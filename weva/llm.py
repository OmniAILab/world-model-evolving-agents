from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable


class LLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        role: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> str: ...

    def generate_scored(self, prompt: str, *, role: str, **kwargs) -> tuple[str, float | None]:
        return self.generate(prompt, role=role, **kwargs), None


class CountingLLM(LLM):
    def __init__(self, inner: LLM):
        self.inner = inner
        self.calls: Counter[str] = Counter()

    def generate(self, prompt: str, *, role: str, **kwargs) -> str:
        self.calls[role] += 1
        return self.inner.generate(prompt, role=role, **kwargs)

    def generate_scored(self, prompt: str, *, role: str, **kwargs) -> tuple[str, float | None]:
        self.calls[role] += 1
        return self.inner.generate_scored(prompt, role=role, **kwargs)

    @property
    def total(self) -> int:
        return sum(self.calls.values())


class ScriptedLLM(LLM):
    def __init__(self, handlers: dict[str, Callable[[str], str]]):
        self.handlers = handlers
        self.log: list[tuple[str, str, str]] = []

    def generate(self, prompt: str, *, role: str, **kwargs) -> str:
        if role not in self.handlers:
            raise KeyError(f"No scripted handler for role '{role}'")
        out = self.handlers[role](prompt)
        self.log.append((role, prompt, out))
        return out


def _logprob_of(x) -> float:
    if isinstance(x, dict):
        return float(x.get("logprob", 0.0))
    return float(getattr(x, "logprob", 0.0))


class LiteLLM(LLM):
    def __init__(
        self,
        model: str,
        send_sampling_params: bool = True,
        min_output_tokens: int = 0,
        logprobs: bool = True,
        **kwargs,
    ):
        try:
            import litellm
        except ImportError as e:
            raise ImportError("Install litellm to use the 'litellm' backend: pip install -e '.[llm]'") from e
        if not model:
            raise ValueError("The litellm backend needs llm.model, e.g. gemini/gemini-3.5-flash")
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        self.litellm = litellm
        self.model = model
        self.send_sampling_params = send_sampling_params
        self.min_output_tokens = min_output_tokens
        self.logprobs = logprobs
        self.kwargs = kwargs

    def _complete(self, prompt, temperature, top_p, max_tokens, logprobs):
        extra = {"logprobs": True} if logprobs else {}
        if self.send_sampling_params:
            extra.update(temperature=temperature, top_p=top_p)
        return self.litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max(max_tokens, self.min_output_tokens),
            drop_params=True,
            **extra,
            **self.kwargs,
        )

    def generate(self, prompt, *, role, temperature=0.0, top_p=1.0, max_tokens=512):
        resp = self._complete(prompt, temperature, top_p, max_tokens, False)
        return resp.choices[0].message.content or ""

    def generate_scored(self, prompt, *, role, temperature=0.0, top_p=1.0, max_tokens=512):
        kw = {"role": role, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        if not self.logprobs:
            return self.generate(prompt, **kw), None
        try:
            resp = self._complete(prompt, temperature, top_p, max_tokens, True)
        except self.litellm.BadRequestError:
            self.logprobs = False
            return self.generate(prompt, **kw), None
        text = resp.choices[0].message.content or ""
        lp = getattr(resp.choices[0], "logprobs", None)
        content = lp.get("content") if isinstance(lp, dict) else getattr(lp, "content", None)
        if not content:
            return text, None
        values = [_logprob_of(x) for x in content]
        return text, sum(values) / len(values)


def create_llm(cfg) -> LLM:
    if cfg.backend == "toy":
        from .envs.toy_llm import make_toy_llm

        return make_toy_llm()
    if cfg.backend == "litellm":
        return LiteLLM(
            cfg.model,
            send_sampling_params=cfg.send_sampling_params,
            min_output_tokens=cfg.min_output_tokens,
            logprobs=cfg.logprobs,
            **cfg.extra,
        )
    if cfg.backend == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend.from_pretrained(cfg.model, adapters=cfg.adapters).llm
    raise ValueError(f"Unknown LLM backend '{cfg.backend}'")
