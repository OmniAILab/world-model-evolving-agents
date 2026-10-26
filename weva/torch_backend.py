from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .components import candidate_lines, world_model_prompt
from .llm import LLM
from .state import parse_state
from .types import State, Transition

PERCEPTOR, WORLD_MODEL, ACTOR = "perceptor", "world_model", "actor"
MODULES = (PERCEPTOR, WORLD_MODEL, ACTOR)
Completion = str | list[tuple[str, float]]
ROLE_TO_MODULE = {"perceptor": PERCEPTOR, "world_model": WORLD_MODEL, "actor": ACTOR}


@dataclass
class TorchBackendConfig:
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lr: float = 1e-6
    world_model_lr: float = 1e-6
    ppo_epochs: int = 1
    sample_temperature: float = 0.9
    max_new_tokens: int = 256
    max_prompt_tokens: int = 3072
    grad_clip: float = 1.0
    unselected_candidate_weight: float = 0.0
    seed: int = 0


class ByteTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    offset = 2
    vocab_size = 258
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [b + self.offset for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return bytes(int(i) - self.offset for i in ids if int(i) >= self.offset).decode(
            "utf-8", errors="ignore"
        )


def tiny_model(seed: int = 0, hidden: int = 32, layers: int = 2):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=ByteTokenizer.vocab_size,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        pad_token_id=ByteTokenizer.pad_token_id,
        bos_token_id=ByteTokenizer.eos_token_id,
        eos_token_id=ByteTokenizer.eos_token_id,
    )
    return LlamaForCausalLM(cfg)


@dataclass
class _Item:
    prompt: str
    completion: Completion
    advantage: float
    old: object = None
    extra: dict = field(default_factory=dict)


class TorchBackend:
    def __init__(self, model, tokenizer, cfg: TorchBackendConfig | None = None, device: str | None = None):
        import torch
        from peft import LoraConfig, get_peft_model

        self.torch = torch
        self.cfg = cfg or TorchBackendConfig()
        torch.manual_seed(self.cfg.seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        for p in model.parameters():
            p.requires_grad = False

        def lora():
            return LoraConfig(
                r=self.cfg.lora_rank,
                lora_alpha=self.cfg.lora_alpha,
                target_modules=list(self.cfg.target_modules),
                lora_dropout=0.0,
            )

        self.model = get_peft_model(model, lora(), adapter_name=PERCEPTOR)
        for name in (WORLD_MODEL, ACTOR):
            self.model.add_adapter(name, lora())
        self.model.to(self.device)
        self._trainable: set[str] = set()
        self._apply_trainable()
        self.optimizers = {
            m: torch.optim.AdamW(
                self.params(m), lr=self.cfg.world_model_lr if m == WORLD_MODEL else self.cfg.lr
            )
            for m in MODULES
        }

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        adapters: str | None = None,
        cfg: TorchBackendConfig | None = None,
        device: str | None = None,
    ) -> TorchBackend:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        backend = cls(model, tok, cfg, device)
        if adapters and Path(adapters).exists():
            backend.load(adapters)
        return backend

    def params(self, module: str) -> list:
        return [p for n, p in self.model.named_parameters() if f".{module}." in n]

    def _apply_trainable(self) -> None:
        for m in MODULES:
            for p in self.params(m):
                p.requires_grad = m in self._trainable

    def set_trainable(self, modules: set[str]) -> None:
        unknown = set(modules) - set(MODULES)
        if unknown:
            raise ValueError(f"Unknown trainable modules: {unknown}")
        self._trainable = set(modules)
        self._apply_trainable()

    def trainable(self) -> set[str]:
        return {m for m in MODULES if any(p.requires_grad for p in self.params(m))}

    @contextmanager
    def _using(self, module: str | None) -> Iterator[None]:
        if module is None:
            with self.model.disable_adapter():
                yield
        else:
            self.model.set_adapter(module)
            self._apply_trainable()
            yield

    def _format(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        return prompt + "\n"

    def _prompt_ids(self, prompt: str) -> list[int]:
        ids = self.tokenizer.encode(self._format(prompt), add_special_tokens=False)
        return ids[-self.cfg.max_prompt_tokens :]

    def _completion(self, completion: Completion) -> tuple[list[int], list[float]]:
        if isinstance(completion, str):
            ids = self.tokenizer.encode(completion, add_special_tokens=False)
            return ids + [self.tokenizer.eos_token_id], [1.0] * (len(ids) + 1)
        text = "".join(t for t, _ in completion)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        bounds, pos = [], 0
        for seg, w in completion:
            bounds.append((pos, pos + len(seg), w))
            pos += len(seg)
        weights = []
        for a, b in self._token_spans(text, ids):
            if a == b:
                hit = [w for s, e, w in bounds if s <= a < e]
            else:
                hit = [w for s, e, w in bounds if a < e and b > s]
            weights.append(max(hit, default=0.0))
        return ids + [self.tokenizer.eos_token_id], weights + [max((w for _, w in completion), default=1.0)]

    def _token_spans(self, text: str, ids: list[int]) -> list[tuple[int, int]]:
        try:
            enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            if list(enc["input_ids"]) == list(ids):
                return [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        except (TypeError, ValueError, KeyError, NotImplementedError):
            pass
        spans, prev = [], 0
        for k in range(1, len(ids) + 1):
            end = max(len(self.tokenizer.decode(ids[:k], skip_special_tokens=True)), prev)
            spans.append((prev, end))
            prev = end
        return spans

    def _completion_ids(self, completion: Completion) -> list[int]:
        return self._completion(completion)[0]

    def token_logprobs(self, module: str | None, prompt: str, completion: Completion, grad: bool = False):
        torch = self.torch
        p_ids, c_ids = self._prompt_ids(prompt), self._completion_ids(completion)
        ids = torch.tensor([p_ids + c_ids], device=self.device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, self._using(module):
            logits = self.model(input_ids=ids).logits[0, len(p_ids) - 1 : -1].float()
            logp = torch.log_softmax(logits, dim=-1)
            target = torch.tensor(c_ids, device=self.device)
            return logp.gather(1, target[:, None]).squeeze(1)

    def generate_text(
        self,
        module: str | None,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> tuple[str, float | None]:
        torch = self.torch
        p_ids = self._prompt_ids(prompt)
        ids = torch.tensor([p_ids], device=self.device)
        kwargs = {"do_sample": temperature > 0}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p)
        self.model.eval()
        with torch.no_grad(), self._using(module):
            out = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_tokens or self.cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        new = [int(i) for i in out[0, len(p_ids) :].tolist()]
        if self.tokenizer.eos_token_id in new:
            new = new[: new.index(self.tokenizer.eos_token_id)]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        lp = self.token_logprobs(module, prompt, text)
        return text, float(lp.mean()) if lp.numel() else None

    def logprob(self, state: State, action: str, successor: State) -> float:
        return float(
            self.token_logprobs(WORLD_MODEL, world_model_prompt(state, action), successor.render()).sum()
        )

    def sample_inductions(self, transitions: Sequence[Transition], n: int) -> list:
        from .training import PerceptorSample

        samples = []
        for t in transitions:
            if not t.source_prompt:
                continue
            texts = [
                self.generate_text(
                    PERCEPTOR, t.source_prompt, temperature=self.cfg.sample_temperature, top_p=1.0
                )[0]
                for _ in range(n)
            ]
            states = [parse_state(x) for x in texts]
            samples.append(PerceptorSample(t, states, prompt=t.source_prompt, texts=texts))
        return samples

    def _grpo(self, module: str, items: list[_Item], clip_eps: float, kl_coef: float, lr: float) -> dict:
        torch = self.torch
        items = [i for i in items if i.advantage != 0.0]
        if not items:
            return {"n": 0, "loss": 0.0, "kl": 0.0, "clip_fraction": 0.0}
        for it in items:
            it.old = self.token_logprobs(module, it.prompt, it.completion).detach()
            it.extra["ref"] = self.token_logprobs(None, it.prompt, it.completion).detach()
            it.extra["w"] = torch.tensor(self._completion(it.completion)[1], device=self.device)
        opt = self.optimizers[module]
        for g in opt.param_groups:
            g["lr"] = lr
        self.model.train()
        stats = {"loss": 0.0, "kl": 0.0, "clip_fraction": 0.0}
        for _ in range(max(1, self.cfg.ppo_epochs)):
            opt.zero_grad()
            stats = {"loss": 0.0, "kl": 0.0, "clip_fraction": 0.0}
            for it in items:
                new = self.token_logprobs(module, it.prompt, it.completion, grad=True)
                ratio = torch.exp(new - it.old)
                a = it.advantage * it.extra["w"]
                surrogate = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
                d = it.extra["ref"] - new
                kl = torch.exp(d) - d - 1
                loss = -(surrogate - kl_coef * kl).mean() / len(items)
                loss.backward()
                stats["loss"] += float(loss.detach())
                stats["kl"] += float(kl.detach().mean()) / len(items)
                stats["clip_fraction"] += float(((ratio - 1).abs() > clip_eps).float().mean()) / len(items)
            torch.nn.utils.clip_grad_norm_(self.params(module), self.cfg.grad_clip)
            opt.step()
        self.model.eval()
        return {"n": len(items), **stats}

    def update_perceptor(self, samples, cfg) -> dict:
        items = [
            _Item(s.prompt, text, a) for s in samples for text, a in zip(s.texts, s.advantages) if s.prompt
        ]
        return self._grpo(PERCEPTOR, items, cfg.clip_eps, cfg.kl_coef, cfg.lr)

    def reinduce(self, transitions: Sequence[Transition]) -> list[Transition]:
        cache: dict[str, State] = {}

        def induce(prompt: str | None, fallback: State) -> State:
            if not prompt:
                return fallback
            if prompt not in cache:
                text, _ = self.generate_text(PERCEPTOR, prompt)
                state = parse_state(text)
                cache[prompt] = state if not state.is_empty() else fallback
            return cache[prompt]

        return [
            dataclasses.replace(
                t, source=induce(t.source_prompt, t.source), target=induce(t.target_prompt, t.target)
            )
            for t in transitions
        ]

    @staticmethod
    def world_model_target(t: Transition) -> str:
        target = t.target.render()
        if t.violated_labels is not None:
            target += "\nVIOLATED: " + ("; ".join(t.violated_labels) or "none")
        return target

    def world_model_nll(self, transitions: Sequence[Transition], grad: bool = False):
        terms = [
            -self.token_logprobs(
                WORLD_MODEL, world_model_prompt(t.source, t.action), self.world_model_target(t), grad=grad
            ).sum()
            for t in transitions
        ]
        return sum(terms) / max(len(terms), 1)

    def update_world_model(
        self, transitions: Sequence[Transition], lr: float | None = None, epochs: int = 1
    ) -> dict:
        if not transitions:
            return {"n": 0}
        opt = self.optimizers[WORLD_MODEL]
        for g in opt.param_groups:
            g["lr"] = lr if lr is not None else self.cfg.world_model_lr
        before = float(self.world_model_nll(transitions))
        self.model.train()
        for _ in range(max(1, epochs)):
            opt.zero_grad()
            loss = self.world_model_nll(transitions, grad=True)
            loss.backward()
            self.torch.nn.utils.clip_grad_norm_(self.params(WORLD_MODEL), self.cfg.grad_clip)
            opt.step()
        self.model.eval()
        return {
            "n": len(transitions),
            "nll_before": before,
            "nll_after": float(self.world_model_nll(transitions)),
        }

    def actor_completion(self, t: Transition) -> list[tuple[str, float]] | None:
        if not t.actor_output:
            return None
        lines = candidate_lines(t.actor_output)
        if not any(action == t.action for _, action in lines):
            return None
        w = self.cfg.unselected_candidate_weight
        return [(line, 1.0 if action == t.action else w) for line, action in lines]

    def actor_items(self, groups) -> list[_Item]:
        items = []
        for g in groups:
            for rollout, a in zip(g.rollouts, g.advantages):
                for t in rollout:
                    completion = self.actor_completion(t)
                    if t.actor_prompt and completion is not None:
                        items.append(_Item(t.actor_prompt, completion, a))
        return items

    def update_actor(self, groups, cfg) -> dict:
        return self._grpo(ACTOR, self.actor_items(groups), cfg.clip_eps, cfg.kl_coef, cfg.lr)

    def adapter_state(self) -> dict:
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if "lora_" in n}

    def snapshot(self) -> dict:
        return {
            "adapters": self.adapter_state(),
            "optimizers": {m: copy.deepcopy(o.state_dict()) for m, o in self.optimizers.items()},
        }

    def restore(self, snap: dict) -> None:
        named = dict(self.model.named_parameters())
        with self.torch.no_grad():
            for n, v in snap["adapters"].items():
                named[n].copy_(v)
        for m, s in snap["optimizers"].items():
            self.optimizers[m].load_state_dict(s)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.torch.save({n: v.cpu() for n, v in self.adapter_state().items()}, path / "adapters.pt")
        (path / "adapters.json").write_text(json.dumps(dataclasses.asdict(self.cfg), indent=2))

    def load(self, path: str | Path) -> None:
        state = self.torch.load(Path(path) / "adapters.pt", map_location=self.device)
        named = dict(self.model.named_parameters())
        missing = [n for n in state if n not in named]
        if missing:
            raise KeyError(f"Adapter weights do not match the model: {missing[:3]}")
        with self.torch.no_grad():
            for n, v in state.items():
                named[n].copy_(v)

    @property
    def llm(self) -> TorchLLM:
        return TorchLLM(self)


class TorchLLM(LLM):
    def __init__(self, backend: TorchBackend):
        self.backend = backend

    def generate_scored(self, prompt, *, role, temperature=0.0, top_p=1.0, max_tokens=512):
        return self.backend.generate_text(ROLE_TO_MODULE.get(role), prompt, temperature, top_p, max_tokens)

    def generate(self, prompt, *, role, temperature=0.0, top_p=1.0, max_tokens=512):
        return self.generate_scored(
            prompt, role=role, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )[0]
