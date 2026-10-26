import sys
from types import SimpleNamespace

from weva import load_config
from weva.llm import LiteLLM, create_llm


class FakeLiteLLM:
    def __init__(self):
        self.calls = []

    def completion(self, **kw):
        self.calls.append(kw)
        logprobs = {"content": [{"logprob": -0.5}, {"logprob": -1.5}]} if kw.get("logprobs") else None
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="V: x"), logprobs=logprobs)]
        )


def test_gemini_is_the_api_backbone_and_receives_the_paper_decoding(monkeypatch):
    fake = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    for cfgs in (
        ["configs/methods/weva_doc.yaml"],
        ["configs/methods/react.yaml", "configs/backbones/gemini.yaml"],
    ):
        cfg = load_config(cfgs)
        assert cfg.llm.backend == "litellm" and cfg.llm.model == "gemini/gemini-3.5-flash"
    llm = create_llm(load_config(["configs/methods/weva_doc.yaml"]).llm)
    assert llm.generate_scored("p", role="perceptor") == ("V: x", None)
    assert "logprobs" not in fake.calls[-1]
    text, mean_lp = LiteLLM("gemini/gemini-3.5-flash").generate_scored("p", role="perceptor", temperature=0.0)
    assert text == "V: x" and mean_lp == -1.0
    llm.generate("p", role="actor", temperature=0.3, top_p=0.9)
    assert fake.calls[-1]["model"] == "gemini/gemini-3.5-flash" and fake.calls[-1]["temperature"] == 0.3
    quiet = LiteLLM("gemini/gemini-3.5-flash", send_sampling_params=False)
    quiet.generate("p", role="actor", temperature=0.3)
    assert "temperature" not in fake.calls[-1] and "top_p" not in fake.calls[-1]


def test_a_rejected_logprob_request_falls_back_to_plain_generation(monkeypatch):
    class Rejected(Exception):
        pass

    class Strict(FakeLiteLLM):
        BadRequestError = Rejected

        def completion(self, **kw):
            if kw.get("logprobs"):
                raise Rejected("logprobs not supported")
            return super().completion(**kw)

    fake = Strict()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    llm = LiteLLM("gemini/gemini-3.5-flash")
    assert llm.generate_scored("p", role="perceptor") == ("V: x", None)
    assert llm.generate_scored("p", role="perceptor") == ("V: x", None)
    assert sum(1 for c in fake.calls if c.get("logprobs")) == 0
