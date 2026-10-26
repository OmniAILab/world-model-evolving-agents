import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

try:
    from weva.torch_backend import ByteTokenizer as _ProbeTokenizer
    from weva.torch_backend import TorchBackend as _Probe
    from weva.torch_backend import tiny_model as _probe_model

    _Probe(_probe_model(), _ProbeTokenizer())
except ImportError as e:
    pytest.skip(f"training backend unavailable: {e}", allow_module_level=True)

from weva.components import perceptor_prompt
from weva.torch_backend import ByteTokenizer, TorchBackend, TorchBackendConfig, tiny_model
from weva.training import (
    ACTOR,
    PERCEPTOR,
    WORLD_MODEL,
    ActorGroup,
    GRPOConfig,
    PerceptorSample,
    Phase2Config,
    _set_trainable,
    phase2_update,
)
from weva.types import State, Transition


def make_backend(**kw):
    cfg = TorchBackendConfig(
        lora_rank=4,
        lora_alpha=8,
        target_modules=("q_proj", "v_proj"),
        lr=5e-2,
        world_model_lr=5e-2,
        max_new_tokens=12,
        **kw,
    )
    return TorchBackend(tiny_model(), ByteTokenizer(), cfg, device="cpu")


def snapshot(backend, module):
    return [p.detach().clone() for p in backend.params(module)]


def changed(before, backend, module):
    return any(not torch.equal(a, b) for a, b in zip(before, backend.params(module)))


def transition(i=0):
    src = State(V=(f"box {i} closed",), C=("key needed",), E=("progress: 0 of 1",))
    tgt = State(V=(f"box {i} open",), C=("key needed",), E=("progress: 1 of 1",))
    return Transition(
        source=src,
        action="open box",
        target=tgt,
        outcome=float(i % 2),
        source_prompt=perceptor_prompt("open the box", f"a closed box {i}", [], []),
        target_prompt=perceptor_prompt("open the box", f"an open box {i}", [], []),
        actor_prompt=f"Propose the next action. state {i}",
        actor_output="look around\nopen box\nsmash box",
    )


def test_freezing_is_enforced_on_real_parameters():
    b = make_backend()
    _set_trainable(b, {PERCEPTOR})
    assert b.trainable() == {PERCEPTOR}
    b.generate_text(WORLD_MODEL, "probe")
    assert b.trainable() == {PERCEPTOR}
    assert not any(p.requires_grad for n, p in b.model.named_parameters() if "lora_" not in n)


def test_world_model_nll_decreases_and_only_the_world_model_moves():
    b = make_backend()
    data = [transition(i) for i in range(2)]
    others = {m: snapshot(b, m) for m in (PERCEPTOR, ACTOR)}
    wm = snapshot(b, WORLD_MODEL)
    _set_trainable(b, {WORLD_MODEL})
    out = None
    for _ in range(5):
        out = b.update_world_model(data, epochs=1)
    first = b.world_model_nll(data)
    assert out["nll_after"] < out["nll_before"] and float(first) == pytest.approx(out["nll_after"], rel=1e-4)
    assert changed(wm, b, WORLD_MODEL)
    assert not any(changed(v, b, m) for m, v in others.items())


def test_actor_items_train_the_sampled_proposal_and_credit_the_selected_candidate():
    b = make_backend()
    t = transition(1)
    items = b.actor_items([ActorGroup("t", [[t]], [1.0], [1.0])])
    assert len(items) == 1 and items[0].prompt == t.actor_prompt
    assert items[0].completion == [("look around\n", 0.0), ("open box\n", 1.0), ("smash box", 0.0)]
    stray = Transition(**{**t.__dict__, "action": "jump"})
    assert b.actor_items([ActorGroup("t", [[stray]], [1.0], [1.0])]) == []
    no_output = Transition(**{**t.__dict__, "actor_output": None})
    assert b.actor_items([ActorGroup("t", [[no_output]], [1.0], [1.0])]) == []
    weights = b._completion(items[0].completion)[1]
    assert weights.count(1.0) == len("open box\n") + 1


def test_actor_grpo_raises_the_selected_candidate_inside_the_sampled_output():
    b = make_backend(ppo_epochs=4)
    good = transition(1)
    bad = Transition(**{**good.__dict__, "action": "smash box"})
    group = ActorGroup("t", [[good], [bad]], [1.0, 0.0], [1.0, -1.0])

    def lp(line):
        return float(
            b.token_logprobs(ACTOR, good.actor_prompt, [("look around\n", 0.0), (line, 1.0)])[
                len("look around\n") : -1
            ].sum()
        )

    before_good, before_bad, frozen = lp("open box\n"), lp("smash box"), snapshot(b, PERCEPTOR)
    _set_trainable(b, {ACTOR})
    stats = b.update_actor([group], GRPOConfig(lr=5e-2, kl_coef=0.001, clip_eps=0.2))
    assert stats["n"] == 2 and lp("open box\n") > before_good and lp("smash box") < before_bad
    assert not changed(frozen, b, PERCEPTOR)


def test_world_model_target_supervises_verified_violations_only_when_checked():
    t = transition(0)
    assert "VIOLATED" not in TorchBackend.world_model_target(t)
    t.violated_labels = ()
    assert TorchBackend.world_model_target(t).endswith("VIOLATED: none")
    t.violated_labels = ("key needed",)
    assert TorchBackend.world_model_target(t).endswith("VIOLATED: key needed")


def test_phase2_labels_violations_with_the_checker_not_with_task_failure():
    from weva.training import NullBackend

    ok, bad = transition(0), transition(1)
    bad = Transition(**{**bad.__dict__, "action": "force box", "outcome": 0.0})
    ok.outcome = 0.0
    groups = [ActorGroup("t", [[ok], [bad]], [0.0, 0.0])]
    checker = lambda c, s, a, tgt: c == "key needed" and a == "force box"
    phase2_update(NullBackend(), [ok, bad], groups, Phase2Config(grpo=GRPOConfig(group_size=2)), [], checker)
    assert ok.violated_labels == () and bad.violated_labels == ("key needed",)


def test_full_phase2_round_updates_each_component_with_real_gradients():
    b = make_backend()
    data = [transition(i) for i in range(2)]
    groups = [ActorGroup("t", [[data[0]], [data[1]]], [0.0, 1.0])]
    before = {m: snapshot(b, m) for m in (PERCEPTOR, WORLD_MODEL, ACTOR)}
    p2 = Phase2Config(grpo=GRPOConfig(group_size=3, lr=5e-2), world_model_lr=5e-2)
    out = phase2_update(b, data, groups, p2, [])
    assert "n" in out["perceptor"] and out["world_model"]["nll_after"] < out["world_model"]["nll_before"]
    assert all(changed(before[m], b, m) for m in (WORLD_MODEL, ACTOR))
    assert b.trainable() == set()


def test_perceptor_grpo_moves_only_the_perceptor_towards_the_better_induction():
    b = make_backend(ppo_epochs=4)
    t = transition(0)
    good, bad = t.target.render(), "V: nothing"
    sample = PerceptorSample(
        t, [t.target, State(V=("nothing",))], t.source_prompt, [good, bad], [1.0, 0.0], [1.0, -1.0]
    )
    lp = lambda: float(b.token_logprobs(PERCEPTOR, t.source_prompt, good).sum())
    before, frozen = lp(), {m: snapshot(b, m) for m in (WORLD_MODEL, ACTOR)}
    _set_trainable(b, {PERCEPTOR})
    assert b.update_perceptor([sample], GRPOConfig(lr=5e-2))["n"] == 2
    assert lp() > before and not any(changed(v, b, m) for m, v in frozen.items())


def test_snapshot_restore_and_save_load(tmp_path):
    b = make_backend()
    snap = b.snapshot()
    _set_trainable(b, {WORLD_MODEL})
    b.update_world_model([transition(0)])
    b.save(tmp_path / "ad")
    trained = snapshot(b, WORLD_MODEL)
    b.restore(snap)
    assert changed(trained, b, WORLD_MODEL)
    b.load(tmp_path / "ad")
    assert not changed(trained, b, WORLD_MODEL)


def test_torch_llm_serves_the_agent_roles():
    b = make_backend()
    text, mean_lp = b.llm.generate_scored("hello", role="perceptor", max_tokens=5)
    assert isinstance(text, str) and mean_lp is None or mean_lp <= 0.0


def test_end_to_end_phase2_with_a_real_model_changes_weights_and_saves_adapters(tmp_path):
    from weva import create_env
    from weva.training import run_phase2

    b = make_backend()
    weights = tmp_path / "ctrl.json"
    weights.write_text(
        '{"W": [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]], "b": [0,0,0,0,0]}'
    )
    before = {m: snapshot(b, m) for m in (WORLD_MODEL, ACTOR)}
    logs = run_phase2(
        ["configs/methods/weva.yaml", "configs/toy.yaml"],
        "configs/training/phase2.yaml",
        create_env("toy-ledger"),
        b,
        str(weights),
        overrides=["run.max_steps=2", "lookahead.horizon=1", "lookahead.branching=2"],
        phase2_overrides=[
            "rounds=1",
            "grpo.group_size=2",
            "grpo.lr=0.05",
            "world_model_lr=0.05",
            f"output_dir={tmp_path / 'p2'}",
        ],
    )
    assert logs[0]["n_transitions"] == 3 * 2 * 2
    assert changed(before[WORLD_MODEL], b, WORLD_MODEL)
    assert (tmp_path / "p2" / "adapters" / "adapters.pt").exists()


class MergingTokenizer(ByteTokenizer):
    MERGED = 999

    def encode(self, text, add_special_tokens=False):
        raw = super().encode(text)
        out, i = [], 0
        while i < len(raw):
            if raw[i : i + 2] == super().encode("\no"):
                out.append(self.MERGED)
                i += 2
            else:
                out.append(raw[i])
                i += 1
        return out

    def decode(self, ids, skip_special_tokens=True):
        flat = []
        for i in ids:
            flat += super().encode("\no") if i == self.MERGED else [i]
        return super().decode(flat)


def test_actor_completion_is_tokenized_once_as_the_whole_generated_output():
    from types import SimpleNamespace

    tok = MergingTokenizer()
    fake = SimpleNamespace(tokenizer=tok)
    fake._token_spans = lambda text, ids: TorchBackend._token_spans(fake, text, ids)
    segments = [("look around\n", 0.0), ("open box\n", 1.0), ("smash box", 0.0)]
    ids, weights = TorchBackend._completion(fake, segments)
    full = "look around\nopen box\nsmash box"
    assert ids[:-1] == tok.encode(full) and tok.MERGED in ids
    separate = [i for s, _ in segments for i in tok.encode(s)]
    assert separate != ids[:-1]
    assert len(weights) == len(ids)
    selected = [tok.decode([i]) for i, w in zip(ids[:-1], weights[:-1]) if w == 1.0]
    assert "".join(selected).replace("\n", "").startswith("open box") and weights[-1] == 1.0
    assert weights[ids.index(tok.MERGED)] == 1.0
