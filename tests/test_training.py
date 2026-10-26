import math

import numpy as np
import pytest

from weva import create_env, load_config
from weva.agent import build_agent
from weva.grpo import group_advantages, grpo_objective, successor_nll
from weva.training import (
    ACTOR,
    CONTROLLER,
    PERCEPTOR,
    WORLD_MODEL,
    NullBackend,
    Phase1Config,
    _set_trainable,
    calibrate_controller,
    collect_round,
    contradiction_ratio,
    load_phase2_config,
    novelty,
    perceptor_reward,
    phase2_update,
    run_phase2,
)
from weva.types import State, Transition

WEVA = ["configs/methods/weva.yaml", "configs/toy.yaml"]
P2 = "configs/training/phase2.yaml"


def test_group_advantages_are_normalised_within_the_group():
    a = np.array(group_advantages([1.0, 0.0, 1.0, 0.0]))
    assert abs(a.mean()) < 1e-9 and abs(a.std() - 1.0) < 1e-3
    assert group_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]


def test_grpo_objective_clips_the_ratio_and_has_zero_kl_at_the_reference():
    out = grpo_objective([[0.0]], [[0.0]], [[0.0]], [1.0])
    assert out["loss"] == pytest.approx(-1.0) and out["kl"] == pytest.approx(0.0)
    out = grpo_objective([[math.log(2.0)]], [[0.0]], [[math.log(2.0)]], [1.0], clip_eps=0.2)
    assert out["loss"] == pytest.approx(-1.2) and out["clip_fraction"] == 1.0
    assert successor_nll([[-1.0, -2.0], [-3.0]]) == pytest.approx(3.0)


def test_eva_signals_follow_their_definitions():
    s = State(V=("a",), C=("keep x", "keep y"))
    checker = lambda c, src, a, t: c == "keep x"
    assert contradiction_ratio(s, "go", None, checker) == 0.5
    assert contradiction_ratio(State(V=("a",)), "go", None, checker) == 0.0
    assert novelty(s, []) == 1.0 and novelty(s, [s]) == pytest.approx(0.0)
    t = Transition(source=s, action="go", target=s, outcome=1.0)
    r = perceptor_reward(NullBackend(), t, [], checker=checker)
    assert r == pytest.approx(1.0 + 0.5 + 1.0 + 0.0)


def _toy_round(group_size=2):
    cfg = load_config(WEVA)
    env = create_env("toy-ledger")
    agent = build_agent(cfg)
    return collect_round(agent, env, env.tasks()[:1], group_size, cfg.run.max_steps)


def test_phase2_alternation_freezes_the_right_modules():
    transitions, groups, _ = _toy_round()
    assert transitions and transitions[0].observation and len(groups[0].outcomes) == 2
    assert all(t.outcome is not None for t in transitions)
    backend = NullBackend()
    p2 = load_phase2_config(P2, ["grpo.group_size=2"])
    phase2_update(backend, transitions, groups, p2, [])
    steps = [name for name, _ in backend.calls]
    assert (
        steps.index("perceptor") < steps.index("reinduce") < steps.index("world_model") < steps.index("actor")
    )
    for name, trainable in backend.calls:
        assert CONTROLLER not in trainable
        if name in ("logprob", "perceptor"):
            assert trainable == {PERCEPTOR}
        if name == "world_model":
            assert trainable == {WORLD_MODEL}
        if name == "actor":
            assert trainable == {ACTOR}
    assert backend.trainable() == set()


def test_confidence_gate_excludes_low_confidence_inductions():
    transitions, groups, _ = _toy_round()
    for t in transitions:
        t.source_confidence = 0.1
    out = phase2_update(NullBackend(), transitions, groups, load_phase2_config(P2, ["grpo.group_size=2"]), [])
    assert out["n_gated_out"] == len(transitions) and out["perceptor"]["n"] == 0


def test_controller_cannot_be_made_trainable_and_unknown_keys_fail():
    with pytest.raises(RuntimeError):
        _set_trainable(NullBackend(), {CONTROLLER})
    with pytest.raises(KeyError):
        load_phase2_config(P2, ["typo=1"])


def test_weva_doc_refuses_parameter_updates(tmp_path):
    weights = tmp_path / "ctrl.json"
    weights.write_text('{"W": [[0,0,0,0,0]], "b": [0]}')
    with pytest.raises(ValueError):
        run_phase2(
            ["configs/methods/weva_doc.yaml", "configs/toy.yaml"],
            P2,
            create_env("toy-ledger"),
            NullBackend(),
            str(weights),
        )


def test_calibration_uses_post_adaptation_episodes_and_phase2_reuses_it(tmp_path):
    env = create_env("toy-ledger")
    with pytest.raises(FileNotFoundError):
        run_phase2(WEVA, P2, env, NullBackend(), str(tmp_path / "missing.json"))
    p1 = Phase1Config(iterations=1, n_envs=1, inner_steps=2, group_size=2, eval_episodes=1, split="test")
    backend = NullBackend()
    weights = calibrate_controller(
        WEVA,
        env,
        str(tmp_path / "ctrl.json"),
        p1,
        backend=backend,
        p2=load_phase2_config(P2, ["grpo.group_size=2"]),
    )
    assert [n for n, _ in backend.calls].count("actor") == p1.inner_steps
    meta = (tmp_path / "ctrl.meta.json").read_text()
    assert '"weight_adaptation": true' in meta and '"inner_steps": 2' in meta
    before = (tmp_path / "ctrl.json").read_text()
    logs = run_phase2(
        WEVA,
        P2,
        env,
        NullBackend(),
        weights,
        phase2_overrides=["rounds=1", "grpo.group_size=2", f"output_dir={tmp_path / 'p2'}"],
    )
    assert len(logs) == 1 and logs[0]["n_transitions"] > 0
    assert (tmp_path / "p2" / "phase2_log.jsonl").exists()
    assert (tmp_path / "ctrl.json").read_text() == before
