from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .components import LearnedController
from .grpo import group_advantages
from .state import plain, sim_psi
from .types import State, Transition

log = logging.getLogger(__name__)

PERCEPTOR, WORLD_MODEL, ACTOR, CONTROLLER = "perceptor", "world_model", "actor", "controller"

Checker = Callable[[str, State, str, State | None], bool]


class SuccessorLogProb(Protocol):
    def logprob(self, state: State, action: str, successor: State) -> float: ...


def predictability_reward(wm: SuccessorLogProb, t: Transition) -> float:
    return float(wm.logprob(t.source, t.action, t.target))


def contradiction_ratio(state: State, action: str, realised: State | None, checker: Checker | None) -> float:
    constraints = [c for c in state.C if plain(c)]
    if checker is None or not constraints:
        return 0.0
    return sum(int(checker(c, state, action, realised)) for c in constraints) / max(1, len(constraints))


def structural_consistency(
    state: State, action: str, realised: State | None, checker: Checker | None
) -> float:
    return 1.0 - contradiction_ratio(state, action, realised, checker)


def violation_labels(t: Transition, checker: Checker) -> tuple[str, ...]:
    return tuple(c for c in t.source.C if plain(c) and checker(c, t.source, t.action, t.target))


def novelty(state: State, seen: Sequence[State]) -> float:
    if not seen:
        return 1.0
    return 1.0 - max(sim_psi(state, s) for s in seen)


@dataclass
class PerceptorRewardWeights:
    task: float = 1.0
    structural_consistency: float = 1.0
    novelty: float = 1.0
    predictability: float = 1.0


def perceptor_reward(
    wm: SuccessorLogProb,
    t: Transition,
    seen: Sequence[State],
    w: PerceptorRewardWeights | None = None,
    checker: Checker | None = None,
) -> float:
    w = w or PerceptorRewardWeights()
    r = w.task * float(t.outcome or 0.0)
    if w.structural_consistency:
        r += w.structural_consistency * structural_consistency(t.source, t.action, t.target, checker)
    if w.novelty:
        r += w.novelty * novelty(t.source, seen)
    if w.predictability:
        r += w.predictability * predictability_reward(wm, t)
    return r


@dataclass
class GRPOConfig:
    group_size: int = 8
    lr: float = 1e-6
    kl_coef: float = 0.001
    clip_eps: float = 0.2


@dataclass
class Phase2Config:
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    rounds: int = 3
    perceptor_reward: PerceptorRewardWeights = field(default_factory=PerceptorRewardWeights)
    confidence_gate: float = 0.6
    invalid_induction_reward: float = -1.0
    train_world_model: bool = True
    world_model_lr: float = 1e-6
    world_model_epochs: int = 1
    backend: dict[str, Any] = field(default_factory=dict)
    split: str = "train"
    output_dir: str = "outputs/phase2"


@dataclass
class Phase1Config:
    lr: float = 0.1
    iterations: int = 200
    lam: float = 0.2
    split: str = "calibration"
    n_envs: int = 50
    inner_steps: int = 5
    group_size: int = 8
    eval_episodes: int = 1


def _load_dataclass(cls, path: str | Path, overrides: list[str] | None, nested: dict[str, type]):
    from .config import apply_overrides, load_raw

    data = apply_overrides(load_raw(path), overrides)
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kw = dict(data)
    for key, sub in nested.items():
        if key in kw:
            kw[key] = sub(**kw[key])
    return cls(**kw)


def load_phase2_config(path: str | Path, overrides: list[str] | None = None) -> Phase2Config:
    return _load_dataclass(
        Phase2Config, path, overrides, {"grpo": GRPOConfig, "perceptor_reward": PerceptorRewardWeights}
    )


def load_phase1_config(path: str | Path, overrides: list[str] | None = None) -> Phase1Config:
    return _load_dataclass(Phase1Config, path, overrides, {})


@dataclass
class PerceptorSample:
    transition: Transition
    candidates: list[State]
    prompt: str | None = None
    texts: list[str] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)


@dataclass
class ActorGroup:
    task_id: str
    rollouts: list[list[Transition]]
    outcomes: list[float]
    advantages: list[float] = field(default_factory=list)


class TrainerBackend(Protocol):
    def set_trainable(self, modules: set[str]) -> None: ...
    def trainable(self) -> set[str]: ...
    def logprob(self, state: State, action: str, successor: State) -> float: ...
    def sample_inductions(self, transitions: Sequence[Transition], n: int) -> list[PerceptorSample]: ...
    def update_perceptor(self, samples: Sequence[PerceptorSample], cfg: GRPOConfig) -> dict: ...
    def reinduce(self, transitions: Sequence[Transition]) -> list[Transition]: ...
    def update_world_model(
        self, transitions: Sequence[Transition], lr: float | None = None, epochs: int = 1
    ) -> dict: ...
    def update_actor(self, groups: Sequence[ActorGroup], cfg: GRPOConfig) -> dict: ...


def _set_trainable(backend: TrainerBackend, modules: set[str]) -> None:
    if CONTROLLER in modules:
        raise RuntimeError("The Controller is fixed after Phase I and must not be updated in Phase II")
    backend.set_trainable(modules)
    if backend.trainable() != modules:
        raise RuntimeError(f"Backend reports trainable={backend.trainable()}, expected {modules}")


def collect_round(agent, env, tasks, group_size: int, max_steps: int, integrate: bool = True):
    from .evaluation import run_episode

    transitions: list[Transition] = []
    groups: list[ActorGroup] = []
    results = []
    base = agent.snapshot() if hasattr(agent, "snapshot") else None
    for task in tasks:
        rollouts, outcomes = [], []
        for _ in range(group_size):
            if base is not None:
                agent.restore(base)
            res = run_episode(agent, env, task, max_steps)
            outcome = float(res.extra.get("score", res.success))
            for t in res.transitions:
                t.outcome = outcome
            results.append(res)
            rollouts.append(res.transitions)
            outcomes.append(outcome)
            transitions.extend(res.transitions)
        groups.append(ActorGroup(task.id, rollouts, outcomes))
    if base is not None:
        agent.restore(base)
        if integrate:
            for res in results:
                for t in res.transitions:
                    agent.memory.record(t)
                agent.memory.end_episode()
    return transitions, groups, results


def phase2_update(
    backend: TrainerBackend,
    transitions: list[Transition],
    groups: list[ActorGroup],
    cfg: Phase2Config,
    seen: list[State],
    checker: Checker | None = None,
) -> dict:
    _set_trainable(backend, {PERCEPTOR})
    if WORLD_MODEL in backend.trainable():
        raise RuntimeError("The World Model must be frozen while the Perceptor is updated")
    eligible = [
        t for t in transitions if t.source_confidence is None or t.source_confidence >= cfg.confidence_gate
    ]
    samples = backend.sample_inductions(eligible, cfg.grpo.group_size)
    for s in samples:
        s.rewards = [
            cfg.invalid_induction_reward
            if c.is_empty()
            else perceptor_reward(
                backend, dataclasses.replace(s.transition, source=c), seen, cfg.perceptor_reward, checker
            )
            for c in s.candidates
        ]
        s.advantages = group_advantages(s.rewards)
    log_p = backend.update_perceptor(samples, cfg.grpo)
    seen.extend(t.source for t in transitions)

    _set_trainable(backend, set())
    transitions = backend.reinduce(transitions)

    if checker is not None:
        for t in transitions:
            t.violated_labels = violation_labels(t, checker)

    log_w: dict = {"skipped": True}
    if cfg.train_world_model:
        _set_trainable(backend, {WORLD_MODEL})
        if PERCEPTOR in backend.trainable():
            raise RuntimeError("The Perceptor must be frozen while the World Model is updated")
        log_w = backend.update_world_model(transitions, cfg.world_model_lr, cfg.world_model_epochs)

    _set_trainable(backend, {ACTOR})
    for g in groups:
        g.advantages = group_advantages(g.outcomes)
    log_a = backend.update_actor(groups, cfg.grpo)
    _set_trainable(backend, set())

    rewards = [r for s in samples for r in s.rewards]
    n_outcomes = sum(len(g.outcomes) for g in groups)
    return {
        "n_transitions": len(transitions),
        "n_gated_out": len(transitions) - len(eligible),
        "mean_perceptor_reward": sum(rewards) / max(len(rewards), 1),
        "mean_task_outcome": sum(o for g in groups for o in g.outcomes) / max(n_outcomes, 1),
        "perceptor": log_p,
        "world_model": log_w,
        "actor": log_a,
    }


def _require_trainable(cfg) -> None:
    if cfg.weights != "trainable":
        raise ValueError(
            f"Method '{cfg.method}' uses weights={cfg.weights!r}; parameter updates need weights: trainable "
            "(WEvA-Doc keeps the weights frozen and adapts through memory only)"
        )


def run_phase2(
    method_config: str | list[str],
    phase2_config: str,
    env,
    backend: TrainerBackend,
    controller_weights: str,
    overrides: list[str] | None = None,
    phase2_overrides: list[str] | None = None,
) -> list[dict]:
    from .agent import build_agent
    from .config import load_config

    if not Path(controller_weights).exists():
        raise FileNotFoundError(f"Phase II needs the Phase I controller weights: {controller_weights}")
    cfg = load_config(
        method_config,
        list(overrides or []) + ["controller.type=learned", f"controller.weights_path={controller_weights}"],
    )
    _require_trainable(cfg)
    p2 = load_phase2_config(phase2_config, phase2_overrides)
    agent = build_agent(cfg, getattr(backend, "llm", None))
    tasks = env.tasks(p2.split)
    out_dir = Path(p2.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict] = []
    seen: list[State] = []
    with open(out_dir / "phase2_log.jsonl", "w", encoding="utf-8") as fh:
        for r in range(p2.rounds):
            transitions, groups, _ = collect_round(agent, env, tasks, p2.grpo.group_size, cfg.run.max_steps)
            row = {"round": r, **phase2_update(backend, transitions, groups, p2, seen, agent.checker)}
            logs.append(row)
            fh.write(json.dumps(row, default=str) + "\n")
            log.info("round %d: %s", r, row)
    if hasattr(backend, "save"):
        backend.save(out_dir / "adapters")
    return logs


@dataclass
class Rollout:
    features: list[list[float]] = field(default_factory=list)
    action_idx: list[int] = field(default_factory=list)
    lookahead_unchanged: int = 0
    lookahead_total: int = 0
    performance: float = 0.0


def controller_reward(r: Rollout, lam: float = 0.2) -> float:
    frac = r.lookahead_unchanged / r.lookahead_total if r.lookahead_total else 0.0
    return r.performance - lam * frac


def reinforce_update(
    ctrl: LearnedController, rollouts: list[Rollout], lr: float = 0.1, lam: float = 0.2
) -> float:
    rewards = np.array([controller_reward(r, lam) for r in rollouts], dtype=float)
    baseline = rewards.mean() if len(rewards) else 0.0
    gW = np.zeros_like(ctrl.W)
    gb = np.zeros_like(ctrl.b)
    n_steps = 0
    for r, R in zip(rollouts, rewards):
        adv = R - baseline
        for x, a in zip(r.features, r.action_idx):
            x = np.asarray(x, dtype=float)
            p = ctrl.probs(x)
            onehot = np.zeros_like(p)
            onehot[a] = 1.0
            gW += adv * np.outer(onehot - p, x)
            gb += adv * (onehot - p)
            n_steps += 1
    if n_steps:
        ctrl.W += lr * gW / n_steps
        ctrl.b += lr * gb / n_steps
    return float(rewards.mean()) if len(rewards) else 0.0



class _RecordingController(LearnedController):
    def __init__(self, base: LearnedController):
        self.__dict__ = base.__dict__
        self.rollout = Rollout()

    def select(self, cs):
        a, i = self.sample(cs)
        self.rollout.features.append(cs.to_vector())
        self.rollout.action_idx.append(i)
        return a


def calibrate_controller(
    method_config: str | list[str],
    env,
    out: str = "outputs/controller.json",
    p1: Phase1Config | None = None,
    overrides: list[str] | None = None,
    backend: TrainerBackend | None = None,
    p2: Phase2Config | None = None,
) -> str:
    from .agent import WEvaAgent
    from .config import load_config
    from .evaluation import run_episode
    from .llm import create_llm

    p1 = p1 or Phase1Config()
    cfg = load_config(method_config, overrides)
    adapt_weights = backend is not None and cfg.weights == "trainable"
    if adapt_weights and p2 is None:
        p2 = Phase2Config()
    llm = getattr(backend, "llm", None) or create_llm(cfg.llm)
    ctrl = LearnedController(seed=cfg.run.seed)
    tasks = env.tasks(p1.split)[: p1.n_envs]
    history = []
    for it in range(p1.iterations):
        rollouts: list[Rollout] = []
        for task in tasks:
            rec = _RecordingController(ctrl)
            agent = WEvaAgent(cfg, llm, controller=rec)
            snap = backend.snapshot() if adapt_weights else None
            seen: list[State] = []
            for _ in range(p1.inner_steps):
                transitions, groups, _ = collect_round(agent, env, [task], p1.group_size, cfg.run.max_steps)
                if adapt_weights:
                    phase2_update(backend, transitions, groups, p2, seen, agent.checker)
            evals = [run_episode(agent, env, task, cfg.run.max_steps) for _ in range(p1.eval_episodes)]
            if snap is not None:
                backend.restore(snap)
            ro = rec.rollout
            records = [s for r in evals for s in r.records]
            ro.lookahead_total = sum(1 for s in records if s.meta_action == "LOOKAHEAD")
            ro.lookahead_unchanged = sum(
                1 for s in records if s.meta_action == "LOOKAHEAD" and not s.lookahead_changed_action
            )
            ro.performance = sum(float(r.success) - 0.5 * float(r.violation) for r in evals) / len(evals)
            rollouts.append(ro)
        mean_reward = reinforce_update(ctrl, rollouts, lr=p1.lr, lam=p1.lam)
        history.append({"iteration": it, "mean_reward": mean_reward})
        log.info("calibration iteration %d: mean post-adaptation reward %.3f", it, mean_reward)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ctrl.save(out)
    meta = {
        "config": method_config,
        "overrides": overrides or [],
        "phase1": dataclasses.asdict(p1),
        "weight_adaptation": adapt_weights,
        "task_ids": [t.id for t in tasks],
        "seed": cfg.run.seed,
        "history": history,
    }
    Path(out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


class NullBackend:
    def __init__(self, world_model=None):
        self.world_model = world_model
        self._trainable: set[str] = set()
        self.calls: list[tuple[str, frozenset[str]]] = []

    def set_trainable(self, modules: set[str]) -> None:
        self._trainable = set(modules)

    def trainable(self) -> set[str]:
        return set(self._trainable)

    def logprob(self, state: State, action: str, successor: State) -> float:
        self.calls.append(("logprob", frozenset(self._trainable)))
        if self.world_model is None:
            return 0.0
        pred = self.world_model.predict(state, action).state
        return math.log(max(sim_psi(pred, successor), 1e-3))

    def sample_inductions(self, transitions, n):
        return [
            PerceptorSample(t, [t.source] * n, t.source_prompt, [t.source.render()] * n) for t in transitions
        ]

    def update_perceptor(self, samples, cfg):
        self.calls.append(("perceptor", frozenset(self._trainable)))
        return {"n": len(samples)}

    def reinduce(self, transitions):
        self.calls.append(("reinduce", frozenset(self._trainable)))
        return list(transitions)

    def update_world_model(self, transitions, lr=None, epochs=1):
        self.calls.append(("world_model", frozenset(self._trainable)))
        return {"n": len(transitions)}

    def update_actor(self, groups, cfg):
        self.calls.append(("actor", frozenset(self._trainable)))
        return {"n_groups": len(groups)}

    def snapshot(self) -> dict:
        return {}

    def restore(self, snap: dict) -> None:
        return None


def make_torch_backend(model_name: str, p2: Phase2Config, adapters: str | None = None):
    from .torch_backend import TorchBackend, TorchBackendConfig

    bcfg = TorchBackendConfig(**{**p2.backend, "lr": p2.grpo.lr, "world_model_lr": p2.world_model_lr})
    return TorchBackend.from_pretrained(model_name, adapters=adapters, cfg=bcfg)
