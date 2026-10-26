from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .envs.base import Env, Task
from .state import field_similarity, sim_psi, strip_terminal
from .types import FIELDS, EpisodeResult, Transition


def ler(results: Iterable[EpisodeResult]) -> float:
    rs = list(results)
    return sum(1 for r in rs if r.violation) / len(rs) if rs else 0.0


def macro_ler(by_env: dict[str, list[EpisodeResult]]) -> float:
    vals = [ler(rs) for rs in by_env.values() if rs]
    return sum(vals) / len(vals) if vals else 0.0


def success_rate(results: Iterable[EpisodeResult]) -> float:
    rs = list(results)
    return sum(1 for r in rs if r.success) / len(rs) if rs else 0.0


def transition_fidelity(transitions: Iterable[Transition]) -> dict[str, float]:
    ts = [t for t in transitions if t.predicted is not None]
    if not ts:
        return {"overall": float("nan"), **{f: float("nan") for f in FIELDS}}
    out = {"overall": sum(sim_psi(t.predicted, t.target) for t in ts) / len(ts)}
    for f in FIELDS:
        vals = []
        for t in ts:
            a, b = t.predicted.get(f), t.target.get(f)
            if f == "E":
                a, b = strip_terminal(a), strip_terminal(b)
            vals.append(field_similarity(a, b))
        out[f] = sum(vals) / len(vals)
    return out


def first_crossing(series: Sequence[float], level: float, below: bool = True) -> int | None:
    for i, v in enumerate(series):
        if (v < level) if below else (v > level):
            return i
    return None


def detection_lead(
    trust_series: Sequence[float],
    success_series: Sequence[float],
    alert: float,
    degradation: float,
) -> int | None:
    t = first_crossing(trust_series, alert, below=True)
    s = first_crossing(success_series, degradation, below=True)
    if t is None or s is None:
        return None
    return s - t


def steps_to_threshold(curve: Sequence[float], threshold: float) -> int | None:
    i = first_crossing(curve, threshold - 1e-12, below=False)
    return None if i is None else i + 1



def _log_step(log: Callable[[str], None], agent: Any, i: int, max_steps: int, obs: str) -> None:
    rec = agent.trace[-1] if getattr(agent, "trace", None) else None
    head = f"  [step {i}/{max_steps}]"
    if rec is not None:
        head += f" {rec.meta_action:<9} trust={rec.trust:.2f} pruned={rec.n_pruned}"
        action = rec.action
    else:
        action = ""
    flat = " ".join(str(obs).split())
    log(f"{head} {action[:90]} -> {flat[:90]}")


def run_episode(
    agent: Any,
    env: Env,
    task: Task,
    max_steps: int = 30,
    log: Callable[[str], None] | None = None,
) -> EpisodeResult:
    obs = env.reset(task)
    calls_before = getattr(agent, "internal_calls", 0)
    agent.start_episode(task.goal)
    env_steps = 0
    for _ in range(max_steps):
        action = agent.act(obs)
        obs, done, info = env.step(action)
        env_steps += 1
        agent.observe_outcome(info)
        if log:
            _log_step(log, agent, env_steps, max_steps, obs)
        if done:
            break
    final = env.evaluate()
    agent.end_episode(success=final["success"], observation=obs)
    return EpisodeResult(
        task_id=task.id,
        success=bool(final["success"]),
        violation=bool(final["violation"]),
        env_steps=env_steps,
        internal_calls=getattr(agent, "internal_calls", 0) - calls_before,
        records=list(getattr(agent, "trace", [])),
        transitions=list(getattr(agent, "transitions", [])),
        extra=dict(final),
    )


def run_benchmark(
    agent: Any,
    env: Env,
    tasks: list[Task] | None = None,
    max_steps: int = 30,
    out_path: str | Path | None = None,
    on_episode: Callable[[EpisodeResult], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[EpisodeResult]:
    tasks = tasks if tasks is not None else env.tasks()
    results: list[EpisodeResult] = []
    with open(out_path, "w", encoding="utf-8") if out_path else nullcontext() as fh:
        for task in tasks:
            if log:
                log(f"=== {task.id}")
            r = run_episode(agent, env, task, max_steps, log)
            if log:
                extra = {k: r.extra[k] for k in ("tgc", "cts") if k in r.extra}
                log(f"  success={r.success} violation={r.violation} steps={r.env_steps} {extra}")
            results.append(r)
            if on_episode:
                on_episode(r)
            if fh:
                row = {
                    "task_id": r.task_id,
                    "success": r.success,
                    "violation": r.violation,
                    "env_steps": r.env_steps,
                    "internal_calls": r.internal_calls,
                    "lookahead_expansions": sum(s.n_expanded for s in r.records),
                    "trace": [asdict(s) for s in r.records],
                }
                fh.write(json.dumps(row) + "\n")
    return results


def scenario_goal_completion(results: list[EpisodeResult]) -> float | None:
    scenarios: dict[str, list[bool]] = {}
    for r in results:
        sid = r.extra.get("scenario_id")
        if sid is not None:
            scenarios.setdefault(sid, []).append(bool(r.success))
    if not scenarios:
        return None
    return sum(all(v) for v in scenarios.values()) / len(scenarios)


def _mean_extra(results: list[EpisodeResult], key: str) -> float | None:
    vals = [float(r.extra[key]) for r in results if key in r.extra and r.extra[key] is not None]
    return sum(vals) / len(vals) if vals else None


def summarise(results: list[EpisodeResult]) -> dict[str, float]:
    tf = transition_fidelity([t for r in results for t in r.transitions])
    out = {
        "n": len(results),
        "success": success_rate(results),
        "LER": ler(results),
        "mean_env_steps": sum(r.env_steps for r in results) / max(len(results), 1),
        "mean_internal_calls": sum(r.internal_calls for r in results) / max(len(results), 1),
        "mean_lookahead_expansions": sum(s.n_expanded for r in results for s in r.records)
        / max(len(results), 1),
        "TF": tf["overall"],
        "TF_V": tf["V"],
        "TF_C": tf["C"],
        "TF_E": tf["E"],
    }
    for key, name in (("score", "score"), ("tgc", "TGC"), ("cts", "CTS")):
        v = _mean_extra(results, key)
        if v is not None:
            out[name] = v
    sgc = scenario_goal_completion(results)
    if sgc is not None:
        out["SGC"] = sgc
    return out
