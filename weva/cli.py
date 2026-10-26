from __future__ import annotations

import argparse
import dataclasses
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from . import __version__
from .agent import build_agent
from .config import load_config
from .envs import create_env
from .evaluation import run_benchmark, summarise

REPO = Path(__file__).resolve().parents[1]


def _env_from_args(env_name: str | None, env_config: str | None):
    kwargs = {}
    if env_config:
        import yaml

        data = yaml.safe_load(Path(env_config).read_text())
        env_name = data["env"]
        kwargs = data.get("kwargs", {})
    return create_env(env_name or "toy-ledger", **kwargs), env_name or "toy-ledger", kwargs


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.set)
    _seed(cfg.run.seed)
    agent = build_agent(cfg)
    env, env_name, env_kwargs = _env_from_args(args.env, args.env_config)
    tasks = env.tasks(args.split)
    out_dir = Path(cfg.run.output_dir) / (args.name or Path(args.config[0]).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config_path": args.config,
        "overrides": args.set,
        "config": dataclasses.asdict(cfg),
        "env": env_name,
        "env_kwargs": env_kwargs,
        "split": args.split,
        "task_ids": [t.id for t in tasks],
        "seed": cfg.run.seed,
        "max_steps": cfg.run.max_steps,
        "weva_version": __version__,
        "git_commit": _git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    try:
        results = run_benchmark(
            agent,
            env,
            tasks=tasks,
            max_steps=cfg.run.max_steps,
            out_path=out_dir / "episodes.jsonl",
            log=(lambda m: print(m, flush=True)) if args.verbose else None,
        )
    finally:
        env.close()
    summary = summarise(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_demo(args: argparse.Namespace) -> None:
    rows = []
    for m in ("react", "weva_h0", "weva"):
        cfg = load_config([REPO / "configs" / "methods" / f"{m}.yaml", REPO / "configs" / "toy.yaml"])
        res = run_benchmark(build_agent(cfg), create_env("toy-ledger"), max_steps=cfg.run.max_steps)
        s = summarise(res)
        rows.append((m, s["success"], s["LER"], s["mean_env_steps"]))
    print(f"{'method':<10}{'success':>9}{'LER':>7}{'env steps':>11}")
    for m, su, le, st in rows:
        print(f"{m:<10}{su:>9.2f}{le:>7.2f}{st:>11.1f}")
    print("\n(scripted toy LLM; illustrates the mechanism, not the paper's benchmark results)")


def _make_backend(kind: str, cfg, p2, args):
    from .training import NullBackend, make_torch_backend

    if kind == "torch":
        return make_torch_backend(cfg.llm.model, p2, adapters=getattr(args, "adapters", None))
    if kind == "null":
        from .components import WorldModel
        from .llm import create_llm

        return NullBackend(WorldModel(create_llm(cfg.llm)))
    return None


def cmd_train_controller(args: argparse.Namespace) -> None:
    from .training import calibrate_controller, load_phase1_config, load_phase2_config

    cfg = load_config(args.config, args.set)
    p1 = load_phase1_config(args.training, args.training_set)
    p2 = load_phase2_config(args.phase2, args.phase2_set)
    env, _, _ = _env_from_args(args.env, args.env_config)
    backend = _make_backend(args.backend, cfg, p2, args)
    try:
        path = calibrate_controller(args.config, env, args.out, p1, args.set, backend, p2)
    finally:
        env.close()
    print(f"saved controller weights to {path}")


def cmd_train_phase2(args: argparse.Namespace) -> None:
    from .training import load_phase2_config, run_phase2

    cfg = load_config(args.config, args.set)
    p2 = load_phase2_config(args.training, args.training_set)
    env, _, _ = _env_from_args(args.env, args.env_config)
    backend = _make_backend(args.backend, cfg, p2, args)
    try:
        logs = run_phase2(
            args.config, args.training, env, backend, args.controller, args.set, args.training_set
        )
    finally:
        env.close()
    print(json.dumps(logs[-1] if logs else {}, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="weva")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="evaluate a method config on an environment")
    r.add_argument("--config", nargs="+", required=True)
    r.add_argument("--env", default=None)
    r.add_argument("--env-config", default=None)
    r.add_argument("--split", default="test")
    r.add_argument("--name", default=None)
    r.add_argument("--set", nargs="*", default=[], help="overrides, e.g. lookahead.horizon=3")
    r.add_argument("--verbose", action="store_true", help="print one line per executed action")
    r.set_defaults(fn=cmd_run)

    d = sub.add_parser("demo", help="offline toy demo: ReAct vs WEvA-H0 vs WEvA")
    d.set_defaults(fn=cmd_demo)

    t = sub.add_parser(
        "train-controller", help="Phase I: calibrate the Controller on post-adaptation performance"
    )
    t.add_argument("--config", nargs="+", default=["configs/methods/weva.yaml"])
    t.add_argument("--env", default=None)
    t.add_argument("--env-config", default=None)
    t.add_argument("--training", default="configs/training/phase1.yaml")
    t.add_argument("--phase2", default="configs/training/phase2.yaml")
    t.add_argument("--backend", choices=["torch", "null", "none"], default="none")
    t.add_argument("--out", default="outputs/controller.json")
    t.add_argument("--set", nargs="*", default=[])
    t.add_argument("--training-set", nargs="*", default=[])
    t.add_argument("--phase2-set", nargs="*", default=[])
    t.set_defaults(fn=cmd_train_controller)

    t2 = sub.add_parser("train-phase2", help="Phase II: alternating Perceptor / World Model / Actor updates")
    t2.add_argument("--config", nargs="+", default=["configs/methods/weva.yaml"])
    t2.add_argument("--training", default="configs/training/phase2.yaml")
    t2.add_argument("--controller", default="outputs/controller.json")
    t2.add_argument("--adapters", default=None)
    t2.add_argument("--env", default=None)
    t2.add_argument("--env-config", default=None)
    t2.add_argument("--backend", choices=["torch", "null"], default="torch")
    t2.add_argument("--set", nargs="*", default=[])
    t2.add_argument("--training-set", nargs="*", default=[])
    t2.set_defaults(fn=cmd_train_phase2)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
