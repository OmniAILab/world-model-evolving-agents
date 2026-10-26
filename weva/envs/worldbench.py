from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

from .base import Env, Task

JUDGED = frozenset({"evaluate_email", "evaluate_note"})


def _load(root: Path) -> dict[str, Any]:
    if not (root / "run_benchmark.py").is_file():
        raise FileNotFoundError(
            f"No WorldBench checkout at {root}. Clone https://github.com/OmniAILab/WorldBench, install its "
            "requirements.txt, and set WORLDBENCH_ROOT or pass root=..."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return {
        "apps": importlib.import_module("apps"),
        "env": importlib.import_module("core.env"),
        "sandbox": importlib.import_module("core.sandbox"),
        "runner": importlib.import_module("run_benchmark"),
        "evaluation": importlib.import_module("evaluation"),
    }


class WorldBenchEnv(Env):
    name = "worldbench"

    def __init__(
        self,
        root: str | None = None,
        benchmark_dir: str | None = None,
        languages: list[str] | None = None,
        personas: list[str] | None = None,
        subtasks: list[str] | None = None,
        max_tasks: int | None = None,
        max_obs_chars: int = 0,
        skip_judged: bool = False,
        work_dir: str = "outputs/worldbench",
    ):
        self.root = Path(root or os.environ.get("WORLDBENCH_ROOT", "WorldBench")).resolve()
        self.wb = _load(self.root)
        self.benchmark_dir = Path(benchmark_dir).resolve() if benchmark_dir else self.root / "data"
        self.languages = languages
        self.personas = personas
        self.subtasks = [str(s) for s in subtasks] if subtasks is not None else None
        self.max_tasks = max_tasks
        self.max_obs_chars = max_obs_chars
        self.skip_judged = skip_judged
        self.work_dir = Path(work_dir).resolve()
        self.env = None

    def tasks(self, split: str = "test") -> list[Task]:
        if split != "test":
            raise ValueError("WorldBench has a single test split")
        out = []
        for lang, persona, sid, config_path, testbed in self.wb["runner"].discover_subtasks(
            str(self.benchmark_dir)
        ):
            if self.languages and lang not in self.languages:
                continue
            if self.personas and persona not in self.personas:
                continue
            if self.subtasks is not None and sid not in self.subtasks:
                continue
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if self.skip_judged and any(e.get("function") in JUDGED for e in config.get("evaluation", [])):
                continue
            task_id = "/".join(x for x in (lang, persona, sid) if x)
            meta = {"config": config, "testbed": testbed, "language": lang}
            out.append(Task(id=task_id, goal=self._goal(config), metadata=meta))
        return out[: self.max_tasks] if self.max_tasks is not None else out

    def _goal(self, config: dict) -> str:
        apps = self.wb["apps"]
        lines = []
        if config.get("persona"):
            lines.append(f"PERSONA: {config['persona']}")
        if config.get("username"):
            lines.append(f"USER: {config['username']}")
        lines.append(f"TASK: {config.get('task', '')}")
        lines.append(
            "ACTIONS: one JSON object per action, on a single line. Files are under /testbed/data. "
            "Switch to an app before using its actions."
        )
        lines.append(
            " - {'app': 'system', 'action': 'switch_app', 'target_app': [ONE OF "
            + ", ".join(apps.AVAILABLE_APPS)
            + "]}"
        )
        for actions in apps.AVAILABLE_ACTIONS.values():
            lines.extend(f" - {module.DEMO}" for module in actions.values())
        lines.append(" - {'app': 'system', 'action': 'finish_task', 'answer': 'None'}")
        return "\n".join(lines)

    def reset(self, task: Task) -> str:
        wb = self.wb
        self.task = task
        self.config = task.metadata["config"]
        self.source = task.metadata["testbed"]
        self.out_dir = self.work_dir / task.id
        self.sandbox = wb["sandbox"].LocalSandbox(str(self.out_dir / "sandbox"))
        self.sandbox.setup(
            source_testbed_dir=self.source if os.path.isdir(self.source) else None,
            source_apps_dir=str(self.root / "apps"),
        )
        self.env = wb["env"].WorldBenchEnv(
            sandbox=self.sandbox,
            task=self.config.get("task", ""),
            verbose=False,
            max_obs_chars=self.max_obs_chars,
        )
        self.env.reset()
        self.changes = 0
        self.done = False
        files = sorted(p.name for p in Path(self.sandbox.testbed_dir, "data").iterdir())
        return self._render(f"Workspace ready. Files in /testbed/data: {', '.join(files)}")

    def _render(self, result: Any) -> str:
        return f"CURRENT APP: {self.env.current_app or 'none'}\nRESULT: {result}"

    def _preservation(self) -> dict:
        return self.wb["runner"].compute_preservation(self.source, self.sandbox.testbed_dir, self.config)

    def _progress(self) -> float:
        fns = self.wb["evaluation"].EVALUATION_FUNCTIONS
        initial = self.wb["runner"].INITIAL_STATE_FUNCTIONS
        items = [e for e in self.config.get("evaluation", []) if e.get("function") not in JUDGED]
        if not items:
            return 0.0
        passed = 0
        for e in items:
            fn = fns.get(e["function"])
            extra = {"initial_dir": self.source} if e["function"] in initial else {}
            with contextlib.suppress(Exception), contextlib.redirect_stdout(io.StringIO()):
                passed += bool(fn and fn(self.sandbox.testbed_dir, e["args"], **extra))
        return passed / len(items)

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        obs, _, done, info = self.env.step(action)
        changes = self._preservation()["n_non_target_changes"]
        violation = changes > self.changes
        self.changes = changes
        self.done = bool(done)
        return (
            self._render(obs),
            self.done,
            {
                "violation": violation,
                "action_executed": bool(info.get("action_executed", False)),
                "progress": self._progress(),
            },
        )

    def evaluate(self) -> dict[str, Any]:
        runner = self.wb["runner"]
        is_pass, details = runner.evaluate_task(self.config, self.sandbox.testbed_dir, self.source)
        preservation = self._preservation()
        self.env.dump_history(str(self.out_dir))
        return {
            "success": bool(is_pass),
            "tgc": float(is_pass),
            "cts": float(is_pass and preservation["preserved"]),
            "violation": not preservation["preserved"],
            "n_modified": preservation["n_modified"],
            "n_deleted": preservation["n_deleted"],
            "evaluations": details,
            "language": self.task.metadata["language"],
        }
