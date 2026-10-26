from __future__ import annotations

from typing import Any

from .base import Env, Task

INVALID_PREFIXES = ("No known action matches that input",)
SPLITS = {"train": "get_variations_train", "dev": "get_variations_dev", "test": "get_variations_test"}


class ScienceWorldEnv(Env):
    name = "scienceworld"

    def __init__(
        self,
        task_names: list[str] | None = None,
        max_variations: int | None = 5,
        step_limit: int = 100,
        simplification: str = "easy",
        calibration_split: str = "dev",
        train_split: str = "train",
        show_valid_actions: int = 0,
    ):
        try:
            from scienceworld import ScienceWorldEnv as _SW
        except ImportError as e:
            raise ImportError(
                "ScienceWorld is not installed: pip install -e '.[scienceworld]' (needs Java)"
            ) from e
        self.sw = _SW("", envStepLimit=step_limit)
        self.task_names = task_names or list(self.sw.get_task_names())
        self.max_variations = max_variations
        self.simplification = simplification
        self.split_alias = {"calibration": calibration_split, "train": train_split}
        self.show_valid_actions = show_valid_actions
        self.task: Task | None = None

    def tasks(self, split: str = "test") -> list[Task]:
        split = self.split_alias.get(split, split)
        if split not in SPLITS:
            raise ValueError(f"Unknown ScienceWorld split '{split}'")
        out: list[Task] = []
        for name in self.task_names:
            self.sw.load(name, 0, self.simplification)
            variations = list(getattr(self.sw, SPLITS[split])())
            if self.max_variations is not None:
                variations = variations[: self.max_variations]
            for v in variations:
                self.sw.load(name, v, self.simplification)
                out.append(
                    Task(
                        id=f"{name}-{v}",
                        goal=self.sw.get_task_description(),
                        metadata={"task": name, "variation": v},
                    )
                )
        return out

    def reset(self, task: Task) -> str:
        self.task = task
        self.sw.load(task.metadata["task"], task.metadata["variation"], self.simplification)
        obs, info = self.sw.reset()
        self.score = float(info.get("score", 0))
        self.done = False
        self.n_invalid = 0
        self.failed = False
        return self._render(obs, info)

    def _render(self, obs: str, info: dict[str, Any]) -> str:
        lines = [obs.strip()]
        if self.show_valid_actions and info.get("valid"):
            lines.append("VALID ACTIONS: " + "; ".join(list(info["valid"])[: self.show_valid_actions]))
        lines.append(f"SCORE: {int(self.score)} of 100")
        return "\n".join(lines)

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        obs, _, done, info = self.sw.step(action.strip())
        invalid = obs.strip().startswith(INVALID_PREFIXES)
        self.score = float(info.get("score", self.score))
        failed_now = self.score < 0 and not self.failed
        self.failed = self.failed or self.score < 0
        self.n_invalid += int(invalid)
        self.done = bool(done) or self.failed
        return (
            self._render(obs, info),
            self.done,
            {
                "violation": invalid or failed_now,
                "invalid_action": invalid,
                "task_failed": failed_now,
                "score": self.score,
                "progress": max(self.score, 0.0) / 100.0,
                "moves": info.get("moves"),
            },
        )

    def evaluate(self) -> dict[str, Any]:
        return {
            "success": self.score >= 100,
            "violation": self.failed,
            "score": max(self.score, 0.0) / 100.0,
            "task_failed": self.failed,
            "invalid_actions": self.n_invalid,
        }

    def close(self) -> None:
        self.sw.close()
