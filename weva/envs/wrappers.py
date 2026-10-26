from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Env, Task


class DynamicsShift(Env):
    def __init__(self, env: Env, at_step: int, mutate: Callable[[Env], None]):
        self.env, self.at_step, self.mutate = env, at_step, mutate
        self.cum_steps = 0
        self.shifted = False
        self.name = f"{env.name}-shift"

    def tasks(self, split: str = "test") -> list[Task]:
        return self.env.tasks(split)

    def reset(self, task: Task) -> str:
        return self.env.reset(task)

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        if not self.shifted and self.cum_steps + 1 >= self.at_step:
            self.mutate(self.env)
            self.shifted = True
        self.cum_steps += 1
        return self.env.step(action)

    def evaluate(self) -> dict[str, Any]:
        return self.env.evaluate()
