from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    id: str
    goal: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Env(ABC):
    name: str = "env"

    @abstractmethod
    def tasks(self, split: str = "test") -> list[Task]: ...

    @abstractmethod
    def reset(self, task: Task) -> str: ...

    @abstractmethod
    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]: ...

    @abstractmethod
    def evaluate(self) -> dict[str, Any]: ...

    def close(self) -> None:
        return None
