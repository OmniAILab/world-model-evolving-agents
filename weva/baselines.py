from __future__ import annotations

from typing import Any, Protocol

from .llm import LLM, CountingLLM
from .prompts import render
from .types import StepRecord


class Agent(Protocol):
    def start_episode(self, goal: str) -> None: ...
    def act(self, observation: str) -> str: ...
    def observe_outcome(self, info: dict[str, Any]) -> None: ...
    def end_episode(self, success: bool | None = None, observation: str | None = None) -> Any: ...
    def freeze(self, flag: bool = True) -> None: ...


class ReActAgent:
    def __init__(self, llm: LLM, max_history: int = 8):
        self.llm = llm if isinstance(llm, CountingLLM) else CountingLLM(llm)
        self.max_history = max_history
        self.trace: list[StepRecord] = []
        self.transitions: list = []
        self.goal = ""
        self._history: list[str] = []
        self._t = 0

    def start_episode(self, goal: str) -> None:
        self.goal, self._history, self._t = goal, [], 0
        self.trace = []

    def act(self, observation: str) -> str:
        prompt = render(
            "react",
            goal=self.goal,
            observation=observation,
            history=" | ".join(self._history[-self.max_history :]) or "none",
        )
        out = self.llm.generate(prompt, role="react", temperature=0.0, max_tokens=64)
        action = out.strip().splitlines()[0] if out.strip() else "noop"
        self._history.append(action)
        self.trace.append(StepRecord(t=self._t, meta_action="ACT", action=action, trust=1.0))
        self._t += 1
        return action

    def observe_outcome(self, info: dict[str, Any]) -> None:
        pass

    def end_episode(self, success: bool | None = None, observation: str | None = None) -> dict:
        return {}

    def freeze(self, flag: bool = True) -> None:
        pass

    @property
    def internal_calls(self) -> int:
        return self.llm.total
