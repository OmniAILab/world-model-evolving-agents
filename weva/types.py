from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

FIELDS = ("V", "C", "E")


@dataclass(frozen=True)
class State:
    V: tuple[str, ...] = ()
    C: tuple[str, ...] = ()
    E: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for f in FIELDS:
            object.__setattr__(self, f, tuple(getattr(self, f)))

    def get(self, name: str) -> tuple[str, ...]:
        return getattr(self, name)

    def replace(self, **kwargs: Any) -> State:
        data = {f: getattr(self, f) for f in FIELDS}
        data.update(kwargs)
        return State(**data)

    def is_empty(self) -> bool:
        return not (self.V or self.C or self.E)

    def render(self) -> str:
        return "\n".join(f"{f}: {'; '.join(self.get(f))}" for f in FIELDS)


class MetaAction(str, Enum):
    LOOKAHEAD = "LOOKAHEAD"
    ACT = "ACT"
    ABSTRACT = "ABSTRACT"
    RETRIEVE = "RETRIEVE"
    ROLLBACK = "ROLLBACK"


@dataclass
class Prediction:
    state: State
    violated: tuple[str, ...] = ()


@dataclass
class Transition:
    source: State
    action: str
    target: State
    step: int = 0
    predicted: State | None = None
    agreement: float | None = None
    verified_violation: bool = False
    failure: bool = False
    success: bool | None = None
    observation: str | None = None
    next_observation: str | None = None
    source_prompt: str | None = None
    target_prompt: str | None = None
    actor_prompt: str | None = None
    actor_output: str | None = None
    violated_labels: tuple[str, ...] | None = None
    source_confidence: float | None = None
    outcome: float | None = None


@dataclass
class Branch:
    actions: list[str]
    states: list[State]
    progress: float
    violations: int = 0
    predictions: list[Prediction] = field(default_factory=list)


@dataclass
class ControlState:
    confidence: float = 0.5
    contradiction: float = 0.0
    novelty: float = 1.0
    progress: float = 0.0
    trust: float = 0.7

    def to_vector(self) -> list[float]:
        return [self.confidence, self.contradiction, self.novelty, self.progress, self.trust]


@dataclass
class StepRecord:
    t: int
    meta_action: str
    action: str
    trust: float
    field_trust: dict[str, float] = field(default_factory=dict)
    n_expanded: int = 0
    n_pruned: int = 0
    lookahead_changed_action: bool = False
    diagnosis: str | None = None


@dataclass
class EpisodeResult:
    task_id: str
    success: bool
    violation: bool
    env_steps: int
    internal_calls: int = 0
    records: list[StepRecord] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
