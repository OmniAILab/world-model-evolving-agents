from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from .state import field_similarity, sim_psi, strip_terminal
from .types import FIELDS, State


class ModelTrust:
    def __init__(
        self,
        window: int = 8,
        initial: float = 0.7,
        weights: Mapping[str, float] | None = None,
    ):
        self.window = window
        self.initial = initial
        self.weights = dict(weights or {})
        self._overall: deque[float] = deque(maxlen=window)
        self._by_field: dict[str, deque[float]] = {f: deque(maxlen=window) for f in FIELDS}
        self.history: list[float] = []

    @property
    def n(self) -> int:
        return len(self._overall)

    @property
    def mu(self) -> float:
        return sum(self._overall) / len(self._overall) if self._overall else self.initial

    def mu_field(self, name: str) -> float:
        d = self._by_field[name]
        return sum(d) / len(d) if d else 1.0

    def field_trust(self) -> dict[str, float]:
        return {f: self.mu_field(f) for f in FIELDS}

    def update(self, predicted: State, realised: State) -> float:
        s = sim_psi(predicted, realised, weights=self.weights)
        self._overall.append(s)
        for f in FIELDS:
            a, b = predicted.get(f), realised.get(f)
            if f == "E":
                a, b = strip_terminal(a), strip_terminal(b)
            self._by_field[f].append(field_similarity(a, b))
        self.history.append(self.mu)
        return s

    def diagnose(self, field_level: bool = True) -> str | None:
        if not self._overall:
            return None
        if not field_level:
            return "scalar"
        return min(FIELDS, key=lambda f: (self.mu_field(f), FIELDS.index(f)))

    def reset(self) -> None:
        self._overall.clear()
        for d in self._by_field.values():
            d.clear()
