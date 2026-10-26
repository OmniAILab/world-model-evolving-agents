from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import MemoryConfig
from .state import plain, sim_psi
from .types import State, Transition

Checker = Callable[[str, State, str, State | None], bool]
Proposer = Callable[[Transition, list[Transition]], str | None]


def verb(action: str) -> str:
    toks = plain(action).split()
    return toks[0] if toks else ""


def format_transition(t: Transition, compressed: bool) -> str:
    added = [e for e in t.target.V if e not in t.source.V]
    removed = [e for e in t.source.V if e not in t.target.V]
    effect = "; ".join([f"+{e}" for e in added] + [f"-{e}" for e in removed]) or "no visible change"
    if compressed:
        return f"{t.action} -> {effect}"
    return f"from [{'; '.join(t.source.V)}] doing '{t.action}' led to [{'; '.join(t.target.V)}] ({effect})"


@dataclass
class Precondition:
    text: str
    evidence: int = 0
    active: bool = False


@dataclass
class RetrievalPolicy:
    levels: tuple[str, ...]
    k: int
    compressed: bool


@dataclass
class RetrievedContext:
    text: str = ""
    transitions: list[Transition] = field(default_factory=list)
    regularities: list[str] = field(default_factory=list)


NO_RETRIEVAL = RetrievalPolicy((), 0, True)
EPISODIC_ONLY = RetrievalPolicy(("episodic",), 3, True)
COMBINED = RetrievalPolicy(("episodic", "semantic"), 5, False)


class MemoryManager:
    def __init__(
        self,
        cfg: MemoryConfig | None = None,
        checker: Checker | None = None,
        proposer: Proposer | None = None,
    ):
        self.cfg = cfg or MemoryConfig()
        self.checker = checker
        self.proposer = proposer
        self.episodic: list[Transition] = []
        self.semantic: dict[str, Counter[str]] = defaultdict(Counter)
        self.preconditions: dict[str, Precondition] = {}
        self._log: list[Transition] = []
        self._failures: list[Transition] = []
        self._successes: list[Transition] = []
        self._disagreements: dict[str, list[Transition]] = defaultdict(list)
        self.frozen = False

    def record(self, t: Transition) -> None:
        if self.frozen:
            return
        self._log.append(t)
        self._log = self._log[-500:]
        (self._failures if t.failure else self._successes).append(t)
        if "episodic" in self.cfg.levels:
            self.episodic.append(t)
        if "semantic" in self.cfg.levels:
            for e in t.target.V:
                if e not in t.source.V:
                    self.semantic[verb(t.action)][f"adds {e}"] += 1
            for e in t.source.V:
                if e not in t.target.V:
                    self.semantic[verb(t.action)][f"removes {e}"] += 1
        if "procedural" in self.cfg.levels:
            self._update_evidence(t)
            if t.failure:
                self._maybe_propose(t)

    def _update_evidence(self, t: Transition) -> None:
        if self.checker is None:
            return
        for p in self.preconditions.values():
            if self.checker(p.text, t.source, t.action, t.target):
                p.evidence += 1 if t.failure else -1

    def _maybe_propose(self, t: Transition) -> None:
        if self.proposer is None:
            return
        key = verb(t.action)
        pool = self._disagreements[key]
        pool.append(t)
        similar = [o for o in pool[:-1] if sim_psi(o.source, t.source) >= self.cfg.recurrence_similarity]
        if len(similar) + 1 < self.cfg.recurrence_min:
            return
        text = self.proposer(t, similar[-3:])
        if text and plain(text) not in {plain(k) for k in self.preconditions}:
            self.preconditions[text] = Precondition(text=text, evidence=len(similar) + 1)

    def end_episode(self) -> dict[str, list[str]]:
        promoted: list[str] = []
        removed: list[str] = []
        if self.frozen or "procedural" not in self.cfg.levels:
            return {"promoted": promoted, "removed": removed}
        failures = self._failures
        successes = self._successes
        for key, p in list(self.preconditions.items()):
            if p.active:
                if p.evidence <= self.cfg.demote_threshold:
                    del self.preconditions[key]
                    removed.append(p.text)
                continue
            if p.evidence <= self.cfg.demote_threshold:
                del self.preconditions[key]
                removed.append(p.text)
                continue
            if p.evidence < self.cfg.promote_threshold:
                continue
            if self.checker is None:
                continue
            rejects_failure = any(self.checker(p.text, t.source, t.action, t.target) for t in failures)
            contradicts = any(self.checker(p.text, t.source, t.action, t.target) for t in successes)
            if rejects_failure and not contradicts:
                p.active = True
                promoted.append(p.text)
            elif contradicts:
                del self.preconditions[key]
                removed.append(p.text)
        return {"promoted": promoted, "removed": removed}

    def active_preconditions(self) -> list[str]:
        if "procedural" not in self.cfg.levels:
            return []
        return [p.text for p in self.preconditions.values() if p.active]

    def novelty(self, state: State) -> float:
        if not self.episodic:
            return 1.0
        return 1.0 - max(sim_psi(state, t.source) for t in self.episodic[-200:])

    def select_policy(self, trust: float) -> RetrievalPolicy:
        mode = self.cfg.retrieval_policy
        if mode == "none":
            return NO_RETRIEVAL
        if mode == "fixed":
            return RetrievalPolicy(("episodic",), self.cfg.retrieval_k, True)
        if trust < self.cfg.retrieval_low:
            return COMBINED
        if trust < self.cfg.retrieval_high:
            return EPISODIC_ONLY
        return NO_RETRIEVAL

    def retrieve(
        self, state: State, trust: float = 1.0, policy: RetrievalPolicy | None = None
    ) -> RetrievedContext:
        policy = policy or self.select_policy(trust)
        ctx = RetrievedContext()
        if policy.k <= 0:
            return ctx
        if "episodic" in policy.levels and self.episodic:
            scored = sorted(self.episodic[-500:], key=lambda t: -sim_psi(state, t.source))
            ctx.transitions = scored[: policy.k]
        if "semantic" in policy.levels:
            for v, counts in self.semantic.items():
                top = ", ".join(f"{k} (x{n})" for k, n in counts.most_common(2))
                ctx.regularities.append(f"{v}: {top}")
            ctx.regularities = ctx.regularities[: policy.k]
        lines = [format_transition(t, policy.compressed) for t in ctx.transitions] + ctx.regularities
        ctx.text = " | ".join(lines)
        return ctx

    def freeze(self, flag: bool = True) -> None:
        self.frozen = flag

    def snapshot(self) -> dict:
        return {
            "episodic": list(self.episodic),
            "semantic": copy.deepcopy(self.semantic),
            "preconditions": copy.deepcopy(self.preconditions),
            "log": list(self._log),
            "failures": list(self._failures),
            "successes": list(self._successes),
            "disagreements": {k: list(v) for k, v in self._disagreements.items()},
            "frozen": self.frozen,
        }

    def restore(self, snap: dict) -> None:
        self.episodic = list(snap["episodic"])
        self.semantic = copy.deepcopy(snap["semantic"])
        self.preconditions = copy.deepcopy(snap["preconditions"])
        self._log = list(snap["log"])
        self._failures = list(snap["failures"])
        self._successes = list(snap["successes"])
        self._disagreements = defaultdict(list, {k: list(v) for k, v in snap["disagreements"].items()})
        self.frozen = snap["frozen"]

    def state_dict(self) -> dict:
        return {
            "preconditions": [
                {"text": p.text, "evidence": p.evidence, "active": p.active}
                for p in self.preconditions.values()
            ],
            "semantic": {k: dict(v) for k, v in self.semantic.items()},
            "n_episodic": len(self.episodic),
        }

    def load_state_dict(self, d: dict) -> None:
        self.preconditions = {p["text"]: Precondition(**p) for p in d.get("preconditions", [])}
        self.semantic = defaultdict(Counter, {k: Counter(v) for k, v in d.get("semantic", {}).items()})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2))

    def load(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text()))



def llm_proposer(llm) -> Proposer:
    from .prompts import render

    def propose(t: Transition, similar: list[Transition]) -> str | None:
        if t.predicted is None:
            return None
        prompt = render(
            "precondition",
            action=t.action,
            predicted=t.predicted.render(),
            realised=t.target.render(),
            similar=" | ".join(format_transition(s, True) for s in similar) or "none",
        )
        out = llm.generate(prompt, role="precondition", temperature=0.0, max_tokens=64)
        text = out.split("PRECONDITION:", 1)[-1].strip().strip(".")
        return text or None

    return propose
