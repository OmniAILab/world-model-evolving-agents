from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .components import Actor, WorldModel
from .config import LookaheadConfig
from .state import default_progress, plain
from .types import Branch, Prediction, State


def _tokens(s: str) -> set[str]:
    return set(plain(s).split())


Checker = Callable[[str, State, str, State | None], bool]


def declared_violation(c: str, pred: Prediction, match_threshold: float = 0.5) -> bool:
    ct = _tokens(c)
    if not ct:
        return False
    for v in pred.violated:
        ft = _tokens(v)
        union = ct | ft
        if union and len(ct & ft) / len(union) >= match_threshold:
            return True
    return False


def active_preconditions(source: State, required: list[str] | tuple[str, ...] = ()) -> list[str]:
    out = [c for c in source.C if _tokens(c)]
    seen = {plain(c) for c in out}
    for c in required:
        if plain(c) not in seen and _tokens(c):
            out.append(c)
            seen.add(plain(c))
    return out


def count_violations(
    source: State,
    pred: Prediction,
    match_threshold: float = 0.5,
    action: str | None = None,
    checker: Checker | None = None,
    required: list[str] | tuple[str, ...] = (),
    mode: str = "declared",
) -> int:
    n = 0
    for c in active_preconditions(source, required):
        declared = mode in ("declared", "both") and declared_violation(c, pred, match_threshold)
        checked = (
            not declared
            and mode in ("checker", "both")
            and checker is not None
            and action is not None
            and checker(c, source, action, pred.state)
        )
        n += int(declared or checked)
    return n


@dataclass
class SearchResult:
    action: str | None
    default_action: str | None
    first_prediction: Prediction | None = None
    branch: Branch | None = None
    n_expanded: int = 0
    n_pruned: int = 0
    n_root_candidates: int = 0
    survivors: list[Branch] = field(default_factory=list)
    root_prompt: str | None = None
    root_output: str | None = None


class VerifiedLookahead:
    def __init__(
        self,
        actor: Actor,
        world_model: WorldModel,
        cfg: LookaheadConfig,
        progress_fn: Callable[[State], float] = default_progress,
        checker: Checker | None = None,
    ):
        self.actor = actor
        self.wm = world_model
        self.cfg = cfg
        self.progress_fn = progress_fn
        self.checker = checker

    def search(
        self, goal: str, state: State, memory: str = "", required: list[str] | tuple[str, ...] = ()
    ) -> SearchResult:
        cfg = self.cfg
        frontier = [Branch(actions=[], states=[state], progress=self.progress_fn(state))]
        expanded = pruned = 0
        default_action: str | None = None
        n_root = 0
        root_prompt: str | None = None
        root_output: str | None = None
        for depth in range(cfg.horizon):
            candidates: list[Branch] = []
            for b in frontier:
                last = b.states[-1]
                acts = self.actor.propose(goal, last, memory, cfg.branching)
                if depth == 0:
                    n_root = len(acts)
                    default_action = acts[0] if acts else None
                    root_prompt, root_output = self.actor.last_prompt, self.actor.last_output
                for a in acts:
                    pred = self.wm.predict(last, a)
                    expanded += 1
                    v = (
                        count_violations(
                            last, pred, cfg.violation_match, a, self.checker, required, cfg.verifier
                        )
                        if cfg.verify
                        else 0
                    )
                    if cfg.verify and v > 0:
                        pruned += 1
                        continue
                    candidates.append(
                        Branch(
                            actions=b.actions + [a],
                            states=b.states + [pred.state],
                            progress=self.progress_fn(pred.state),
                            violations=b.violations + v,
                            predictions=b.predictions + [pred],
                        )
                    )
            if not candidates:
                frontier = []
                break
            candidates.sort(key=lambda br: -br.progress)
            frontier = candidates[: cfg.beam]
        if not frontier or len(frontier[0].actions) < cfg.horizon:
            return SearchResult(
                None, default_action, None, None, expanded, pruned, n_root, [], root_prompt, root_output
            )
        best = frontier[0]
        return SearchResult(
            action=best.actions[0],
            default_action=default_action,
            first_prediction=best.predictions[0],
            branch=best,
            n_expanded=expanded,
            n_pruned=pruned,
            n_root_candidates=n_root,
            survivors=frontier,
            root_prompt=root_prompt,
            root_output=root_output,
        )
