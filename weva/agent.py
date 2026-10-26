from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from .baselines import ReActAgent
from .components import Actor, Controller, Perceptor, WorldModel, create_controller, llm_checker
from .config import WEvaConfig
from .llm import LLM, CountingLLM, create_llm
from .memory import COMBINED, MemoryManager, llm_proposer
from .search import VerifiedLookahead, active_preconditions
from .state import default_progress
from .trust import ModelTrust
from .types import ControlState, MetaAction, State, StepRecord, Transition


class WEvaAgent:
    def __init__(
        self,
        cfg: WEvaConfig,
        llm: LLM,
        memory: MemoryManager | None = None,
        controller: Controller | None = None,
        progress_fn: Callable[[State], float] = default_progress,
    ):
        self.cfg = cfg
        self.llm = llm if isinstance(llm, CountingLLM) else CountingLLM(llm)
        mode = cfg.state.representation
        self.perceptor = Perceptor(self.llm, mode)
        self.world_model = WorldModel(self.llm, mode)
        self.actor = Actor(self.llm, cfg.llm.actor_temperature, cfg.llm.actor_top_p)
        self.checker = llm_checker(self.llm)
        self.search = VerifiedLookahead(
            self.actor, self.world_model, cfg.lookahead, progress_fn, self.checker
        )
        self.controller = controller or create_controller(cfg.controller, self.llm)
        self.memory = memory or MemoryManager(
            cfg.memory, checker=self.checker, proposer=llm_proposer(self.llm)
        )
        self.progress_fn = progress_fn
        self.trust = ModelTrust(cfg.controller.window, cfg.controller.initial_trust, cfg.state.field_weights)
        self.trace: list[StepRecord] = []
        self.transitions: list[Transition] = []
        self.reset()

    def reset(self) -> None:
        self.goal = ""
        self._t = 0
        self._window: list[str] = []
        self._window_action: str | None = None
        self._pending: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}
        self._confidence = 0.5
        self._contradiction = 0.0
        self._progress = 0.0
        self._last_consistent: State | None = None

    def start_episode(self, goal: str) -> None:
        self.reset()
        self.trust.reset()
        self.goal = goal
        self.trace = []
        self.transitions = []

    def observe_outcome(self, info: dict[str, Any]) -> None:
        self._last_info = info or {}
        if "progress" in self._last_info:
            self._progress = float(self._last_info["progress"])

    def end_episode(
        self, success: bool | None = None, observation: str | None = None
    ) -> dict[str, list[str]]:
        if observation is not None and self._pending is not None:
            self._close_window(observation)
            sigma = self._perceive(observation)
            self._close_transition(sigma, observation)
            self._pending = None
        if success is not None and self.transitions:
            self.transitions[-1].success = success
        return self.memory.end_episode()

    def freeze(self, flag: bool = True) -> None:
        self.memory.freeze(flag)

    def snapshot(self) -> dict[str, Any]:
        return {
            "memory": self.memory.snapshot(),
            "controller": copy.deepcopy(self.controller.get_state()),
            "trust": copy.deepcopy(self.trust),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        self.memory.restore(snap["memory"])
        self.controller.set_state(copy.deepcopy(snap["controller"]))
        self.trust = copy.deepcopy(snap["trust"])

    @property
    def internal_calls(self) -> int:
        return self.llm.total

    def _close_window(self, observation: str) -> None:
        if self._window_action is None:
            return
        n = self.cfg.state.window_obs_chars
        obs = " / ".join(line.strip() for line in observation.splitlines() if line.strip())
        entry = self._window_action if n <= 0 else f"{self._window_action} -> {obs[:n]}"
        self._window.append(entry)
        self._window_action = None

    def _perceive(self, observation: str, focus: str | None = None) -> State:
        s = self.perceptor.induce(
            self.goal,
            observation,
            self._window[-self.cfg.state.window :],
            self.memory.active_preconditions(),
            focus=focus,
        )
        promoted = [c for c in self.memory.active_preconditions() if c not in s.C]
        return s.replace(C=s.C + tuple(promoted)) if promoted else s

    def _contradiction_ratio(self, source: State, action: str, realised: State) -> float:
        constraints = active_preconditions(realised, self.memory.active_preconditions())
        if "violation" in self._last_info and not self._last_info["violation"]:
            return 0.0
        hits = sum(int(self.checker(c, source, action, realised)) for c in constraints)
        return hits / max(1, len(constraints))

    def _close_transition(self, sigma: State, observation: str) -> Transition:
        p = self._pending
        predicted = p["predicted"]
        agreement = self.trust.update(predicted, sigma) if predicted is not None else None
        tr = Transition(
            source=p["source"],
            action=p["action"],
            target=sigma,
            step=self._t - 1,
            predicted=predicted,
            agreement=agreement,
            verified_violation=bool(self._last_info.get("violation", False)),
            observation=p["observation"],
            next_observation=observation,
            source_prompt=p["source_prompt"],
            target_prompt=self.perceptor.last_prompt,
            actor_prompt=p["actor_prompt"],
            actor_output=p["actor_output"],
            source_confidence=p["source_confidence"],
        )
        tr.failure = tr.verified_violation or (
            agreement is not None and agreement < self.cfg.memory.disagree_threshold
        )
        self.memory.record(tr)
        self.transitions.append(tr)
        if not tr.failure:
            self._last_consistent = sigma
        self._contradiction = self._contradiction_ratio(tr.source, tr.action, sigma)
        return tr

    def act(self, observation: str) -> str:
        cfg = self.cfg
        self._close_window(observation)
        sigma = self._perceive(observation)
        diagnosis: str | None = None

        if self._pending is not None:
            self._close_transition(sigma, observation)
            c = cfg.correction
            if c.enabled and self.trust.n >= c.min_observations and self.trust.mu < cfg.controller.alert:
                diagnosis = self.trust.diagnose(c.field_diagnosis)
                if c.revise_context and diagnosis in ("V", "E", "scalar"):
                    sigma = self._perceive(observation, focus=diagnosis)

        source_prompt = self.perceptor.last_prompt
        if self.perceptor.last_confidence is not None:
            self._confidence = self.perceptor.last_confidence
        source_confidence = self.perceptor.last_confidence

        cs = ControlState(
            confidence=self._confidence,
            contradiction=self._contradiction,
            novelty=self.memory.novelty(sigma),
            progress=self._progress,
            trust=self.trust.mu,
        )
        meta = self.controller.select(cs)
        if cfg.lookahead.horizon <= 0 and meta == MetaAction.LOOKAHEAD:
            meta = MetaAction.ACT

        mem_text = self.memory.retrieve(sigma, trust=self.trust.mu).text
        action: str | None = None
        predicted_next: State | None = None
        n_exp = n_prune = 0
        changed = False
        proposal: tuple[str | None, str | None] = (None, None)
        fallback = cfg.controller.confidence_fallback == "candidates" and source_confidence is None

        if meta == MetaAction.LOOKAHEAD:
            res = self.search.search(self.goal, sigma, mem_text, self.memory.active_preconditions())
            n_exp, n_prune = res.n_expanded, res.n_pruned
            if fallback and res.n_root_candidates:
                self._confidence = 1.0 / res.n_root_candidates
            if res.action is not None:
                action = res.action
                proposal = (res.root_prompt, res.root_output)
                predicted_next = res.first_prediction.state if res.first_prediction else None
                changed = res.default_action is not None and res.default_action != action
        elif meta == MetaAction.ABSTRACT:
            sigma = self._perceive(observation, focus="abstract")
            source_prompt = self.perceptor.last_prompt
        elif meta == MetaAction.RETRIEVE:
            mem_text = self.memory.retrieve(sigma, trust=self.trust.mu, policy=COMBINED).text
        elif meta == MetaAction.ROLLBACK and self._last_consistent is not None:
            sigma = self._last_consistent

        branching = max(cfg.lookahead.branching, 1)
        if action is None:
            cands = self.actor.propose(self.goal, sigma, mem_text, branching) or ["noop"]
            action = cands[0]
            proposal = (self.actor.last_prompt, self.actor.last_output)
            if fallback:
                self._confidence = 1.0 / len(cands)

        if predicted_next is None and cfg.correction.monitor_on_act:
            predicted_next = self.world_model.predict(sigma, action).state

        self._pending = {
            "source": sigma,
            "action": action,
            "predicted": predicted_next,
            "observation": observation,
            "source_prompt": source_prompt,
            "source_confidence": source_confidence,
            "actor_prompt": proposal[0],
            "actor_output": proposal[1],
        }
        self._window_action = f"{self._t}: {action}"
        self.trace.append(
            StepRecord(
                t=self._t,
                meta_action=meta.value,
                action=action,
                trust=self.trust.mu,
                field_trust=self.trust.field_trust(),
                n_expanded=n_exp,
                n_pruned=n_prune,
                lookahead_changed_action=changed,
                diagnosis=diagnosis,
            )
        )
        self._t += 1
        return action


def build_agent(cfg: WEvaConfig, llm: LLM | None = None) -> Any:
    llm = llm or create_llm(cfg.llm)
    if cfg.method == "react":
        return ReActAgent(llm)
    if cfg.method == "weva":
        agent = WEvaAgent(cfg, llm)
        if cfg.run.freeze_memory:
            agent.freeze()
        return agent
    raise ValueError(
        f"Unknown method '{cfg.method}'. External baselines follow the Agent protocol in weva.baselines"
    )
