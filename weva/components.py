from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np

from .config import ControllerConfig
from .llm import LLM
from .prompts import render
from .state import parse_prediction, parse_state
from .types import ControlState, MetaAction, Prediction, State


def perceptor_prompt(
    goal: str, observation: str, window: list[str], preconditions: list[str], focus: str | None = None
) -> str:
    prompt = render(
        "perceptor",
        goal=goal,
        observation=observation,
        window=" | ".join(window) if window else "none",
        preconditions="; ".join(preconditions) if preconditions else "none",
    )
    if focus:
        prompt += f"FOCUS: re-induce {focus} over the extended window.\n"
    return prompt


def world_model_prompt(state: State, action: str) -> str:
    return render(
        "world_model",
        variables="; ".join(state.V) or "none",
        constraints="; ".join(state.C) or "none",
        outcomes="; ".join(state.E) or "none",
        action=action,
    )


def actor_prompt(goal: str, state: State, memory: str, B: int) -> str:
    return render(
        "actor",
        goal=goal,
        variables="; ".join(state.V) or "none",
        constraints="; ".join(state.C) or "none",
        outcomes="; ".join(state.E) or "none",
        memory=memory or "none",
        B=B,
    )


def verify_prompt(precondition: str, source: State, action: str, target: State | None) -> str:
    return render(
        "verify",
        precondition=precondition,
        source=source.render().replace("\n", "; "),
        action=action,
        target=target.render().replace("\n", "; ") if target is not None else "unknown",
    )


class Perceptor:
    def __init__(self, llm: LLM, mode: str = "quasi_symbolic", max_tokens: int = 512):
        self.llm = llm
        self.mode = mode
        self.max_tokens = max_tokens
        self.last_prompt: str | None = None
        self.last_confidence: float | None = None

    def induce(
        self,
        goal: str,
        observation: str,
        window: list[str],
        preconditions: list[str],
        focus: str | None = None,
    ) -> State:
        prompt = perceptor_prompt(goal, observation, window, preconditions, focus)
        self.last_prompt = prompt
        self.last_confidence = None
        if self.mode == "raw":
            return State(V=(observation,))
        out, mean_logprob = self.llm.generate_scored(
            prompt, role="perceptor", temperature=0.0, max_tokens=self.max_tokens
        )
        if mean_logprob is not None:
            self.last_confidence = math.exp(mean_logprob)
        state = parse_state(out, self.mode)
        if state.is_empty():
            state = State(V=(observation[:200],))
        return state


class WorldModel:
    def __init__(self, llm: LLM, mode: str = "quasi_symbolic", max_tokens: int = 512):
        self.llm = llm
        self.mode = "free_text" if mode == "raw" else mode
        self.max_tokens = max_tokens

    def predict(self, state: State, action: str) -> Prediction:
        out = self.llm.generate(
            world_model_prompt(state, action), role="world_model", temperature=0.0, max_tokens=self.max_tokens
        )
        return parse_prediction(out, source=state, mode=self.mode)


def candidate_lines(output: str) -> list[tuple[str, str]]:
    return [
        (line, re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip())
        for line in output.splitlines(keepends=True)
    ]


def parse_candidates(output: str, B: int) -> list[str]:
    actions: list[str] = []
    for _, a in candidate_lines(output):
        if a and not a.startswith("```") and a not in actions:
            actions.append(a)
    return actions[:B]


class Actor:
    def __init__(self, llm: LLM, temperature: float = 0.3, top_p: float = 0.9, max_tokens: int = 256):
        self.llm = llm
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.last_prompt: str | None = None
        self.last_output: str | None = None

    def propose(self, goal: str, state: State, memory: str = "", B: int = 3) -> list[str]:
        prompt = actor_prompt(goal, state, memory, B)
        out = self.llm.generate(
            prompt, role="actor", temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens
        )
        self.last_prompt, self.last_output = prompt, out
        return parse_candidates(out, B)


def llm_checker(llm: LLM):
    def check(c: str, source: State, action: str, target: State | None) -> bool:
        out = llm.generate(
            verify_prompt(c, source, action, target), role="verifier", temperature=0.0, max_tokens=4
        )
        return out.strip().upper().startswith("YES")

    return check


class Controller(ABC):
    @abstractmethod
    def select(self, cs: ControlState) -> MetaAction: ...

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict) -> None:
        return None


class FixedController(Controller):
    def __init__(self, action: MetaAction = MetaAction.LOOKAHEAD):
        self.action = action

    def select(self, cs: ControlState) -> MetaAction:
        return self.action


class ThresholdController(Controller):
    def __init__(self, high: float = 0.7, mid: float = 0.55, low: float = 0.4):
        self.high, self.mid, self.low = high, mid, low
        self._last: MetaAction | None = None

    def select(self, cs: ControlState) -> MetaAction:
        mu = cs.trust
        if mu >= self.high:
            a = MetaAction.LOOKAHEAD
        elif mu >= self.mid:
            a = MetaAction.LOOKAHEAD if cs.contradiction < 0.5 else MetaAction.ACT
        elif mu >= self.low:
            a = MetaAction.RETRIEVE if cs.novelty > 0.5 else MetaAction.ACT
        else:
            a = MetaAction.ROLLBACK if self._last == MetaAction.ABSTRACT else MetaAction.ABSTRACT
        self._last = a
        return a

    def get_state(self) -> dict:
        return {"last": self._last}

    def set_state(self, state: dict) -> None:
        self._last = state.get("last")


class LLMController(Controller):
    def __init__(self, llm: LLM):
        self.llm = llm

    def select(self, cs: ControlState) -> MetaAction:
        prompt = render(
            "controller",
            conf=f"{cs.confidence:.2f}",
            delta=f"{cs.contradiction:.2f}",
            novelty=f"{cs.novelty:.2f}",
            progress=f"{cs.progress:.2f}",
            trust=f"{cs.trust:.2f}",
        )
        out = self.llm.generate(prompt, role="controller", temperature=0.0, max_tokens=8)
        m = re.search(r"LOOKAHEAD|ACT|ABSTRACT|RETRIEVE|ROLLBACK", out.upper())
        return MetaAction(m.group(0)) if m else MetaAction.ACT


class LearnedController(Controller):
    ACTIONS: ClassVar[list[MetaAction]] = list(MetaAction)

    def __init__(self, dim: int = 5, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((len(self.ACTIONS), dim))
        self.b = np.zeros(len(self.ACTIONS))

    def probs(self, x: np.ndarray) -> np.ndarray:
        z = self.W @ x + self.b
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def select(self, cs: ControlState) -> MetaAction:
        p = self.probs(np.asarray(cs.to_vector(), dtype=float))
        return self.ACTIONS[int(np.argmax(p))]

    def sample(self, cs: ControlState) -> tuple[MetaAction, int]:
        p = self.probs(np.asarray(cs.to_vector(), dtype=float))
        i = int(self.rng.choice(len(self.ACTIONS), p=p))
        return self.ACTIONS[i], i

    def state_dict(self) -> dict:
        return {"W": self.W.tolist(), "b": self.b.tolist()}

    def get_state(self) -> dict:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_state(self, state: dict) -> None:
        self.W, self.b = state["W"].copy(), state["b"].copy()

    def load_state_dict(self, d: dict) -> None:
        self.W, self.b = np.asarray(d["W"], dtype=float), np.asarray(d["b"], dtype=float)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict()))

    def load(self, path: str | Path) -> None:
        self.load_state_dict(json.loads(Path(path).read_text()))


def create_controller(cfg: ControllerConfig, llm: LLM | None = None) -> Controller:
    if not cfg.trust_gating or cfg.type == "fixed":
        return FixedController(MetaAction(cfg.fixed_action))
    if cfg.type == "threshold":
        return ThresholdController(cfg.high, cfg.mid, cfg.low)
    if cfg.type == "llm":
        if llm is None:
            raise ValueError("LLMController needs an llm")
        return LLMController(llm)
    if cfg.type == "learned":
        c = LearnedController()
        if cfg.weights_path:
            c.load(cfg.weights_path)
        return c
    raise ValueError(f"Unknown controller type '{cfg.type}'")
