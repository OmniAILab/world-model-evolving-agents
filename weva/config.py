from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    backend: str = "toy"
    model: str | None = None
    actor_temperature: float = 0.3
    actor_top_p: float = 0.9
    max_tokens: int = 512
    adapters: str | None = None
    send_sampling_params: bool = True
    min_output_tokens: int = 0
    logprobs: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateConfig:
    representation: str = "quasi_symbolic"
    field_weights: dict[str, float] = field(default_factory=lambda: {"V": 1.0, "C": 1.0, "E": 1.0})
    window: int = 8
    window_obs_chars: int = 500


@dataclass
class LookaheadConfig:
    horizon: int = 2
    branching: int = 3
    beam: int = 2
    verify: bool = True
    violation_match: float = 0.5
    verifier: str = "both"


@dataclass
class ControllerConfig:
    type: str = "threshold"
    fixed_action: str = "LOOKAHEAD"
    trust_gating: bool = True
    high: float = 0.7
    mid: float = 0.55
    low: float = 0.4
    initial_trust: float = 0.7
    window: int = 8
    alert: float = 0.55
    weights_path: str | None = None
    confidence_fallback: str = "candidates"


@dataclass
class MemoryConfig:
    levels: tuple[str, ...] = ("episodic", "semantic", "procedural")
    retrieval_policy: str = "adaptive"
    retrieval_k: int = 3
    promote_threshold: int = 2
    demote_threshold: int = -2
    recurrence_min: int = 2
    recurrence_similarity: float = 0.5
    disagree_threshold: float = 0.6
    retrieval_low: float = 0.4
    retrieval_high: float = 0.7


@dataclass
class CorrectionConfig:
    enabled: bool = True
    field_diagnosis: bool = True
    revise_context: bool = True
    monitor_on_act: bool = True
    min_observations: int = 2


@dataclass
class RunConfig:
    max_steps: int = 30
    seed: int = 0
    output_dir: str = "outputs"
    freeze_memory: bool = False


@dataclass
class WEvaConfig:
    method: str = "weva"
    weights: str = "frozen"
    llm: LLMConfig = field(default_factory=LLMConfig)
    state: StateConfig = field(default_factory=StateConfig)
    lookahead: LookaheadConfig = field(default_factory=LookaheadConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _build(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        sub = _SUBCONFIGS.get(cls, {}).get(key)
        if sub is not None and isinstance(value, dict):
            kwargs[key] = _build(sub, value)
        elif key == "levels":
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


_SUBCONFIGS: dict[type, dict[str, type]] = {
    WEvaConfig: {
        "llm": LLMConfig,
        "state": StateConfig,
        "lookahead": LookaheadConfig,
        "controller": ControllerConfig,
        "memory": MemoryConfig,
        "correction": CorrectionConfig,
        "run": RunConfig,
    }
}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_raw(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base_", None)
    if base:
        base_path = (path.parent / base).resolve()
        data = deep_merge(load_raw(base_path), data)
    return data


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def from_dict(data: dict[str, Any]) -> WEvaConfig:
    return _build(WEvaConfig, data)


def load_config(
    path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    overrides: list[str] | None = None,
) -> WEvaConfig:
    paths = [path] if isinstance(path, (str, Path)) else list(path or [])
    data: dict[str, Any] = {}
    for p in paths:
        data = deep_merge(data, load_raw(p))
    cfg = from_dict(apply_overrides(data, overrides))
    if cfg.weights not in ("trainable", "frozen"):
        raise ValueError(f"weights must be 'trainable' or 'frozen', got {cfg.weights!r}")
    return cfg
