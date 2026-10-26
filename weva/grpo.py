from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def group_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    r = np.asarray(rewards, dtype=float)
    if r.size == 0:
        return []
    std = r.std()
    if std < eps:
        return [0.0] * r.size
    return ((r - r.mean()) / (std + eps)).tolist()


def kl_k3(logp: np.ndarray, logp_ref: np.ndarray) -> np.ndarray:
    d = logp_ref - logp
    return np.exp(d) - d - 1.0


def grpo_objective(
    logp: Sequence[Sequence[float]],
    logp_old: Sequence[Sequence[float]],
    logp_ref: Sequence[Sequence[float]],
    advantages: Sequence[float],
    clip_eps: float = 0.2,
    kl_coef: float = 0.001,
) -> dict[str, float]:
    per_sample, kls, clipped = [], [], []
    for lp, lo, lr, a in zip(logp, logp_old, logp_ref, advantages):
        lp, lo, lr = (np.asarray(x, dtype=float) for x in (lp, lo, lr))
        ratio = np.exp(lp - lo)
        surrogate = np.minimum(ratio * a, np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a)
        kl = kl_k3(lp, lr)
        per_sample.append(float(np.mean(surrogate - kl_coef * kl)))
        kls.append(float(np.mean(kl)))
        clipped.append(float(np.mean(np.abs(ratio - 1.0) > clip_eps)))
    n = max(len(per_sample), 1)
    return {
        "loss": -sum(per_sample) / n,
        "kl": sum(kls) / n,
        "clip_fraction": sum(clipped) / n,
    }


def successor_nll(token_logprobs: Sequence[Sequence[float]]) -> float:
    if not token_logprobs:
        return 0.0
    return float(np.mean([-float(np.sum(t)) for t in token_logprobs]))
