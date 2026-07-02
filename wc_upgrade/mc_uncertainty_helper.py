from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np


def jitter_prob(probs: Iterable[float], epsilon: float = 0.008, seed: int | None = None) -> np.ndarray:
    """Apply a light, symmetric probability jitter and renormalize.

    Args:
        probs: 3-class probability vector in [A, D, H] order.
        epsilon: noise scale, kept intentionally small.
        seed: optional deterministic seed for reproducibility.
    """
    p = np.asarray(list(probs), dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, epsilon, size=p.shape)
    noise -= noise.mean()
    q = np.clip(p + noise, 1e-9, None)
    q = q / q.sum()
    return q


def summarize_probs(samples: Iterable[Iterable[float]]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(list(samples), dtype=float)
    if arr.size == 0:
        mean = np.array([np.nan, np.nan, np.nan], dtype=float)
        std = np.array([np.nan, np.nan, np.nan], dtype=float)
        return mean, std
    return arr.mean(axis=0), arr.std(axis=0)
