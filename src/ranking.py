"""Pengurutan kandidat dan Proposisi beta* (Protokol Bagian 3)."""
from __future__ import annotations

import numpy as np


def beta_star(delta: float, rating_range: float, n: int) -> float:
    """beta* = delta / (2 * Delta * (n - 1)). Untuk n < 2 tidak ada pasangan, kembalikan inf."""
    if n < 2:
        return float("inf")
    return delta / (2.0 * rating_range * (n - 1))


def additive_score(O: np.ndarray, P: np.ndarray, beta: float) -> np.ndarray:
    """S(i) = O(i) + beta * P(i)."""
    return O + beta * P


def rank_order(primary: np.ndarray, secondary: np.ndarray | None, random_key: np.ndarray) -> np.ndarray:
    """Urutan indeks kandidat, dari skor tertinggi.

    Urut menurut primary, lalu secondary (bila ada), lalu random_key untuk seri yang tersisa.
    - aturan + acak : primary = O, secondary = None
    - aturan + P(i) : primary = O, secondary = P   (setara S = O + beta P dengan beta < beta*)
    - Pairwise Mean : primary = P, secondary = None
    """
    keys = [random_key]
    if secondary is not None:
        keys.append(-secondary)
    keys.append(-primary)
    return np.lexsort(tuple(keys))
