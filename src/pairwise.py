"""Perbandingan berpasangan antar kandidat dari rating tetangga (Protokol Bagian 2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairwiseStats:
    """Statistik per pasangan (i, j) untuk n kandidat.

    W[i, j]   = w_ij, jumlah tetangga dengan d_u(i, j) > 0
    Dsum[i, j] = jumlah d_u(i, j) atas U_ij
    Cnt[i, j] = |U_ij|, jumlah tetangga yang merating i dan j
    """

    W: np.ndarray
    Dsum: np.ndarray
    Cnt: np.ndarray

    @property
    def n(self) -> int:
        return self.W.shape[0]


def pairwise_stats(Rn: np.ndarray) -> PairwiseStats:
    """Menghitung w_ij, jumlah d_u, dan |U_ij| dari rating tetangga.

    Rn: matriks (m tetangga x n kandidat) dengan NaN untuk rating kosong.
    Dihitung per user sesuai Persamaan 3-5 (bukan dari tanda rata-rata margin).
    """
    m, n = Rn.shape
    W = np.zeros((n, n))
    Dsum = np.zeros((n, n))
    Cnt = np.zeros((n, n))
    for u in range(m):
        idx = np.flatnonzero(~np.isnan(Rn[u]))
        if idx.size < 2:
            continue
        r = Rn[u, idx]
        d = r[:, None] - r[None, :]
        block = np.ix_(idx, idx)
        W[block] += d > 0
        Dsum[block] += d
        Cnt[block] += 1
    np.fill_diagonal(Cnt, 0)
    np.fill_diagonal(Dsum, 0)
    return PairwiseStats(W=W, Dsum=Dsum, Cnt=Cnt)
