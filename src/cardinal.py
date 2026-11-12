"""Komponen kardinal p_ij dan P(i) (Protokol Bagian 2 dan 4)."""
from __future__ import annotations

import numpy as np

from pairwise import PairwiseStats

# P(i) adalah jumlah bilangan rasional; dibulatkan agar galat floating-point
# tidak dihitung sebagai "skor berbeda" (kritik reviewer terhadap USR).
P_DECIMALS = 9


def p_matrix(ps: PairwiseStats) -> np.ndarray:
    """p_ij = rata-rata d_u(i, j) atas U_ij; 0 bila U_ij kosong."""
    p = np.zeros_like(ps.Dsum)
    has = ps.Cnt > 0
    p[has] = ps.Dsum[has] / ps.Cnt[has]
    return p


def cardinal_score(ps: PairwiseStats) -> np.ndarray:
    """P(i) = sum_{j != i} p_ij, terhadap seluruh kandidat."""
    return np.round(p_matrix(ps).sum(axis=1), P_DECIMALS)
