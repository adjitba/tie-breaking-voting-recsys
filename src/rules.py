"""Skor ordinal O(i) aturan diskret (Protokol Bagian 5).

Setiap aturan menerima Rn (rating tetangga x kandidat, NaN = kosong) dan
PairwiseStats dari Rn, lalu mengembalikan skor O(i) per kandidat.
"""
from __future__ import annotations

import numpy as np

from pairwise import PairwiseStats


def copeland(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
    """Copeland standar: O(i) = sum_j sign(w_ij - w_ji). Bilangan bulat, delta = 1."""
    return np.sign(ps.W - ps.W.T).sum(axis=1)


def maximin(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
    """Maximin (Simpson), versi margin: O(i) = min_{j != i} (w_ij - w_ji). Bilangan bulat, delta = 1.

    Pasangan tanpa tetangga bersama punya margin 0.
    """
    M = (ps.W - ps.W.T).astype(float)
    n = M.shape[0]
    if n < 2:
        return np.zeros(n)
    np.fill_diagonal(M, np.inf)
    return M.min(axis=1)


def borda(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
    """Borda standar: O(i) = sum_j w_ij. Bilangan bulat, delta = 1."""
    return ps.W.sum(axis=1)


def borda_net(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
    """Borda net (baseline lama yang di paper IS disebut 'Copeland'): sum_j (w_ij - w_ji)."""
    return (ps.W - ps.W.T).sum(axis=1)


def plurality(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
    """Plurality: setiap tetangga memberi 1 suara untuk kandidat favoritnya (rating tertinggi
    di antara kandidat yang ia rating).

    Bila seorang tetangga punya beberapa favorit yang seri, suaranya dibagi rata
    (1/t per item), sejalan dengan konvensi pembagian rata untuk Borda di
    Felfernig dkk. (2018). Keputusan ini masih terbuka di protokol (Bagian 10).
    """
    score = np.zeros(Rn.shape[1])
    for u in range(Rn.shape[0]):
        rated = ~np.isnan(Rn[u])
        if not rated.any():
            continue
        best = np.nanmax(Rn[u])
        fav = rated & (Rn[u] == best)
        score[fav] += 1.0 / fav.sum()
    return np.round(score, 9)


def make_approval(threshold: float):
    """Approval: jumlah tetangga yang memberi rating >= threshold pada i. Bilangan bulat, delta = 1."""

    def approval(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
        return np.nansum(Rn >= threshold, axis=0).astype(float)

    return approval


def least_misery(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
    """Least Misery: rating terendah yang diterima i dari tetangga. delta = granularitas skala."""
    return np.nanmin(Rn, axis=0)


def most_pleasure(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
    """Most Pleasure (Masthoff, 2004): rating tertinggi yang diterima i dari tetangga. delta = granularitas skala."""
    return np.nanmax(Rn, axis=0)


def build_rules_masthoff(relevance_threshold: float) -> dict:
    """Enam strategi Masthoff (2004) yang skornya diskret (bisa seri)."""
    return {
        "plurality": plurality,
        "borda": borda,
        "copeland": copeland,
        "approval": make_approval(relevance_threshold),
        "least_misery": least_misery,
        "most_pleasure": most_pleasure,
    }


# ---------------------------------------------------------------------------
# Kandidat aturan voting baru (results/kandidat/PROTOKOL_KANDIDAT.md)
# ---------------------------------------------------------------------------

def _shrunk_share(votes: np.ndarray, n_i: np.ndarray) -> np.ndarray:
    """(votes + m * p) / (n_i + m), p = sum votes / sum n_i, m = rata-rata n_i."""
    m = n_i.mean()
    p = votes.sum() / n_i.sum() if n_i.sum() > 0 else 0.0
    return np.round((votes + m * p) / (n_i + m), 9)


def shrunk_plurality(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
    """SP(i): proporsi suara Plurality di antara perating i, disusutkan ke proporsi rata-rata."""
    return _shrunk_share(plurality(Rn, ps), (~np.isnan(Rn)).sum(axis=0).astype(float))


def make_shrunk_approval(threshold: float):
    def shrunk_approval(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
        """SA(i): proporsi approval di antara perating i, disusutkan ke proporsi rata-rata."""
        A = np.nansum(Rn >= threshold, axis=0).astype(float)
        return _shrunk_share(A, (~np.isnan(Rn)).sum(axis=0).astype(float))
    return shrunk_approval


def make_quality_approval(threshold: float, r_min: float, r_max: float):
    def quality_approval(Rn: np.ndarray, ps: PairwiseStats) -> np.ndarray:
        """QA(i) = A(i) * b(i), b = (B(i) - r_min) / Delta, B = rata-rata tersusut."""
        M = ~np.isnan(Rn)
        n = M.sum(axis=0).astype(float)
        s = np.nansum(Rn, axis=0)
        m = n.mean()
        mu = s.sum() / n.sum()
        b = ((s + m * mu) / (n + m) - r_min) / (r_max - r_min)
        A = np.nansum(Rn >= threshold, axis=0).astype(float)
        return np.round(A * b, 9)
    return quality_approval


def build_rules_candidates(relevance_threshold: float, r_min: float, r_max: float) -> dict:
    return {
        "shrunk_plurality": shrunk_plurality,
        "shrunk_approval": make_shrunk_approval(relevance_threshold),
        "quality_approval": make_quality_approval(relevance_threshold, r_min, r_max),
    }


def build_rules(relevance_threshold: float) -> dict:
    """Enam aturan utama protokol, dengan ambang Approval sesuai dataset."""
    return {
        "copeland": copeland,
        "maximin": maximin,
        "borda": borda,
        "plurality": plurality,
        "approval": make_approval(relevance_threshold),
        "least_misery": least_misery,
    }


# Kompatibilitas dengan skrip pilot (skala 1-5)
RULES = {**build_rules(4.0), "borda_net": borda_net}
