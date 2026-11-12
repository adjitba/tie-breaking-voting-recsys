"""Sinyal pembanding kontinu untuk RQ3 (Protokol Bagian 5)."""
from __future__ import annotations

import numpy as np

from cardinal import cardinal_score
from pairwise import PairwiseStats

DECIMALS = 9  # pembulatan agar galat floating-point tidak menciptakan "skor berbeda"


def pairwise_mean(Rn, ps: PairwiseStats, **_) -> np.ndarray:
    """Urut menurut P(i) saja (ujung kardinal kerangka)."""
    return cardinal_score(ps)


def mean_rating(Rn, ps, **_) -> np.ndarray:
    """Rata-rata rating item dari tetangga yang merating."""
    return np.round(np.nanmean(Rn, axis=0), DECIMALS)


def support(Rn, ps, **_) -> np.ndarray:
    """Jumlah tetangga yang merating item."""
    return (~np.isnan(Rn)).sum(axis=0).astype(float)


def rating_sum(Rn, ps, **_) -> np.ndarray:
    """Jumlah rating dari tetangga, rating kosong = 0 (skor klasik user-kNN Top-N, tanpa bobot)."""
    return np.round(np.nansum(Rn, axis=0), DECIMALS)


def shrunk_mean(Rn, ps, **_) -> np.ndarray:
    """Rata-rata tersusut (Bayesian average) per kandidat:
        B(i) = (s_i + m * mu_C) / (n_i + m)
    s_i = jumlah rating tetangga untuk i, n_i = jumlah tetangga yang merating i,
    mu_C = rata-rata semua rating tetangga atas kandidat, m = rata-rata n_i atas kandidat.
    Tanpa parameter yang di-tuning; m dan mu_C dihitung per user target."""
    M = ~np.isnan(Rn)
    n = M.sum(axis=0).astype(float)
    s = np.nansum(Rn, axis=0)
    mu = s.sum() / n.sum()
    m = n.mean()
    return np.round((s + m * mu) / (n + m), DECIMALS)


def resnick(Rn, ps, *, neighbor_sims, neighbor_means, target_mean, **_) -> np.ndarray:
    """Prediksi rating Resnick: r_t + sum_u s_u (r_ui - r_u) / sum_u |s_u| atas tetangga yang merating i."""
    M = ~np.isnan(Rn)
    dev = np.where(M, Rn - neighbor_means[:, None], 0.0)
    num = (neighbor_sims[:, None] * dev).sum(axis=0)
    den = (np.abs(neighbor_sims)[:, None] * M).sum(axis=0)
    pred = target_mean + np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.round(pred, DECIMALS)


SIGNALS = {
    "pairwise_mean": pairwise_mean,
    "mean_rating": mean_rating,
    "support": support,
    "rating_sum": rating_sum,
    "resnick": resnick,
    "shrunk_mean": shrunk_mean,
}
