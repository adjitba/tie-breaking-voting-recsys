"""Uji statistik (Protokol Bagian 9)."""
from __future__ import annotations

import numpy as np
from scipy import stats as st
from scipy.stats import rankdata, wilcoxon

TOST_MARGIN_REL = 0.05  # 5% dari rata-rata metrik versi acak (diputuskan 19 Oktober 2026)


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation; selisih nol dibuang (sesuai Wilcoxon)."""
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    t_pos, t_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((t_pos - t_neg) / (t_pos + t_neg))


def bootstrap_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    """CI persentil untuk rata-rata selisih."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def t_ci(diff: np.ndarray, level: float):
    """CI berbasis t untuk rata-rata selisih berpasangan."""
    n = len(diff)
    m = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    if not np.isfinite(se) or se == 0:
        return float(m), float(m)
    h = st.t.ppf(0.5 + level / 2, n - 1) * se
    return float(m - h), float(m + h)


def tost(diff: np.ndarray, margin: float) -> float:
    """p-value TOST (dua uji t satu arah) untuk H1: -margin < rata-rata selisih < +margin."""
    n = len(diff)
    m = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    if not np.isfinite(se) or se == 0:
        return 0.0 if abs(m) < margin else 1.0
    p_lower = 1 - st.t.cdf((m + margin) / se, n - 1)
    p_upper = st.t.cdf((m - margin) / se, n - 1)
    return float(max(p_lower, p_upper))


def equivalence_verdict(ci90: tuple, margin: float) -> str:
    """Empat kategori hasil (Protokol Bagian 9), berdasarkan CI 90% dan margin."""
    lo, hi = ci90
    if -margin < lo and hi < margin:
        return "setara" if lo <= 0 <= hi else "beda kecil, setara secara praktis"
    if hi <= -margin or lo >= margin:
        return "berbeda bermakna"
    return "tidak konklusif"


def paired_test(a: np.ndarray, b: np.ndarray, margin: float | None = None, seed: int = 0) -> dict:
    """Wilcoxon dua sisi untuk a - b, effect size, CI t 95% dan 90%, serta TOST bila margin diberikan."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff = a - b
    n_nonzero = int((np.abs(diff) > 1e-12).sum())
    p = float(wilcoxon(diff).pvalue) if n_nonzero > 0 else 1.0
    ci95 = t_ci(diff, 0.95)
    ci90 = t_ci(diff, 0.90)
    out = {
        "n_users": len(diff),
        "n_nonzero": n_nonzero,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci95_low": ci95[0], "ci95_high": ci95[1],
        "ci90_low": ci90[0], "ci90_high": ci90[1],
        "rank_biserial": rank_biserial(diff),
        "p_value": p,
    }
    if margin is not None:
        out["margin"] = margin
        out["p_tost"] = tost(diff, margin)
        out["verdict"] = equivalence_verdict(ci90, margin)
    return out


def holm(pvalues: list[float]) -> list[float]:
    """Koreksi Holm-Bonferroni; mengembalikan p-value terkoreksi dalam urutan asal."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
