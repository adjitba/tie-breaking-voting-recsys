"""Metrik evaluasi (Protokol Bagian 8).

Fungsi utama untuk eksperimen penuh adalah `expected_topn`: Precision@N dan NDCG@N
sebagai NILAI HARAPAN PASTI atas semua urutan item yang skornya kembar
(McSherry & Najork, 2008; rumus yang sama dipakai Guo dkk. 2026 dan Ma & Cheng 2026).
Fungsi berbasis urutan tunggal (precision_at, ndcg_at) tetap ada untuk pilot dan tes.
"""
from __future__ import annotations

import numpy as np


def _discounts(n: int) -> np.ndarray:
    """1 / log2(p + 1) untuk posisi p = 1..n."""
    return 1.0 / np.log2(np.arange(2, n + 2))


def sort_desc(*score_columns: np.ndarray) -> np.ndarray:
    """Indeks urutan menurun secara leksikografis (kolom pertama paling menentukan).

    Seri yang tersisa dibiarkan dalam urutan indeks; fungsi pemanggil yang
    memperlakukannya sebagai kelompok seri.
    """
    keys = [np.arange(len(score_columns[0]))]  # kunci terakhir untuk stabilitas
    keys += [-np.asarray(c, float) for c in reversed(score_columns)]
    return np.lexsort(tuple(keys))


def tie_blocks(*score_columns: np.ndarray):
    """Kelompok seri setelah pengurutan: daftar (indeks_urut, awal, ukuran)."""
    order = sort_desc(*score_columns)
    keys = np.column_stack(score_columns)[order]
    n = len(order)
    if n == 0:
        return order, []
    change = np.any(keys[1:] != keys[:-1], axis=1)
    starts = np.concatenate(([0], np.flatnonzero(change) + 1))
    sizes = np.diff(np.concatenate((starts, [n])))
    return order, list(zip(starts.tolist(), sizes.tolist()))


def expected_topn(score_columns, is_relevant: np.ndarray, N: int, n_relevant_ideal: int,
                  values: np.ndarray | None = None) -> dict:
    """Nilai harapan Precision@N dan NDCG@N di bawah pemecah seri acak seragam.

    score_columns : tuple kolom skor (mis. (O,) untuk aturan + acak, (O, P) untuk aturan + P(i)).
    is_relevant   : bool per kandidat.
    n_relevant_ideal : jumlah item relevan untuk IDCG.

    Untuk kelompok seri berukuran g yang memuat r item relevan dan menempati posisi a+1..a+g:
        E[hit di Top-N]   = (r/g) * max(0, min(a+g, N) - a)
        E[DCG kelompok]   = (r/g) * sum_{p=a+1}^{min(a+g, N)} 1/log2(p+1)

    values (opsional) : nilai per kandidat, mis. popularitas item. Dikembalikan sebagai "VAL",
        nilai harapan rata-rata `values` atas item di Top-N:
        E[jumlah kelompok] = (jumlah values kelompok / g) * max(0, min(a+g, N) - a),
        lalu dibagi min(N, jumlah kandidat).
    """
    order, blocks = tie_blocks(*score_columns)
    rel_sorted = np.asarray(is_relevant, bool)[order]
    disc = _discounts(max(N, 1))
    vals_sorted = None if values is None else np.asarray(values, float)[order]
    val_sum = 0.0
    hits = 0.0
    dcg = 0.0
    boundary_size = 0
    changeable = False
    for a, g in blocks:
        if a >= N:
            break
        r = rel_sorted[a:a + g].sum()
        end = min(a + g, N)
        frac = r / g
        hits += frac * (end - a)
        dcg += frac * disc[a:end].sum()
        if vals_sorted is not None:
            val_sum += vals_sorted[a:a + g].mean() * (end - a)
        if a + g > N:  # kelompok terpotong batas Top-N
            boundary_size = g
            changeable = 0 < r < g
    idcg = disc[: min(N, n_relevant_ideal)].sum()
    out_val = {} if vals_sorted is None else {"VAL": val_sum / min(N, len(order))}
    return {
        **out_val,
        "P": hits / N,
        "NDCG": dcg / idcg if idcg > 0 else 0.0,
        "tie_at_boundary": boundary_size > 0,
        "boundary_group_size": boundary_size,
        "changeable": changeable,
    }


def usr(*score_columns: np.ndarray) -> float:
    """Unique Score Ratio: jumlah kombinasi skor unik dibagi jumlah kandidat."""
    keys = np.column_stack(score_columns)
    if len(keys) == 0:
        return float("nan")
    return len(np.unique(keys, axis=0)) / len(keys)


# ---------------------------------------------------------------------------
# Fungsi untuk satu urutan tertentu (dipakai pilot dan tes)
# ---------------------------------------------------------------------------

def precision_at(ranked_items: np.ndarray, relevant: set, N: int) -> float:
    """Proporsi item relevan di Top-N (penyebut selalu N)."""
    top = ranked_items[:N]
    return sum(1 for i in top if i in relevant) / N


def ndcg_at(ranked_items: np.ndarray, relevant: set, N: int) -> float:
    """NDCG@N dengan relevansi biner; IDCG dari min(N, |relevant|)."""
    if not relevant:
        return 0.0
    top = ranked_items[:N]
    dcg = sum(1.0 / np.log2(pos + 2) for pos, i in enumerate(top) if i in relevant)
    idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(N, len(relevant))))
    return dcg / idcg


def tie_at_boundary(*score_columns: np.ndarray, N: int) -> bool:
    """True bila kandidat di posisi N dan N+1 punya skor yang sama persis."""
    keys = np.column_stack(score_columns)
    if len(keys) <= N:
        return False
    order = sort_desc(*score_columns)
    s = keys[order]
    return bool(np.array_equal(s[N - 1], s[N]))
