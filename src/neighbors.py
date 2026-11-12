"""Kemiripan (Pearson dan GMCC), tetangga, dan kandidat (Protokol Bagian 7).

GMCC (Global Mean-Centered Correlation; Wiyono dkk., Information Sciences 749 (2026) 123502)
sama dengan Pearson kecuali titik pusatnya: satu rata-rata global mu dari seluruh rating latih,
bukan rata-rata tiap user. Karena itu GMCC tetap terdefinisi untuk user yang semua ratingnya sama
(Proposisi 3 paper GMCC), selama nilai itu tidak tepat sama dengan mu.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RatingMatrix:
    """Matriks rating data latih. R berisi NaN untuk sel kosong; M adalah masker rating."""

    users: np.ndarray  # ID user asli, urut
    items: np.ndarray  # ID item asli, urut
    R: np.ndarray  # (n_users, n_items)
    M: np.ndarray  # bool, (n_users, n_items)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, items: np.ndarray | None = None) -> "RatingMatrix":
        users = np.sort(df["user"].unique())
        items = np.sort(df["item"].unique()) if items is None else np.asarray(items)
        u_pos = {u: i for i, u in enumerate(users)}
        i_pos = {it: j for j, it in enumerate(items)}
        R = np.full((len(users), len(items)), np.nan)
        rows = df["user"].map(u_pos).to_numpy()
        cols = df["item"].map(i_pos).to_numpy()
        R[rows, cols] = df["rating"].to_numpy()
        return cls(users=users, items=items, R=R, M=~np.isnan(R))


def _centers(R: np.ndarray, M: np.ndarray, center: str) -> np.ndarray:
    """Titik pusat per user: 'user' = rata-rata rating latih user (Pearson), 'global' = mu (GMCC)."""
    if center == "user":
        return np.nansum(R, axis=1) / M.sum(axis=1)
    if center == "global":
        return np.full(R.shape[0], np.nansum(R) / M.sum())
    raise ValueError(center)


def pearson_matrix(R: np.ndarray, M: np.ndarray, min_common: int = 3, center: str = "user",
                   return_common: bool = False):
    """Pearson antar user (Persamaan 1 paper); center='global' memberi GMCC.

    Rata-rata user r̄_u dihitung dari seluruh rating latih user tersebut; penjumlahan
    hanya atas item yang dirating bersama. Pasangan dengan < min_common item bersama
    atau penyebut nol diberi NaN.
    """
    Mf = M.astype(float)
    means = _centers(R, M, center)
    C = np.where(M, R - means[:, None], 0.0)
    num = C @ C.T
    sq = (C**2) @ Mf.T  # sq[u, v] = jumlah c_ui^2 atas item yang dirating u dan v
    den = np.sqrt(sq * sq.T)
    common = Mf @ Mf.T
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = num / den
    sim[(common < min_common) | (den == 0)] = np.nan
    np.fill_diagonal(sim, np.nan)
    return (sim, common) if return_common else sim


def pearson_matrix_sparse(R: np.ndarray, M: np.ndarray, min_common: int = 3, center: str = "user",
                          return_common: bool = False):
    """Pearson yang sama dengan pearson_matrix, dihitung lewat matriks jarang (untuk dataset besar).

    Hasil identik (hingga presisi float32) dengan pearson_matrix; memori O(user^2) float32.
    """
    from scipy import sparse

    Mf = sparse.csr_matrix(M.astype(np.float32))
    means = _centers(R, M, center).astype(np.float32)
    C = sparse.csr_matrix(np.where(M, R - means[:, None], 0.0).astype(np.float32))
    C2 = C.multiply(C).tocsr()
    num = (C @ C.T).toarray()
    sq = (C2 @ Mf.T).toarray()
    common = (Mf @ Mf.T).toarray()
    den = np.sqrt(sq * sq.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = (num / den).astype(np.float32)
    del num, sq
    sim[(common < min_common) | (den == 0)] = np.nan
    np.fill_diagonal(sim, np.nan)
    return (sim, common) if return_common else sim


def gmcc_matrix(R, M, min_common=3, return_common=False):
    return pearson_matrix(R, M, min_common, "global", return_common)


def gmcc_matrix_sparse(R, M, min_common=3, return_common=False):
    return pearson_matrix_sparse(R, M, min_common, "global", return_common)


SIM_DECIMALS = 6  # kemiripan dibulatkan sebelum diurutkan, agar galat float32 tidak memecah seri


def top_k_neighbors(sim_row: np.ndarray, k: int, common_row: np.ndarray | None = None) -> np.ndarray:
    """Indeks k user dengan kemiripan tertinggi (NaN diabaikan).

    Kemiripan yang seri (setelah dibulatkan SIM_DECIMALS desimal) diurutkan menurut jumlah item
    yang dirating bersama (lebih banyak bukti didahulukan), lalu menurut indeks user. Seri ini
    jarang pada Pearson, tetapi sering pada GMCC untuk user yang ratingnya konstan (nilai tepat 1).
    """
    valid = np.flatnonzero(~np.isnan(sim_row))
    key = np.round(sim_row[valid].astype(np.float64), SIM_DECIMALS)
    if common_row is None:
        order = np.lexsort((valid, -key))
    else:
        order = np.lexsort((valid, -common_row[valid], -key))
    return valid[order[:k]]


def candidate_items(M: np.ndarray, neighbors: np.ndarray, target: int, min_support: int = 2) -> np.ndarray:
    """Item yang dirating minimal min_support tetangga dan belum dirating user target."""
    support = M[neighbors].sum(axis=0)
    return np.flatnonzero((support >= min_support) & ~M[target])
