"""Pemuatan dataset, filter, dan pembagian data (Protokol Bagian 6)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["user", "item", "rating", "timestamp"]


def load_ml100k(path: str | Path) -> pd.DataFrame:
    """Membaca u.data MovieLens 100K (dipisah tab: user, item, rating, timestamp)."""
    df = pd.read_csv(path, sep="\t", names=COLUMNS, engine="python")
    return df.astype({"user": int, "item": int, "rating": float, "timestamp": int})


def filter_min_counts(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Menghapus user dengan < min_user rating dan item dengan < min_item rating.

    Diulang sampai stabil, karena menghapus item bisa membuat user jatuh di bawah
    ambang (dan sebaliknya).
    """
    out = df
    while True:
        n_before = len(out)
        item_counts = out["item"].value_counts()
        out = out[out["item"].isin(item_counts[item_counts >= min_item].index)]
        user_counts = out["user"].value_counts()
        out = out[out["user"].isin(user_counts[user_counts >= min_user].index)]
        if len(out) == n_before:
            return out.reset_index(drop=True)


def split_random_per_user(
    df: pd.DataFrame, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split acak per user: round(test_frac * n_u) rating (minimal 1) menjadi data uji.

    User diproses berurutan menurut ID agar hasil identik untuk seed yang sama.
    """
    rng = np.random.default_rng(seed)
    test_idx: list[int] = []
    for _, group in df.sort_values(["user", "item"]).groupby("user", sort=True):
        idx = group.index.to_numpy()
        n_test = max(1, int(round(test_frac * len(idx))))
        test_idx.extend(rng.choice(idx, size=n_test, replace=False).tolist())
    is_test = df.index.isin(test_idx)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def split_temporal_per_user(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split per user seperti paper lama: int(train_frac * n_u) rating TERLAMA menjadi
    data latih, sisanya (terbaru) data uji. Seri waktu diurutkan stabil menurut item."""
    train_idx, test_idx = [], []
    for _, g in df.sort_values(["user", "timestamp", "item"], kind="stable").groupby("user", sort=True):
        n_train = int(len(g) * train_frac)
        train_idx.extend(g.index[:n_train].tolist())
        test_idx.extend(g.index[n_train:].tolist())
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def describe(df: pd.DataFrame) -> dict:
    """Statistik dasar dataset."""
    n_users, n_items, n_ratings = df["user"].nunique(), df["item"].nunique(), len(df)
    return {
        "users": int(n_users),
        "items": int(n_items),
        "ratings": int(n_ratings),
        "sparsity": float(1 - n_ratings / (n_users * n_items)),
    }
