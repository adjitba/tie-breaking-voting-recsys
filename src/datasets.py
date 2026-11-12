"""Konfigurasi dan pemuat dataset (Tabel 3 naskah).

Dataset configurations and loaders (Table 3 of the paper). The ML-1M entry is kept for
completeness but is not used in the paper.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from data import load_ml100k

DATASETS = {
    "ml-100k": {
        "path": "data/ml-100k/u.data",
        "loader": "ml100k",
        "r_min": 1.0, "r_max": 5.0,
        "relevance_threshold": 4.0,
        "min_user_ratings": 20, "min_item_ratings": 5,
        "provisional": [],
    },
    "ml-1m": {
        # Dataset KONFIRMASI (Protokol Bagian 24): tidak dipakai untuk merancang metode
        "path": "data/ml-1m/ratings.dat",
        "loader": "ml1m",
        "r_min": 1.0, "r_max": 5.0,
        "relevance_threshold": 4.0,
        "min_user_ratings": 20, "min_item_ratings": 5,  # sama dengan ML-100K
        "provisional": [],
    },
    "filmtrust": {
        # LibRec: ratings.txt, baris "user item rating" dipisah spasi
        "path": "data/filmtrust/ratings.txt",
        "loader": "whitespace",
        "r_min": 0.5, "r_max": 4.0,
        "relevance_threshold": 3.5,  # dua tingkat teratas dari 8 tingkat (setara >= 4 pada skala 1-5)
        "min_user_ratings": 5, "min_item_ratings": 5,
        "provisional": [],
    },
    "amazon-digital-music": {
        # Amazon Review Data 2018 (Ni, Li & McAuley, 2019), 5-core: Digital_Music_5.json (atau .json.gz),
        # satu objek JSON per baris. Versi 2014 (64.706 ulasan) servernya sudah dipensiunkan.
        "path": "data/amazon-digital-music/Digital_Music_5.json",
        "loader": "amazon_json",
        "r_min": 1.0, "r_max": 5.0,
        "relevance_threshold": 4.0,
        "min_user_ratings": 5, "min_item_ratings": 5,  # mengikuti 5-core
        "provisional": [],
    },
}


def _load_ml1m(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="::", header=None, engine="python", names=["user", "item", "rating", "timestamp"])
    return df.astype({"user": int, "item": int, "rating": float, "timestamp": int})


def _load_whitespace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, :3]
    df.columns = ["user", "item", "rating"]
    df = df.astype({"user": int, "item": int, "rating": float})
    # Pasangan (user, item) ganda (FilmTrust memuat 3): dipertahankan kemunculan terakhir di file,
    # agar tidak masuk latih dan uji sekaligus dan nilai tetap pada skala aslinya (kelipatan 0,5)
    return df.drop_duplicates(["user", "item"], keep="last").reset_index(drop=True)


def _load_amazon_json(path: Path) -> pd.DataFrame:
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append((rec["reviewerID"], rec["asin"], float(rec["overall"]), int(rec.get("unixReviewTime", 0))))
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])
    # Ulasan ganda (user, item): dipertahankan ulasan terbaru, agar rating tetap bilangan bulat 1-5
    df = (df.sort_values(["user", "item", "timestamp"], kind="stable")
            .drop_duplicates(["user", "item"], keep="last").reset_index(drop=True))
    # ID string -> bilangan bulat
    df["user"] = df["user"].astype("category").cat.codes
    df["item"] = df["item"].astype("category").cat.codes
    return df


def load_dataset(name: str, root: Path) -> pd.DataFrame:
    cfg = DATASETS[name]
    path = root / cfg["path"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset {name} tidak ditemukan di {path}")
    if cfg["loader"] == "ml100k":
        return load_ml100k(path)[["user", "item", "rating", "timestamp"]]
    if cfg["loader"] == "ml1m":
        return _load_ml1m(path)
    if cfg["loader"] == "whitespace":
        return _load_whitespace(path)
    if cfg["loader"] == "amazon_json":
        return _load_amazon_json(path)
    raise ValueError(cfg["loader"])
