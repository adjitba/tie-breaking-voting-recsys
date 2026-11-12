"""Waktu komputasi: pra-hitung kemiripan GMCC (sekali) dan pemberian skor per user target.

Mengukur, untuk setiap user target pada split seed 0 dan k = 30:
- online : pemilihan tetangga dan kandidat (dari matriks kemiripan yang sudah dihitung)
- rule   : skor aturan O(i) (termasuk statistik berpasangan untuk Borda dan Copeland)
- B      : rata-rata tersusut B(i) dan pengurutan leksikografis (O, B)
- P      : selisih rata-rata berpasangan P(i) (termasuk statistik berpasangan) dan pengurutan (O, P)
Setiap pengukuran diulang 3 kali; yang dipakai adalah waktu tercepat (mengurangi derau sistem).

Jalankan:  python scripts/timing.py --datasets ml-100k filmtrust amazon-digital-music
Keluaran:  results/timing/per_user.csv, results/timing/summary.csv
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardinal import cardinal_score  # noqa: E402
from data import filter_min_counts, split_random_per_user  # noqa: E402
from datasets import DATASETS, load_dataset  # noqa: E402
from neighbors import RatingMatrix, candidate_items, gmcc_matrix, gmcc_matrix_sparse, top_k_neighbors  # noqa: E402
from pairwise import pairwise_stats  # noqa: E402
from rules import build_rules_masthoff  # noqa: E402
from signals import shrunk_mean  # noqa: E402

K, SEED, REP = 30, 0, 3
PAIRWISE = {"borda", "copeland"}


def best(fn):
    t = []
    for _ in range(REP):
        t0 = time.perf_counter()
        out = fn()
        t.append(time.perf_counter() - t0)
    return min(t), out


def run(ds: str) -> pd.DataFrame:
    cfg = DATASETS[ds]
    df = filter_min_counts(load_dataset(ds, ROOT), cfg["min_user_ratings"], cfg["min_item_ratings"])
    rules = build_rules_masthoff(cfg["relevance_threshold"])
    train, _ = split_random_per_user(df, 0.2, SEED)
    rm = RatingMatrix.from_frame(train, items=np.sort(df["item"].unique()))
    fn = gmcc_matrix_sparse if len(rm.users) > 3000 else gmcc_matrix
    t0 = time.perf_counter()
    sim, common = fn(rm.R, rm.M, 3, return_common=True)
    t_sim = time.perf_counter() - t0
    rows = []
    for u in range(len(rm.users)):
        def online():
            nb = top_k_neighbors(sim[u], K, common[u])
            return nb, (candidate_items(rm.M, nb, u, 2) if len(nb) else np.array([], int))
        t_on, (nb, cand) = best(online)
        if len(nb) == 0 or len(cand) < 2:
            continue
        Rn = rm.R[np.ix_(nb, cand)]
        row = {"dataset": ds, "user": u, "n_cand": len(cand), "n_neigh": len(nb), "online_ms": 1e3 * t_on}
        for name, rule in rules.items():
            def score():
                ps = pairwise_stats(Rn) if name in PAIRWISE else None
                return rule(Rn, ps), ps
            t_r, (O, ps) = best(score)
            t_b, _ = best(lambda: np.lexsort((-shrunk_mean(Rn, None), -np.asarray(O, float))))
            t_p, _ = best(lambda: np.lexsort((-cardinal_score(ps if ps is not None else pairwise_stats(Rn)),
                                              -np.asarray(O, float))))
            row[f"{name}|rule_ms"] = 1e3 * t_r
            row[f"{name}|B_ms"] = 1e3 * t_b
            row[f"{name}|P_ms"] = 1e3 * t_p
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["t_sim"] = t_sim
    print(f"{ds}: GMCC {t_sim:.1f} s, {len(out)} user", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ml-100k", "filmtrust", "amazon-digital-music"])
    a = ap.parse_args()
    out = ROOT / "results" / "timing"
    out.mkdir(parents=True, exist_ok=True)
    frames, summ = [], []
    for ds in a.datasets:
        d = run(ds)
        frames.append(d)
        for c in [c for c in d.columns if c.endswith("_ms")]:
            summ.append({"dataset": ds, "measure": c, "median_ms": d[c].median(), "p95_ms": d[c].quantile(0.95),
                         "mean_ms": d[c].mean(), "gmcc_s": d.attrs["t_sim"], "users": len(d),
                         "median_cand": d.n_cand.median()})
    pd.concat(frames).to_csv(out / "per_user.csv", index=False)
    s = pd.DataFrame(summ)
    s["machine"] = f"{platform.processor() or platform.machine()}, Python {platform.python_version()}"
    s.to_csv(out / "summary.csv", index=False)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
