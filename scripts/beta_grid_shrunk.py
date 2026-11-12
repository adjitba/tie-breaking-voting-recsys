"""Uji bobot di atas batas beta*: S(i) = O(i) + c * beta*_u * b(i), untuk enam strategi Masthoff.

b(i) = (B(i) - r_min) / Delta adalah rata-rata tersusut yang dinormalkan ke [0, 1].
beta*_u = delta_u / rho_b, dengan delta_u = selisih positif terkecil skor aturan pada kandidat user u
dan rho_b = rentang b(i) pada kandidat yang sama (Proposisi 1).
- c < 1   : B(i) hanya pemecah seri (harus identik dengan pengurutan leksikografis (O, B))
- c > 1   : B(i) boleh membalik keputusan aturan
- "B saja": kandidat diurutkan menurut B(i) saja

Pengaturan: k = 30, N = 10, protokol A, seed 0-4, sama dengan eksperimen utama.
Jalankan:  python scripts/beta_grid_shrunk.py --datasets ml-100k filmtrust amazon-digital-music
Keluaran:  results/beta_shrunk/per_user_<ds>_seed<s>.csv dan results/beta_shrunk/means.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import filter_min_counts, split_random_per_user  # noqa: E402
from datasets import DATASETS, load_dataset  # noqa: E402
from metrics import expected_topn  # noqa: E402
from neighbors import RatingMatrix, candidate_items, gmcc_matrix, gmcc_matrix_sparse, top_k_neighbors  # noqa: E402
from pairwise import pairwise_stats  # noqa: E402
from rules import build_rules_masthoff  # noqa: E402
from signals import shrunk_mean  # noqa: E402

MULTS = [0.5, 2, 5, 20, 100]
K, N, SEEDS = 30, 10, [0, 1, 2, 3, 4]


def min_gap(O: np.ndarray) -> float:
    u = np.unique(np.round(O, 9))
    return float(np.diff(u).min()) if len(u) > 1 else np.nan


def run(ds: str, out: Path, seeds) -> None:
    cfg = DATASETS[ds]
    df = filter_min_counts(load_dataset(ds, ROOT), cfg["min_user_ratings"], cfg["min_item_ratings"])
    thr, r_min, r_max = cfg["relevance_threshold"], cfg["r_min"], cfg["r_max"]
    rules = build_rules_masthoff(thr)
    for seed in seeds:
        rows = []
        train, test = split_random_per_user(df, 0.2, seed)
        rm = RatingMatrix.from_frame(train, items=np.sort(df["item"].unique()))
        fn = gmcc_matrix_sparse if len(rm.users) > 3000 else gmcc_matrix
        sim, common = fn(rm.R, rm.M, 3, return_common=True)
        item_pop = rm.M.sum(axis=0).astype(float)
        test_rel = test[test["rating"] >= thr].groupby("user")["item"].apply(set).to_dict()
        for u, user in enumerate(rm.users):
            relevant = test_rel.get(user, set())
            if not relevant:
                continue
            neigh = top_k_neighbors(sim[u], K, common[u])
            if len(neigh) == 0:
                continue
            cand = candidate_items(rm.M, neigh, u, 2)
            if len(cand) < 2:
                continue
            Rn = rm.R[np.ix_(neigh, cand)]
            ps = pairwise_stats(Rn)
            B = shrunk_mean(Rn, ps)
            b = np.round((B - r_min) / (r_max - r_min), 9)
            rho = float(b.max() - b.min())
            rel = np.array([i in relevant for i in rm.items[cand]])
            pop = item_pop[cand]
            row = {"dataset": ds, "seed": seed, "user": int(user)}

            def ev(name, cols):
                e = expected_topn(cols, rel, N, len(relevant), pop)
                row[f"{name}|P"], row[f"{name}|NDCG"], row[f"{name}|POP"] = e["P"], e["NDCG"], e["VAL"]

            for rname, rule in rules.items():
                O = np.round(rule(Rn, ps).astype(float), 9)
                ev(f"{rname}|random", (O,))
                ev(f"{rname}|tiebreak", (O, b))
                d = min_gap(O)
                for c in MULTS:
                    if np.isnan(d) or rho == 0:  # semua skor aturan sama atau b konstan: S = urutan b
                        ev(f"{rname}|c={c:g}", (O, b))
                    else:
                        ev(f"{rname}|c={c:g}", (np.round(O + c * (d / rho) * b, 9),))
            ev("B_only", (b,))
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"per_user_{ds}_seed{seed}.csv", index=False)
        print(f"  {ds} seed {seed} selesai ({len(rows)} baris)", flush=True)
        del sim, common, rm, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ml-100k", "filmtrust", "amazon-digital-music"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    a = ap.parse_args()
    out = ROOT / "results" / "beta_shrunk"
    out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        run(ds, out, a.seeds)


if __name__ == "__main__":
    main()
