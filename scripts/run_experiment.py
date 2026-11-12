"""Eksperimen penuh (Protokol Bagian 7): satu dataset, beberapa k, beberapa split, protokol A dan B.

Jalankan dari folder proyek:
    python scripts/run_experiment.py --dataset ml-100k
    python scripts/run_experiment.py --dataset ml-100k --k 30 --seeds 0      # uji cepat

Keluaran: results/<dataset>/per_user_seed<s>_k<k>.csv (satu baris per user per protokol)
dan results/<dataset>/run_info.json.

Catatan desain:
- Skor setiap metode dihitung SEKALI atas seluruh kandidat. Protokol A meranking semua
  kandidat; protokol B hanya mengevaluasi urutan kandidat yang ada di data uji user.
- Akurasi dihitung sebagai nilai harapan pasti atas urutan item yang seri (metrics.expected_topn).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardinal import cardinal_score  # noqa: E402
from data import describe, filter_min_counts, split_random_per_user, split_temporal_per_user  # noqa: E402
from datasets import DATASETS, load_dataset  # noqa: E402
from metrics import expected_topn, usr  # noqa: E402
from neighbors import (RatingMatrix, candidate_items, gmcc_matrix, gmcc_matrix_sparse,  # noqa: E402
                       pearson_matrix, pearson_matrix_sparse, top_k_neighbors)

SIMILARITIES = {"pearson": (pearson_matrix, pearson_matrix_sparse), "gmcc": (gmcc_matrix, gmcc_matrix_sparse)}
from pairwise import pairwise_stats  # noqa: E402
from rules import build_rules, build_rules_candidates, build_rules_masthoff  # noqa: E402
from signals import SIGNALS  # noqa: E402

COMMON = {
    "test_frac": 0.2,
    "sim_min_common": 3,
    "candidate_min_support": 2,
    "N_values": [5, 10],
}


def evaluate(cols, is_rel, n_ideal, prefix, row, Ns, diag=False, values=None):
    for N in Ns:
        e = expected_topn(cols, is_rel, N, n_ideal, values)
        row[f"{prefix}__P@{N}"] = e["P"]
        row[f"{prefix}__NDCG@{N}"] = e["NDCG"]
        if values is not None:  # popularitas rata-rata item di Top-N (hanya protokol A)
            row[f"{prefix}__POP@{N}"] = e["VAL"]
        if diag:
            row[f"{prefix}__tie@{N}"] = e["tie_at_boundary"]
            row[f"{prefix}__changeable@{N}"] = e["changeable"]
            row[f"{prefix}__tiesize@{N}"] = e["boundary_group_size"]


OPTS = {"rules": "main", "tiebreakers": None, "protocols": ["A", "B"], "signals": True}


def run_split(df, ds_cfg, seed, ks, out_dir, split="random", similarity="gmcc"):
    if split == "temporal":
        train, test = split_temporal_per_user(df, 1 - COMMON["test_frac"])
    else:
        train, test = split_random_per_user(df, COMMON["test_frac"], seed)
    rm = RatingMatrix.from_frame(train, items=np.sort(df["item"].unique()))
    big = len(rm.users) > 3000  # dataset besar: hitung kemiripan lewat matriks jarang (hasil identik)
    dense_fn, sparse_fn = SIMILARITIES[similarity]
    sim, common = (sparse_fn if big else dense_fn)(rm.R, rm.M, COMMON["sim_min_common"], return_common=True)
    n_defined = int((~np.isnan(sim)).any(axis=1).sum())
    print(f"  seed {seed}: {n_defined} dari {len(rm.users)} user punya minimal satu kemiripan terdefinisi", flush=True)
    means = np.nanmean(rm.R, axis=1)
    thr = ds_cfg["relevance_threshold"]
    if OPTS["rules"] == "masthoff":
        rules = build_rules_masthoff(thr)
    elif OPTS["rules"] == "candidates":
        rules = build_rules_candidates(thr, ds_cfg["r_min"], ds_cfg["r_max"])
    else:
        rules = build_rules(thr)
    item_pop = rm.M.sum(axis=0).astype(float)  # popularitas = jumlah user latih yang merating item
    test_items = test.groupby("user")["item"].apply(set).to_dict()
    test_rel = test[test["rating"] >= thr].groupby("user")["item"].apply(set).to_dict()
    Ns = COMMON["N_values"]

    for k in ks:
        t0 = time.time()
        rows = []
        for u_idx, user in enumerate(rm.users):
            relevant = test_rel.get(user, set())
            neigh = top_k_neighbors(sim[u_idx], k, common[u_idx])
            if len(neigh) == 0:
                continue
            cand = candidate_items(rm.M, neigh, u_idx, COMMON["candidate_min_support"])
            if len(cand) < 2:
                continue
            Rn = rm.R[np.ix_(neigh, cand)]
            ps = pairwise_stats(Rn)
            P = cardinal_score(ps)
            scores = {}
            for name, rule in rules.items():
                scores[name] = rule(Rn, ps)
            sig = {
                name: fn(Rn, ps, neighbor_sims=sim[u_idx, neigh], neighbor_means=means[neigh],
                         target_mean=means[u_idx])
                for name, fn in SIGNALS.items()
            }
            cand_items = rm.items[cand]
            # Lima pemecah seri (Protokol Bagian 7). None = acak (nilai harapan pasti).
            # "id": ID item terkecil didahulukan, meniru urutan bawaan pengurutan stabil.
            tiebreakers = {
                "random": None,
                "id": -cand_items.astype(float),
                "P": P,
                "mean": sig["mean_rating"],
                "support": sig["support"],
                "shrunk": sig["shrunk_mean"],
            }
            if OPTS["tiebreakers"] is not None:
                tiebreakers = {t: v for t, v in tiebreakers.items() if t in OPTS["tiebreakers"]}
            pop_all = item_pop[cand]
            is_rel_all = np.array([i in relevant for i in cand_items])
            in_test = np.array([i in test_items.get(user, set()) for i in cand_items])

            protocols = {
                "A": (np.ones(len(cand), bool), len(relevant)),
                "B": (in_test, int((is_rel_all & in_test).sum())),
            }
            for proto, (mask, n_ideal) in protocols.items():
                if proto not in OPTS["protocols"]:
                    continue
                if n_ideal == 0 or mask.sum() < 2:
                    continue
                is_rel = is_rel_all[mask]
                pop = pop_all[mask] if proto == "A" else None
                row = {"seed": seed, "k": k, "protocol": proto, "user": int(user),
                       "n_neighbors": len(neigh), "n_candidates": int(mask.sum()),
                       "n_relevant": n_ideal}
                for name, O in scores.items():
                    Om, Pm = O[mask], P[mask]
                    for tb, T in tiebreakers.items():
                        cols = (Om,) if T is None else (Om, T[mask])
                        evaluate(cols, is_rel, n_ideal, f"{name}_{tb}", row, Ns, diag=tb in ("random", "P"), values=pop)
                    # oracle: item relevan didahulukan di dalam kelompok seri (batas atas, bukan metode)
                    evaluate((Om, is_rel.astype(float)), is_rel, n_ideal, f"{name}_oracle", row, Ns, values=pop)
                    row[f"{name}__usr_O"] = usr(Om)
                    row[f"{name}__usr_OP"] = usr(Om, Pm)
                if OPTS["signals"]:
                    for name, s in sig.items():
                        evaluate((s[mask],), is_rel, n_ideal, name, row, Ns, values=pop)
                rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"per_user_seed{seed}_k{k}.csv", index=False)
        print(f"  seed {seed}, k {k}: {len(rows)} baris, {time.time() - t0:.1f} dtk", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-100k", choices=sorted(DATASETS))
    ap.add_argument("--k", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", choices=["random", "temporal"], default="random",
                    help="temporal = seperti paper lama (hanya satu pembagian; --seeds diabaikan)")
    ap.add_argument("--min-common", type=int, default=COMMON["sim_min_common"],
                    help="minimal item yang dirating bersama agar kemiripan dihitung (folder berakhiran -min<m> bila bukan 3)")
    ap.add_argument("--similarity", choices=sorted(SIMILARITIES), default="gmcc",
                    help="gmcc = similarity utama; pearson = arsip pembanding (folder berakhiran -pearson)")
    ap.add_argument("--rules", choices=["main", "masthoff", "candidates"], default="main",
                    help="masthoff = enam strategi diskret Masthoff (2004), termasuk Most Pleasure")
    ap.add_argument("--N", type=int, nargs="+", default=COMMON["N_values"], help="panjang daftar Top-N")
    ap.add_argument("--tiebreakers", nargs="+", default=None,
                    help="subset pemecah seri: random id P mean support shrunk (default semua)")
    ap.add_argument("--protocols", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--no-signals", action="store_true", help="tidak mengevaluasi sinyal kontinu sebagai ranking mandiri")
    ap.add_argument("--tag", default="", help="akhiran nama folder hasil, mis. -masthoff")
    args = ap.parse_args()
    COMMON["N_values"] = args.N
    OPTS.update(rules=args.rules, tiebreakers=args.tiebreakers, protocols=args.protocols, signals=not args.no_signals)

    ds_cfg = DATASETS[args.dataset]
    if ds_cfg["provisional"]:
        print(f"PERINGATAN: parameter SEMENTARA untuk {args.dataset}: {ds_cfg['provisional']}")
    out_dir = ROOT / "results" / (args.dataset + ("-temporal" if args.split == "temporal" else "")
                                  + ("-pearson" if args.similarity == "pearson" else "")
                                  + (f"-min{args.min_common}" if args.min_common != 3 else "")
                                  + args.tag)
    COMMON["sim_min_common"] = args.min_common
    if args.split == "temporal":
        args.seeds = [0]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(args.dataset, ROOT)
    if "timestamp" not in raw.columns:
        if args.split == "temporal":
            raise SystemExit(f"{args.dataset} tidak punya timestamp; temporal split tidak bisa dipakai")
        raw = raw.assign(timestamp=0)
    df = filter_min_counts(raw, ds_cfg["min_user_ratings"], ds_cfg["min_item_ratings"])
    info = {"dataset": args.dataset, "split": args.split, "similarity": args.similarity, "config": ds_cfg, "common": COMMON, "k": args.k, "seeds": args.seeds,
            "opts": OPTS,
            "raw": describe(raw), "after_filter": describe(df)}
    print(json.dumps(info["after_filter"]))
    t0 = time.time()
    for seed in args.seeds:
        run_split(df, ds_cfg, seed, args.k, out_dir, args.split, args.similarity)
    info["runtime_seconds"] = round(time.time() - t0, 1)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
