"""Analisis pendukung naskah studi empiris (bingkai B(i)).

Memakai hasil per user dari run '-masthoff' (enam strategi x acak/P(i)/B(i)/oracle) dan
'-candidates' (rating sum). Protokol A; user dirata-rata atas 5 split.

Keluaran di results/paper/:
- vs_ratingsum.csv : setiap strategi (acak dan B(i)) dibanding rating sum, TOST 5%, Holm per dataset
- headroom.csv     : headroom oracle dan bagian headroom yang direalisasi P(i) dan B(i)
- popularity.csv   : perubahan relatif POP@N oleh P(i) dan B(i) terhadap acak
- k_trend.csv      : jumlah verdict 'berbeda bermakna' (naik) B(i) vs acak per k
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_candidates import per_user  # noqa: E402
from stats import TOST_MARGIN_REL, holm, paired_test  # noqa: E402

DATASETS = {"ml-100k": "ML-100K", "filmtrust": "FilmTrust", "amazon-digital-music": "Amazon"}
RULES = ["plurality", "borda", "copeland", "approval", "least_misery", "most_pleasure"]
NS = [5, 10, 20, 50]
OUT = ROOT / "results" / "paper"


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for ds, lab in DATASETS.items():
        a = per_user(f"{ds}-masthoff")
        b = per_user(f"{ds}-candidates")
        keep = ["k", "user"] + [c for c in b.columns if c.startswith("rating_sum__")]
        g_all = a.merge(b[keep], on=["k", "user"], how="inner")
        assert len(g_all) == len(a) == len(b)
        rr = []
        for k, g in g_all.groupby("k"):
            for N in NS:
                for m in ["P", "NDCG"]:
                    y = g[f"rating_sum__{m}@{N}"].to_numpy()
                    for r in RULES:
                        for v in ["random", "shrunk"]:
                            x = g[f"{r}_{v}__{m}@{N}"].to_numpy()
                            res = paired_test(x, y, TOST_MARGIN_REL * y.mean())
                            rr.append({"dataset": lab, "k": k, "N": N, "metric": m, "rule": r, "variant": v, **res,
                                       "rel_diff": res["mean_diff"] / res["mean_b"]})
        t = pd.DataFrame(rr)
        t["p_holm"] = holm(t["p_value"].tolist())
        rows.append(t)
    vs = pd.concat(rows, ignore_index=True)
    vs.to_csv(OUT / "vs_ratingsum.csv", index=False)

    means = pd.read_csv(ROOT / "results" / "masthoff" / "means.csv")
    tests = pd.read_csv(ROOT / "results" / "masthoff" / "tests.csv")
    p = means.pivot_table(index=["dataset", "k", "rule", "N", "metric"], columns="variant", values="mean").reset_index()
    p["headroom"] = p.oracle / p.random - 1
    for v in ["P", "shrunk"]:
        p[f"share_{v}"] = (p[v] - p.random) / (p.oracle - p.random)
        p[f"gain_{v}"] = p[v] / p.random - 1
    p[p.metric != "POP"].to_csv(OUT / "headroom.csv", index=False)
    p[p.metric == "POP"].to_csv(OUT / "popularity.csv", index=False)

    s = tests[tests.comparison == "shrunk_vs_random"].copy()
    s["gain"] = (s.verdict == "berbeda bermakna") & (s.mean_diff > 0)
    s["loss"] = (s.verdict == "berbeda bermakna") & (s.mean_diff < 0)
    s.groupby(["k", "metric"])[["gain", "loss"]].sum().reset_index().to_csv(OUT / "k_trend.csv", index=False)
    print("selesai")


if __name__ == "__main__":
    main()
