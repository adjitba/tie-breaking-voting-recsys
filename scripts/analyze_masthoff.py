"""Analisis eksperimen enam strategi Masthoff (2004) x tiga varian pemecah seri.

Jalankan setelah:
    python scripts/run_experiment.py --dataset <ds> --k 5 10 15 20 30 --rules masthoff \
        --N 5 10 20 50 --tiebreakers random P shrunk --protocols A --no-signals --tag=-masthoff

Varian: acak (nilai harapan pasti), P(i), B(i) (rata-rata tersusut); oracle sebagai batas atas.
Unit analisis = user (dirata-rata atas 5 split). Protokol A saja.

Keluaran di results/masthoff/:
- means.csv     : rata-rata P@N, NDCG@N, POP@N per dataset, k, aturan, varian, N
- tests.csv     : P(i) vs acak, B(i) vs acak, B(i) vs P(i): Wilcoxon (Holm per dataset), CI 90%, TOST 5%
- ties.csv      : proporsi user dengan seri di batas Top-N dan proporsi user yang bisa berubah
- RINGKASAN.md  : tabel ringkas
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stats import TOST_MARGIN_REL, holm, paired_test  # noqa: E402

DATASETS = {"ml-100k": "ML-100K", "filmtrust": "FilmTrust", "amazon-digital-music": "Amazon"}
RULES = ["plurality", "borda", "copeland", "approval", "least_misery", "most_pleasure"]
RULE_LABEL = {"plurality": "Plurality", "borda": "Borda", "copeland": "Copeland", "approval": "Approval",
              "least_misery": "Least Misery", "most_pleasure": "Most Pleasure"}
VARIANTS = ["random", "P", "shrunk", "oracle"]
VAR_LABEL = {"random": "Acak", "P": "P(i)", "shrunk": "B(i)", "oracle": "Oracle"}
NS = [5, 10, 20, 50]
METRICS = ["P", "NDCG", "POP"]
COMPARISONS = [("P", "random"), ("shrunk", "random"), ("shrunk", "P")]
SHORT = {"setara": "E", "beda kecil, setara secara praktis": "N", "tidak konklusif": "I", "berbeda bermakna": "R"}


def per_user(ds: str) -> pd.DataFrame:
    d = ROOT / "results" / f"{ds}-masthoff"
    files = sorted(d.glob("per_user_seed*_k*.csv"))
    if not files:
        raise FileNotFoundError(d)
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw = raw[raw.protocol == "A"]
    num = raw.drop(columns=["seed", "protocol"]).apply(pd.to_numeric, errors="coerce")
    return num.groupby(["k", "user"], as_index=False).mean()


def main():
    out = ROOT / "results" / "masthoff"
    out.mkdir(exist_ok=True)
    means, tests, ties = [], [], []
    for ds, lab in DATASETS.items():
        pu = per_user(ds)
        rows_t = []
        for k, g in pu.groupby("k"):
            for r in RULES:
                for N in NS:
                    ties.append({"dataset": lab, "k": k, "rule": r, "N": N, "n_users": len(g),
                                 "tie_rate": g[f"{r}_random__tie@{N}"].mean(),
                                 "changeable_rate": g[f"{r}_random__changeable@{N}"].mean()})
                    for m in METRICS:
                        for v in VARIANTS:
                            means.append({"dataset": lab, "k": k, "rule": r, "N": N, "metric": m, "variant": v,
                                          "mean": g[f"{r}_{v}__{m}@{N}"].mean()})
                        if m == "POP":
                            continue
                        for a_v, b_v in COMPARISONS:
                            a = g[f"{r}_{a_v}__{m}@{N}"].to_numpy()
                            b = g[f"{r}_{b_v}__{m}@{N}"].to_numpy()
                            res = paired_test(a, b, TOST_MARGIN_REL * b.mean())
                            rows_t.append({"dataset": lab, "k": k, "rule": r, "N": N, "metric": m,
                                           "comparison": f"{a_v}_vs_{b_v}", **res,
                                           "rel_diff": res["mean_diff"] / res["mean_b"] if res["mean_b"] else float("nan")})
        t = pd.DataFrame(rows_t)
        t["p_holm"] = holm(t["p_value"].tolist())  # Holm atas semua perbandingan dalam satu dataset
        tests.append(t)
        print(f"{lab}: {len(pu)} baris user x k", flush=True)
    means, tests, ties = pd.DataFrame(means), pd.concat(tests, ignore_index=True), pd.DataFrame(ties)
    means.to_csv(out / "means.csv", index=False)
    tests.to_csv(out / "tests.csv", index=False)
    ties.to_csv(out / "ties.csv", index=False)
    (out / "RINGKASAN.md").write_text(summary(means, tests, ties), encoding="utf-8")
    print((out / "RINGKASAN.md").read_text(encoding="utf-8"))


def summary(means, tests, ties) -> str:
    L = ["# Ringkasan: enam strategi Masthoff x tiga varian pemecah seri (protokol A)", "",
         "Varian: Acak = nilai harapan pemecah seri acak; P(i); B(i) = rata-rata tersusut. Oracle = batas atas.",
         "Verdict TOST margin 5%: E = setara, N = beda kecil, I = tidak konklusif, R = berbeda bermakna.", ""]
    L += ["## Proporsi user dengan seri di batas Top-N (k = 30)", "",
          "| Dataset | Aturan | @5 | @10 | @20 | @50 |", "| --- | --- | --- | --- | --- | --- |"]
    t30 = ties[ties.k == 30]
    for (ds, r), g in t30.groupby(["dataset", "rule"], sort=False):
        x = g.set_index("N").tie_rate
        L.append(f"| {ds} | {RULE_LABEL[r]} | " + " | ".join(f"{100 * x[n]:.0f}%" for n in NS) + " |")
    for m in ["P", "NDCG"]:
        L += ["", f"## {m}@N (k = 30): Acak / P(i) / B(i) / Oracle", "",
              "| Dataset | Aturan | @5 | @10 | @20 | @50 |", "| --- | --- | --- | --- | --- | --- |"]
        mm = means[(means.k == 30) & (means.metric == m)]
        for (ds, r), g in mm.groupby(["dataset", "rule"], sort=False):
            cells = []
            for n in NS:
                x = g[g.N == n].set_index("variant")["mean"]
                cells.append(" / ".join(f"{x[v]:.4f}" for v in VARIANTS))
            L.append(f"| {ds} | {RULE_LABEL[r]} | " + " | ".join(cells) + " |")
    L += ["", "## Jumlah verdict atas 3 dataset x 5 k x 4 N = 60 konfigurasi per aturan", "",
          "| Metrik | Perbandingan | Aturan | E | N | I | R (naik) | R (turun) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in ["P", "NDCG"]:
        for comp in ["P_vs_random", "shrunk_vs_random", "shrunk_vs_P"]:
            for r in RULES:
                g = tests[(tests.metric == m) & (tests.comparison == comp) & (tests.rule == r)]
                v = g.verdict.map(SHORT)
                up = ((v == "R") & (g.mean_diff > 0)).sum()
                down = ((v == "R") & (g.mean_diff < 0)).sum()
                L.append(f"| {m} | {comp} | {RULE_LABEL[r]} | {(v == 'E').sum()} | {(v == 'N').sum()} | "
                         f"{(v == 'I').sum()} | {up} | {down} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
