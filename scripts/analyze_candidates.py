"""Analisis kandidat aturan voting baru (results/kandidat/PROTOKOL_KANDIDAT.md).

Menggabungkan hasil per user dari run '-masthoff' (enam aturan Masthoff, pemecahan seri acak)
dan run '-candidates' (tiga kandidat dan rating sum). Kedua run memakai split dan seed yang sama,
sehingga perbandingan berpasangan per user.

Keluaran di results/kandidat/: means.csv, tests.csv, KEPUTUSAN.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stats import TOST_MARGIN_REL, holm, paired_test  # noqa: E402

DATASETS = {"ml-100k": "ML-100K", "filmtrust": "FilmTrust", "amazon-digital-music": "Amazon"}
CANDS = ["shrunk_plurality", "shrunk_approval", "quality_approval"]
BASE = {"plurality": "plurality_random", "approval": "approval_random", "borda": "borda_random",
        "copeland": "copeland_random", "least_misery": "least_misery_random",
        "most_pleasure": "most_pleasure_random", "rating_sum": "rating_sum"}
KEY_BASE = ["plurality", "approval", "rating_sum"]
NS = [5, 10, 20, 50]
METRICS = ["P", "NDCG", "POP"]


def per_user(tag: str) -> pd.DataFrame:
    files = sorted((ROOT / "results" / tag).glob("per_user_seed*_k*.csv"))
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw = raw[raw.protocol == "A"].drop(columns=["seed", "protocol"])
    raw = raw.apply(pd.to_numeric, errors="coerce").copy()
    return raw.groupby(["k", "user"], as_index=False).mean()


def col(method: str, m: str, N: int) -> str:
    return f"{method}__{m}@{N}" if method == "rating_sum" else f"{method}__{m}@{N}"


def main():
    out = ROOT / "results" / "kandidat"
    out.mkdir(exist_ok=True)
    means, tests = [], []
    for ds, lab in DATASETS.items():
        a = per_user(f"{ds}-masthoff")
        b = per_user(f"{ds}-candidates")
        keep_b = ["k", "user"] + [c for c in b.columns if any(c.startswith(x) for x in CANDS + ["rating_sum__"])]
        g_all = a.merge(b[keep_b], on=["k", "user"], how="inner")
        assert len(g_all) == len(a) == len(b), (lab, len(a), len(b), len(g_all))
        methods = {c: f"{c}_random" for c in CANDS} | BASE
        rows = []
        for k, g in g_all.groupby("k"):
            for N in NS:
                for m in METRICS:
                    for name, pref in methods.items():
                        means.append({"dataset": lab, "k": k, "N": N, "metric": m, "method": name,
                                      "mean": g[f"{pref}__{m}@{N}"].mean()})
                    if m == "POP":
                        continue
                    for c in CANDS:
                        x = g[f"{c}_random__{m}@{N}"].to_numpy()
                        for bname, bpref in BASE.items():
                            y = g[f"{bpref}__{m}@{N}"].to_numpy()
                            res = paired_test(x, y, TOST_MARGIN_REL * y.mean())
                            rows.append({"dataset": lab, "k": k, "N": N, "metric": m, "candidate": c,
                                         "baseline": bname, **res, "rel_diff": res["mean_diff"] / res["mean_b"]})
        t = pd.DataFrame(rows)
        t["p_holm"] = holm(t["p_value"].tolist())
        tests.append(t)
        print(lab, len(g_all), flush=True)
    means, tests = pd.DataFrame(means), pd.concat(tests, ignore_index=True)
    means.to_csv(out / "means.csv", index=False)
    tests.to_csv(out / "tests.csv", index=False)
    (out / "KEPUTUSAN.md").write_text(decision(means, tests), encoding="utf-8")
    print((out / "KEPUTUSAN.md").read_text(encoding="utf-8"))


def decision(means, tests) -> str:
    L = ["# Hasil uji kandidat terhadap kriteria yang ditetapkan sebelumnya", "",
         "Kriteria: pada k = 30, verdict 'berbeda bermakna' dan positif terhadap Plurality, Approval, dan rating sum,",
         "untuk P@10 DAN NDCG@10, di minimal 2 dari 3 dataset.", ""]
    t = tests[(tests.k == 30) & (tests.N == 10) & tests.baseline.isin(KEY_BASE)].copy()
    t["ok"] = (t.verdict == "berbeda bermakna") & (t.mean_diff > 0)
    for c in CANDS:
        tc = t[t.candidate == c]
        per_ds = tc.groupby("dataset").ok.all()
        n_ok = int(per_ds.sum())
        L.append(f"## {c}: memenuhi di {n_ok} dari 3 dataset -> {'BERHASIL' if n_ok >= 2 else 'TIDAK BERHASIL'}")
        L.append("")
        L.append("| Dataset | Metrik | Pembanding | Kandidat | Pembanding | Selisih relatif | Verdict |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, x in tc.iterrows():
            L.append(f"| {x.dataset} | {x.metric}@10 | {x.baseline} | {x.mean_a:.4f} | {x.mean_b:.4f} | "
                     f"{100 * x.rel_diff:+.1f}% | {x.verdict} |")
        L.append("")
    m = means[(means.k == 30) & (means.N == 10)].pivot_table(index=["dataset", "method"], columns="metric",
                                                             values="mean")
    L += ["## Semua metode, k = 30, @10", "", "| Dataset | Metode | P@10 | NDCG@10 | POP@10 |",
          "| --- | --- | --- | --- | --- |"]
    for (ds, meth), x in m.iterrows():
        L.append(f"| {ds} | {meth} | {x['P']:.4f} | {x['NDCG']:.4f} | {x['POP']:.1f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
