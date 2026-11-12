"""Gambar dan tabel temuan utama eksperimen Masthoff (jalankan setelah analyze_masthoff.py).

Keluaran di results/masthoff/:
- Figure_M1.pdf/.png : kenaikan relatif P(i) dan B(i) terhadap Acak, P@N dan NDCG@N vs N (k = 30)
- Table_M1.tex/.csv  : P@10 dan NDCG@10 (k = 30) per dataset dan aturan, dengan verdict TOST
- Table_M2.tex/.csv  : jumlah verdict P(i) vs Acak dan B(i) vs Acak per aturan (60 konfigurasi)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "masthoff"
RULES = ["plurality", "borda", "copeland", "approval", "least_misery", "most_pleasure"]
RL = {"plurality": "Plurality", "borda": "Borda", "copeland": "Copeland", "approval": "Approval",
      "least_misery": "Least Misery", "most_pleasure": "Most Pleasure"}
DS = ["ML-100K", "FilmTrust", "Amazon"]
NS = [5, 10, 20, 50]
C = {"P": "#eb6834", "shrunk": "#2a78d6"}
VL = {"P": "P(i)", "shrunk": "B(i)"}
STY = {"ML-100K": ("-", "o"), "FilmTrust": ("--", "s"), "Amazon": (":", "^")}
GREY = "#8a8a8a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.edgecolor": "#6b6b6b",
                     "xtick.color": "#6b6b6b", "ytick.color": "#6b6b6b", "legend.frameon": False})


def figure_m1(means: pd.DataFrame):
    m = means[means.k == 30].pivot_table(index=["dataset", "rule", "metric", "N"], columns="variant", values="mean")
    fig, axes = plt.subplots(2, 6, figsize=(7.6, 3.4), sharex=True)
    for row, met in enumerate(["P", "NDCG"]):
        for col, r in enumerate(RULES):
            ax = axes[row, col]
            for ds in DS:
                ls, mk = STY[ds]
                for v in ["P", "shrunk"]:
                    y = [100 * (m.loc[(ds, r, met, n), v] / m.loc[(ds, r, met, n), "random"] - 1) for n in NS]
                    ax.plot(NS, y, color=C[v], ls=ls, marker=mk, ms=3, lw=1.1)
            ax.axhspan(-5, 5, color="#ececec", lw=0, zorder=0)  # margin kesetaraan 5%
            ax.axhline(0, color=GREY, lw=0.8)
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, -7), max(hi, 7))
            ax.set_xscale("log")
            ax.set_xticks(NS, [str(n) for n in NS])
            ax.minorticks_off()
            if row == 0:
                ax.set_title(RL[r])
            if col == 0:
                ax.set_ylabel(f"Change in {met}@N\nvs random (%)")
            if row == 1:
                ax.set_xlabel("N")
    handles = [plt.Line2D([], [], color=C[v], lw=1.6, label=VL[v]) for v in ["P", "shrunk"]]
    handles += [plt.Line2D([], [], color=GREY, ls=STY[d][0], marker=STY[d][1], ms=3, label=d) for d in DS]
    handles += [plt.Rectangle((0, 0), 1, 1, color="#ececec", label="±5% margin")]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(OUT / "Figure_M1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure_M1.png", dpi=300, bbox_inches="tight")


def signs(s: str) -> str:
    return re.sub(r"(?<!\^\{)(?<![\w$^])([+-])(\d)", r"$\1$\2", s)


def table_m1(means, tests, ties):
    m = means[(means.k == 30) & (means.N == 10)].pivot_table(index=["dataset", "rule", "metric"],
                                                             columns="variant", values="mean")
    t = tests[(tests.k == 30) & (tests.N == 10)].set_index(["dataset", "rule", "metric", "comparison"])
    tr = ties[(ties.k == 30) & (ties.N == 10)].set_index(["dataset", "rule"]).tie_rate
    rows, L = [], [r"\begin{table}[!htbp]", r"\centering",
                   r"\caption{P@10 and NDCG@10 of the six discrete aggregation strategies of Masthoff with random, "
                   r"$P(i)$ and $B(i)$ tie-breaking ($k=30$). Ties: proportion of users with a tie at the Top-10 cut-off. "
                   r"Bold: relevant gain over random tie-breaking; italic: inconclusive; plain: equivalent or negligible "
                   r"difference (TOST, margin 5\%).}",
                   r"\label{tab:masthoff}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
                   r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                   r" & & \multicolumn{3}{c}{P@10} & \multicolumn{3}{c}{NDCG@10} \\",
                   r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                   r"Strategy & Ties & Random & $P(i)$ & $B(i)$ & Random & $P(i)$ & $B(i)$ \\", r"\midrule"]

    def cell(ds, r, met, v):
        val = m.loc[(ds, r, met), v]
        s = f"{val:.4f}"
        if v != "random":
            x = t.loc[(ds, r, met, f"{v}_vs_random")]
            if x.verdict == "berbeda bermakna" and x.mean_diff > 0:
                s = r"\textbf{" + s + "}"
            elif x.verdict == "tidak konklusif":
                s = r"\emph{" + s + "}"
        return s, val

    for ds in DS:
        L.append(r"\multicolumn{8}{l}{\emph{" + ds + r"}} \\")
        for r in RULES:
            cells, rec = [], {"dataset": ds, "rule": RL[r], "tie_rate": tr[(ds, r)]}
            for met in ["P", "NDCG"]:
                for v in ["random", "P", "shrunk"]:
                    s, val = cell(ds, r, met, v)
                    cells.append(s)
                    rec[f"{met}@10_{v}"] = val
            rows.append(rec)
            L.append(f"{RL[r]} & {100 * tr[(ds, r)]:.0f}\\% & " + " & ".join(cells) + r" \\")
        if ds != DS[-1]:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "Table_M1.tex").write_text("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "Table_M1.csv", index=False)


def table_m2(tests):
    V = ["setara", "beda kecil, setara secara praktis", "tidak konklusif", "berbeda bermakna"]
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{TOST verdicts of $P(i)$ and $B(i)$ against random tie-breaking over 60 configurations per strategy "
         r"(3 datasets $\times$ 5 neighborhood sizes $\times$ 4 list lengths). Each cell gives E/N/I/R: equivalent, "
         r"negligible difference, inconclusive, relevant difference. No relevant difference was a loss.}",
         r"\label{tab:masthoff-verdicts}", r"\small",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{2}{c}{P@N} & \multicolumn{2}{c}{NDCG@N} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Strategy & $P(i)$ & $B(i)$ & $P(i)$ & $B(i)$ \\", r"\midrule"]
    rows = []
    for r in RULES:
        cells, rec = [], {"rule": RL[r]}
        for met in ["P", "NDCG"]:
            for comp in ["P_vs_random", "shrunk_vs_random"]:
                g = tests[(tests.rule == r) & (tests.metric == met) & (tests.comparison == comp)]
                c = "/".join(str(int((g.verdict == v).sum())) for v in V)
                cells.append(c)
                rec[f"{met}_{comp}"] = c
        rows.append(rec)
        L.append(f"{RL[r]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "Table_M2.tex").write_text("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "Table_M2.csv", index=False)


if __name__ == "__main__":
    means = pd.read_csv(OUT / "means.csv")
    tests = pd.read_csv(OUT / "tests.csv")
    ties = pd.read_csv(OUT / "ties.csv")
    figure_m1(means)
    table_m1(means, tests, ties)
    table_m2(tests)
    (OUT / "Table_M1.tex").write_text(signs((OUT / "Table_M1.tex").read_text()))
    print("selesai")
