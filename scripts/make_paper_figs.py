"""Gambar dan tabel naskah studi empiris (bingkai B(i)).

Jalankan setelah analyze_masthoff.py, analyze_candidates.py, analyze_paper.py dan beta_grid_shrunk.py.
Keluaran:
- <fig-dir>/Figure_1.pdf : proporsi user dengan seri di batas Top-N, per strategi (k = 30)
- <fig-dir>/Figure_2.pdf : perubahan relatif P(i) dan B(i) terhadap acak (= Figure_M1)
- <fig-dir>/Figure_3.pdf : bobot di atas beta* (B(i)), P@10 dan NDCG@10 relatif terhadap pemecah seri
- <tab-dir>/table_main.tex, table_verdicts.tex, table_ratingsum.tex
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_masthoff_figs as mm  # noqa: E402

FIG = ROOT / "results" / "paper" / "figures"  # ubah dengan --fig-dir, mis. naskah/figures
TAB = ROOT / "results" / "paper" / "tables"   # ubah dengan --tab-dir, mis. naskah/tables
RES = ROOT / "results"
RULES, RL, DS, NS, STY = mm.RULES, mm.RL, mm.DS, mm.NS, mm.STY
INK = "#3d3d3a"
W = 5.31  # lebar teks elsarticle 1p (384 pt); gambar dicetak 1:1 sehingga huruf 7.5-9 pt tetap terbaca
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 8})
POS = {r: divmod(i, 3) for i, r in enumerate(RULES)}  # aturan -> (baris, kolom) dalam kisi 2 x 3
DSKEY = {"ml-100k": "ML-100K", "filmtrust": "FilmTrust", "amazon-digital-music": "Amazon"}
VSHORT = {"setara": "E", "beda kecil, setara secara praktis": "N", "tidak konklusif": "I", "berbeda bermakna": "R"}


def legend_ds(fig, extra=(), y=1.0, ncol=3):
    h = [plt.Line2D([], [], color=INK, ls=STY[d][0], marker=STY[d][1], ms=3.5, label=d) for d in DS] + list(extra)
    fig.legend(handles=h, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), frameon=False)


def figure_1(ties):
    t = ties[ties.k == 30]
    fig, axes = plt.subplots(2, 3, figsize=(W, 3.7), sharey=True, sharex=True)
    for r in RULES:
        ax = axes[POS[r]]
        for ds in DS:
            ls, mk = STY[ds]
            y = [100 * t[(t.dataset == ds) & (t.rule == r) & (t.N == n)].tie_rate.iloc[0] for n in NS]
            ax.plot(NS, y, color=INK, ls=ls, marker=mk, ms=3.5, lw=1.2)
        ax.set_title(RL[r])
        ax.set_xscale("log")
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.minorticks_off()
        ax.set_ylim(0, 105)
        if POS[r][1] == 0:
            ax.set_ylabel("Users with a tie (%)")
        if POS[r][0] == 1:
            ax.set_xlabel("List length N")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    legend_ds(fig, y=0.93)
    fig.savefig(FIG / "Figure_1.pdf", bbox_inches="tight")
    fig.savefig(RES / "paper" / "Figure_1.png", dpi=200, bbox_inches="tight")


def figure_2(means):
    m = means[means.k == 30].pivot_table(index=["dataset", "rule", "metric", "N"], columns="variant", values="mean")
    fig, axes = plt.subplots(4, 3, figsize=(W, 6.9), sharex=True)
    for b, met in enumerate(["P", "NDCG"]):
        for r in RULES:
            row, col = POS[r]
            ax = axes[2 * b + row, col]
            for ds in DS:
                ls, mk = STY[ds]
                for v in ["P", "shrunk"]:
                    y = [100 * (m.loc[(ds, r, met, n), v] / m.loc[(ds, r, met, n), "random"] - 1) for n in NS]
                    ax.plot(NS, y, color=mm.C[v], ls=ls, marker=mk, ms=3, lw=1.1)
            ax.axhspan(-5, 5, color="#ececec", lw=0, zorder=0)
            ax.axhline(0, color="#8a8a8a", lw=0.8)
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, -7), max(hi, 7))
            ax.set_xscale("log")
            ax.set_xticks(NS, [str(n) for n in NS])
            ax.minorticks_off()
            ax.set_title(f"{RL[r]}, {met}@N")
            if col == 0:
                ax.set_ylabel("Change (%)")
            if 2 * b + row == 3:
                ax.set_xlabel("List length N")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    extra = [plt.Line2D([], [], color=mm.C[v], lw=1.8, label=mm.VL[v]) for v in ["P", "shrunk"]]
    extra += [plt.Rectangle((0, 0), 1, 1, color="#ececec", label="\u00b15% margin")]
    legend_ds(fig, extra, y=0.94, ncol=3)
    fig.savefig(FIG / "Figure_2.pdf", bbox_inches="tight")
    fig.savefig(RES / "paper" / "Figure_2.png", dpi=200, bbox_inches="tight")


def beta_means():
    d = pd.concat([pd.read_csv(f) for f in sorted((RES / "beta_shrunk").glob("per_user_*_seed*.csv"))])
    d = d.drop(columns=["seed"]).groupby(["dataset", "user"], as_index=False).mean()  # user = unit analisis
    long = d.melt(id_vars=["dataset", "user"])
    long[["rule", "variant", "metric"]] = long.variable.str.rsplit("|", n=2, expand=True)
    bonly = long[long.variable.str.startswith("B_only")].copy()
    bonly[["variant", "metric"]] = bonly.variable.str.split("|", expand=True)
    rest = long[~long.variable.str.startswith("B_only")]
    m = rest.groupby(["dataset", "rule", "variant", "metric"]).value.mean().reset_index()
    b = bonly.groupby(["dataset", "metric"]).value.mean().reset_index()
    m["dataset"] = m.dataset.map(DSKEY)
    b["dataset"] = b.dataset.map(DSKEY)
    m.to_csv(RES / "beta_shrunk" / "means.csv", index=False)
    b.to_csv(RES / "beta_shrunk" / "means_B_only.csv", index=False)
    return m, b


XV = ["c=0.5", "c=2", "c=5", "c=20", "c=100", "B_only"]
XL = ["0.5", "2", "5", "20", "100", r"$B$"]


def figure_3(m, b):
    fig, axes = plt.subplots(4, 3, figsize=(W, 7.2), sharex=True)
    for blk, met in enumerate(["P", "NDCG"]):
        for r in RULES:
            row, col = POS[r]
            ax = axes[2 * blk + row, col]
            for ds in DS:
                ls, mk = STY[ds]
                base = m[(m.dataset == ds) & (m.rule == r) & (m.variant == "tiebreak") & (m.metric == met)].value.iloc[0]
                y = []
                for v in XV:
                    if v == "B_only":
                        val = b[(b.dataset == ds) & (b.metric == met)].value.iloc[0]
                    else:
                        val = m[(m.dataset == ds) & (m.rule == r) & (m.variant == v) & (m.metric == met)].value.iloc[0]
                    y.append(100 * (val / base - 1))
                ax.plot(range(len(XV)), y, color=INK, ls=ls, marker=mk, ms=3, lw=1.1)
            ax.axhspan(-5, 5, color="#ececec", lw=0, zorder=0)
            ax.axhline(0, color="#8a8a8a", lw=0.8)
            ax.axvline(0.5, color="#8a8a8a", lw=0.7, ls=(0, (2, 2)))
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, -7), max(hi, 7))
            ax.set_xticks(range(len(XV)), XL)
            ax.set_title(f"{RL[r]}, {met}@10")
            if col == 0:
                ax.set_ylabel("Change (%)")
            if 2 * blk + row == 3:
                ax.set_xlabel(r"$\beta/\beta^*$")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    legend_ds(fig, [plt.Rectangle((0, 0), 1, 1, color="#ececec", label="\u00b15% margin")], y=0.95, ncol=4)
    fig.savefig(FIG / "Figure_3.pdf", bbox_inches="tight")
    fig.savefig(RES / "paper" / "Figure_3.png", dpi=200, bbox_inches="tight")


def table_ratingsum():
    v = pd.read_csv(RES / "paper" / "vs_ratingsum.csv")
    v = v[(v.k == 30) & (v.N == 10) & (v.variant == "shrunk")].set_index(["dataset", "rule", "metric"])
    km = pd.read_csv(RES / "kandidat" / "means.csv")
    rs = km[(km.k == 30) & (km.N == 10) & (km.method == "rating_sum")].set_index(["dataset", "metric"])["mean"]
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Relative difference (\%) between each strategy with $B(i)$ tie-breaking and the rating sum "
         r"($k=30$, $N=10$). Bold: relevant difference; italic: inconclusive; plain: equivalent or negligible "
         r"difference (TOST, margin 5\%). The first row gives the rating sum.}",
         r"\label{tab:ratingsum}", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{ML-100K} & \multicolumn{2}{c}{FilmTrust} & \multicolumn{2}{c}{Amazon} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Strategy & P@10 & NDCG@10 & P@10 & NDCG@10 & P@10 & NDCG@10 \\", r"\midrule",
         "Rating sum & " + " & ".join(f"{rs[(ds, met)]:.4f}" for ds in DS for met in ["P", "NDCG"]) + r" \\",
         r"\midrule"]
    for r in RULES:
        cells = []
        for ds in DS:
            for met in ["P", "NDCG"]:
                x = v.loc[(ds, r, met)]
                s = f"{100 * x.rel_diff:+.1f}".replace("+", "$+$").replace("-", "$-$")
                if x.verdict == "berbeda bermakna":
                    s = r"\textbf{" + s + "}"
                elif x.verdict == "tidak konklusif":
                    s = r"\emph{" + s + "}"
                cells.append(s)
        L.append(f"{RL[r]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table_ratingsum.tex").write_text("\n".join(L) + "\n")


def main():
    global FIG, TAB
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", type=Path, default=FIG)
    ap.add_argument("--tab-dir", type=Path, default=TAB)
    a = ap.parse_args()
    FIG, TAB = a.fig_dir.resolve(), a.tab_dir.resolve()
    (RES / "paper").mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    means = pd.read_csv(RES / "masthoff" / "means.csv")
    tests = pd.read_csv(RES / "masthoff" / "tests.csv")
    ties = pd.read_csv(RES / "masthoff" / "ties.csv")
    figure_1(ties)
    figure_2(means)
    mm.table_m1(means, tests, ties)
    mm.table_m2(tests)
    (TAB / "table_main.tex").write_text(mm.signs((RES / "masthoff" / "Table_M1.tex").read_text()))
    (TAB / "table_verdicts.tex").write_text((RES / "masthoff" / "Table_M2.tex").read_text())
    table_ratingsum()
    m, b = beta_means()
    figure_3(m, b)
    print("selesai")


if __name__ == "__main__":
    main()
