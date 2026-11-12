# Tie-breaking in voting-based recommendation

Code and summary results for the paper

> **Tie-breaking in voting-based recommendation: an order-preserving bound and an empirical study of six aggregation strategies.**
> Teguh Bharata Adji, Sunu Wibirama, Hanung Adi Nugroho. Manuscript submitted to *Array* (Elsevier), 2026.
> Department of Electrical and Information Engineering, Universitas Gadjah Mada, Indonesia. Corresponding author: adji@ugm.ac.id
>
> Repository: https://github.com/adjitba/tie-breaking-voting-recsys

The study measures how often the six discrete aggregation strategies of Masthoff (Plurality, Borda, Copeland, Approval, Least Misery and Most Pleasure) produce ties at the Top-N cut-off in user-based neighborhood recommendation. It compares three tie-breakers (random, pairwise mean difference P(i), shrunk mean B(i)) and tests the order-preserving bound β\* = δ/ρ_T, below which a secondary score only breaks ties.

All accuracy metrics (Precision@N, NDCG@N) are computed as **exact expectations over the orders of tied items** (McSherry & Najork, 2008), so the results do not depend on random seeds for tie-breaking or on item identifiers.

## Repository layout

| Path | Content |
| --- | --- |
| `src/` | Library code: data loading and splitting (`data.py`, `datasets.py`), GMCC similarity and neighborhoods (`neighbors.py`), pairwise statistics (`pairwise.py`, `cardinal.py`), voting rules and candidate rules (`rules.py`), secondary scores incl. the shrunk mean and rating sum (`signals.py`), β\* and lexicographic ranking (`ranking.py`), exact expected metrics (`metrics.py`), statistics incl. TOST and Holm (`stats.py`) |
| `scripts/` | Experiment, analysis and figure scripts (see *Reproducing the paper*) |
| `tests/` | 385 automated tests (`pytest`), e.g. expected metrics vs. brute-force enumeration of tie orders, hand-computed examples |
| `data/` | Place the datasets here (not included, see `data/README.md`) |
| `results/` | Summary results used in the paper (per-user results: see *Per-user results*) |

## Installation

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests             # all tests should pass
```

## Data

The datasets are publicly available but are not redistributed here. See [`data/README.md`](data/README.md) for download links and the expected file locations:

- `data/ml-100k/u.data` (MovieLens 100K)
- `data/filmtrust/ratings.txt` (FilmTrust)
- `data/amazon-digital-music/Digital_Music_5.json` (Amazon Review Data 2018, 5-core Digital Music)

## Reproducing the paper

Run all commands from the repository root. `reproduce.sh` (Linux/macOS) and `reproduce.ps1` (Windows) run the full pipeline.

**1. Main experiment: six strategies × {random, P(i), B(i), oracle}** (Tables 4–5, Figures 1–2)

```bash
for ds in ml-100k filmtrust amazon-digital-music; do
  python scripts/run_experiment.py --dataset $ds --k 5 10 15 20 30 --rules masthoff \
      --N 5 10 20 50 --tiebreakers random P shrunk --protocols A --no-signals --tag=-masthoff
done
```

**2. Rating sum and the three candidate rules** (Table 6, Section 3.5)

```bash
for ds in ml-100k filmtrust amazon-digital-music; do
  python scripts/run_experiment.py --dataset $ds --k 5 10 15 20 30 --rules candidates \
      --N 5 10 20 50 --tiebreakers random --protocols A --tag=-candidates
done
```

The candidate rules and their success criterion were fixed before this run; see `results/kandidat/PROTOKOL_KANDIDAT.md`.

**3. Weights above β\*** (Figure 3)

```bash
python scripts/beta_grid_shrunk.py --datasets ml-100k filmtrust
for s in 0 1 2 3 4; do python scripts/beta_grid_shrunk.py --datasets amazon-digital-music --seeds $s; done
```

Amazon is run one seed at a time to limit memory use (about 8 GB RAM is enough).

**4. Analyses, figures and tables**

```bash
python scripts/analyze_masthoff.py      # results/masthoff/: means, tests (TOST, Wilcoxon-Holm), tie rates
python scripts/analyze_candidates.py    # results/kandidat/: candidate rules vs. baselines
python scripts/analyze_paper.py         # results/paper/: rating-sum comparison, headroom, popularity
python scripts/make_paper_figs.py       # results/paper/figures and results/paper/tables
```

Runtime on a laptop: about 1–2 hours for steps 1–3, mostly for Amazon.

### Mapping to the paper

| Paper | Produced by | File |
| --- | --- | --- |
| Figure 1 (tie rates) | `make_paper_figs.py` | `results/paper/figures/Figure_1.pdf` |
| Figure 2 (gains of P(i), B(i)) | `make_masthoff_figs.py` via `make_paper_figs.py` | `results/paper/figures/Figure_2.pdf` |
| Figure 3 (weights above β\*) | `make_paper_figs.py` | `results/paper/figures/Figure_3.pdf` |
| Table 4 (verdict counts) | `make_masthoff_figs.py` | `results/paper/tables/table_verdicts.tex` |
| Table 5 (P@10, NDCG@10) | `make_masthoff_figs.py` | `results/paper/tables/table_main.tex` |
| Table 6 (vs. rating sum) | `make_paper_figs.py` | `results/paper/tables/table_ratingsum.tex` |
| Section 3.5 (candidate rules) | `analyze_candidates.py` | `results/kandidat/KEPUTUSAN.md`, `tests.csv` |
| Section 3.6 (computation time) | `timing.py` | `results/timing/summary.csv` |

## Summary results

`results/` contains the aggregated results reported in the paper:

- `masthoff/means.csv`: mean P@N, NDCG@N and POP@N per dataset, k, rule, tie-breaker and N
- `masthoff/tests.csv`: paired tests per configuration (mean difference, 90% CI, TOST verdict, Wilcoxon p, Holm-adjusted p, rank-biserial r)
- `masthoff/ties.csv`: tie rate and changeable rate at the cut-off
- `kandidat/`: candidate rules, pre-specified protocol and decision
- `paper/`: comparison with the rating sum, oracle headroom, popularity, effect of k
- `beta_shrunk/`: accuracy of O(i) + c·β\*·b(i) for c ∈ {0.5, 2, 5, 20, 100} and B(i) alone

Verdict labels in the CSV files are in Indonesian: `setara` = equivalent, `beda kecil, setara secara praktis` = negligible difference, `tidak konklusif` = inconclusive, `berbeda bermakna` = relevant difference.

## Per-user results

The per-user results (one row per user, split and neighborhood size) are too large for this repository. They are available from the corresponding author upon reasonable request, or can be regenerated with steps 1–3 of *Reproducing the paper*.

## Computation time

`scripts/timing.py` measures, for every target user (first split, k = 30), the time to select neighbors and candidates from the precomputed similarities, to compute the primary score of each rule, and to break ties with B(i). `results/timing/summary.csv` gives the median, 95th percentile and mean per dataset and rule, and the time to compute all GMCC similarities.

```bash
python scripts/timing.py --datasets ml-100k filmtrust amazon-digital-music
```

## Implementation notes

- Similarity: global mean-centered correlation (GMCC), computed for user pairs with at least three co-rated items.
- Candidates: items not rated by the target user and rated by at least two neighbors.
- Each neighbor casts one unweighted vote; similarity only selects the neighbors.
- Ties after a secondary score are treated as random (exact expectation). Continuous scores are rounded to nine decimals before sorting.
- Splits: random 80/20 per user, seeds 0–4; metrics are averaged per user over the five splits before testing.
- Code comments and console messages are partly in Indonesian.

## Citation

If you use this code, please cite the paper above (citation will be updated after publication) and this repository (see `CITATION.cff`).

## License

Code: MIT License (see `LICENSE`). The datasets are subject to their own licenses.
