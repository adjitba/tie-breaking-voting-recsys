# Reproduces all results, figures and tables of the paper. Run from the repository root (PowerShell).
$ErrorActionPreference = "Stop"
python -m pytest -q tests
foreach ($ds in "ml-100k", "filmtrust", "amazon-digital-music") {
  python scripts/run_experiment.py --dataset $ds --k 5 10 15 20 30 --rules masthoff `
      --N 5 10 20 50 --tiebreakers random P shrunk --protocols A --no-signals --tag=-masthoff
  python scripts/run_experiment.py --dataset $ds --k 5 10 15 20 30 --rules candidates `
      --N 5 10 20 50 --tiebreakers random --protocols A --tag=-candidates
}
python scripts/beta_grid_shrunk.py --datasets ml-100k filmtrust
foreach ($s in 0, 1, 2, 3, 4) { python scripts/beta_grid_shrunk.py --datasets amazon-digital-music --seeds $s }
python scripts/analyze_masthoff.py
python scripts/analyze_candidates.py
python scripts/analyze_paper.py
python scripts/make_paper_figs.py
