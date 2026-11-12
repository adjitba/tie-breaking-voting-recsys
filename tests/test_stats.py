import numpy as np
import pytest

from stats import equivalence_verdict, holm, paired_test, rank_biserial, tost


def test_rank_biserial():
    assert rank_biserial(np.array([1.0, 2.0, 0.0])) == 1.0
    assert rank_biserial(np.array([1.0, -1.0])) == 0.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_verdict_empat_kategori():
    assert equivalence_verdict((-0.0002, 0.0007), 0.0031) == "setara"
    assert equivalence_verdict((0.0005, 0.0010), 0.0031) == "beda kecil, setara secara praktis"
    assert equivalence_verdict((0.0040, 0.0060), 0.0031) == "berbeda bermakna"
    assert equivalence_verdict((0.0018, 0.0063), 0.0020) == "tidak konklusif"


def test_tost_dan_paired():
    rng = np.random.default_rng(0)
    b = rng.random(2000)
    a = b + rng.normal(0, 0.001, 2000)
    r = paired_test(a, b, margin=0.01)
    assert r["verdict"] == "setara" and r["p_tost"] < 0.05
    assert tost(np.full(10, 0.5) + rng.normal(0, 0.01, 10), 0.01) > 0.5
