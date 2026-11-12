"""Uji Maximin, Plurality, dan sinyal pembanding dengan contoh 3 tetangga, 4 kandidat."""
import numpy as np
import pytest

from pairwise import pairwise_stats
from rules import build_rules, maximin, plurality
from signals import mean_rating, rating_sum, resnick, support

nan = np.nan
#               A    B    C    D
EX1 = np.array([[5, 4, 3, nan],
                [3, 5, nan, 4],
                [4, nan, 5, 2]])


def test_maximin_contoh1():
    # Margin w_ij - w_ji: A: vs B 0, vs C 0, vs D 0 -> min 0
    # B: vs A 0, vs C +1, vs D +1 -> 0 ; C: vs A 0, vs B -1, vs D +1 -> -1 ; D: vs A 0, vs B -1, vs C -1 -> -1
    assert maximin(EX1, pairwise_stats(EX1)).tolist() == [0, 0, -1, -1]


def test_plurality_contoh1():
    # u1 favorit A (5), u2 favorit B (5), u3 favorit C (5)
    assert plurality(EX1, pairwise_stats(EX1)).tolist() == [1, 1, 1, 0]


def test_plurality_favorit_seri_dibagi_rata():
    R = np.array([[5, 5, 3.0], [4, nan, 2.0]])
    assert plurality(R, pairwise_stats(R)).tolist() == pytest.approx([1.5, 0.5, 0.0])


def test_build_rules_enam_aturan():
    rules = build_rules(4.0)
    assert list(rules) == ["copeland", "maximin", "borda", "plurality", "approval", "least_misery"]
    ps = pairwise_stats(EX1)
    assert rules["approval"](EX1, ps).tolist() == [2, 2, 1, 1]
    assert build_rules(3.0)["approval"](EX1, ps).tolist() == [3, 2, 2, 1]


def test_sinyal_dasar():
    ps = pairwise_stats(EX1)
    assert mean_rating(EX1, ps).tolist() == pytest.approx([4, 4.5, 4, 3])
    assert support(EX1, ps).tolist() == [3, 2, 2, 2]
    assert rating_sum(EX1, ps).tolist() == [12, 9, 8, 6]


def test_resnick_manual():
    sims = np.array([1.0, 0.5, -0.5])
    means = np.array([4.0, 4.0, 11 / 3])
    pred = resnick(EX1, pairwise_stats(EX1), neighbor_sims=sims, neighbor_means=means, target_mean=3.0)
    # Item A: num = 1*(5-4) + 0.5*(3-4) + (-0.5)*(4-11/3) = 1 - 0.5 - 1/6 ; den = 1 + 0.5 + 0.5 = 2
    assert pred[0] == pytest.approx(3.0 + (1 - 0.5 - 1 / 6) / 2)


def test_shrunk_mean_contoh_manual():
    # Contoh di diskusi: A (5,5), B (4,4,5,3), C (5,4,4), D (5); mu_C = 4,4; m = 2,5
    from signals import shrunk_mean
    nan = np.nan
    Rn = np.array([[5, 4, 5, 5],
                   [5, 4, 4, nan],
                   [nan, 5, 4, nan],
                   [nan, 3, nan, nan]], dtype=float)
    B = shrunk_mean(Rn, None)
    assert B == pytest.approx([21 / 4.5, 27 / 6.5, 24 / 5.5, 16 / 3.5])


def test_most_pleasure_dan_daftar_masthoff():
    import numpy as np
    from pairwise import pairwise_stats
    from rules import build_rules_masthoff, most_pleasure
    nan = np.nan
    R = np.array([[5, 4, nan], [3, nan, 2], [nan, 5, 1]], float)
    assert most_pleasure(R, pairwise_stats(R)).tolist() == [5, 5, 2]
    assert list(build_rules_masthoff(4.0)) == ["plurality", "borda", "copeland", "approval",
                                               "least_misery", "most_pleasure"]


def test_kandidat_aturan_baru_contoh_manual():
    import numpy as np
    from pairwise import pairwise_stats
    from rules import make_quality_approval, make_shrunk_approval, shrunk_plurality
    nan = np.nan
    # v1 favorit X,Z (5); v2 favorit Y (5); v3 favorit X,W (4)
    R = np.array([[5, 3, 5, nan], [nan, 5, 3, 2], [4, nan, nan, 4]], float)
    ps = pairwise_stats(R)
    # V = [1, 1, .5, .5], n = [2, 2, 2, 2], m = 2, p = 3/8
    sp = shrunk_plurality(R, ps)
    assert np.allclose(sp, (np.array([1, 1, .5, .5]) + 2 * 3 / 8) / 4)
    # A(>=4) = [2, 1, 1, 1], q = 5/8
    sa = make_shrunk_approval(4.0)(R, ps)
    assert np.allclose(sa, (np.array([2, 1, 1, 1]) + 2 * 5 / 8) / 4)
    # B: mu = 31/8, m = 2 -> B = [(9+7.75)/4, (8+7.75)/4, (8+7.75)/4, (6+7.75)/4]
    qa = make_quality_approval(4.0, 1.0, 5.0)(R, ps)
    B = (np.array([9, 8, 8, 6]) + 7.75) / 4
    assert np.allclose(qa, np.array([2, 1, 1, 1]) * (B - 1) / 4)
