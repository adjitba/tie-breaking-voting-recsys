"""Uji dengan contoh angka yang dihitung manual dalam diskusi perancangan protokol."""
import numpy as np
import pytest

from cardinal import cardinal_score, p_matrix
from metrics import ndcg_at, precision_at, tie_at_boundary, usr
from pairwise import pairwise_stats
from ranking import additive_score, beta_star, rank_order
from rules import RULES, borda, borda_net, copeland, least_misery

nan = np.nan

# Contoh utama: 3 tetangga, 4 kandidat (A, B, C, D)
#        A    B    C    D
EX1 = np.array([
    [5, 4, 3, nan],  # u1
    [3, 5, nan, 4],  # u2
    [4, nan, 5, 2],  # u3
])


def test_contoh1_w_dan_p():
    ps = pairwise_stats(EX1)
    A, B, C, D = range(4)
    # w_ij dan w_ji per pasangan (Langkah 1)
    assert (ps.W[A, B], ps.W[B, A]) == (1, 1)
    assert (ps.W[A, C], ps.W[C, A]) == (1, 1)
    assert (ps.W[A, D], ps.W[D, A]) == (1, 1)
    assert (ps.W[B, C], ps.W[C, B]) == (1, 0)
    assert (ps.W[B, D], ps.W[D, B]) == (1, 0)
    assert (ps.W[C, D], ps.W[D, C]) == (1, 0)
    p = p_matrix(ps)
    assert p[A, B] == pytest.approx(-0.5)
    assert p[A, C] == pytest.approx(0.5)
    assert p[A, D] == pytest.approx(0.5)
    assert p[B, C] == pytest.approx(1.0)
    assert p[B, D] == pytest.approx(1.0)
    assert p[C, D] == pytest.approx(3.0)
    assert np.allclose(p, -p.T)


def test_contoh1_skor_ordinal_dan_kardinal():
    ps = pairwise_stats(EX1)
    assert copeland(None, ps).tolist() == [0, 2, 0, -2]
    assert borda(None, ps).tolist() == [3, 3, 2, 1]
    assert borda_net(None, ps).tolist() == [0, 2, 0, -2]
    P = cardinal_score(ps)
    assert P.tolist() == pytest.approx([0.5, 2.5, 1.5, -4.5])
    assert P.sum() == pytest.approx(0.0)


def test_contoh1_urutan_akhir():
    ps = pairwise_stats(EX1)
    P = cardinal_score(ps)
    key = np.zeros(4)
    # Copeland + P(i): B > C > A > D
    assert rank_order(copeland(None, ps), P, key).tolist() == [1, 2, 0, 3]
    # Borda + P(i): B > A > C > D
    assert rank_order(borda(None, ps), P, key).tolist() == [1, 0, 2, 3]
    # Skor aditif dengan beta = 0,04 (Langkah 7)
    S = additive_score(borda(None, ps), P, 0.04)
    assert S.tolist() == pytest.approx([3.02, 3.10, 2.06, 0.82])


def test_contoh1_usr():
    ps = pairwise_stats(EX1)
    O, P = copeland(None, ps), cardinal_score(ps)
    assert usr(O) == 0.75
    assert usr(O, P) == 1.0


def test_beta_star_bentuk_umum():
    assert beta_star(1, 4, 4) == pytest.approx(1 / 24)  # ML-100K, n = 4
    assert beta_star(1, 3.5, 4) == pytest.approx(1 / 21)  # FilmTrust, n = 4
    assert beta_star(1, 9, 4) == pytest.approx(1 / 54)  # skala 1-10


def test_sifat1_selalu_terdefinisi():
    # X dan Y tidak punya tetangga bersama, tetapi P(i) tetap memutus seri
    #               X    Y    Z
    R = np.array([[5, nan, 3],
                  [nan, 4, 3]])
    ps = pairwise_stats(R)
    assert ps.Cnt[0, 1] == 0
    assert copeland(None, ps).tolist() == [1, 1, -2]
    assert cardinal_score(ps).tolist() == pytest.approx([2, 1, -3])
    assert rank_order(copeland(None, ps), cardinal_score(ps), np.zeros(3)).tolist() == [0, 1, 2]


def test_sifat2_transitif_pada_siklus():
    # A > B, B > C, C > A secara berpasangan; P(i) tetap memberi urutan C > B > A
    #               A    B    C
    R = np.array([[5, 4, nan],
                  [nan, 5, 4],
                  [3, nan, 5]])
    ps = pairwise_stats(R)
    assert copeland(None, ps).tolist() == [0, 0, 0]
    assert cardinal_score(ps).tolist() == pytest.approx([-1, 0, 1])
    assert rank_order(copeland(None, ps), cardinal_score(ps), np.zeros(3)).tolist() == [2, 1, 0]


@pytest.mark.parametrize("seed", range(200))
def test_proposisi_beta_star_setara_leksikografis(seed):
    """Untuk beta < beta*, urutan S = O + beta P identik dengan urutan leksikografis (O, lalu P)."""
    rng = np.random.default_rng(seed)
    m, n = rng.integers(2, 8), rng.integers(2, 12)
    R = rng.integers(1, 6, size=(m, n)).astype(float)
    R[rng.random((m, n)) < 0.4] = nan
    ps = pairwise_stats(R)
    O, P = copeland(None, ps), cardinal_score(ps)
    beta = 0.999 * beta_star(1.0, 4.0, n)
    S = additive_score(O, P, beta)
    key = rng.random(n)
    lex = rank_order(O, P, key)
    add = rank_order(np.round(S, 12), None, key)
    assert lex.tolist() == add.tolist()


def test_tie_rate_di_batas():
    def t(scores, N=2):
        return tie_at_boundary(np.array(scores, dtype=float), N=N)

    assert t([5, 4, 4, 2]) is True
    assert t([5, 5, 3, 1]) is False
    assert t([6, 4, 3, 3]) is False
    assert t([3, 3, 3, 1]) is True
    assert t([3, 3], N=2) is False  # kandidat tidak lebih dari N
    # Dengan kolom kedua (P), seri di batas bisa terputus
    O = np.array([5, 4, 4, 2.0]); P = np.array([0, 1, 2, 0.0])
    assert tie_at_boundary(O, P, N=2) is False


def test_precision_ndcg():
    ranked = np.array([10, 20, 30])
    rel = {20, 99}
    assert precision_at(ranked, rel, 2) == 0.5
    expected = (1 / np.log2(3)) / (1 / np.log2(2) + 1 / np.log2(3))
    assert ndcg_at(ranked, rel, 2) == pytest.approx(expected)


def test_approval_dan_least_misery_contoh1():
    ps = pairwise_stats(EX1)
    # Approval (rating >= 4): A dari u1, u3 = 2; B dari u1, u2 = 2; C dari u3 = 1; D dari u2 = 1
    assert RULES["approval"](EX1, ps).tolist() == [2, 2, 1, 1]
    # Least Misery: A min(5,3,4) = 3; B min(4,5) = 4; C min(3,5) = 3; D min(4,2) = 2
    assert least_misery(EX1, ps).tolist() == [3, 4, 3, 2]
