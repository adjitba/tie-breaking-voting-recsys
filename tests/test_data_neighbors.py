import numpy as np
import pandas as pd
import pytest

from data import filter_min_counts, split_random_per_user
from neighbors import RatingMatrix, candidate_items, pearson_matrix, pearson_matrix_sparse, top_k_neighbors

nan = np.nan


def test_pearson_manual():
    # u0 dan u1 merating 3 item bersama; rata-rata dihitung dari seluruh rating user
    R = np.array([[5, 3, 1, nan],
                  [4, 3, 2, 5],
                  [nan, nan, 1, 2]])
    sim = pearson_matrix(R, ~np.isnan(R), min_common=3)
    m0, m1 = 3.0, 3.5
    a = np.array([5, 3, 1]) - m0
    b = np.array([4, 3, 2]) - m1
    expected = (a @ b) / np.sqrt((a @ a) * (b @ b))
    assert sim[0, 1] == pytest.approx(expected)
    assert sim[1, 0] == pytest.approx(expected)
    assert np.isnan(sim[0, 2])  # hanya 1 item bersama
    assert np.isnan(sim[0, 0])


def test_top_k_dan_kandidat():
    sim_row = np.array([nan, 0.9, 0.5, nan, 0.9])
    assert top_k_neighbors(sim_row, 2).tolist() == [1, 4]
    M = np.array([[1, 0, 0, 0],
                  [1, 1, 0, 1],
                  [0, 1, 1, 1],
                  [1, 1, 1, 0]], dtype=bool)
    # target 0, tetangga 1, 2, 3: item 1 (3 tetangga), 2 (2), 3 (2); item 0 sudah dirating target
    assert candidate_items(M, np.array([1, 2, 3]), 0, min_support=2).tolist() == [1, 2, 3]
    assert candidate_items(M, np.array([1, 2]), 0, min_support=2).tolist() == [1, 3]


def test_filter_dan_split():
    rows = [(u, i, 4.0, 0) for u in range(5) for i in range(10)] + [(9, 0, 5.0, 0)]
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])
    f = filter_min_counts(df, min_user=5, min_item=3)
    assert 9 not in set(f["user"])
    train, test = split_random_per_user(f, 0.2, seed=0)
    assert len(train) + len(test) == len(f)
    assert (test.groupby("user").size() == 2).all()
    train2, test2 = split_random_per_user(f, 0.2, seed=0)
    assert test.equals(test2)


def test_rating_matrix():
    df = pd.DataFrame({"user": [3, 3, 7], "item": [10, 20, 10], "rating": [4.0, 2.0, 5.0]})
    rm = RatingMatrix.from_frame(df)
    assert rm.users.tolist() == [3, 7]
    assert rm.items.tolist() == [10, 20]
    assert rm.M.tolist() == [[True, True], [True, False]]


def test_pearson_sparse_sama_dengan_dense():
    rng = np.random.default_rng(1)
    R = rng.integers(1, 6, size=(40, 30)).astype(float)
    R[rng.random(R.shape) < 0.6] = nan
    M = ~np.isnan(R)
    a = pearson_matrix(R, M, 3)
    b = pearson_matrix_sparse(R, M, 3)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-5)


def test_gmcc_manual():
    from neighbors import gmcc_matrix
    R = np.array([[5, 3, 1, nan],
                  [4, 3, 2, 5],
                  [nan, nan, 1, 2]])
    M = ~np.isnan(R)
    mu = np.nansum(R) / M.sum()  # 31 / 9
    sim = gmcc_matrix(R, M, min_common=3)
    a = np.array([5, 3, 1]) - mu
    b = np.array([4, 3, 2]) - mu
    assert sim[0, 1] == pytest.approx((a @ b) / np.sqrt((a @ a) * (b @ b)))
    assert np.isnan(sim[0, 2])


def test_gmcc_profil_konstan_proposisi3():
    # u0 dan u1 memberi 5 pada semua item: Pearson tidak terdefinisi, GMCC = sign((5-mu)(5-mu)) = 1
    from neighbors import gmcc_matrix
    R = np.array([[5, 5, 5, 5],
                  [5, 5, 5, nan],
                  [1, 2, 1, 3]], dtype=float)
    M = ~np.isnan(R)
    assert np.isnan(pearson_matrix(R, M, 3)[0, 1])
    g = gmcc_matrix(R, M, 3)
    assert g[0, 1] == pytest.approx(1.0)
    mu = np.nansum(R) / M.sum()
    y = np.array([1, 2, 1, 3]) - mu  # u0 dan u2 berbagi 4 item
    assert g[0, 2] == pytest.approx(np.sign(5 - mu) * y.sum() / np.sqrt(4 * (y ** 2).sum()))


def test_gmcc_sparse_sama_dengan_dense():
    from neighbors import gmcc_matrix, gmcc_matrix_sparse
    rng = np.random.default_rng(2)
    R = rng.integers(1, 6, size=(40, 30)).astype(float)
    R[rng.random(R.shape) < 0.6] = nan
    R[0, ~np.isnan(R[0])] = 5.0  # satu user konstan
    M = ~np.isnan(R)
    a, ca = gmcc_matrix(R, M, 3, return_common=True)
    b, cb = gmcc_matrix_sparse(R, M, 3, return_common=True)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-5)
    assert np.array_equal(ca, cb)


def test_top_k_seri_dipecah_jumlah_item_bersama():
    sim_row = np.array([nan, 1.0, 0.9999999999, 1.0, 0.5])
    common = np.array([0, 3, 9, 7, 20])
    # 1.0 dan 0.9999999999 seri setelah pembulatan; urutan menurut item bersama: 2 (9), 3 (7), 1 (3)
    assert top_k_neighbors(sim_row, 3, common).tolist() == [2, 3, 1]
    assert top_k_neighbors(sim_row, 3).tolist() == [1, 2, 3]
