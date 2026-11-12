"""Nilai harapan pasti harus sama dengan rata-rata atas SEMUA permutasi di dalam kelompok seri."""
import itertools

import numpy as np
import pytest

from metrics import expected_topn, ndcg_at, precision_at, tie_blocks


def brute_force(scores_cols, is_rel, N, n_rel_ideal):
    """Rata-rata atas semua urutan yang konsisten dengan skor (enumerasi permutasi kelompok seri)."""
    order, blocks = tie_blocks(*scores_cols)
    perms_per_block = [list(itertools.permutations(order[a:a + g])) for a, g in blocks]
    items = np.arange(len(is_rel))
    rel = set(items[np.asarray(is_rel, bool)].tolist())
    # Agar IDCG sama dengan expected_topn, tambahkan item relevan fiktif di luar kandidat
    extra = list(range(10_000, 10_000 + max(0, n_rel_ideal - len(rel))))
    rel_ideal = rel | set(extra)
    p_sum = n_sum = 0.0
    count = 0
    for combo in itertools.product(*perms_per_block):
        ranked = np.array([i for block in combo for i in block])
        p_sum += precision_at(ranked, rel, N)
        n_sum += ndcg_at(ranked, rel_ideal, N)
        count += 1
    return p_sum / count, n_sum / count


@pytest.mark.parametrize("seed", range(150))
def test_expected_sama_dengan_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 8))
    O = rng.integers(0, 3, n).astype(float)
    P = np.round(rng.integers(-2, 3, n) / 2, 1)
    is_rel = rng.random(n) < 0.4
    N = int(rng.integers(1, n + 1))
    n_ideal = int(is_rel.sum()) + int(rng.integers(0, 3))
    for cols in [(O,), (O, P)]:
        exp = expected_topn(cols, is_rel, N, max(n_ideal, 1))
        bp, bn = brute_force(cols, is_rel, N, max(n_ideal, 1))
        assert exp["P"] == pytest.approx(bp)
        assert exp["NDCG"] == pytest.approx(bn)


def test_contoh_guo_semua_seri():
    # 31 kandidat seri, 1 relevan: E[Hit@10] = 10/31 (contoh di Guo dkk. 2026)
    O = np.zeros(31)
    is_rel = np.zeros(31, bool); is_rel[0] = True
    e = expected_topn((O,), is_rel, 10, 1)
    assert e["P"] * 10 == pytest.approx(10 / 31)
    assert e["NDCG"] == pytest.approx(sum(1 / np.log2(p + 1) for p in range(1, 11)) / 31, abs=1e-4)
    assert e["NDCG"] == pytest.approx(0.1466, abs=1e-4)


def test_diagnosis_batas():
    # Skor 5, 4, 4, 2 dengan N = 2: kelompok {4, 4} terpotong batas
    O = np.array([5, 4, 4, 2.0])
    e = expected_topn((O,), np.array([True, True, False, False]), 2, 2)
    assert e["tie_at_boundary"] and e["boundary_group_size"] == 2 and e["changeable"]
    e2 = expected_topn((O,), np.array([True, False, False, False]), 2, 1)
    assert e2["tie_at_boundary"] and not e2["changeable"]  # kelompok tanpa item relevan
    e3 = expected_topn((np.array([5, 5, 3, 1.0]),), np.array([True, False, True, False]), 2, 2)
    assert not e3["tie_at_boundary"]  # seri seluruhnya di dalam Top-2


def test_expected_values_brute_force():
    import itertools
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.integers(3, 7))
        O = rng.integers(0, 3, n).astype(float)
        vals = rng.integers(1, 100, n).astype(float)
        rel = rng.random(n) < 0.5
        for N in (2, 3, 10):
            got = expected_topn((O,), rel, N, max(1, rel.sum()), values=vals)["VAL"]
            tot, cnt = 0.0, 0
            for perm in itertools.permutations(range(n)):
                perm = np.array(perm)
                # urutan sah: menurun menurut O, seri boleh berurutan apa saja
                if np.all(np.diff(O[perm]) <= 0):
                    tot += vals[perm[:N]].mean()
                    cnt += 1
            assert got == pytest.approx(tot / cnt)
