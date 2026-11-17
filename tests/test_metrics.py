from app.metrics import (ResultRow, compute_optimal, curves, fold_assignment, level_table,
                         pass_at_1, pool_rows, subsample_curve)
from app.schema import SampleRecord


def S(ans, ok, prm=0.5, tokens=10):
    return SampleRecord(0, "", ans, ok, [prm], tokens, 0.1)


def test_subsample_full_pool_exact():
    pool = [S("3", True), S("3", True), S("5", False)]
    assert subsample_curve(pool, [3], "majority")[3] == 1.0


def test_subsample_k1_equals_pass1():
    pool = [S("3", True), S("5", False), S("5", False), S("7", False)]
    assert abs(subsample_curve(pool, [1], "majority", repeats=4000)[1] - 0.25) < 0.03
    assert pass_at_1(pool) == 0.25


def test_bon_prefers_high_prm():
    pool = [S("3", True, .9), S("5", False, .2), S("5", False, .2)]
    assert subsample_curve(pool, [3], "bon")[3] == 1.0
    assert subsample_curve(pool, [3], "majority")[3] == 0.0


def test_no_answer_samples_never_win():
    pool = [S(None, False), S(None, False), S("3", True)]
    assert subsample_curve(pool, [3], "majority")[3] == 1.0
    assert subsample_curve([S(None, False)], [1], "majority")[1] == 0.0


def test_subsample_deterministic():
    pool = [S(str(i % 3), i % 3 == 0, 0.1 * i) for i in range(10)]
    assert subsample_curve(pool, [2, 4], "bon", seed=1) == subsample_curve(pool, [2, 4], "bon", seed=1)


def test_pool_rows_shape():
    pool = [S("3", True, tokens=10), S("5", False, tokens=20)]
    rows = pool_rows("q1", pool, [1, 2], repeats=10)
    keys = {(r.method, r.budget) for r in rows}
    assert keys == {("majority", 1), ("majority", 2), ("bon", 1), ("bon", 2), ("pass@1", 1)}
    p1 = [r for r in rows if r.method == "pass@1"][0]
    assert p1.accuracy == 0.5 and p1.tokens == 15
    assert [r for r in rows if r.method == "majority" and r.budget == 2][0].tokens == 30


def test_compute_optimal_picks_per_level_best():
    levels = {f"a{i}": 1 for i in range(4)} | {f"b{i}": 2 for i in range(4)}
    rows = [ResultRow(q, "majority", 4, 1.0 if q[0] == "a" else 0.0, 0, 0) for q in levels] + \
           [ResultRow(q, "bon", 4, 0.0 if q[0] == "a" else 1.0, 0, 0) for q in levels]
    assert compute_optimal(rows, levels)[4] == 1.0
    assert curves(rows)["majority"][4] == 0.5


def test_compute_optimal_uses_held_out_fold():
    levels = {f"q{i}": 1 for i in range(4)}
    folds = fold_assignment(levels, seed=0)
    assert sorted(folds.values()) == [0, 0, 1, 1]
    # each method is perfect only on the fold where the other fold would never pick it
    rows = [ResultRow(q, "majority", 4, float(folds[q] == 0), 0, 0) for q in levels] + \
           [ResultRow(q, "bon", 4, float(folds[q] == 1), 0, 0) for q in levels]
    assert compute_optimal(rows, levels)[4] < 1.0


def test_level_table():
    levels = {"a": 1, "b": 1, "c": 2}
    rows = [ResultRow("a", "bon", 4, 1.0, 0, 0), ResultRow("b", "bon", 4, 0.0, 0, 0),
            ResultRow("c", "bon", 4, 1.0, 0, 0)]
    t = level_table(rows, levels)
    assert {"level": 1, "method": "bon", "budget": 4, "accuracy": 0.5, "n": 2} in t
    assert {"level": 2, "method": "bon", "budget": 4, "accuracy": 1.0, "n": 1} in t
