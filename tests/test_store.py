from app.metrics import ResultRow
from app.schema import SampleRecord
from app.store import Store


def test_runs_roundtrip(tmp_path):
    st = Store(tmp_path / "runs" / "t.db")
    rid = st.create_run("bench", {"per_level": 2}, {"policy_model": "m"})
    run = st.get_run(rid)
    assert run["kind"] == "bench" and run["config"] == {"per_level": 2} and run["lock"] == {"policy_model": "m"}
    assert run["status"] == "created"
    st.set_status(rid, "running")
    assert st.get_run(rid)["status"] == "running"
    assert [r["id"] for r in st.list_runs("bench")] == [rid]
    assert st.list_runs("live") == []


def test_rows_samples_and_items(tmp_path):
    st = Store(tmp_path / "t.db")
    rid = st.create_run("bench", {}, {})
    st.save_rows(rid, [ResultRow("q1", "bon", 4, 0.5, 40, 1.0)])
    st.save_rows(rid, [ResultRow("q1", "bon", 4, 0.75, 40, 1.0)])  # replace, not duplicate
    assert st.rows(rid) == [ResultRow("q1", "bon", 4, 0.75, 40, 1.0)]
    st.add_samples(rid, "q1", "pool", 4, [SampleRecord(0, "t", "3", True, [0.1, 0.9], 5, 0.2)])
    assert st.samples(rid, "q1")[0].prm == 0.9
    st.mark_done(rid, "q1", "pool", 4, "A100", {"x": 1})
    assert st.done_items(rid) == {("q1", "pool", 4)}
