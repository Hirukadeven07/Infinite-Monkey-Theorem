from app.aggregate import answer_key, majority, tally, weighted_vote


def test_answer_key_normalizes_forms():
    assert answer_key(r"\dfrac{1}{2}") == answer_key(r"\frac{1}{2}")
    assert answer_key(r"\left( 3, \frac{\pi}{2} \right)") == answer_key(r"(3,\frac{\pi}{2})")
    assert answer_key(r"90^\circ") == answer_key("90")
    assert answer_key(r"\text{(B)}") == answer_key("(B)")
    assert answer_key(None) is None


def test_majority_groups_equivalent_forms():
    assert majority([r"\frac{1}{2}", "3", r"\dfrac{1}{2}"]) == r"\frac{1}{2}"


def test_majority_tie_earliest_wins():
    assert majority(["3", "5"]) == "3"


def test_none_never_beats_real_answer():  # Review Focus 1
    assert majority([None, None, "3"]) == "3"
    assert weighted_vote([None, None, "3"], [0.9, 0.9, 0.1]) == "3"
    assert majority([None, None]) is None
    assert majority([]) is None


def test_weighted_vote_sums_scores():
    assert weighted_vote(["3", "3", "5"], [0.2, 0.2, 0.9]) == "5"
    assert weighted_vote(["3", "3", "5"], [0.5, 0.5, 0.9]) == "3"


def test_tally_shape():
    t = tally(["3", None, "3"], None, [True, False, True])
    assert t[0] == {"answer": "3", "votes": 2, "weight": 2.0, "correct": True}
    assert t[1]["answer"] == "(no answer)" and t[1]["correct"] is False


def test_tally_weighted_order():
    t = tally(["3", "5", "5"], [0.9, 0.1, 0.1], [True, False, False])
    assert [e["answer"] for e in t] == ["3", "5"]
    assert t[0]["weight"] == 0.9 and t[1]["votes"] == 2
