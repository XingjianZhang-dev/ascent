from experiments.prepare_babilong_semantic_holdout import semantic_fingerprint


def test_semantic_fingerprint_ignores_distractor_text_and_target() -> None:
    left = {
        "input": "Noise one. Mary went to the garden.",
        "question": "Where is Mary? ",
        "target": "garden",
    }
    right = {
        "input": "Different noise. Mary travelled to the garden.",
        "question": "where is mary?",
        "target": "wrong-on-purpose",
    }
    assert semantic_fingerprint("qa1", left) == semantic_fingerprint("qa1", right)


def test_semantic_fingerprint_changes_with_event_sequence() -> None:
    left = {
        "input": "Mary went to the garden.",
        "question": "Where is Mary?",
    }
    right = {
        "input": "Mary went to the office.",
        "question": "Where is Mary?",
    }
    assert semantic_fingerprint("qa1", left) != semantic_fingerprint("qa1", right)
