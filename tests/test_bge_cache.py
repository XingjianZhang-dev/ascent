import numpy as np

from experiments.prepare_babilong_bge_cache import bm25_scores, ranks_descending


def test_bm25_scores_lexical_match_without_task_vocabulary() -> None:
    scores = bm25_scores(
        ["Aster entered north.", "Beryl carried widget."], "Where is widget?"
    )
    assert scores[1] > scores[0]


def test_rank_ties_prefer_recent_passage() -> None:
    ranks = ranks_descending(np.asarray([1.0, 1.0, 0.0]))
    assert ranks.tolist() == [2, 1, 3]
