import pytest

from experiments.analyze_babilong_qrag_direct_peer import cluster_summary


def test_cluster_summary_uses_panel_t_interval() -> None:
    result = cluster_summary([0.1] * 10)
    assert result["clusters"] == 10
    assert result["degrees_of_freedom"] == 9
    assert result["mean"] == pytest.approx(0.1)
    assert result["ci95_low"] == pytest.approx(0.1)
    assert result["ci95_high"] == pytest.approx(0.1)


def test_cluster_summary_rejects_one_panel() -> None:
    with pytest.raises(ValueError, match="at least two panels"):
        cluster_summary([0.1])
