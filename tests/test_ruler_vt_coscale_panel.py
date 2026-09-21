from experiments.analyze_ruler_vt_coscale_panel import cluster_summary
from experiments.analyze_ruler_vt_coscale import endpoint_suffix


def test_cluster_summary_uses_registered_three_seed_t_interval() -> None:
    result = cluster_summary([0.01, 0.02, 0.03])
    assert result["mean"] == 0.02
    assert result["degrees_of_freedom"] == 2
    assert result["ci95_low"] < 0.02 < result["ci95_high"]


def test_cluster_summary_rejects_unregistered_seed_count() -> None:
    try:
        cluster_summary([0.01, 0.02])
    except ValueError as error:
        assert "three or four" in str(error)
    else:
        raise AssertionError("two-seed panel should be rejected")


def test_alternative_middle_endpoint_has_unambiguous_result_suffix() -> None:
    assert endpoint_suffix("pythia-1b") == "1b"
    assert endpoint_suffix("pythia-1.4b") == "1p4b"
