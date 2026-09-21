import pytest

from experiments.analyze_latent_scale_curve import slug, three_seed_cluster_summary


def test_three_seed_cluster_interval_uses_seed_variation() -> None:
    result = three_seed_cluster_summary([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["ci95_low"] < 0.0
    assert result["ci95_high"] > 4.0


def test_cluster_summary_rejects_wrong_number_of_seeds() -> None:
    with pytest.raises(ValueError):
        three_seed_cluster_summary([1.0, 2.0])


def test_result_slug_is_stable() -> None:
    assert slug("pythia-2.8b") == "pythia_2p8b"
