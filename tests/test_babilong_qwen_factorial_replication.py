import pytest

from experiments.analyze_babilong_qwen_factorial_replication import (
    T95_DF9,
    T_ONE_SIDED_BONFERRONI_DF9_M6,
    expected_model_files,
    mean_ci,
)


def test_qwen_replication_interval_uses_ten_panel_df9_critical() -> None:
    values = [0.01 * index for index in range(1, 11)]
    interval = mean_ci(values)
    assert interval["mean"] == pytest.approx(0.055)
    assert interval["ci95_low"] == pytest.approx(
        interval["mean"] - T95_DF9 * interval["standard_error"]
    )


def test_familywise_directional_bound_is_stricter_than_individual_bound() -> None:
    values = [0.01 * index for index in range(1, 11)]
    individual = mean_ci(values)
    familywise = mean_ci(values, T_ONE_SIDED_BONFERRONI_DF9_M6)
    assert familywise["ci95_low"] < individual["ci95_low"]


def test_expected_model_files_preserves_single_and_sharded_manifests() -> None:
    single = {"model_artifact": {"path": "model.safetensors"}}
    shards = {
        "model_artifacts": [
            {"path": "model-00001-of-00002.safetensors"},
            {"path": "model-00002-of-00002.safetensors"},
        ]
    }
    assert expected_model_files(single) == [single["model_artifact"]]
    assert expected_model_files(shards) == shards["model_artifacts"]
