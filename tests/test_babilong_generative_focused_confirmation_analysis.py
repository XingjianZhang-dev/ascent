from experiments.analyze_babilong_generative_focused_confirmation import (
    focused_estimands,
    mean_ci,
)


def metric(values):
    return [{"ascent": {"mean": value}, "gain": {"mean": value}} for value in values]


def test_mean_ci_requires_all_ten_panels():
    try:
        mean_ci([1.0] * 9)
    except ValueError as error:
        assert "ten panels" in str(error)
    else:
        raise AssertionError("nine panels must not be accepted")


def test_focused_estimands_use_registered_paired_contrasts():
    results = {
        ("smollm2-135m-instruct", 2): metric([0.10] * 10),
        ("smollm2-360m-instruct", 2): metric([0.30] * 10),
        ("smollm2-360m-instruct", 3): metric([0.35] * 10),
        ("smollm2-1p7b-instruct", 2): metric([0.50] * 10),
        ("smollm2-1p7b-instruct", 3): metric([0.70] * 10),
    }
    summary = focused_estimands(results)
    assert abs(summary["first_coscale_excess"]["mean"] - 0.20) < 1e-12
    assert abs(summary["second_registered_interaction"]["mean"] - 0.15) < 1e-12
    assert abs(summary["upper_diagonal_gain_increment"]["mean"] - 0.40) < 1e-12
    assert abs(summary["upper_coscale_excess"]["mean"] - 0.35) < 1e-12
