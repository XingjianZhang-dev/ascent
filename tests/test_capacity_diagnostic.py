from ascent.capacity_diagnostic import exact_kv_comparison


def test_clean_exact_kv_dominates_noisy_code_on_simple_recall() -> None:
    config = {
        "num_labels": 16,
        "crossover_probability": 0.18,
        "num_writes": 512,
        "metadata_bits_per_slot": 10,
        "scales": [
            {"name": "small", "state_budget_bits": 8192, "fingerprint_bits": 20, "ascent_rounds": 1},
            {"name": "large", "state_budget_bits": 20480, "fingerprint_bits": 28, "ascent_rounds": 4},
        ],
    }
    rows = exact_kv_comparison(config)
    assert all(row["exact_kv_dominates"] for row in rows)
    assert all(row["raw_minus_ascent_gain_nats"] > 0.0 for row in rows)

