import pytest

from ascent.gpt_neox_replay import relative_layer_index


def test_relative_layer_index_uses_registered_floor_rule() -> None:
    assert relative_layer_index(24, 0.5) == 12
    assert relative_layer_index(32, 0.25) == 8


@pytest.mark.parametrize("layers,rho", [(1, 0.5), (12, 0.0), (12, 1.0)])
def test_relative_layer_index_rejects_invalid_settings(layers: int, rho: float) -> None:
    with pytest.raises(ValueError):
        relative_layer_index(layers, rho)
