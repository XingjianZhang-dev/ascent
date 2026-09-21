from __future__ import annotations

import pytest

from ascent.state_accounting import (
    affine_head_parameters,
    latent_state_bytes,
    state_element_ratio,
)


def test_dense_state_and_probe_accounting() -> None:
    assert latent_state_bytes(18, 896, 2) == 32_256
    assert affine_head_parameters(896, 11) == 9_867
    assert state_element_ratio(18, 896, 494_032_768) == pytest.approx(
        3.264560783142223e-5
    )


@pytest.mark.parametrize(
    "function,args",
    [
        (latent_state_bytes, (0, 896, 2)),
        (affine_head_parameters, (896, 1)),
        (state_element_ratio, (18, 896, 0)),
    ],
)
def test_accounting_rejects_nonpositive_dimensions(function, args) -> None:
    with pytest.raises(ValueError):
        function(*args)
