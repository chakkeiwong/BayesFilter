from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from bayesfilter.highdim.cubature_genut_adapters import reduced_sir_candidate_adapter
from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS
from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
    STRUCTURAL_UKF_SCOPE,
)


def test_reduced_sir_is_fail_closed_mechanics_only() -> None:
    try:
        reduced_sir_candidate_adapter()
    except ValueError as exc:
        assert "artificial mechanics fixture" in str(exc)
    else:
        raise AssertionError("reduced SIR must not be constructible as an actual model")

    adapter = reduced_sir_candidate_adapter(mechanics_fixture_only=True)
    assert adapter.state_dimension == 2


def test_existing_chapter18b_structural_target_is_in_executable_registry() -> None:
    structural = [cell for cell in EXECUTABLE_CELLS if cell.cell_id == "STR-UKF"]
    assert len(structural) == 1
    assert structural[0].parameter_dim == 5
    assert structural[0].parameter_names == ("rho", "sigma", "phi", "gamma", "R")
    assert "structural" in structural[0].target_description
    assert STRUCTURAL_UKF_SCOPE == "STR-UKF-five-probit-T100-structural-innovation-v1"
