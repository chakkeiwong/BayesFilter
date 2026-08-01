from __future__ import annotations

import argparse

import pytest

from docs.benchmarks import diagnose_ledh_predator_generalized_fd_root_cause as diagnostic


@pytest.mark.parametrize(
    ("row", "time_steps", "num_particles", "chunk_size"),
    (
        ("predator-prey", 1, 2, 2),
        ("generalized-sv", 2, 8, 4),
    ),
)
def test_reference_diagnostic_binds_identical_value_graph_and_manual_jvp(
    row: str,
    time_steps: int,
    num_particles: int,
    chunk_size: int,
) -> None:
    cli = argparse.Namespace(
        row=row,
        time_steps=time_steps,
        num_particles=num_particles,
        seed=81120,
        sinkhorn_iterations=1,
        row_chunk_size=chunk_size,
        col_chunk_size=chunk_size,
        particle_chunk_size=chunk_size,
    )

    result = diagnostic.diagnose(cli)

    assert result["objective_route_equality"]["status"] == "pass"
    assert result["precision"] == {
        "dtype": "float64",
        "tf32_mode": "disabled",
        "execution_target": "cpu_only_reference",
        "cuda_visible_devices": "-1",
        "jit_compile": False,
    }
    assert result["prepared_input_fingerprint"]["tensor_leaf_count"] > 0
    assert len(result["parameters"]) == len(result["parameter_names"])
    if row == "generalized-sv":
        assert result["manual_jvp_vs_full_transport_autodiff_status"] == "pass"
    for parameter in result["parameters"]:
        assert "full_transport_autodiff_score" in parameter
        assert "manual_vs_full_transport_autodiff" in parameter
        assert parameter["finite_difference_ladder"][0]["strategy"] == "legacy_absolute"
        assert len(parameter["finite_difference_ladder"]) == 1 + len(
            diagnostic.RELATIVE_STEP_COEFFICIENTS
        )
        assert all(entry["effective_step"] > 0.0 for entry in parameter["finite_difference_ladder"])
        assert all(
            "versus_full_transport_autodiff" in entry
            for entry in parameter["finite_difference_ladder"]
        )
