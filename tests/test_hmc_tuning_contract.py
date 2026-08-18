from __future__ import annotations

from pathlib import Path

import pytest

import bayesfilter
from bayesfilter.inference import (
    BootstrapFixedMassAdapter,
    HMCTuningScope,
    active_hmc_tuning_routes,
    hmc_tuning_route_record,
    hmc_tuning_route_registry_payload,
    mass_artifact_signature,
    require_active_hmc_tuning_route,
)
from bayesfilter.inference.hmc import PrecomputedMassArtifact
from bayesfilter.inference import hmc_budget_ladder, hmc_kernel_tuning, hmc_tuning
from scripts.inventory_hmc_tuning_routes import inventory_payload


def test_route_registry_has_exactly_two_active_interfaces() -> None:
    active = active_hmc_tuning_routes()

    assert tuple(record.interface_name for record in active) == (
        "tune_hmc_kernel",
        "tune_fixed_transport_hmc_kernel",
    )
    assert all(record.artifact_authority for record in active)
    assert hmc_tuning_route_registry_payload()["schema"] == (
        "bayesfilter.hmc_tuning_route_registry.v1"
    )


def test_historical_route_cannot_claim_active_authority() -> None:
    record = hmc_tuning_route_record("tune_hmc_kernel_robust_broad_grid")

    assert record.role == "diagnostic"
    assert record.replacement == "tune_hmc_kernel"
    assert record.artifact_authority is False
    with pytest.raises(ValueError, match="not active"):
        require_active_hmc_tuning_route(record.interface_name)


def test_inventory_has_no_unclassified_or_stale_routes() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = inventory_payload(root)

    assert payload["unclassified"] == ()
    assert payload["stale_registry_entries"] == ()


def test_tuning_scope_binds_backend_coordinates_and_transport() -> None:
    scope = HMCTuningScope(
        target_scope="gaussian_oracle",
        adapter_signature="adapter-v1",
        coordinate_signature="theta-v1",
        transport_signature="affine-v1",
        parameter_dimension=2,
        backend="tensorflow_tfp",
        dtype="float64",
        xla_enabled=True,
        chain_execution_mode="tf_function",
    )

    assert scope.payload()["transport_signature"] == "affine-v1"
    assert scope.payload()["parameter_dimension"] == 2


def test_public_compatibility_replacements_export_from_both_package_layers() -> None:
    assert bayesfilter.BootstrapFixedMassAdapter is BootstrapFixedMassAdapter
    assert bayesfilter.mass_artifact_signature is mass_artifact_signature
    assert "HMCTuningScope" in bayesfilter.__all__
    assert "mass_artifact_signature" in bayesfilter.__all__


def test_all_legacy_mass_signature_paths_delegate_to_one_authority() -> None:
    import numpy as np

    artifact = PrecomputedMassArtifact.from_covariance(
        position=np.zeros(2),
        covariance=np.array([[2.0, 0.25], [0.25, 1.0]]),
        adapter_signature="tuning-contract-mass-v1",
        covariance_source="test",
        jitter=0.0,
    )

    signatures = {
        mass_artifact_signature(artifact),
        hmc_kernel_tuning._mass_artifact_signature(artifact),
        hmc_tuning._mass_artifact_signature(artifact),
        hmc_budget_ladder._mass_artifact_signature(artifact),
    }
    assert len(signatures) == 1
