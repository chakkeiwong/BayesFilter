"""Construct the typed deterministic-field tuning inputs without launching HMC."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import tensorflow as tf

from bayesfilter.inference import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    FourChainMeanBandAcceptancePolicy,
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    TensorFlowHMCKernelTuningConfig,
    bind_neural_force_hmc_tuning_runner,
    tune_hmc_kernel,
)


def build_binding() -> Any:
    """Bind a deterministic non-gradient proposal field to an exact potential."""

    def proposal_field(position: tf.Tensor) -> tf.Tensor:
        position = tf.convert_to_tensor(position, tf.float64)
        return tf.stack(
            (
                position[..., 0] + 0.2 * position[..., 1],
                0.4 * position[..., 1],
            ),
            axis=-1,
        )

    def exact_endpoint_potential(position: tf.Tensor) -> tf.Tensor:
        position = tf.convert_to_tensor(position, tf.float64)
        return 0.5 * tf.reduce_sum(tf.square(position), axis=-1)

    return bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(
            function=proposal_field,
            identity="docs-deterministic-proposal-field-v1",
            semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
            coordinate_system="raw",
        ),
        target=FrozenTargetPotential(
            function=exact_endpoint_potential,
            identity="docs-exact-endpoint-potential-v1",
            coordinate_system="raw",
        ),
        target_scope="docs_neural_force_binding",
    )


def build_mechanics_config() -> TensorFlowHMCKernelTuningConfig:
    """Declare an intentionally tiny, non-authoritative documentation budget."""

    return TensorFlowHMCKernelTuningConfig(
        parameter_dimension=2,
        evidence_role="diagnostic_only",
        mass_window_results=(1,),
        step_adaptation_results=1,
        verification_results=1,
        max_leapfrog_steps=1,
        initial_step_size=0.1,
        budget_provenance="one-step documentation diagnostic",
        initial_step_size_provenance="documentation convenience hypothesis",
        geometry_provenance="unit-scale documentation diagnostic",
        target_scope="docs_neural_force_binding",
        acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(0.0, 1.0),
            per_chain_band=(0.0, 1.0),
        ),
        target_accept_prob=0.70,
        verification_repair_rounds=0,
        step_repair_factor=2.0,
        mass_shrinkage=0.10,
        covariance_jitter=1.0e-9,
        eigenvalue_floor=1.0e-9,
        max_condition_number=1.0e8,
        seed=(20260905, 1),
    )


def tune_deterministic_field(
    *,
    adapter: Any,
    initial_position: Any,
    parameter_scales: Any,
    output_dir: str | Path,
) -> Any:
    """Show the only valid public dispatch for this mechanics-only branch."""

    return tune_hmc_kernel(
        adapter=adapter,
        initial_position=initial_position,
        parameter_scales=parameter_scales,
        config=build_mechanics_config(),
        runner_binding=build_binding(),
        output_dir=output_dir,
    )


def main() -> Mapping[str, Any]:
    """Return both typed payloads; no tuner or chain runner is called."""

    binding_payload = build_binding().payload()
    config_payload = build_mechanics_config().payload()
    assert binding_payload["artifact_authority"] is False
    assert binding_payload["force_semantics"] == (
        DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS
    )
    assert binding_payload["endpoint_target_coordinate_system"] == "raw"
    assert config_payload["artifact_authority"] is False
    assert config_payload["trajectory_candidate_policy"] == (
        "powers_of_two_then_explicit_cap"
    )
    return {"binding": binding_payload, "config": config_payload}


if __name__ == "__main__":
    print(main()["binding"]["binding_hash"])
