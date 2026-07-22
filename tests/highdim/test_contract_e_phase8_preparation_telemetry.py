from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as preparation


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase5-tiny-fixture-freeze-v2-2026-07-14.json"
)
KWARGS = {
    "steps": 2,
    "balance_steps": 100,
    "row_chunk_size": 4,
    "col_chunk_size": 4,
}


def _convert(value: Any) -> Any:
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, str):
        return float(Fraction(value))
    return value


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _prepared_fixture() -> dict[str, tf.Tensor]:
    fixture = _fixture()
    return canonical._as_prepared_tensors(
        {
            "observations": _convert(fixture["observations"]),
            "initial_noise": _convert(fixture["initial_noise"]),
            "transition_noise": _convert(fixture["transition_noise"]),
            "fixed_reset_mask": fixture["fixed_reset_mask"],
            "residual_design": _convert(fixture["residual_design"]),
            "prepared_ridge": _convert(fixture["prepared_ridge"]),
            "epsilon": _convert(fixture["transport"]["epsilon"]),
            "scaling": _convert(fixture["transport"]["scaling"]),
        }
    )


def _build(dtype: tf.dtypes.DType = tf.float64) -> dict[str, Any]:
    return preparation.prepare_contract_e_lgssm_inputs(
        observations=[[0.1, -0.2, 0.3], [0.0, 0.4, -0.1]],
        estimator_seeds=[81120, 81121],
        num_particles=4,
        fixed_reset_mask=[[True, False], [False, True]],
        prepared_ridge=[[0.25, 0.5], [0.75, 1.0]],
        epsilon=0.5,
        scaling=0.75,
        sinkhorn_steps=2,
        balance_steps=100,
        row_chunk_size=4,
        col_chunk_size=4,
        dtype=dtype,
    )


def test_preparation_is_repeatable_domain_separated_and_source_bound() -> None:
    first = _build()
    second = _build()
    assert first["identity"] == second["identity"]
    for name in first["prepared"]:
        tf.debugging.assert_equal(first["prepared"][name], second["prepared"][name])
    tensors = first["prepared"]
    assert tensors["initial_noise"].shape == (2, 4, 3)
    assert tensors["transition_noise"].shape == (2, 2, 4, 3)
    assert tensors["residual_design"].shape == (2, 2, 4, 3)
    assert bool(
        tf.reduce_any(tensors["initial_noise"][0] != tensors["initial_noise"][1])
    )
    assert bool(
        tf.reduce_any(
            tensors["transition_noise"][:, 0] != tensors["residual_design"][:, 0]
        )
    )
    identity = first["identity"]
    assert identity["rng_algorithm"] == "philox"
    assert identity["key_encoding"] == "[root_seed, domain_tag]"
    assert identity["root_seeds_in_order"] == [81120, 81121]
    assert identity["balance_steps"] == 100
    assert set(identity["tensor_sha256"]) == set(tensors)


def test_preparation_identity_binds_terminal_balance_steps() -> None:
    first = _build()
    second = preparation.prepare_contract_e_lgssm_inputs(
        observations=[[0.1, -0.2, 0.3], [0.0, 0.4, -0.1]],
        estimator_seeds=[81120, 81121],
        num_particles=4,
        fixed_reset_mask=[[True, False], [False, True]],
        prepared_ridge=[[0.25, 0.5], [0.75, 1.0]],
        epsilon=0.5,
        scaling=0.75,
        sinkhorn_steps=2,
        balance_steps=99,
        row_chunk_size=4,
        col_chunk_size=4,
        dtype=tf.float64,
    )
    first_identity = dict(first["identity"])
    second_identity = dict(second["identity"])
    assert first_identity.pop("balance_steps") == 100
    assert second_identity.pop("balance_steps") == 99
    assert first_identity == second_identity


def test_residual_design_is_centered_in_float64_before_final_cast() -> None:
    residual = _build()["prepared"]["residual_design"]
    tf.debugging.assert_near(
        tf.reduce_sum(residual, axis=2),
        tf.zeros([2, 2, 3], tf.float64),
        atol=2.0e-15,
        rtol=0.0,
    )
    float32 = _build(tf.float32)["prepared"]
    assert float32["residual_design"].dtype == tf.float32
    assert float32["initial_noise"].dtype == tf.float32


@pytest.mark.parametrize(
    "override, match",
    [
        ({"estimator_seeds": [81120, 81120]}, "unique"),
        ({"prepared_ridge": [[0.0, 0.5], [0.75, 1.0]]}, "strictly positive"),
        ({"fixed_reset_mask": [[True, False]]}, "fixed_reset_mask"),
        ({"num_particles": 1}, "greater than one"),
    ],
)
def test_preparation_rejects_invalid_explicit_inputs(
    override: dict[str, Any], match: str
) -> None:
    kwargs = {
        "observations": [[0.1, -0.2, 0.3], [0.0, 0.4, -0.1]],
        "estimator_seeds": [81120, 81121],
        "num_particles": 4,
        "fixed_reset_mask": [[True, False], [False, True]],
        "prepared_ridge": [[0.25, 0.5], [0.75, 1.0]],
        "epsilon": 0.5,
        "scaling": 0.75,
        "sinkhorn_steps": 2,
        "balance_steps": 100,
        "row_chunk_size": 4,
        "col_chunk_size": 4,
        "dtype": tf.float64,
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=match):
        preparation.prepare_contract_e_lgssm_inputs(**kwargs)


def test_telemetry_matches_exact_reset_core_definitions() -> None:
    tensors = _prepared_fixture()
    theta = tf.constant(_convert(_fixture()["center_theta"]), tf.float64)
    result = canonical._canonical_primal_core(theta, tensors, **KWARGS)
    assert result["quotient_mass_history"].shape == (2, 2, 4)
    assert result["target_covariance_history"].shape == (2, 2, 3, 3)
    assert result["gap_chol_diagonal_history"].shape == (2, 2, 3)
    tf.debugging.assert_equal(result["active_reset_history"], tensors["fixed_reset_mask"])
    tf.debugging.assert_equal(result["realized_ridge_history"], tensors["prepared_ridge"])
    tf.debugging.assert_equal(
        result["quotient_row_residual_history"],
        tf.reduce_max(tf.abs(result["quotient_mass_history"] - 1.0), axis=2),
    )
    tf.debugging.assert_equal(
        result["mean_residual_history"],
        result["output_mean_history"] - result["target_mean_history"],
    )
    tf.debugging.assert_equal(
        result["raw_covariance_residual_history"],
        result["output_covariance_history"]
        - result["target_covariance_history"],
    )
    tf.debugging.assert_equal(
        result["raw_covariance_prediction_error_history"],
        result["raw_covariance_residual_history"]
        - result["predicted_raw_covariance_residual_history"],
    )


def test_superseded_tiny_fixture_chunk_is_rejected() -> None:
    with pytest.raises(ValueError, match="wrong under"):
        canonical._canonical_primal_core(
            tf.constant(_convert(_fixture()["center_theta"]), tf.float64),
            _prepared_fixture(),
            steps=2,
            balance_steps=0,
            row_chunk_size=2,
            col_chunk_size=2,
        )
