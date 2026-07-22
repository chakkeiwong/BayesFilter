from __future__ import annotations

import ast
import inspect
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.highdim import ledh_contract_e_streaming_tf as streaming
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


ROOT = Path(__file__).resolve().parents[2]
PHASE4_FIXTURE = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase4-small-fixture-freeze-2026-07-13.json"
)
PHASE5_FIXTURE = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase5-tiny-fixture-freeze-v2-2026-07-14.json"
)


def _convert(value: Any) -> Any:
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, str):
        return float(Fraction(value))
    return value


def _phase4() -> dict[str, Any]:
    return json.loads(PHASE4_FIXTURE.read_text(encoding="utf-8"))


def _phase5() -> dict[str, Any]:
    return json.loads(PHASE5_FIXTURE.read_text(encoding="utf-8"))


def _phase4_tensor(name: str) -> tf.Tensor:
    return tf.constant(_convert(_phase4()[name]), tf.float64)


def _dense_plan(
    scaled: tf.Tensor,
    logw: tf.Tensor,
    epsilon: tf.Tensor,
    epsilon0: tf.Tensor,
    scaling: tf.Tensor,
) -> tf.Tensor:
    fixture = _phase4()
    particle_count = tf.shape(scaled)[1]
    log_n = tf.math.log(tf.cast(particle_count, tf.float64))
    uniform_logw = -log_n * tf.ones_like(logw)
    cost = annealed_transport_tf._filterflow_exact_cost(scaled, scaled)
    alpha, beta, _running, _iterations = (
        annealed_transport_tf._filterflow_manual_dense_finite_sinkhorn_outputs(
            logw,
            uniform_logw,
            cost,
            cost,
            cost,
            cost,
            epsilon=epsilon,
            epsilon0=epsilon0,
            scaling=scaling,
            steps=int(fixture["finite_sinkhorn_steps"]),
        )
    )
    return annealed_transport_tf._filterflow_exact_transport_from_potentials(
        scaled,
        alpha,
        beta,
        epsilon,
        logw,
        tf.cast(particle_count, tf.float64),
    )


def test_streaming_marginals_match_exact_dense_coupling_definitions() -> None:
    fixture = _phase4()
    scaled = _phase4_tensor("scaled_geometry")
    particles = _phase4_tensor("source_particles")
    weights = _phase4_tensor("normalized_weights")
    logw = tf.math.log(weights)
    epsilon = _phase4_tensor("epsilon")
    epsilon0 = _phase4_tensor("epsilon0")
    scaling = _phase4_tensor("scaling")
    tiling = fixture["chunk_tilings"][0]
    kwargs = {
        "steps": int(fixture["finite_sinkhorn_steps"]),
        "row_chunk_size": int(tiling["row_chunk_size"]),
        "col_chunk_size": int(tiling["col_chunk_size"]),
    }
    actual = streaming._streaming_row_quotient_forward_core(
        scaled, particles, logw, epsilon, epsilon0, scaling, **kwargs
    )
    plan = _dense_plan(scaled, logw, epsilon, epsilon0, scaling)
    dense_row_mass = tf.reduce_sum(plan, axis=2)
    dense_column_mass = tf.reduce_sum(plan, axis=1)
    dense_post_quotient_column_mass = tf.reduce_sum(
        plan / dense_row_mass[:, :, None], axis=1
    )
    dense_column_target = tf.cast(tf.shape(plan)[1], tf.float64) * weights

    tf.debugging.assert_near(actual["mass"], dense_row_mass, atol=3e-15, rtol=3e-15)
    tf.debugging.assert_near(
        actual["column_mass"], dense_column_mass, atol=3e-15, rtol=3e-15
    )
    tf.debugging.assert_near(
        actual["post_quotient_column_mass"],
        dense_post_quotient_column_mass,
        atol=3e-15,
        rtol=3e-15,
    )
    tf.debugging.assert_equal(actual["row_target"], tf.ones_like(dense_row_mass))
    tf.debugging.assert_near(
        actual["column_target"], dense_column_target, atol=3e-15, rtol=3e-15
    )
    tf.debugging.assert_near(
        actual["row_signed_residual"],
        dense_row_mass - 1.0,
        atol=3e-15,
        rtol=3e-15,
    )
    tf.debugging.assert_near(
        actual["column_signed_residual"],
        dense_column_mass - dense_column_target,
        atol=3e-15,
        rtol=3e-15,
    )
    tf.debugging.assert_near(
        actual["post_quotient_column_signed_residual"],
        dense_post_quotient_column_mass - dense_column_target,
        atol=3e-15,
        rtol=3e-15,
    )


def test_canonical_gap_telemetry_is_exact_and_scalar_score_are_preserved() -> None:
    fixture = _phase5()
    prepared = canonical._as_prepared_tensors(
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
    theta = tf.constant(_convert(fixture["center_theta"]), tf.float64)
    kwargs = {
        "steps": 2,
        "balance_steps": 0,
        "row_chunk_size": 2,
        "col_chunk_size": 2,
    }
    primal = canonical._canonical_primal_core(theta, prepared, **kwargs)
    tangent = canonical._canonical_manual_jvp_core(theta, prepared, **kwargs)

    tf.debugging.assert_equal(
        primal["covariance_gap_history"],
        primal["target_covariance_history"] - primal["plus_covariance_history"],
    )
    tf.debugging.assert_near(
        primal["covariance_gap_eigenvalue_history"],
        tf.linalg.eigvalsh(primal["covariance_gap_history"]),
        atol=2e-15,
        rtol=2e-15,
    )
    tf.debugging.assert_near(
        tf.reduce_sum(primal["covariance_gap_eigenvalue_history"], axis=-1),
        tf.linalg.trace(primal["covariance_gap_history"]),
        atol=2e-15,
        rtol=2e-15,
    )
    assert float(primal["objective"]).hex() == "-0x1.55564a66d9846p+2"
    assert [float(value).hex() for value in tangent["score"].numpy()] == [
        "-0x1.c993b9119c773p-2",
        "-0x1.cad12b05cc706p-3",
        "0x1.cc0ca41e05578p-5",
        "-0x1.c6af389364ccep+1",
        "-0x1.2fce89f3bf0c9p+2",
    ]


def test_production_marginal_reporter_has_no_two_particle_axis_allocation() -> None:
    source = inspect.getsource(streaming._streaming_column_masses_from_potentials_core)
    tree = ast.parse(source)
    assert "_filterflow_exact_cost" not in source
    assert "_filterflow_exact_transport_from_potentials" not in source
    assert "num_particles, num_particles" not in source
    assert "particle_count, particle_count" not in source
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            names = [
                item.id for item in node.elts if isinstance(item, ast.Name)
            ]
            assert names.count("particle_count") < 2
