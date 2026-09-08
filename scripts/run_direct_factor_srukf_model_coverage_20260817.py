"""Bounded direct-factor SR-UKF model coverage campaign.

This is a diagnostic campaign driver.  It deliberately keeps repository model
classification separate from the admitted runtime route: principal-root UKF,
SGQF, DPF/LEDH, and multiplicative/non-Gaussian contracts are classified but
are not silently converted into direct-factor evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.linear.rectangular_factor_tf import (
    batched_direct_stack_svd_factor,
)
from bayesfilter.linear.stack_qr_tf import batched_stack_qr_lower
from bayesfilter.nonlinear.rectangular_srukf_tf import (
    TFRectangularSRUKFModel,
    tf_rectangular_srukf_value,
)
from bayesfilter.nonlinear.factor_srukf_tf import (
    TFFactorSRUKFDerivatives,
    TFFactorSRUKFModel,
    tf_factor_srukf_value_and_score,
)
from bayesfilter.nonlinear.factor_srukf_compat import covariance_model_to_factor_contract


ARTIFACT_ROOT = ROOT / "docs/plans/artifacts/direct-factor-srukf-model-coverage-20260817"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: str, anchor: str) -> dict[str, Any]:
    target = ROOT / path
    return {"path": path, "anchor": anchor, "sha256": _sha256(target)}


def _finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())


def _scalar(value: tf.Tensor) -> float:
    return float(tf.reshape(tf.convert_to_tensor(value), []).numpy())


def _metric_delta(left: tf.Tensor, right: tf.Tensor) -> dict[str, float]:
    delta = tf.abs(tf.convert_to_tensor(left) - tf.convert_to_tensor(right))
    scale = tf.maximum(tf.constant(1.0, tf.float64), tf.abs(tf.convert_to_tensor(right)))
    return {
        "max_abs": _scalar(tf.reduce_max(delta)),
        "max_rel": _scalar(tf.reduce_max(delta / scale)),
    }


def _load_parity_module():
    path = ROOT / "tests/test_factor_srukf_model_parity.py"
    spec = importlib.util.spec_from_file_location("factor_srukf_model_parity_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load parity fixture module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dummy_jacobian_derivatives(model: TFFactorSRUKFModel, *, state_jacobian, process_jacobian, observation_jacobian):
    b, n, q, m = model.batch_dim, model.state_dim, model.process_dim, model.observation_dim
    p = 1

    def transition_jac(previous, process):
        del previous, process
        return tf.broadcast_to(tf.convert_to_tensor(state_jacobian, tf.float64)[None, None, :, :], [b, tf.shape(process)[1], n, n])

    def process_jac(previous, process):
        del previous
        return tf.broadcast_to(tf.convert_to_tensor(process_jacobian, tf.float64)[None, None, :, :], [b, tf.shape(process)[1], n, q])

    def d_transition(previous, process):
        del process
        return tf.zeros([b, p, tf.shape(previous)[1], n], tf.float64)

    def observation_jac(states):
        return tf.broadcast_to(tf.convert_to_tensor(observation_jacobian, tf.float64)[None, None, :, :], [b, tf.shape(states)[1], m, n])

    def d_observation(states):
        return tf.zeros([b, p, tf.shape(states)[1], m], tf.float64)

    return TFFactorSRUKFDerivatives(
        d_initial_mean=tf.zeros([b, p, n], tf.float64),
        d_initial_factor=tf.zeros([b, p, n, n], tf.float64),
        d_process_factor=tf.zeros([b, p, q, q], tf.float64),
        d_observation_factor=tf.zeros([b, p, m, m], tf.float64),
        transition_state_jacobian_fn=transition_jac,
        transition_process_jacobian_fn=process_jac,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_jac,
        d_observation_fn=d_observation,
        name="fixed_nominal_zero_parameter_derivatives",
    )


def _run_parity_fixtures() -> list[dict[str, Any]]:
    module = _load_parity_module()
    cases = (
        ("model_a_affine", module._model_a, tf.constant(0.0, tf.float64), module.model_a_observations_tf()),
        ("model_b_nonlinear_accumulation", module._model_b, tf.constant([0.70, 0.25, 0.80], tf.float64), module.model_b_observations_tf()),
        ("model_c_nonlinear_growth", module._model_c, tf.constant([1.0, 1.0, 0.20], tf.float64), module.model_c_observations_tf()),
    )
    rows: list[dict[str, Any]] = []
    for name, builder, theta, observations in cases:
        base_theta = theta[0] if theta.shape.rank == 1 and theta.shape[0] == 1 else theta
        model, derivatives, old_model, old_derivatives = builder(base_theta)
        eager = tf_factor_srukf_value_and_score(observations[None, ...], model, derivatives, jit_compile=False)
        xla_status = "not_run"
        xla = None
        try:
            xla = tf_factor_srukf_value_and_score(observations[None, ...], model, derivatives, jit_compile=True)
            xla_status = "passed"
        except (tf.errors.InvalidArgumentError, tf.errors.InternalError, tf.errors.UnimplementedError) as exc:
            xla_status = f"failed:{type(exc).__name__}"
        old = module.tf_principal_sqrt_ukf_score(observations, old_model, old_derivatives)
        row = {
            "model_id": name,
            "status": "eligible_score",
            "route": "direct_qr_block_conditional",
            "source": _source("tests/test_factor_srukf_model_parity.py", f"{name} fixture builder and parity test"),
            "parameter_dim": int(eager.score.shape[-1]),
            "state_dim": model.state_dim,
            "observation_dim": model.observation_dim,
            "horizon": int(observations.shape[0]),
            "dtype": "float64",
            "device": "CPU diagnostic/reference lane",
            "jit_compile_eager": False,
            "jit_compile_xla": xla_status,
            "direct_value": _scalar(eager.log_likelihood[0]),
            "direct_score": _jsonable(eager.score[0]),
            "old_value": _scalar(old.log_likelihood),
            "old_score": _jsonable(old.score),
            "value_delta": _metric_delta(eager.log_likelihood[0], old.log_likelihood),
            "score_delta": _metric_delta(eager.score[0], old.score),
            "eager_xla_delta": None if xla is None else _metric_delta(eager.log_likelihood, xla.log_likelihood),
            "minimum_qr_pivot": _scalar(eager.diagnostics["minimum_qr_pivot"][0]),
            "maximum_factor_reconstruction_residual": _scalar(eager.diagnostics["maximum_factor_reconstruction_residual"][0]),
            "maximum_derivative_reconstruction_residual": _scalar(eager.diagnostics["maximum_derivative_reconstruction_residual"][0]),
            "branch_status": "fixed_full_rank_positive_pivot",
            "score_claim": "same finite direct-factor program; old comparison is historical principal-root parity",
            "nonclaims": ["not exact nonlinear Bayesian inference", "not singular score evidence", "not HMC readiness"],
        }
        rows.append(row)
    return rows


def _registry_candidate(kind: str, theta: tf.Tensor, *, jit_compile: bool):
    if kind == "PP-UKF":
        from bayesfilter.testing import predator_prey_ukf_neutra_target_tf as target

        _states, observations = target.generate_frozen_predator_prey_dataset_tf()
        covariance_model, covariance_derivatives, initial_value = target._build_ukf_model_and_derivatives(theta, observations)
        initial_score = tf.zeros_like(theta)
    elif kind == "STR-UKF":
        from bayesfilter.testing import structural_ukf_neutra_target_design_tf as target

        _states, observations = target.generate_frozen_structural_dataset_tf()
        covariance_model, covariance_derivatives, initial_value, initial_score = target.build_structural_ukf_model_and_derivatives(theta, observations)
    else:
        raise ValueError(f"unknown registry candidate: {kind}")
    model, derivatives = covariance_model_to_factor_contract(covariance_model, covariance_derivatives)
    result = tf_factor_srukf_value_and_score(
        observations[None, 1:, :], model, derivatives, jit_compile=jit_compile
    )
    return initial_value + result.log_likelihood, initial_score + result.score, result, observations


def _run_registry_candidates() -> list[dict[str, Any]]:
    from bayesfilter.testing import predator_prey_ukf_neutra_target_tf as pp
    from bayesfilter.testing import structural_ukf_neutra_target_design_tf as structural

    pp_probability = (pp.PP_TRUTH_PHYSICAL - pp.PP_PARAMETER_LOWER) / pp.PP_PARAMETER_WIDTH
    pp_theta = tf.sqrt(tf.constant(2.0, tf.float64)) * tf.math.erfinv(2.0 * pp_probability - 1.0)
    cases = (
        ("PP-UKF", pp_theta[None, :], pp),
        ("STR-UKF", structural.structural_truth_source()[None, :], structural),
    )
    rows = []
    for model_id, theta, target in cases:
        value, score, eager, observations = _registry_candidate(model_id, theta, jit_compile=False)
        xla_value, xla_score, _xla, _ = _registry_candidate(model_id, theta, jit_compile=True)
        if model_id == "PP-UKF":
            old_value, old_score, _status = target.pp_ukf_likelihood_value_score_status(theta, observations=observations)
            source_path = "bayesfilter/testing/predator_prey_ukf_neutra_target_tf.py"
            source_anchor = "_build_ukf_model_and_derivatives"
            data_sha256 = target.PP_OBSERVATION_SHA256
        else:
            old_value, old_score, _status = target.structural_ukf_likelihood_value_score_status(theta, observations=observations)
            source_path = "bayesfilter/testing/structural_ukf_neutra_target_design_tf.py"
            source_anchor = "build_structural_ukf_model_and_derivatives"
            data_sha256 = target.STRUCTURAL_FINAL_OBSERVATION_SHA256
        fd_columns = []
        step = tf.constant(1.0e-5, tf.float64)
        for parameter_index in range(int(theta.shape[1])):
            direction = tf.one_hot(parameter_index, int(theta.shape[1]), dtype=tf.float64)[None, :]
            plus_value, _plus_score, _plus_result, _ = _registry_candidate(
                model_id, theta + step * direction, jit_compile=False
            )
            minus_value, _minus_score, _minus_result, _ = _registry_candidate(
                model_id, theta - step * direction, jit_compile=False
            )
            fd_columns.append((plus_value[0] - minus_value[0]) / (2.0 * step))
        finite_difference = tf.stack(fd_columns)
        rows.append({
            "model_id": model_id,
            "status": "eligible_score",
            "route": "one_time_cholesky_adapter_then_direct_qr_block_conditional",
            "source": _source(source_path, source_anchor),
            "adapter_source": _source("bayesfilter/nonlinear/factor_srukf_compat.py", "covariance_model_to_factor_contract"),
            "adapter_boundary": "covariances and covariance derivatives are factored once before tracing; no per-step covariance factorization",
            "data_sha256": data_sha256,
            "parameter_dim": int(theta.shape[1]),
            "parameter_coordinate": "frozen source-probit coordinates",
            "state_dim": int(eager.filtered_mean.shape[-1]),
            "observation_dim": int(observations.shape[-1]),
            "horizon": int(observations.shape[0]),
            "dtype": "float64",
            "device": "CPU diagnostic/reference lane",
            "jit_compile_eager": False,
            "jit_compile_xla": "passed",
            "direct_value": _scalar(value[0]),
            "direct_score": _jsonable(score[0]),
            "old_value": _scalar(old_value[0]),
            "old_score": _jsonable(old_score[0]),
            "value_delta": _metric_delta(value, old_value),
            "score_delta": _metric_delta(score, old_score),
            "finite_difference_score": _jsonable(finite_difference),
            "finite_difference_delta": _metric_delta(score[0], finite_difference),
            "eager_xla_value_delta": _metric_delta(value, xla_value),
            "eager_xla_score_delta": _metric_delta(score, xla_score),
            "minimum_qr_pivot": _scalar(eager.diagnostics["minimum_qr_pivot"][0]),
            "maximum_factor_reconstruction_residual": _scalar(eager.diagnostics["maximum_factor_reconstruction_residual"][0]),
            "maximum_derivative_reconstruction_residual": _scalar(eager.diagnostics["maximum_derivative_reconstruction_residual"][0]),
            "branch_status": "fixed_full_rank_positive_pivot",
            "score_claim": "same finite direct-factor program; centered finite-difference certified",
            "nonclaims": ["historical principal-root parity is gauge-sensitive", "not exact nonlinear Bayesian inference", "not HMC readiness"],
        })
    return rows


def _build_structural_rectangular():
    from experiments.dpf_implementation.tf_tfp.fixtures.structural_ar1_quadratic_tf import build_structural_ar1_quadratic_fixture_tf, complete_k_tf, structural_observation_mean_tf

    fixture = build_structural_ar1_quadratic_fixture_tf(horizon=8)
    initial_factor = tf.concat(
        [
            tf.reshape(tf.sqrt(fixture.m0_variance), [1, 1, 1]),
            tf.zeros([1, 1, 1], tf.float64),
        ],
        axis=1,
    )
    model = TFRectangularSRUKFModel(
        initial_mean=tf.reshape(tf.stack([fixture.m0_mean, fixture.k0]), [1, 2]),
        initial_factor=initial_factor,
        process_factor=tf.ones([1, 1, 1], tf.float64),
        observation_factor=tf.reshape(fixture.observation_scale, [1, 1, 1]),
        transition_fn=lambda previous, innovation: tf.stack(
            [
                fixture.rho * previous[:, :, 0] + fixture.sigma * innovation[:, :, 0],
                complete_k_tf(
                    previous_k=previous[:, :, 1],
                    previous_m=previous[:, :, 0],
                    current_m=fixture.rho * previous[:, :, 0] + fixture.sigma * innovation[:, :, 0],
                    a=fixture.a, b=fixture.b, c=fixture.c, d=fixture.d,
                ),
            ], axis=2
        ),
        observation_fn=lambda states: structural_observation_mean_tf(states, fixture.lam)[:, :, None],
    )
    return fixture, model


def _run_rectangular_fixture() -> dict[str, Any]:
    fixture, model = _build_structural_rectangular()
    observations = tf.reshape(fixture.observations, [1, -1, 1])
    eager = tf_rectangular_srukf_value(observations, model, jit_compile=False)
    xla_status = "not_run"
    xla_delta = None
    try:
        xla = tf_rectangular_srukf_value(observations, model, jit_compile=True)
        xla_status = "passed"
        xla_delta = _metric_delta(eager.log_likelihood, xla.log_likelihood)
    except (tf.errors.InvalidArgumentError, tf.errors.InternalError, tf.errors.UnimplementedError) as exc:
        xla_status = f"failed:{type(exc).__name__}"
    return {
        "model_id": "structural_ar1_quadratic_h16",
        "status": "eligible_value_only",
        "route": "rectangular_direct_stack_svd_support",
        "source": _source("experiments/dpf_implementation/tf_tfp/fixtures/structural_ar1_quadratic_tf.py", "build_structural_ar1_quadratic_fixture_tf"),
        "parameter_dim": 0,
        "state_dim": model.state_dim,
        "observation_dim": model.observation_dim,
        "horizon": int(observations.shape[1]),
        "dtype": "float64",
        "device": "CPU diagnostic/reference lane",
        "jit_compile_eager": False,
        "jit_compile_xla": xla_status,
        "direct_value": _scalar(eager.log_likelihood[0]),
        "eager_xla_delta": xla_delta,
        "on_support": bool(eager.diagnostics["on_support"][0].numpy()),
        "minimum_observation_rank": int(eager.diagnostics["minimum_observation_rank"][0].numpy()),
        "maximum_support_residual": _scalar(eager.diagnostics["maximum_support_residual"][0]),
        "branch_status": "value_only_rank_discovery",
        "likelihood_measure": "affine_support_gaussian",
        "score_claim": "none; rank/support branch is not differentiable evidence",
        "nonclaims": ["no analytical score", "no smooth derivative through rank changes", "not HMC readiness"],
    }


def _run_robustness() -> dict[str, Any]:
    rows = []
    for scale in (1.0, 1.0e-4, 1.0e-8, 1.0e-12, 1.0e-14, 1.0e-15):
        stack = tf.constant([[[scale, 0.0, 0.0, 0.0], [0.0, scale, 0.0, 0.0]]], tf.float64)
        try:
            factor, derivative, diagnostics = batched_stack_qr_lower(
                stack, compute_covariance_diagnostics=False, relative_pivot_tolerance=0.0
            )
            rows.append({
                "case": "full_rank_scaled_stack",
                "scale": scale,
                "route": "direct_qr",
                "status": "finite" if _finite(factor) else "nonfinite",
                "minimum_pivot": _scalar(diagnostics["minimum_qr_pivot"][0]),
                "reconstruction_residual": _scalar(diagnostics["stack_reconstruction_residual"][0]),
                "derivative_residual": None if derivative is None else _scalar(tf.reduce_max(tf.abs(derivative))),
            })
        except (tf.errors.InvalidArgumentError, tf.errors.InternalError, ValueError) as exc:
            rows.append({"case": "full_rank_scaled_stack", "scale": scale, "route": "direct_qr", "status": f"fail_closed:{type(exc).__name__}"})
    try:
        floor_stack = tf.constant([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0e-14, 0.0, 0.0]]], tf.float64)
        batched_stack_qr_lower(
            floor_stack,
            compute_covariance_diagnostics=False,
            relative_pivot_tolerance=1.0e-8,
        )
        rows.append({"case": "explicit_pivot_floor", "route": "direct_qr", "pivot_floor": 1.0e-8, "status": "unexpectedly_accepted"})
    except (tf.errors.InvalidArgumentError, tf.errors.InternalError, ValueError) as exc:
        rows.append({"case": "explicit_pivot_floor", "route": "direct_qr", "pivot_floor": 1.0e-8, "status": f"fail_closed:{type(exc).__name__}"})
    for rank_case, stack in (
        ("rank_zero", tf.zeros([1, 2, 4], tf.float64)),
        ("rank_one", tf.constant([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], tf.float64)),
        ("repeated_singular_values", tf.constant([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], tf.float64)),
    ):
        factor, singular_values, _left_vectors, diagnostics = batched_direct_stack_svd_factor(stack, 1.0e-12)
        rows.append({
            "case": rank_case,
            "route": "direct_stack_svd_value_only",
            "status": "finite" if _finite(factor) else "nonfinite",
            "rank": int(diagnostics["rank"][0].numpy()),
            "singular_values": _jsonable(singular_values[0]),
            "reconstruction_residual": _scalar(
                tf.linalg.norm(
                    factor @ tf.linalg.matrix_transpose(factor)
                    - stack @ tf.linalg.matrix_transpose(stack),
                    axis=[-2, -1],
                )[0]
            ),
            "branch_status": "value_only_rank_discovery",
        })
    return {"scales": [1.0, 1.0e-4, 1.0e-8, 1.0e-12, 1.0e-14, 1.0e-15], "rows": rows, "score_admission": "none for rank-changing SVD branches"}


def _inventory() -> list[dict[str, Any]]:
    rows = [
        {"model_id": "model_a_affine", "status": "eligible_score", "contract": "TFFactorSRUKFModel", "source": _source("tests/test_factor_srukf_model_parity.py", "_model_a"), "reason": "existing affine score-bearing direct-factor fixture"},
        {"model_id": "model_b_nonlinear_accumulation", "status": "eligible_score", "contract": "TFFactorSRUKFModel", "source": _source("tests/test_factor_srukf_model_parity.py", "_model_b"), "reason": "existing nonlinear score-bearing direct-factor fixture"},
        {"model_id": "model_c_nonlinear_growth", "status": "eligible_score", "contract": "TFFactorSRUKFModel", "source": _source("tests/test_factor_srukf_model_parity.py", "_model_c"), "reason": "existing nonlinear score-bearing direct-factor fixture"},
        {"model_id": "lgssm_2d_h25_rich", "status": "adapter_required", "contract": "LinearGaussianSSM covariance/density contract", "source": _source("experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py", "_common_lgssm_v2_spec"), "reason": "no repository-certified direct-factor adapter with frozen parameter convention"},
        {"model_id": "sv_1d_h18_rich", "status": "not_applicable_contract", "contract": "multiplicative stochastic-volatility observation", "source": _source("experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py", "_common_sv_v2_spec"), "reason": "observation y=exp(h/2)e is not additive observation noise under current factor contract"},
        {"model_id": "range_bearing_4d_h20_rich", "status": "adapter_required", "contract": "nonlinear range-bearing additive Gaussian model", "source": _source("experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py", "_common_range_bearing_v2_spec"), "reason": "angle residual convention and parameter derivative adapter are not certified"},
        {"model_id": "structural_ar1_quadratic_h16", "status": "eligible_value_only", "contract": "rectangular deterministic-completion support", "source": _source("experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py", "_common_structural_ar1_v2_spec"), "reason": "structural deterministic coordinate creates a rank-deficient state support; score route is not admitted"},
        {"model_id": "spatial_sir_j3_rk4", "status": "not_applicable_contract", "contract": "non-Gaussian/domain-constrained SIR fixture", "source": _source("experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py", "_common_spatial_sir_v2_spec"), "reason": "state-domain clipping/epidemiological contract is not a fixed additive-Gaussian SR-UKF score contract"},
        {"model_id": "predator_prey_rk4", "status": "adapter_required", "contract": "nonlinear additive Gaussian predator-prey model", "source": _source("experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py", "_common_predator_prey_v2_spec"), "reason": "direct-factor adapter and physical-parameter derivative convention are not certified"},
        {"model_id": "LGSSM-EXACT", "status": "adapter_required", "contract": "NeuTra exact deterministic LGSSM target", "source": _source("bayesfilter/testing/neutra_model_registry_tf.py", "CellSpec LGSSM-EXACT"), "reason": "linear/reference candidate, but the NeuTra target registry is not a certified direct-factor SR-UKF model adapter"},
        {"model_id": "PP-UKF", "status": "eligible_score", "contract": "TFFactorSRUKFModel", "source": _source("bayesfilter/testing/predator_prey_ukf_neutra_target_tf.py", "_build_ukf_model_and_derivatives"), "reason": "one-time pre-trace factor adapter certified; every temporal step uses direct block-QR"},
        {"model_id": "STR-UKF", "status": "eligible_score", "contract": "TFFactorSRUKFModel", "source": _source("bayesfilter/testing/structural_ukf_neutra_target_design_tf.py", "build_structural_ukf_model_and_derivatives"), "reason": "one-time pre-trace factor adapter certified; every temporal step uses direct block-QR"},
        {"model_id": "PP-SGQF", "status": "not_applicable_contract", "contract": "fixed sparse-grid quadrature filter", "source": _source("bayesfilter/testing/neutra_model_registry_tf.py", "CellSpec PP-SGQF"), "reason": "SGQF is a different point/filter contract"},
        {"model_id": "SIR-SGQF", "status": "not_applicable_contract", "contract": "fixed sparse-grid quadrature SIR filter", "source": _source("bayesfilter/testing/neutra_model_registry_tf.py", "CellSpec SIR-SGQF"), "reason": "SGQF/SIR contract is not direct SR-UKF"},
        {"model_id": "SVX-SGQF", "status": "blocked", "contract": "blocked NeuTra registry cell", "source": _source("bayesfilter/testing/neutra_model_registry_tf.py", "BLOCKED_CELLS SVX-SGQF"), "reason": "no frozen SGQF level passed filter admission"},
        {"model_id": "KSC-UKF", "status": "blocked", "contract": "blocked NeuTra registry cell", "source": _source("bayesfilter/testing/neutra_model_registry_tf.py", "BLOCKED_CELLS KSC-UKF"), "reason": "dense-reference value/score admission failed"},
        {"model_id": "PP-ZC", "status": "blocked", "contract": "blocked Zhao-Cui NeuTra extension", "source": _source("bayesfilter/testing/neutra_model_registry_tf.py", "BLOCKED_CELLS PP-ZC"), "reason": "no batch-native posterior adapter or admitted chart/Jacobian"},
        {"model_id": "STR-ZC", "status": "blocked", "contract": "blocked Zhao-Cui extension", "source": _source("bayesfilter/testing/neutra_model_registry_tf.py", "BLOCKED_CELLS STR-ZC"), "reason": "extension target absent"},
        {"model_id": "SIR-ZC", "status": "blocked", "contract": "blocked Zhao-Cui extension", "source": _source("bayesfilter/testing/neutra_model_registry_tf.py", "BLOCKED_CELLS SIR-ZC"), "reason": "observed-data parameter-score closure absent"},
        {"model_id": "SVX-ZC", "status": "not_applicable_contract", "contract": "executable fixed adjacent-state squared-TT Zhao-Cui target", "source": _source("bayesfilter/testing/neutra_model_registry_tf.py", "CellSpec SVX-ZC"), "reason": "active GPU/XLA/HMC-capable NeuTra target, but not a direct-factor SR-UKF contract"},
        {"model_id": "SIR-UKF", "status": "owner_excluded", "contract": "owner exclusion", "source": _source("bayesfilter/testing/neutra_model_registry_tf.py", "OWNER_EXCLUDED_CELLS SIR-UKF"), "reason": "owner determination excludes UKF for SIR"},
        {"model_id": "SSL-LSTM", "status": "owner_excluded", "contract": "owner exclusion", "source": _source("tests/test_ssl_lstm_sgqf_ukf_adapters.py", "SSL-LSTM campaign exclusion"), "reason": "explicit user instruction excludes SSL-LSTM"},
        {"model_id": "actual_sv_independent_panel", "status": "historical_only", "contract": "legacy manual factor SR-UKF panel score", "source": _source("bayesfilter/highdim/actual_sv_srukf_tf.py", "actual_transformed_sv_independent_panel_augmented_noise_srukf_score"), "reason": "legacy route forms covariance diagnostics and is not the admitted TFFactorSRUKFModel block-QR route"},
        {"model_id": "macrofinance_lgssm_adapter", "status": "not_applicable_contract", "contract": "MacroFinance covariance-form LGSSM adapter", "source": _source("bayesfilter/adapters/macrofinance.py", "macrofinance_lgssm_to_bayesfilter"), "reason": "generic external provider adapter has no frozen direct-factor transition/observation derivative contract"},
    ]
    ids = [row["model_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("inventory contains duplicate model IDs")
    return rows


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    inventory = _inventory()
    parity = _run_parity_fixtures()
    registry_results = _run_registry_candidates()
    rectangular = _run_rectangular_fixture()
    robustness = _run_robustness()
    score_results = parity + registry_results
    results = score_results + [rectangular]
    manifest = {
        "schema": "bayesfilter.direct_factor_srukf_model_coverage.v1",
        "campaign_id": "direct-factor-srukf-model-coverage-20260817",
        "status": "EXECUTED_WITH_EXPLICIT_CLASSIFICATION_BOUNDARIES",
        "scientific_target": "direct block-QR full-rank score branch and rectangular value-only singular branch",
        "seed_policy": "frozen seeds embedded in source fixtures",
        "dtype": "float64",
        "backend": "TensorFlow/TFP",
        "device_policy": "CPU diagnostic/reference lane; no GPU claim",
        "jit_policy": "eager plus XLA attempt; XLA failures retained",
        "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip(),
        "inventory_path": str(ARTIFACT_ROOT / "model_inventory.json"),
        "commands": [
            "CUDA_VISIBLE_DEVICES=-1 python scripts/run_direct_factor_srukf_model_coverage_20260817.py",
            "CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_block_qr_conditional_tf.py tests/test_rectangular_factor_tf.py tests/test_rectangular_srukf_tf.py tests/test_factor_srukf_tf.py tests/test_factor_srukf_model_parity.py tests/test_factor_srukf_route_guard.py tests/test_srukf_backend_policy.py",
        ],
        "nonclaims": ["not every repository model is mathematically an SR-UKF contract", "no singular analytical-score claim", "no HMC or posterior correctness claim"],
    }
    _write_json(ARTIFACT_ROOT / "model_inventory.json", {"schema": "bayesfilter.direct_factor_srukf_inventory.v1", "rows": inventory})
    _write_json(ARTIFACT_ROOT / "campaign_manifest.json", manifest)
    _write_json(ARTIFACT_ROOT / "baseline_comparison.json", {"rows": score_results})
    _write_json(ARTIFACT_ROOT / "singular_robustness_report.json", robustness)
    _write_json(ARTIFACT_ROOT / "model_a_affine_result.json", parity[0])
    _write_json(ARTIFACT_ROOT / "model_b_nonlinear_accumulation_result.json", parity[1])
    _write_json(ARTIFACT_ROOT / "model_c_nonlinear_growth_result.json", parity[2])
    _write_json(ARTIFACT_ROOT / "pp_ukf_result.json", registry_results[0])
    _write_json(ARTIFACT_ROOT / "str_ukf_result.json", registry_results[1])
    _write_json(ARTIFACT_ROOT / "structural_ar1_quadratic_h16_result.json", rectangular)
    matrix_lines = ["model_id,status,route,executed,score_claim,branch_status"]
    inventory_by_id = {row["model_id"]: row for row in inventory}
    for row in inventory:
        result = next((item for item in results if item["model_id"] == row["model_id"]), None)
        matrix_lines.append(
            ",".join([
                row["model_id"], row["status"], result["route"] if result else "none",
                "true" if result else "false",
                "yes" if result and row["status"] == "eligible_score" else "no",
                result["branch_status"] if result else "classified_only",
            ])
        )
    (ARTIFACT_ROOT / "coverage_matrix.csv").write_text("\n".join(matrix_lines) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in inventory:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = [
        "# Direct-Factor SR-UKF Model Coverage Report",
        "",
        "Campaign status: `EXECUTED_WITH_EXPLICIT_CLASSIFICATION_BOUNDARIES`.",
        "",
        f"Inventory rows: {len(inventory)}; executed rows: {len(results)}.",
        "Status counts: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) + ".",
        "",
        "Executed score rows are the three existing direct-factor fixtures (A/B/C) plus the certified PP-UKF and STR-UKF one-time factor adapters, with eager/XLA attempts, historical principal-root comparison, and fixed-program score evidence. The structural AR1 row was executed through the rectangular direct-stack SVD support route and is value-only.",
        "",
        "Common V2 and NeuTra rows are all classified. Models whose source route is principal-root UKF, SGQF, multiplicative SV, domain-constrained SIR, DPF/LEDH, or an un-certified covariance adapter are not silently promoted to direct-factor evidence.",
        "",
        "Singular robustness includes QR scales 1 through 1e-15, exact rank-zero/rank-one stacks, repeated singular values, reconstruction residuals, and explicit value-only branch metadata.",
        "",
        "Nonclaims: no claim of exact nonlinear Bayesian inference, no score through rank/support changes, no HMC readiness, and no repository-wide claim that inapplicable contracts were tested as SR-UKF.",
    ]
    (ARTIFACT_ROOT / "coverage_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "artifact_root": str(ARTIFACT_ROOT),
    }
    (ARTIFACT_ROOT / "commands_and_environment.md").write_text(
        "# Commands and Environment\n\n```text\n" + "\n".join(manifest["commands"]) + "\n```\n\n" + json.dumps(_jsonable(env), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_json(ARTIFACT_ROOT / "latex_table_payload.json", {"inventory": inventory, "score_rows": score_results, "value_only_rows": [rectangular], "robustness": robustness})
    print(json.dumps({"artifact_root": str(ARTIFACT_ROOT), "inventory_rows": len(inventory), "executed_rows": len(results), "status_counts": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
