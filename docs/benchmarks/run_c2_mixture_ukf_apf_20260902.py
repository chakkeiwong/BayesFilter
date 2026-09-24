"""Execute the bounded C2 mixture-UKF/APF campaign phases.

Phase 0 is a CPU reference/XLA check.  It freezes the linear-Gaussian fixture,
exercises the repository-owned batched UKF and K=1 APF endpoints, and writes a
fresh manifest and result.  The smoke phase is a bounded CPU/reference run at
the frozen C2 horizon.  Phase 2 first runs a one-branch GPU entry pilot and then
the declared twelve-branch N=8192 serious expansion.  Phase 3 reuses the exact
target and paired contract to compare fixed-topology K=2/K=4 mixtures after a
disjoint offset calibration; larger rows remain a separate, explicitly gated
step.

This runner is diagnostic infrastructure.  It does not claim C2 posterior
correctness, low-variance importance sampling, or production readiness.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Mapping, Sequence

# This TensorFlow build eagerly creates logical GPUs when the growth override
# is present during import, which prevents the repository helper from setting
# and verifying per-device growth. Defer the inherited value through import;
# restore it before the helper is called so the realized policy is explicit.
_DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH = os.environ.pop(
    "TF_FORCE_GPU_ALLOW_GROWTH", None
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_mixture_ukf_lgssm_phase0_v1.json"
PLAN_PATH = ROOT / "docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md"
MODULE_PATH = ROOT / "bayesfilter/highdim/c2_mixture_ukf_apf_tf.py"
sys.path.insert(0, str(ROOT))

import tensorflow as tf

# Importing ``bayesfilter.highdim`` executes a large package initializer with
# module-level TensorFlow constants.  On this build those constants can create
# logical GPUs, so all algorithm imports are deferred until after the Phase 2
# memory policy has been installed.  CPU phases call this loader explicitly.
_ALGORITHM_MODULES_LOADED = False


def _load_algorithm_modules() -> None:
    global _ALGORITHM_MODULES_LOADED
    global BatchedUKFConfig, complete_gaussian_mixture_log_density
    global compile_k1_apf_proposal, gaussian_log_density, make_batched_ukf_kernel
    global C2_UKF_ROUTE_ID, compile_c2_per_ancestor_ukf_apf_k1
    global C2_UKF_MIXTURE_ROUTE_ID, compile_c2_per_ancestor_ukf_apf_mixture
    global C2_UKF_DEFENSIVE_ROUTE_ID, compile_c2_per_ancestor_ukf_apf_defensive_mixture
    global C2StochasticVolatilityFrozenAPFModel
    global compile_c2_bootstrap_proposal_branch
    global compile_c2_transformed_student_proposal_branch
    global prepare_frozen_proposal_apf_program
    if _ALGORITHM_MODULES_LOADED:
        return
    from bayesfilter.highdim.c2_mixture_ukf_apf_tf import (
        BatchedUKFConfig as _BatchedUKFConfig,
        complete_gaussian_mixture_log_density as _complete_gaussian_mixture_log_density,
        compile_k1_apf_proposal as _compile_k1_apf_proposal,
        gaussian_log_density as _gaussian_log_density,
        make_batched_ukf_kernel as _make_batched_ukf_kernel,
    )
    from bayesfilter.highdim.c2_mixture_ukf_apf_c2_adapter import (
        DEFENSIVE_MIXTURE_ROUTE_ID as _c2_ukf_defensive_route_id,
        MIXTURE_ROUTE_ID as _c2_ukf_mixture_route_id,
        ROUTE_ID as _c2_ukf_route_id,
        compile_c2_per_ancestor_ukf_apf_defensive_mixture as _compile_c2_ukf_defensive_mixture,
        compile_c2_per_ancestor_ukf_apf_k1 as _compile_c2_per_ancestor_ukf_apf_k1,
        compile_c2_per_ancestor_ukf_apf_mixture as _compile_c2_per_ancestor_ukf_apf_mixture,
    )
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
        C2StochasticVolatilityFrozenAPFModel as _c2_model,
        compile_c2_bootstrap_proposal_branch as _compile_bootstrap,
        compile_c2_transformed_student_proposal_branch as _compile_student,
    )
    from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
        prepare_frozen_proposal_apf_program as _prepare_program,
    )
    BatchedUKFConfig = _BatchedUKFConfig
    complete_gaussian_mixture_log_density = _complete_gaussian_mixture_log_density
    compile_k1_apf_proposal = _compile_k1_apf_proposal
    gaussian_log_density = _gaussian_log_density
    make_batched_ukf_kernel = _make_batched_ukf_kernel
    C2_UKF_ROUTE_ID = _c2_ukf_route_id
    compile_c2_per_ancestor_ukf_apf_k1 = _compile_c2_per_ancestor_ukf_apf_k1
    C2_UKF_MIXTURE_ROUTE_ID = _c2_ukf_mixture_route_id
    compile_c2_per_ancestor_ukf_apf_mixture = _compile_c2_per_ancestor_ukf_apf_mixture
    C2_UKF_DEFENSIVE_ROUTE_ID = _c2_ukf_defensive_route_id
    compile_c2_per_ancestor_ukf_apf_defensive_mixture = _compile_c2_ukf_defensive_mixture
    C2StochasticVolatilityFrozenAPFModel = _c2_model
    compile_c2_bootstrap_proposal_branch = _compile_bootstrap
    compile_c2_transformed_student_proposal_branch = _compile_student
    prepare_frozen_proposal_apf_program = _prepare_program
    _ALGORITHM_MODULES_LOADED = True


DTYPE = tf.float64
PHASE_ID = "c2_mixture_ukf_apf_phase0_preflight_v1"
RESULT_SCHEMA = "c2_mixture_ukf_apf_phase0_result_v1"
C2_FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
C2_ADAPTER_PATH = ROOT / "bayesfilter/highdim/c2_mixture_ukf_apf_c2_adapter.py"
C2_MODEL_PATH = ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py"
C2_EXACT_PATH = ROOT / "bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py"
PHASE4_PLAN_PATH = ROOT / "docs/plans/c2-mixture-ukf-apf-phase4-execution-20260903.md"
PHASE1_ID = "c2_mixture_ukf_apf_phase1_smoke_v1"
PHASE1_RESULT_SCHEMA = "c2_mixture_ukf_apf_phase1_result_v1"
SEED = (20260903, 17)
# Absolute tolerances remain useful for small, independently recomposed
# quantities.  The APF identity below is a subtraction-heavy floating-point
# rearrangement, so its validity check uses a scale-aware backward-error bound.
ABS_TOL = 2.0e-10
FLOAT64_EPSILON = sys.float_info.epsilon
APF_IDENTITY_BACKWARD_FACTOR = 32.0
APF_IDENTITY_BACKWARD_BOUND = APF_IDENTITY_BACKWARD_FACTOR * FLOAT64_EPSILON


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("phase0", "smoke", "phase2-entry", "serious", "phase3", "phase4", "phase4-repair"), required=True
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rows", default="256,1024")
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument(
        "--jit-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use XLA for the CPU reference kernel (default: true).",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _workspace_manifest() -> Mapping[str, object]:
    status = _git("status", "--porcelain=v1")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status": status,
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite artifact value: {value!r}")
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_fixture() -> Mapping[str, object]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "c2_mixture_ukf_lgssm_phase0_v1":
        raise ValueError("unexpected Phase 0 fixture schema")
    return payload


def _tensor(value: object) -> tf.Tensor:
    return tf.convert_to_tensor(value, dtype=DTYPE)


def _fixture_inputs(fixture: Mapping[str, object]) -> tuple[tf.Tensor, ...]:
    means = _tensor(fixture["ancestor_means"])
    count = int(means.shape[0])
    process = tf.broadcast_to(_tensor(fixture["process_covariance"]), [count, 2, 2])
    observation_covariance = tf.broadcast_to(
        _tensor(fixture["observation_covariance"]), [count, 2, 2]
    )
    return (
        means,
        _tensor(fixture["ancestor_covariances"]),
        process,
        _tensor(fixture["observation"]),
        observation_covariance,
    )


def _linear_kernel(*, jit_compile: bool, fixture: Mapping[str, object]):
    transition_matrix = _tensor(fixture["transition_matrix"])
    transition_offset = _tensor(fixture["transition_offset"])
    observation_matrix = _tensor(fixture["observation_matrix"])
    observation_offset = _tensor(fixture["observation_offset"])
    inputs = _fixture_inputs(fixture)
    batch_size = int(inputs[0].shape[0])

    def transition_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ij,bpj->bpi", transition_matrix, points) + transition_offset

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ij,bpj->bpi", observation_matrix, points) + observation_offset

    return make_batched_ukf_kernel(
        batch_size=batch_size,
        state_dim=2,
        observation_dim=2,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        config=BatchedUKFConfig(),
        jit_compile=jit_compile,
    )


def _independent_kalman(fixture: Mapping[str, object]) -> Mapping[str, tf.Tensor]:
    means, covariances, process, observation, observation_covariance = _fixture_inputs(fixture)
    transition_matrix = _tensor(fixture["transition_matrix"])
    transition_offset = _tensor(fixture["transition_offset"])
    observation_matrix = _tensor(fixture["observation_matrix"])
    observation_offset = _tensor(fixture["observation_offset"])
    predicted_mean = tf.einsum("ij,bj->bi", transition_matrix, means) + transition_offset
    predicted_covariance = tf.einsum(
        "ij,bjk,lk->bil", transition_matrix, covariances, transition_matrix
    ) + process
    observation_mean = tf.einsum("ij,bj->bi", observation_matrix, predicted_mean) + observation_offset
    innovation_covariance = tf.einsum(
        "ij,bjk,lk->bil", observation_matrix, predicted_covariance, observation_matrix
    ) + observation_covariance
    cross_covariance = tf.einsum(
        "bij,lj->bil", predicted_covariance, observation_matrix
    )
    innovation = observation[None, :] - observation_mean
    gain = tf.transpose(
        tf.linalg.solve(
            innovation_covariance,
            tf.transpose(cross_covariance, [0, 2, 1]),
        ),
        [0, 2, 1],
    )
    posterior_mean = predicted_mean + tf.einsum("bio,bo->bi", gain, innovation)
    raw_posterior = predicted_covariance - tf.matmul(
        tf.matmul(gain, innovation_covariance), gain, transpose_b=True
    )
    posterior_covariance = 0.5 * (raw_posterior + tf.transpose(raw_posterior, [0, 2, 1]))
    chol = tf.linalg.cholesky(innovation_covariance)
    solved = tf.linalg.cholesky_solve(chol, innovation[:, :, None])[:, :, 0]
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=1)
    log_likelihood = -0.5 * (
        2.0 * math.log(2.0 * math.pi)
        + logdet
        + tf.reduce_sum(innovation * solved, axis=1)
    )
    return {
        "predicted_mean": predicted_mean,
        "predicted_covariance": predicted_covariance,
        "predicted_observation_mean": observation_mean,
        "innovation_covariance": innovation_covariance,
        "cross_covariance": cross_covariance,
        "posterior_mean": posterior_mean,
        "posterior_covariance": posterior_covariance,
        "innovation_log_likelihood": log_likelihood,
    }


def _max_abs(value: tf.Tensor) -> float:
    return float(tf.reduce_max(tf.abs(tf.convert_to_tensor(value, DTYPE))).numpy())


def _run_checks(fixture: Mapping[str, object], *, jit_compile: bool) -> Mapping[str, object]:
    inputs = _fixture_inputs(fixture)
    kernel = _linear_kernel(jit_compile=jit_compile, fixture=fixture)
    result = kernel(*inputs)
    expected = _independent_kalman(fixture)
    parity_errors = {
        name: _max_abs(result[name] - value) for name, value in expected.items()
    }
    proposal = compile_k1_apf_proposal(
        kernel,
        prior_means=inputs[0],
        prior_covariances=inputs[1],
        process_covariances=inputs[2],
        observation=inputs[3],
        observation_covariances=inputs[4],
        log_parent_weights=_tensor(fixture["log_parent_weights"]),
        seed=SEED,
    )
    ancestor = proposal["ancestor_indices"]
    parent = tf.gather(inputs[0], ancestor)
    transition_matrix = _tensor(fixture["transition_matrix"])
    transition_offset = _tensor(fixture["transition_offset"])
    transition_mean = tf.einsum("ij,bj->bi", transition_matrix, parent) + transition_offset
    transition_chol = tf.broadcast_to(
        tf.linalg.cholesky(_tensor(fixture["process_covariance"])),
        [int(ancestor.shape[0]), 2, 2],
    )
    transition_log_density = gaussian_log_density(
        proposal["samples"], transition_mean, transition_chol
    )
    observation_matrix = _tensor(fixture["observation_matrix"])
    observation_offset = _tensor(fixture["observation_offset"])
    observation_mean = tf.einsum(
        "ij,bj->bi", observation_matrix, proposal["samples"]
    ) + observation_offset
    observation_chol = tf.broadcast_to(
        tf.linalg.cholesky(_tensor(fixture["observation_covariance"])),
        [int(ancestor.shape[0]), 2, 2],
    )
    observation_log_density = gaussian_log_density(
        tf.broadcast_to(inputs[3][None, :], [int(ancestor.shape[0]), 2]),
        observation_mean,
        observation_chol,
    )
    log_target = (
        tf.gather(_tensor(fixture["log_parent_weights"]), ancestor)
        + transition_log_density
        + observation_log_density
    )
    log_weight = log_target - proposal["selected_log_q"] - tf.gather(
        proposal["log_ancestor_probabilities"], ancestor
    )
    apf_identity_error = _max_abs(
        tf.gather(proposal["log_ancestor_probabilities"], ancestor)
        + proposal["selected_log_q"]
        + log_weight
        - log_target
    )

    base_mean = result["posterior_mean"]
    base_chol = result["posterior_cholesky"]
    points = base_mean + _tensor([[0.4, -0.25], [-0.2, 0.3], [0.1, 0.05], [-0.3, -0.1]])
    component_offsets = tf.broadcast_to(
        _tensor([[[0.0, 0.0], [0.7, -0.2]]]), [4, 2, 2]
    )
    component_means = base_mean[:, None, :] + component_offsets
    component_chol = tf.broadcast_to(base_chol[:, None, :, :], [4, 2, 2, 2])
    component_weights = tf.broadcast_to(_tensor([[0.35, 0.65]]), [4, 2])
    complete = complete_gaussian_mixture_log_density(
        points, component_means, component_chol, component_weights
    )
    component_logs = tf.stack(
        [
            gaussian_log_density(points, component_means[:, 0, :], base_chol),
            gaussian_log_density(points, component_means[:, 1, :], base_chol),
        ],
        axis=1,
    )
    independent_complete = tf.reduce_logsumexp(
        component_logs + tf.math.log(component_weights), axis=1
    )
    permuted = complete_gaussian_mixture_log_density(
        points,
        tf.gather(component_means, [1, 0], axis=1),
        tf.gather(component_chol, [1, 0], axis=1),
        tf.gather(component_weights, [1, 0], axis=1),
    )

    raw_kernel = make_batched_ukf_kernel(
        batch_size=4,
        state_dim=2,
        observation_dim=1,
        transition_fn=lambda value: value,
        observation_fn=lambda value: tf.zeros([4, 5, 1], DTYPE),
        config=BatchedUKFConfig(),
        jit_compile=jit_compile,
    )
    raw = raw_kernel(
        inputs[0], inputs[1], inputs[2], tf.constant([1.2], DTYPE),
        tf.broadcast_to(tf.constant([[0.25]], DTYPE), [4, 1, 1]),
    )
    transformed_kernel = make_batched_ukf_kernel(
        batch_size=4,
        state_dim=2,
        observation_dim=1,
        transition_fn=lambda value: value,
        observation_fn=lambda value: tf.exp(0.5 * value[..., :1]),
        config=BatchedUKFConfig(),
        jit_compile=jit_compile,
    )
    transformed = transformed_kernel(
        inputs[0], inputs[1], inputs[2], tf.constant([1.2], DTYPE),
        tf.broadcast_to(tf.constant([[0.25]], DTYPE), [4, 1, 1]),
    )

    invalid_covariance = tf.tensor_scatter_nd_update(
        inputs[1], tf.constant([[0, 0, 0]], tf.int32), tf.constant([-0.5], DTYPE)
    )
    invalid = kernel(
        inputs[0], invalid_covariance, inputs[2], inputs[3], inputs[4]
    )
    parity_pass = all(error <= ABS_TOL for error in parity_errors.values())
    mixture_permutation_error = _max_abs(complete - permuted)
    mixture_independent_error = _max_abs(complete - independent_complete)
    checks = {
        "linear_kalman_parity": parity_pass,
        "apf_w_over_a_identity": apf_identity_error <= ABS_TOL,
        "ancestor_probability_normalization": abs(
            float(tf.reduce_logsumexp(proposal["log_ancestor_probabilities"]).numpy())
        )
        <= ABS_TOL,
        "complete_mixture_independent_recomposition": mixture_independent_error <= ABS_TOL,
        "complete_mixture_permutation_invariance": mixture_permutation_error <= ABS_TOL,
        "raw_signed_observation_zero_gain": _max_abs(raw["cross_covariance"]) <= ABS_TOL,
        "transformed_observation_has_cross_covariance": _max_abs(
            transformed["cross_covariance"]
        )
        > 1.0e-4,
        "invalid_covariance_reported": not bool(invalid["finite"].numpy())
        and not bool(invalid["valid_rows"][0].numpy()),
        "per_ancestor_observation_response": _max_abs(
            result["posterior_mean"] - result["predicted_mean"]
        )
        > 1.0e-4,
    }
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "parity_errors": parity_errors,
        "apf_identity_max_abs_error": apf_identity_error,
        "mixture_independent_max_abs_error": mixture_independent_error,
        "mixture_permutation_max_abs_error": mixture_permutation_error,
        "raw_zero_gain_max_abs": _max_abs(raw["cross_covariance"]),
        "transformed_cross_covariance_max_abs": _max_abs(
            transformed["cross_covariance"]
        ),
        "per_ancestor_update_max_abs": _max_abs(
            result["posterior_mean"] - result["predicted_mean"]
        ),
        "ancestor_log_probability_min": float(
            tf.reduce_min(proposal["log_ancestor_probabilities"]).numpy()
        ),
        "ancestor_log_probability_max": float(
            tf.reduce_max(proposal["log_ancestor_probabilities"]).numpy()
        ),
        "ukf_valid_rows": result["valid_rows"],
        "ukf_finite": result["finite"],
        "input_signature": [
            tuple(spec.shape.as_list()) for spec in kernel.input_signature or ()
        ],
        "jit_compile_requested": bool(jit_compile),
    }


def _result_markdown(payload: Mapping[str, object]) -> str:
    checks = payload["checks"]
    all_checks_pass = checks["all_checks_pass"]
    lines = [
        "# C2 Mixture-UKF/APF Phase 0 Result",
        "",
        f"Phase: `{PHASE_ID}`",
        f"Status: `{payload['status']}`",
        "",
        "## Decision",
        "",
        "| Item | Status |",
        "| --- | --- |",
        f"| All Phase 0 checks | `{all_checks_pass}` |",
        f"| Continuation | `{payload['continuation']}` |",
        "| Scientific claim | diagnostic only; no C2 posterior or production claim |",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, value in checks.items():
        lines.append(f"| `{name}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Phase-close record",
            "",
            "| Field | Entry |",
            "| --- | --- |",
            f"| Failure class | `{payload['failure_class']}` |",
            f"| Repair | `{payload['repair']}` |",
            f"| Next-phase refresh | `{payload['next_phase_refresh']}` |",
            f"| Budget remaining | `{payload['budget_remaining']}` |",
            f"| MathDevMCP/Lean | `{payload['formal_tool_status']}` |",
            "",
            "The linear fixture is an independent mechanics check. The APF "
            "w/a identity is checked pointwise on the realized draw. "
            "Complete-mixture permutation and raw/transformed observation "
            "checks are negative/diagnostic controls; they do not establish "
            "low variance or filtering correctness.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_c2_fixture() -> Mapping[str, object]:
    payload = json.loads(C2_FIXTURE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_id") != "bayesfilter.c2_sv_frozen_fixture.v1":
        raise ValueError("unexpected C2 fixture schema")
    dimension = int(payload.get("state_dimension", 0))
    horizon = int(payload.get("horizon", 0))
    observations = payload.get("observations")
    if dimension < 1 or horizon < 2 or not isinstance(observations, list):
        raise ValueError("C2 fixture has invalid dimensions")
    if len(observations) != horizon:
        raise ValueError("C2 fixture horizon does not match observations")
    return payload


def _parse_rows(value: str) -> tuple[int, ...]:
    rows: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        count = int(token)
        if count < 2:
            raise ValueError("particle rows must be at least two")
        rows.append(count)
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("--rows must contain one or more distinct positive counts")
    return tuple(rows)


def _c2_model_and_inputs(
    fixture: Mapping[str, object],
) -> tuple[C2StochasticVolatilityFrozenAPFModel, tf.Tensor, tf.Tensor]:
    dimension = int(fixture["state_dimension"])
    theta = tf.constant(
        [float(fixture["gamma"]), math.log(float(fixture["beta"]))], DTYPE
    )
    transition = _tensor(fixture["transition_matrix"])
    coupling = transition - theta[0] * tf.eye(dimension, dtype=DTYPE)
    model = C2StochasticVolatilityFrozenAPFModel(
        coupling_matrix=coupling, sigma=float(fixture["sigma"])
    )
    observations = tf.constant(fixture["observations"], DTYPE)
    observations = tf.ensure_shape(observations, [int(fixture["horizon"]), dimension])
    return model, theta, observations


def _finite_tensor(value: object) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())


def _tensor_float(value: object) -> float:
    tensor = tf.convert_to_tensor(value, DTYPE)
    if not _finite_tensor(tensor):
        raise ValueError("attempted to serialize a non-finite tensor")
    return float(tensor.numpy())


def _tensor_vector(value: object) -> list[float]:
    tensor = tf.reshape(tf.convert_to_tensor(value, DTYPE), [-1])
    if not _finite_tensor(tensor):
        raise ValueError("attempted to serialize a non-finite tensor vector")
    return [float(item) for item in tensor.numpy().tolist()]


def _scaled_backward_error(
    residual: object, terms: Sequence[object]
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return absolute residual, scale, and normalized backward error."""

    residual_tensor = tf.convert_to_tensor(residual, DTYPE)
    term_tensors = [tf.convert_to_tensor(term, DTYPE) for term in terms]
    if not term_tensors:
        raise ValueError("at least one scale term is required")
    scale = tf.maximum(
        tf.constant(1.0, DTYPE),
        tf.add_n([tf.abs(term) for term in term_tensors]),
    )
    absolute = tf.abs(residual_tensor)
    return absolute, scale, absolute / scale


def _central_difference(
    program, theta: tf.Tensor, parameter_index: int, step: float = 1.0e-5
) -> tf.Tensor:
    direction = tf.one_hot(parameter_index, int(theta.shape[0]), dtype=DTYPE)
    plus = program.evaluate(theta + tf.constant(step, DTYPE) * direction)
    minus = program.evaluate(theta - tf.constant(step, DTYPE) * direction)
    return (plus["log_likelihood"] - minus["log_likelihood"]) / tf.constant(
        2.0 * step, DTYPE
    )


def _apf_identity_and_law_diagnostics(
    model: C2StochasticVolatilityFrozenAPFModel,
    branch,
    theta: tf.Tensor,
    expected_final_log_weights: tf.Tensor | None = None,
) -> Mapping[str, object]:
    """Recompute the conditional APF law and report scale-aware roundoff.

    The expression ``a + q + (target - a - q) - target`` is an algebraic
    identity, but its absolute floating-point residual grows with the size of
    the log terms.  A fixed absolute threshold therefore rejects valid
    realizations with large log-weight spread.  We retain the absolute
    residual for observability and gate it with a normalized backward error.
    """

    particle_count = branch.particle_count
    initial_target = model.initial_log_density(theta, branch.states[0]) + model.observation_log_density(
        theta, branch.states[0], branch.observations[0], 0
    )
    initial_log_unnormalized = (
        branch.initial_log_base_mass
        + initial_target
        - branch.initial_log_proposal_density
    )
    initial_increment = tf.reduce_logsumexp(initial_log_unnormalized)
    log_weights = initial_log_unnormalized - initial_increment
    max_identity_error = tf.constant(0.0, DTYPE)
    max_identity_scale = tf.constant(1.0, DTYPE)
    max_identity_backward_error = tf.constant(0.0, DTYPE)
    max_auxiliary_normalization_error = tf.abs(
        tf.reduce_logsumexp(branch.auxiliary_log_probabilities, axis=1)
    ) if branch.time_steps > 1 else tf.zeros([0], DTYPE)
    max_base_normalization_error = tf.abs(
        tf.reduce_logsumexp(branch.transition_log_base_mass, axis=1)
    ) if branch.time_steps > 1 else tf.zeros([0], DTYPE)

    for time_index in range(1, branch.time_steps):
        ancestors = branch.ancestors[time_index - 1]
        previous = tf.gather(branch.states[time_index - 1], ancestors)
        current = branch.states[time_index]
        selected_previous_log_weights = tf.gather(log_weights, ancestors)
        selected_log_a = tf.gather(
            branch.auxiliary_log_probabilities[time_index - 1], ancestors
        )
        transition_log_density = model.transition_log_density(
            theta, previous, current, time_index
        )
        observation_log_density = model.observation_log_density(
            theta, current, branch.observations[time_index], time_index
        )
        selected_log_q = branch.transition_log_proposal_density[time_index - 1]
        exact_conditional_target = (
            selected_previous_log_weights
            + transition_log_density
            + observation_log_density
        )
        corrected_log_weight = exact_conditional_target - selected_log_a - selected_log_q
        identity_residual = (
            selected_log_a
            + selected_log_q
            + corrected_log_weight
            - exact_conditional_target
        )
        identity_absolute, identity_scale, identity_relative = _scaled_backward_error(
            identity_residual,
            (
                selected_log_a,
                selected_log_q,
                corrected_log_weight,
                exact_conditional_target,
            ),
        )
        identity_error = tf.reduce_max(identity_absolute)
        identity_backward_error = tf.reduce_max(
            identity_relative
        )
        max_identity_error = tf.maximum(max_identity_error, identity_error)
        max_identity_scale = tf.maximum(max_identity_scale, tf.reduce_max(identity_scale))
        max_identity_backward_error = tf.maximum(
            max_identity_backward_error, identity_backward_error
        )
        log_unnormalized = (
            branch.transition_log_base_mass[time_index - 1] + corrected_log_weight
        )
        increment = tf.reduce_logsumexp(log_unnormalized)
        log_weights = log_unnormalized - increment

    final_weight_parity_error = tf.constant(0.0, DTYPE)
    final_weight_parity_scale = tf.constant(1.0, DTYPE)
    final_weight_parity_backward_error = tf.constant(0.0, DTYPE)
    if expected_final_log_weights is not None:
        expected = tf.ensure_shape(
            tf.convert_to_tensor(expected_final_log_weights, DTYPE), [particle_count]
        )
        final_residual = log_weights - expected
        final_absolute, final_scale, final_relative = _scaled_backward_error(
            final_residual, (log_weights, expected)
        )
        final_weight_parity_error = tf.reduce_max(final_absolute)
        final_weight_parity_scale = tf.reduce_max(final_scale)
        final_weight_parity_backward_error = tf.reduce_max(
            final_relative
        )
    finite = (
        _finite_tensor(log_weights)
        and _finite_tensor(max_auxiliary_normalization_error)
        and _finite_tensor(max_identity_backward_error)
        and _finite_tensor(final_weight_parity_backward_error)
    )
    if branch.time_steps > 1:
        finite = finite and _finite_tensor(max_base_normalization_error)
    return {
        "max_identity_abs_error": _tensor_float(max_identity_error),
        "max_identity_scale": _tensor_float(max_identity_scale),
        "max_identity_backward_error": _tensor_float(max_identity_backward_error),
        "identity_backward_error_bound": APF_IDENTITY_BACKWARD_BOUND,
        "max_auxiliary_normalization_abs_error": _tensor_float(
            tf.reduce_max(max_auxiliary_normalization_error)
            if branch.time_steps > 1
            else tf.constant(0.0, DTYPE)
        ),
        "max_base_mass_normalization_abs_error": _tensor_float(
            tf.reduce_max(max_base_normalization_error)
            if branch.time_steps > 1
            else tf.constant(0.0, DTYPE)
        ),
        "finite": bool(finite),
        "final_log_weight_parity_max_abs_error": _tensor_float(
            final_weight_parity_error
        ),
        "final_log_weight_parity_scale": _tensor_float(final_weight_parity_scale),
        "final_log_weight_parity_backward_error": _tensor_float(
            final_weight_parity_backward_error
        ),
        "final_log_weight_parity_backward_error_bound": APF_IDENTITY_BACKWARD_BOUND,
        "final_log_weights": _tensor_vector(log_weights),
        "particle_count": particle_count,
    }


def _branch_proposal_diagnostics(compilation) -> Mapping[str, object]:
    branch = compilation.branch
    auxiliary_error = (
        tf.reduce_max(tf.abs(tf.reduce_logsumexp(branch.auxiliary_log_probabilities, axis=1)))
        if branch.time_steps > 1
        else tf.constant(0.0, DTYPE)
    )
    base_error = (
        tf.reduce_max(tf.abs(tf.reduce_logsumexp(branch.transition_log_base_mass, axis=1)))
        if branch.time_steps > 1
        else tf.constant(0.0, DTYPE)
    )
    transition_q = branch.transition_log_proposal_density
    summary: dict[str, object] = {
        "auxiliary_normalization_max_abs_error": _tensor_float(auxiliary_error),
        "base_mass_normalization_max_abs_error": _tensor_float(base_error),
        "transition_log_q_min": _tensor_float(tf.reduce_min(transition_q))
        if branch.time_steps > 1
        else 0.0,
        "transition_log_q_max": _tensor_float(tf.reduce_max(transition_q))
        if branch.time_steps > 1
        else 0.0,
        "proposal_diagnostic_rows": len(compilation.proposal_diagnostics),
    }
    if compilation.manifest.get("route_id") in (
        C2_UKF_ROUTE_ID,
        C2_UKF_MIXTURE_ROUTE_ID,
        C2_UKF_DEFENSIVE_ROUTE_ID,
    ):
        means = tf.stack(
            [tf.convert_to_tensor(row["posterior_mean"], DTYPE) for row in compilation.proposal_diagnostics]
        )
        lookahead = tf.stack(
            [tf.convert_to_tensor(row["lookahead_log_likelihood"], DTYPE) for row in compilation.proposal_diagnostics]
        )
        innovation_quadratic = tf.stack(
            [
                tf.convert_to_tensor(row["innovation_quadratic"], DTYPE)
                for row in compilation.proposal_diagnostics
                if "innovation_quadratic" in row
            ]
        ) if any("innovation_quadratic" in row for row in compilation.proposal_diagnostics) else tf.zeros([0, branch.particle_count], DTYPE)
        recomposition = tf.stack(
            [tf.convert_to_tensor(row["proposal_density_recomposition_max_abs"], DTYPE) for row in compilation.proposal_diagnostics]
        )
        posterior_minimum = tf.stack(
            [tf.convert_to_tensor(row["posterior_min_eigenvalue"], DTYPE) for row in compilation.proposal_diagnostics]
        )
        summary.update(
            {
                "per_ancestor_posterior_mean_spread_max": _tensor_float(
                    tf.reduce_max(tf.reduce_max(means, axis=1) - tf.reduce_min(means, axis=1))
                ),
                "lookahead_spread_max": _tensor_float(
                    tf.reduce_max(tf.reduce_max(lookahead, axis=1) - tf.reduce_min(lookahead, axis=1))
                ),
                "proposal_density_recomposition_max_abs": _tensor_float(tf.reduce_max(recomposition)),
                "posterior_min_eigenvalue": _tensor_float(tf.reduce_min(posterior_minimum)),
                "innovation_quadratic_mean_by_time": _tensor_vector(
                    tf.reduce_mean(innovation_quadratic, axis=1)
                )
                if int(innovation_quadratic.shape[0]) > 0
                else [],
                "innovation_quadratic_max_by_time": _tensor_vector(
                    tf.reduce_max(innovation_quadratic, axis=1)
                )
                if int(innovation_quadratic.shape[0]) > 0
                else [],
            }
        )
        if compilation.manifest.get("route_id") == C2_UKF_MIXTURE_ROUTE_ID:
            component_minimum = tf.stack(
                [
                    tf.convert_to_tensor(
                        row["component_minimum_eigenvalue"], DTYPE
                    )
                    for row in compilation.proposal_diagnostics
                ]
            )
            moment_error = tf.stack(
                [
                    tf.convert_to_tensor(
                        row["moment_recomposition_max_abs"], DTYPE
                    )
                    for row in compilation.proposal_diagnostics
                ]
            )
            label_permutation = tf.stack(
                [
                    tf.convert_to_tensor(
                        row["label_permutation_max_abs"], DTYPE
                    )
                    for row in compilation.proposal_diagnostics
                ]
            )
            summary.update(
                {
                    "component_minimum_eigenvalue": _tensor_float(
                        tf.reduce_min(component_minimum)
                    ),
                    "moment_recomposition_max_abs": _tensor_float(
                        tf.reduce_max(moment_error)
                    ),
                    "label_permutation_max_abs": _tensor_float(
                        tf.reduce_max(label_permutation)
                    ),
                    "component_count": int(compilation.manifest["component_count"]),
                    "offset": float(compilation.manifest["offset"]),
                }
            )
        if compilation.manifest.get("route_id") == C2_UKF_DEFENSIVE_ROUTE_ID:
            component_minimum = tf.stack(
                [
                    tf.convert_to_tensor(
                        row["component_minimum_eigenvalue"], DTYPE
                    )
                    for row in compilation.proposal_diagnostics
                ]
            )
            moment_error = tf.stack(
                [
                    tf.convert_to_tensor(
                        row["local_moment_recomposition_max_abs"], DTYPE
                    )
                    for row in compilation.proposal_diagnostics
                ]
            )
            label_permutation = tf.stack(
                [
                    tf.convert_to_tensor(
                        row["label_permutation_max_abs"], DTYPE
                    )
                    for row in compilation.proposal_diagnostics
                ]
            )
            epsilon_spread = tf.stack(
                [tf.convert_to_tensor(row["epsilon_spread"], DTYPE) for row in compilation.proposal_diagnostics]
            )
            summary.update(
                {
                    "component_minimum_eigenvalue": _tensor_float(
                        tf.reduce_min(component_minimum)
                    ),
                    "moment_recomposition_max_abs": _tensor_float(
                        tf.reduce_max(moment_error)
                    ),
                    "label_permutation_max_abs": _tensor_float(
                        tf.reduce_max(label_permutation)
                    ),
                    "epsilon_spread_max": _tensor_float(tf.reduce_max(epsilon_spread)),
                    "local_component_count": int(
                        compilation.manifest["local_component_count"]
                    ),
                    "nu": float(compilation.manifest["nu"]),
                    "epsilon_min": float(compilation.manifest["epsilon_min"]),
                    "epsilon_max": float(compilation.manifest["epsilon_max"]),
                }
            )
    finite = _finite_tensor(transition_q) and _finite_tensor(auxiliary_error) and _finite_tensor(base_error)
    summary["finite"] = bool(finite)
    return summary


def _evaluate_c2_candidate(
    *,
    label: str,
    compilation,
    model: C2StochasticVolatilityFrozenAPFModel,
    theta: tf.Tensor,
    check_xla: bool,
) -> Mapping[str, object]:
    branch = compilation.branch
    program = prepare_frozen_proposal_apf_program(model, branch)
    result = program.evaluate(theta)
    finite = bool(result["finite"].numpy()) and _finite_tensor(result["log_likelihood"]) and _finite_tensor(result["score"])
    score_error = float("inf")
    finite_difference: list[float] = []
    if finite:
        finite_difference_tensor = tf.stack(
            [_central_difference(program, theta, index) for index in range(int(theta.shape[0]))]
        )
        finite_difference = _tensor_vector(finite_difference_tensor)
        score_error = _tensor_float(tf.reduce_max(tf.abs(result["score"] - finite_difference_tensor)))

    nonjit_value_error = float("inf")
    nonjit_score_error = float("inf")
    nonjit_error_message = ""
    try:
        nonjit_result = program.compiled(jit_compile=False)(theta)
        nonjit_value_error = _tensor_float(tf.abs(result["log_likelihood"] - nonjit_result["log_likelihood"]))
        nonjit_score_error = _tensor_float(tf.reduce_max(tf.abs(result["score"] - nonjit_result["score"])))
    except Exception as exc:  # pragma: no cover - backend-specific diagnostic
        nonjit_error_message = f"{type(exc).__name__}: {exc}"

    xla_value_error = None
    xla_score_error = None
    xla_error_message = ""
    if check_xla:
        try:
            xla_result = program.compiled(jit_compile=True)(theta)
            xla_value_error = _tensor_float(tf.abs(result["log_likelihood"] - xla_result["log_likelihood"]))
            xla_score_error = _tensor_float(tf.reduce_max(tf.abs(result["score"] - xla_result["score"])))
        except Exception as exc:  # pragma: no cover - backend-specific diagnostic
            xla_error_message = f"{type(exc).__name__}: {exc}"

    apf = _apf_identity_and_law_diagnostics(
        model, branch, theta, expected_final_log_weights=result["final_log_weights"]
    )
    proposal = _branch_proposal_diagnostics(compilation)
    checks = {
        "finite_exact_value_and_score": finite,
        "analytical_score_central_fd": score_error <= 2.0e-7,
        "complete_denominator_and_law_finite": bool(proposal["finite"]),
        "apf_w_over_a_identity": apf["max_identity_backward_error"]
        <= APF_IDENTITY_BACKWARD_BOUND,
        "apf_recomputed_final_weight_parity": apf[
            "final_log_weight_parity_backward_error"
        ]
        <= APF_IDENTITY_BACKWARD_BOUND,
        "auxiliary_rows_normalized": apf["max_auxiliary_normalization_abs_error"] <= ABS_TOL,
        "nonjit_exact_program_parity": nonjit_value_error <= 2.0e-10 and nonjit_score_error <= 2.0e-10,
    }
    if check_xla:
        checks["xla_exact_program_parity"] = (
            xla_value_error is not None
            and xla_score_error is not None
            and xla_value_error <= 2.0e-10
            and xla_score_error <= 2.0e-10
        )
    if label in (
        "ukf_apf_k1_defensive",
        "ukf_apf_k2_defensive",
        "ukf_apf_k4_defensive",
    ):
        checks["per_ancestor_observation_response"] = (
            proposal.get("per_ancestor_posterior_mean_spread_max", 0.0) > 1.0e-4
            and proposal.get("lookahead_spread_max", 0.0) > 1.0e-4
        )
        checks["smooth_defensive_gate_response"] = (
            proposal.get("epsilon_spread_max", 0.0) > 1.0e-8
        )
        checks["defensive_fraction_bounds"] = (
            0.0 < float(proposal.get("epsilon_min", -1.0))
            <= float(proposal.get("epsilon_max", -1.0))
            < 1.0
        )
        checks["component_covariance_spd"] = (
            proposal.get("component_minimum_eigenvalue", float("-inf")) > 0.0
        )
        checks["local_moment_recomposition"] = (
            proposal.get("moment_recomposition_max_abs", float("inf")) <= ABS_TOL
        )
        checks["complete_mixture_label_invariance"] = (
            proposal.get("label_permutation_max_abs", float("inf")) <= ABS_TOL
        )
    elif label == "ukf_apf_k1":
        checks["per_ancestor_observation_response"] = (
            proposal.get("per_ancestor_posterior_mean_spread_max", 0.0) > 1.0e-4
            and proposal.get("lookahead_spread_max", 0.0) > 1.0e-4
        )
        checks["complete_k1_density_recomposition"] = (
            proposal.get("proposal_density_recomposition_max_abs", float("inf")) <= ABS_TOL
        )
    elif label in ("ukf_apf_k2", "ukf_apf_k4"):
        checks["per_ancestor_observation_response"] = (
            proposal.get("per_ancestor_posterior_mean_spread_max", 0.0) > 1.0e-4
            and proposal.get("lookahead_spread_max", 0.0) > 1.0e-4
        )
        checks["component_covariance_spd"] = (
            proposal.get("component_minimum_eigenvalue", float("-inf")) > 0.0
        )
        checks["moment_recomposition"] = (
            proposal.get("moment_recomposition_max_abs", float("inf")) <= ABS_TOL
        )
        checks["complete_mixture_label_invariance"] = (
            proposal.get("label_permutation_max_abs", float("inf")) <= ABS_TOL
        )
    return {
        "label": label,
        "compiler_id": compilation.compiler_id,
        "branch_id": branch.branch_id,
        "particle_count": branch.particle_count,
        "time_steps": branch.time_steps,
        "program_id": program.program_id,
        "output_device": str(result["log_likelihood"].device),
        "log_likelihood": _tensor_float(result["log_likelihood"]),
        "score": _tensor_vector(result["score"]),
        "finite_difference_score": finite_difference,
        "score_max_abs_error": score_error,
        "minimum_ess": _tensor_float(result["minimum_ess"]),
        "ess_by_time": _tensor_vector(result["ess_by_time"]),
        "maximum_log_weight_spread": _tensor_float(result["maximum_log_weight_spread"]),
        "maximum_normalized_weight_by_time": _tensor_vector(result["maximum_normalized_weight_by_time"]),
        "proposal": proposal,
        "apf": apf,
        "parity": {
            "nonjit_value_max_abs": nonjit_value_error,
            "nonjit_score_max_abs": nonjit_score_error,
            "nonjit_error": nonjit_error_message,
            "xla_value_max_abs": xla_value_error,
            "xla_score_max_abs": xla_score_error,
            "xla_error": xla_error_message,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "interpretation": "descriptive finite-program smoke; ESS is not a correctness or superiority claim",
    }


def _phase1_result_markdown(payload: Mapping[str, object]) -> str:
    records = payload["records"]
    lines = [
        "# C2 Mixture-UKF/APF Phase 1 Smoke Result",
        "",
        f"Phase: `{PHASE1_ID}`",
        f"Status: `{payload['status']}`",
        f"Continuation: `{payload['continuation']}`",
        "",
        "The run is a CPU/reference smoke at the frozen C2 horizon. It checks the",
        "generic UKF/APF call chain and the exact frozen finite program. ESS and",
        "log-likelihood contrasts are descriptive; this artifact makes no",
        "posterior, unbiasedness, superiority, or production claim.",
        "",
        "## Decision",
        "",
        "| Item | Status |",
        "| --- | --- |",
        f"| Required K=1 checks | `{payload['required_checks_pass']}` |",
        f"| Comparator records complete | `{payload['comparator_records_complete']}` |",
        f"| Failure class | `{payload['failure_class']}` |",
        f"| Budget remaining | `{payload['budget_remaining']}` |",
        "",
        "## Candidate records",
        "",
        "| Row | Branch | Candidate | log L | min ESS | score FD max | checks |",
        "| ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['particle_count']} | {record['branch_index']} | "
            f"`{record['label']}` | {record.get('log_likelihood', 'error')} | "
            f"{record.get('minimum_ess', 'error')} | "
            f"{record.get('score_max_abs_error', 'error')} | "
            f"`{record.get('all_checks_pass', False)}` |"
        )
    lines.extend(
        [
            "",
            "## Required diagnostics",
            "",
            "The JSON record contains APF identity errors, auxiliary/base-mass",
            "normalization, complete K=1 density recomposition, observation",
            "sensitivity, exact-program eager/non-JIT/XLA parity, and raw per-time",
            "ESS. A comparator failure is retained as candidate evidence and is not",
            "silently converted into a UKF implementation failure.",
            "",
            "## Inference status",
            "",
            "| Evidence class | Status |",
            "| --- | --- |",
            "| Hard validity vetoes | determined from the required K=1 checks above |",
            "| Statistically supported ranking | not attempted; one smoke branch is descriptive |",
            "| Descriptive differences | recorded per row and candidate |",
            "| Default readiness | not evaluated |",
            "| Next evidence | close Phase 1, refresh Phase 2, then run the declared serious GPU ladder |",
            "",
            f"Formal math status: `{payload['formal_tool_status']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_phase1_smoke(args: argparse.Namespace) -> Path:
    _load_algorithm_modules()
    fixture = _load_c2_fixture()
    rows = _parse_rows(args.rows)
    if int(args.branches) < 1 or int(args.branches) > 12:
        raise ValueError("--branches must lie in [1, 12]")
    output = _make_output_root(args.output_root)
    model, theta, observations = _c2_model_and_inputs(fixture)
    records: list[dict[str, object]] = []

    for row in rows:
        for branch_index in range(int(args.branches)):
            seed = 91000 + 1009 * int(branch_index) + int(row)
            candidate_specs = (
                (
                    "ukf_apf_k1",
                    lambda: compile_c2_per_ancestor_ukf_apf_k1(
                        model=model,
                        observations=observations,
                        theta_reference=theta,
                        particle_count=row,
                        seed=seed,
                        jit_compile=bool(args.jit_compile),
                    ),
                    True,
                ),
                (
                    "bootstrap",
                    lambda: compile_c2_bootstrap_proposal_branch(
                        model=model,
                        observations=observations,
                        theta_reference=theta,
                        particle_count=row,
                        seed=seed,
                        jit_compile_sampler=bool(args.jit_compile),
                    ),
                    False,
                ),
                (
                    "transformed_student_nu8",
                    lambda: compile_c2_transformed_student_proposal_branch(
                        model=model,
                        observations=observations,
                        theta_reference=theta,
                        nu=8.0,
                        particle_count=row,
                        seed=seed,
                        jit_compile_sampler=bool(args.jit_compile),
                    ),
                    False,
                ),
            )
            for label, compiler, check_xla in candidate_specs:
                record: dict[str, object] = {
                    "particle_count": row,
                    "branch_index": branch_index,
                    "seed": seed,
                    "label": label,
                    "requested_jit_compile": bool(args.jit_compile),
                }
                try:
                    compilation = compiler()
                    evaluated = _evaluate_c2_candidate(
                        label=label,
                        compilation=compilation,
                        model=model,
                        theta=theta,
                        check_xla=check_xla,
                    )
                    record.update(evaluated)
                except Exception as exc:  # pragma: no cover - retained run evidence
                    record.update(
                        {
                            "all_checks_pass": False,
                            "failure_class": "candidate_or_comparator_failure",
                            "error": f"{type(exc).__name__}: {exc}",
                            "interpretation": "candidate failure retained; inspect before promotion",
                        }
                    )
                records.append(record)

    required_records = [
        record for record in records if record.get("label") == "ukf_apf_k1"
    ]
    comparator_records = [
        record for record in records if record.get("label") != "ukf_apf_k1"
    ]
    required_checks_pass = bool(required_records) and all(
        bool(record.get("all_checks_pass", False)) for record in required_records
    )
    comparator_complete = bool(comparator_records) and all(
        "error" not in record for record in comparator_records
    )
    any_comparator_failure = any(
        "error" in record or not bool(record.get("all_checks_pass", False))
        for record in comparator_records
    )
    if required_checks_pass:
        status = "PASS_PHASE1_SMOKE" if not any_comparator_failure else "PASS_PHASE1_SMOKE_WITH_COMPARATOR_DIAGNOSTIC"
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "candidate_failure" if any_comparator_failure else "none"
        repair = (
            "retain comparator diagnostic and refresh Phase 2; no K=1 validity blocker"
            if any_comparator_failure
            else "none; required K=1 checks passed"
        )
    else:
        status = "VETO_PHASE1_SMOKE"
        continuation = "CONTINUATION_VETO_PHASE1_REQUIRED_CHECK"
        failure_class = "implementation_or_numerical_validity"
        repair = "repair the required K=1 check before opening Phase 2"

    payload = {
        "schema_version": PHASE1_RESULT_SCHEMA,
        "phase": PHASE1_ID,
        "status": status,
        "continuation": continuation,
        "failure_class": failure_class,
        "repair": repair,
        "next_phase_refresh": "Phase 2 serious GPU ladder remains closed until this smoke close note is reviewed",
        "budget_remaining": "four CPU-hours and six GPU-hours campaign budget; smoke consumed one bounded ladder",
        "formal_tool_status": "Phase 0 MathDevMCP/Lean audit recorded in phase0-attempt02/formal-audit.md",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
            "physical_devices": [device.name for device in tf.config.list_physical_devices()],
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "execution_lane": "cpu_reference_with_optional_xla",
        },
        "fixture": {
            "path": str(C2_FIXTURE_PATH.relative_to(ROOT)),
            "sha256": _sha256_file(C2_FIXTURE_PATH),
            "schema_id": fixture["schema_id"],
            "horizon": int(fixture["horizon"]),
            "state_dimension": int(fixture["state_dimension"]),
        },
        "sources": {
            "plan": {"path": str(PLAN_PATH.relative_to(ROOT)), "sha256": _sha256_file(PLAN_PATH)},
            "driver": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _sha256_file(Path(__file__).resolve())},
            "generic_kernel": {"path": str(MODULE_PATH.relative_to(ROOT)), "sha256": _sha256_file(MODULE_PATH)},
            "c2_adapter": {"path": str(C2_ADAPTER_PATH.relative_to(ROOT)), "sha256": _sha256_file(C2_ADAPTER_PATH)},
            "c2_model": {"path": str(C2_MODEL_PATH.relative_to(ROOT)), "sha256": _sha256_file(C2_MODEL_PATH)},
            "exact_evaluator": {"path": str(C2_EXACT_PATH.relative_to(ROOT)), "sha256": _sha256_file(C2_EXACT_PATH)},
        },
        "workspace": _workspace_manifest(),
        "rows": rows,
        "branches": int(args.branches),
        "seed_policy": "paired candidate seed per row and branch; candidate topology remains frozen after construction",
        "required_checks_pass": required_checks_pass,
        "comparator_records_complete": comparator_complete,
        "records": records,
        "nonclaims": (
            "no posterior correctness claim",
            "no unbiased likelihood claim",
            "no statistical superiority claim from one smoke branch",
            "no production or default-readiness claim",
        ),
    }
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "c2_mixture_ukf_apf_phase1_manifest_v1",
            "phase": PHASE1_ID,
            "command": " ".join(sys.argv),
            "plan_sha256": payload["sources"]["plan"]["sha256"],
            "workspace": payload["workspace"],
            "environment": payload["environment"],
            "fixture": payload["fixture"],
            "rows": rows,
            "branches": int(args.branches),
            "jit_compile_requested": bool(args.jit_compile),
        },
    )
    _write_json(output / "result.json", payload)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_phase1_result_markdown(payload), encoding="utf-8")
    return output


PHASE2_DEGREE = 6
PHASE2_RANK = 6
PHASE2_FIT_ROWS = 8192
PHASE2_SWEEPS = 32
PHASE2_RIDGE = 1.0e-10
PHASE2_TAU = 1.0e-6
PHASE2_NU = 8.0
PHASE2_ID = "c2_mixture_ukf_apf_phase2_entry_v1"
PHASE2_RESULT_SCHEMA = "c2_mixture_ukf_apf_phase2_entry_result_v1"
PHASE2_SERIOUS_ID = "c2_mixture_ukf_apf_phase2_serious_n8192_v1"
PHASE2_SERIOUS_RESULT_SCHEMA = "c2_mixture_ukf_apf_phase2_serious_n8192_result_v1"
PHASE2_SERIOUS_BRANCHES = 12
PHASE2_PAIRED_T_CRITICAL_DF11 = 2.200985
PHASE2_FAMILIES = (
    "ukf_apf_k1",
    "bootstrap_conditional",
    "transformed_student_nu8",
    "gaussian_hint_marginal",
    "stationary_independence",
    "retained_tt",
)

PHASE3_ID = "c2_mixture_ukf_apf_phase3_fixed_split_v1"
PHASE3_RESULT_SCHEMA = "c2_mixture_ukf_apf_phase3_fixed_split_result_v1"
PHASE3_MANIFEST_SCHEMA = "c2_mixture_ukf_apf_phase3_fixed_split_manifest_v1"
PHASE3_CALIBRATION_SCHEMA = "c2_mixture_ukf_apf_phase3_offset_calibration_v1"
PHASE3_CALIBRATION_HORIZON = 10
PHASE3_CALIBRATION_ROWS = 64
PHASE3_CALIBRATION_SEED = (20260903, 7301)
PHASE3_CLAIM_SEED_BASE = 97000
PHASE3_OFFSETS = (0.20, 0.35, 0.50)
PHASE3_SPD_MARGIN = 1.0e-6
PHASE3_BRANCHES = 12
PHASE3_PAIRED_T_CRITICAL_DF11 = PHASE2_PAIRED_T_CRITICAL_DF11
PHASE3_FAMILIES = (
    "ukf_apf_k1",
    "ukf_apf_k2",
    "ukf_apf_k4",
    "bootstrap_conditional",
    "transformed_student_nu8",
    "gaussian_hint_marginal",
    "stationary_independence",
    "retained_tt",
)

PHASE4_ID = "c2_mixture_ukf_apf_phase4_smooth_student_defensive_v1"
PHASE4_RESULT_SCHEMA = "c2_mixture_ukf_apf_phase4_smooth_student_defensive_result_v1"
PHASE4_MANIFEST_SCHEMA = "c2_mixture_ukf_apf_phase4_smooth_student_defensive_manifest_v1"
PHASE4_CALIBRATION_SCHEMA = "c2_mixture_ukf_apf_phase4_defensive_calibration_v1"
PHASE4_REPAIR_ID = "c2_mixture_ukf_apf_phase4_backward_error_repair_v1"
PHASE4_REPAIR_RESULT_SCHEMA = "c2_mixture_ukf_apf_phase4_backward_error_repair_result_v1"
PHASE4_REPAIR_MANIFEST_SCHEMA = "c2_mixture_ukf_apf_phase4_backward_error_repair_manifest_v1"
PHASE4_FIRST_ATTEMPT_PATH = ROOT / "docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase4-smooth-defensive-n8192-attempt01"
PHASE4_CALIBRATION_HORIZON = 10
PHASE4_CALIBRATION_ROWS = 64
PHASE4_CALIBRATION_SEED = (20260903, 7401)
PHASE4_CLAIM_SEED_BASE = 98000
PHASE4_NU_LADDER = (5.0, 8.0)
PHASE4_EPSILON_LADDER = ((0.05, 0.20), (0.10, 0.30))
PHASE4_GATE_CENTER = 4.0
PHASE4_GATE_TEMPERATURE = 8.0
PHASE4_BRANCHES = 12
PHASE4_FAMILIES = (
    "ukf_apf_k1_defensive",
    "ukf_apf_k2_defensive",
    "ukf_apf_k4_defensive",
    "ukf_apf_k1",
    "ukf_apf_k2",
    "ukf_apf_k4",
    "bootstrap_conditional",
    "transformed_student_nu8",
    "gaussian_hint_marginal",
    "stationary_independence",
    "retained_tt",
)


def _phase2_frozen_adapter(fixture: Mapping[str, object]):
    """Build the TensorFlow-only adapter used to create a fresh TT snapshot."""

    from bayesfilter.highdim.squared_tt_engine_v0_tf import DensityKernelAdapter

    dimension = int(fixture["state_dimension"])
    transition = _tensor(fixture["transition_matrix"])
    process_chol = tf.linalg.cholesky(_tensor(fixture["process_covariance"]))
    initial_chol = tf.linalg.cholesky(_tensor(fixture["stationary_covariance"]))
    beta = tf.constant(float(fixture["beta"]), DTYPE)

    def mvn(states: tf.Tensor, means: tf.Tensor, chol: tf.Tensor) -> tf.Tensor:
        centered = states - means
        whitened = tf.transpose(
            tf.linalg.triangular_solve(chol, tf.transpose(centered), lower=True)
        )
        return -0.5 * (
            tf.constant(dimension * math.log(2.0 * math.pi), DTYPE)
            + tf.reduce_sum(tf.square(whitened), axis=1)
        ) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))

    def transition_log_density(current: tf.Tensor, previous: tf.Tensor) -> tf.Tensor:
        means = tf.linalg.matmul(previous, transition, transpose_b=True)
        return mvn(current, means, process_chol)

    def observation_log_density(current: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        observed = tf.ensure_shape(tf.convert_to_tensor(observation, DTYPE), [dimension])
        return tf.reduce_sum(
            -0.5 * tf.constant(math.log(2.0 * math.pi), DTYPE)
            - tf.math.log(beta)
            - 0.5 * current
            - 0.5 * tf.square(observed)[None, :]
            * tf.exp(-current)
            / tf.square(beta),
            axis=1,
        )

    def initial_log_density(current: tf.Tensor) -> tf.Tensor:
        return mvn(current, tf.zeros_like(current), initial_chol)

    return DensityKernelAdapter(
        state_dim=dimension,
        transition_log_density=transition_log_density,
        observation_log_density=observation_log_density,
        initial_log_density=initial_log_density,
    )


def _phase2_hint_factory(fixture: Mapping[str, object], horizon: int):
    hints = tuple(fixture["moment_hints"][:horizon])
    cursor = {"value": 0}

    def initial_hint(_observation: tf.Tensor):
        if cursor["value"] != 0:
            raise RuntimeError("phase2 frozen hints were consumed out of order")
        cursor["value"] = 1
        return _tensor(hints[0]["mean"]), _tensor(hints[0]["covariance"])

    def predictive_hint(time_index: int, _observation: tf.Tensor):
        if int(time_index) != cursor["value"]:
            raise RuntimeError("phase2 frozen hints were consumed out of order")
        cursor["value"] += 1
        return _tensor(hints[int(time_index)]["mean"]), _tensor(
            hints[int(time_index)]["covariance"]
        )

    return initial_hint, predictive_hint


def _save_phase2_snapshot(snapshot, snapshot_api, output_root: Path) -> Mapping[str, object]:
    metadata, tensors = snapshot_api.gaussian_xla_retained_proposal_snapshot_parts(snapshot)
    step_root = output_root / "snapshots" / f"t{int(snapshot.time_index):02d}"
    step_root.mkdir(parents=True, exist_ok=False)
    tensor_rows: dict[str, object] = {}
    for name, tensor in tensors.items():
        path = step_root / f"{name}.tensor"
        tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
        tensor_rows[name] = {
            "path": str(path.relative_to(output_root)),
            "sha256": _sha256_file(path),
        }
    metadata = dict(metadata)
    metadata["snapshot_fingerprint"] = snapshot_api.gaussian_xla_retained_proposal_snapshot_fingerprint(snapshot)
    metadata["tensor_files"] = tensor_rows
    metadata_path = step_root / "metadata.json"
    _write_json(metadata_path, metadata)
    return {
        "time_index": int(snapshot.time_index),
        "snapshot_fingerprint": metadata["snapshot_fingerprint"],
        "metadata_path": str(metadata_path.relative_to(output_root)),
        "metadata_sha256": _sha256_file(metadata_path),
        "tensor_count": len(tensors),
    }


def _configure_phase2_gpu() -> Mapping[str, object]:
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    if _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH is not None:
        # Restore only after all module imports have completed; restoring
        # earlier lets an import-time TensorFlow op create logical devices.
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if not logical_gpus:
        raise RuntimeError("Phase 2 requires at least one logical TensorFlow GPU")
    with tf.device("/GPU:0"):
        placement_probe = tf.reduce_sum(tf.ones([32], DTYPE))
    if "GPU" not in str(placement_probe.device).upper():
        raise RuntimeError(
            f"Phase 2 placement probe did not execute on GPU: {placement_probe.device}"
        )
    return {
        "memory_policy": memory_policy,
        "logical_devices": [device.name for device in logical_gpus],
        "placement_probe_device": str(placement_probe.device),
        "placement_probe_value": _tensor_float(placement_probe),
    }


def _prepare_phase2_proposals(
    *,
    fixture: Mapping[str, object],
    model: C2StochasticVolatilityFrozenAPFModel,
    theta: tf.Tensor,
    observations: tf.Tensor,
    output_root: Path,
) -> Mapping[str, object]:
    """Capture fresh retained-TT, hint, and stationary proposals for Phase 2."""

    from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
        retained_proposal_from_transition_snapshot,
    )
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
        FrozenGaussianStateProposal,
        stationary_gaussian_proposals,
        transformed_student_proposals,
    )
    from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
    import bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf as snapshot_api

    horizon = int(observations.shape[0])
    adapter = _phase2_frozen_adapter(fixture)
    initial_hint, predictive_hint = _phase2_hint_factory(fixture, horizon)
    config = EngineConfig(
        basis_degree=PHASE2_DEGREE,
        rank=PHASE2_RANK,
        row_count=PHASE2_FIT_ROWS,
        sweeps=PHASE2_SWEEPS,
        ridge=PHASE2_RIDGE,
        tau=PHASE2_TAU,
        coordinate_half_width=3.0,
        seed=98000 + 100 * int(fixture["state_dimension"]) + 10 * PHASE2_DEGREE + PHASE2_RANK,
        row_design="sobol",
    )
    run_identity_payload = {
        "fixture_sha256": _sha256_file(C2_FIXTURE_PATH),
        "horizon": horizon,
        "degree": PHASE2_DEGREE,
        "rank": PHASE2_RANK,
        "fit_rows": PHASE2_FIT_ROWS,
        "sweeps": PHASE2_SWEEPS,
        "ridge": PHASE2_RIDGE,
        "tau": PHASE2_TAU,
    }
    run_identity = hashlib.sha256(
        json.dumps(run_identity_payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    started = time.perf_counter()
    _, direct_diagnostics, snapshots = snapshot_api.run_value_filter_branch_axis_gaussian_xla_retained_proposal_diagnostic(
        adapter,
        observations,
        config,
        predictive_moment_hint=predictive_hint,
        initial_moment_hint=initial_hint,
        capture_steps=tuple(range(1, horizon)),
        run_identity=run_identity,
        defensive_nu=PHASE2_NU,
    )
    fit_seconds = time.perf_counter() - started
    if set(snapshots) != set(range(1, horizon)):
        raise ValueError("fresh retained-TT snapshot set is incomplete")
    snapshot_rows = [
        _save_phase2_snapshot(snapshots[time_index], snapshot_api, output_root)
        for time_index in range(1, horizon)
    ]
    tt_proposals = tuple(
        retained_proposal_from_transition_snapshot(snapshots[time_index])
        for time_index in range(1, horizon)
    )
    hint_proposals = tuple(
        FrozenGaussianStateProposal(
            mean=snapshots[time_index].coordinate_offset,
            chol=snapshots[time_index].coordinate_matrix,
            time_index=time_index,
            family="gaussian_hint_marginal",
        )
        for time_index in range(1, horizon)
    )
    stationary = stationary_gaussian_proposals(model, theta, horizon)
    defensive = transformed_student_proposals(
        model=model, observations=observations, theta_reference=theta, nu=PHASE2_NU
    )
    proposal_manifest = {
        "schema_id": "c2_mixture_ukf_apf_phase2_proposal_manifest_v1",
        "run_identity": run_identity,
        "fit_seconds": fit_seconds,
        "config": {
            "basis_degree": PHASE2_DEGREE,
            "rank": PHASE2_RANK,
            "row_count": PHASE2_FIT_ROWS,
            "sweeps": PHASE2_SWEEPS,
            "ridge": PHASE2_RIDGE,
            "tau": PHASE2_TAU,
            "row_design": "sobol",
        },
        "snapshots": snapshot_rows,
        "retained_tt": [proposal.manifest_payload() for proposal in tt_proposals],
        "gaussian_hint": [proposal.manifest_payload() for proposal in hint_proposals],
        "stationary": [proposal.manifest_payload() for proposal in stationary],
        "transformed_student": [proposal.manifest_payload() for proposal in defensive],
        "direct_diagnostics": direct_diagnostics,
    }
    _write_json(output_root / "proposal_manifest.json", proposal_manifest)
    return {
        "tt_proposals": tt_proposals,
        "hint_proposals": hint_proposals,
        "stationary_proposals": stationary,
        "defensive_proposals": defensive,
        "run_identity": run_identity,
        "fit_seconds": fit_seconds,
        "snapshot_rows": snapshot_rows,
    }


def _phase3_calibration_observations(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    theta: tf.Tensor,
    horizon: int,
    seed: tuple[int, int] = PHASE3_CALIBRATION_SEED,
) -> tf.Tensor:
    """Generate a deterministic calibration bank independent of the claim bank.

    The claim fixture has a frozen observation seed.  Calibration uses a
    separate stateless seed and a model-generated state/observation path, so
    offset selection cannot inspect the claim observations.  The recursion is
    expressed with ``tf.while_loop`` and remains outside the claim score path.
    """

    horizon = int(horizon)
    dimension = model.state_dim()
    if horizon < 2:
        raise ValueError("Phase 3 calibration horizon must be at least two")
    transition = model.transition_matrix(theta)
    stationary_covariance, _ = model.stationary_covariance_and_derivative(theta)
    stationary_cholesky = tf.linalg.cholesky(stationary_covariance)
    seed = tuple(int(value) for value in seed)
    if len(seed) != 2:
        raise ValueError("calibration seed must contain two integers")
    process_normals = tf.random.stateless_normal(
        [horizon, dimension], seed, dtype=DTYPE
    )
    observation_normals = tf.random.stateless_normal(
        [horizon, dimension],
        [seed[0], seed[1] + 1],
        dtype=DTYPE,
    )
    states = tf.TensorArray(
        DTYPE, size=horizon, clear_after_read=False, element_shape=[dimension]
    )
    initial_state = tf.linalg.matvec(stationary_cholesky, process_normals[0])
    states = states.write(0, initial_state)

    def condition(index: tf.Tensor, _states: tf.TensorArray) -> tf.Tensor:
        return index < horizon

    def body(index: tf.Tensor, state_array: tf.TensorArray):
        previous = state_array.read(index - 1)
        current = tf.linalg.matvec(transition, previous) + tf.constant(
            float(model.sigma), DTYPE
        ) * process_normals[index]
        return index + 1, state_array.write(index, current)

    _, states = tf.while_loop(
        condition,
        body,
        (tf.constant(1, tf.int32), states),
        parallel_iterations=1,
    )
    latent_states = states.stack()
    observations = tf.exp(0.5 * latent_states + theta[1]) * observation_normals
    tf.debugging.assert_all_finite(
        observations, "Phase 3 calibration observations must be finite"
    )
    return tf.ensure_shape(observations, [horizon, dimension])


def _phase3_calibrate_offset(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    theta: tf.Tensor,
    output_root: Path,
) -> Mapping[str, object]:
    """Select the largest predeclared split offset passing calibration gates."""

    observations = _phase3_calibration_observations(
        model=model, theta=theta, horizon=PHASE3_CALIBRATION_HORIZON
    )
    observation_path = output_root / "calibration_observations.tensor"
    tf.io.write_file(str(observation_path), tf.io.serialize_tensor(observations))
    rows: list[dict[str, object]] = []
    for offset in PHASE3_OFFSETS:
        for component_count in (2, 4):
            started = time.perf_counter()
            row: dict[str, object] = {
                "offset": float(offset),
                "component_count": component_count,
                "particle_count": PHASE3_CALIBRATION_ROWS,
                "horizon": PHASE3_CALIBRATION_HORIZON,
                "seed": 86000 + 100 * component_count + int(round(1000 * offset)),
            }
            try:
                compilation = compile_c2_per_ancestor_ukf_apf_mixture(
                    model=model,
                    observations=observations,
                    theta_reference=theta,
                    particle_count=PHASE3_CALIBRATION_ROWS,
                    seed=int(row["seed"]),
                    component_count=component_count,
                    offset=float(offset),
                    jit_compile=True,
                )
                component_minimum = min(
                    float(tf.reduce_min(item["component_minimum_eigenvalue"]).numpy())
                    for item in compilation.proposal_diagnostics
                )
                moment_error = max(
                    float(item["moment_recomposition_max_abs"].numpy())
                    for item in compilation.proposal_diagnostics
                )
                label_error = max(
                    float(item["label_permutation_max_abs"].numpy())
                    for item in compilation.proposal_diagnostics
                )
                density_error = max(
                    float(item["proposal_density_recomposition_max_abs"].numpy())
                    for item in compilation.proposal_diagnostics
                )
                finite = all(
                    bool(item["proposal_finite"].numpy())
                    and bool(item["exact_prefix_finite"].numpy())
                    for item in compilation.proposal_diagnostics
                )
                row.update(
                    {
                        "finite": finite,
                        "component_minimum_eigenvalue": component_minimum,
                        "moment_recomposition_max_abs": moment_error,
                        "label_permutation_max_abs": label_error,
                        "proposal_density_recomposition_max_abs": density_error,
                        "all_checks_pass": bool(
                            finite
                            and component_minimum > PHASE3_SPD_MARGIN
                            and moment_error <= ABS_TOL
                            and label_error <= ABS_TOL
                            and density_error <= ABS_TOL
                        ),
                    }
                )
            except Exception as exc:  # pragma: no cover - retained calibration evidence
                row.update(
                    {
                        "all_checks_pass": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            row["wall_seconds"] = time.perf_counter() - started
            rows.append(row)

    valid_offsets = [
        offset
        for offset in PHASE3_OFFSETS
        if all(
            bool(row.get("all_checks_pass", False))
            for row in rows
            if float(row["offset"]) == float(offset)
        )
    ]
    selected = max(valid_offsets) if valid_offsets else None
    payload: dict[str, object] = {
        "schema_version": PHASE3_CALIBRATION_SCHEMA,
        "calibration_seed": list(PHASE3_CALIBRATION_SEED),
        "observation_path": str(observation_path.relative_to(output_root)),
        "observation_sha256": _sha256_file(observation_path),
        "horizon": PHASE3_CALIBRATION_HORIZON,
        "particle_count": PHASE3_CALIBRATION_ROWS,
        "offset_ladder": list(PHASE3_OFFSETS),
        "spd_margin": PHASE3_SPD_MARGIN,
        "rows": rows,
        "valid_offsets": valid_offsets,
        "selected_offset": selected,
        "selection_rule": "largest predeclared offset for which both K=2 and K=4 pass all calibration checks",
        "claim_observations_not_used": True,
    }
    _write_json(output_root / "calibration.json", payload)
    markdown = [
        "# Phase 3 Offset Calibration",
        "",
        f"Selected offset: `{selected if selected is not None else 'none'}`",
        f"Calibration observations: `{payload['observation_path']}`",
        "",
        "The bank uses an independent stateless model-generated path and is not",
        "the claim observation sequence. The largest predeclared dimensionless",
        "offset is selected only when both K=2 and K=4 pass SPD, moment, label,",
        "density, and finite checks.",
        "",
        "| Offset | K | min component eigenvalue | moment error | label error | density error | pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        markdown.append(
            f"| {row['offset']} | {row['component_count']} | "
            f"{row.get('component_minimum_eigenvalue', 'error')} | "
            f"{row.get('moment_recomposition_max_abs', 'error')} | "
            f"{row.get('label_permutation_max_abs', 'error')} | "
            f"{row.get('proposal_density_recomposition_max_abs', 'error')} | "
            f"`{row.get('all_checks_pass', False)}` |"
        )
    (output_root / "calibration.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    if selected is None:
        raise RuntimeError("no predeclared Phase 3 offset passed calibration")
    return {
        "observations": observations,
        "selected_offset": float(selected),
        "rows": rows,
        "payload": payload,
    }


def _phase4_calibrate_controls(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    theta: tf.Tensor,
    output_root: Path,
) -> Mapping[str, object]:
    """Select a defensive Student/gate configuration on an independent bank."""

    observations = _phase3_calibration_observations(
        model=model,
        theta=theta,
        horizon=PHASE4_CALIBRATION_HORIZON,
        seed=PHASE4_CALIBRATION_SEED,
    )
    observation_path = output_root / "calibration_observations.tensor"
    tf.io.write_file(str(observation_path), tf.io.serialize_tensor(observations))
    configurations = [
        (f"nu{int(nu)}_eps{int(round(100 * eps_min)):02d}_{int(round(100 * eps_max)):02d}", nu, eps_min, eps_max)
        for nu in PHASE4_NU_LADDER
        for eps_min, eps_max in PHASE4_EPSILON_LADDER
    ]
    rows: list[dict[str, object]] = []
    for config_index, (config_id, nu, eps_min, eps_max) in enumerate(configurations):
        for local_component_count in (1, 2, 4):
            started = time.perf_counter()
            row: dict[str, object] = {
                "config_id": config_id,
                "nu": float(nu),
                "epsilon_min": float(eps_min),
                "epsilon_max": float(eps_max),
                "gate_center": PHASE4_GATE_CENTER,
                "gate_temperature": PHASE4_GATE_TEMPERATURE,
                "local_component_count": local_component_count,
                "offset": PHASE3_OFFSETS[-1],
                "particle_count": PHASE4_CALIBRATION_ROWS,
                "horizon": PHASE4_CALIBRATION_HORIZON,
                "seed": 87000 + 1000 * config_index + local_component_count,
            }
            try:
                compilation = compile_c2_per_ancestor_ukf_apf_defensive_mixture(
                    model=model,
                    observations=observations,
                    theta_reference=theta,
                    particle_count=PHASE4_CALIBRATION_ROWS,
                    seed=int(row["seed"]),
                    local_component_count=local_component_count,
                    offset=PHASE3_OFFSETS[-1],
                    nu=float(nu),
                    epsilon_min=float(eps_min),
                    epsilon_max=float(eps_max),
                    gate_center=PHASE4_GATE_CENTER,
                    gate_temperature=PHASE4_GATE_TEMPERATURE,
                    jit_compile=True,
                )
                proposal = _branch_proposal_diagnostics(compilation)
                exact = prepare_frozen_proposal_apf_program(
                    model, compilation.branch
                ).evaluate(theta)
                row.update(
                    {
                        "finite": bool(exact["finite"].numpy()) and bool(proposal["finite"]),
                        "minimum_ess": _tensor_float(exact["minimum_ess"]),
                        "component_minimum_eigenvalue": proposal.get(
                            "component_minimum_eigenvalue", float("-inf")
                        ),
                        "moment_recomposition_max_abs": proposal.get(
                            "moment_recomposition_max_abs", float("inf")
                        ),
                        "label_permutation_max_abs": proposal.get(
                            "label_permutation_max_abs", float("inf")
                        ),
                        "proposal_density_recomposition_max_abs": proposal.get(
                            "proposal_density_recomposition_max_abs", float("inf")
                        ),
                        "epsilon_spread_max": proposal.get("epsilon_spread_max", 0.0),
                        "all_checks_pass": bool(
                            bool(exact["finite"].numpy())
                            and bool(proposal["finite"])
                            and float(proposal.get("component_minimum_eigenvalue", -1.0))
                            > PHASE3_SPD_MARGIN
                            and float(proposal.get("moment_recomposition_max_abs", float("inf")))
                            <= ABS_TOL
                            and float(proposal.get("label_permutation_max_abs", float("inf")))
                            <= ABS_TOL
                            and float(
                                proposal.get(
                                    "proposal_density_recomposition_max_abs", float("inf")
                                )
                            )
                            <= ABS_TOL
                            and float(proposal.get("epsilon_spread_max", 0.0)) > 1.0e-8
                        ),
                    }
                )
            except Exception as exc:  # pragma: no cover - retained calibration evidence
                row.update(
                    {
                        "all_checks_pass": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            row["wall_seconds"] = time.perf_counter() - started
            rows.append(row)

    valid_config_ids = [
        config_id
        for config_id, _, _, _ in configurations
        if all(
            bool(row.get("all_checks_pass", False))
            for row in rows
            if row.get("config_id") == config_id
        )
    ]
    config_scores = {
        config_id: min(
            float(row.get("minimum_ess", 0.0))
            for row in rows
            if row.get("config_id") == config_id
        )
        for config_id in valid_config_ids
    }
    selected_id = max(valid_config_ids, key=lambda value: config_scores[value]) if valid_config_ids else None
    selected = next(
        (
            {
                "config_id": config_id,
                "nu": float(nu),
                "epsilon_min": float(eps_min),
                "epsilon_max": float(eps_max),
                "gate_center": PHASE4_GATE_CENTER,
                "gate_temperature": PHASE4_GATE_TEMPERATURE,
                "offset": PHASE3_OFFSETS[-1],
                "local_component_counts": [1, 2, 4],
            }
            for config_id, nu, eps_min, eps_max in configurations
            if config_id == selected_id
        ),
        None,
    )
    payload: dict[str, object] = {
        "schema_version": PHASE4_CALIBRATION_SCHEMA,
        "calibration_seed": list(PHASE4_CALIBRATION_SEED),
        "observation_path": str(observation_path.relative_to(output_root)),
        "observation_sha256": _sha256_file(observation_path),
        "horizon": PHASE4_CALIBRATION_HORIZON,
        "particle_count": PHASE4_CALIBRATION_ROWS,
        "offset": PHASE3_OFFSETS[-1],
        "nu_ladder": list(PHASE4_NU_LADDER),
        "epsilon_ladder": [list(pair) for pair in PHASE4_EPSILON_LADDER],
        "gate_center": PHASE4_GATE_CENTER,
        "gate_temperature": PHASE4_GATE_TEMPERATURE,
        "rows": rows,
        "valid_config_ids": valid_config_ids,
        "config_scores": config_scores,
        "selected": selected,
        "selection_rule": "largest minimum calibration ESS across K=1, K=2, K=4 among configurations passing all proposal checks",
        "claim_observations_not_used": True,
    }
    _write_json(output_root / "calibration.json", payload)
    markdown = [
        "# Phase 4 Defensive Calibration",
        "",
        f"Selected configuration: `{selected_id if selected_id is not None else 'none'}`",
        f"Calibration observations: `{payload['observation_path']}`",
        "",
        "The bank is independent of the claim observation sequence. A configuration",
        "is eligible only when K=1, K=2, and K=4 pass finite, support, SPD,",
        "moment, label, density, and smooth-gate checks. Selection maximizes the",
        "minimum calibration ESS across those three local topologies.",
        "",
        "| Config | nu | eps min | eps max | K | min ESS | min eig | eps spread | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        markdown.append(
            f"| {row['config_id']} | {row['nu']} | {row['epsilon_min']} | "
            f"{row['epsilon_max']} | {row['local_component_count']} | "
            f"{row.get('minimum_ess', 'error')} | "
            f"{row.get('component_minimum_eigenvalue', 'error')} | "
            f"{row.get('epsilon_spread_max', 'error')} | "
            f"`{row.get('all_checks_pass', False)}` |"
        )
    (output_root / "calibration.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    if selected is None:
        raise RuntimeError("no Phase 4 defensive configuration passed calibration")
    return {"observations": observations, "selected": selected, "rows": rows, "payload": payload}


def _phase2_candidate_specs(
    *,
    model,
    observations: tf.Tensor,
    theta: tf.Tensor,
    proposal_sets: Mapping[str, object],
    particle_count: int,
    seed: int,
):
    """Return the frozen six-family compiler set for one paired branch."""

    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
        compile_c2_independent_proposal_branch,
    )

    return (
        (
            "ukf_apf_k1",
            lambda: compile_c2_per_ancestor_ukf_apf_k1(
                model=model,
                observations=observations,
                theta_reference=theta,
                particle_count=particle_count,
                seed=seed,
                jit_compile=True,
            ),
            True,
        ),
        (
            "bootstrap_conditional",
            lambda: compile_c2_bootstrap_proposal_branch(
                model=model,
                observations=observations,
                theta_reference=theta,
                particle_count=particle_count,
                seed=seed,
                jit_compile_sampler=True,
            ),
            False,
        ),
        (
            "transformed_student_nu8",
            lambda: compile_c2_transformed_student_proposal_branch(
                model=model,
                observations=observations,
                theta_reference=theta,
                nu=PHASE2_NU,
                particle_count=particle_count,
                seed=seed,
                jit_compile_sampler=True,
            ),
            False,
        ),
        (
            "gaussian_hint_marginal",
            lambda: compile_c2_independent_proposal_branch(
                model=model,
                observations=observations,
                theta_reference=theta,
                transition_proposals=proposal_sets["gaussian_hint_marginal"],
                particle_count=particle_count,
                seed=seed,
                family="gaussian_hint_marginal",
                jit_compile_sampler=True,
            ),
            False,
        ),
        (
            "stationary_independence",
            lambda: compile_c2_independent_proposal_branch(
                model=model,
                observations=observations,
                theta_reference=theta,
                transition_proposals=proposal_sets["stationary_independence"],
                particle_count=particle_count,
                seed=seed,
                family="stationary_independence",
                jit_compile_sampler=True,
            ),
            False,
        ),
        (
            "retained_tt",
            lambda: compile_c2_independent_proposal_branch(
                model=model,
                observations=observations,
                theta_reference=theta,
                transition_proposals=proposal_sets["retained_tt"],
                particle_count=particle_count,
                seed=seed,
                family="retained_tt",
                jit_compile_sampler=True,
            ),
            False,
        ),
    )


def _phase3_candidate_specs(
    *,
    model,
    observations: tf.Tensor,
    theta: tf.Tensor,
    proposal_sets: Mapping[str, object],
    particle_count: int,
    seed: int,
    mixture_offset: float,
):
    """Return Phase 2 families plus the fixed K=2/K=4 split arms."""

    phase2_specs = _phase2_candidate_specs(
        model=model,
        observations=observations,
        theta=theta,
        proposal_sets=proposal_sets,
        particle_count=particle_count,
        seed=seed,
    )
    mixture_specs = (
        (
            "ukf_apf_k2",
            lambda: compile_c2_per_ancestor_ukf_apf_mixture(
                model=model,
                observations=observations,
                theta_reference=theta,
                particle_count=particle_count,
                seed=seed,
                component_count=2,
                offset=mixture_offset,
                jit_compile=True,
            ),
            True,
        ),
        (
            "ukf_apf_k4",
            lambda: compile_c2_per_ancestor_ukf_apf_mixture(
                model=model,
                observations=observations,
                theta_reference=theta,
                particle_count=particle_count,
                seed=seed,
                component_count=4,
                offset=mixture_offset,
                jit_compile=True,
            ),
            True,
        ),
    )
    return (phase2_specs[0], *mixture_specs, *phase2_specs[1:])


def _phase4_defensive_specs(
    *,
    model,
    observations: tf.Tensor,
    theta: tf.Tensor,
    particle_count: int,
    seed: int,
    mixture_offset: float,
    defensive_config: Mapping[str, object],
):
    """Return only the three defensive arms for a bounded repair replay."""

    config = dict(defensive_config)
    return tuple(
        (
            f"ukf_apf_k{component_count}_defensive",
            lambda component_count=component_count: compile_c2_per_ancestor_ukf_apf_defensive_mixture(
                model=model,
                observations=observations,
                theta_reference=theta,
                particle_count=particle_count,
                seed=seed,
                local_component_count=component_count,
                offset=float(mixture_offset),
                nu=float(config["nu"]),
                epsilon_min=float(config["epsilon_min"]),
                epsilon_max=float(config["epsilon_max"]),
                gate_center=float(config["gate_center"]),
                gate_temperature=float(config["gate_temperature"]),
                jit_compile=True,
            ),
            True,
        )
        for component_count in (1, 2, 4)
    )


def _phase4_candidate_specs(
    *,
    model,
    observations: tf.Tensor,
    theta: tf.Tensor,
    proposal_sets: Mapping[str, object],
    particle_count: int,
    seed: int,
    mixture_offset: float,
    defensive_config: Mapping[str, object],
):
    """Return defensive arms followed by the unchanged Phase 3 ladder."""

    defensive_specs = _phase4_defensive_specs(
        model=model,
        observations=observations,
        theta=theta,
        particle_count=particle_count,
        seed=seed,
        mixture_offset=mixture_offset,
        defensive_config=defensive_config,
    )
    phase3_specs = _phase3_candidate_specs(
        model=model,
        observations=observations,
        theta=theta,
        proposal_sets=proposal_sets,
        particle_count=particle_count,
        seed=seed,
        mixture_offset=mixture_offset,
    )
    return (*defensive_specs, *phase3_specs)


def _run_phase2_candidates(
    *,
    output_root: Path,
    model,
    observations: tf.Tensor,
    theta: tf.Tensor,
    proposal_sets: Mapping[str, object],
    particle_count: int,
    branch_count: int,
) -> list[dict[str, object]]:
    """Evaluate all paired proposal families and checkpoint every record."""

    records: list[dict[str, object]] = []
    for branch_index in range(branch_count):
        seed = 97000 + int(particle_count) + 1009 * int(branch_index)
        candidate_specs = _phase2_candidate_specs(
            model=model,
            observations=observations,
            theta=theta,
            proposal_sets=proposal_sets,
            particle_count=particle_count,
            seed=seed,
        )
        for label, compiler, check_xla in candidate_specs:
            branch_started = time.perf_counter()
            record: dict[str, object] = {
                "particle_count": particle_count,
                "branch_index": branch_index,
                "seed": seed,
                "label": label,
                "requested_jit_compile": True,
            }
            try:
                compilation = compiler()
                evaluated = _evaluate_c2_candidate(
                    label=label,
                    compilation=compilation,
                    model=model,
                    theta=theta,
                    check_xla=check_xla,
                )
                evaluated = dict(evaluated)
                evaluated["gpu_output"] = "GPU" in str(
                    evaluated.get("output_device", "")
                ).upper()
                evaluated["branch_wall_seconds"] = time.perf_counter() - branch_started
                evaluated["checks"] = dict(evaluated["checks"])
                evaluated["checks"]["gpu_output"] = bool(evaluated["gpu_output"])
                evaluated["all_checks_pass"] = all(
                    bool(value) for value in evaluated["checks"].values()
                )
                record.update(_jsonable(evaluated))
            except Exception as exc:  # pragma: no cover - retained run evidence
                record.update(
                    {
                        "all_checks_pass": False,
                        "failure_class": "candidate_or_comparator_failure",
                        "error": f"{type(exc).__name__}: {exc}",
                        "branch_wall_seconds": time.perf_counter() - branch_started,
                        "gpu_output": False,
                        "interpretation": "candidate failure retained; inspect before promotion",
                    }
                )
            records.append(record)
            _write_json(output_root / "branch_results.partial.json", records)
    return records


def _run_phase3_candidates(
    *,
    output_root: Path,
    model,
    observations: tf.Tensor,
    theta: tf.Tensor,
    proposal_sets: Mapping[str, object],
    particle_count: int,
    branch_count: int,
    mixture_offset: float,
) -> list[dict[str, object]]:
    """Evaluate Phase 3 families with paired seeds and resumable checkpoints."""

    records: list[dict[str, object]] = []
    for branch_index in range(branch_count):
        seed = PHASE3_CLAIM_SEED_BASE + int(particle_count) + 1009 * int(branch_index)
        candidate_specs = _phase3_candidate_specs(
            model=model,
            observations=observations,
            theta=theta,
            proposal_sets=proposal_sets,
            particle_count=particle_count,
            seed=seed,
            mixture_offset=mixture_offset,
        )
        for label, compiler, check_xla in candidate_specs:
            branch_started = time.perf_counter()
            record: dict[str, object] = {
                "particle_count": particle_count,
                "branch_index": branch_index,
                "seed": seed,
                "label": label,
                "mixture_offset": mixture_offset,
                "requested_jit_compile": True,
            }
            try:
                compilation = compiler()
                evaluated = _evaluate_c2_candidate(
                    label=label,
                    compilation=compilation,
                    model=model,
                    theta=theta,
                    check_xla=check_xla,
                )
                evaluated = dict(evaluated)
                evaluated["gpu_output"] = "GPU" in str(
                    evaluated.get("output_device", "")
                ).upper()
                evaluated["branch_wall_seconds"] = time.perf_counter() - branch_started
                evaluated["checks"] = dict(evaluated["checks"])
                evaluated["checks"]["gpu_output"] = bool(evaluated["gpu_output"])
                evaluated["all_checks_pass"] = all(
                    bool(value) for value in evaluated["checks"].values()
                )
                record.update(_jsonable(evaluated))
            except Exception as exc:  # pragma: no cover - retained run evidence
                record.update(
                    {
                        "all_checks_pass": False,
                        "failure_class": "candidate_or_comparator_failure",
                        "error": f"{type(exc).__name__}: {exc}",
                        "branch_wall_seconds": time.perf_counter() - branch_started,
                        "gpu_output": False,
                        "interpretation": "candidate failure retained; inspect before promotion",
                    }
                )
            records.append(record)
            _write_json(output_root / "branch_results.partial.json", records)
    return records


def _run_phase4_candidates(
    *,
    output_root: Path,
    model,
    observations: tf.Tensor,
    theta: tf.Tensor,
    proposal_sets: Mapping[str, object],
    particle_count: int,
    branch_count: int,
    mixture_offset: float,
    defensive_config: Mapping[str, object],
) -> list[dict[str, object]]:
    """Evaluate Phase 4 defensive and unchanged comparator families."""

    records: list[dict[str, object]] = []
    for branch_index in range(branch_count):
        seed = PHASE4_CLAIM_SEED_BASE + int(particle_count) + 1009 * int(branch_index)
        candidate_specs = _phase4_candidate_specs(
            model=model,
            observations=observations,
            theta=theta,
            proposal_sets=proposal_sets,
            particle_count=particle_count,
            seed=seed,
            mixture_offset=mixture_offset,
            defensive_config=defensive_config,
        )
        for label, compiler, check_xla in candidate_specs:
            branch_started = time.perf_counter()
            record: dict[str, object] = {
                "particle_count": particle_count,
                "branch_index": branch_index,
                "seed": seed,
                "label": label,
                "mixture_offset": mixture_offset,
                "defensive_config": dict(defensive_config),
                "requested_jit_compile": True,
            }
            try:
                compilation = compiler()
                evaluated = _evaluate_c2_candidate(
                    label=label,
                    compilation=compilation,
                    model=model,
                    theta=theta,
                    check_xla=check_xla,
                )
                evaluated = dict(evaluated)
                evaluated["gpu_output"] = "GPU" in str(
                    evaluated.get("output_device", "")
                ).upper()
                evaluated["branch_wall_seconds"] = time.perf_counter() - branch_started
                evaluated["checks"] = dict(evaluated["checks"])
                evaluated["checks"]["gpu_output"] = bool(evaluated["gpu_output"])
                evaluated["all_checks_pass"] = all(
                    bool(value) for value in evaluated["checks"].values()
                )
                record.update(_jsonable(evaluated))
            except Exception as exc:  # pragma: no cover - retained run evidence
                record.update(
                    {
                        "all_checks_pass": False,
                        "failure_class": "candidate_or_comparator_failure",
                        "error": f"{type(exc).__name__}: {exc}",
                        "branch_wall_seconds": time.perf_counter() - branch_started,
                        "gpu_output": False,
                        "interpretation": "candidate failure retained; inspect before promotion",
                    }
                )
            records.append(record)
            _write_json(output_root / "branch_results.partial.json", records)
    return records


def _phase3_candidate_summary(
    records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Aggregate descriptive ESS, per-time ESS, and cost diagnostics."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("label", "unknown")), []).append(record)
    summary: dict[str, object] = {}
    for label, rows in sorted(grouped.items()):
        evaluated = [row for row in rows if "error" not in row]
        valid = [row for row in evaluated if bool(row.get("all_checks_pass", False))]
        if not evaluated:
            summary[label] = {
                "record_count": len(rows),
                "evaluated_count": 0,
                "valid_count": 0,
            }
            continue
        # Descriptive ESS/cost aggregates use every evaluated record; validity
        # is reported separately so a failed check cannot be presented as a
        # valid record.
        ess_values = [float(row["minimum_ess"]) for row in evaluated]
        times = [float(row.get("branch_wall_seconds", 0.0)) for row in evaluated]
        ess_by_time = [
            [float(value) for value in row.get("ess_by_time", ())]
            for row in evaluated
        ]
        time_means = []
        if ess_by_time and all(len(values) == len(ess_by_time[0]) for values in ess_by_time):
            for index in range(len(ess_by_time[0])):
                time_means.append(
                    sum(values[index] for values in ess_by_time) / len(ess_by_time)
                )
        mean_ess = sum(ess_values) / len(ess_values)
        mean_time = sum(times) / len(times)
        summary[label] = {
            "record_count": len(rows),
            "evaluated_count": len(evaluated),
            "valid_count": len(valid),
            "minimum_ess_mean": mean_ess,
            "minimum_ess_min": min(ess_values),
            "minimum_ess_max": max(ess_values),
            "branch_seconds_mean": mean_time,
            "branch_seconds_min": min(times),
            "branch_seconds_max": max(times),
            "cost_seconds_per_mean_minimum_ess": mean_time / mean_ess
            if mean_ess > 0.0
            else float("inf"),
            "per_time_ess_mean": time_means,
        }
    return summary


def _phase2_paired_summary(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Summarize paired branch contrasts without ranking candidates."""

    by_label: dict[str, dict[int, Mapping[str, object]]] = {}
    for record in records:
        label = str(record.get("label", ""))
        branch = int(record.get("branch_index", -1))
        by_label.setdefault(label, {})[branch] = record
    reference = by_label.get("ukf_apf_k1", {})

    def interval(values: Sequence[float]) -> Mapping[str, object]:
        count = len(values)
        mean = sum(values) / count if count else None
        if count > 1:
            centered = [value - mean for value in values]
            standard_deviation = math.sqrt(
                sum(value * value for value in centered) / (count - 1)
            )
            half_width = PHASE2_PAIRED_T_CRITICAL_DF11 * standard_deviation / math.sqrt(count)
            lower = mean - half_width
            upper = mean + half_width
        else:
            standard_deviation = None
            lower = None
            upper = None
        return {
            "n": count,
            "mean": mean,
            "sample_sd": standard_deviation,
            "descriptive_95_t_low": lower,
            "descriptive_95_t_high": upper,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "positive_count": sum(value > 0.0 for value in values),
            "negative_count": sum(value < 0.0 for value in values),
        }

    comparisons: dict[str, object] = {}
    for label, candidate in sorted(by_label.items()):
        if label == "ukf_apf_k1":
            continue
        shared = sorted(
            index
            for index in set(reference).intersection(candidate)
            if "minimum_ess" in reference[index]
            and "minimum_ess" in candidate[index]
            and "log_likelihood" in reference[index]
            and "log_likelihood" in candidate[index]
        )
        ess_delta = [
            float(reference[index]["minimum_ess"])
            - float(candidate[index]["minimum_ess"])
            for index in shared
        ]
        log_likelihood_delta = [
            float(reference[index]["log_likelihood"])
            - float(candidate[index]["log_likelihood"])
            for index in shared
        ]
        comparisons[label] = {
            "branch_indices": shared,
            "ess_k1_minus_comparator": interval(ess_delta),
            "log_likelihood_k1_minus_comparator": interval(log_likelihood_delta),
        }
    return {
        "method": "paired_descriptive_student_t_interval",
        "t_critical": PHASE2_PAIRED_T_CRITICAL_DF11,
        "degrees_of_freedom": 11,
        "interpretation": "intervals are descriptive with twelve paired branches; they do not establish superiority or posterior correctness",
        "comparisons": comparisons,
    }


def _interval_summary(values: Sequence[float]) -> Mapping[str, object]:
    """Return the predeclared twelve-branch descriptive interval."""

    count = len(values)
    mean = sum(values) / count if count else None
    if count > 1:
        centered = [value - mean for value in values]
        standard_deviation = math.sqrt(
            sum(value * value for value in centered) / (count - 1)
        )
        half_width = (
            PHASE2_PAIRED_T_CRITICAL_DF11
            * standard_deviation
            / math.sqrt(count)
        )
        lower = mean - half_width
        upper = mean + half_width
    else:
        standard_deviation = None
        lower = None
        upper = None
    return {
        "n": count,
        "mean": mean,
        "sample_sd": standard_deviation,
        "descriptive_95_t_low": lower,
        "descriptive_95_t_high": upper,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "positive_count": sum(value > 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
    }


def _phase4_promotion_summary(
    records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Evaluate the Phase 4 promotion screen and conditional adversaries.

    The primary screen is the plan's defensive-versus-K=1 paired log ESS
    ratio.  Matched local-arm ratios are reported as a mechanism diagnostic,
    and the cheap-adversary table is evaluated at the declared salient times.
    These diagnostics never alter the exact finite-program value.
    """

    by_label: dict[str, dict[int, Mapping[str, object]]] = {}
    for record in records:
        if "minimum_ess" not in record or "branch_index" not in record:
            continue
        by_label.setdefault(str(record.get("label", "")), {})[
            int(record["branch_index"])
        ] = record

    defensive = (
        "ukf_apf_k1_defensive",
        "ukf_apf_k2_defensive",
        "ukf_apf_k4_defensive",
    )
    local_baselines = {
        "ukf_apf_k1_defensive": "ukf_apf_k1",
        "ukf_apf_k2_defensive": "ukf_apf_k2",
        "ukf_apf_k4_defensive": "ukf_apf_k4",
    }
    cheap = (
        "bootstrap_conditional",
        "transformed_student_nu8",
        "gaussian_hint_marginal",
        "stationary_independence",
    )

    def log_ratio(candidate: str, baseline: str) -> Mapping[str, object]:
        candidate_rows = by_label.get(candidate, {})
        baseline_rows = by_label.get(baseline, {})
        shared = sorted(set(candidate_rows).intersection(baseline_rows))
        values = [
            math.log(
                float(candidate_rows[index]["minimum_ess"])
                / float(baseline_rows[index]["minimum_ess"])
            )
            for index in shared
            if float(candidate_rows[index]["minimum_ess"]) > 0.0
            and float(baseline_rows[index]["minimum_ess"]) > 0.0
        ]
        result = dict(_interval_summary(values))
        result["branch_indices"] = shared
        result["screen_pass"] = bool(
            len(values) == PHASE4_BRANCHES
            and result["descriptive_95_t_low"] is not None
            and result["descriptive_95_t_low"] > 0.0
            and result["positive_count"] >= 10
        )
        return result

    primary = {
        label: log_ratio(label, "ukf_apf_k1") for label in defensive
    }
    matched = {
        label: log_ratio(label, local_baselines[label]) for label in defensive
    }

    conditional: dict[str, object] = {}
    for label in defensive:
        rows = by_label.get(label, {})
        if not rows:
            conditional[label] = {"screen_pass": False, "reason": "no records"}
            continue
        horizon = max(
            len(row.get("ess_by_time", ())) for row in rows.values()
        )
        mean_ess_by_time = [
            sum(float(row["ess_by_time"][time_index]) for row in rows.values())
            / len(rows)
            for time_index in range(horizon)
            if all(time_index < len(row.get("ess_by_time", ())) for row in rows.values())
        ]
        minimum_time = min(
            range(len(mean_ess_by_time)), key=lambda index: mean_ess_by_time[index]
        )
        innovation_by_time = [
            sum(
                float(row["proposal"]["innovation_quadratic_mean_by_time"][index])
                for row in rows.values()
            )
            / len(rows)
            for index in range(
                min(
                    len(row.get("proposal", {}).get("innovation_quadratic_mean_by_time", ()))
                    for row in rows.values()
                )
            )
        ]
        largest_innovation_time = (
            1 + max(range(len(innovation_by_time)), key=lambda index: innovation_by_time[index])
            if innovation_by_time
            else None
        )
        salient = []
        for time_index in (3, 4, minimum_time, largest_innovation_time, horizon - 1):
            if time_index is not None and 0 <= int(time_index) < horizon:
                if int(time_index) not in salient:
                    salient.append(int(time_index))
        adversary_rows: dict[str, object] = {}
        for cheap_label in cheap:
            cheap_rows = by_label.get(cheap_label, {})
            per_time: dict[str, object] = {}
            losses = []
            for time_index in salient:
                shared = sorted(set(rows).intersection(cheap_rows))
                values = [
                    float(rows[index]["ess_by_time"][time_index])
                    - float(cheap_rows[index]["ess_by_time"][time_index])
                    for index in shared
                    if time_index < len(rows[index].get("ess_by_time", ()))
                    and time_index < len(cheap_rows[index].get("ess_by_time", ()))
                ]
                interval = dict(_interval_summary(values))
                interval["loss_mean"] = bool(
                    interval["mean"] is not None and interval["mean"] < 0.0
                )
                if interval["loss_mean"]:
                    losses.append(time_index)
                per_time[str(time_index)] = interval
            adversary_rows[cheap_label] = {
                "per_time": per_time,
                "loss_times": losses,
                "passes": not losses,
            }
        conditional[label] = {
            "minimum_ess_time": minimum_time,
            "largest_innovation_time": largest_innovation_time,
            "salient_time_indices": salient,
            "mean_ess_by_time": mean_ess_by_time,
            "mean_innovation_quadratic_by_time": innovation_by_time,
            "adversaries": adversary_rows,
            "screen_pass": all(
                bool(row["passes"]) for row in adversary_rows.values()
            ),
        }

    return {
        "primary_baseline": "ukf_apf_k1",
        "primary": primary,
        "matched_local_diagnostic": matched,
        "conditional_heuristic": conditional,
        "promotion_pass": bool(
            all(bool(value["screen_pass"]) for value in primary.values())
            and all(bool(value.get("screen_pass", False)) for value in conditional.values())
        ),
        "interpretation": (
            "paired t intervals are descriptive with twelve branches; the "
            "heuristic table is a promotion veto, not a tuning target"
        ),
    }


def _phase2_result_markdown(payload: Mapping[str, object]) -> str:
    """Render a Phase 2 result without promoting descriptive metrics."""

    records = payload.get("records", ())
    serious = payload.get("campaign_kind") == "serious"
    title = (
        "# C2 Mixture-UKF/APF Phase 2 Serious N=8192 Result"
        if serious
        else "# C2 Mixture-UKF/APF Phase 2 Entry Result"
    )
    rows = payload.get("rows", (PHASE2_FIT_ROWS,))
    branches = payload.get("branches", 1)
    description = (
        f"This is the bounded serious GPU expansion: one fresh retained-TT fit and "
        f"{branches} paired branches at rows {rows} for each declared proposal family."
        if serious
        else "This is the bounded GPU entry pilot: one fresh retained-TT fit and one paired branch at N=8192 for each declared proposal family."
    )
    lines = [
        title,
        "",
        f"Phase: `{payload.get('phase', PHASE2_ID)}`",
        f"Status: `{payload.get('status', 'unknown')}`",
        f"Continuation: `{payload.get('continuation', 'unknown')}`",
        "",
        description,
        "The exact",
        "C2 target and analytical score are evaluated by the shared frozen finite",
        "program. ESS and log-likelihood differences are descriptive and do not",
        "establish posterior correctness, unbiasedness, superiority, or readiness.",
        "",
        "## Decision",
        "",
        "| Item | Status |",
        "| --- | --- |",
        f"| K=1 validity checks | `{payload.get('required_k1_checks_pass', False)}` |",
        f"| Comparator records complete | `{payload.get('comparator_records_complete', False)}` |",
        f"| Fresh proposal fit | `{payload.get('proposal_fit_status', 'unknown')}` |",
        f"| Failure class | `{payload.get('failure_class', 'unknown')}` |",
        f"| Fit seconds | `{payload.get('fit_seconds', 'unknown')}` |",
        f"| Budget/continuation | `{payload.get('budget_remaining', 'unknown')}` |",
        "",
        "## Candidate records",
        "",
        "| Candidate | N | log L | min ESS | score FD max | GPU output | checks |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| `{record.get('label', 'unknown')}` | "
            f"{record.get('particle_count', 'error')} | "
            f"{record.get('log_likelihood', 'error')} | "
            f"{record.get('minimum_ess', 'error')} | "
            f"{record.get('score_max_abs_error', 'error')} | "
            f"{record.get('gpu_output', 'unknown')} | "
            f"`{record.get('all_checks_pass', False)}` |"
        )
    lines.extend(
        [
            "",
            "## Inference status",
            "",
            "| Evidence class | Status |",
            "| --- | --- |",
            "| Hard validity vetoes | required K=1 checks and GPU/XLA provenance |",
            "| Statistically supported ranking | not attempted; branch metrics remain descriptive |",
            "| Descriptive differences | recorded in `result.json` only |",
            "| Default readiness | not evaluated |",
            "| Next evidence | review branch uncertainty and cost before any larger row |",
            "",
            "The fresh snapshots and their hashes are in `proposal_manifest.json`.",
            "Any candidate error is retained as a candidate/comparator failure; it",
            "is not silently relabeled as a generic UKF failure.",
            "",
            f"Failure detail: `{payload.get('error', '')}`",
        ]
    )
    summary = payload.get("paired_uncertainty")
    if isinstance(summary, Mapping) and summary.get("comparisons"):
        lines.extend(
            [
                "",
                "## Paired descriptive contrasts",
                "",
                "Intervals below are paired descriptive t intervals (df=11); they are",
                "not superiority or posterior-correctness evidence.",
                "",
                "| Comparator | ESS K1-minus-comparator mean [95%] | log L K1-minus-comparator mean [95%] |",
                "| --- | --- | --- |",
            ]
        )
        for label, comparison in summary["comparisons"].items():
            ess = comparison["ess_k1_minus_comparator"]
            log_likelihood = comparison["log_likelihood_k1_minus_comparator"]
            def fmt(value: object) -> str:
                return "n/a" if value is None else f"{float(value):.6g}"

            lines.append(
                f"| `{label}` | {fmt(ess['mean'])} "
                f"[{fmt(ess['descriptive_95_t_low'])}, {fmt(ess['descriptive_95_t_high'])}] | "
                f"{fmt(log_likelihood['mean'])} "
                f"[{fmt(log_likelihood['descriptive_95_t_low'])}, {fmt(log_likelihood['descriptive_95_t_high'])}] |"
            )
    if serious:
        lines.insert(
            lines.index("## Inference status"),
            f"Expected raw branch records: `{payload.get('expected_record_count', 'unknown')}`; "
            f"actual: `{payload.get('actual_record_count', 'unknown')}`. Branch timing and "
            "paired uncertainty summaries are in `result.json`.",
        )
    return "\n".join(lines) + "\n"


def run_phase2_entry(args: argparse.Namespace) -> Path:
    """Run the bounded one-branch GPU pilot before opening the serious ladder."""

    fixture = _load_c2_fixture()
    rows = _parse_rows(args.rows)
    if rows != (PHASE2_FIT_ROWS,):
        raise ValueError(
            f"Phase 2 entry is fixed to one row count N={PHASE2_FIT_ROWS}; got {rows}"
        )
    if int(args.branches) != 1:
        raise ValueError("Phase 2 entry is fixed to one paired branch")
    if not bool(args.jit_compile):
        raise ValueError("Phase 2 entry requires XLA; --no-jit-compile is not in scope")

    output = _make_output_root(args.output_root)
    records: list[dict[str, object]] = []
    gpu_info: Mapping[str, object] = {}
    prepared: Mapping[str, object] = {}
    fit_seconds: float | None = None
    global_error = ""
    started = time.perf_counter()

    try:
        # Device policy is established before constructing model tensors or a
        # graph. The context also makes the candidate output-device check
        # meaningful for this entry pilot.
        gpu_info = _configure_phase2_gpu()
        _load_algorithm_modules()
        with tf.device("/GPU:0"):
            model, theta, observations = _c2_model_and_inputs(fixture)
            prepared = _prepare_phase2_proposals(
                fixture=fixture,
                model=model,
                theta=theta,
                observations=observations,
                output_root=output,
            )
            fit_seconds = float(prepared["fit_seconds"])
            proposal_sets = {
                "retained_tt": prepared["tt_proposals"],
                "gaussian_hint_marginal": prepared["hint_proposals"],
                "stationary_independence": prepared["stationary_proposals"],
            }
            records.extend(
                _run_phase2_candidates(
                    output_root=output,
                    model=model,
                    observations=observations,
                    theta=theta,
                    proposal_sets=proposal_sets,
                    particle_count=PHASE2_FIT_ROWS,
                    branch_count=1,
                )
            )
    except Exception as exc:  # pragma: no cover - retained setup/fit evidence
        global_error = f"{type(exc).__name__}: {exc}"

    required_records = [
        record for record in records if record.get("label") == "ukf_apf_k1"
    ]
    comparator_records = [
        record for record in records if record.get("label") != "ukf_apf_k1"
    ]
    required_k1_checks_pass = bool(required_records) and all(
        bool(record.get("all_checks_pass", False)) for record in required_records
    )
    comparator_complete = len(comparator_records) == len(PHASE2_FAMILIES) - 1 and all(
        "error" not in record for record in comparator_records
    )
    if global_error:
        status = "PHASE2_ENTRY_ATTEMPT_FAILED"
        continuation = "REPAIR_AND_RETRY_WITHIN_PHASE2_ENTRY_BUDGET"
        failure_class = "setup_or_fit_infrastructure_failure"
        repair = "preserve this attempt, classify the setup/fit failure, and retry with the unchanged entry contract"
    elif required_k1_checks_pass:
        status = (
            "PASS_PHASE2_ENTRY"
            if comparator_complete
            else "PASS_PHASE2_ENTRY_WITH_COMPARATOR_DIAGNOSTIC"
        )
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "none" if comparator_complete else "candidate_failure"
        repair = (
            "none; all six candidate records completed"
            if comparator_complete
            else "retain comparator failures and refresh Phase 2 cost before expansion"
        )
    else:
        status = "VETO_PHASE2_ENTRY"
        continuation = "REPAIR_REQUIRED_BEFORE_PHASE2_EXPANSION"
        failure_class = "implementation_or_numerical_validity"
        repair = "repair the required K=1 validity or GPU/XLA check before expansion"

    source_paths = {
        "plan": PLAN_PATH,
        "driver": Path(__file__).resolve(),
        "generic_kernel": MODULE_PATH,
        "c2_adapter": C2_ADAPTER_PATH,
        "c2_model": C2_MODEL_PATH,
        "exact_evaluator": C2_EXACT_PATH,
    }
    source_manifest = {
        key: {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256_file(path),
        }
        for key, path in source_paths.items()
    }
    payload = {
        "schema_version": PHASE2_RESULT_SCHEMA,
        "phase": PHASE2_ID,
        "status": status,
        "continuation": continuation,
        "failure_class": failure_class,
        "repair": repair,
        "error": global_error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "proposal_fit_status": "complete" if prepared else "not_complete",
        "fit_seconds": fit_seconds,
        "required_k1_checks_pass": required_k1_checks_pass,
        "comparator_records_complete": comparator_complete,
        "rows": rows,
        "branches": int(args.branches),
        "seed_policy": "one paired seed shared across all six frozen proposal families",
        "gpu": gpu_info,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
            "physical_devices": [device.name for device in tf.config.list_physical_devices()],
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "execution_lane": "trusted_gpu_xla_entry_pilot",
            "jit_compile": True,
        },
        "fixture": {
            "path": str(C2_FIXTURE_PATH.relative_to(ROOT)),
            "sha256": _sha256_file(C2_FIXTURE_PATH),
            "schema_id": fixture["schema_id"],
            "horizon": int(fixture["horizon"]),
            "state_dimension": int(fixture["state_dimension"]),
        },
        "sources": source_manifest,
        "workspace": _workspace_manifest(),
        "records": records,
        "paired_uncertainty": _phase2_paired_summary(records),
        "nonclaims": (
            "no posterior correctness claim",
            "no unbiased likelihood claim",
            "no statistical superiority claim from one branch",
            "no production or default-readiness claim",
        ),
        "budget_remaining": "entry pilot consumed one bounded GPU attempt; refresh cost before any expansion",
    }
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "c2_mixture_ukf_apf_phase2_entry_manifest_v1",
            "phase": PHASE2_ID,
            "command": " ".join(sys.argv),
            "plan_sha256": source_manifest["plan"]["sha256"],
            "workspace": payload["workspace"],
            "environment": payload["environment"],
            "gpu": gpu_info,
            "fixture": payload["fixture"],
            "rows": rows,
            "branches": int(args.branches),
            "proposal_families": PHASE2_FAMILIES,
        },
    )
    _write_json(output / "result.json", payload)
    _write_json(output / "branch_results.json", records)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_phase2_result_markdown(payload), encoding="utf-8")
    return output


def run_phase2_serious(args: argparse.Namespace) -> Path:
    """Run the first serious expansion: twelve paired branches at N=8192."""

    fixture = _load_c2_fixture()
    rows = _parse_rows(args.rows)
    if rows != (PHASE2_FIT_ROWS,):
        raise ValueError(
            "the first serious expansion is fixed to one row count "
            f"N={PHASE2_FIT_ROWS}; larger rows require a refreshed plan"
        )
    branches = int(args.branches)
    if branches != PHASE2_SERIOUS_BRANCHES:
        raise ValueError(
            f"the first serious expansion is fixed to {PHASE2_SERIOUS_BRANCHES} paired branches"
        )
    if not bool(args.jit_compile):
        raise ValueError("the serious expansion requires XLA; --no-jit-compile is not in scope")

    output = _make_output_root(args.output_root)
    records: list[dict[str, object]] = []
    gpu_info: Mapping[str, object] = {}
    prepared: Mapping[str, object] = {}
    fit_seconds: float | None = None
    global_error = ""
    started = time.perf_counter()

    try:
        # Configure the device before loading high-dimensional modules or
        # constructing model tensors. This is a hard provenance requirement.
        gpu_info = _configure_phase2_gpu()
        _load_algorithm_modules()
        with tf.device("/GPU:0"):
            model, theta, observations = _c2_model_and_inputs(fixture)
            prepared = _prepare_phase2_proposals(
                fixture=fixture,
                model=model,
                theta=theta,
                observations=observations,
                output_root=output,
            )
            fit_seconds = float(prepared["fit_seconds"])
            proposal_sets = {
                "retained_tt": prepared["tt_proposals"],
                "gaussian_hint_marginal": prepared["hint_proposals"],
                "stationary_independence": prepared["stationary_proposals"],
            }
            records.extend(
                _run_phase2_candidates(
                    output_root=output,
                    model=model,
                    observations=observations,
                    theta=theta,
                    proposal_sets=proposal_sets,
                    particle_count=PHASE2_FIT_ROWS,
                    branch_count=branches,
                )
            )
    except Exception as exc:  # pragma: no cover - retained setup/fit evidence
        global_error = f"{type(exc).__name__}: {exc}"

    required_records = [
        record for record in records if record.get("label") == "ukf_apf_k1"
    ]
    comparator_records = [
        record for record in records if record.get("label") != "ukf_apf_k1"
    ]
    expected_required = branches
    expected_total = branches * len(PHASE2_FAMILIES)
    required_k1_checks_pass = (
        len(required_records) == expected_required
        and all(bool(record.get("all_checks_pass", False)) for record in required_records)
    )
    comparator_complete = (
        len(comparator_records) == expected_total - expected_required
        and all("error" not in record for record in comparator_records)
    )
    all_candidate_checks_pass = (
        len(records) == expected_total
        and all(bool(record.get("all_checks_pass", False)) for record in records)
    )
    if global_error:
        status = "PHASE2_SERIOUS_ATTEMPT_FAILED"
        continuation = "REPAIR_AND_RETRY_WITHIN_PHASE2_SERIOUS_BUDGET"
        failure_class = "setup_or_fit_infrastructure_failure"
        repair = "preserve this attempt, classify the setup/fit failure, and retry the unchanged N=8192 contract"
    elif not required_k1_checks_pass:
        status = "VETO_PHASE2_SERIOUS"
        continuation = "REPAIR_REQUIRED_BEFORE_LARGER_ROW"
        failure_class = "implementation_or_numerical_validity"
        repair = "repair the required K=1 validity, branch coverage, or GPU/XLA check before expanding rows"
    elif not all_candidate_checks_pass:
        status = "PASS_PHASE2_SERIOUS_WITH_COMPARATOR_DIAGNOSTIC"
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "candidate_failure"
        repair = "retain comparator diagnostics; do not relabel them as a K=1 failure"
    else:
        status = "PASS_PHASE2_SERIOUS_N8192"
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "none"
        repair = "none; all paired records passed the declared finite-program checks"

    branch_seconds = sum(
        float(record.get("branch_wall_seconds", 0.0))
        for record in records
        if isinstance(record.get("branch_wall_seconds"), (int, float))
    )
    source_paths = {
        "plan": PLAN_PATH,
        "driver": Path(__file__).resolve(),
        "generic_kernel": MODULE_PATH,
        "c2_adapter": C2_ADAPTER_PATH,
        "c2_model": C2_MODEL_PATH,
        "exact_evaluator": C2_EXACT_PATH,
    }
    source_manifest = {
        key: {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256_file(path),
        }
        for key, path in source_paths.items()
    }
    payload = {
        "schema_version": PHASE2_SERIOUS_RESULT_SCHEMA,
        "phase": PHASE2_SERIOUS_ID,
        "campaign_kind": "serious",
        "status": status,
        "continuation": continuation,
        "failure_class": failure_class,
        "repair": repair,
        "error": global_error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "proposal_fit_status": "complete" if prepared else "not_complete",
        "fit_seconds": fit_seconds,
        "branch_seconds_sum": branch_seconds,
        "required_k1_checks_pass": required_k1_checks_pass,
        "comparator_records_complete": comparator_complete,
        "all_candidate_checks_pass": all_candidate_checks_pass,
        "expected_record_count": expected_total,
        "actual_record_count": len(records),
        "rows": rows,
        "branches": branches,
        "seed_policy": "paired seed per branch shared across all six frozen proposal families",
        "gpu": gpu_info,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
            "physical_devices": [device.name for device in tf.config.list_physical_devices()],
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "execution_lane": "trusted_gpu_xla_serious_n8192",
            "jit_compile": True,
        },
        "fixture": {
            "path": str(C2_FIXTURE_PATH.relative_to(ROOT)),
            "sha256": _sha256_file(C2_FIXTURE_PATH),
            "schema_id": fixture["schema_id"],
            "horizon": int(fixture["horizon"]),
            "state_dimension": int(fixture["state_dimension"]),
        },
        "sources": source_manifest,
        "workspace": _workspace_manifest(),
        "records": records,
        "paired_uncertainty": _phase2_paired_summary(records),
        "nonclaims": (
            "no posterior correctness claim",
            "no unbiased likelihood claim",
            "no statistical superiority claim from twelve branches without a declared uncertainty analysis",
            "no production or default-readiness claim",
        ),
        "budget_remaining": "larger rows remain closed pending review of this run's cost, memory, and validity record",
    }
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "c2_mixture_ukf_apf_phase2_serious_manifest_v1",
            "phase": PHASE2_SERIOUS_ID,
            "command": " ".join(sys.argv),
            "plan_sha256": source_manifest["plan"]["sha256"],
            "workspace": payload["workspace"],
            "environment": payload["environment"],
            "gpu": gpu_info,
            "fixture": payload["fixture"],
            "rows": rows,
            "branches": branches,
            "proposal_families": PHASE2_FAMILIES,
            "expected_record_count": expected_total,
        },
    )
    _write_json(output / "branch_results.json", records)
    _write_json(output / "result.json", payload)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_phase2_result_markdown(payload), encoding="utf-8")
    return output


def _phase3_result_markdown(payload: Mapping[str, object]) -> str:
    """Render the Phase 3 close record without promoting descriptive metrics."""

    calibration = payload.get("calibration", {})
    records = payload.get("records", ())
    lines = [
        "# C2 Mixture-UKF/APF Phase 3 Fixed-Split Result",
        "",
        f"Phase: {payload.get('phase', PHASE3_ID)}",
        f"Status: {payload.get('status', 'unknown')}",
        f"Continuation: {payload.get('continuation', 'unknown')}",
        "",
        "This run calibrates a dimensionless Cholesky-column split on an",
        "independent model-generated observation bank, freezes the selected",
        "offset, and evaluates K=1, K=2, K=4, and the declared comparator",
        "families on the untouched C2 observation sequence. The exact target,",
        "complete proposal denominator, and frozen analytical score are shared",
        "across all records. ESS and log-likelihood differences are descriptive.",
        "",
        "## Decision",
        "",
        "| Item | Status |",
        "| --- | --- |",
        f"| Calibration checks | {payload.get('calibration_pass', False)} |",
        f"| Selected offset | {payload.get('selected_offset', 'none')} |",
        f"| Required K=1/K=2/K=4 checks | {payload.get('required_mixture_checks_pass', False)} |",
        f"| Records complete | {payload.get('records_complete', False)} |",
        f"| All candidate checks | {payload.get('all_candidate_checks_pass', False)} |",
        f"| Failure class | {payload.get('failure_class', 'unknown')} |",
        f"| Fit seconds | {payload.get('fit_seconds', 'unknown')} |",
        f"| Branch seconds (sum) | {payload.get('branch_seconds_sum', 'unknown')} |",
        "",
        "## Calibration",
        "",
        "| Offset | K | Minimum component eigenvalue | Moment error | Label error | Density error | Pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    calibration_rows = (
        calibration.get("rows", ()) if isinstance(calibration, Mapping) else ()
    )
    for row in calibration_rows:
        lines.append(
            f"| {row.get('offset', 'error')} | {row.get('component_count', 'error')} | "
            f"{row.get('component_minimum_eigenvalue', 'error')} | "
            f"{row.get('moment_recomposition_max_abs', 'error')} | "
            f"{row.get('label_permutation_max_abs', 'error')} | "
            f"{row.get('proposal_density_recomposition_max_abs', 'error')} | "
            f"{row.get('all_checks_pass', False)} |"
        )
    lines.extend(
        [
            "",
            "## Candidate records",
            "",
            (
                "This repair artifact lists the 36 replayed defensive records; "
                "the 96 unchanged comparator records are inherited from the "
                "immutable parent attempt and are included in aggregate and "
                "promotion calculations by hash."
                if payload.get("campaign_kind") == "phase4_diagnostic_repair"
                else "The table lists every evaluated claim record."
            ),
            "",
            "| Candidate | N | log L | min ESS | branch seconds | score FD max | GPU | checks |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            f"| {record.get('label', 'unknown')} | "
            f"{record.get('particle_count', 'error')} | "
            f"{record.get('log_likelihood', 'error')} | "
            f"{record.get('minimum_ess', 'error')} | "
            f"{record.get('branch_wall_seconds', 'error')} | "
            f"{record.get('score_max_abs_error', 'error')} | "
            f"{record.get('gpu_output', 'unknown')} | "
            f"{record.get('all_checks_pass', False)} |"
        )
    lines.extend(
        [
            "",
            "## Candidate summary",
            "",
            "These aggregates are descriptive and do not establish a statistical",
            "ranking or posterior correctness.",
            "",
            "| Candidate | Valid records | Mean min ESS | Min/max min ESS | Mean seconds | Seconds per mean min ESS |",
            "| --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for label, summary in payload.get("candidate_summary", {}).items():
        lines.append(
            f"| {label} | {summary.get('valid_count', 0)} | "
            f"{summary.get('minimum_ess_mean', 'n/a')} | "
            f"{summary.get('minimum_ess_min', 'n/a')} / {summary.get('minimum_ess_max', 'n/a')} | "
            f"{summary.get('branch_seconds_mean', 'n/a')} | "
            f"{summary.get('cost_seconds_per_mean_minimum_ess', 'n/a')} |"
        )
    paired = payload.get("paired_uncertainty", {})
    if isinstance(paired, Mapping) and paired.get("comparisons"):
        lines.extend(
            [
                "",
                "## Paired descriptive contrasts",
                "",
                "Intervals are paired descriptive t intervals with twelve branches;",
                "they are not superiority or posterior-correctness evidence.",
                "",
                "| Comparator | ESS K1-minus-comparator mean [95%] | log L K1-minus-comparator mean [95%] |",
                "| --- | --- | --- |",
            ]
        )
        for label, comparison in paired["comparisons"].items():
            ess = comparison["ess_k1_minus_comparator"]
            log_likelihood = comparison["log_likelihood_k1_minus_comparator"]

            def fmt(value: object) -> str:
                return "n/a" if value is None else f"{float(value):.6g}"

            lines.append(
                f"| {label} | {fmt(ess['mean'])} "
                f"[{fmt(ess['descriptive_95_t_low'])}, {fmt(ess['descriptive_95_t_high'])}] | "
                f"{fmt(log_likelihood['mean'])} "
                f"[{fmt(log_likelihood['descriptive_95_t_low'])}, {fmt(log_likelihood['descriptive_95_t_high'])}] |"
            )
    lines.extend(
        [
            "",
            "## Inference status",
            "",
            "| Evidence class | Status |",
            "| --- | --- |",
            "| Hard validity vetoes | calibration and required K=1/K=2/K=4 finite-program checks |",
            "| Statistically supported ranking | not established; twelve-branch contrasts are descriptive |",
            "| Descriptive differences | candidate summary and raw records above |",
            "| Default readiness | not evaluated |",
            "| Next evidence | review this cost/ESS result before Phase 4 defensive gating |",
            "",
            f"Failure detail: {payload.get('error', '')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _phase4_result_markdown(payload: Mapping[str, object]) -> str:
    """Render the Phase 4 close record without promoting descriptive metrics."""

    calibration = payload.get("calibration", {})
    lines = [
        "# C2 Mixture-UKF/APF Phase 4 Smooth Student-Defensive Result",
        "",
        f"Phase: `{payload.get('phase', PHASE4_ID)}`",
        f"Status: `{payload.get('status', 'unknown')}`",
        f"Continuation: `{payload.get('continuation', 'unknown')}`",
        "",
        "The run selects a smooth innovation-based Student defensive fraction on",
        "an independent calibration bank, freezes it, and evaluates the complete",
        "Gaussian-plus-Student proposal density in the exact C2 finite program.",
        "ESS and log-likelihood differences are descriptive.",
        "",
        "## Decision",
        "",
        "| Item | Status |",
        "| --- | --- |",
        f"| Calibration checks | {payload.get('calibration_pass', False)} |",
        f"| Selected configuration | {payload.get('selected_config_id', 'none')} |",
        f"| Required defensive checks | {payload.get('required_defensive_checks_pass', payload.get('replay_valid', False))} |",
        f"| Replay branch/program parity | {payload.get('replay_parity_pass', 'n/a')} |",
        f"| Parent source/fixture/control identity | {payload.get('parent_source_consistency', 'n/a')}/{payload.get('parent_fixture_consistency', 'n/a')}/{payload.get('parent_control_consistency', 'n/a')} |",
        f"| Records complete | {payload.get('records_complete', False)} |",
        f"| All candidate checks | {payload.get('all_candidate_checks_pass', False)} |",
        f"| Failure class | {payload.get('failure_class', 'unknown')} |",
        f"| Fit seconds | {payload.get('fit_seconds', 'unknown')} |",
        f"| Branch seconds (sum) | {payload.get('branch_seconds_sum', 'unknown')} |",
        "",
        "## Calibration",
        "",
        "| Config | nu | eps min | eps max | K | min ESS | min eig | epsilon spread | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    calibration_rows = calibration.get("rows", ()) if isinstance(calibration, Mapping) else ()
    for row in calibration_rows:
        lines.append(
            f"| {row.get('config_id', 'error')} | {row.get('nu', 'error')} | "
            f"{row.get('epsilon_min', 'error')} | {row.get('epsilon_max', 'error')} | "
            f"{row.get('local_component_count', 'error')} | {row.get('minimum_ess', 'error')} | "
            f"{row.get('component_minimum_eigenvalue', 'error')} | "
            f"{row.get('epsilon_spread_max', 'error')} | {row.get('all_checks_pass', False)} |"
        )
    lines.extend(
        [
            "",
            "## Candidate summary",
            "",
            "These aggregates are descriptive and do not establish a statistical",
            "ranking or posterior correctness.",
            "",
            "| Candidate | Evaluated | Valid | Mean min ESS | Min/max min ESS | Mean seconds |",
            "| --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for label, summary in payload.get("candidate_summary", {}).items():
        lines.append(
            f"| {label} | {summary.get('evaluated_count', summary.get('valid_count', 0))} | "
            f"{summary.get('valid_count', 0)} | "
            f"{summary.get('minimum_ess_mean', 'n/a')} | "
            f"{summary.get('minimum_ess_min', 'n/a')} / {summary.get('minimum_ess_max', 'n/a')} | "
            f"{summary.get('branch_seconds_mean', 'n/a')} |"
        )
    paired = payload.get("paired_uncertainty", {})
    if isinstance(paired, Mapping) and paired.get("comparisons"):
        lines.extend(
            [
                "",
                "## Paired descriptive contrasts",
                "",
                "The reference is K=1; intervals use twelve paired branches and are",
                "descriptive rather than superiority or posterior-correctness evidence.",
                "",
                "| Comparator | ESS K1-minus-comparator mean [95%] | log L mean [95%] |",
                "| --- | --- | --- |",
            ]
        )
        for label, comparison in paired["comparisons"].items():
            ess = comparison["ess_k1_minus_comparator"]
            log_likelihood = comparison["log_likelihood_k1_minus_comparator"]
            lines.append(
                f"| {label} | {ess.get('mean', 'n/a')} "
                f"[{ess.get('descriptive_95_t_low', 'n/a')}, {ess.get('descriptive_95_t_high', 'n/a')}] | "
                f"{log_likelihood.get('mean', 'n/a')} "
                f"[{log_likelihood.get('descriptive_95_t_low', 'n/a')}, {log_likelihood.get('descriptive_95_t_high', 'n/a')}] |"
            )
    promotion = payload.get("promotion", {})
    if isinstance(promotion, Mapping) and promotion:
        lines.extend(
            [
                "",
                "## Promotion and heuristic screen",
                "",
                "The primary screen is the predeclared paired log minimum-ESS ratio "
                "against K=1. Intervals are descriptive with twelve branches. "
                "The conditional cheap-adversary screen is a promotion veto, not "
                "a tuning objective.",
                "",
                "| Defensive arm | log ESS ratio mean [95%] | positive contrasts | primary screen | heuristic screen |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        conditional = promotion.get("conditional_heuristic", {})
        for label, comparison in promotion.get("primary", {}).items():
            interval = comparison
            heuristic_pass = conditional.get(label, {}).get("screen_pass", False)
            lines.append(
                f"| {label} | {interval.get('mean', 'n/a')} "
                f"[{interval.get('descriptive_95_t_low', 'n/a')}, "
                f"{interval.get('descriptive_95_t_high', 'n/a')}] | "
                f"{interval.get('positive_count', 0)}/12 | "
                f"{interval.get('screen_pass', False)} | {heuristic_pass} |"
            )
        lines.extend(
            [
                "",
                "Conditional salient-time details (zero-based time indices) are "
                "stored in `result.json`; a negative paired mean against any cheap "
                "adversary is a promotion veto.",
            ]
        )
    lines.extend(
        [
            "",
            "## Inference status",
            "",
            "| Evidence class | Status |",
            "| --- | --- |",
            f"| Hard validity vetoes | {payload.get('validity_veto_summary', 'see records')} |",
            "| Statistically supported ranking | not established; twelve-branch contrasts are descriptive |",
            "| Descriptive differences | candidate summary and raw JSON records above |",
            "| Default readiness | not evaluated |",
            "| Next evidence | use the refreshed Phase 5 recursive-map plan; do not promote this defensive arm |",
            "",
            f"Failure detail: {payload.get('error', '')}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_phase3(args: argparse.Namespace) -> Path:
    """Run the fixed-topology K=2/K=4 repair on the unchanged C2 contract."""

    fixture = _load_c2_fixture()
    rows = _parse_rows(args.rows)
    if rows != (PHASE2_FIT_ROWS,):
        raise ValueError(
            f"Phase 3 is fixed to one row count N={PHASE2_FIT_ROWS}; got {rows}"
        )
    branches = int(args.branches)
    if branches != PHASE3_BRANCHES:
        raise ValueError(
            f"Phase 3 is fixed to {PHASE3_BRANCHES} paired branches; got {branches}"
        )
    if not bool(args.jit_compile):
        raise ValueError("Phase 3 requires XLA; --no-jit-compile is not in scope")

    output = _make_output_root(args.output_root)
    records: list[dict[str, object]] = []
    gpu_info: Mapping[str, object] = {}
    prepared: Mapping[str, object] = {}
    calibration: Mapping[str, object] = {}
    fit_seconds: float | None = None
    global_error = ""
    started = time.perf_counter()

    try:
        gpu_info = _configure_phase2_gpu()
        _load_algorithm_modules()
        with tf.device("/GPU:0"):
            model, theta, observations = _c2_model_and_inputs(fixture)
            calibration = _phase3_calibrate_offset(
                model=model, theta=theta, output_root=output
            )
            selected_offset = float(calibration["selected_offset"])
            prepared = _prepare_phase2_proposals(
                fixture=fixture,
                model=model,
                theta=theta,
                observations=observations,
                output_root=output,
            )
            fit_seconds = float(prepared["fit_seconds"])
            proposal_sets = {
                "retained_tt": prepared["tt_proposals"],
                "gaussian_hint_marginal": prepared["hint_proposals"],
                "stationary_independence": prepared["stationary_proposals"],
            }
            records.extend(
                _run_phase3_candidates(
                    output_root=output,
                    model=model,
                    observations=observations,
                    theta=theta,
                    proposal_sets=proposal_sets,
                    particle_count=PHASE2_FIT_ROWS,
                    branch_count=branches,
                    mixture_offset=selected_offset,
                )
            )
    except Exception as exc:  # pragma: no cover - retained setup/fit evidence
        global_error = f"{type(exc).__name__}: {exc}"

    selected_offset = calibration.get("selected_offset") if calibration else None
    calibration_pass = selected_offset is not None and all(
        bool(row.get("all_checks_pass", False))
        for row in calibration.get("rows", ())
        if float(row.get("offset", -1.0)) == float(selected_offset)
    )
    required_labels = {"ukf_apf_k1", "ukf_apf_k2", "ukf_apf_k4"}
    required_records = [
        record for record in records if record.get("label") in required_labels
    ]
    required_mixture_checks_pass = (
        len(required_records) == branches * len(required_labels)
        and all(bool(record.get("all_checks_pass", False)) for record in required_records)
    )
    expected_total = branches * len(PHASE3_FAMILIES)
    records_complete = len(records) == expected_total
    all_candidate_checks_pass = records_complete and all(
        bool(record.get("all_checks_pass", False)) for record in records
    )
    promotion = _phase4_promotion_summary(records) if records else {}
    if global_error:
        status = "PHASE3_ATTEMPT_FAILED"
        continuation = "REPAIR_AND_RETRY_WITHIN_PHASE3_BUDGET"
        failure_class = "setup_or_fit_infrastructure_failure"
        repair = "preserve this attempt, classify the setup/calibration/fit failure, and retry the unchanged Phase 3 contract"
    elif not calibration_pass:
        status = "VETO_PHASE3_CALIBRATION"
        continuation = "REPAIR_REQUIRED_BEFORE_PHASE3_CLAIM"
        failure_class = "tuning_or_numerical_validity"
        repair = "repair or recalibrate the predeclared offset on the independent bank before claim replay"
    elif not required_mixture_checks_pass:
        status = "VETO_PHASE3_REQUIRED_MIXTURE"
        continuation = "REPAIR_REQUIRED_BEFORE_PHASE4"
        failure_class = "implementation_or_numerical_validity"
        repair = "repair the required K=1/K=2/K=4 finite-program, score, denominator, or call-chain check"
    elif not records_complete:
        status = "PHASE3_INCOMPLETE"
        continuation = "REPAIR_AND_RETRY_WITHIN_PHASE3_BUDGET"
        failure_class = "infrastructure_or_harness"
        repair = "preserve partial records and retry missing branches under the unchanged contract"
    elif not all_candidate_checks_pass:
        status = "PASS_PHASE3_WITH_CANDIDATE_DIAGNOSTIC"
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "candidate_failure"
        repair = "retain comparator/candidate failures; refresh Phase 4 with the valid required arms"
    else:
        status = "PASS_PHASE3_FIXED_SPLITS"
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "none"
        repair = "none; all required and comparator finite-program checks passed"

    branch_seconds = sum(
        float(record.get("branch_wall_seconds", 0.0))
        for record in records
        if isinstance(record.get("branch_wall_seconds"), (int, float))
    )
    source_paths = {
        "plan": PLAN_PATH,
        "driver": Path(__file__).resolve(),
        "generic_kernel": MODULE_PATH,
        "c2_adapter": C2_ADAPTER_PATH,
        "c2_model": C2_MODEL_PATH,
        "exact_evaluator": C2_EXACT_PATH,
    }
    source_manifest = {
        key: {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256_file(path),
        }
        for key, path in source_paths.items()
    }
    payload: dict[str, object] = {
        "schema_version": PHASE3_RESULT_SCHEMA,
        "phase": PHASE3_ID,
        "campaign_kind": "phase3",
        "status": status,
        "continuation": continuation,
        "failure_class": failure_class,
        "repair": repair,
        "error": global_error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "proposal_fit_status": "complete" if prepared else "not_complete",
        "fit_seconds": fit_seconds,
        "branch_seconds_sum": branch_seconds,
        "calibration_pass": calibration_pass,
        "selected_offset": selected_offset,
        "required_mixture_checks_pass": required_mixture_checks_pass,
        "records_complete": records_complete,
        "all_candidate_checks_pass": all_candidate_checks_pass,
        "expected_record_count": expected_total,
        "actual_record_count": len(records),
        "rows": rows,
        "branches": branches,
        "families": PHASE3_FAMILIES,
        "seed_policy": "paired seed per branch shared across K=1, K=2, K=4, and all six comparator families",
        "gpu": gpu_info,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
            "physical_devices": [device.name for device in tf.config.list_physical_devices()],
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "execution_lane": "trusted_gpu_xla_phase3_fixed_split",
            "jit_compile": True,
        },
        "fixture": {
            "path": str(C2_FIXTURE_PATH.relative_to(ROOT)),
            "sha256": _sha256_file(C2_FIXTURE_PATH),
            "schema_id": fixture["schema_id"],
            "horizon": int(fixture["horizon"]),
            "state_dimension": int(fixture["state_dimension"]),
        },
        "calibration": calibration.get("payload", {}) if calibration else {},
        "sources": source_manifest,
        "workspace": _workspace_manifest(),
        "records": records,
        "candidate_summary": _phase3_candidate_summary(records),
        "paired_uncertainty": _phase2_paired_summary(records),
        "nonclaims": (
            "no posterior correctness claim",
            "no unbiased likelihood claim",
            "no statistical superiority claim from twelve branches",
            "no production or default-readiness claim",
        ),
        "budget_remaining": "larger rows remain closed pending Phase 3 validity, cost, and ESS review",
    }
    _write_json(
        output / "manifest.json",
        {
            "schema_version": PHASE3_MANIFEST_SCHEMA,
            "phase": PHASE3_ID,
            "command": " ".join(sys.argv),
            "plan_sha256": source_manifest["plan"]["sha256"],
            "workspace": payload["workspace"],
            "environment": payload["environment"],
            "gpu": gpu_info,
            "fixture": payload["fixture"],
            "rows": rows,
            "branches": branches,
            "proposal_families": PHASE3_FAMILIES,
            "selected_offset": selected_offset,
            "expected_record_count": expected_total,
        },
    )
    _write_json(output / "branch_results.json", records)
    _write_json(output / "result.json", payload)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_phase3_result_markdown(payload), encoding="utf-8")
    return output


def run_phase4(args: argparse.Namespace) -> Path:
    """Run the smooth Student-defensive repair on the unchanged C2 contract."""

    fixture = _load_c2_fixture()
    rows = _parse_rows(args.rows)
    if rows != (PHASE2_FIT_ROWS,):
        raise ValueError(
            f"Phase 4 is fixed to one row count N={PHASE2_FIT_ROWS}; got {rows}"
        )
    branches = int(args.branches)
    if branches != PHASE4_BRANCHES:
        raise ValueError(
            f"Phase 4 is fixed to {PHASE4_BRANCHES} paired branches; got {branches}"
        )
    if not bool(args.jit_compile):
        raise ValueError("Phase 4 requires XLA; --no-jit-compile is not in scope")

    output = _make_output_root(args.output_root)
    records: list[dict[str, object]] = []
    gpu_info: Mapping[str, object] = {}
    prepared: Mapping[str, object] = {}
    calibration: Mapping[str, object] = {}
    fit_seconds: float | None = None
    global_error = ""
    started = time.perf_counter()

    try:
        gpu_info = _configure_phase2_gpu()
        _load_algorithm_modules()
        with tf.device("/GPU:0"):
            model, theta, observations = _c2_model_and_inputs(fixture)
            calibration = _phase4_calibrate_controls(
                model=model, theta=theta, output_root=output
            )
            selected = dict(calibration["selected"])
            prepared = _prepare_phase2_proposals(
                fixture=fixture,
                model=model,
                theta=theta,
                observations=observations,
                output_root=output,
            )
            fit_seconds = float(prepared["fit_seconds"])
            proposal_sets = {
                "retained_tt": prepared["tt_proposals"],
                "gaussian_hint_marginal": prepared["hint_proposals"],
                "stationary_independence": prepared["stationary_proposals"],
            }
            records.extend(
                _run_phase4_candidates(
                    output_root=output,
                    model=model,
                    observations=observations,
                    theta=theta,
                    proposal_sets=proposal_sets,
                    particle_count=PHASE2_FIT_ROWS,
                    branch_count=branches,
                    mixture_offset=float(selected["offset"]),
                    defensive_config=selected,
                )
            )
    except Exception as exc:  # pragma: no cover - retained setup/fit evidence
        global_error = f"{type(exc).__name__}: {exc}"

    selected = calibration.get("selected") if calibration else None
    selected_config_id = selected.get("config_id") if isinstance(selected, Mapping) else None
    selected_rows = [
        row
        for row in calibration.get("rows", ())
        if row.get("config_id") == selected_config_id
    ] if calibration else []
    calibration_pass = (
        selected_config_id is not None
        and len(selected_rows) == 3
        and all(bool(row.get("all_checks_pass", False)) for row in selected_rows)
    )
    required_labels = {
        "ukf_apf_k1_defensive",
        "ukf_apf_k2_defensive",
        "ukf_apf_k4_defensive",
    }
    required_records = [
        record for record in records if record.get("label") in required_labels
    ]
    required_defensive_checks_pass = (
        len(required_records) == branches * len(required_labels)
        and all(bool(record.get("all_checks_pass", False)) for record in required_records)
    )
    expected_total = branches * len(PHASE4_FAMILIES)
    records_complete = len(records) == expected_total
    all_candidate_checks_pass = records_complete and all(
        bool(record.get("all_checks_pass", False)) for record in records
    )
    if global_error:
        status = "PHASE4_ATTEMPT_FAILED"
        continuation = "REPAIR_AND_RETRY_WITHIN_PHASE4_BUDGET"
        failure_class = "setup_or_fit_infrastructure_failure"
        repair = "preserve this attempt, classify the setup/calibration/fit failure, and retry the unchanged Phase 4 contract"
    elif not calibration_pass:
        status = "VETO_PHASE4_CALIBRATION"
        continuation = "REPAIR_REQUIRED_BEFORE_PHASE4_CLAIM"
        failure_class = "tuning_or_numerical_validity"
        repair = "repair or recalibrate the predeclared defensive controls on the independent bank"
    elif not required_defensive_checks_pass:
        status = "VETO_PHASE4_REQUIRED_DEFENSIVE"
        continuation = "REPAIR_REQUIRED_BEFORE_NEXT_PHASE"
        failure_class = "implementation_or_numerical_validity"
        repair = "repair the required defensive support, complete-density, score, or call-chain check"
    elif not records_complete:
        status = "PHASE4_INCOMPLETE"
        continuation = "REPAIR_AND_RETRY_WITHIN_PHASE4_BUDGET"
        failure_class = "infrastructure_or_harness"
        repair = "preserve partial records and retry missing branches under the unchanged contract"
    elif not all_candidate_checks_pass:
        status = "VETO_PHASE4_CANDIDATE_VALIDITY"
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "implementation_or_numerical_validity"
        repair = "repair the failed finite-program diagnostic and replay the affected defensive arms"
    elif not bool(promotion.get("promotion_pass", False)):
        status = "VETO_PHASE4_PROMOTION"
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "candidate_failure"
        repair = "retain the valid defensive candidate as a negative result and refresh Phase 5 with the unchanged proposal ladder"
    else:
        status = "PASS_PHASE4_DEFENSIVE"
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "none"
        repair = "none; all required and comparator finite-program checks passed"

    branch_seconds = sum(
        float(record.get("branch_wall_seconds", 0.0))
        for record in records
        if isinstance(record.get("branch_wall_seconds"), (int, float))
    )
    source_paths = {
        "plan": PLAN_PATH,
        "phase4_plan": PHASE4_PLAN_PATH,
        "driver": Path(__file__).resolve(),
        "generic_kernel": MODULE_PATH,
        "c2_adapter": C2_ADAPTER_PATH,
        "c2_model": C2_MODEL_PATH,
        "exact_evaluator": C2_EXACT_PATH,
    }
    source_manifest = {
        key: {"path": str(path.relative_to(ROOT)), "sha256": _sha256_file(path)}
        for key, path in source_paths.items()
    }
    payload: dict[str, object] = {
        "schema_version": PHASE4_RESULT_SCHEMA,
        "phase": PHASE4_ID,
        "campaign_kind": "phase4",
        "status": status,
        "continuation": continuation,
        "failure_class": failure_class,
        "repair": repair,
        "error": global_error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "proposal_fit_status": "complete" if prepared else "not_complete",
        "fit_seconds": fit_seconds,
        "branch_seconds_sum": branch_seconds,
        "calibration_pass": calibration_pass,
        "selected_config_id": selected_config_id,
        "selected_config": selected,
        "required_defensive_checks_pass": required_defensive_checks_pass,
        "records_complete": records_complete,
        "all_candidate_checks_pass": all_candidate_checks_pass,
        "validity_veto_summary": (
            "all evaluated records passed the declared finite-program checks"
            if all_candidate_checks_pass
            else "one or more evaluated records failed a declared finite-program check"
        ),
        "promotion": promotion,
        "expected_record_count": expected_total,
        "actual_record_count": len(records),
        "rows": rows,
        "branches": branches,
        "families": PHASE4_FAMILIES,
        "seed_policy": "paired seed per branch shared across defensive K=1/K=2/K=4 and Phase 3 comparator families",
        "gpu": gpu_info,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
            "physical_devices": [device.name for device in tf.config.list_physical_devices()],
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "execution_lane": "trusted_gpu_xla_phase4_smooth_student_defensive",
            "jit_compile": True,
        },
        "fixture": {
            "path": str(C2_FIXTURE_PATH.relative_to(ROOT)),
            "sha256": _sha256_file(C2_FIXTURE_PATH),
            "schema_id": fixture["schema_id"],
            "horizon": int(fixture["horizon"]),
            "state_dimension": int(fixture["state_dimension"]),
        },
        "calibration": calibration.get("payload", {}) if calibration else {},
        "sources": source_manifest,
        "workspace": _workspace_manifest(),
        "records": records,
        "candidate_summary": _phase3_candidate_summary(records),
        "paired_uncertainty": _phase2_paired_summary(records),
        "nonclaims": (
            "no posterior correctness claim",
            "no unbiased likelihood claim",
            "no statistical superiority claim from twelve branches",
            "no production or default-readiness claim",
        ),
        "budget_remaining": "larger rows remain closed; Phase 5 recursive-map work is bounded by the refreshed campaign budget",
    }
    _write_json(
        output / "manifest.json",
        {
            "schema_version": PHASE4_MANIFEST_SCHEMA,
            "phase": PHASE4_ID,
            "command": " ".join(sys.argv),
            "plan_sha256": source_manifest["plan"]["sha256"],
            "phase4_plan_sha256": source_manifest["phase4_plan"]["sha256"],
            "workspace": payload["workspace"],
            "environment": payload["environment"],
            "gpu": gpu_info,
            "fixture": payload["fixture"],
            "rows": rows,
            "branches": branches,
            "proposal_families": PHASE4_FAMILIES,
            "selected_config": selected,
            "expected_record_count": expected_total,
        },
    )
    _write_json(output / "branch_results.json", records)
    _write_json(output / "result.json", payload)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_phase4_result_markdown(payload), encoding="utf-8")
    return output


def _scale_free_difference(left: object, right: object) -> float:
    """Compute a relative max error for replay parity diagnostics."""

    left_tensor = tf.convert_to_tensor(left, DTYPE)
    right_tensor = tf.convert_to_tensor(right, DTYPE)
    scale = tf.maximum(
        tf.constant(1.0, DTYPE), tf.abs(left_tensor) + tf.abs(right_tensor)
    )
    return _tensor_float(tf.reduce_max(tf.abs(left_tensor - right_tensor) / scale))


def _phase4_replay_markdown(payload: Mapping[str, object]) -> str:
    """Render a repair replay with its immutable parent linkage."""

    base = _phase4_result_markdown(payload)
    parent = payload.get("parent_attempt", {})
    lines = [
        base.rstrip(),
        "",
        "## Diagnostic repair provenance",
        "",
        f"Parent attempt: `{parent.get('path', 'unknown')}`",
        f"Parent result SHA-256: `{parent.get('result_sha256', 'unknown')}`",
        f"Replayed defensive records: `{payload.get('replayed_record_count', 0)}`",
        f"Inherited comparator records: `{payload.get('inherited_record_count', 0)}`",
        f"Replay branch/program parity: `{payload.get('replay_parity_pass', False)}`",
        "",
        "The parent directory is immutable evidence. The repair changes only the",
        "diagnostic gate from an absolute cancellation residual to a normalized",
        "backward-error bound and adds final-log-weight recurrence parity; it does",
        "not alter the target, proposal law, samples, seeds, or promotion rule.",
        "",
        "Replay parity details are in `replay_parity.json`; the repaired defensive",
        "records are in `branch_results.json`.",
    ]
    return "\n".join(lines) + "\n"


def run_phase4_repair(args: argparse.Namespace) -> Path:
    """Replay all defensive arms after the scale-aware identity repair."""

    fixture = _load_c2_fixture()
    rows = _parse_rows(args.rows)
    if rows != (PHASE2_FIT_ROWS,):
        raise ValueError(
            f"Phase 4 repair is fixed to one row count N={PHASE2_FIT_ROWS}; got {rows}"
        )
    branches = int(args.branches)
    if branches != PHASE4_BRANCHES:
        raise ValueError(
            f"Phase 4 repair is fixed to {PHASE4_BRANCHES} paired branches; got {branches}"
        )
    if not bool(args.jit_compile):
        raise ValueError("Phase 4 repair requires XLA; --no-jit-compile is not in scope")
    parent_result_path = PHASE4_FIRST_ATTEMPT_PATH / "result.json"
    parent_manifest_path = PHASE4_FIRST_ATTEMPT_PATH / "manifest.json"
    if not parent_result_path.is_file() or not parent_manifest_path.is_file():
        raise FileNotFoundError("the immutable Phase 4 parent attempt is missing")
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_result.get("actual_record_count") != branches * len(PHASE4_FAMILIES):
        raise ValueError("parent Phase 4 attempt does not contain the declared records")
    parent_records = parent_result.get("records", ())
    if not isinstance(parent_records, list):
        raise ValueError("parent Phase 4 records are not a list")

    output = _make_output_root(args.output_root)
    records: list[dict[str, object]] = []
    gpu_info: Mapping[str, object] = {}
    calibration: Mapping[str, object] = {}
    global_error = ""
    started = time.perf_counter()
    try:
        gpu_info = _configure_phase2_gpu()
        _load_algorithm_modules()
        with tf.device("/GPU:0"):
            model, theta, observations = _c2_model_and_inputs(fixture)
            calibration = _phase4_calibrate_controls(
                model=model, theta=theta, output_root=output
            )
            selected = dict(calibration["selected"])
            for branch_index in range(branches):
                seed = PHASE4_CLAIM_SEED_BASE + PHASE2_FIT_ROWS + 1009 * branch_index
                for label, compiler, check_xla in _phase4_defensive_specs(
                    model=model,
                    observations=observations,
                    theta=theta,
                    particle_count=PHASE2_FIT_ROWS,
                    seed=seed,
                    mixture_offset=float(selected["offset"]),
                    defensive_config=selected,
                ):
                    branch_started = time.perf_counter()
                    record: dict[str, object] = {
                        "particle_count": PHASE2_FIT_ROWS,
                        "branch_index": branch_index,
                        "seed": seed,
                        "label": label,
                        "mixture_offset": float(selected["offset"]),
                        "defensive_config": dict(selected),
                        "requested_jit_compile": True,
                        "repair_attempt": PHASE4_REPAIR_ID,
                    }
                    try:
                        compilation = compiler()
                        evaluated = dict(
                            _evaluate_c2_candidate(
                                label=label,
                                compilation=compilation,
                                model=model,
                                theta=theta,
                                check_xla=check_xla,
                            )
                        )
                        evaluated["gpu_output"] = "GPU" in str(
                            evaluated.get("output_device", "")
                        ).upper()
                        evaluated["branch_wall_seconds"] = (
                            time.perf_counter() - branch_started
                        )
                        evaluated["checks"] = dict(evaluated["checks"])
                        evaluated["checks"]["gpu_output"] = bool(
                            evaluated["gpu_output"]
                        )
                        evaluated["all_checks_pass"] = all(
                            bool(value) for value in evaluated["checks"].values()
                        )
                        record.update(_jsonable(evaluated))
                    except Exception as exc:  # pragma: no cover - retained replay evidence
                        record.update(
                            {
                                "all_checks_pass": False,
                                "failure_class": "candidate_or_comparator_failure",
                                "error": f"{type(exc).__name__}: {exc}",
                                "branch_wall_seconds": time.perf_counter()
                                - branch_started,
                                "gpu_output": False,
                                "interpretation": "repair replay failure retained",
                            }
                        )
                    records.append(record)
                    _write_json(output / "branch_results.partial.json", records)
    except Exception as exc:  # pragma: no cover - retained setup/replay evidence
        global_error = f"{type(exc).__name__}: {exc}"

    selected = calibration.get("selected") if calibration else None
    selected_config_id = (
        selected.get("config_id") if isinstance(selected, Mapping) else None
    )
    selected_rows = [
        row
        for row in calibration.get("rows", ())
        if row.get("config_id") == selected_config_id
    ] if calibration else []
    calibration_pass = (
        selected_config_id is not None
        and len(selected_rows) == 3
        and all(bool(row.get("all_checks_pass", False)) for row in selected_rows)
    )
    replay_complete = len(records) == branches * 3
    replay_valid = replay_complete and all(
        bool(record.get("all_checks_pass", False)) for record in records
    )
    parent_by_key = {
        (int(record.get("branch_index", -1)), str(record.get("label", ""))): record
        for record in parent_records
    }
    replay_parity_rows: list[dict[str, object]] = []
    for record in records:
        key = (int(record.get("branch_index", -1)), str(record.get("label", "")))
        parent = parent_by_key.get(key)
        parity: dict[str, object] = {"branch_index": key[0], "label": key[1]}
        if parent is None or "error" in record:
            parity["pass"] = False
            parity["reason"] = "missing parent or replay error"
        else:
            parity["branch_id_equal"] = record.get("branch_id") == parent.get("branch_id")
            parity["program_id_equal"] = record.get("program_id") == parent.get("program_id")
            parity["minimum_ess_relative_error"] = _scale_free_difference(
                record.get("minimum_ess"), parent.get("minimum_ess")
            )
            parity["log_likelihood_relative_error"] = _scale_free_difference(
                record.get("log_likelihood"), parent.get("log_likelihood")
            )
            parity["score_relative_error"] = _scale_free_difference(
                record.get("score"), parent.get("score")
            )
            parity["ess_by_time_relative_error"] = _scale_free_difference(
                record.get("ess_by_time"), parent.get("ess_by_time")
            )
            parity["pass"] = bool(
                parity["branch_id_equal"]
                and parity["program_id_equal"]
                and parity["minimum_ess_relative_error"] <= 1.0e-12
                and parity["log_likelihood_relative_error"] <= 1.0e-12
                and parity["score_relative_error"] <= 1.0e-12
                and parity["ess_by_time_relative_error"] <= 1.0e-12
            )
        replay_parity_rows.append(parity)
    replay_parity_pass = bool(replay_parity_rows) and all(
        bool(row.get("pass", False)) for row in replay_parity_rows
    )
    inherited_records = [
        record
        for record in parent_records
        if str(record.get("label", "")) not in {
            "ukf_apf_k1_defensive",
            "ukf_apf_k2_defensive",
            "ukf_apf_k4_defensive",
        }
    ]
    combined_records: list[Mapping[str, object]] = []
    replacement = {
        (int(record.get("branch_index", -1)), str(record.get("label", ""))): record
        for record in records
    }
    for parent in parent_records:
        key = (int(parent.get("branch_index", -1)), str(parent.get("label", "")))
        combined_records.append(replacement.get(key, parent))
    parent_source_paths = {
        "generic_kernel": MODULE_PATH,
        "c2_adapter": C2_ADAPTER_PATH,
        "c2_model": C2_MODEL_PATH,
        "exact_evaluator": C2_EXACT_PATH,
    }
    parent_source_consistency = all(
        parent_result.get("sources", {}).get(name, {}).get("sha256")
        == _sha256_file(path)
        for name, path in parent_source_paths.items()
    )
    parent_fixture_consistency = (
        parent_result.get("fixture", {}).get("sha256") == _sha256_file(C2_FIXTURE_PATH)
    )
    parent_control_consistency = (
        parent_result.get("selected_config_id") == selected_config_id
        and parent_result.get("selected_config", {}).get("offset")
        == (selected.get("offset") if isinstance(selected, Mapping) else None)
    )
    promotion = _phase4_promotion_summary(combined_records)
    all_candidate_checks_pass = (
        replay_valid
        and replay_parity_pass
        and parent_source_consistency
        and parent_fixture_consistency
        and parent_control_consistency
        and all(bool(record.get("all_checks_pass", False)) for record in inherited_records)
    )
    if global_error:
        status = "PHASE4_REPAIR_ATTEMPT_FAILED"
        continuation = "REPAIR_AND_RETRY_WITHIN_PHASE4_BUDGET"
        failure_class = "infrastructure_or_harness"
        repair = "preserve the replay and repair setup under the unchanged contract"
    elif not calibration_pass:
        status = "VETO_PHASE4_REPAIR_CALIBRATION"
        continuation = "REPAIR_REQUIRED_BEFORE_NEXT_PHASE"
        failure_class = "tuning_or_numerical_validity"
        repair = "repair the independent calibration replay"
    elif not (
        parent_source_consistency
        and parent_fixture_consistency
        and parent_control_consistency
    ):
        status = "VETO_PHASE4_REPAIR_PARENT_MISMATCH"
        continuation = "CONTINUATION_VETO_PHASE4"
        failure_class = "target_or_math"
        repair = "parent source, fixture, or frozen-control identity changed; stop rather than reinterpret the replay"
    elif not replay_valid or not replay_parity_pass:
        status = "VETO_PHASE4_REPAIR_VALIDITY"
        continuation = "CONTINUATION_VETO_PHASE4"
        failure_class = "implementation_or_numerical_validity"
        repair = "the scale-aware identity or replay parity still fails; do not enter Phase 5"
    elif not bool(promotion.get("promotion_pass", False)):
        status = "PASS_PHASE4_REPAIR_WITH_PROMOTION_VETO"
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "candidate_failure"
        repair = "retain the valid defensive candidate as a negative result and enter the refreshed Phase 5 plan"
    else:
        status = "PASS_PHASE4_REPAIR_DEFENSIVE"
        continuation = "CONTINUE_NO_REAL_BLOCKER"
        failure_class = "none"
        repair = "none; validity and the declared promotion screen passed"

    branch_seconds = sum(
        float(record.get("branch_wall_seconds", 0.0))
        for record in records
        if isinstance(record.get("branch_wall_seconds"), (int, float))
    )
    source_paths = {
        "plan": PLAN_PATH,
        "phase4_plan": PHASE4_PLAN_PATH,
        "driver": Path(__file__).resolve(),
        "generic_kernel": MODULE_PATH,
        "c2_adapter": C2_ADAPTER_PATH,
        "c2_model": C2_MODEL_PATH,
        "exact_evaluator": C2_EXACT_PATH,
    }
    source_manifest = {
        key: {"path": str(path.relative_to(ROOT)), "sha256": _sha256_file(path)}
        for key, path in source_paths.items()
    }
    payload: dict[str, object] = {
        "schema_version": PHASE4_REPAIR_RESULT_SCHEMA,
        "phase": PHASE4_REPAIR_ID,
        "campaign_kind": "phase4_diagnostic_repair",
        "status": status,
        "continuation": continuation,
        "failure_class": failure_class,
        "repair": repair,
        "error": global_error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "branch_seconds_sum": branch_seconds,
        "calibration_pass": calibration_pass,
        "selected_config_id": selected_config_id,
        "selected_config": selected,
        "replay_complete": replay_complete,
        "replay_valid": replay_valid,
        "replay_parity_pass": replay_parity_pass,
        "parent_source_consistency": parent_source_consistency,
        "parent_fixture_consistency": parent_fixture_consistency,
        "parent_control_consistency": parent_control_consistency,
        "required_defensive_checks_pass": replay_valid,
        "records_complete": len(combined_records) == branches * len(PHASE4_FAMILIES),
        "all_candidate_checks_pass": all_candidate_checks_pass,
        "validity_veto_summary": (
            "all replay and inherited comparator records passed the repaired checks"
            if all_candidate_checks_pass
            else "the repaired replay or inherited comparator checks did not all pass"
        ),
        "expected_record_count": branches * len(PHASE4_FAMILIES),
        "actual_record_count": len(combined_records),
        "replayed_record_count": len(records),
        "inherited_record_count": len(inherited_records),
        "rows": rows,
        "branches": branches,
        "families": PHASE4_FAMILIES,
        "gpu": gpu_info,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get(
                "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
            ),
            "physical_devices": [
                device.name for device in tf.config.list_physical_devices()
            ],
            "logical_devices": [
                device.name for device in tf.config.list_logical_devices()
            ],
            "execution_lane": "trusted_gpu_xla_phase4_backward_error_repair",
            "jit_compile": True,
        },
        "fixture": {
            "path": str(C2_FIXTURE_PATH.relative_to(ROOT)),
            "sha256": _sha256_file(C2_FIXTURE_PATH),
            "schema_id": fixture["schema_id"],
            "horizon": int(fixture["horizon"]),
            "state_dimension": int(fixture["state_dimension"]),
        },
        "calibration": calibration.get("payload", {}) if calibration else {},
        "sources": source_manifest,
        "workspace": _workspace_manifest(),
        "parent_attempt": {
            "path": str(PHASE4_FIRST_ATTEMPT_PATH.relative_to(ROOT)),
            "result_sha256": _sha256_file(parent_result_path),
            "manifest_sha256": _sha256_file(parent_manifest_path),
            "result_schema": parent_result.get("schema_version"),
            "manifest_schema": parent_manifest.get("schema_version"),
        },
        "repaired_identity_policy": {
            "absolute_residual_retained": True,
            "normalized_backward_error_bound": APF_IDENTITY_BACKWARD_BOUND,
            "factor": APF_IDENTITY_BACKWARD_FACTOR,
            "float64_epsilon": FLOAT64_EPSILON,
            "final_log_weight_recurrence_parity": True,
        },
        "records": records,
        "candidate_summary": _phase3_candidate_summary(combined_records),
        "paired_uncertainty": _phase2_paired_summary(combined_records),
        "promotion": promotion,
        "nonclaims": (
            "no posterior correctness claim",
            "no unbiased likelihood claim",
            "no statistical superiority claim from twelve branches",
            "no production or default-readiness claim",
        ),
        "budget_remaining": "Phase 5 is bounded to the refreshed recursive-map fixture and representation pilot",
    }
    _write_json(output / "manifest.json", {
        "schema_version": PHASE4_REPAIR_MANIFEST_SCHEMA,
        "phase": PHASE4_REPAIR_ID,
        "command": " ".join(sys.argv),
        "plan_sha256": source_manifest["plan"]["sha256"],
        "phase4_plan_sha256": source_manifest["phase4_plan"]["sha256"],
        "parent_attempt": payload["parent_attempt"],
        "workspace": payload["workspace"],
        "environment": payload["environment"],
        "gpu": gpu_info,
        "fixture": payload["fixture"],
        "rows": rows,
        "branches": branches,
        "proposal_families": [
            "ukf_apf_k1_defensive",
            "ukf_apf_k2_defensive",
            "ukf_apf_k4_defensive",
        ],
        "selected_config": selected,
        "expected_record_count": branches * 3,
        "repaired_identity_policy": payload["repaired_identity_policy"],
        "parent_source_consistency": parent_source_consistency,
        "parent_fixture_consistency": parent_fixture_consistency,
        "parent_control_consistency": parent_control_consistency,
    })
    _write_json(output / "replay_parity.json", replay_parity_rows)
    _write_json(output / "branch_results.json", records)
    _write_json(output / "result.json", payload)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_phase4_replay_markdown(payload), encoding="utf-8")
    return output


def _make_output_root(value: str) -> Path:
    output = (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def run_phase0(args: argparse.Namespace) -> Path:
    _load_algorithm_modules()
    fixture = _load_fixture()
    output = _make_output_root(args.output_root)
    checks = _run_checks(fixture, jit_compile=bool(args.jit_compile))
    all_pass = bool(checks["all_checks_pass"])
    payload = {
        "schema_version": RESULT_SCHEMA,
        "phase": PHASE_ID,
        "status": "PASS_PHASE0_PREFLIGHT" if all_pass else "VETO_PHASE0_PREFLIGHT",
        "continuation": "CONTINUE_NO_REAL_BLOCKER" if all_pass else "CONTINUATION_VETO_PHASE0_CHECK",
        "failure_class": "none" if all_pass else "implementation_or_numerical_validity",
        "repair": "none; all bounded checks passed" if all_pass else "repair failing check before Phase 1",
        "next_phase_refresh": "Phase 1 API uses the frozen [B,D], [B,D,D], and [B,M,M] signatures" if all_pass else "blocked until Phase 0 repair",
        "budget_remaining": "campaign budget unchanged; Phase 0 consumed one bounded preflight",
        "formal_tool_status": "not run in this driver; MathDevMCP/Lean status must be recorded separately",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
            "physical_devices": [device.name for device in tf.config.list_physical_devices()],
            "logical_devices": [device.name for device in tf.config.list_logical_devices()],
            "execution_lane": "cpu_reference_with_optional_xla",
        },
        "fixture": {
            "path": str(FIXTURE_PATH.relative_to(ROOT)),
            "sha256": _sha256_file(FIXTURE_PATH),
            "schema_version": fixture["schema_version"],
        },
        "sources": {
            "plan": {
                "path": str(PLAN_PATH.relative_to(ROOT)),
                "sha256": _sha256_file(PLAN_PATH),
            },
            "kernel": {
                "path": str(MODULE_PATH.relative_to(ROOT)),
                "sha256": _sha256_file(MODULE_PATH),
            },
        },
        "workspace": _workspace_manifest(),
        "seed": SEED,
        "rows_argument": str(args.rows),
        "branches_argument": int(args.branches),
        "checks": checks,
    }
    _write_json(output / "manifest.json", {
        "schema_version": "c2_mixture_ukf_apf_phase0_manifest_v1",
        "phase": PHASE_ID,
        "command": " ".join(sys.argv),
        "plan_sha256": payload["sources"]["plan"]["sha256"],
        "workspace": payload["workspace"],
        "environment": payload["environment"],
        "fixture": payload["fixture"],
        "seed": SEED,
    })
    _write_json(output / "result.json", payload)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(_result_markdown(payload), encoding="utf-8")
    return output


def main() -> int:
    args = _parse_args()
    if int(args.branches) < 1:
        raise ValueError("--branches must be positive")
    if args.phase == "phase0":
        output = run_phase0(args)
        phase = PHASE_ID
    elif args.phase == "smoke":
        output = run_phase1_smoke(args)
        phase = PHASE1_ID
    elif args.phase == "phase2-entry":
        output = run_phase2_entry(args)
        phase = PHASE2_ID
    elif args.phase == "phase3":
        output = run_phase3(args)
        phase = PHASE3_ID
    elif args.phase == "phase4":
        output = run_phase4(args)
        phase = PHASE4_ID
    elif args.phase == "phase4-repair":
        output = run_phase4_repair(args)
        phase = PHASE4_REPAIR_ID
    else:
        output = run_phase2_serious(args)
        phase = PHASE2_SERIOUS_ID
    print(json.dumps({"phase": phase, "output_root": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
