"""Bounded C2 compatibility smoke for the generic complete-DMIS kernel.

This is a CPU-only diagnostic.  It combines a real retained Hermite proposal
object with a fixed Student component, evaluates the exact C2 transition and
observation factors, and checks the generic frozen-bank tangent against a
central finite difference.  It does not estimate the C2 marginal likelihood
or alter any claim-bearing C2 route.
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

import tensorflow as tf

from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
    GaussianHermiteRetainedProposal,
    stateless_proposal_random_inputs,
)
from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
    C2StochasticVolatilityFrozenAPFModel,
)
from bayesfilter.highdim.frozen_dmis_control_variate_tf import (
    DTYPE,
    ROUTE_CLASSIFICATION,
    ROUTE_ID,
    complete_mixture_log_density,
    frozen_importance_directional_estimate,
    frozen_importance_estimate,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DEFAULT = ROOT / (
    "docs/benchmarks/artifacts/"
    "c2_generic_dmis_compatibility_20260901/attempt01"
)
STATE_DIM = 2
BANK_COUNT = 32
SEED = 20260901
THETA = tf.constant([0.58, math.log(0.4)], DTYPE)
OBSERVATION = tf.constant([0.35, -0.22], DTYPE)
PARENT = tf.constant([[0.20, -0.10]], DTYPE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(OUTPUT_DEFAULT))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"refusing to serialize non-finite float {value!r}")
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported artifact value {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _make_retained_proposal() -> GaussianHermiteRetainedProposal:
    first = tf.constant([1.0, 0.12], DTYPE)
    second = tf.constant([1.0, -0.08], DTYPE)
    z_h = tf.reduce_sum(tf.square(first)) * tf.reduce_sum(tf.square(second))
    return GaussianHermiteRetainedProposal(
        prefix_core_values=(
            tf.reshape(first, [1, 2, 1]),
            tf.reshape(second, [1, 2, 1]),
        ),
        suffix_gram=tf.ones([1, 1], DTYPE),
        z_h=z_h,
        tau_abs=tf.constant(0.02, DTYPE),
        coordinate_offset=tf.zeros([STATE_DIM], DTYPE),
        coordinate_matrix=tf.eye(STATE_DIM, dtype=DTYPE),
        defensive_nu=5.0,
        time_index=1,
        source_snapshot_fingerprint="0" * 64,
    )


def _student_log_density(
    states: tf.Tensor, mean: tf.Tensor, chol: tf.Tensor, degrees: float
) -> tf.Tensor:
    centered = states - mean[None, :]
    whitened = tf.transpose(
        tf.linalg.triangular_solve(chol, tf.transpose(centered), lower=True)
    )
    nu = tf.constant(degrees, DTYPE)
    dimension = tf.cast(tf.shape(states)[1], DTYPE)
    log_normalizer = (
        tf.math.lgamma((nu + dimension) / 2.0)
        - tf.math.lgamma(nu / 2.0)
        - 0.5 * dimension * tf.math.log(nu * tf.constant(math.pi, DTYPE))
        - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    )
    return log_normalizer - 0.5 * (nu + dimension) * tf.reduce_sum(
        tf.math.log1p(tf.square(whitened) / nu), axis=1
    )


def _sample_student(
    count: int, mean: tf.Tensor, chol: tf.Tensor, degrees: float
) -> tf.Tensor:
    normal = tf.random.stateless_normal(
        [count, STATE_DIM], [SEED, 71], dtype=DTYPE
    )
    chi_square = tf.random.stateless_gamma(
        [count, STATE_DIM],
        [SEED, 72],
        alpha=tf.constant(degrees / 2.0, DTYPE),
        beta=tf.constant(0.5, DTYPE),
        dtype=DTYPE,
    )
    standard_student = normal / tf.sqrt(chi_square / degrees)
    return mean[None, :] + tf.einsum("ij,nj->ni", chol, standard_student)


def _target_and_score(
    model: C2StochasticVolatilityFrozenAPFModel,
    theta: tf.Tensor,
    states: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    parents = tf.broadcast_to(PARENT, [tf.shape(states)[0], STATE_DIM])
    transition_log = model.transition_log_density(theta, parents, states, 1)
    observation_log = model.observation_log_density(
        theta, states, OBSERVATION, 1
    )
    log_target = transition_log + observation_log
    target = tf.exp(log_target)
    score = model.transition_log_density_parameter_score(
        theta, parents, states, 1
    ) + model.observation_log_density_parameter_score(
        theta, states, OBSERVATION, 1
    )
    return target, score, log_target


def _evaluate_value(
    model: C2StochasticVolatilityFrozenAPFModel,
    theta: tf.Tensor,
    states: tf.Tensor,
    component_logs: tf.Tensor,
    alpha: tf.Tensor,
    masses: tf.Tensor,
) -> tf.Tensor:
    target, _, _ = _target_and_score(model, theta, states)
    return frozen_importance_estimate(
        target, component_logs, alpha, masses
    )["normalizer"]


def run(output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    model = C2StochasticVolatilityFrozenAPFModel(
        coupling_matrix=tf.constant([[0.0, 0.04], [-0.02, 0.0]], DTYPE),
        sigma=1.0,
    )
    retained = _make_retained_proposal()
    mixture_uniforms, hermite_uniforms, defensive_inputs = (
        stateless_proposal_random_inputs(retained, BANK_COUNT, (SEED, 11))
    )
    retained_sample = retained.compiled_sampler(
        BANK_COUNT, jit_compile=False
    )(mixture_uniforms, hermite_uniforms, defensive_inputs)
    retained_states = retained_sample["physical_points"]

    student_mean = tf.constant([0.0, 0.0], DTYPE)
    student_chol = tf.linalg.diag(tf.constant([1.35, 0.95], DTYPE))
    student_states = _sample_student(
        BANK_COUNT, student_mean, student_chol, degrees=5.0
    )
    states = tf.concat([retained_states, student_states], axis=0)
    retained_log_q = retained.physical_log_density(states)
    student_log_q = _student_log_density(
        states, student_mean, student_chol, degrees=5.0
    )
    component_logs = tf.stack([retained_log_q, student_log_q], axis=1)
    alpha = tf.constant([0.5, 0.5], DTYPE)
    masses = tf.fill(
        [2 * BANK_COUNT], tf.constant(1.0 / (2.0 * BANK_COUNT), DTYPE)
    )

    target, score, log_target = _target_and_score(model, THETA, states)
    log_q = complete_mixture_log_density(component_logs, alpha)
    manual_log_q = tf.math.log(
        0.5 * tf.exp(component_logs[:, 0])
        + 0.5 * tf.exp(component_logs[:, 1])
    )
    tf.debugging.assert_near(log_q, manual_log_q, atol=1e-13, rtol=1e-13)
    value = frozen_importance_estimate(target, component_logs, alpha, masses)

    component_tangent = tf.zeros([2 * BANK_COUNT, 2, 2], DTYPE)
    tangent = frozen_importance_directional_estimate(
        target,
        component_logs,
        alpha,
        masses,
        target[:, None] * score,
        component_tangent,
    )
    step = tf.constant(1.0e-6, DTYPE)
    finite_difference = []
    for index in range(2):
        direction = tf.one_hot(index, 2, dtype=DTYPE)
        finite_difference.append(
            (
                _evaluate_value(
                    model, THETA + step * direction, states, component_logs,
                    alpha, masses
                )
                - _evaluate_value(
                    model, THETA - step * direction, states, component_logs,
                    alpha, masses
                )
            )
            / (2.0 * step)
        )
    finite_difference_tensor = tf.stack(finite_difference)
    tf.debugging.assert_near(
        tangent["normalizer_tangent"],
        finite_difference_tensor,
        atol=2.0e-7,
        rtol=2.0e-7,
    )
    if not bool(value["valid"].numpy()) or not bool(value["dmis_valid"].numpy()):
        raise RuntimeError("generic C2 compatibility value failed validity flags")
    if not bool(tangent["tangent_valid"].numpy()):
        raise RuntimeError("generic C2 compatibility tangent failed validity flag")

    script_path = Path(__file__).resolve()
    manifest = {
        "status": "passed",
        "classification": "compatibility_diagnostic_only",
        "scientific_claims": [],
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "model_id": "c2_sv_gamma_log_beta_stationary_v1",
        "state_dimension": STATE_DIM,
        "bank_count_per_component": BANK_COUNT,
        "total_rows": 2 * BANK_COUNT,
        "component_weights": alpha,
        "base_mass_policy": "alpha_over_component_bank_count",
        "target": "exact C2 transition times observation at fixed parent",
        "proposal_components": [
            "GaussianHermiteRetainedProposal (retained TT plus its declared floor)",
            "fixed product Student-t, nu=5, diagonal scale",
        ],
        "complete_mixture_checked": True,
        "proposal_rows_frozen": True,
        "jit_compile_sampler": False,
        "device_policy": "CPU-only diagnostic; CUDA_VISIBLE_DEVICES=-1",
        "seed": SEED,
        "theta_reference": THETA,
        "observation": OBSERVATION,
        "parent": PARENT,
        "retained_proposal_manifest": retained.manifest_payload(),
        "normalizer": value["normalizer"],
        "log_normalizer": value["log_normalizer"],
        "dmis_normalizer": value["dmis_normalizer"],
        "target_weight_ess": value["target_effective_sample_size"],
        "target_weight_ess_fraction": value[
            "target_effective_sample_size_fraction"
        ],
        "residual_second_moment": value["residual_second_moment"],
        "normalizer_tangent": tangent["normalizer_tangent"],
        "finite_difference_tangent": finite_difference_tensor,
        "maximum_tangent_abs_error": tf.reduce_max(
            tf.abs(tangent["normalizer_tangent"] - finite_difference_tensor)
        ),
        "minimum_log_density": tf.reduce_min(log_q),
        "maximum_log_density": tf.reduce_max(log_q),
        "minimum_log_target": tf.reduce_min(log_target),
        "maximum_log_target": tf.reduce_max(log_target),
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_short": _git_value("status", "--short"),
        "command": [sys.executable, *sys.argv],
        "script_sha256": _sha256(script_path),
        "module_sha256": _sha256(
            ROOT / "bayesfilter/highdim/frozen_dmis_control_variate_tf.py"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_root / "result.json", manifest)
    (output_root / "result.md").write_text(
        "# Generic DMIS C2 Compatibility Smoke\n\n"
        "Status: `PASSED` (CPU-only compatibility diagnostic; no C2 likelihood "
        "or default claim).\n\n"
        f"The run used {2 * BANK_COUNT} frozen rows: equal banks from a retained "
        "Hermite proposal and a fixed nu=5 Student component. The exact C2 "
        "transition-observation target was evaluated at a fixed parent, and "
        "the complete two-component denominator was checked against an "
        "independent recomposition.\n\n"
        f"- Normalizer: `{float(value['normalizer'].numpy()):.16g}`\n"
        f"- Log normalizer: `{float(value['log_normalizer'].numpy()):.16g}`\n"
        f"- Target-weight ESS fraction: "
        f"`{float(value['target_effective_sample_size_fraction'].numpy()):.8g}`\n"
        f"- Maximum tangent finite-difference error: "
        f"`{float(manifest['maximum_tangent_abs_error'].numpy()):.8g}`\n"
        "- Complete-mixture and validity flags: `true`\n\n"
        "The tangent check freezes rows, masses, proposal logs, and component "
        "labels; it verifies the generic finite-program derivative, not an "
        "adaptive proposal derivative. Full machine-readable provenance is in "
        "`result.json`.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = _parse_args()
    run(Path(args.output_root).resolve())


if __name__ == "__main__":
    main()
