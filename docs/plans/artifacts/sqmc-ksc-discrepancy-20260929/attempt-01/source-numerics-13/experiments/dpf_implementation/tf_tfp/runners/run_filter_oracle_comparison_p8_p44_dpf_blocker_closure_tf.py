"""Run P8 P44 DPF blocker-closure measurements.

P8 is a CPU-only TensorFlow/TFP execution phase.  It compares the
`dpf_bootstrap_ot` and `dpf_ledh_pfpf_ot` routes against same-target P44 dense
references, with an explicit P44-M2 dim-1 adapter gate before the full panel.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-p8-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

import bayesfilter.highdim as highdim
from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    normalize_log_weights_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    finite_tensor,
    load_json,
    max_sinkhorn_residual,
    scalar,
    stable_digest,
    tensor_to_json,
    utc_now,
    write_json,
    write_text,
)


MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p8_p44_dpf_blocker_closure_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p8-p44-dpf-blocker-closure-plan-2026-06-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p8-p44-dpf-blocker-closure-result-2026-06-09.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p8-claude-review-ledger-2026-06-09.md"
)
AMENDED_P6_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p6-amended-with-p8-dpf-metrics-2026-06-09.md"
)
JSON_PATH = (
    OUTPUT_DIR
    / "dpf_filter_oracle_comparison_p8_p44_dpf_blocker_closure_2026-06-09.json"
)
REPORT_PATH = (
    REPORT_DIR
    / "dpf-filter-oracle-comparison-p8-p44-dpf-blocker-closure-2026-06-09.md"
)

SEEDS = [101, 202, 303, 404, 505]
BASE_PARTICLE_COUNTS = [128, 256]
TRIGGERED_PARTICLE_COUNT = 512
GATE_PARTICLE_COUNT = 128
TARGET_IDS = [
    "p44_m2_cubic_additive_gaussian_panel",
    "p44_m3_quadratic_observation_panel",
    "p44_m4_nonlinear_transition_h2_panel",
]
METHOD_IDS = ["dpf_bootstrap_ot", "dpf_ledh_pfpf_ot"]
DTYPE = tf.float64
FD_STEP = 1e-5
INITIAL_QUADRATURE_ORDER = 241

TARGET_SPECS = {
    "p44_m2_cubic_additive_gaussian_panel": {
        "family": "m2",
        "theta": [0.25, math.log(0.14), math.log(0.10), 0.04, 0.35],
        "dense_order": 241,
        "horizon": 2,
        "parameterization": "(rho_raw, log_q, log_r, raw_initial_mean, cubic_raw)",
    },
    "p44_m3_quadratic_observation_panel": {
        "family": "m3",
        "theta": [0.18, math.log(0.13), math.log(0.09), 0.02],
        "dense_order": 281,
        "horizon": 2,
        "parameterization": "(rho_raw, log_q, log_r, raw_initial_mean)",
    },
    "p44_m4_nonlinear_transition_h2_panel": {
        "family": "m4",
        "theta": [0.22, math.log(0.12), math.log(0.09), 0.03, 0.40],
        "dense_order": 241,
        "horizon": 2,
        "parameterization": "(rho_raw, log_q, log_r, raw_initial_mean, nonlin_raw)",
    },
}


@dataclass(frozen=True)
class AxisModel:
    target_id: str
    family: str
    axis: int
    horizon: int

    def parameter_dim(self) -> int:
        return 4 if self.family == "m3" else 5

    def state_dim(self) -> int:
        return 1

    def observation_dim(self) -> int:
        return 1

    def observations(self) -> tf.Tensor:
        return _observations(self.family, self.axis + 1, self.horizon)[:, self.axis : self.axis + 1]

    def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
        points = _rows(x0, 1, "x0")[:, 0]
        parts = _axis_part(self.family, theta, self.axis)
        if self.family == "m4":
            nodes, weights = highdim.legendre_gauss_nodes_weights(INITIAL_QUADRATURE_ORDER)
            raw = 6.0 * tf.cast(nodes, DTYPE)
            scaled_weights = 6.0 * tf.cast(weights, DTYPE)
            raw_prior = tfp.distributions.Normal(
                loc=parts["raw_initial_mean"],
                scale=tf.sqrt(parts["raw_initial_variance"]),
            )
            transition = tfp.distributions.Normal(
                loc=_transition_mean_scalar(raw, parts),
                scale=tf.sqrt(parts["transition_variance"]),
            )
            log_terms = (
                tf.math.log(scaled_weights)[tf.newaxis, :]
                + raw_prior.log_prob(raw)[tf.newaxis, :]
                + transition.log_prob(points[:, tf.newaxis])
            )
            return tf.reduce_logsumexp(log_terms, axis=1)
        return tfp.distributions.Normal(
            loc=parts["initial_mean"],
            scale=tf.sqrt(parts["initial_variance"]),
        ).log_prob(points)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_prev: tf.Tensor,
        x_next: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del t
        previous = _rows(x_prev, 1, "x_prev")[:, 0]
        current = _rows(x_next, 1, "x_next")[:, 0]
        parts = _axis_part(self.family, theta, self.axis)
        loc = (
            _transition_mean_scalar(previous, parts)
            if self.family == "m4"
            else parts["rho"] * previous
        )
        return tfp.distributions.Normal(
            loc=loc,
            scale=tf.sqrt(parts["transition_variance"]),
        ).log_prob(current)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        x_t: tf.Tensor,
        y_t: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del t
        points = _rows(x_t, 1, "x_t")[:, 0]
        observation = tf.reshape(tf.convert_to_tensor(y_t, dtype=DTYPE), [1])[0]
        parts = _axis_part(self.family, theta, self.axis)
        if self.family == "m2":
            loc = points + parts["cubic"] * tf.pow(points, 3)
        elif self.family == "m3":
            loc = tf.square(points)
        elif self.family == "m4":
            loc = points
        else:
            raise ValueError(f"unknown P44 family {self.family}")
        return tfp.distributions.Normal(
            loc=loc,
            scale=tf.sqrt(parts["observation_variance"]),
        ).log_prob(observation)

    def manifest_payload(self) -> dict[str, object]:
        return {
            "family": f"p44_{self.family}_scalar_axis_model",
            "target_id": self.target_id,
            "axis": int(self.axis),
            "horizon": int(self.horizon),
            "timing": "observe_initial_latent_at_t0_then_transition_between_observations",
        }


class P8ValidationError(ValueError):
    """Raised when a P8 artifact violates the reviewed phase contract."""


def _observations(family: str, dim: int, horizon: int) -> tf.Tensor:
    if family == "m2":
        values = tf.constant(
            [
                [0.10, -0.04, 0.07],
                [-0.02, 0.06, -0.05],
            ],
            dtype=DTYPE,
        )
        return values[:, : int(dim)]
    if family == "m3":
        values = tf.constant(
            [
                [0.16, 0.11, 0.20],
                [0.09, 0.18, 0.13],
            ],
            dtype=DTYPE,
        )
        return values[:, : int(dim)]
    if family == "m4":
        values = tf.constant(
            [
                [0.08, -0.03, 0.06],
                [-0.02, 0.05, -0.04],
                [0.04, -0.01, 0.03],
                [0.01, 0.02, -0.02],
            ],
            dtype=DTYPE,
        )
        return values[: int(horizon), : int(dim)]
    raise ValueError(f"unknown P44 family {family}")


def _physical_parts(family: str, theta: tf.Tensor, dim: int) -> dict[str, tf.Tensor]:
    theta = tf.convert_to_tensor(theta, dtype=DTYPE)
    if family == "m2":
        rho_scale = tf.constant([1.00, 0.86, 0.72], dtype=DTYPE)[:dim]
        q_scale = tf.constant([0.90, 1.15, 1.35], dtype=DTYPE)[:dim]
        r_scale = tf.constant([1.00, 1.20, 0.85], dtype=DTYPE)[:dim]
        mean_scale = tf.constant([1.00, -0.50, 0.25], dtype=DTYPE)[:dim]
        cubic_scale = tf.constant([1.00, 0.80, 1.20], dtype=DTYPE)[:dim]
        raw_initial_variance = tf.constant([0.55, 0.70, 0.90], dtype=DTYPE)[:dim]
        rho = 0.45 * tf.tanh(theta[0]) * rho_scale
        transition_variance = tf.exp(theta[1]) * q_scale
        observation_variance = tf.exp(theta[2]) * r_scale
        raw_initial_mean = theta[3] * mean_scale
        initial_mean = rho * raw_initial_mean
        initial_variance = tf.square(rho) * raw_initial_variance + transition_variance
        cubic = 0.04 * tf.tanh(theta[4]) * cubic_scale
        return {
            "rho": rho,
            "transition_variance": transition_variance,
            "observation_variance": observation_variance,
            "raw_initial_mean": raw_initial_mean,
            "raw_initial_variance": raw_initial_variance,
            "initial_mean": initial_mean,
            "initial_variance": initial_variance,
            "cubic": cubic,
        }
    if family == "m3":
        rho_scale = tf.constant([1.00, 0.88, 0.76], dtype=DTYPE)[:dim]
        q_scale = tf.constant([0.90, 1.12, 1.30], dtype=DTYPE)[:dim]
        r_scale = tf.constant([1.00, 1.18, 0.90], dtype=DTYPE)[:dim]
        mean_scale = tf.constant([1.00, -0.50, 0.25], dtype=DTYPE)[:dim]
        raw_initial_variance = tf.constant([0.50, 0.68, 0.82], dtype=DTYPE)[:dim]
        rho = 0.40 * tf.tanh(theta[0]) * rho_scale
        transition_variance = tf.exp(theta[1]) * q_scale
        observation_variance = tf.exp(theta[2]) * r_scale
        raw_initial_mean = theta[3] * mean_scale
        initial_mean = rho * raw_initial_mean
        initial_variance = tf.square(rho) * raw_initial_variance + transition_variance
        return {
            "rho": rho,
            "transition_variance": transition_variance,
            "observation_variance": observation_variance,
            "raw_initial_mean": raw_initial_mean,
            "raw_initial_variance": raw_initial_variance,
            "initial_mean": initial_mean,
            "initial_variance": initial_variance,
        }
    if family == "m4":
        rho_scale = tf.constant([1.00, 0.88, 0.76], dtype=DTYPE)[:dim]
        q_scale = tf.constant([0.90, 1.12, 1.30], dtype=DTYPE)[:dim]
        r_scale = tf.constant([1.00, 1.18, 0.90], dtype=DTYPE)[:dim]
        mean_scale = tf.constant([1.00, -0.50, 0.25], dtype=DTYPE)[:dim]
        nonlin_scale = tf.constant([1.00, 0.75, 1.15], dtype=DTYPE)[:dim]
        raw_initial_variance = tf.constant([0.50, 0.68, 0.82], dtype=DTYPE)[:dim]
        rho = 0.42 * tf.tanh(theta[0]) * rho_scale
        transition_variance = tf.exp(theta[1]) * q_scale
        observation_variance = tf.exp(theta[2]) * r_scale
        raw_initial_mean = theta[3] * mean_scale
        nonlin = 0.05 * tf.tanh(theta[4]) * nonlin_scale
        return {
            "rho": rho,
            "transition_variance": transition_variance,
            "observation_variance": observation_variance,
            "raw_initial_mean": raw_initial_mean,
            "raw_initial_variance": raw_initial_variance,
            "nonlin": nonlin,
            "initial_mean": rho * raw_initial_mean,
            "initial_variance": tf.square(rho) * raw_initial_variance + transition_variance,
        }
    raise ValueError(f"unknown P44 family {family}")


def _axis_part(family: str, theta: tf.Tensor, axis: int) -> dict[str, tf.Tensor]:
    parts = _physical_parts(family, theta, axis + 1)
    return {key: value[axis] for key, value in parts.items()}


def _rows(values: tf.Tensor, width: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, dtype=DTYPE)
    if tensor.shape.rank == 1:
        tensor = tensor[tf.newaxis, :]
    if tensor.shape.rank != 2 or tensor.shape[1] != int(width):
        raise ValueError(f"{name} has wrong shape")
    return tensor


def _transition_mean_scalar(values: tf.Tensor, parts: dict[str, tf.Tensor]) -> tf.Tensor:
    values = tf.convert_to_tensor(values, dtype=DTYPE)
    return parts["rho"] * values + parts["nonlin"] * tf.math.tanh(values)


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _dense_config(axis: int, order: int) -> highdim.FixedBranchFilterConfig:
    return highdim.FixedBranchFilterConfig(
        fit_config=None,
        density_tau=0.0,
        normalizer_floor=1e-14,
        denominator_floor=1e-14,
        retained_storage_byte_budget=20_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], dtype=DTYPE),
                matrix=tf.constant([[6.0]], dtype=DTYPE),
            ),
        ),
        measure_convention=_convention(),
        deterministic_seed=f"p44-p8-dense-axis-{axis}-order-{order}",
        fit_quadrature_order=int(order),
    )


def _dense_axis_value(theta: tf.Tensor, target_id: str, axis: int) -> tf.Tensor:
    spec = TARGET_SPECS[target_id]
    model = AxisModel(
        target_id=target_id,
        family=str(spec["family"]),
        axis=axis,
        horizon=int(spec["horizon"]),
    )
    return highdim.FixedBranchSquaredTTFilter(
        _dense_config(axis, int(spec["dense_order"]))
    ).log_likelihood(
        model,
        theta,
        model.observations(),
    ).log_likelihood


def _dense_panel_value(theta: tf.Tensor, target_id: str, dim: int) -> tf.Tensor:
    terms = [_dense_axis_value(theta, target_id, axis) for axis in range(dim)]
    return tf.reduce_sum(tf.stack(terms))


def _value_and_score(
    value_fn: Callable[[tf.Tensor], tf.Tensor],
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    theta = tf.convert_to_tensor(theta, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = value_fn(theta)
    score = tape.gradient(value, theta)
    if score is None:
        raise P8ValidationError("GradientTape returned None")
    return value, score


def _reference_row(target_id: str, dim: int) -> dict[str, Any]:
    theta = tf.constant(TARGET_SPECS[target_id]["theta"], dtype=DTYPE)
    value, score = _value_and_score(
        lambda current_theta: _dense_panel_value(current_theta, target_id, dim),
        theta,
    )
    observations = _observations(
        str(TARGET_SPECS[target_id]["family"]),
        dim,
        int(TARGET_SPECS[target_id]["horizon"]),
    )
    return {
        "target_id": target_id,
        "dim": int(dim),
        "reference_id": "p44_fixed_branch_dense_refined_quadrature",
        "dense_order": int(TARGET_SPECS[target_id]["dense_order"]),
        "horizon": int(TARGET_SPECS[target_id]["horizon"]),
        "observation_count": int(observations.shape[0]) * int(observations.shape[1]),
        "theta": tensor_to_json(theta),
        "log_likelihood": scalar(value),
        "score": tensor_to_json(score),
        "score_norm": scalar(tf.linalg.norm(score)),
        "finite": bool(finite_tensor(value) and finite_tensor(score)),
        "parameterization": TARGET_SPECS[target_id]["parameterization"],
        "scalar_sign_convention": "score = d log_likelihood / d theta",
        "target_timing": "observe_initial_latent_at_t0_then_transition_between_observations",
        "source": "P44 test target definitions transcribed into P8 runner",
    }


def _dpf_row(
    method_id: str,
    target_id: str,
    dim: int,
    seed: int,
    num_particles: int,
    reference: dict[str, Any],
) -> dict[str, Any]:
    theta = tf.constant(TARGET_SPECS[target_id]["theta"], dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value, diagnostics = _dpf_panel_scalar(
            method_id=method_id,
            target_id=target_id,
            dim=dim,
            theta=theta,
            seed=seed,
            num_particles=num_particles,
        )
    gradient = tape.gradient(value, theta)
    if gradient is None:
        gradient = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=DTYPE))
    reference_score = tf.constant(reference["score"], dtype=DTYPE)
    value_error = value - tf.constant(reference["log_likelihood"], dtype=DTYPE)
    gradient_error = tf.cast(gradient, DTYPE) - reference_score
    relative_score_error = tf.linalg.norm(gradient_error) / tf.maximum(
        tf.constant(1.0, dtype=DTYPE),
        tf.linalg.norm(reference_score),
    )
    observation_count = int(reference["observation_count"])
    finite = bool(finite_tensor(value) and finite_tensor(gradient))
    branch_records = diagnostics["branch_records"]
    return {
        "target_id": target_id,
        "dim": int(dim),
        "method_id": method_id,
        "seed": int(seed),
        "num_particles": int(num_particles),
        "value": scalar(value),
        "value_error": scalar(value_error),
        "per_observation_value_error": scalar(
            value_error / tf.constant(float(observation_count), dtype=DTYPE)
        ),
        "fixed_branch_gradient": tensor_to_json(gradient),
        "gradient_error": tensor_to_json(gradient_error),
        "gradient_error_norm": scalar(tf.linalg.norm(gradient_error)),
        "relative_score_error": scalar(relative_score_error),
        "reference_score_norm": float(reference["score_norm"]),
        "finite": finite,
        "gradient_object": "fixed_branch_score",
        "stochastic_score_claim": "not_claimed",
        "branch_signature": diagnostics["branch_signature"],
        "branch_record_count": len(branch_records),
        "branch_records": branch_records,
        "diagnostics": diagnostics["summary"],
    }


def _dpf_panel_scalar(
    *,
    method_id: str,
    target_id: str,
    dim: int,
    theta: tf.Tensor,
    seed: int,
    num_particles: int,
) -> tuple[tf.Tensor, dict[str, Any]]:
    spec = TARGET_SPECS[target_id]
    value = tf.constant(0.0, dtype=DTYPE)
    all_records: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    for axis in range(dim):
        model = AxisModel(
            target_id=target_id,
            family=str(spec["family"]),
            axis=axis,
            horizon=int(spec["horizon"]),
        )
        axis_value, axis_diag = _dpf_axis_scalar(
            method_id=method_id,
            model=model,
            theta=theta,
            seed=int(seed),
            num_particles=int(num_particles),
        )
        value = value + axis_value
        all_records.extend(axis_diag["branch_records"])
        signatures.extend(axis_diag["branch_signature"])
    summary = _branch_summary(all_records)
    return value, {
        "branch_records": all_records,
        "branch_signature": signatures,
        "summary": summary,
    }


def _dpf_axis_scalar(
    *,
    method_id: str,
    model: AxisModel,
    theta: tf.Tensor,
    seed: int,
    num_particles: int,
) -> tuple[tf.Tensor, dict[str, Any]]:
    observations = model.observations()
    if method_id == "dpf_bootstrap_ot":
        particles = _sample_initial_exact(model, theta, num_particles, seed)
    elif method_id == "dpf_ledh_pfpf_ot":
        particles = _sample_initial_ledh_proposal(model, theta, num_particles, seed)
    else:
        raise ValueError(f"unknown method_id {method_id}")
    log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
    value = tf.constant(0.0, dtype=DTYPE)
    records: list[dict[str, Any]] = []
    signature: list[dict[str, Any]] = []

    for time_index, observation in enumerate(tf.unstack(observations, axis=0)):
        if method_id == "dpf_bootstrap_ot":
            if time_index > 0:
                particles = _sample_transition(model, theta, particles, seed, time_index)
            proposal_particles = particles
            corrected = log_weights + model.observation_log_density(
                theta,
                proposal_particles,
                observation,
                time_index,
            )
            ledh_diag: dict[str, Any] = {}
        else:
            ancestors = particles
            if time_index == 0:
                pre_flow = particles
                prior_mean, prior_variance = _initial_ledh_gaussian_moments(model, theta)
                target_prior_or_transition = lambda x: model.initial_log_density(theta, x)
            else:
                prior_mean = _transition_mean(model, theta, ancestors)
                prior_variance = _transition_variance(model, theta)
                pre_flow = _sample_transition(model, theta, ancestors, seed, time_index)
                target_prior_or_transition = lambda x: model.transition_log_density(
                    theta,
                    ancestors,
                    x,
                    time_index,
                )
            flow = _ledh_scalar_flow(
                model=model,
                theta=theta,
                pre_flow_particles=pre_flow,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                observation=observation,
            )
            proposal_particles = flow["post_flow_particles"]
            corrected = (
                log_weights
                + target_prior_or_transition(proposal_particles)
                + model.observation_log_density(theta, proposal_particles, observation, time_index)
                - flow["pre_flow_log_density"]
                + flow["forward_log_det"]
            )
            ledh_diag = dict(flow["diagnostics"])

        weights, increment = normalize_log_weights_tf(corrected)
        value = value + increment
        log_weights = corrected - increment
        ess = 1.0 / tf.reduce_sum(weights * weights)
        do_resample = bool((ess < tf.constant(0.5 * num_particles, dtype=DTYPE)).numpy())
        record: dict[str, Any] = {
            "target_id": model.target_id,
            "axis": int(model.axis),
            "family": model.family,
            "method_id": method_id,
            "seed": int(seed),
            "num_particles": int(num_particles),
            "time_index": int(time_index),
            "ess": scalar(ess),
            "ess_ratio": scalar(ess / tf.cast(num_particles, DTYPE)),
            "resampled": bool(do_resample),
            "resampling_method": "none",
            "transport_active_mask_status": "not_triggered",
            "finite_corrected_log_weights": bool(finite_tensor(corrected)),
            "min_corrected_log_weight": scalar(tf.reduce_min(corrected)),
            "max_corrected_log_weight": scalar(tf.reduce_max(corrected)),
            "target_timing": "initial_observation_no_transition" if time_index == 0 else "transition_then_observation",
            **ledh_diag,
        }
        if do_resample:
            transport = annealed_transport_resample_tf(
                proposal_particles,
                log_weights,
                epsilon=0.7,
                scaling=0.9,
                convergence_threshold=1e-3,
                max_iterations=80,
                ess_mask=tf.constant([True], dtype=tf.bool),
                transport_gradient_mode="filterflow_clipped",
                application_mode="active_rows_only",
            )
            particles = tf.cast(transport.particles, DTYPE)
            log_weights = tf.cast(transport.log_weights, DTYPE)
            record.update(
                {
                    "resampling_method": "filterflow_style_annealed_transport_tf",
                    "transport_active_mask_status": "triggered",
                    **dict(transport.diagnostics),
                }
            )
        else:
            particles = proposal_particles
        records.append(record)
        signature.append(_structural_branch_signature(record))

    return value, {"branch_records": records, "branch_signature": signature}


def _sample_initial_exact(
    model: AxisModel,
    theta: tf.Tensor,
    num_particles: int,
    seed: int,
) -> tf.Tensor:
    parts = _axis_part(model.family, theta, model.axis)
    if model.family == "m4":
        raw = _sample_normal(
            parts["raw_initial_mean"],
            parts["raw_initial_variance"],
            num_particles,
            seed,
            10_000 + 97 * model.axis,
        )
        noise = _sample_normal(
            tf.constant(0.0, dtype=DTYPE),
            parts["transition_variance"],
            num_particles,
            seed,
            11_000 + 97 * model.axis,
        )
        samples = _transition_mean_scalar(raw, parts) + noise
    else:
        samples = _sample_normal(
            parts["initial_mean"],
            parts["initial_variance"],
            num_particles,
            seed,
            10_000 + 97 * model.axis,
        )
    return tf.reshape(samples, [num_particles, 1])


def _sample_initial_ledh_proposal(
    model: AxisModel,
    theta: tf.Tensor,
    num_particles: int,
    seed: int,
) -> tf.Tensor:
    mean, variance = _initial_ledh_gaussian_moments(model, theta)
    samples = _sample_normal(mean, variance, num_particles, seed, 12_000 + 97 * model.axis)
    return tf.reshape(samples, [num_particles, 1])


def _sample_transition(
    model: AxisModel,
    theta: tf.Tensor,
    ancestors: tf.Tensor,
    seed: int,
    time_index: int,
) -> tf.Tensor:
    mean = _transition_mean(model, theta, ancestors)
    variance = _transition_variance(model, theta)
    noise = _sample_normal(
        tf.zeros([int(ancestors.shape[0])], dtype=DTYPE),
        variance,
        int(ancestors.shape[0]),
        seed,
        20_000 + 997 * model.axis + int(time_index),
    )
    return tf.reshape(tf.reshape(mean, [-1]) + noise, [int(ancestors.shape[0]), 1])


def _transition_mean(model: AxisModel, theta: tf.Tensor, ancestors: tf.Tensor) -> tf.Tensor:
    parts = _axis_part(model.family, theta, model.axis)
    previous = _rows(ancestors, 1, "ancestors")[:, 0]
    if model.family == "m4":
        mean = _transition_mean_scalar(previous, parts)
    else:
        mean = parts["rho"] * previous
    return tf.reshape(mean, [-1, 1])


def _transition_variance(model: AxisModel, theta: tf.Tensor) -> tf.Tensor:
    return _axis_part(model.family, theta, model.axis)["transition_variance"]


def _initial_ledh_gaussian_moments(model: AxisModel, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    parts = _axis_part(model.family, theta, model.axis)
    if model.family != "m4":
        return parts["initial_mean"], parts["initial_variance"]
    nodes, weights = highdim.legendre_gauss_nodes_weights(INITIAL_QUADRATURE_ORDER)
    raw = 6.0 * tf.cast(nodes, DTYPE)
    scaled_weights = 6.0 * tf.cast(weights, DTYPE)
    raw_prior = tfp.distributions.Normal(
        loc=parts["raw_initial_mean"],
        scale=tf.sqrt(parts["raw_initial_variance"]),
    )
    unnormalized = scaled_weights * tf.exp(raw_prior.log_prob(raw))
    mass = tf.reduce_sum(unnormalized)
    normalized = unnormalized / mass
    transition_mean = _transition_mean_scalar(raw, parts)
    mean = tf.reduce_sum(normalized * transition_mean)
    second = tf.reduce_sum(normalized * tf.square(transition_mean))
    variance = tf.maximum(
        second - tf.square(mean) + parts["transition_variance"],
        tf.constant(1e-10, dtype=DTYPE),
    )
    return mean, variance


def _sample_normal(
    loc: tf.Tensor,
    variance: tf.Tensor,
    num_particles: int,
    seed: int,
    salt: int,
) -> tf.Tensor:
    loc = tf.cast(loc, DTYPE)
    variance = tf.cast(variance, DTYPE)
    shape = [num_particles]
    normal = tf.random.stateless_normal(shape, seed=_seed_pair(seed, salt), dtype=DTYPE)
    return loc + tf.sqrt(variance) * normal


def _ledh_scalar_flow(
    *,
    model: AxisModel,
    theta: tf.Tensor,
    pre_flow_particles: tf.Tensor,
    prior_mean: tf.Tensor,
    prior_variance: tf.Tensor,
    observation: tf.Tensor,
) -> dict[str, Any]:
    x0 = _rows(pre_flow_particles, 1, "pre_flow_particles")[:, 0]
    prior_mean_vec = tf.reshape(tf.cast(prior_mean, DTYPE), [-1])
    if int(prior_mean_vec.shape[0]) == 1:
        prior_mean_vec = tf.fill(tf.shape(x0), prior_mean_vec[0])
    prior_variance = tf.maximum(tf.cast(prior_variance, DTYPE), tf.constant(1e-10, dtype=DTYPE))
    observation_value = tf.reshape(tf.cast(observation, DTYPE), [1])[0]
    parts = _axis_part(model.family, theta, model.axis)
    predicted = _observation_mean(model.family, parts, x0)
    jacobian = _observation_jacobian(model.family, parts, x0)
    observation_variance = parts["observation_variance"]
    pseudo_observation = jacobian * x0 + (observation_value - predicted)
    posterior_precision = 1.0 / prior_variance + tf.square(jacobian) / observation_variance
    posterior_variance = 1.0 / posterior_precision
    posterior_mean = posterior_variance * (
        prior_mean_vec / prior_variance + jacobian * pseudo_observation / observation_variance
    )
    transform_scale = tf.sqrt(posterior_variance / prior_variance)
    post = posterior_mean + transform_scale * (x0 - prior_mean_vec)
    forward_log_det = tf.math.log(transform_scale)
    pre_flow_log_density = tfp.distributions.Normal(
        loc=prior_mean_vec,
        scale=tf.sqrt(prior_variance),
    ).log_prob(x0)
    diagnostics = {
        "ledh_component_id": "p8_scalar_local_affine_ledh",
        "ledh_map_convention": "x1 = local_posterior_mean + sqrt(P_post/P_prior) * (x0 - prior_mean)",
        "ledh_local_linearization": "per_particle_observation_jacobian",
        "ledh_residual_convention": "observed_minus_predicted",
        "pfpf_correction": "log_exact_target_prior_or_transition_plus_observation_minus_q0_plus_forward_logdet",
        "finite_pre_flow": bool(finite_tensor(x0)),
        "finite_post_flow": bool(finite_tensor(post)),
        "finite_forward_log_det": bool(finite_tensor(forward_log_det)),
        "finite_pre_flow_log_density": bool(finite_tensor(pre_flow_log_density)),
        "min_observation_jacobian": scalar(tf.reduce_min(jacobian)),
        "max_observation_jacobian": scalar(tf.reduce_max(jacobian)),
        "min_abs_observation_jacobian": scalar(tf.reduce_min(tf.abs(jacobian))),
        "min_forward_log_det": scalar(tf.reduce_min(forward_log_det)),
        "max_forward_log_det": scalar(tf.reduce_max(forward_log_det)),
        "max_abs_forward_log_det": scalar(tf.reduce_max(tf.abs(forward_log_det))),
        "min_jacobian_singular_value": scalar(tf.reduce_min(tf.abs(transform_scale))),
        "max_jacobian_singular_value": scalar(tf.reduce_max(tf.abs(transform_scale))),
    }
    return {
        "post_flow_particles": tf.reshape(post, [-1, 1]),
        "pre_flow_log_density": pre_flow_log_density,
        "forward_log_det": forward_log_det,
        "diagnostics": diagnostics,
    }


def _observation_mean(family: str, parts: dict[str, tf.Tensor], values: tf.Tensor) -> tf.Tensor:
    if family == "m2":
        return values + parts["cubic"] * tf.pow(values, 3)
    if family == "m3":
        return tf.square(values)
    if family == "m4":
        return values
    raise ValueError(f"unknown P44 family {family}")


def _observation_jacobian(family: str, parts: dict[str, tf.Tensor], values: tf.Tensor) -> tf.Tensor:
    if family == "m2":
        return 1.0 + 3.0 * parts["cubic"] * tf.square(values)
    if family == "m3":
        return 2.0 * values
    if family == "m4":
        return tf.ones_like(values)
    raise ValueError(f"unknown P44 family {family}")


def _branch_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ess_values = [float(row["ess"]) for row in records]
    transport_records = [
        row for row in records if row.get("resampling_method") == "filterflow_style_annealed_transport_tf"
    ]
    return {
        "branch_record_count": len(records),
        "ess_min": min(ess_values) if ess_values else None,
        "ess_mean": statistics.fmean(ess_values) if ess_values else None,
        "resampling_count": sum(1 for row in records if row["resampled"]),
        "transport_trigger_count": len(transport_records),
        "max_sinkhorn_residual": max_sinkhorn_residual(records),
        "all_corrected_log_weights_finite": all(
            bool(row.get("finite_corrected_log_weights", False)) for row in records
        ),
    }


def _structural_branch_signature(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_id": record["target_id"],
        "axis": int(record["axis"]),
        "method_id": record["method_id"],
        "time_index": int(record["time_index"]),
        "resampled": bool(record["resampled"]),
        "resampling_method": record["resampling_method"],
        "transport_active_mask_status": record["transport_active_mask_status"],
        "ledh_local_linearization": record.get("ledh_local_linearization", "not_applicable"),
    }


def _directions(size: int) -> tf.Tensor:
    eye = tf.eye(size, dtype=DTYPE)
    mixed_a = tf.cast(tf.range(1, size + 1), DTYPE)
    mixed_a = mixed_a / tf.linalg.norm(mixed_a)
    mixed_b = tf.where(
        tf.math.floormod(tf.range(size), 2) == 0,
        tf.ones([size], dtype=DTYPE),
        -tf.ones([size], dtype=DTYPE),
    )
    mixed_b = mixed_b / tf.linalg.norm(mixed_b)
    return tf.concat([eye, mixed_a[tf.newaxis, :], mixed_b[tf.newaxis, :]], axis=0)


def _directional_residual(
    *,
    target_id: str,
    dim: int,
    method_id: str,
    seed: int,
    num_particles: int,
    ad_gradient: list[float],
) -> dict[str, Any]:
    theta = tf.constant(TARGET_SPECS[target_id]["theta"], dtype=DTYPE)
    gradient = tf.constant(ad_gradient, dtype=DTYPE)
    base_value, base_diag = _dpf_panel_scalar(
        method_id=method_id,
        target_id=target_id,
        dim=dim,
        theta=theta,
        seed=seed,
        num_particles=num_particles,
    )
    del base_value
    base_signature = base_diag["branch_signature"]
    residuals = []
    stable = True
    for direction in tf.unstack(_directions(int(theta.shape[0])), axis=0):
        plus_theta = theta + tf.constant(FD_STEP, dtype=DTYPE) * direction
        minus_theta = theta - tf.constant(FD_STEP, dtype=DTYPE) * direction
        plus_value, plus_diag = _dpf_panel_scalar(
            method_id=method_id,
            target_id=target_id,
            dim=dim,
            theta=plus_theta,
            seed=seed,
            num_particles=num_particles,
        )
        minus_value, minus_diag = _dpf_panel_scalar(
            method_id=method_id,
            target_id=target_id,
            dim=dim,
            theta=minus_theta,
            seed=seed,
            num_particles=num_particles,
        )
        stable = stable and plus_diag["branch_signature"] == base_signature
        stable = stable and minus_diag["branch_signature"] == base_signature
        fd = (plus_value - minus_value) / tf.constant(2.0 * FD_STEP, dtype=DTYPE)
        ad = tf.tensordot(direction, gradient, axes=1)
        residuals.append(scalar(tf.abs(ad - fd)))
    max_residual = max(residuals) if residuals else float("nan")
    return {
        "target_id": target_id,
        "dim": int(dim),
        "method_id": method_id,
        "seed": int(seed),
        "num_particles": int(num_particles),
        "fd_step": FD_STEP,
        "direction_count": int(_directions(int(theta.shape[0])).shape[0]),
        "directional_abs_residuals": residuals,
        "max_abs_directional_residual": max_residual,
        "branch_signature_stable": bool(stable),
        "branch_signature": base_signature,
    }


def _row_summary(
    *,
    target_id: str,
    dim: int,
    method_id: str,
    num_particles: int,
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
    directional: dict[str, Any] | None,
) -> dict[str, Any]:
    subset = [
        row
        for row in rows
        if row["target_id"] == target_id
        and row["dim"] == dim
        and row["method_id"] == method_id
        and row["num_particles"] == num_particles
    ]
    value_errors = [float(row["value_error"]) for row in subset]
    per_observation_errors = [float(row["per_observation_value_error"]) for row in subset]
    gradient_error_norms = [float(row["gradient_error_norm"]) for row in subset]
    relative_errors = [float(row["relative_score_error"]) for row in subset]
    reference_norm = float(reference["score_norm"])
    direction_max = (
        float(directional["max_abs_directional_residual"])
        if directional is not None
        else float("nan")
    )
    diagnostic_pass = (
        _all_finite(per_observation_errors)
        and _all_finite(relative_errors)
        and abs(statistics.fmean(per_observation_errors)) <= 0.50
        and statistics.fmean(relative_errors) <= 1.00
    )
    promotion_pass = (
        diagnostic_pass
        and _ci_contains_band(_ci95(per_observation_errors), -0.10, 0.10)
        and _rmse(per_observation_errors) <= 0.25
        and statistics.fmean(relative_errors) <= 0.35
        and _rmse(gradient_error_norms) <= max(1.0, 0.50 * reference_norm)
        and max(relative_errors) <= 0.75
        and direction_max <= 0.25 * max(1.0, reference_norm)
        and bool(directional and directional["branch_signature_stable"])
    )
    if promotion_pass:
        decision = "PROMOTED_EXACT_TARGET_CLOSENESS"
    elif diagnostic_pass:
        decision = "DIAGNOSTIC_ONLY_MEASURED"
    else:
        decision = "FAILED_NUMERIC_BANDS"
    return {
        "target_id": target_id,
        "dim": int(dim),
        "method_id": method_id,
        "num_particles": int(num_particles),
        "seed_count": len(subset),
        "mean_value_error": statistics.fmean(value_errors),
        "value_error_standard_error": _standard_error(value_errors),
        "value_error_ci95": _ci95(value_errors),
        "value_error_rmse": _rmse(value_errors),
        "mean_per_observation_value_error": statistics.fmean(per_observation_errors),
        "per_observation_value_error_ci95": _ci95(per_observation_errors),
        "per_observation_value_error_rmse": _rmse(per_observation_errors),
        "max_abs_per_observation_value_error": max(abs(value) for value in per_observation_errors),
        "score_rmse": _rmse(gradient_error_norms),
        "score_error_norm_standard_error": _standard_error(gradient_error_norms),
        "score_error_norm_ci95": _ci95(gradient_error_norms),
        "mean_relative_score_error": statistics.fmean(relative_errors),
        "max_relative_score_error": max(relative_errors),
        "reference_score_norm": reference_norm,
        "all_rows_finite": all(bool(row["finite"]) for row in subset),
        "branch_record_count": sum(int(row["branch_record_count"]) for row in subset),
        "directional_residual_max": direction_max,
        "directional_branch_signature_stable": bool(
            directional and directional["branch_signature_stable"]
        ),
        "row_decision": decision,
    }


def _third_count_reasons(
    small: dict[str, Any],
    large: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    ci = large["per_observation_value_error_ci95"]
    if ci[0] > 0.0 or ci[1] < 0.0:
        reasons.append("per_observation_value_ci95_excludes_zero")
    if large["mean_relative_score_error"] > 1.00:
        reasons.append("mean_relative_score_error_above_diagnostic_band")
    if not bool(large["all_rows_finite"]):
        reasons.append("nonfinite_large_count_row")
    if _safe_ratio(
        large["per_observation_value_error_rmse"],
        small["per_observation_value_error_rmse"],
    ) > 0.75:
        reasons.append("value_rmse_reduction_less_than_25_percent")
    return reasons


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _ci95(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    half = 1.96 * _standard_error(values)
    return [mean - half, mean + half]


def _rmse(values: list[float]) -> float:
    return math.sqrt(statistics.fmean([float(value) ** 2 for value in values]))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-15:
        return float("inf") if abs(float(numerator)) >= 1e-15 else 0.0
    return abs(float(numerator)) / abs(float(denominator))


def _ci_contains_band(ci: list[float], lower: float, upper: float) -> bool:
    return lower <= float(ci[0]) and float(ci[1]) <= upper


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2_147_483_647, int(salt) % 2_147_483_647], dtype=tf.int32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P8_P44_DPF_BLOCKER_CLOSURE_VALIDATED")
        return 0

    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    amended = _amended_p6_markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    write_text(REPO_ROOT / AMENDED_P6_PATH, amended)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    references = [_reference_row(target_id, dim) for target_id in TARGET_IDS for dim in (1, 2, 3)]
    reference_by_key = {
        _combo_key(row["target_id"], int(row["dim"]), "reference", 0): row
        for row in references
    }
    ledh_conventions = [
        _ledh_convention_record(target_id, dim) for target_id in TARGET_IDS for dim in (1, 2, 3)
    ]
    rows: list[dict[str, Any]] = []
    directional_residuals: list[dict[str, Any]] = []

    gate_reference = reference_by_key[
        _combo_key("p44_m2_cubic_additive_gaussian_panel", 1, "reference", 0)
    ]
    for method_id in METHOD_IDS:
        for seed in SEEDS:
            rows.append(
                _safe_dpf_row(
                    method_id=method_id,
                    target_id="p44_m2_cubic_additive_gaussian_panel",
                    dim=1,
                    seed=seed,
                    num_particles=GATE_PARTICLE_COUNT,
                    reference=gate_reference,
                )
            )
        representative = next(
            row
            for row in rows
            if row["target_id"] == "p44_m2_cubic_additive_gaussian_panel"
            and row["dim"] == 1
            and row["method_id"] == method_id
            and row["seed"] == SEEDS[0]
            and row["num_particles"] == GATE_PARTICLE_COUNT
        )
        directional_residuals.append(
            _safe_directional_residual(
                target_id="p44_m2_cubic_additive_gaussian_panel",
                dim=1,
                method_id=method_id,
                seed=SEEDS[0],
                num_particles=GATE_PARTICLE_COUNT,
                ad_gradient=representative["fixed_branch_gradient"],
            )
        )

    gate = _m2_dim1_gate(rows, directional_residuals, gate_reference, ledh_conventions)
    if gate["status"] != "PASS_P8_M2_DIM1_ADAPTER_GATE":
        return _payload(
            decision="BLOCKED_P8_P44_DPF_ADAPTER_GATE",
            references=references,
            ledh_conventions=ledh_conventions,
            rows=rows,
            directional_residuals=directional_residuals,
            row_summaries=[],
            m2_dim1_gate=gate,
            veto_diagnostics=_veto_diagnostics(rows, references, directional_residuals, gate),
        )

    completed = {
        (
            row["target_id"],
            int(row["dim"]),
            row["method_id"],
            int(row["seed"]),
            int(row["num_particles"]),
        )
        for row in rows
    }
    for target_id in TARGET_IDS:
        for dim in (1, 2, 3):
            reference = reference_by_key[_combo_key(target_id, dim, "reference", 0)]
            for method_id in METHOD_IDS:
                for num_particles in BASE_PARTICLE_COUNTS:
                    for seed in SEEDS:
                        key = (target_id, dim, method_id, seed, num_particles)
                        if key in completed:
                            continue
                        rows.append(
                            _safe_dpf_row(
                                method_id=method_id,
                                target_id=target_id,
                                dim=dim,
                                seed=seed,
                                num_particles=num_particles,
                                reference=reference,
                            )
                        )
                        completed.add(key)

    preliminary_summaries: list[dict[str, Any]] = []
    triggered: dict[tuple[str, int, str], list[str]] = {}
    for target_id in TARGET_IDS:
        for dim in (1, 2, 3):
            reference = reference_by_key[_combo_key(target_id, dim, "reference", 0)]
            for method_id in METHOD_IDS:
                summaries = {}
                for num_particles in BASE_PARTICLE_COUNTS:
                    summary = _row_summary(
                        target_id=target_id,
                        dim=dim,
                        method_id=method_id,
                        num_particles=num_particles,
                        rows=rows,
                        reference=reference,
                        directional=None,
                    )
                    summaries[num_particles] = summary
                    preliminary_summaries.append(summary)
                reasons = _third_count_reasons(
                    summaries[BASE_PARTICLE_COUNTS[0]],
                    summaries[BASE_PARTICLE_COUNTS[1]],
                )
                if reasons:
                    triggered[(target_id, dim, method_id)] = reasons

    for (target_id, dim, method_id), reasons in triggered.items():
        del reasons
        reference = reference_by_key[_combo_key(target_id, dim, "reference", 0)]
        for seed in SEEDS:
            rows.append(
                _safe_dpf_row(
                    method_id=method_id,
                    target_id=target_id,
                    dim=dim,
                    seed=seed,
                    num_particles=TRIGGERED_PARTICLE_COUNT,
                    reference=reference,
                )
            )

    final_count_by_combo: dict[tuple[str, int, str], int] = {}
    for target_id in TARGET_IDS:
        for dim in (1, 2, 3):
            for method_id in METHOD_IDS:
                final_count_by_combo[(target_id, dim, method_id)] = (
                    TRIGGERED_PARTICLE_COUNT
                    if (target_id, dim, method_id) in triggered
                    else BASE_PARTICLE_COUNTS[-1]
                )

    for (target_id, dim, method_id), num_particles in final_count_by_combo.items():
        representative = next(
            row
            for row in rows
            if row["target_id"] == target_id
            and row["dim"] == dim
            and row["method_id"] == method_id
            and row["seed"] == SEEDS[0]
            and row["num_particles"] == num_particles
        )
        directional_residuals.append(
            _safe_directional_residual(
                target_id=target_id,
                dim=dim,
                method_id=method_id,
                seed=SEEDS[0],
                num_particles=num_particles,
                ad_gradient=representative["fixed_branch_gradient"],
            )
        )

    final_directional = {
        (row["target_id"], int(row["dim"]), row["method_id"], int(row["num_particles"])): row
        for row in directional_residuals
    }
    row_summaries: list[dict[str, Any]] = []
    for target_id in TARGET_IDS:
        for dim in (1, 2, 3):
            reference = reference_by_key[_combo_key(target_id, dim, "reference", 0)]
            for method_id in METHOD_IDS:
                counts = list(BASE_PARTICLE_COUNTS)
                if (target_id, dim, method_id) in triggered:
                    counts.append(TRIGGERED_PARTICLE_COUNT)
                for num_particles in counts:
                    directional = final_directional.get((target_id, dim, method_id, num_particles))
                    summary = _row_summary(
                        target_id=target_id,
                        dim=dim,
                        method_id=method_id,
                        num_particles=num_particles,
                        rows=rows,
                        reference=reference,
                        directional=directional,
                    )
                    summary["third_particle_count_triggered"] = (
                        num_particles == TRIGGERED_PARTICLE_COUNT
                    )
                    summary["third_particle_count_reasons"] = triggered.get(
                        (target_id, dim, method_id),
                        [],
                    )
                    summary["final_reported_particle_count"] = final_count_by_combo[
                        (target_id, dim, method_id)
                    ]
                    row_summaries.append(summary)

    decision = (
        "PASS_P8_P44_DPF_BLOCKER_CLOSURE_READY_FOR_REVIEW"
        if _full_execution_structurally_complete(rows, references, directional_residuals, row_summaries)
        else "BLOCKED_P8_P44_DPF_ADAPTER_GATE"
    )
    return _payload(
        decision=decision,
        references=references,
        ledh_conventions=ledh_conventions,
        rows=rows,
        directional_residuals=directional_residuals,
        row_summaries=row_summaries,
        m2_dim1_gate=gate,
        veto_diagnostics=_veto_diagnostics(rows, references, directional_residuals, gate),
    )


def _safe_dpf_row(
    *,
    method_id: str,
    target_id: str,
    dim: int,
    seed: int,
    num_particles: int,
    reference: dict[str, Any],
) -> dict[str, Any]:
    try:
        return _dpf_row(
            method_id=method_id,
            target_id=target_id,
            dim=dim,
            seed=seed,
            num_particles=num_particles,
            reference=reference,
        )
    except Exception as exc:  # preserve measured blocker rather than losing the run
        theta_dim = len(TARGET_SPECS[target_id]["theta"])
        return {
            "target_id": target_id,
            "dim": int(dim),
            "method_id": method_id,
            "seed": int(seed),
            "num_particles": int(num_particles),
            "value": float("nan"),
            "value_error": float("nan"),
            "per_observation_value_error": float("nan"),
            "fixed_branch_gradient": [float("nan")] * theta_dim,
            "gradient_error": [float("nan")] * theta_dim,
            "gradient_error_norm": float("nan"),
            "relative_score_error": float("nan"),
            "reference_score_norm": float(reference["score_norm"]),
            "finite": False,
            "gradient_object": "fixed_branch_score",
            "stochastic_score_claim": "not_claimed",
            "branch_signature": [],
            "branch_record_count": 0,
            "branch_records": [],
            "diagnostics": {
                "blocked_reason": type(exc).__name__,
                "blocked_message": str(exc),
            },
            "row_decision": "BLOCKED_WITH_REVIEWED_REASON",
        }


def _safe_directional_residual(
    *,
    target_id: str,
    dim: int,
    method_id: str,
    seed: int,
    num_particles: int,
    ad_gradient: list[float],
) -> dict[str, Any]:
    try:
        return _directional_residual(
            target_id=target_id,
            dim=dim,
            method_id=method_id,
            seed=seed,
            num_particles=num_particles,
            ad_gradient=ad_gradient,
        )
    except Exception as exc:
        return {
            "target_id": target_id,
            "dim": int(dim),
            "method_id": method_id,
            "seed": int(seed),
            "num_particles": int(num_particles),
            "fd_step": FD_STEP,
            "direction_count": 0,
            "directional_abs_residuals": [],
            "max_abs_directional_residual": float("nan"),
            "branch_signature_stable": False,
            "branch_signature": [],
            "blocked_reason": type(exc).__name__,
            "blocked_message": str(exc),
        }


def _payload(
    *,
    decision: str,
    references: list[dict[str, Any]],
    ledh_conventions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    directional_residuals: list[dict[str, Any]],
    row_summaries: list[dict[str, Any]],
    m2_dim1_gate: dict[str, Any],
    veto_diagnostics: dict[str, bool],
) -> dict[str, Any]:
    manifest = environment_manifest(
        command="CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-p8-mpl python -m "
        + MODULE_PATH,
        pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
    )
    manifest.update(
        {
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "review_ledger_path": REVIEW_LEDGER_PATH,
            "amended_p6_path": AMENDED_P6_PATH,
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "seed_list": list(SEEDS),
            "particle_counts": list(BASE_PARTICLE_COUNTS),
            "triggered_particle_count": TRIGGERED_PARTICLE_COUNT,
            "data_version": "P44 local deterministic target fixtures",
        }
    )
    return {
        "metadata_date": "2026-06-09",
        "created_at_utc": utc_now(),
        "phase": "P8",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": (
            "Can the P44 DPF blocker rows be converted from historical N/A "
            "cells to measured same-target value and fixed-branch gradient rows?"
        ),
        "review_gate": {
            "plan_review_status": "VERDICT_AGREE_ITERATION_5",
            "plan_path": PLAN_PATH,
            "ledger_path": REVIEW_LEDGER_PATH,
        },
        "targets": TARGET_IDS,
        "methods": METHOD_IDS,
        "rows": rows,
        "reference_rows": references,
        "ledh_convention_records": ledh_conventions,
        "row_summaries": row_summaries,
        "directional_residuals": directional_residuals,
        "branch_records": _flatten_branch_records(rows),
        "m2_dim1_gate": m2_dim1_gate,
        "veto_diagnostics": veto_diagnostics,
        "decision_table": _decision_table(decision, veto_diagnostics, row_summaries),
        "post_run_red_team": _post_run_red_team(decision),
        "run_manifest": manifest,
        "nonclaims": _nonclaims(),
    }


def _combo_key(target_id: str, dim: int, method_id: str, num_particles: int) -> str:
    return f"{target_id}|dim={int(dim)}|{method_id}|n={int(num_particles)}"


def _ledh_convention_record(target_id: str, dim: int) -> dict[str, Any]:
    spec = TARGET_SPECS[target_id]
    family = str(spec["family"])
    theta = tf.constant(spec["theta"], dtype=DTYPE)
    parts = _physical_parts(family, theta, dim)
    observation_jacobian = {
        "m2": "1 + 3*cubic*x^2",
        "m3": "2*x",
        "m4": "1",
    }[family]
    transition_prior_mean = {
        "m2": "rho*x_prev; initial mean rho*raw_initial_mean",
        "m3": "rho*x_prev; initial mean rho*raw_initial_mean",
        "m4": "rho*x_prev + nonlin*tanh(x_prev); initial proposal uses quadrature moment of raw pre-transition mixture",
    }[family]
    return {
        "target_id": target_id,
        "dim": int(dim),
        "method_id": "dpf_ledh_pfpf_ot",
        "transition_prior_mean": transition_prior_mean,
        "transition_covariance": tensor_to_json(tf.linalg.diag(parts["transition_variance"])),
        "observation_function": {
            "m2": "h(x)=x+cubic*x^3",
            "m3": "h(x)=x^2",
            "m4": "h(x)=x",
        }[family],
        "observation_jacobian": observation_jacobian,
        "observation_covariance": tensor_to_json(tf.linalg.diag(parts["observation_variance"])),
        "residual_convention": "observed_minus_predicted",
        "local_linearization_point": "per-particle pre-flow proposal",
        "pfpf_corrected_weight_formula": "log_w + log_target_prior_or_transition + log_observation - log_q0 + forward_log_det",
        "initial_timing": "t0 observes initial latent; no transition before first observation",
        "complete": True,
    }


def _m2_dim1_gate(
    rows: list[dict[str, Any]],
    directional_residuals: list[dict[str, Any]],
    reference: dict[str, Any],
    ledh_conventions: list[dict[str, Any]],
) -> dict[str, Any]:
    gate_rows = [
        row
        for row in rows
        if row["target_id"] == "p44_m2_cubic_additive_gaussian_panel"
        and int(row["dim"]) == 1
        and int(row["num_particles"]) == GATE_PARTICLE_COUNT
    ]
    reasons: list[str] = []
    if len(gate_rows) != len(SEEDS) * len(METHOD_IDS):
        reasons.append("missing_gate_rows")
    if not bool(reference["finite"]):
        reasons.append("nonfinite_dense_reference")
    if any(not bool(row["finite"]) for row in gate_rows):
        reasons.append("nonfinite_gate_value_or_score")
    if any(int(row.get("branch_record_count", 0)) <= 0 for row in gate_rows):
        reasons.append("missing_branch_records")
    if any(
        row["method_id"] == "dpf_ledh_pfpf_ot"
        and not _has_ledh_convention(ledh_conventions, row["target_id"], int(row["dim"]))
        for row in gate_rows
    ):
        reasons.append("missing_ledh_convention_record")
    gate_directionals = [
        row
        for row in directional_residuals
        if row["target_id"] == "p44_m2_cubic_additive_gaussian_panel"
        and int(row["dim"]) == 1
        and int(row["num_particles"]) == GATE_PARTICLE_COUNT
    ]
    if len(gate_directionals) != len(METHOD_IDS):
        reasons.append("missing_directional_residuals")
    for row in gate_directionals:
        if not bool(row["branch_signature_stable"]):
            reasons.append(f"{row['method_id']}_directional_branch_instability")
        if not math.isfinite(float(row["max_abs_directional_residual"])):
            reasons.append(f"{row['method_id']}_directional_residual_nonfinite")
        elif float(row["max_abs_directional_residual"]) > 0.25 * max(
            1.0,
            float(reference["score_norm"]),
        ):
            reasons.append(f"{row['method_id']}_directional_residual_above_gate_band")
    return {
        "status": "PASS_P8_M2_DIM1_ADAPTER_GATE" if not reasons else "FAIL_P8_M2_DIM1_ADAPTER_GATE",
        "target_id": "p44_m2_cubic_additive_gaussian_panel",
        "dim": 1,
        "num_particles": GATE_PARTICLE_COUNT,
        "seeds": list(SEEDS),
        "methods": list(METHOD_IDS),
        "reason": "all gate checks passed" if not reasons else "; ".join(sorted(set(reasons))),
        "reasons": sorted(set(reasons)),
    }


def _has_ledh_convention(
    ledh_conventions: list[dict[str, Any]],
    target_id: str,
    dim: int,
) -> bool:
    return any(
        row["target_id"] == target_id and int(row["dim"]) == int(dim) and bool(row["complete"])
        for row in ledh_conventions
    )


def _full_execution_structurally_complete(
    rows: list[dict[str, Any]],
    references: list[dict[str, Any]],
    directional_residuals: list[dict[str, Any]],
    row_summaries: list[dict[str, Any]],
) -> bool:
    if len(references) != len(TARGET_IDS) * 3:
        return False
    if any(not bool(row["finite"]) for row in references):
        return False
    final_summaries = [
        row
        for row in row_summaries
        if int(row["num_particles"]) == int(row["final_reported_particle_count"])
    ]
    if len(final_summaries) != len(TARGET_IDS) * 3 * len(METHOD_IDS):
        return False
    if any(int(row["seed_count"]) != len(SEEDS) for row in final_summaries):
        return False
    if any(int(row["branch_record_count"]) <= 0 for row in final_summaries):
        return False
    if len(directional_residuals) < len(TARGET_IDS) * 3 * len(METHOD_IDS):
        return False
    return bool(rows)


def _veto_diagnostics(
    rows: list[dict[str, Any]],
    references: list[dict[str, Any]],
    directional_residuals: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, bool]:
    return {
        "dense_reference_missing_or_nonfinite": len(references) != len(TARGET_IDS) * 3
        or any(not bool(row["finite"]) for row in references),
        "m2_dim1_adapter_gate_failed": gate["status"] != "PASS_P8_M2_DIM1_ADAPTER_GATE",
        "dpf_row_nonfinite": any(not bool(row["finite"]) for row in rows),
        "missing_branch_records": any(int(row.get("branch_record_count", 0)) <= 0 for row in rows),
        "missing_directional_residuals": not directional_residuals,
        "directional_branch_instability": any(
            not bool(row.get("branch_signature_stable", False)) for row in directional_residuals
        ),
        "directional_residual_nonfinite": any(
            not math.isfinite(float(row.get("max_abs_directional_residual", float("nan"))))
            for row in directional_residuals
        ),
        "fixed_branch_gradient_mislabeled_stochastic_score": any(
            row.get("gradient_object") != "fixed_branch_score"
            or row.get("stochastic_score_claim") != "not_claimed"
            for row in rows
        ),
        "historical_p6_overwritten": False,
        "runner_numerics_not_implemented": False,
    }


def _flatten_branch_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        diagnostics = row.get("diagnostics", {})
        if "blocked_reason" in diagnostics:
            continue
        for branch in row.get("branch_records", []):
            records.append(
                {
                    "target_id": row["target_id"],
                    "dim": int(row["dim"]),
                    "method_id": row["method_id"],
                    "seed": int(row["seed"]),
                    "num_particles": int(row["num_particles"]),
                    **branch,
                }
            )
    return records


def _decision_table(
    decision: str,
    veto_diagnostics: dict[str, bool],
    row_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    final_summaries = [
        row
        for row in row_summaries
        if int(row.get("num_particles", -1)) == int(row.get("final_reported_particle_count", -2))
    ]
    failed = [row for row in final_summaries if row.get("row_decision") == "FAILED_NUMERIC_BANDS"]
    promoted = [
        row
        for row in final_summaries
        if row.get("row_decision") == "PROMOTED_EXACT_TARGET_CLOSENESS"
    ]
    diagnostic = [
        row
        for row in final_summaries
        if row.get("row_decision") == "DIAGNOSTIC_ONLY_MEASURED"
    ]
    return [
        {
            "decision": decision,
            "primary_criterion_status": (
                f"{len(final_summaries)} final measured rows; "
                f"{len(promoted)} promoted, {len(diagnostic)} diagnostic, {len(failed)} failed numeric bands"
            ),
            "veto_diagnostic_status": {
                key: value for key, value in veto_diagnostics.items() if bool(value)
            }
            or "no structural vetoes",
            "main_uncertainty": "fixed-branch DPF gradients and finite particle counts do not establish stochastic-score correctness",
            "next_justified_action": "run Claude read-only result review and inspect failed/diagnostic row labels",
            "not_concluded": "no production, HMC, GPU, public API, or universal DPF superiority claim",
        }
    ]


def _post_run_red_team(decision: str) -> dict[str, Any]:
    if decision == "BLOCKED_P8_P44_DPF_ADAPTER_GATE":
        return {
            "strongest_alternative_explanation": "The gate may fail because of adapter/tuning choices rather than because DPF cannot handle P44.",
            "what_result_would_overturn": "A reviewed M2 dim-1 rerun with finite rows, stable directional residuals, and branch records passing the adapter gate.",
            "weakest_part_of_evidence": "Execution stopped before M3/M4 or higher dimensions by design.",
            "blocker_manifest_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        }
    return {
        "strongest_alternative_explanation": "Measured rows may reflect finite-particle and fixed-branch behavior rather than stable stochastic-score performance.",
        "what_result_would_overturn": "Multi-ladder reruns with larger particles or independently implemented same-target adapters that reverse the value/score conclusions.",
        "weakest_part_of_evidence": "Only paired seeds and finite particle counts are used; no full stochastic-resampling gradient claim is made.",
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "review_gate",
        "rows",
        "reference_rows",
        "row_summaries",
        "directional_residuals",
        "branch_records",
        "m2_dim1_gate",
        "veto_diagnostics",
        "decision_table",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P8ValidationError(f"missing payload fields {sorted(missing)}")
    manifest = payload["run_manifest"]
    for field in (
        "command",
        "pre_import_cuda_visible_devices",
        "json_path",
        "report_path",
        "plan_path",
        "result_path",
        "amended_p6_path",
        "seed_list",
        "particle_counts",
    ):
        if field not in manifest:
            raise P8ValidationError(f"run_manifest missing {field}")
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise P8ValidationError("TensorFlow was not forced CPU-only before import")
    if payload["review_gate"]["plan_review_status"] != "VERDICT_AGREE_ITERATION_5":
        raise P8ValidationError("plan review gate is not closed")
    if payload["decision"] not in {
        "BLOCKED_P8_P44_DPF_ADAPTER_GATE",
        "PASS_P8_P44_DPF_BLOCKER_CLOSURE_READY_FOR_REVIEW",
    }:
        raise P8ValidationError(f"invalid P8 decision {payload['decision']}")
    if payload["decision"] == "PASS_P8_P44_DPF_BLOCKER_CLOSURE_READY_FOR_REVIEW":
        if not payload["row_summaries"]:
            raise P8ValidationError("passing payload missing row summaries")
        if not payload["directional_residuals"]:
            raise P8ValidationError("passing payload missing directional residuals")
        if not payload["branch_records"]:
            raise P8ValidationError("passing payload missing branch records")
        required_branch_fields = {
            "ess",
            "ess_ratio",
            "resampled",
            "resampling_method",
            "transport_active_mask_status",
            "time_index",
        }
        for record in payload["branch_records"]:
            missing_branch = required_branch_fields.difference(record)
            if missing_branch:
                raise P8ValidationError(
                    f"branch record missing fields {sorted(missing_branch)}"
                )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P8 Result: P44 DPF Blocker Closure",
        "",
        "metadata_date: 2026-06-09",
        "phase: P8",
        f"status: {payload['decision']}",
        "",
        "## Review Gate",
        "",
        f"- plan review: `{payload['review_gate']['plan_review_status']}`",
        f"- plan: `{payload['review_gate']['plan_path']}`",
        f"- ledger: `{payload['review_gate']['ledger_path']}`",
        "",
        "## Gate Status",
        "",
        f"- P44-M2 dim-1 adapter gate: `{payload['m2_dim1_gate']['status']}`",
        f"- reason: `{payload['m2_dim1_gate'].get('reason', 'N/A')}`",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["decision_table"]:
        lines.append(
            f"| `{row['decision']}` | {row['primary_criterion_status']} | "
            f"{row['veto_diagnostic_status']} | {row['main_uncertainty']} | "
            f"{row['next_justified_action']} | {row['not_concluded']} |"
        )
    lines.extend(
        [
            "",
            "## Run Manifest",
            "",
            f"- command: `{payload['run_manifest']['command']}`",
            f"- git branch: `{payload['run_manifest']['branch']}`",
            f"- git commit: `{payload['run_manifest']['commit']}`",
            f"- CPU/GPU status: CPU-only, `CUDA_VISIBLE_DEVICES={payload['run_manifest']['pre_import_cuda_visible_devices']}` before TensorFlow import",
            f"- visible GPU devices: `{payload['run_manifest']['gpu_devices_visible']}`",
            f"- seeds: `{payload['run_manifest']['seed_list']}`",
            f"- particle counts: `{payload['run_manifest']['particle_counts']}`",
            f"- JSON: `{payload['run_manifest']['json_path']}`",
            f"- report: `{payload['run_manifest']['report_path']}`",
            f"- result: `{payload['run_manifest']['result_path']}`",
            f"- amended P6 display: `{payload['run_manifest']['amended_p6_path']}`",
            f"- wall time seconds: `{payload['run_manifest']['wall_time_seconds']}`",
            "",
            "## Nonclaims",
            "",
            *[f"- {item}" for item in payload["nonclaims"]],
            "",
        ]
    )
    return "\n".join(lines)


def _amended_p6_markdown(payload: dict[str, Any]) -> str:
    final_rows = [
        row
        for row in payload["row_summaries"]
        if int(row["num_particles"]) == int(row["final_reported_particle_count"])
    ]
    lines = [
        "# P6 Amended Display With P8 DPF Metrics",
        "",
        "metadata_date: 2026-06-09",
        f"status: {payload['decision']}",
        "",
        "This file amends the historical P6 display only; it does not overwrite the original P6 result artifact.",
        "",
        f"- P8 JSON: `{payload['run_manifest']['json_path']}`",
        f"- P8 result: `{payload['run_manifest']['result_path']}`",
        f"- P8 M2 dim-1 gate: `{payload['m2_dim1_gate']['status']}`",
        "",
        "## Filled P44 DPF Cells",
        "",
        "| Historical P6 target | DPF method | Dim | Final particles | Filled value cell | Filled gradient cell | P8 row decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in final_rows:
        lines.append(
            f"| `{row['target_id']}` | `{row['method_id']}` | {row['dim']} | "
            f"{row['num_particles']} | value RMSE/obs `{row['per_observation_value_error_rmse']:.6g}` | "
            f"mean relative score `{row['mean_relative_score_error']:.6g}` | "
            f"`{row['row_decision']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Rows marked `PROMOTED_EXACT_TARGET_CLOSENESS` satisfy the reviewed P8 promotion bands at the final reported particle count. Rows marked `DIAGNOSTIC_ONLY_MEASURED` fill the historical `N/A` cells with finite measured value and fixed-branch gradient metrics, but they do not satisfy all promotion bands.",
            "",
            "The amended cells use fixed-branch AD score metrics only. They do not claim full stochastic-resampling score correctness, HMC readiness, production readiness, GPU readiness, or universal DPF superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def _nonclaims() -> list[str]:
    return [
        "P8 does not overwrite historical P6 results.",
        "DPF gradients are fixed-branch AD gradients, not full stochastic-resampling scores.",
        "A filled N/A cell is not a promotion unless the predeclared P8 bands pass.",
        "No HMC, production, public API, GPU, or default-policy readiness is concluded.",
    ]


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_digest"}
    return stable_digest(stable)


if __name__ == "__main__":
    raise SystemExit(main())
