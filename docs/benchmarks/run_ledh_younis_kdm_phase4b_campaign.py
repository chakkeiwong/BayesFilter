#!/usr/bin/env python3
"""Diagnostic matrix-LGSSM campaign for the complete Phase 4B KDM route.

Numerical filter kernels are the existing TensorFlow analytical endpoints.
Every accepted result consumes their validity flags on the host. Statistics
are post-run diagnostic TensorFlow reductions; no NumPy runtime path is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/bayesfilter-ledh-younis-kdm-phase4b-campaign-amendment-20260909.md"
RHO_GRID = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6)
MARKS = (
    "responsibility_conditional_mean_no_scatter_v1",
    "fixed_selected_component_mark_v1",
)
SCOPES = ((32, 5), (64, 20))
SPLITS = {"preflight": 0, "calibration": 1, "pilot": 2, "validation": 3}
MODEL = {
    "theta": 0.72,
    "A_base": [[0.0, 0.12], [-0.08, 0.10]],
    "A_direction": [[1.0, 0.0], [0.0, 0.7]],
    "H": [[1.0, 0.25], [-0.15, 0.9]],
    "Q": [[0.12, 0.025], [0.025, 0.09]],
    "R": [[0.22, 0.035], [0.035, 0.18]],
    "P0": [[0.8, 0.12], [0.12, 0.6]],
}
CONTROLS = {
    "flow_substeps": 6,
    "reset_policy": "contract_e",
    "reset_epsilon": 2.0,
    "reset_sinkhorn_steps": 4,
    "reset_balance_steps": 2,
    "reset_ridge": 1e-5,
    "correction_steps": 1,
    "correction_strength": 0.2,
    "correction_lm_damping": 1e-2,
    "correction_lm_scale_floor": 1e-4,
    "correction_trust_radius": 0.5,
    "pairwise_steps": 1,
    "pairwise_strength": 0.02,
    "pairwise_rms_cap": 2.0,
    "coordinate_cap": 0.95,
    "coordinate_cap_power": 8,
    "annealed_stages": 1,
}
SOURCE_PATHS = (
    "docs/benchmarks/run_ledh_younis_kdm_phase4b_campaign.py",
    "bayesfilter/highdim/ledh_canonical_score_tf.py",
    "bayesfilter/highdim/ledh_canonical_score_stages_tf.py",
    "bayesfilter/highdim/ledh_canonical_reset_score_tf.py",
    "bayesfilter/highdim/higher_moment_contract_e.py",
    "bayesfilter/highdim/ledh_younis_kdm_tf.py",
    "bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py",
    "bayesfilter/highdim/ledh_younis_kdm_resampling_tf.py",
    "bayesfilter/highdim/ledh_younis_kdm_lgssm_reference_tf.py",
    "docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex",
    PLAN,
)


class CampaignVeto(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.seconds = seconds

    @property
    def remaining(self):
        return self.seconds - (time.monotonic() - self.started)

    def check(self, required=0.0):
        if self.remaining <= required:
            raise BudgetExhausted(
                f"remaining {self.remaining:.1f}s cannot cover projected {required:.1f}s"
            )


def seed_key(attempt, scope, split, replicate):
    if not (1 <= attempt <= 100 and 0 <= scope <= 2 and 0 <= replicate < 1000):
        raise ValueError("seed coordinate outside the declared schedule")
    if split not in SPLITS:
        raise ValueError("unknown split")
    return [
        attempt * 1_000_000 + scope * 100_000 + SPLITS[split] * 10_000 + replicate,
        20260909,
    ]


def _write(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _float(x):
    return float(tf.reshape(x, []).numpy())


def _bool(x):
    return bool(tf.reduce_all(x).numpy())


def _stack(trace):
    keys = (
        "higher_moment_valid",
        "higher_moment_pairwise_configured",
        "higher_moment_pairwise_target_mask",
        "higher_moment_maximum_pairwise_pre_cap_particle_rms",
        "higher_moment_maximum_pairwise_post_cap_particle_rms",
        "higher_moment_minimum_pairwise_particle_cap_scale",
        "higher_moment_fraction_coordinatewise_cap_active",
        "higher_moment_minimum_coordinatewise_cap_derivative",
    )
    return {key: tf.stack([step[key] for step in trace]) for key in keys}


def _compact(result):
    """Retain validity and already-computed cap diagnostics at the consumer."""
    finite = tf.math.is_finite(result["value"]) & tf.reduce_all(
        tf.math.is_finite(result["score"])
    )
    output = {
        "value": result["value"],
        "score": tf.reshape(result["score"], []),
        "valid": result["valid"] & finite,
    }
    if "higher_moment_valid" in result:
        output.update(
            {
                "higher_moment_valid": tf.reduce_all(result["higher_moment_valid"]),
                "pairwise_configured": tf.reduce_all(
                    result["higher_moment_pairwise_configured"]
                ),
                "pairwise_mask_count": tf.reduce_sum(
                    tf.cast(result["higher_moment_pairwise_target_mask"] > 0, tf.int32)
                ),
                "pairwise_pre_cap_rms": tf.reduce_max(
                    result["higher_moment_maximum_pairwise_pre_cap_particle_rms"]
                ),
                "pairwise_post_cap_rms": tf.reduce_max(
                    result["higher_moment_maximum_pairwise_post_cap_particle_rms"]
                ),
                "pairwise_min_cap_scale": tf.reduce_min(
                    result["higher_moment_minimum_pairwise_particle_cap_scale"]
                ),
                "coordinate_cap_active_fraction": tf.reduce_mean(
                    result["higher_moment_fraction_coordinatewise_cap_active"]
                ),
                "coordinate_min_cap_derivative": tf.reduce_min(
                    result["higher_moment_minimum_coordinatewise_cap_derivative"]
                ),
            }
        )
        output["valid"] &= output["higher_moment_valid"]
    if "pair_count" in result:
        output.update(
            {
                "pair_count": result["pair_count"],
                "anchor_log_ratio_error": result["maximum_anchor_log_ratio_error"],
                "raw_weight_tangent_max": tf.reduce_max(
                    tf.abs(result["d_importance_weight_sums"])
                ),
                "minimum_bandwidth_eigenvalue": result["minimum_bandwidth_eigenvalue"],
                "responsibility_entropy": -tf.reduce_mean(
                    tf.reduce_sum(
                        tf.math.xlogy(
                            result["responsibilities"], result["responsibilities"]
                        ),
                        axis=-1,
                    )
                ),
            }
        )
    if "minimum_ess" in result:
        output["minimum_ess"] = result["minimum_ess"]
    return output


class Engine:
    """Stable shapes for one (N,T) scope, using the full shared executor."""

    def __init__(self, n, horizon, *, dtype=tf.float64, jit_compile=True):
        from bayesfilter.highdim.ledh_canonical_score_tf import (
            NonlinearScoreModel,
            canonical_value_and_analytical_score,
        )
        from bayesfilter.highdim.ledh_younis_kdm_integrated_tf import (
            make_integrated_linear_gaussian_kdm_kernel,
        )
        from bayesfilter.highdim.ledh_younis_kdm_resampling_tf import (
            make_resampling_anchor_kernel,
            make_resampling_replay_kernel,
        )
        from bayesfilter.highdim.ledh_younis_kdm_lgssm_reference_tf import (
            make_bootstrap_lgssm_fixed_stream_kernel,
            matrix_lgssm_value_and_directional_score,
        )

        if n % 4 or horizon < 2:
            raise ValueError("this matrix fixture requires N divisible by 4 and T>=2")
        self.n, self.horizon, self.dtype = n, horizon, dtype
        self.theta = tf.constant([MODEL["theta"]], dtype)
        self.matrices = {
            k: tf.constant(v, dtype) for k, v in MODEL.items() if k != "theta"
        }
        a, da, h, q, r, p0 = [
            self.matrices[k] for k in ("A_base", "A_direction", "H", "Q", "R", "P0")
        ]
        self.options = dict(
            CONTROLS,
            reset_design=tf.tile(
                tf.concat([tf.eye(2, dtype=dtype), -tf.eye(2, dtype=dtype)], 0),
                [n // 4, 1],
            ),
        )
        self.model = NonlinearScoreModel(
            transition_mean_fn=lambda theta, x: tf.linalg.matmul(
                x, a + theta[0] * da, transpose_b=True
            ),
            transition_mean_tangent_fn=lambda theta, x, dx: tf.linalg.matmul(
                x, da, transpose_b=True
            )
            + tf.linalg.matmul(dx, a + theta[0] * da, transpose_b=True),
            observation_fn=lambda x: tf.linalg.matmul(x, h, transpose_b=True),
            observation_jacobian_fn=lambda x: tf.broadcast_to(
                h, [tf.shape(x)[0], 2, 2]
            ),
            observation_tangent_fn=lambda x, dx: tf.linalg.matmul(
                dx, h, transpose_b=True
            ),
            observation_jacobian_tangent_fn=lambda x, dx: tf.zeros(
                [tf.shape(x)[0], 2, 2], dtype
            ),
            process_covariance=q,
            observation_covariance=r,
        )
        common = dict(
            theta_dimension=1,
            particle_count=n,
            state_dimension=2,
            observation_dimension=2,
            horizon=horizon,
            canonical_options=self.options,
            dtype=dtype,
            jit_compile=jit_compile,
        )
        self.anchor = {
            mark: make_resampling_anchor_kernel(
                self.model, covariance_mark_policy=mark, **common
            )
            for mark in MARKS
        }
        self.replay = {
            mark: make_resampling_replay_kernel(
                self.model, covariance_mark_policy=mark, **common
            )
            for mark in MARKS
        }
        self.phase4a = make_integrated_linear_gaussian_kdm_kernel(self.model, **common)
        self.phase4a_zero = make_integrated_linear_gaussian_kdm_kernel(
            self.model, bandwidth_is_zero=True, **common
        )
        self.bootstrap = make_bootstrap_lgssm_fixed_stream_kernel(
            transition_matrix_base=a,
            transition_matrix_direction=da,
            observation_matrix=h,
            process_covariance=q,
            observation_covariance=r,
            particle_count=n,
            horizon=horizon,
            dtype=dtype,
            jit_compile=jit_compile,
        )
        signature = [
            tf.TensorSpec([1], dtype),
            tf.TensorSpec([n, 2], dtype),
            tf.TensorSpec([n, 2, 2], dtype),
            tf.TensorSpec([horizon, n, 2], dtype),
            tf.TensorSpec([horizon, 2], dtype),
        ]

        @tf.function(
            input_signature=signature, jit_compile=jit_compile, autograph=False
        )
        def atom(theta, states, covs, noises, observations):
            value, score, trace = canonical_value_and_analytical_score(
                self.model,
                theta,
                states,
                covs,
                noises,
                observations,
                with_score=True,
                return_trace=True,
                **self.options,
            )
            result = {
                "value": value,
                "score": score,
                "valid": tf.constant(True),
                **_stack(trace),
            }
            return _compact(result)

        self.atom = atom
        self.atom_endpoint = canonical_value_and_analytical_score

        @tf.function(
            input_signature=[
                tf.TensorSpec([1], dtype),
                tf.TensorSpec([horizon, 2], dtype),
            ],
            jit_compile=jit_compile,
            autograph=False,
        )
        def oracle(theta, observations):
            value, score = matrix_lgssm_value_and_directional_score(
                observations, a + theta[0] * da, da, h, tf.zeros([2], dtype), p0, q, r
            )
            return {
                "value": value,
                "score": score,
                "valid": tf.math.is_finite(value) & tf.math.is_finite(score),
            }

        self.oracle = oracle

        # Compact wrappers leave all value/score operations intact. They keep
        # the required validity/cap diagnostics while avoiding full trace copies.
        def compact_kernel(kernel):
            def packed(*args):
                return _compact(kernel(*args))

            return tf.function(
                packed,
                input_signature=kernel.input_signature,
                jit_compile=jit_compile,
                autograph=False,
            )

        self.compact_anchors = {
            mark: compact_kernel(self.anchor[mark]) for mark in MARKS
        }

        def packed_a(*args):
            return _compact(self.phase4a(*args))

        self.compact_a = tf.function(
            packed_a,
            input_signature=self.phase4a.input_signature,
            jit_compile=jit_compile,
            autograph=False,
        )

        @tf.function(
            input_signature=[tf.TensorSpec([2], tf.int32)],
            jit_compile=jit_compile,
            autograph=False,
        )
        def generate(seed):
            def normal(shape, role):
                return tf.random.stateless_normal(
                    shape,
                    tf.random.experimental.stateless_fold_in(seed, role),
                    dtype=dtype,
                )

            def uniform(shape, role):
                return tf.random.stateless_uniform(
                    shape,
                    tf.random.experimental.stateless_fold_in(seed, role),
                    dtype=dtype,
                )

            lp, lq, lr = (
                tf.linalg.cholesky(p0),
                tf.linalg.cholesky(q),
                tf.linalg.cholesky(r),
            )
            state = tf.linalg.matvec(lp, normal([2], 1))
            process_noise, observation_noise = (
                normal([horizon, 2], 2),
                normal([horizon, 2], 3),
            )
            observations = []
            for t in range(horizon):
                state = tf.linalg.matvec(
                    a + self.theta[0] * da, state
                ) + tf.linalg.matvec(lq, process_noise[t])
                observations.append(
                    tf.linalg.matvec(h, state)
                    + tf.linalg.matvec(lr, observation_noise[t])
                )
            return {
                "initial_states": tf.linalg.matmul(
                    normal([n, 2], 4), lp, transpose_b=True
                ),
                "initial_covariances": tf.broadcast_to(p0, [n, 2, 2]),
                "noises": normal([horizon, n, 2], 5),
                "observations": tf.stack(observations),
                "uniforms": uniform([horizon, n], 6),
                "kdm_noises": normal([horizon, n, 2], 7),
                "bootstrap_offsets": uniform([horizon], 8),
            }

        self.generate = generate

    def shared(self, path, theta=None):
        return (
            self.theta if theta is None else theta,
            path["initial_states"],
            path["initial_covariances"],
            path["noises"],
            path["observations"],
        )

    def kdm_inputs(self, path, rho, theta=None):
        bandwidth = tf.broadcast_to(
            tf.constant(rho**2, self.dtype) * self.matrices["Q"], [self.horizon, 2, 2]
        )
        return (*self.shared(path, theta), bandwidth, tf.zeros_like(bandwidth))

    def a_inputs(self, path, rho, theta=None):
        bandwidth = tf.broadcast_to(
            tf.constant(rho**2, self.dtype) * self.matrices["Q"],
            [self.horizon, self.n, 2, 2],
        )
        return (
            *self.shared(path, theta),
            self.matrices["H"],
            tf.zeros([2, 2], self.dtype),
            bandwidth,
            tf.zeros_like(bandwidth),
        )

    def raw(self, method, path, rho=0.0, mark=MARKS[0], theta=None):
        theta = self.theta if theta is None else theta
        if method == "atom":
            return self.atom(*self.shared(path, theta))
        if method == "oracle":
            return self.oracle(theta, path["observations"])
        if method == "bootstrap":
            return self.bootstrap(
                theta,
                path["initial_states"],
                path["noises"],
                path["observations"],
                path["bootstrap_offsets"],
            )
        if method == "phase4a":
            return self.compact_a(*self.a_inputs(path, rho, theta))
        if method == "phase4b":
            return self.compact_anchors[mark](
                *self.kdm_inputs(path, rho, theta), path["uniforms"], path["kdm_noises"]
            )
        raise ValueError(method)

    def evaluate(self, method, path, rho=0.0, mark=MARKS[0], budget=None):
        if budget is not None:
            budget.check()
        started = time.monotonic()
        result = (
            _compact(self.raw(method, path, rho, mark))
            if method == "bootstrap"
            else self.raw(method, path, rho, mark)
        )
        if not _bool(result["valid"]):
            raise CampaignVeto(f"{method} invalid at rho={rho}, mark={mark}")
        row = {k: v.numpy().item() for k, v in result.items()}
        if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
            raise CampaignVeto(f"nonfinite {method} diagnostic at rho={rho}")
        if method in ("atom", "phase4a", "phase4b"):
            if (
                not row["pairwise_configured"]
                or row["pairwise_mask_count"] != 2 * self.horizon
                or row["pairwise_pre_cap_rms"] <= 0
            ):
                raise CampaignVeto(f"{method} did not execute the pairwise correction")
        if method == "phase4b" and row["pair_count"] != self.horizon * self.n**2:
            raise CampaignVeto("missing full-mixture pairs")
        row["seconds"] = time.monotonic() - started
        return row


def _check_close(name, actual, expected, tolerance):
    error = _float(tf.reduce_max(tf.abs(actual - expected)))
    scale = max(1.0, _float(tf.reduce_max(tf.abs(expected))))
    if not math.isfinite(error) or error > tolerance * scale:
        raise CampaignVeto(f"{name}: error={error}, tolerance={tolerance * scale}")
    return error / scale


def preflight(engine, attempt=1):
    """Discriminating endpoint checks on the campaign's actual matrix model."""
    path = engine.generate(tf.constant(seed_key(attempt, 2, "preflight", 0), tf.int32))
    eps = tf.constant(1e-5, engine.dtype)
    plus, minus = engine.theta + eps, engine.theta - eps
    a = engine.matrices["A_base"] + engine.theta[0] * engine.matrices["A_direction"]
    radius = _float(tf.reduce_max(tf.abs(tf.linalg.eigvals(a))))
    if radius >= 1.0:
        raise CampaignVeto("unstable matrix fixture")
    checks = {"spectral_radius": radius, "methods": {}}
    for method in ("oracle", "atom", "phase4a"):
        center = engine.raw(method, path, rho=0.2)
        upper, lower = (
            engine.raw(method, path, rho=0.2, theta=plus),
            engine.raw(method, path, rho=0.2, theta=minus),
        )
        if not all(_bool(v["valid"]) for v in (center, upper, lower)):
            raise CampaignVeto(f"invalid preflight {method}")
        fd = (upper["value"] - lower["value"]) / (2 * eps)
        checks["methods"][method] = {
            "fd_relative_error": _check_close(method, center["score"], fd, 2e-5)
        }
    atom = engine.raw("atom", path)
    zero = engine.phase4a_zero(*engine.a_inputs(path, 0.0))
    if not _bool(zero["valid"]):
        raise CampaignVeto("invalid Phase 4A zero branch")
    checks["zero_value_error"] = _check_close(
        "zero value", zero["value"], atom["value"], 1e-10
    )
    checks["zero_score_error"] = _check_close(
        "zero score", zero["score"], atom["score"], 1e-10
    )
    for mark in MARKS:
        anchor = engine.anchor[mark](
            *engine.kdm_inputs(path, 0.2), path["uniforms"], path["kdm_noises"]
        )
        bank = (
            anchor["fixed_samples"],
            anchor["fixed_proposal_log_densities"],
            anchor["component_indices"],
        )
        replay = engine.replay[mark](*engine.kdm_inputs(path, 0.2), *bank)
        upper = engine.replay[mark](*engine.kdm_inputs(path, 0.2, plus), *bank)
        lower = engine.replay[mark](*engine.kdm_inputs(path, 0.2, minus), *bank)
        if not all(_bool(v["valid"]) for v in (anchor, replay, upper, lower)):
            raise CampaignVeto("invalid Phase 4B preflight")
        row = engine.evaluate("phase4b", path, 0.2, mark)
        row["fd_relative_error"] = _check_close(
            "replay score",
            replay["score"],
            (upper["value"] - lower["value"]) / (2 * eps),
            2e-5,
        )
        row["replay_value_error"] = _check_close(
            "replay value", replay["value"], anchor["value"], 1e-10
        )
        row["replay_score_error"] = _check_close(
            "replay score identity", replay["score"], anchor["score"], 1e-10
        )
        row["raw_weight_recurrence_error"] = _check_close(
            "raw weights",
            replay["incoming_log_weights"][1:],
            replay["outgoing_log_weights"][:-1],
            0.0,
        )
        row["raw_tangent_recurrence_error"] = _check_close(
            "raw tangents",
            replay["d_incoming_log_weights"][1:],
            replay["d_outgoing_log_weights"][:-1],
            0.0,
        )
        if not _bool(
            anchor["component_indices"]
            == tf.broadcast_to(tf.range(engine.n), [engine.horizon, engine.n])
        ):
            raise CampaignVeto(
                "uniform stratified sampling did not select each component once"
            )
        checks["methods"][mark] = row
    # Fixed-stream bootstrap FD requires the realized indices to agree.
    center = engine.raw("bootstrap", path)
    for step in (1e-5, 1e-6, 1e-7):
        upper = engine.raw(
            "bootstrap", path, theta=engine.theta + tf.constant(step, engine.dtype)
        )
        lower = engine.raw(
            "bootstrap", path, theta=engine.theta - tf.constant(step, engine.dtype)
        )
        if _bool(center["ancestor_indices"] == upper["ancestor_indices"]) and _bool(
            center["ancestor_indices"] == lower["ancestor_indices"]
        ):
            fd = (upper["value"] - lower["value"]) / tf.constant(2 * step, engine.dtype)
            checks["methods"]["bootstrap"] = {
                "step": step,
                "fd_relative_error": _check_close(
                    "bootstrap", center["score"], fd, 2e-5
                ),
            }
            break
    else:
        raise CampaignVeto("bootstrap preflight crossed resampling boundaries")
    invalid_offsets = tf.tensor_scatter_nd_update(
        path["bootstrap_offsets"],
        [[engine.horizon - 1]],
        tf.constant([1.25], engine.dtype),
    )
    try:
        invalid = engine.bootstrap(
            engine.theta,
            path["initial_states"],
            path["noises"],
            path["observations"],
            invalid_offsets,
        )
        if _bool(invalid["valid"]):
            raise CampaignVeto("bootstrap accepted an out-of-domain resampling offset")
        checks["bootstrap_invalid_offset_rejected"] = "returned_valid_flag"
    except tf.errors.InvalidArgumentError:
        checks["bootstrap_invalid_offset_rejected"] = "tensorflow_invalid_argument"
    checks["status"] = "PASS"
    return checks


def _mean(values):
    return _float(tf.reduce_mean(tf.constant(values, tf.float64)))


def _variance(values):
    x = tf.constant(values, tf.float64)
    return _float(
        tf.reduce_sum(tf.square(x - tf.reduce_mean(x)))
        / tf.cast(len(values) - 1, tf.float64)
    )


@tf.function(
    input_signature=[tf.TensorSpec([None], tf.float64), tf.TensorSpec([2], tf.int32)],
    jit_compile=False,
    autograph=False,
)
def _bootstrap_means(values, seed):
    # Explicit host-side, non-XLA reporting exception: variable group sizes do
    # not retrace a GPU scientific kernel, and all resamples remain paired.
    n = tf.shape(values)[0]
    indices = tf.random.stateless_uniform(
        [4000, n], seed, minval=0, maxval=n, dtype=tf.int32
    )
    return tf.sort(tf.reduce_mean(tf.gather(values, indices), axis=1))


def interval(values, seed):
    if len(values) < 2:
        return [None, None]
    with tf.device("/CPU:0"):
        means = _bootstrap_means(
            tf.constant(values, tf.float64), tf.constant(seed, tf.int32)
        )
    return [_float(means[99]), _float(means[3899])]


def power_requirement(pilot_candidate_errors, pilot_atom_errors, calibration_mse):
    if (
        len(pilot_candidate_errors) != len(pilot_atom_errors)
        or len(pilot_atom_errors) < 2
    ):
        raise ValueError("power needs paired pilot errors")
    if not math.isfinite(calibration_mse) or calibration_mse <= 0:
        raise CampaignVeto("no positive finite calibration MSE for power design")
    margin_errors = [
        b * b - 0.9 * a * a for b, a in zip(pilot_candidate_errors, pilot_atom_errors)
    ]
    variance = _variance(margin_errors)
    raw = math.ceil((1.96 + 0.84) ** 2 * variance / (0.1 * calibration_mse) ** 2)
    required = max(100, 20 * math.ceil(raw / 20))
    return {
        "required_paths": required,
        "executed_target_paths": min(500, required),
        "nominal_power_adequate": required <= 500,
        "pilot_margin_variance": variance,
        "design_true_relative_mse_reduction": 0.20,
        "required_relative_mse_reduction": 0.10,
    }


def summarize(rows, split_threshold, power, seed):
    groups = {
        "all": rows,
        "low_absolute_exact_score": [
            r for r in rows if abs(r["oracle"]["score"]) <= split_threshold
        ],
        "high_absolute_exact_score": [
            r for r in rows if abs(r["oracle"]["score"]) > split_threshold
        ],
    }
    report, vetoes = {}, []
    for group_index, (name, group) in enumerate(groups.items()):
        errors = {
            m: [r[m]["score"] - r["oracle"]["score"] for r in group]
            for m in ("atom", "bootstrap", "phase4a", "phase4b")
        }
        if not group:
            report[name] = {"count": 0, "status": "INSUFFICIENT_CONDITIONAL_EVIDENCE"}
            vetoes.append(name + ":empty")
            continue
        data = {"count": len(group), "methods": {}, "phase4b_comparisons": {}}
        for method, values in errors.items():
            variance = _variance(values) if len(values) > 1 else None
            data["methods"][method] = {
                "bias": _mean(values),
                "sample_variance": variance,
                "mse": _mean([v * v for v in values]),
                "mean_error_mcse": math.sqrt(variance / len(values))
                if variance is not None
                else None,
            }
        candidate = errors["phase4b"]
        for i, comparator in enumerate(("atom", "bootstrap", "phase4a")):
            diffs = [b * b - a * a for b, a in zip(candidate, errors[comparator])]
            ci = interval(diffs, [seed, group_index * 10 + i])
            mean = _mean(diffs)
            data["phase4b_comparisons"][comparator] = {
                "mean_mse_difference": mean,
                "mcse": math.sqrt(_variance(diffs) / len(diffs))
                if len(diffs) > 1
                else None,
                "paired_ci95": ci,
                "point_loss": mean > 0,
                "statistically_supported_loss": ci[0] is not None and ci[0] > 0,
                "statistically_supported_improvement": ci[1] is not None and ci[1] < 0,
            }
            if name != "all" and mean > 0:
                vetoes.append(name + ":point_loss_to_" + comparator)
        margin = [b * b - 0.9 * a * a for b, a in zip(candidate, errors["atom"])]
        data["ten_percent_margin_ci95"] = interval(margin, [seed, 100 + group_index])
        data["ten_percent_margin_mean"] = _mean(margin)
        if name != "all" and len(group) < 30:
            vetoes.append(name + ":fewer_than_30_paths")
        report[name] = data
    primary = report["all"]["ten_percent_margin_ci95"][1]
    criterion = primary is not None and primary < 0
    if not power["nominal_power_adequate"]:
        vetoes.append("underpowered_at_500_path_cap")
    if len(rows) != power["executed_target_paths"]:
        vetoes.append("incomplete_validation")
    if not criterion:
        vetoes.append("ten_percent_primary_criterion_not_met")
    return {
        "groups": report,
        "promotion_vetoes": vetoes,
        "primary_criterion_pass": criterion,
        "heuristic_dominance_verdict": "PASS_SCREEN"
        if not any("point_loss" in v for v in vetoes)
        else "VETO",
        "decision": "NOMINATE_FOR_BROADER_VALIDATION" if not vetoes else "NO_PROMOTION",
        "ranking_scope": "only comparisons whose paired interval excludes zero",
        "default_readiness": False,
    }


def campaign_scope(engine, scope, args, budget, out):
    n, horizon = engine.n, engine.horizon
    rows_path = out / f"n{n}t{horizon}_paths.jsonl"
    settings = [("phase4b", rho, mark) for rho in RHO_GRID for mark in MARKS] + [
        ("phase4a", rho, MARKS[0]) for rho in RHO_GRID
    ]
    calibration_scores = {setting: [] for setting in settings}
    atom_errors, exact_scores, timings = [], [], []
    progress = {"scope": [n, horizon], "phase": "calibration", "paths_complete": 0}
    with rows_path.open("x") as handle:

        def retain(split, replicate, row):
            handle.write(
                json.dumps(
                    {"split": split, "replicate": replicate, **row}, allow_nan=False
                )
                + "\n"
            )
            handle.flush()

        for replicate in range(40):
            budget.check()
            path = engine.generate(
                tf.constant(
                    seed_key(args.attempt, scope, "calibration", replicate), tf.int32
                )
            )
            oracle = engine.evaluate("oracle", path, budget=budget)
            atom = engine.evaluate("atom", path, budget=budget)
            atom_errors.append(atom["score"] - oracle["score"])
            exact_scores.append(abs(oracle["score"]))
            candidates = []
            for method, rho, mark in settings:
                row = engine.evaluate(method, path, rho, mark, budget)
                error = row["score"] - oracle["score"]
                calibration_scores[(method, rho, mark)].append(error)
                candidates.append({"method": method, "rho": rho, "mark": mark, **row})
            retain(
                "calibration",
                replicate,
                {
                    "seed": seed_key(args.attempt, scope, "calibration", replicate),
                    "oracle": oracle,
                    "atom": atom,
                    "candidates": candidates,
                },
            )
            timings.append(
                atom["seconds"]
                + oracle["seconds"]
                + sum(r["seconds"] for r in candidates)
            )
            progress["paths_complete"] = replicate + 1
            progress["remaining_seconds"] = budget.remaining
            _write(out / "progress.json", progress)
            if replicate == 1:
                # Compile cost has already been paid. Reserve minimum 100-path
                # validation plus pilot using the measured warm four-arm cost.
                warm_candidate = max(r["seconds"] for r in candidates)
                minimum_cost = (40 - replicate - 1) * timings[-1] + 130 * (
                    atom["seconds"] + oracle["seconds"] + 3 * warm_candidate
                )
                budget.check(required=1.25 * minimum_cost)
            if (replicate + 1) % 5 == 0:
                print(json.dumps({"scope": [n, horizon], **progress}), flush=True)
        mse = {
            key: _mean([e * e for e in values])
            for key, values in calibration_scores.items()
        }
        chosen_b = min((key for key in settings if key[0] == "phase4b"), key=mse.get)
        chosen_a = min((key for key in settings if key[0] == "phase4a"), key=mse.get)
        sorted_exact = sorted(exact_scores)
        threshold = (sorted_exact[19] + sorted_exact[20]) / 2
        selection = {
            "phase4b": list(chosen_b),
            "phase4a": list(chosen_a),
            "absolute_exact_score_split": threshold,
            "canonical_calibration_mse": _mean([e * e for e in atom_errors]),
            "calibration": [
                {"method": key[0], "rho": key[1], "mark": key[2], "mse": value}
                for key, value in mse.items()
            ],
        }
        _write(out / f"n{n}t{horizon}_selection.json", selection)

        def selected_row(split, replicate):
            budget.check()
            key = seed_key(args.attempt, scope, split, replicate)
            path = engine.generate(tf.constant(key, tf.int32))
            row = {
                method: engine.evaluate(method, path, budget=budget)
                for method in ("oracle", "atom", "bootstrap")
            }
            row["phase4a"] = engine.evaluate(
                "phase4a", path, chosen_a[1], chosen_a[2], budget
            )
            row["phase4b"] = engine.evaluate(
                "phase4b", path, chosen_b[1], chosen_b[2], budget
            )
            retain(split, replicate, {"seed": key, **row})
            return row

        pilot = [selected_row("pilot", i) for i in range(30)]
        power = power_requirement(
            [r["phase4b"]["score"] - r["oracle"]["score"] for r in pilot],
            [r["atom"]["score"] - r["oracle"]["score"] for r in pilot],
            selection["canonical_calibration_mse"],
        )
        selection["power"] = power
        _write(out / f"n{n}t{horizon}_selection.json", selection)
        seconds_per_path = _mean(
            [
                sum(
                    r[m]["seconds"]
                    for m in ("oracle", "atom", "bootstrap", "phase4a", "phase4b")
                )
                for r in pilot[1:]
            ]
        )
        budget.check(required=1.25 * power["executed_target_paths"] * seconds_per_path)
        validation = []
        for i in range(power["executed_target_paths"]):
            validation.append(selected_row("validation", i))
            if (i + 1) % 20 == 0:
                progress = {
                    "scope": [n, horizon],
                    "phase": "validation",
                    "paths_complete": i + 1,
                    "paths_planned": power["executed_target_paths"],
                    "remaining_seconds": budget.remaining,
                }
                _write(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        result = {
            "scope": [n, horizon],
            "selection": selection,
            "validation": summarize(validation, threshold, power, 20260909 + scope),
            "absent_comparators": [
                "unproved zero-mean auxiliary control variate",
                "separate innovation-jitter arm",
            ],
            "path_rows": str(rows_path.relative_to(ROOT)),
            "warm_seconds_per_validation_path": seconds_per_path,
        }
        _write(out / f"n{n}t{horizon}_result.json", result)
        return result


def _manifest(args, memory_policy):
    return {
        "schema": "bayesfilter.kdm_phase4b_matrix_campaign.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "plan": PLAN,
        "model": MODEL,
        "numerical_controls": CONTROLS,
        "rho_grid": RHO_GRID,
        "mark_policies": MARKS,
        "dtype": "float64",
        "tf32": False,
        "jit_compile": True,
        "mode": args.mode,
        "tensorflow": tf.__version__,
        "python": sys.version,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unset"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu_memory_policy": memory_policy,
        "budget_seconds": args.budget_seconds,
        "seed_scheme": "[attempt*1000000+scope*100000+split*10000+replicate,20260909]; role folded in TensorFlow",
        "splits": SPLITS,
        "attempt": args.attempt,
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS
        },
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "nonclaims": [
            "Not ATOM-FINITE score parity",
            "No DSGE or HMC claim",
            "No canonical/default promotion",
            "Numerical controls are unpromoted diagnostic hypotheses",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "campaign"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--budget-seconds", type=float, default=2700.0)
    args = parser.parse_args()
    seed_key(args.attempt, 0, "preflight", 0)
    if not 0 < args.budget_seconds <= 2700:
        parser.error("budget must be positive and no larger than 2700 seconds")
    out = args.output_dir.resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(args.budget_seconds)
    terminal = {"status": "RUNNING", "scopes": [], "plan": PLAN}
    exit_code = 0
    try:
        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )

        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        manifest = _manifest(args, memory_policy)
        _write(out / "manifest.json", manifest)
        with tf.device("/GPU:0"):
            if args.mode == "preflight":
                terminal["preflight"] = preflight(Engine(8, 2), args.attempt)
                terminal["status"] = "PREFLIGHT_PASS"
            else:
                for scope, (n, horizon) in enumerate(SCOPES):
                    budget.check()
                    engine = Engine(n, horizon)
                    terminal["scopes"].append(
                        campaign_scope(engine, scope, args, budget, out)
                    )
                    _write(out / "result.json", terminal)
                    del engine
                terminal["status"] = "CAMPAIGN_COMPLETE"
    except BudgetExhausted as exc:
        terminal.update(status="UNDER_BUDGETED", error=str(exc))
        exit_code = 3
    except Exception as exc:
        terminal.update(
            status="IMPLEMENTATION_OR_NUMERICAL_VETO",
            error=f"{type(exc).__name__}: {exc}",
        )
        exit_code = 2
        import traceback

        terminal["traceback"] = traceback.format_exc()
    finally:
        terminal["wall_time_seconds"] = time.monotonic() - budget.started
        terminal["remaining_budget_seconds"] = max(0.0, budget.remaining)
        terminal["result_path"] = str((out / "result.json").relative_to(ROOT))
        if "manifest" in locals():
            terminal["gpu_allocator_bytes"] = tf.config.experimental.get_memory_info(
                "GPU:0"
            )
        _write(out / "result.json", terminal)
        print(
            json.dumps(
                {"status": terminal["status"], "result": str(out / "result.json")}
            ),
            flush=True,
        )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
