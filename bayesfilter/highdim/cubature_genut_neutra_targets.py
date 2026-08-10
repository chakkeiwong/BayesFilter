"""Frozen four-model posterior adapters for batch-native GenUT NeuTra training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_batch_adapters import (
    diagonal_lgssm_batch_adapter,
    ksc_mixture_sv_batch_adapter,
    parameterized_austria_sir_batch_adapter,
    predator_prey_batch_adapter,
)
from bayesfilter.highdim.cubature_genut_batch_tf import (
    BatchCandidateModelAdapter,
    batch_finite_value,
    batch_finite_value_score,
)
from bayesfilter.highdim.cubature_genut_candidate import cubature_design
from bayesfilter.inference.posterior_adapter import ValueScoreCapability


_LOG_TWO_PI = math.log(2.0 * math.pi)
_LGSSM_MATRIX = (
    (1.0, 0.25, -0.15),
    (0.2, 1.1, 0.3),
    (-0.1, 0.35, 0.9),
)
_LGSSM_LOWER = (-0.95, -0.95, -0.95, 0.05, 0.05)
_LGSSM_UPPER = (0.95, 0.95, 0.95, 2.0, 2.0)
_PP_LOWER = (0.1, 110.0, 20.0, 0.1, 0.0, 0.0)
_PP_UPPER = (1.1, 130.0, 30.0, 1.1, 1.0, 1.0)
_ADMISSION_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "docs/benchmarks/artifacts/genut_four_model_neutra_readiness_20260804/aggregate_attempt04/result.json"
)


@dataclass(frozen=True)
class GenUTControls:
    epsilon: float
    sinkhorn_steps: int
    balance_steps: int
    ridge: float
    higher_moment_correction_steps: int = 4
    higher_moment_strength: float = 0.2
    higher_moment_floor: float = 1.0e-5
    tuning_scope: str = "unreviewed"
    tuning_artifact: str = "unreviewed"

    def payload(self) -> Mapping[str, Any]:
        return {
            "epsilon": float(self.epsilon),
            "sinkhorn_steps": int(self.sinkhorn_steps),
            "balance_steps": int(self.balance_steps),
            "ridge": float(self.ridge),
            "higher_moment_correction_steps": int(
                self.higher_moment_correction_steps
            ),
            "higher_moment_strength": float(self.higher_moment_strength),
            "higher_moment_floor": float(self.higher_moment_floor),
            "tuning_scope": self.tuning_scope,
            "tuning_artifact": self.tuning_artifact,
        }


_WARM_START_CONTROLS = GenUTControls(
    epsilon=2.0,
    sinkhorn_steps=8,
    balance_steps=8,
    ridge=1.0e-5,
    tuning_scope="cross_scope_warm_start_hypothesis_only",
    tuning_artifact=(
        "docs/benchmarks/artifacts/"
        "moment_retuned_genut_whole_leaderboard_20260723/attempt05_final/result.json"
    ),
)
_LGSSM_CONTROLS = GenUTControls(
    epsilon=2.0,
    sinkhorn_steps=8,
    balance_steps=8,
    ridge=1.0e-5,
    tuning_scope="lgssm_T50_N1008_fp32_tf32_initial_observation_first",
    tuning_artifact=(
        "docs/benchmarks/artifacts/"
        "moment_retuned_genut_whole_leaderboard_20260723/attempt05_final/result.json"
    ),
)
_AUSTRIA_CONTROLS = GenUTControls(
    epsilon=8.0,
    sinkhorn_steps=8,
    balance_steps=8,
    ridge=1.0e-5,
    tuning_scope="austria_sir_T20_N1008_current_source",
    tuning_artifact=(
        "docs/benchmarks/artifacts/genut_austria_antithetic_ensemble_20260803/"
        "tuning_attempt01/result.json"
    ),
)


class GenUTNeuTraTargetAdapter:
    """Frozen deterministic posterior formed from one finite GenUT program."""

    dtype = tf.float64

    def __init__(
        self,
        *,
        model_id: str,
        parameter_names: tuple[str, ...],
        filter_adapter: BatchCandidateModelAdapter,
        observations: tf.Tensor,
        initial_noise: tf.Tensor,
        process_noise: tf.Tensor,
        design: tf.Tensor,
        controls: GenUTControls,
        chart: str,
        transition_before_first_observation: bool,
        data_id: str,
        control_status: str,
    ) -> None:
        self.model_id = str(model_id)
        self.parameter_names = tuple(parameter_names)
        self.parameter_dim = len(self.parameter_names)
        self.filter_adapter = filter_adapter
        self.observations = tf.convert_to_tensor(observations, tf.float32)
        self.initial_noise = tf.convert_to_tensor(initial_noise, tf.float32)
        self.process_noise = tf.convert_to_tensor(process_noise, tf.float32)
        self.design = tf.convert_to_tensor(design, tf.float32)
        self.controls = controls
        self.chart = str(chart)
        self.transition_before_first_observation = bool(
            transition_before_first_observation
        )
        self.data_id = str(data_id)
        self.control_status = str(control_status)
        self.target_scope = f"GENUT-{self.model_id}-fixed-posterior-v1"
        payload = {
            "schema": "bayesfilter.genut_neutra_target.v1",
            "model_id": self.model_id,
            "parameter_names": self.parameter_names,
            "chart": self.chart,
            "data_id": self.data_id,
            "observations_sha256": _tensor_hash(self.observations),
            "initial_noise_sha256": _tensor_hash(self.initial_noise),
            "process_noise_sha256": _tensor_hash(self.process_noise),
            "design_sha256": _tensor_hash(self.design),
            "controls": dict(self.controls.payload()),
            "control_status": self.control_status,
            "transition_before_first_observation": (
                self.transition_before_first_observation
            ),
            "backend": "tensorflow_batch_native_genut_forward_jvp_fp32",
            "deterministic_ops_required": True,
        }
        self._target_signature = _semantic_hash(payload)
        self._adapter_signature = _semantic_hash(
            payload
            | {
                "adapter": (
                    "bayesfilter.highdim.cubature_genut_neutra_targets:"
                    "GenUTNeuTraTargetAdapter"
                )
            }
        )

    @property
    def target_signature(self) -> str:
        return self._target_signature

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=False,
            runtime_backend="tensorflow_batch_native_genut_forward_jvp_fp32",
            evidence_path=(
                "docs/plans/"
                "bayesfilter-genut-four-model-neutra-readiness-plan-2026-08-04.md"
            ),
            target_scope=self.target_scope,
            nonclaims=(
                "engineering readiness target only",
                "no HMC convergence or posterior correctness claim",
                f"control_status={self.control_status}",
            ),
        )

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        return _posterior_value_score_status(self, theta)

    def batch_value_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
        return _posterior_value_status(self, theta)


def _normal_cdf_density(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    probability = 0.5 * (
        1.0 + tf.math.erf(values / tf.sqrt(tf.constant(2.0, values.dtype)))
    )
    density = tf.exp(
        -0.5 * tf.square(values) - 0.5 * tf.cast(_LOG_TWO_PI, values.dtype)
    )
    return probability, density


def _filter_theta_and_jacobian(
    target: GenUTNeuTraTargetAdapter, theta: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Map posterior coordinates to filter coordinates and posterior terms."""

    if target.chart == "austria_identity_normal":
        filter_theta = theta
        derivative = tf.ones_like(theta)
        scale = tf.constant(0.5, theta.dtype)
        prior = tf.reduce_sum(
            -0.5 * tf.square(theta / scale)
            - tf.math.log(scale)
            - 0.5 * tf.cast(_LOG_TWO_PI, theta.dtype),
            axis=1,
        )
        posterior_score = -theta / tf.square(scale)
        return filter_theta, derivative, prior, posterior_score

    probability, density = _normal_cdf_density(theta)
    if target.chart == "lgssm_five_probit_box":
        lower = tf.constant(_LGSSM_LOWER, theta.dtype)
        width = tf.constant(
            tuple(upper - lower for lower, upper in zip(_LGSSM_LOWER, _LGSSM_UPPER)),
            theta.dtype,
        )
        physical = lower[None, :] + width[None, :] * probability
        derivative = width[None, :] * density
        prior = tf.fill(tf.shape(theta)[:1], -tf.reduce_sum(tf.math.log(width)))
        jacobian = tf.reduce_sum(
            tf.math.log(width)[None, :]
            - 0.5 * tf.square(theta)
            - 0.5 * tf.cast(_LOG_TWO_PI, theta.dtype),
            axis=1,
        )
        return physical, derivative, prior + jacobian, -theta

    if target.chart == "ksc_two_probit_box":
        physical = 0.1 + 0.8 * probability
        log_beta = tf.math.log(physical[:, 1])
        filter_theta = tf.stack([theta[:, 0], log_beta], axis=1)
        derivative = tf.stack(
            [tf.ones_like(theta[:, 0]), 0.8 * density[:, 1] / physical[:, 1]],
            axis=1,
        )
        prior = tf.fill(
            tf.shape(theta)[:1], -2.0 * tf.math.log(tf.constant(0.8, theta.dtype))
        )
        jacobian = tf.reduce_sum(
            tf.math.log(tf.constant(0.8, theta.dtype))
            - 0.5 * tf.square(theta)
            - 0.5 * tf.cast(_LOG_TWO_PI, theta.dtype),
            axis=1,
        )
        return filter_theta, derivative, prior + jacobian, -theta

    if target.chart == "predator_prey_six_probit_box":
        lower = tf.constant(_PP_LOWER, theta.dtype)
        width = tf.constant(
            tuple(upper - lower for lower, upper in zip(_PP_LOWER, _PP_UPPER)),
            theta.dtype,
        )
        physical = lower[None, :] + width[None, :] * probability
        derivative = width[None, :] * density
        prior = tf.fill(tf.shape(theta)[:1], -tf.reduce_sum(tf.math.log(width)))
        jacobian = tf.reduce_sum(
            tf.math.log(width)[None, :]
            - 0.5 * tf.square(theta)
            - 0.5 * tf.cast(_LOG_TWO_PI, theta.dtype),
            axis=1,
        )
        return physical, derivative, prior + jacobian, -theta
    raise ValueError(f"unknown GenUT posterior chart: {target.chart}")


def _core_kwargs(target: GenUTNeuTraTargetAdapter) -> Mapping[str, Any]:
    controls = target.controls
    return {
        "epsilon": controls.epsilon,
        "sinkhorn_steps": controls.sinkhorn_steps,
        "balance_steps": controls.balance_steps,
        "ridge": controls.ridge,
        "transition_before_first_observation": (
            target.transition_before_first_observation
        ),
        "higher_moment_correction_steps": (
            controls.higher_moment_correction_steps
        ),
        "higher_moment_strength": controls.higher_moment_strength,
        "higher_moment_floor": controls.higher_moment_floor,
    }


def _normalized_status(
    diagnostics: Mapping[str, tf.Tensor], posterior_finite: tf.Tensor
) -> Mapping[str, tf.Tensor]:
    valid = diagnostics["program_valid"] & posterior_finite
    leading = tf.shape(valid)
    return {
        "status_code": tf.where(valid, tf.zeros(leading, tf.int32), tf.ones(leading, tf.int32)),
        "valid_pre_regularized_score": valid,
        "floor_count_value": tf.zeros(leading, tf.int32),
        # The generic NeuTra schema predates GenUT. There is no innovation
        # covariance here; zero is an unavailable sentinel, never a GenUT gate.
        "min_innovation_eigenvalue": tf.zeros(leading, tf.float64),
        "min_innovation_eigenvalue_available": tf.zeros(leading, tf.bool),
        "innovation_condition_estimate": tf.ones(leading, tf.float64),
        "innovation_condition_estimate_available": tf.zeros(leading, tf.bool),
        "minimum_covariance_gap_eigenvalue": tf.cast(
            diagnostics["minimum_covariance_gap_eigenvalue"], tf.float64
        ),
        "max_mean_residual": tf.cast(diagnostics["max_mean_residual"], tf.float64),
        "max_row_residual": tf.cast(diagnostics["max_row_residual"], tf.float64),
        "max_col_residual": tf.cast(diagnostics["max_col_residual"], tf.float64),
        "maximum_skew_residual": tf.cast(
            diagnostics["maximum_skew_residual"], tf.float64
        ),
        "maximum_kurtosis_residual": tf.cast(
            diagnostics["maximum_kurtosis_residual"], tf.float64
        ),
    }


def _posterior_value_score_status(
    target: GenUTNeuTraTargetAdapter, theta: Any
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    values = tf.convert_to_tensor(theta, tf.float64)
    if values.shape.rank != 2:
        raise ValueError("GenUT NeuTra target requires theta shape [batch, parameter]")
    if values.shape[-1] is not None and int(values.shape[-1]) != target.parameter_dim:
        raise ValueError("GenUT NeuTra target has the wrong parameter dimension")
    finite_input = tf.reduce_all(tf.math.is_finite(values), axis=1)
    safe_values = tf.where(finite_input[:, None], values, tf.zeros_like(values))
    filter_theta, chain, prior_jacobian, prior_jacobian_score = (
        _filter_theta_and_jacobian(target, safe_values)
    )
    likelihood, filter_score, diagnostics = batch_finite_value_score(
        target.filter_adapter,
        tf.cast(filter_theta, tf.float32),
        target.observations,
        target.initial_noise,
        target.process_noise,
        target.design,
        **_core_kwargs(target),
    )
    value = tf.cast(likelihood, tf.float64) + prior_jacobian
    score = tf.cast(filter_score, tf.float64) * chain + prior_jacobian_score
    posterior_finite = (
        finite_input
        & tf.math.is_finite(value)
        & tf.reduce_all(tf.math.is_finite(score), axis=1)
    )
    status = _normalized_status(diagnostics, posterior_finite)
    nan = tf.constant(float("nan"), tf.float64)
    return (
        tf.where(status["valid_pre_regularized_score"], value, nan),
        tf.where(
            status["valid_pre_regularized_score"][:, None],
            score,
            tf.fill(tf.shape(score), nan),
        ),
        status,
    )


def _posterior_value_status(
    target: GenUTNeuTraTargetAdapter, theta: Any
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    values = tf.convert_to_tensor(theta, tf.float64)
    if values.shape.rank != 2:
        raise ValueError("GenUT endpoint requires theta shape [batch, parameter]")
    finite_input = tf.reduce_all(tf.math.is_finite(values), axis=1)
    safe_values = tf.where(finite_input[:, None], values, tf.zeros_like(values))
    filter_theta, _chain, prior_jacobian, _score = _filter_theta_and_jacobian(
        target, safe_values
    )
    likelihood, diagnostics = batch_finite_value(
        target.filter_adapter,
        tf.cast(filter_theta, tf.float32),
        target.observations,
        target.initial_noise,
        target.process_noise,
        target.design,
        **_core_kwargs(target),
    )
    value = tf.cast(likelihood, tf.float64) + prior_jacobian
    status = _normalized_status(
        diagnostics, finite_input & tf.math.is_finite(value)
    )
    return (
        tf.where(
            status["valid_pre_regularized_score"],
            value,
            tf.constant(float("nan"), tf.float64),
        ),
        status,
    )


def make_genut_neutra_target(
    model: str,
    *,
    particle_count: int = 1008,
    noise_seed: int = 140000,
    controls: GenUTControls | None = None,
) -> GenUTNeuTraTargetAdapter:
    """Build one candidate target without a posterior-row fallback.

    This generic factory does not establish training admission. Serious NeuTra
    training must use :func:`make_admitted_genut_neutra_target`.
    """

    name = str(model).lower().replace("-", "_")
    supplied_scope_controls = controls is not None
    if particle_count <= 0:
        raise ValueError("particle_count must be positive")
    if name == "lgssm":
        observations = _lgssm_observations()
        dimension, observation_dimension, horizon = 3, 3, 50
        adapter = diagonal_lgssm_batch_adapter(
            observation_matrix=tf.constant(_LGSSM_MATRIX, tf.float32)
        )
        parameter_names = (
            "phi1_source_probit",
            "phi2_source_probit",
            "phi3_source_probit",
            "q_scale_source_probit",
            "r_scale_source_probit",
        )
        chart = "lgssm_five_probit_box"
        transition_first = False
        data_id = "benchmark_lgssm_m3_T50_seed81100"
        selected = controls or _LGSSM_CONTROLS
        control_status = (
            "repository_tuning_artifact_bound"
            if supplied_scope_controls
            else "scope_tuned_current_scalar_source_batch_revalidation_required"
        )
    elif name in ("ksc", "ksc_sv"):
        from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
            generate_frozen_exact_sv_dataset_tf,
        )
        from bayesfilter.testing.ksc_ukf_neutra_target_tf import (
            transformed_ksc_observations,
        )

        _states, raw = generate_frozen_exact_sv_dataset_tf()
        observations = tf.cast(transformed_ksc_observations(raw), tf.float32)
        dimension, observation_dimension, horizon = 1, 1, 1000
        adapter = ksc_mixture_sv_batch_adapter()
        parameter_names = ("gamma_source_probit", "beta_source_probit")
        chart = "ksc_two_probit_box"
        transition_first = False
        data_id = "zhao_cui_sv_ksc_T1000_seed81101"
        selected = controls or _WARM_START_CONTROLS
        control_status = (
            "repository_tuning_artifact_bound"
            if supplied_scope_controls
            else "requires_T1000_tuning_before_claim"
        )
    elif name in ("austria", "austria_sir", "sir"):
        from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
            SIR_OBSERVATION_SHA256,
            generate_frozen_sir_dataset_tf,
        )

        _states, observations64, _all = generate_frozen_sir_dataset_tf()
        observations = tf.cast(observations64, tf.float32)
        dimension, observation_dimension, horizon = 18, 9, 20
        adapter = parameterized_austria_sir_batch_adapter()
        parameter_names = (
            "log_kappa_scale",
            "log_nu_scale",
            "log_observation_noise_scale",
        )
        chart = "austria_identity_normal"
        transition_first = True
        data_id = f"austria_sir_y1_y20_sha256_{SIR_OBSERVATION_SHA256}"
        selected = controls or _AUSTRIA_CONTROLS
        control_status = (
            "repository_tuning_artifact_bound"
            if supplied_scope_controls
            else "scope_tuned_current_source"
        )
    elif name in ("predator_prey", "pp"):
        from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
            PP_OBSERVATION_SHA256,
            generate_frozen_predator_prey_dataset_tf,
        )

        _states, observations64 = generate_frozen_predator_prey_dataset_tf()
        observations = tf.cast(observations64, tf.float32)
        dimension, observation_dimension, horizon = 2, 2, 20
        adapter = predator_prey_batch_adapter()
        parameter_names = tuple(f"{name}_source_probit" for name in "rKasuv")
        chart = "predator_prey_six_probit_box"
        transition_first = False
        data_id = f"predator_prey_T20_sha256_{PP_OBSERVATION_SHA256}"
        selected = controls or _WARM_START_CONTROLS
        control_status = (
            "repository_tuning_artifact_bound"
            if supplied_scope_controls
            else "event_order_mismatch_requires_exact_scope_tuning"
        )
    else:
        raise ValueError(f"unknown GenUT NeuTra model: {model}")
    initial_noise = tf.random.stateless_normal(
        [particle_count, dimension], [noise_seed, 101], dtype=tf.float32
    )
    process_noise = tf.random.stateless_normal(
        [horizon, particle_count, dimension],
        [noise_seed, 102],
        dtype=tf.float32,
    )
    return GenUTNeuTraTargetAdapter(
        model_id=name,
        parameter_names=parameter_names,
        filter_adapter=adapter,
        observations=tf.ensure_shape(
            observations, [horizon, observation_dimension]
        ),
        initial_noise=initial_noise,
        process_noise=process_noise,
        design=cubature_design(
            dim=dimension, num_particles=particle_count, dtype=tf.float32
        ),
        controls=selected,
        chart=chart,
        transition_before_first_observation=transition_first,
        data_id=data_id,
        control_status=control_status,
    )


def make_admitted_genut_neutra_target(model: str) -> GenUTNeuTraTargetAdapter:
    """Build a target admitted by the repository-owned readiness aggregate."""

    if not _ADMISSION_ARTIFACT.is_file():
        raise RuntimeError("GenUT NeuTra admission artifact is missing")
    aggregate = json.loads(_ADMISSION_ARTIFACT.read_text(encoding="utf-8"))
    if (
        aggregate.get("schema")
        != "bayesfilter.genut_four_model_neutra_readiness_aggregate.v1"
        or aggregate.get("passed") is not True
    ):
        raise RuntimeError("GenUT NeuTra admission artifact is invalid")
    name = str(model).lower().replace("-", "_")
    aliases = {"ksc": "ksc_sv", "pp": "predator_prey"}
    name = aliases.get(name, name)
    entry = aggregate.get("models", {}).get(name)
    if not isinstance(entry, dict) or not entry.get(
        "ready_for_target_specific_serious_neutra_training", False
    ):
        raise ValueError(f"GenUT target is not admitted for serious NeuTra training: {name}")
    expected_tf32 = bool(entry["tf32_enabled"])
    if tf.config.experimental.tensor_float_32_execution_enabled() is not expected_tf32:
        raise RuntimeError(
            f"GenUT admitted arithmetic requires tf32_enabled={expected_tf32}"
        )
    if os.environ.get("TF_DETERMINISTIC_OPS") != "1":
        raise RuntimeError("GenUT admitted target requires TF_DETERMINISTIC_OPS=1")
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        raise RuntimeError("GenUT admitted target requires a visible GPU")
    if not all(
        tf.config.experimental.get_memory_growth(device) for device in physical_gpus
    ):
        raise RuntimeError("GenUT admitted target requires verified GPU memory growth")
    controls_payload = entry["controls"]
    controls = GenUTControls(
        epsilon=float(controls_payload["epsilon"]),
        sinkhorn_steps=int(controls_payload["sinkhorn_steps"]),
        balance_steps=int(controls_payload["balance_steps"]),
        ridge=float(controls_payload["ridge"]),
        higher_moment_correction_steps=int(
            controls_payload["higher_moment_correction_steps"]
        ),
        higher_moment_strength=float(controls_payload["higher_moment_strength"]),
        higher_moment_floor=float(controls_payload["higher_moment_floor"]),
        tuning_scope=str(controls_payload["tuning_scope"]),
        tuning_artifact=str(controls_payload["tuning_artifact"]),
    )
    target = make_genut_neutra_target(name, particle_count=1008, controls=controls)
    if target.target_signature != entry["target_signature"]:
        raise RuntimeError("GenUT admitted target signature mismatch")
    if target.control_status != "repository_tuning_artifact_bound":
        raise RuntimeError("GenUT admitted target is not tuning-artifact bound")
    return target


def _lgssm_observations() -> tf.Tensor:
    generator = tf.random.Generator.from_seed(81100)
    phi = tf.constant([0.72, 0.55, 0.35], tf.float64)
    q_scale = tf.constant(0.35, tf.float64)
    r_scale = tf.constant(0.45, tf.float64)
    observation_matrix = tf.constant(_LGSSM_MATRIX, tf.float64)
    state = (
        q_scale
        / tf.sqrt(1.0 - tf.square(phi))
        * generator.normal([3], dtype=tf.float64)
    )
    observations = tf.TensorArray(tf.float64, size=50)
    observations = observations.write(
        0,
        tf.linalg.matvec(observation_matrix, state)
        + r_scale * generator.normal([3], dtype=tf.float64),
    )
    for index in range(1, 50):
        state = phi * state + q_scale * generator.normal([3], dtype=tf.float64)
        observations = observations.write(
            index,
            tf.linalg.matvec(observation_matrix, state)
            + r_scale * generator.normal([3], dtype=tf.float64),
        )
    return tf.cast(observations.stack(), tf.float32)


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "GenUTControls",
    "GenUTNeuTraTargetAdapter",
    "make_admitted_genut_neutra_target",
    "make_genut_neutra_target",
]
