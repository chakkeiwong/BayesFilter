"""Deterministic multidimensional LGSSM HMC tuning driver.

This script is intentionally staged.  The initial ``fixture`` stage only
materializes a T=120 lower-triangular LGSSM data artifact from a versioned JSON
config.  Later stages may add XLA compile, geometry/mass, kernel tuning, and
final recovery gates, but those stages must remain deterministic and must not
delegate tuning decisions to an agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf
import tensorflow_probability as tfp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference import (  # noqa: E402
    HMCGeometryScaledBudgetTimingPolicy,
    HMCKernelTuningConfig,
    LowRankSPDQuadraticGeometryConfig,
    ValueScoreCapability,
    covariance_from_precision,
    fit_low_rank_spd_quadratic_geometry,
    tune_hmc_kernel,
)
from bayesfilter.runtime import stable_config_hash  # noqa: E402
from bayesfilter.testing import multidim_triangular_lgssm_tf as triangular  # noqa: E402


SCRIPT_NAME = "run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py"
CONFIG_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_tuning_config.v1"
FIXTURE_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_tuning_fixture.v1"
XLA_SCORE_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_tuning_xla_score_gate.v1"
GEOMETRY_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_tuning_geometry.v1"
MASS_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_tuning_mass.v1"
KERNEL_TUNING_SCHEMA = "bayesfilter.deterministic_lgssm_hmc_tuning_kernel.v1"
FINAL_RECOVERY_SCHEMA = (
    "bayesfilter.deterministic_lgssm_hmc_final_recovery_result.v1"
)
PRIVATE_TUNING_REPLAY_SCHEMA = (
    "bayesfilter.deterministic_lgssm_hmc_private_tuning_replay.v1"
)
PRIVATE_TUNING_REPLAY_REFERENCE_SCHEMA = (
    "bayesfilter.deterministic_lgssm_hmc_private_tuning_replay_reference.v1"
)
TARGET_SCOPE = "bayesfilter_multidim_lower_triangular_lgssm_t120_hmc_2026_07_09"
DEFAULT_CONFIG_PATH = (
    ROOT / "docs/benchmarks/configs/multidim_lgssm_serious_hmc_tuning_2026_07_09.json"
)
NONCLAIMS = (
    "fixture generation stage only",
    "not an HMC run",
    "not HMC convergence evidence",
    "not posterior recovery evidence",
    "not sampler superiority evidence",
    "not production readiness",
    "not default readiness",
    "not a DSGE claim",
)


@dataclass(frozen=True)
class DeterministicLGSSMHMCConfig:
    """Normalized config wrapper for the deterministic tuning driver."""

    payload: Mapping[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> "DeterministicLGSSMHMCConfig":
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("schema") != CONFIG_SCHEMA:
            raise ValueError("unsupported deterministic LGSSM HMC config schema")
        return cls(payload=payload, path=config_path)

    @property
    def hash(self) -> str:
        return f"sha256:{stable_config_hash(self.payload)}"

    @property
    def horizon(self) -> int:
        return int(self.payload["truth_and_data"]["horizon"])

    @property
    def fixture_path(self) -> Path:
        return ROOT / self.payload["artifact_paths"]["fixture"]

    @property
    def xla_compile_path(self) -> Path:
        return ROOT / self.payload["artifact_paths"]["xla_compile"]

    @property
    def geometry_path(self) -> Path:
        return ROOT / self.payload["artifact_paths"]["geometry"]

    @property
    def mass_path(self) -> Path:
        return ROOT / self.payload["artifact_paths"]["mass"]

    @property
    def kernel_tuning_path(self) -> Path:
        return ROOT / self.payload["artifact_paths"]["kernel_tuning"]

    @property
    def private_tuning_replay_path(self) -> Path:
        return self.artifact_root / "private_diagnostics" / "kernel_tuning_replay.json"

    @property
    def artifact_root(self) -> Path:
        return ROOT / self.payload["artifact_paths"]["root"]

    @property
    def source_contract_path(self) -> Path:
        return ROOT / self.payload["source_contract"]["path"]

    @property
    def final_result_path(self) -> Path:
        return ROOT / self.payload["artifact_paths"]["final_result"]

    @property
    def no_overwrite(self) -> bool:
        return bool(
            self.payload.get("execution_policy", {}).get(
                "artifact_no_overwrite", False
            )
        )

    def validate_fixture_stage(self) -> None:
        if self.horizon != 120:
            raise ValueError("serious LGSSM fixture horizon must be T=120")
        execution = self.payload["execution_policy"]
        if execution.get("required_environment", {}).get("CUDA_VISIBLE_DEVICES") != "-1":
            raise ValueError("fixture config must require CUDA_VISIBLE_DEVICES=-1")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise ValueError("fixture generation requires CUDA_VISIBLE_DEVICES=-1")
        if execution.get("jit_compile") is not True:
            raise ValueError("target-path jit_compile must be true in config")
        if execution.get("use_xla") is not True:
            raise ValueError("target-path use_xla must be true in config")
        if execution.get("jit_compile_false_runtime_allowed") is not False:
            raise ValueError("jit_compile=false runtime must be disallowed")
        if execution.get("gpu_sample_generation_allowed") is not False:
            raise ValueError("GPU sample generation must be disallowed")
        if execution.get("runtime_gradient_tape_allowed") is not False:
            raise ValueError("runtime autodiff tape must be disallowed")


class DeterministicLGSSMPosteriorAdapter:
    """Graph-native LGSSM posterior adapter for deterministic HMC tuning."""

    parameter_dim = 18

    def __init__(
        self,
        *,
        observations: Any,
        contract: Mapping[str, Any],
        parameter_names: Sequence[str],
        evidence_path: str,
    ) -> None:
        names = tuple(str(item) for item in parameter_names)
        if len(names) != self.parameter_dim:
            raise ValueError("parameter_names length mismatch")
        self._observations = tf.convert_to_tensor(observations, dtype=tf.float64)
        self._contract = dict(contract)
        self._parameter_names = names
        self._evidence_path = str(evidence_path)

    def adapter_signature(self) -> str:
        return stable_config_hash(
            {
                "adapter": "deterministic_multidim_lower_triangular_lgssm",
                "target_scope": TARGET_SCOPE,
                "parameter_names": self._parameter_names,
                "observation_shape": tuple(self._observations.shape.as_list()),
                "contract_id": self._contract.get("contract_id"),
            }
        )

    def parameter_names(self) -> tuple[str, ...]:
        return self._parameter_names

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_manual_lgssm_svd_graph_status_score",
            evidence_path=self._evidence_path,
            target_scope=TARGET_SCOPE,
            nonclaims=(
                "Phase 4 XLA compile authority only",
                "not posterior convergence evidence",
                "not posterior recovery evidence",
                "not production readiness",
            ),
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score = self._log_prob_and_grad_tensor(theta)
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self._log_prob_and_grad_tensor(theta)

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        theta_tensor = self._validate_theta_tensor(theta)
        if theta_tensor.shape.rank == 1:
            return self._single_target_status_telemetry(theta_tensor)
        flat_theta = tf.reshape(theta_tensor, (-1, self.parameter_dim))
        (
            status_code,
            valid_pre_regularized_score,
            floor_count_value,
            min_innovation_eigenvalue,
            innovation_condition_estimate,
        ) = tf.map_fn(
            self._single_target_status_tuple,
            flat_theta,
            fn_output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.bool),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float64),
                tf.TensorSpec(shape=(), dtype=tf.float64),
            ),
        )
        leading_shape = tf.shape(theta_tensor)[:-1]
        return {
            "status_code": tf.reshape(status_code, leading_shape),
            "valid_pre_regularized_score": tf.reshape(
                valid_pre_regularized_score,
                leading_shape,
            ),
            "floor_count_value": tf.reshape(floor_count_value, leading_shape),
            "min_innovation_eigenvalue": tf.reshape(
                min_innovation_eigenvalue,
                leading_shape,
            ),
            "innovation_condition_estimate": tf.reshape(
                innovation_condition_estimate,
                leading_shape,
            ),
        }

    def _log_prob_and_grad_tensor(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        theta_tensor = self._validate_theta_tensor(theta)
        if theta_tensor.shape.rank == 1:
            return self._single_log_prob_and_grad(theta_tensor)
        flat_theta = tf.reshape(theta_tensor, (-1, self.parameter_dim))
        values, scores = tf.map_fn(
            self._single_log_prob_and_grad,
            flat_theta,
            fn_output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.float64),
                tf.TensorSpec(shape=(self.parameter_dim,), dtype=tf.float64),
            ),
        )
        leading_shape = tf.shape(theta_tensor)[:-1]
        score_shape = tf.concat(
            [leading_shape, tf.constant([self.parameter_dim], dtype=tf.int32)],
            axis=0,
        )
        return tf.reshape(values, leading_shape), tf.reshape(scores, score_shape)

    def _single_log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _likelihood, _likelihood_score = (
            triangular.lower_triangular_lgssm_log_prob_and_score(
                theta,
                self._observations,
                self._contract,
            )
        )
        return value, score

    def _single_target_status_telemetry(self, theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
        (
            _value,
            _score,
            _likelihood,
            _likelihood_score,
            status,
        ) = triangular.lower_triangular_lgssm_log_prob_score_status(
            theta,
            self._observations,
            self._contract,
        )
        return status

    def _single_target_status_tuple(
        self,
        theta: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        status = self._single_target_status_telemetry(theta)
        return (
            status["status_code"],
            status["valid_pre_regularized_score"],
            status["floor_count_value"],
            status["min_innovation_eigenvalue"],
            status["innovation_condition_estimate"],
        )

    def _validate_theta_tensor(self, theta: Any) -> tf.Tensor:
        tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
        if tensor.shape.rank is None:
            raise ValueError("theta must have static rank")
        if tensor.shape.rank < 1:
            raise ValueError("theta must have rank at least 1")
        trailing = tensor.shape[-1]
        if trailing is None:
            raise ValueError("theta must have static trailing dimension")
        if int(trailing) != self.parameter_dim:
            raise ValueError("theta trailing dimension must match parameter_dim")
        return tensor


def build_fixture(config: DeterministicLGSSMHMCConfig) -> Mapping[str, Any]:
    """Build the deterministic T=120 LGSSM fixture payload."""

    config.validate_fixture_stage()
    contract = triangular.load_lower_triangular_lgssm_contract(config.source_contract_path)
    raw_truth = triangular.raw_truth_from_contract(contract)
    materialized = triangular.materialize_lower_triangular_lgssm_from_raw(
        raw_truth,
        contract,
    )
    horizon = config.horizon
    seed = tuple(int(item) for item in config.payload["truth_and_data"]["simulation_seed"])
    if len(seed) != 2:
        raise ValueError("simulation_seed must contain two integers")
    states, observations = simulate_lgssm(
        materialized,
        horizon=horizon,
        seed=seed,
    )
    residual = triangular.lyapunov_residual_tf(
        materialized.transition_matrix,
        materialized.stationary_covariance,
        materialized.model.transition_covariance,
    )
    constrained = {
        "transition_matrix": materialized.transition_matrix.numpy(),
        "process_std": materialized.process_std.numpy(),
        "process_covariance": materialized.model.transition_covariance.numpy(),
        "observation_matrix": materialized.model.observation_matrix.numpy(),
        "observation_std": materialized.observation_std.numpy(),
        "observation_covariance": materialized.model.observation_covariance.numpy(),
        "stationary_initial_mean": materialized.model.initial_mean.numpy(),
        "stationary_initial_covariance": materialized.stationary_covariance.numpy(),
    }
    diagnostics = fixture_diagnostics(
        states=states,
        observations=observations,
        materialized=materialized,
        lyapunov_residual=residual,
    )
    payload = {
        "schema": FIXTURE_SCHEMA,
        "created_at_utc": "deterministic_fixture_stage_no_wallclock",
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "config_path": str(config.path.relative_to(ROOT) if config.path.is_absolute() else config.path),
        "config_hash": config.hash,
        "stage": "fixture",
        "source_contract": dict(config.payload["source_contract"]),
        "horizon": horizon,
        "seed": seed,
        "parameter_names": tuple(config.payload["model"]["parameter_names"]),
        "raw_truth": raw_truth.numpy(),
        "constrained_truth": constrained,
        "states": states,
        "observations": observations,
        "diagnostics": diagnostics,
        "execution_policy": config.payload["execution_policy"],
        "nonclaims": NONCLAIMS,
    }
    normalized = json_ready(payload)
    normalized["artifact_hash"] = f"sha256:{stable_config_hash(normalized)}"
    return normalized


def build_xla_score_gate(config: DeterministicLGSSMHMCConfig) -> Mapping[str, Any]:
    """Compile and evaluate the T=120 fixture value/score path with XLA."""

    config.validate_fixture_stage()
    fixture = load_json(config.fixture_path)
    if fixture.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("fixture schema mismatch")
    if fixture.get("config_hash") != config.hash:
        raise ValueError("fixture config hash mismatch")
    raw = tf.constant(fixture["raw_truth"], dtype=tf.float64)
    compiled = build_compiled_value_score(config, fixture)
    contract = triangular.load_lower_triangular_lgssm_contract(config.source_contract_path)
    adapter = DeterministicLGSSMPosteriorAdapter(
        observations=fixture["observations"],
        contract=contract,
        parameter_names=fixture["parameter_names"],
        evidence_path=str(config.xla_compile_path.relative_to(ROOT)),
    )

    start = time.perf_counter()
    value, score = compiled(raw)
    compile_and_execute_s = time.perf_counter() - start
    hlo_metadata = xla_hlo_metadata(compiled, raw)
    second_start = time.perf_counter()
    value_second, score_second = compiled(tf.identity(raw))
    warm_execute_s = time.perf_counter() - second_start
    concrete_count = len(compiled._list_all_concrete_functions_for_serialization())
    value_np = value.numpy()
    score_np = score.numpy()
    score_second_np = score_second.numpy()
    finite_value = bool(np.all(np.isfinite(value_np)))
    finite_score = bool(np.all(np.isfinite(score_np)))
    target_status = adapter.target_status_telemetry(raw)
    target_status_payload = {
        key: np.asarray(value.numpy()).tolist()
        for key, value in target_status.items()
    }
    target_status_valid = bool(
        np.asarray(target_status["valid_pre_regularized_score"].numpy()).item()
        and np.asarray(target_status["status_code"].numpy()).item() == 0
        and np.asarray(target_status["floor_count_value"].numpy()).item() == 0
    )
    stable_second_call = bool(
        np.allclose(value_np, value_second.numpy(), rtol=0.0, atol=0.0)
        and np.allclose(score_np, score_second_np, rtol=0.0, atol=0.0)
    )
    passed = bool(
        finite_value
        and finite_score
        and target_status_valid
        and stable_second_call
        and concrete_count == 1
        and config.payload["execution_policy"]["jit_compile"] is True
    )
    payload = {
        "schema": XLA_SCORE_SCHEMA,
        "created_at_utc": "deterministic_xla_score_stage_no_wallclock",
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "config_path": str(config.path.relative_to(ROOT) if config.path.is_absolute() else config.path),
        "config_hash": config.hash,
        "fixture_path": str(config.fixture_path.relative_to(ROOT)),
        "fixture_hash": fixture["artifact_hash"],
        "stage": "xla_score",
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "runtime_autodiff_tape_executed": False,
        "value": float(value_np),
        "score": score_np,
        "score_shape": score_np.shape,
        "finite_value": finite_value,
        "finite_score": finite_score,
        "target_status_telemetry": target_status_payload,
        "target_status_valid": target_status_valid,
        "concrete_function_count": concrete_count,
        "hlo_metadata": hlo_metadata,
        "compile_and_first_execute_seconds": float(compile_and_execute_s),
        "warm_execute_seconds": float(warm_execute_s),
        "stable_second_call": stable_second_call,
        "passed": passed,
        "vetoes": [] if passed else xla_score_vetoes(
            finite_value=finite_value,
            finite_score=finite_score,
            target_status_valid=target_status_valid,
            stable_second_call=stable_second_call,
            concrete_count=concrete_count,
        ),
        "metric_roles": {
            "passed": "primary_phase4_pass_fail",
            "finite_value": "veto_diagnostic",
            "finite_score": "veto_diagnostic",
            "target_status_telemetry": "veto_diagnostic",
            "target_status_valid": "veto_diagnostic",
            "concrete_function_count": "retrace_veto_if_not_one",
            "hlo_metadata": "explanatory_only",
            "compile_and_first_execute_seconds": "explanatory_only",
            "warm_execute_seconds": "explanatory_only",
            "value": "explanatory_only",
            "score": "explanatory_only",
        },
        "nonclaims": (
            "XLA value/score compile gate only",
            "not an HMC run",
            "not HMC tuning evidence",
            "not HMC convergence evidence",
            "not posterior recovery evidence",
            "not sampler superiority evidence",
            "not production readiness",
            "not default readiness",
        ),
    }
    normalized = json_ready(payload)
    normalized["artifact_hash"] = f"sha256:{stable_config_hash(normalized)}"
    return normalized


def build_kernel_tuning(config: DeterministicLGSSMHMCConfig) -> Mapping[str, Any]:
    """Run deterministic serious HMC kernel tuning through BayesFilter APIs."""

    config.validate_fixture_stage()
    fixture = load_json(config.fixture_path)
    xla_gate = load_json(config.xla_compile_path)
    geometry = load_json(config.geometry_path)
    mass = load_json(config.mass_path)
    if fixture.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("fixture schema mismatch")
    if xla_gate.get("schema") != XLA_SCORE_SCHEMA or xla_gate.get("passed") is not True:
        raise ValueError("Phase 6 requires a passing XLA score gate")
    if geometry.get("schema") != GEOMETRY_SCHEMA or geometry.get("passed") is not True:
        raise ValueError("Phase 6 requires passing geometry artifact")
    if mass.get("schema") != MASS_SCHEMA or mass.get("passed") is not True:
        raise ValueError("Phase 6 requires passing mass artifact")
    contract = triangular.load_lower_triangular_lgssm_contract(config.source_contract_path)
    adapter = DeterministicLGSSMPosteriorAdapter(
        observations=fixture["observations"],
        contract=contract,
        parameter_names=fixture["parameter_names"],
        evidence_path=str(config.xla_compile_path.relative_to(ROOT)),
    )
    tuning_config = kernel_tuning_config_from_config(config)
    initial_position = np.asarray(mass["center"], dtype=float)
    initial_covariance = np.asarray(mass["mass_covariance"], dtype=float)
    start = time.perf_counter()
    result = tune_hmc_kernel(
        adapter=adapter,
        initial_position=initial_position,
        initial_covariance=initial_covariance,
        parameter_scales=np.asarray(mass["scale"], dtype=float),
        config=tuning_config,
        output_dir=config.artifact_root / "kernel_tuning_public",
    )
    elapsed_s = time.perf_counter() - start
    private_replay_reference: Mapping[str, Any] = {
        "schema": PRIVATE_TUNING_REPLAY_REFERENCE_SCHEMA,
        "available": False,
        "reason": "tuning_result_not_passed",
        "path_publicized": False,
        "hmc_mechanics_publicized": False,
        "mass_arrays_publicized": False,
    }
    if result.passed:
        private_replay = build_private_tuning_replay_payload(
            config=config,
            result=result,
            fixture=fixture,
            xla_gate=xla_gate,
            geometry=geometry,
            mass=mass,
        )
        write_json(
            private_replay,
            config.private_tuning_replay_path,
            overwrite=not config.no_overwrite,
        )
        private_replay_file_sha256 = file_sha256(config.private_tuning_replay_path)
        private_replay_reference = {
            "schema": PRIVATE_TUNING_REPLAY_REFERENCE_SCHEMA,
            "available": True,
            "artifact_hash": private_replay["artifact_hash"],
            "file_sha256": private_replay_file_sha256,
            "byte_count": config.private_tuning_replay_path.stat().st_size,
            "private_loop_final_kernel_hash": private_replay[
                "private_loop_final_kernel_hash"
            ],
            "public_final_kernel_hash": private_replay["public_final_kernel_hash"],
            "path_publicized": False,
            "hmc_mechanics_publicized": False,
            "mass_arrays_publicized": False,
        }
    tuner_payload = result.payload(include_internal_diagnostics=False)
    final_kernel_payload = result.final_kernel_payload
    final_kernel_hash = result.final_kernel_hash
    xla_confirmed = bool(
        tuning_config.use_xla
        and tuner_payload["config"]["use_xla"] is True
        and xla_gate.get("jit_compile") is True
        and xla_gate.get("passed") is True
    )
    hard_vetoes = tuple(str(item) for item in result.hard_vetoes)
    passed = bool(
        result.passed
        and final_kernel_payload is not None
        and final_kernel_hash is not None
        and xla_confirmed
        and not hard_vetoes
    )
    vetoes = kernel_tuning_vetoes(
        result_passed=bool(result.passed),
        final_kernel_payload=final_kernel_payload,
        final_kernel_hash=final_kernel_hash,
        xla_confirmed=xla_confirmed,
        hard_vetoes=hard_vetoes,
    )
    payload = {
        "schema": KERNEL_TUNING_SCHEMA,
        "created_at_utc": "deterministic_kernel_tuning_stage_no_wallclock",
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "config_path": str(config.path.relative_to(ROOT) if config.path.is_absolute() else config.path),
        "config_hash": config.hash,
        "fixture_path": str(config.fixture_path.relative_to(ROOT)),
        "fixture_hash": fixture["artifact_hash"],
        "xla_compile_path": str(config.xla_compile_path.relative_to(ROOT)),
        "xla_compile_hash": xla_gate["artifact_hash"],
        "geometry_path": str(config.geometry_path.relative_to(ROOT)),
        "geometry_hash": geometry["artifact_hash"],
        "mass_path": str(config.mass_path.relative_to(ROOT)),
        "mass_hash": mass["artifact_hash"],
        "stage": "kernel_tuning",
        "target_scope": TARGET_SCOPE,
        "adapter_signature": adapter.adapter_signature(),
        "parameter_names": fixture["parameter_names"],
        "jit_compile": True,
        "use_xla": True,
        "xla_confirmed": xla_confirmed,
        "jit_compile_false_runtime_executed": False,
        "runtime_autodiff_tape_executed": False,
        "elapsed_seconds": float(elapsed_s),
        "tuning_config": tuning_config.payload(),
        "tuner_public_payload": tuner_payload,
        "final_status": result.final_status,
        "diagnostic_role": result.diagnostic_role,
        "hard_vetoes": hard_vetoes,
        "repair_triggers": result.repair_triggers,
        "final_kernel_payload": final_kernel_payload,
        "final_kernel_hash": final_kernel_hash,
        "private_replay_reference": private_replay_reference,
        "final_kernel_requires_serious_sampling_pass": True,
        "passed": passed,
        "vetoes": vetoes,
        "metric_roles": {
            "passed": "primary_phase6_pass_fail",
            "xla_confirmed": "veto_diagnostic",
            "hard_vetoes": "veto_diagnostic",
            "final_kernel_payload": "veto_if_missing",
            "final_kernel_requires_serious_sampling_pass": "boundary_nonclaim",
            "final_status": "veto_diagnostic",
            "repair_triggers": "explanatory_only",
            "elapsed_seconds": "explanatory_only",
        },
        "nonclaims": (
            "kernel tuning artifact only",
            "not posterior convergence evidence",
            "not posterior recovery evidence",
            "not sampler superiority evidence",
            "not production readiness",
            "not default readiness",
        ),
    }
    normalized = json_ready(payload)
    normalized["artifact_hash"] = f"sha256:{stable_config_hash(normalized)}"
    return normalized


def build_private_tuning_replay_payload(
    *,
    config: DeterministicLGSSMHMCConfig,
    result: Any,
    fixture: Mapping[str, Any],
    xla_gate: Mapping[str, Any],
    geometry: Mapping[str, Any],
    mass: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the ignored replay payload consumed by retained Phase 7 HMC."""

    loop = result.tune_verify_repair_loop
    if result.passed is not True or result.final_status != "passed":
        raise ValueError("private replay requires a passed tuning result")
    if result.hard_vetoes:
        raise ValueError("private replay rejects hard-vetoed tuning results")
    if loop is None or loop.passed is not True or loop.final_kernel_payload is None:
        raise ValueError("private replay requires a passed private tuning loop")
    if loop.final_kernel_hash is None:
        raise ValueError("private replay requires a private loop kernel hash")

    tuning_payload = dict(result.payload(include_internal_diagnostics=True))
    tuning_payload["tune_verify_repair_loop"] = loop.payload(
        include_final_mass_arrays=True
    )
    private_final = tuning_payload["tune_verify_repair_loop"]["final_kernel_payload"]
    if not isinstance(private_final, Mapping):
        raise ValueError("private replay final kernel payload is missing")
    required_private_fields = {
        "adapted_mass_artifact_payload",
        "step_size",
        "num_leapfrog_steps",
    }
    missing = sorted(required_private_fields.difference(private_final))
    if missing:
        raise ValueError(f"private replay final kernel fields missing: {missing}")
    payload = {
        "schema": PRIVATE_TUNING_REPLAY_SCHEMA,
        "config_hash": config.hash,
        "fixture_hash": fixture["artifact_hash"],
        "xla_compile_hash": xla_gate["artifact_hash"],
        "geometry_hash": geometry["artifact_hash"],
        "mass_hash": mass["artifact_hash"],
        "target_scope": TARGET_SCOPE,
        "adapter_signature": result.adapter_signature,
        "public_final_kernel_hash": result.final_kernel_hash,
        "private_loop_final_kernel_hash": loop.final_kernel_hash,
        "selected_step_hash": private_final["selected_step_hash"],
        "selected_trajectory_hash": private_final["selected_trajectory_hash"],
        "tuning_payload": tuning_payload,
        "private_replay_payload": True,
        "public_reconstruction_api": False,
        "nonclaims": (
            "private exact frozen-kernel replay payload only",
            "not posterior convergence evidence",
            "not posterior recovery evidence",
            "not sampler superiority evidence",
            "not production or default readiness evidence",
        ),
    }
    normalized = json_ready(payload)
    normalized["artifact_hash"] = f"sha256:{stable_config_hash(normalized)}"
    return normalized


def build_final_recovery(
    config: DeterministicLGSSMHMCConfig,
    *,
    phase7_config_path: str | Path,
) -> Mapping[str, Any]:
    """Independently verify retained samples and evaluate fixture recovery."""

    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds,
        rank_normalized_hmc_diagnostics,
    )
    from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
        PHASE7_CONFIG_SCHEMA_V3,
        DeterministicLGSSMPhase7Config,
        validate_phase7_v3_inputs,
    )

    phase7 = DeterministicLGSSMPhase7Config.load(phase7_config_path)
    if phase7.payload.get("schema") != PHASE7_CONFIG_SCHEMA_V3:
        raise ValueError("final recovery requires the fresh Phase 7 V3 config")
    if phase7.payload["source_tuning_config_hash"] != config.hash:
        raise ValueError("final recovery tuning/Phase 7 config mismatch")
    if phase7.artifact_root.resolve() != config.artifact_root.resolve():
        raise ValueError("final recovery artifact-root mismatch")
    preflight = validate_phase7_v3_inputs(
        phase7,
        require_fresh_outputs=False,
    )
    terminal_path = phase7.artifact_path("public_result")
    private_path = phase7.artifact_path("private_retained_samples")
    terminal = load_json(terminal_path)
    if terminal.get("passed") is not True or terminal.get("smoke") is not False:
        raise ValueError("final recovery requires a passing serious terminal result")
    if terminal.get("config_hash") != phase7.hash:
        raise ValueError("serious terminal config hash mismatch")
    expected_replay_hash = phase7.payload["governed_source_references"][
        "private_replay"
    ]["artifact_hash"]
    with np.load(private_path, allow_pickle=False) as archive:
        retained = np.asarray(archive["retained_raw_samples"], dtype=np.float64)
        final_states = np.asarray(archive["final_worker_states"], dtype=np.float64)
        archive_config_hash = str(archive["config_hash"].item())
        archive_replay_hash = str(archive["private_replay_hash"].item())
    if retained.ndim != 3 or retained.shape[1:] != (4, 18):
        raise ValueError("retained archive must have shape [draw, 4, 18]")
    if retained.shape[0] != int(terminal["retained_results_per_chain"]):
        raise ValueError("retained archive draw count disagrees with terminal result")
    if final_states.shape != (2, 2, 18):
        raise ValueError("retained archive final worker-state shape mismatch")
    if not np.all(np.isfinite(retained)) or not np.all(np.isfinite(final_states)):
        raise ValueError("retained archive contains nonfinite values")
    if archive_config_hash != phase7.hash or archive_replay_hash != expected_replay_hash:
        raise ValueError("retained archive provenance mismatch")
    private_file_sha256 = file_sha256(private_path)
    private_reference = terminal.get("private_retained_sample_reference")
    if not isinstance(private_reference, Mapping) or (
        private_reference.get("file_sha256") != private_file_sha256
        or int(private_reference.get("byte_count", -1)) != private_path.stat().st_size
        or private_reference.get("shape_verified") is not True
        or private_reference.get("finite_verified") is not True
        or private_reference.get("provenance_verified") is not True
    ):
        raise ValueError("retained archive terminal reference mismatch")
    parameter_names = tuple(config.payload["model"]["parameter_names"])
    gate = config.payload["final_recovery_gate"]
    thresholds = RankNormalizedHMCThresholds(
        rhat_max=float(gate["r_hat_threshold"]),
        bulk_ess_min=float(gate["bulk_ess_min_per_parameter"]),
        tail_ess_min=float(gate["tail_ess_min_per_parameter"]),
    )
    diagnostics = rank_normalized_hmc_diagnostics(
        retained,
        parameter_names=parameter_names,
        thresholds=thresholds,
    )
    terminal_diagnostics = terminal.get("final_diagnostics")
    diagnostics_agree = _convergence_diagnostics_agree(
        diagnostics,
        terminal_diagnostics,
    )
    if not diagnostics_agree:
        raise ValueError("recomputed diagnostics disagree with serious terminal result")
    fixture = load_json(config.fixture_path)
    truth = np.asarray(fixture["raw_truth"], dtype=np.float64)
    if truth.shape != (18,) or not np.all(np.isfinite(truth)):
        raise ValueError("fixture raw truth is invalid")
    pooled = retained.reshape(-1, retained.shape[-1])
    posterior_mean = np.mean(pooled, axis=0)
    posterior_sd = np.std(pooled, axis=0, ddof=1)
    quantiles = np.quantile(pooled, [0.05, 0.50, 0.95], axis=0, method="linear")
    diagnostic_rows = {
        str(row["parameter"]): row for row in diagnostics["parameter_diagnostics"]
    }
    half = retained.shape[0] // 2
    split_raw = np.reshape(
        np.stack((retained[:half], retained[-half:]), axis=2),
        (half, 2 * retained.shape[1], retained.shape[2]),
    )
    mean_ess = np.asarray(
        tfp.mcmc.effective_sample_size(
            tf.convert_to_tensor(split_raw, dtype=tf.float64),
            filter_beyond_positive_pairs=True,
            cross_chain_dims=1,
        ).numpy(),
        dtype=np.float64,
    )
    if mean_ess.shape != (18,) or not np.all(np.isfinite(mean_ess)) or np.any(
        mean_ess <= 0.0
    ):
        raise ValueError("raw-mean ESS is nonfinite or invalid")
    prior_scales = geometry_scale_from_config(config)
    max_z = float(gate["truth_distance_max_abs_z"])
    rows = []
    for index, name in enumerate(parameter_names):
        sd = float(posterior_sd[index])
        mean_mcse = sd / np.sqrt(float(mean_ess[index]))
        recovery_z = (
            abs(float(posterior_mean[index]) - float(truth[index])) / sd
            if sd > 0.0
            else float("inf")
        )
        rows.append(
            {
                "parameter": name,
                "truth": float(truth[index]),
                "posterior_mean": float(posterior_mean[index]),
                "posterior_sd": sd,
                "mean_mcse": float(mean_mcse),
                "mean_ess": float(mean_ess[index]),
                "mean_mcse_definition": (
                    "posterior_sd / sqrt(split-chain cross-chain ESS of raw draws)"
                ),
                "q05": float(quantiles[0, index]),
                "q50": float(quantiles[1, index]),
                "q95": float(quantiles[2, index]),
                "abs_mean_minus_truth_over_sd": float(recovery_z),
                "prior_sd": float(prior_scales[index]),
                "posterior_to_prior_sd_ratio": float(sd / prior_scales[index]),
                "convergence_passed": bool(diagnostic_rows[name]["passed"]),
                "recovery_passed": bool(
                    np.isfinite(sd)
                    and sd > 0.0
                    and np.isfinite(mean_mcse)
                    and np.isfinite(recovery_z)
                    and recovery_z <= max_z
                ),
            }
        )
    recovery_passed = all(row["recovery_passed"] for row in rows)
    passed = bool(
        diagnostics["passed"]
        and diagnostics_agree
        and recovery_passed
        and not diagnostics.get("hard_vetoes")
    )
    payload = {
        "schema": FINAL_RECOVERY_SCHEMA,
        "stage": "final_recovery",
        "passed": passed,
        "decision": (
            "PASS_SINGLE_FIXTURE_FULL_ESTIMATION_RECOVERY_SCREEN"
            if passed
            else "FAIL_SINGLE_FIXTURE_FULL_ESTIMATION_RECOVERY_SCREEN"
        ),
        "config_path": str(config.path.resolve().relative_to(ROOT)),
        "config_hash": config.hash,
        "phase7_config_path": str(Path(phase7_config_path).resolve().relative_to(ROOT)),
        "phase7_config_hash": phase7.hash,
        "phase7_terminal_path": str(terminal_path.relative_to(ROOT)),
        "phase7_terminal_artifact_hash": terminal.get("artifact_hash"),
        "preflight_artifact_hash": preflight["artifact_hash"],
        "private_retained_samples": {
            "file_sha256": private_file_sha256,
            "byte_count": private_path.stat().st_size,
            "shape": tuple(int(item) for item in retained.shape),
            "config_hash_verified": True,
            "private_replay_hash_verified": True,
            "terminal_reference_verified": True,
            "all_finite": True,
        },
        "diagnostics": diagnostics,
        "terminal_diagnostics_agree": diagnostics_agree,
        "recovery_threshold_max_abs_z": max_z,
        "parameter_recovery": tuple(rows),
        "max_abs_mean_minus_truth_over_sd": max(
            row["abs_mean_minus_truth_over_sd"] for row in rows
        ),
        "all_parameter_recovery_passed": recovery_passed,
        "metric_roles": {
            "diagnostics": "promotion_and_veto_criterion",
            "parameter_recovery": "single_fixture_recovery_screen",
            "posterior_to_prior_sd_ratio": "explanatory_only",
            "mean_mcse": "explanatory_uncertainty_diagnostic",
        },
        "nonclaims": (
            "single deterministic synthetic-fixture recovery screen only",
            "not calibrated coverage evidence",
            "not estimator generality or robustness evidence",
            "not sampler superiority evidence",
            "not GPU, production, or default readiness evidence",
        ),
    }
    normalized = json_ready(payload)
    normalized["artifact_hash"] = f"sha256:{stable_config_hash(normalized)}"
    return normalized


def _convergence_diagnostics_agree(
    recomputed: Mapping[str, Any],
    terminal: Any,
    *,
    atol: float = 1e-12,
) -> bool:
    if not isinstance(terminal, Mapping):
        return False
    exact_fields = (
        "schema",
        "passed",
        "input_all_finite",
        "diagnostics_all_finite",
        "draw_count_per_chain",
        "chain_count",
        "parameter_count",
        "split_draw_count_per_chain",
        "split_chain_count",
    )
    if any(recomputed.get(name) != terminal.get(name) for name in exact_fields):
        return False
    if recomputed.get("definitions") != terminal.get("definitions") or (
        tuple(recomputed.get("hard_vetoes", ()))
        != tuple(terminal.get("hard_vetoes", ()))
    ):
        return False
    for name in ("max_rhat", "min_bulk_ess", "min_tail_ess"):
        if not np.isclose(
            float(recomputed[name]),
            float(terminal.get(name, float("nan"))),
            rtol=0.0,
            atol=atol,
        ):
            return False
    left_rows = recomputed.get("parameter_diagnostics", ())
    right_rows = terminal.get("parameter_diagnostics", ())
    if len(left_rows) != len(right_rows):
        return False
    numeric_fields = (
        "rank_normalized_split_rhat",
        "folded_rank_normalized_split_rhat",
        "rhat",
        "bulk_ess",
        "tail_ess",
        "lower_tail_ess",
        "upper_tail_ess",
    )
    for left, right in zip(left_rows, right_rows, strict=True):
        if left.get("parameter") != right.get("parameter") or left.get(
            "passed"
        ) != right.get("passed"):
            return False
        if any(
            not np.isclose(
                float(left[name]),
                float(right.get(name, float("nan"))),
                rtol=0.0,
                atol=atol,
            )
            for name in numeric_fields
        ):
            return False
    return True


def kernel_tuning_config_from_config(
    config: DeterministicLGSSMHMCConfig,
) -> HMCKernelTuningConfig:
    settings = config.payload["kernel_tuning"]
    if settings.get("preset") != "serious":
        raise ValueError("Phase 6 requires HMCKernelTuningConfig.serious")
    if settings.get("use_xla") is not True:
        raise ValueError("Phase 6 requires use_xla=true")
    if settings.get("chain_execution_mode") != "tf_function":
        raise ValueError("Phase 6 requires chain_execution_mode=tf_function")
    staged_timeout_policy = HMCGeometryScaledBudgetTimingPolicy().staged_timeout_policy()
    return HMCKernelTuningConfig.serious(
        target_accept_prob=float(settings["target_accept_prob"]),
        acceptance_band=tuple(float(item) for item in settings["acceptance_band"]),
        repair_band=tuple(float(item) for item in settings["repair_band"]),
        max_leapfrog_steps=int(settings["max_leapfrog_steps"]),
        bootstrap_max_repairs=int(settings["bootstrap_max_repairs"]),
        max_attempts=int(settings["max_attempts"]),
        seed=tuple(int(item) for item in settings["seed"]),
        chain_execution_mode=str(settings["chain_execution_mode"]),
        use_xla=bool(settings["use_xla"]),
        target_scope=TARGET_SCOPE,
        target_status_trace_policy=str(settings["target_status_trace_policy"]),
        verification_chunk_max_results=(
            None
            if settings.get("verification_chunk_max_results") is None
            else int(settings["verification_chunk_max_results"])
        ),
        verification_min_retained_results_for_pass=(
            None
            if settings.get("verification_min_retained_results_for_pass") is None
            else int(settings["verification_min_retained_results_for_pass"])
        ),
        allow_geometry_fallback=bool(settings["allow_geometry_fallback"]),
        geometry_position_role="prior_mean_raw_coordinates_truth_fixture",
        negative_hessian_source="phase5_low_rank_quadratic_precision",
        public_timeout_budget_s=staged_timeout_policy.global_cap_s,
        staged_timeout_policy=staged_timeout_policy,
    )


def kernel_tuning_vetoes(
    *,
    result_passed: bool,
    final_kernel_payload: Mapping[str, Any] | None,
    final_kernel_hash: str | None,
    xla_confirmed: bool,
    hard_vetoes: Sequence[str],
) -> list[str]:
    vetoes: list[str] = []
    if not result_passed:
        vetoes.append("tuner_not_passed")
    if final_kernel_payload is None:
        vetoes.append("final_kernel_payload_missing")
    if final_kernel_hash is None:
        vetoes.append("final_kernel_hash_missing")
    if not xla_confirmed:
        vetoes.append("xla_not_confirmed")
    vetoes.extend(f"hard_veto:{item}" for item in hard_vetoes)
    return vetoes


def build_geometry_and_mass(config: DeterministicLGSSMHMCConfig) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Build deterministic quadratic geometry and mass initializer artifacts."""

    config.validate_fixture_stage()
    fixture = load_json(config.fixture_path)
    xla_gate = load_json(config.xla_compile_path)
    if fixture.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("fixture schema mismatch")
    if fixture.get("config_hash") != config.hash:
        raise ValueError("fixture config hash mismatch")
    if xla_gate.get("schema") != XLA_SCORE_SCHEMA:
        raise ValueError("XLA score artifact schema mismatch")
    if xla_gate.get("passed") is not True:
        raise ValueError("Phase 5 requires a passing XLA score gate")
    if xla_gate.get("config_hash") != config.hash:
        raise ValueError("XLA score config hash mismatch")
    raw = np.asarray(fixture["raw_truth"], dtype=float)
    scale = geometry_scale_from_config(config)
    value_score_fn = build_compiled_value_score(config, fixture)
    geometry_config = geometry_config_from_config(config)
    geometry_result = fit_low_rank_spd_quadratic_geometry(
        value_score_fn,
        raw,
        scale=scale,
        config=geometry_config,
    )
    geometry_payload = {
        "schema": GEOMETRY_SCHEMA,
        "created_at_utc": "deterministic_geometry_stage_no_wallclock",
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "config_path": str(config.path.relative_to(ROOT) if config.path.is_absolute() else config.path),
        "config_hash": config.hash,
        "fixture_path": str(config.fixture_path.relative_to(ROOT)),
        "fixture_hash": fixture["artifact_hash"],
        "xla_compile_path": str(config.xla_compile_path.relative_to(ROOT)),
        "xla_compile_hash": xla_gate["artifact_hash"],
        "stage": "geometry",
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "runtime_autodiff_tape_executed": False,
        "geometry_config": geometry_config.payload(),
        "center_role": "prior_mean_raw_coordinates_truth_fixture",
        "center": raw,
        "scale": scale,
        "parameter_names": fixture["parameter_names"],
        "low_rank_geometry": geometry_result.payload(include_arrays=True),
        "passed": bool(geometry_result.accepted and geometry_result.precision is not None),
        "vetoes": [] if geometry_result.accepted and geometry_result.precision is not None else [
            f"geometry_not_accepted:{geometry_result.status}"
        ],
        "metric_roles": {
            "passed": "primary_phase5_geometry_pass_fail",
            "accepted": "veto_diagnostic",
            "precision_eigen_summary": "veto_diagnostic",
            "covariance_eigen_summary": "veto_diagnostic",
            "holdout_rmse": "veto_diagnostic",
            "center_refinement": "explanatory_only",
        },
        "nonclaims": (
            "geometry initializer only",
            "not an HMC run",
            "not HMC tuning evidence",
            "not HMC convergence evidence",
            "not posterior recovery evidence",
            "not certified MAP covariance",
            "not sampler superiority evidence",
            "not production readiness",
            "not default readiness",
        ),
    }
    geometry_payload = json_ready(geometry_payload)
    geometry_payload["artifact_hash"] = f"sha256:{stable_config_hash(geometry_payload)}"
    if geometry_payload["passed"] is not True:
        return geometry_payload, build_mass_blocker_from_geometry_payload(
            config,
            geometry_payload,
        )
    mass_payload = build_mass_from_geometry_payload(config, geometry_payload)
    return geometry_payload, mass_payload


def build_mass_from_geometry_payload(
    config: DeterministicLGSSMHMCConfig,
    geometry_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Convert accepted geometry precision into a deterministic mass artifact."""

    if geometry_payload.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError("geometry schema mismatch")
    if geometry_payload.get("passed") is not True:
        raise ValueError("cannot build mass from rejected geometry")
    low_rank = geometry_payload["low_rank_geometry"]
    precision = np.asarray(low_rank["precision"], dtype=float)
    mass_cfg = config.payload["mass_conversion"]
    mass = covariance_from_precision(
        precision,
        source=str(mass_cfg["source"]),
        jitter=float(mass_cfg["jitter"]),
        eigenvalue_floor=float(mass_cfg["eigenvalue_floor"]),
        max_condition_number=float(mass_cfg["max_condition_number"]),
        dense=bool(mass_cfg["dense"]),
    )
    regularized_precision = np.asarray(mass.regularized_precision, dtype=float)
    covariance = np.asarray(mass.covariance, dtype=float)
    factor = np.linalg.cholesky(covariance)
    reconstruction_error = float(
        np.max(np.abs(regularized_precision @ covariance - np.eye(precision.shape[0])))
    )
    factor_reconstruction_error = float(
        np.max(np.abs(factor @ factor.T - covariance))
    )
    condition_ok = bool(
        mass.precision_eigen_summary is not None
        and float(mass.precision_eigen_summary["condition_number"])
        <= float(mass_cfg["max_condition_number"]) * (1.0 + 1.0e-8)
    )
    reconstruction_ok = bool(
        reconstruction_error <= float(mass_cfg["reconstruction_max_abs_error_tolerance"])
    )
    passed = bool(
        mass.precision_eigen_summary is not None
        and mass.precision_eigen_summary["positive"]
        and mass.covariance_eigen_summary is not None
        and mass.covariance_eigen_summary["positive"]
        and condition_ok
        and reconstruction_ok
        and np.all(np.isfinite(factor))
    )
    vetoes = mass_vetoes(
        precision_summary=mass.precision_eigen_summary,
        covariance_summary=mass.covariance_eigen_summary,
        condition_ok=condition_ok,
        reconstruction_ok=reconstruction_ok,
        factor_finite=bool(np.all(np.isfinite(factor))),
    )
    payload = {
        "schema": MASS_SCHEMA,
        "created_at_utc": "deterministic_mass_stage_no_wallclock",
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "config_path": str(config.path.relative_to(ROOT) if config.path.is_absolute() else config.path),
        "config_hash": config.hash,
        "geometry_path": str(config.geometry_path.relative_to(ROOT)),
        "geometry_hash": geometry_payload["artifact_hash"],
        "stage": "mass",
        "center_role": geometry_payload["center_role"],
        "center": geometry_payload["center"],
        "scale": geometry_payload["scale"],
        "parameter_names": geometry_payload["parameter_names"],
        "mass_config": mass_cfg,
        "regularized_precision": regularized_precision,
        "mass_covariance": covariance,
        "factor": factor,
        "precision_eigen_summary": mass.precision_eigen_summary,
        "mass_covariance_eigen_summary": mass.covariance_eigen_summary,
        "regularization_report": mass.regularization_report,
        "precision_covariance_identity_max_abs_error": reconstruction_error,
        "factor_covariance_max_abs_error": factor_reconstruction_error,
        "passed": passed,
        "vetoes": vetoes,
        "metric_roles": {
            "passed": "primary_phase5_mass_pass_fail",
            "precision_eigen_summary": "veto_diagnostic",
            "mass_covariance_eigen_summary": "veto_diagnostic",
            "precision_covariance_identity_max_abs_error": "veto_diagnostic",
            "factor_covariance_max_abs_error": "veto_diagnostic",
            "regularization_report": "explanatory_only",
        },
        "nonclaims": (
            "mass initializer only",
            "not an HMC run",
            "not HMC tuning evidence",
            "not HMC convergence evidence",
            "not posterior recovery evidence",
            "not certified MAP covariance",
            "not sampler superiority evidence",
            "not production readiness",
            "not default readiness",
        ),
    }
    normalized = json_ready(payload)
    normalized["artifact_hash"] = f"sha256:{stable_config_hash(normalized)}"
    return normalized


def build_mass_blocker_from_geometry_payload(
    config: DeterministicLGSSMHMCConfig,
    geometry_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = {
        "schema": MASS_SCHEMA,
        "created_at_utc": "deterministic_mass_stage_no_wallclock",
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "config_path": str(config.path.relative_to(ROOT) if config.path.is_absolute() else config.path),
        "config_hash": config.hash,
        "geometry_path": str(config.geometry_path.relative_to(ROOT)),
        "geometry_hash": geometry_payload["artifact_hash"],
        "stage": "mass",
        "passed": False,
        "vetoes": tuple(geometry_payload.get("vetoes", ())) or ("geometry_not_passed",),
        "blocked_before_mass_conversion": True,
        "metric_roles": {
            "passed": "primary_phase5_mass_pass_fail",
            "vetoes": "veto_diagnostic",
        },
        "nonclaims": (
            "mass blocker artifact only",
            "not an HMC run",
            "not HMC tuning evidence",
            "not HMC convergence evidence",
            "not posterior recovery evidence",
        ),
    }
    normalized = json_ready(payload)
    normalized["artifact_hash"] = f"sha256:{stable_config_hash(normalized)}"
    return normalized


def build_compiled_value_score(
    config: DeterministicLGSSMHMCConfig,
    fixture: Mapping[str, Any],
) -> Any:
    """Return the fixture-bound XLA-compiled value/score callable."""

    contract = triangular.load_lower_triangular_lgssm_contract(config.source_contract_path)
    observations = tf.constant(fixture["observations"], dtype=tf.float64)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _likelihood, _likelihood_score = (
            triangular.lower_triangular_lgssm_log_prob_and_score(
                theta,
                observations,
                contract,
            )
        )
        return value, score

    return compiled


def geometry_config_from_config(
    config: DeterministicLGSSMHMCConfig,
) -> LowRankSPDQuadraticGeometryConfig:
    settings = config.payload["geometry_initializer"]
    return LowRankSPDQuadraticGeometryConfig(
        rank=int(settings["rank"]),
        sample_count=int(settings["sample_count"]),
        min_samples_per_parameter=int(settings["min_samples_per_parameter"]),
        trust_radius=float(settings["trust_radius"]),
        pilot_radius=float(settings["pilot_radius"]),
        pilot_direction_count=int(settings["pilot_direction_count"]),
        holdout_fraction=float(settings["holdout_fraction"]),
        eigenvalue_floor=float(settings["eigenvalue_floor"]),
        max_condition_number=float(settings["max_condition_number"]),
        fit_max_iterations=int(settings["fit_max_iterations"]),
        fit_tolerance=float(settings["fit_tolerance"]),
        holdout_rmse_abs_tolerance=float(settings["holdout_rmse_abs_tolerance"]),
        holdout_rmse_rel_tolerance=float(settings["holdout_rmse_rel_tolerance"]),
        seed=tuple(int(item) for item in settings["seed"]),
    )


def geometry_scale_from_config(config: DeterministicLGSSMHMCConfig) -> np.ndarray:
    scales = config.payload["prior"]["scale_by_block"]
    return np.asarray(
        [
            *([float(scales["diagonal_raw"])] * 4),
            *([float(scales["lower_raw"])] * 6),
            *([float(scales["log_process_std"])] * 4),
            *([float(scales["log_observation_std"])] * 4),
        ],
        dtype=float,
    )


def mass_vetoes(
    *,
    precision_summary: Mapping[str, Any] | None,
    covariance_summary: Mapping[str, Any] | None,
    condition_ok: bool,
    reconstruction_ok: bool,
    factor_finite: bool,
) -> list[str]:
    vetoes: list[str] = []
    if precision_summary is None or precision_summary.get("positive") is not True:
        vetoes.append("regularized_precision_not_spd")
    if covariance_summary is None or covariance_summary.get("positive") is not True:
        vetoes.append("mass_covariance_not_spd")
    if not condition_ok:
        vetoes.append("regularized_precision_condition_above_cap")
    if not reconstruction_ok:
        vetoes.append("precision_covariance_reconstruction_failed")
    if not factor_finite:
        vetoes.append("mass_factor_nonfinite")
    return vetoes


def xla_hlo_metadata(compiled: Any, raw: tf.Tensor) -> Mapping[str, Any]:
    """Return bounded XLA compiler-IR metadata when TensorFlow exposes it."""

    try:
        hlo_text = compiled.experimental_get_compiler_ir(raw)(stage="hlo")
    except (AttributeError, TypeError, ValueError, tf.errors.OpError) as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:240],
        }
    hlo_bytes = hlo_text.encode("utf-8")
    return {
        "available": True,
        "stage": "hlo",
        "byte_count": len(hlo_bytes),
        "sha256": stable_config_hash({"hlo_text": hlo_text}),
    }


def xla_score_vetoes(
    *,
    finite_value: bool,
    finite_score: bool,
    target_status_valid: bool,
    stable_second_call: bool,
    concrete_count: int,
) -> list[str]:
    vetoes: list[str] = []
    if not finite_value:
        vetoes.append("value_nonfinite")
    if not finite_score:
        vetoes.append("score_nonfinite")
    if not target_status_valid:
        vetoes.append("target_status_invalid")
    if not stable_second_call:
        vetoes.append("second_call_not_stable")
    if concrete_count != 1:
        vetoes.append("unexpected_retrace_count")
    return vetoes


def simulate_lgssm(
    materialized: triangular.LowerTriangularLGSSMMaterialization,
    *,
    horizon: int,
    seed: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if int(horizon) <= 1:
        raise ValueError("horizon must be greater than one")
    state_dim = int(materialized.transition_matrix.shape[0])
    observation_dim = int(materialized.model.observation_matrix.shape[0])
    base_seed = tf.constant(seed, dtype=tf.int32)
    initial_noise = tf.random.stateless_normal(
        (state_dim,),
        seed=base_seed,
        dtype=tf.float64,
    )
    process_noise = tf.random.stateless_normal(
        (horizon - 1, state_dim),
        seed=base_seed + tf.constant([1009, 17], dtype=tf.int32),
        dtype=tf.float64,
    )
    observation_noise = tf.random.stateless_normal(
        (horizon, observation_dim),
        seed=base_seed + tf.constant([2003, 29], dtype=tf.int32),
        dtype=tf.float64,
    )
    initial_factor = tf.linalg.cholesky(materialized.stationary_covariance)
    process_factor = tf.linalg.cholesky(materialized.model.transition_covariance)
    observation_factor = tf.linalg.cholesky(materialized.model.observation_covariance)
    states = []
    state = materialized.model.initial_mean + tf.linalg.matvec(
        initial_factor,
        initial_noise,
    )
    states.append(state)
    for index in range(horizon - 1):
        state = (
            materialized.model.transition_offset
            + tf.linalg.matvec(materialized.transition_matrix, state)
            + tf.linalg.matvec(process_factor, process_noise[index])
        )
        states.append(state)
    state_tensor = tf.stack(states, axis=0)
    observations = (
        state_tensor @ tf.transpose(materialized.model.observation_matrix)
        + materialized.model.observation_offset[tf.newaxis, :]
        + observation_noise @ tf.transpose(observation_factor)
    )
    return state_tensor.numpy(), observations.numpy()


def fixture_diagnostics(
    *,
    states: np.ndarray,
    observations: np.ndarray,
    materialized: triangular.LowerTriangularLGSSMMaterialization,
    lyapunov_residual: tf.Tensor,
) -> Mapping[str, Any]:
    transition = materialized.transition_matrix.numpy()
    stationary = materialized.stationary_covariance.numpy()
    eigvals = np.linalg.eigvals(transition)
    stationary_eigvals = np.linalg.eigvalsh(stationary)
    process_std = materialized.process_std.numpy()
    observation_std = materialized.observation_std.numpy()
    return {
        "state_shape": states.shape,
        "observation_shape": observations.shape,
        "transition_eigenvalues": eigvals,
        "transition_spectral_radius": float(np.max(np.abs(eigvals))),
        "stationarity_margin": float(1.0 - np.max(np.abs(eigvals))),
        "stationary_covariance_eigenvalues": stationary_eigvals,
        "stationary_covariance_min_eigenvalue": float(np.min(stationary_eigvals)),
        "lyapunov_max_abs_residual": float(
            tf.reduce_max(tf.abs(lyapunov_residual)).numpy()
        ),
        "state_mean": np.mean(states, axis=0),
        "state_std_empirical": np.std(states, axis=0),
        "observation_mean": np.mean(observations, axis=0),
        "observation_std_empirical": np.std(observations, axis=0),
        "max_abs_state": float(np.max(np.abs(states))),
        "max_abs_observation": float(np.max(np.abs(observations))),
        "process_to_observation_std_ratio_truth": process_std / observation_std,
    }


def write_json(
    payload: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = True,
) -> None:
    if not overwrite and path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [
                {"real": float(np.real(item)), "imag": float(np.imag(item))}
                for item in value.reshape(-1)
            ] if value.ndim == 1 else json_ready(value.tolist())
        return value.tolist()
    if isinstance(value, np.generic):
        if np.iscomplexobj(value):
            return {
                "real": float(np.real(value)),
                "imag": float(np.imag(value)),
            }
        return value.item()
    return value


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to deterministic LGSSM HMC tuning config JSON.",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "fixture",
            "xla_score",
            "geometry_mass",
            "kernel_tuning",
            "burnin_sampling",
            "final_recovery",
        ),
        default="fixture",
        help="Driver stage to execute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path override for fixture stage.",
    )
    parser.add_argument(
        "--phase7-config",
        type=Path,
        default=None,
        help="Explicit Phase 7 config for smoke, serious, and recovery stages.",
    )
    parser.add_argument(
        "--phase7-smoke",
        action="store_true",
        help="Run only the non-promoting tiny Phase 7 engineering smoke.",
    )
    parser.add_argument(
        "--phase7-output-dir",
        type=Path,
        default=None,
        help="Override Phase 7 public/private outputs, intended for /tmp smoke artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = DeterministicLGSSMHMCConfig.load(args.config)
    if args.stage == "fixture":
        _assert_fresh_outputs(config, (config.fixture_path,))
        payload = build_fixture(config)
        output = ROOT / args.output if args.output is not None and not args.output.is_absolute() else args.output
        if config.no_overwrite and int(
            config.payload["truth_and_data"]["fixture_hash_repetition_count"]
        ) != 2:
            raise ValueError("fresh fixture requires exactly two deterministic builds")
        if config.no_overwrite:
            repeated = build_fixture(config)
            if payload != repeated:
                raise ValueError("fresh fixture repeated generation mismatch")
        write_json(
            payload,
            config.fixture_path if output is None else output,
            overwrite=not config.no_overwrite,
        )
        return 0
    if args.stage == "xla_score":
        _assert_fresh_outputs(config, (config.xla_compile_path,))
        payload = build_xla_score_gate(config)
        output = ROOT / args.output if args.output is not None and not args.output.is_absolute() else args.output
        write_json(
            payload,
            config.xla_compile_path if output is None else output,
            overwrite=not config.no_overwrite,
        )
        return 0
    if args.stage == "geometry_mass":
        if args.output is not None:
            raise ValueError("--output is not supported for geometry_mass; two artifacts are written")
        _assert_fresh_outputs(config, (config.geometry_path, config.mass_path))
        geometry_payload, mass_payload = build_geometry_and_mass(config)
        write_json(
            geometry_payload,
            config.geometry_path,
            overwrite=not config.no_overwrite,
        )
        write_json(
            mass_payload,
            config.mass_path,
            overwrite=not config.no_overwrite,
        )
        return 0
    if args.stage == "kernel_tuning":
        _assert_fresh_outputs(
            config,
            (
                config.kernel_tuning_path,
                config.private_tuning_replay_path,
                config.artifact_root / "kernel_tuning_public",
            ),
        )
        payload = build_kernel_tuning(config)
        output = ROOT / args.output if args.output is not None and not args.output.is_absolute() else args.output
        write_json(
            payload,
            config.kernel_tuning_path if output is None else output,
            overwrite=not config.no_overwrite,
        )
        return 0
    if args.stage == "burnin_sampling":
        if args.output is not None:
            raise ValueError(
                "--output is not supported for burnin_sampling; use the Phase 7 config"
            )
        from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (
            DeterministicLGSSMPhase7Config,
            run_phase7,
        )

        if args.phase7_config is None:
            raise ValueError("burnin_sampling requires --phase7-config")
        phase7_config = DeterministicLGSSMPhase7Config.load(args.phase7_config)
        if phase7_config.payload["source_tuning_config_hash"] != config.hash:
            raise ValueError("tuning and Phase 7 config mismatch")

        output_dir = args.phase7_output_dir
        if args.phase7_smoke and output_dir is None:
            raise ValueError("--phase7-smoke requires --phase7-output-dir")
        if args.phase7_smoke and not output_dir.resolve().is_relative_to(Path("/tmp")):
            raise ValueError("Phase 7 smoke output must be under /tmp")
        if args.phase7_smoke and output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("Phase 7 smoke output directory must be empty")
        if not args.phase7_smoke and output_dir is not None:
            raise ValueError("serious Phase 7 must use its reviewed artifact paths")
        result = run_phase7(
            phase7_config,
            smoke=bool(args.phase7_smoke),
            output_override=(
                None if output_dir is None else output_dir / "burnin_sampling.json"
            ),
            progress_override=(
                None
                if output_dir is None
                else output_dir / "burnin_sampling_progress.json"
            ),
            private_samples_override=(
                None
                if output_dir is None
                else output_dir / "private_diagnostics" / "phase7_retained_samples.npz"
            ),
        )
        return 0 if result.get("passed") is True else 1
    if args.stage == "final_recovery":
        if args.output is not None:
            raise ValueError("--output is not supported for final_recovery")
        if args.phase7_config is None:
            raise ValueError("final_recovery requires --phase7-config")
        payload = build_final_recovery(
            config,
            phase7_config_path=args.phase7_config,
        )
        write_json(
            payload,
            config.final_result_path,
            overwrite=not config.no_overwrite,
        )
        return 0 if payload.get("passed") is True else 1
    raise ValueError(f"unsupported stage: {args.stage}")


def _assert_fresh_outputs(
    config: DeterministicLGSSMHMCConfig,
    paths: Sequence[Path],
) -> None:
    if not config.no_overwrite:
        return
    collisions = tuple(str(path) for path in paths if path.exists())
    if collisions:
        raise FileExistsError(f"fresh-run output collision: {collisions}")


if __name__ == "__main__":
    raise SystemExit(main())
