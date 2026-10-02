"""Phase 20 CPU-hidden LGSSM NeuTra HMC reference validation.

This helper runs a bounded fixed-transport LGSSM NeuTra HMC validation against
a deterministic two-dimensional quadrature reference posterior.  It is a
Phase 20 engineering gate only.  It does not train NeuTra, use GPU sample
generation, run a non-JIT fallback, tune HMC, or establish broad HMC,
production, default, DSGE, nonlinear-SSM, or scientific validity claims.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesfilter.runtime.device_policy import assert_cpu_only_env, ensure_cpu_only_env
from bayesfilter.testing.neutra_cpu_multicore_hmc_chain_harness_tf import (
    DEFAULT_PHASE16_ARTIFACT_DIR,
    DEFAULT_PHASE17_PAYLOAD_PATH,
    DEFAULT_PHASE18_OUTPUT_PATH,
    DEFAULT_PHASE19_OUTPUT_PATH,
    EXPECTED_PHASE19_ADAPTER_SIGNATURE,
    EXPECTED_PHASE19_FIXED_TRANSPORT_ADAPTER_SIGNATURE,
    EXPECTED_PHASE19_TARGET_SIGNATURE,
)


PHASE20_ROUTE = "phase20_lgssm_reference_hmc_validation"
PHASE20_TARGET_SCOPE = "lgssm-neutra-phase20-fixed-transport-reference-hmc-validation"
PHASE20_EVIDENCE_PATH = (
    "bayesfilter/testing/neutra_lgssm_reference_hmc_validation_tf.py"
)
DEFAULT_SEED = 20260707
DEFAULT_PHASE20_OUTPUT_PATH = DEFAULT_PHASE16_ARTIFACT_DIR / (
    "lgssm_static_qr_exact_kalman_affine_neutra_"
    "phase20_lgssm_reference_hmc_validation_seed20260707.json"
)

NEUTRA_LGSSM_REFERENCE_HMC_VALIDATION_NONCLAIMS = (
    "Phase 20 LGSSM reference HMC validation for one static QR fixture only",
    "deterministic 2D quadrature reference over exact LGSSM likelihood target",
    "no analytic Gaussian posterior claim",
    "no HMC sampler superiority claim",
    "no HMC tuning optimality claim",
    "no production readiness claim",
    "no default execution readiness claim",
    "no nonlinear SSM validity claim",
    "no DSGE or c603 validity claim",
    "no broad NeuTra validity claim",
    "no scientific validity claim",
)


class NeuTraLGSSMReferenceHMCValidationError(RuntimeError):
    """Raised when the Phase 20 validation contract is violated."""


@dataclass(frozen=True)
class NeuTraLGSSMReferenceHMCValidationConfig:
    """Configuration for the Phase 20 LGSSM reference HMC gate."""

    payload_path: Path = DEFAULT_PHASE17_PAYLOAD_PATH
    phase18_diagnostic_path: Path = DEFAULT_PHASE18_OUTPUT_PATH
    phase19_harness_path: Path = DEFAULT_PHASE19_OUTPUT_PATH
    output_path: Path = DEFAULT_PHASE20_OUTPUT_PATH
    seed: int = DEFAULT_SEED
    worker_count: int = 2
    chain_count: int = 4
    num_results: int = 64
    num_burnin_steps: int = 64
    num_leapfrog_steps: int = 4
    step_size: float = 0.05
    jit_compile: bool = True
    require_cpu_hidden: bool = True
    quadrature_grid_size: int = 121
    quadrature_extent: float = 5.0
    posterior_mean_abs_tolerance: float = 0.35
    posterior_cov_abs_tolerance: float = 0.65
    min_acceptance_rate: float = 0.01
    max_acceptance_rate: float = 1.0
    target_signature: str = EXPECTED_PHASE19_TARGET_SIGNATURE
    adapter_signature: str = EXPECTED_PHASE19_ADAPTER_SIGNATURE
    fixed_transport_adapter_signature: str = (
        EXPECTED_PHASE19_FIXED_TRANSPORT_ADAPTER_SIGNATURE
    )

    def normalized(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.lgssm_reference_hmc_validation_config.v1",
            "phase": PHASE20_ROUTE,
            "payload_path": str(self.payload_path),
            "phase18_diagnostic_path": str(self.phase18_diagnostic_path),
            "phase19_harness_path": str(self.phase19_harness_path),
            "output_path": str(self.output_path),
            "seed": int(self.seed),
            "worker_count": int(self.worker_count),
            "chain_count": int(self.chain_count),
            "num_results": int(self.num_results),
            "num_burnin_steps": int(self.num_burnin_steps),
            "num_leapfrog_steps": int(self.num_leapfrog_steps),
            "step_size": float(self.step_size),
            "jit_compile": bool(self.jit_compile),
            "jit_compile_false_runtime_allowed": False,
            "require_cpu_hidden": bool(self.require_cpu_hidden),
            "quadrature_grid_size": int(self.quadrature_grid_size),
            "quadrature_extent": float(self.quadrature_extent),
            "posterior_mean_abs_tolerance": float(self.posterior_mean_abs_tolerance),
            "posterior_cov_abs_tolerance": float(self.posterior_cov_abs_tolerance),
            "min_acceptance_rate": float(self.min_acceptance_rate),
            "max_acceptance_rate": float(self.max_acceptance_rate),
            "execution_target": "cpu_hidden_multicore_full_chain_hmc",
            "training_execution_target": "not_run",
            "gpu_sample_generation_policy": "forbidden",
            "hmc_policy": "fixed_kernel_full_chain_tfp_hmc_no_adaptation",
            "reference_posterior_policy": (
                "deterministic_2d_quadrature_over_exact_lgssm_likelihood"
            ),
            "target_scope": PHASE20_TARGET_SCOPE,
            "full_chain_xla_diagnostic_authority_scope": "phase20_only",
            "expected_target_signature": str(self.target_signature),
            "expected_adapter_signature": str(self.adapter_signature),
            "expected_fixed_transport_adapter_signature": str(
                self.fixed_transport_adapter_signature
            ),
            "nonclaims": NEUTRA_LGSSM_REFERENCE_HMC_VALIDATION_NONCLAIMS,
        }


def run_lgssm_reference_hmc_validation(
    config: NeuTraLGSSMReferenceHMCValidationConfig | None = None,
) -> Mapping[str, Any]:
    """Run the Phase 20 CPU-hidden fixed-transport HMC validation."""

    cfg = NeuTraLGSSMReferenceHMCValidationConfig() if config is None else config
    start = time.monotonic()
    _validate_config(cfg)
    payload = _read_json_mapping(cfg.payload_path, "Phase 17 payload")
    phase18 = _read_json_mapping(cfg.phase18_diagnostic_path, "Phase 18 diagnostic")
    phase19 = _read_json_mapping(cfg.phase19_harness_path, "Phase 19 harness")
    payload_hash = _file_sha256(cfg.payload_path)
    phase18_hash = _file_sha256(cfg.phase18_diagnostic_path)
    phase19_hash = _file_sha256(cfg.phase19_harness_path)
    _validate_provenance(cfg, payload=payload, phase18=phase18, phase19=phase19)
    reference = _run_reference_posterior_worker(cfg)
    worker_inputs = [
        {
            "worker_index": index,
            "payload_path": str(cfg.payload_path),
            "seed": int(cfg.seed) + 2003 * (index + 1),
            "chain_count": _worker_chain_count(
                worker_index=index,
                worker_count=int(cfg.worker_count),
                chain_count=int(cfg.chain_count),
            ),
            "num_results": int(cfg.num_results),
            "num_burnin_steps": int(cfg.num_burnin_steps),
            "num_leapfrog_steps": int(cfg.num_leapfrog_steps),
            "step_size": float(cfg.step_size),
            "jit_compile": bool(cfg.jit_compile),
            "target_signature": str(cfg.target_signature),
            "adapter_signature": str(cfg.adapter_signature),
        }
        for index in range(int(cfg.worker_count))
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(cfg.worker_count),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        workers = list(executor.map(_worker_run_hmc_validation, worker_inputs))
    workers = sorted(workers, key=lambda row: int(row["worker_index"]))
    aggregate = _aggregate_worker_samples(workers)
    residuals = _posterior_residuals(
        sample_summary=aggregate["posterior_sample_summary"],
        reference_posterior=reference,
    )
    sample_diagnostics = _sample_diagnostics(
        workers=workers,
        aggregate=aggregate,
        residuals=residuals,
        config=cfg,
    )
    boundary_checks = {
        "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "worker_count_match": len(workers) == int(cfg.worker_count),
        "worker_return_codes_zero": all(
            int(row.get("return_code", 1)) == 0 for row in workers
        ),
        "worker_seeds_distinct": len(
            {int(row.get("seed", -1)) for row in workers}
        )
        == len(workers),
        "jit_compile_true": bool(cfg.jit_compile) is True,
        "jit_compile_false_runtime_not_executed": True,
        "training_not_executed": True,
        "gpu_sample_generation_not_executed": True,
        "posterior_validation_executed": True,
        "reference_posterior_present": bool(reference),
        "sample_diagnostics_present": bool(sample_diagnostics),
    }
    pass_checks = {
        **boundary_checks,
        "all_workers_passed": all(bool(row.get("passed")) for row in workers),
        "finite_samples": bool(sample_diagnostics.get("finite_samples")),
        "finite_target_log_prob": bool(
            sample_diagnostics.get("finite_target_log_prob")
        ),
        "acceptance_rate_in_bounds": bool(
            sample_diagnostics.get("acceptance_rate_in_bounds")
        ),
        "posterior_mean_residual_within_tolerance": bool(
            residuals["mean_max_abs_residual"]
            <= float(cfg.posterior_mean_abs_tolerance)
        ),
        "posterior_cov_residual_within_tolerance": bool(
            residuals["cov_max_abs_residual"]
            <= float(cfg.posterior_cov_abs_tolerance)
        ),
    }
    passed = bool(all(pass_checks.values()))
    artifact = {
        "schema": "bayesfilter.neutra.lgssm_reference_hmc_validation_result.v1",
        "phase": PHASE20_ROUTE,
        "passed": passed,
        "decision": (
            "PASS_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION"
            if passed
            else "BLOCK_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION"
        ),
        "config": cfg.normalized(),
        "payload_path": str(cfg.payload_path),
        "payload_file_sha256": payload_hash,
        "phase18_diagnostic_path": str(cfg.phase18_diagnostic_path),
        "phase18_diagnostic_file_sha256": phase18_hash,
        "phase19_harness_path": str(cfg.phase19_harness_path),
        "phase19_harness_file_sha256": phase19_hash,
        "target_signature": str(cfg.target_signature),
        "adapter_signature": str(cfg.adapter_signature),
        "fixed_transport_adapter_signature": str(
            cfg.fixed_transport_adapter_signature
        ),
        "phase20_base_adapter_signature": _single_worker_field(
            workers,
            "phase20_base_adapter_signature",
        ),
        "phase20_fixed_transport_adapter_signature": _single_worker_field(
            workers,
            "phase20_fixed_transport_adapter_signature",
        ),
        "transport_hash": _single_worker_field(workers, "transport_hash"),
        "artifact_signature": _single_worker_field(workers, "artifact_signature"),
        "reference_posterior": reference,
        "posterior_residuals": residuals,
        "sample_diagnostics": sample_diagnostics,
        "worker_count": int(cfg.worker_count),
        "chain_count": int(cfg.chain_count),
        "num_results": int(cfg.num_results),
        "num_burnin_steps": int(cfg.num_burnin_steps),
        "num_leapfrog_steps": int(cfg.num_leapfrog_steps),
        "step_size": float(cfg.step_size),
        "workers": workers,
        "boundary_checks": boundary_checks,
        "pass_checks": pass_checks,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "tensorflow_imported_in_parent": _tensorflow_imported(),
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "training_executed": False,
        "gpu_sample_generation_executed": False,
        "posterior_validation_executed": True,
        "full_chain_hmc_executed": any(
            bool(row.get("full_chain_hmc_executed")) for row in workers
        ),
        "hmc_tuning_executed": False,
        "uncertainty_limitations": (
            "bounded short chains only",
            "few workers and chains",
            "posterior residuals are Phase 20 fixture diagnostics only",
            "R-hat and ESS are unavailable unless explicitly recorded by worker",
        ),
        "nonclaims": NEUTRA_LGSSM_REFERENCE_HMC_VALIDATION_NONCLAIMS,
        "elapsed_seconds": time.monotonic() - start,
    }
    artifact = {
        **artifact,
        "artifact_hash": f"sha256:{_stable_payload_sha256(artifact)}",
        "artifact_hash_semantics": "stable_json_sha256_excluding_artifact_hash_fields",
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.output_path, artifact)
    return artifact


def phase20_error_payload(
    error: BaseException,
    *,
    config: NeuTraLGSSMReferenceHMCValidationConfig,
) -> Mapping[str, Any]:
    """Build a Phase 20 blocker payload without any non-JIT fallback."""

    payload = {
        "schema": "bayesfilter.neutra.lgssm_reference_hmc_validation_result.v1",
        "phase": PHASE20_ROUTE,
        "passed": False,
        "decision": "BLOCK_PHASE20_LGSSM_REFERENCE_HMC_VALIDATION",
        "config": config.normalized(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "jit_compile": bool(config.jit_compile),
        "jit_compile_false_runtime_executed": False,
        "training_executed": False,
        "gpu_sample_generation_executed": False,
        "posterior_validation_executed": False,
        "full_chain_hmc_executed": False,
        "hmc_tuning_executed": False,
        "nonclaims": NEUTRA_LGSSM_REFERENCE_HMC_VALIDATION_NONCLAIMS,
    }
    return {
        **payload,
        "artifact_hash": f"sha256:{_stable_payload_sha256(payload)}",
        "artifact_hash_semantics": "stable_json_sha256_excluding_artifact_hash_fields",
    }


def _compute_reference_posterior(
    config: NeuTraLGSSMReferenceHMCValidationConfig,
) -> Mapping[str, Any]:
    ensure_cpu_only_env()
    assert_cpu_only_env()
    import tensorflow as tf

    from bayesfilter.ssm import (
        stable_ssm_posterior_adapter_signature,
        stable_ssm_target_signature,
    )
    from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
        make_lgssm_generic_target_fixture,
    )

    fixture = make_lgssm_generic_target_fixture()
    target_signature = stable_ssm_target_signature(fixture.contract)
    adapter_signature = stable_ssm_posterior_adapter_signature(fixture.adapter)
    if target_signature != str(config.target_signature):
        raise NeuTraLGSSMReferenceHMCValidationError(
            "reference target signature mismatch"
        )
    if adapter_signature != str(config.adapter_signature):
        raise NeuTraLGSSMReferenceHMCValidationError(
            "reference adapter signature mismatch"
        )
    center = tf.convert_to_tensor(fixture.source_target.initial_parameters, dtype=tf.float64)
    grid_size = int(config.quadrature_grid_size)
    extent = float(config.quadrature_extent)
    axis0 = tf.linspace(center[0] - extent, center[0] + extent, grid_size)
    axis1 = tf.linspace(center[1] - extent, center[1] + extent, grid_size)
    mesh0, mesh1 = tf.meshgrid(axis0, axis1, indexing="ij")
    points = tf.stack([tf.reshape(mesh0, [-1]), tf.reshape(mesh1, [-1])], axis=1)
    values, _scores = fixture.adapter.log_prob_and_grad(points)
    values = tf.convert_to_tensor(values, dtype=tf.float64)
    max_log = tf.reduce_max(values)
    weights = tf.exp(values - max_log)
    weight_sum = tf.reduce_sum(weights)
    normalized = weights / weight_sum
    mean = tf.reduce_sum(points * normalized[:, tf.newaxis], axis=0)
    centered = points - mean
    covariance = tf.matmul(
        centered,
        centered * normalized[:, tf.newaxis],
        transpose_a=True,
    )
    finite = bool(
        tf.reduce_all(tf.math.is_finite(points)).numpy()
        and tf.reduce_all(tf.math.is_finite(values)).numpy()
        and tf.reduce_all(tf.math.is_finite(weights)).numpy()
        and tf.reduce_all(tf.math.is_finite(mean)).numpy()
        and tf.reduce_all(tf.math.is_finite(covariance)).numpy()
    )
    payload = {
        "schema": "bayesfilter.neutra.lgssm_quadrature_reference_posterior.v1",
        "reference_type": "deterministic_2d_quadrature",
        "analytic_exact_posterior": False,
        "target": "static_qr_lgssm_parameter_posterior",
        "likelihood_semantics": "exact_lgssm_kalman_log_likelihood",
        "prior_semantics": "zero_mean_independent_gaussian",
        "target_signature": target_signature,
        "adapter_signature": adapter_signature,
        "grid_size_per_axis": grid_size,
        "point_count": int(points.shape[0]),
        "center": _json_safe(center),
        "extent": extent,
        "axis0_min": float(axis0[0].numpy()),
        "axis0_max": float(axis0[-1].numpy()),
        "axis1_min": float(axis1[0].numpy()),
        "axis1_max": float(axis1[-1].numpy()),
        "log_normalizer_relative": float(
            (max_log + tf.math.log(weight_sum)).numpy()
        ),
        "mean": _json_safe(mean),
        "covariance": _json_safe(covariance),
        "finite": finite,
        "nonclaims": (
            "deterministic quadrature reference for this 2D fixture only",
            "not an analytic Gaussian posterior",
            "not a broad posterior correctness proof",
        ),
    }
    return {
        **payload,
        "reference_hash": f"sha256:{_stable_payload_sha256(payload)}",
        "reference_hash_semantics": "stable_json_sha256_excluding_hash_fields",
    }


def _run_reference_posterior_worker(
    config: NeuTraLGSSMReferenceHMCValidationConfig,
) -> Mapping[str, Any]:
    """Compute the TF quadrature reference outside the parent process."""

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        future = executor.submit(_compute_reference_posterior, config)
        return future.result()


def _worker_run_hmc_validation(worker_input: Mapping[str, Any]) -> Mapping[str, Any]:
    ensure_cpu_only_env()
    worker_start = time.monotonic()
    try:
        assert_cpu_only_env()
        import tensorflow as tf

        from bayesfilter.inference import (
            FixedTransportValueScoreAdapter,
            FullChainHMCConfig,
            load_frozen_neutra_artifact,
            run_full_chain_tfp_hmc,
            stable_frozen_neutra_artifact_signature,
        )
        from bayesfilter.ssm import (
            build_ssm_posterior_adapter,
            stable_ssm_posterior_adapter_signature,
            stable_ssm_target_signature,
        )
        from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
            LGSSM_GENERIC_TARGET_XLA_HMC_NONCLAIMS,
            lgssm_gaussian_prior_log_prob_and_grad,
            lgssm_qr_log_likelihood_and_grad,
            make_lgssm_generic_target_contract,
            make_lgssm_generic_target_fixture,
        )

        payload_path = Path(str(worker_input["payload_path"]))
        payload = _read_json_mapping(payload_path, "Phase 17 payload")
        loaded = load_frozen_neutra_artifact(
            payload,
            expected_target_signature=str(worker_input["target_signature"]),
        )
        fixture = make_lgssm_generic_target_fixture()
        target_signature = stable_ssm_target_signature(fixture.contract)
        adapter_signature = stable_ssm_posterior_adapter_signature(fixture.adapter)
        if target_signature != str(worker_input["target_signature"]):
            raise NeuTraLGSSMReferenceHMCValidationError(
                "worker target signature mismatch"
            )
        if adapter_signature != str(worker_input["adapter_signature"]):
            raise NeuTraLGSSMReferenceHMCValidationError(
                "worker adapter signature mismatch"
            )
        contract = make_lgssm_generic_target_contract(fixture.source_target)
        base_adapter = build_ssm_posterior_adapter(
            contract=contract,
            prior_log_prob_and_grad=lambda theta: lgssm_gaussian_prior_log_prob_and_grad(
                theta,
                prior_scale=fixture.source_target.prior_scale,
            ),
            filter_log_likelihood_and_grad=lambda theta: (
                lgssm_qr_log_likelihood_and_grad(
                    theta,
                    source_target=fixture.source_target,
                )
            ),
            dtype=tf.float64,
            target_scope=PHASE20_TARGET_SCOPE,
            evidence_path=PHASE20_EVIDENCE_PATH,
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            nonclaims=LGSSM_GENERIC_TARGET_XLA_HMC_NONCLAIMS
            + (
                "Phase 20 full-chain XLA diagnostic authority only",
                "no global target adapter promotion",
            ),
        )
        adapter = FixedTransportValueScoreAdapter(
            base_adapter=base_adapter,
            transport=loaded.transport,
            target_scope=PHASE20_TARGET_SCOPE,
            evidence_path=PHASE20_EVIDENCE_PATH,
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            nonclaims=NEUTRA_LGSSM_REFERENCE_HMC_VALIDATION_NONCLAIMS,
        )
        capability = adapter.value_score_capability()
        if not capability.is_accepted_full_chain_xla_diagnostic_authority:
            raise NeuTraLGSSMReferenceHMCValidationError(
                "Phase 20 adapter lacks scoped full-chain XLA diagnostic authority"
            )
        chain_count = int(worker_input["chain_count"])
        dimension = int(loaded.manifest.dimension)
        seed = int(worker_input["seed"])
        initial_state = _worker_initial_state(
            chain_count=chain_count,
            dimension=dimension,
            seed=seed,
            tf=tf,
        )
        hmc_config = FullChainHMCConfig(
            num_results=int(worker_input["num_results"]),
            num_burnin_steps=int(worker_input["num_burnin_steps"]),
            step_size=float(worker_input["step_size"]),
            num_leapfrog_steps=int(worker_input["num_leapfrog_steps"]),
            seed=(seed, seed + 17),
            use_xla=True,
            trace_policy="standard",
            target_status_trace_policy="none",
            target_scope=PHASE20_TARGET_SCOPE,
        )
        run_result = run_full_chain_tfp_hmc(
            adapter,
            initial_state,
            hmc_config,
        )
        samples = tf.convert_to_tensor(run_result.samples, dtype=tf.float64)
        flat_latent = tf.reshape(samples, [-1, dimension])
        flat_position = adapter.latent_to_position(flat_latent)
        position_samples = tf.reshape(
            flat_position,
            tf.concat([tf.shape(samples)[:-1], [dimension]], axis=0),
        )
        sample_mean = tf.reduce_mean(flat_position, axis=0)
        centered = flat_position - sample_mean
        total = tf.shape(flat_position)[0]
        denominator = tf.maximum(tf.cast(total - 1, tf.float64), 1.0)
        sample_covariance = tf.matmul(centered, centered, transpose_a=True) / denominator
        finite_latent_samples = bool(tf.reduce_all(tf.math.is_finite(samples)).numpy())
        finite_position_samples = bool(
            tf.reduce_all(tf.math.is_finite(position_samples)).numpy()
        )
        diagnostics = _json_safe(run_result.diagnostics)
        metadata = _json_safe(run_result.metadata)
        acceptance_rate = diagnostics.get("acceptance_rate")
        min_target_log_prob = diagnostics.get("min_target_log_prob")
        max_target_log_prob = diagnostics.get("max_target_log_prob")
        finite_target_log_prob = _finite_or_none(min_target_log_prob) and _finite_or_none(
            max_target_log_prob
        )
        passed = bool(
            finite_latent_samples
            and finite_position_samples
            and finite_target_log_prob
            and metadata.get("jit_compile") is True
            and capability.is_accepted_full_chain_xla_diagnostic_authority
        )
        return {
            "worker_index": int(worker_input["worker_index"]),
            "pid": os.getpid(),
            "return_code": 0 if passed else 1,
            "passed": passed,
            "seed": seed,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tensorflow_version": tf.__version__,
            "target_signature": target_signature,
            "adapter_signature": adapter_signature,
            "phase20_base_adapter_signature": base_adapter.adapter_signature(),
            "phase20_fixed_transport_adapter_signature": adapter.adapter_signature(),
            "artifact_signature": stable_frozen_neutra_artifact_signature(loaded),
            "transport_hash": loaded.manifest.transport_hash,
            "value_score_capability": {
                "value_score_authority": capability.value_score_authority,
                "xla_hmc_ready": bool(capability.xla_hmc_ready),
                "accepted_xla_hmc_authority": bool(
                    capability.is_accepted_xla_hmc_authority
                ),
                "full_chain_xla_diagnostic_ready": bool(
                    capability.full_chain_xla_diagnostic_ready
                ),
                "accepted_full_chain_xla_diagnostic_authority": bool(
                    capability.is_accepted_full_chain_xla_diagnostic_authority
                ),
                "target_scope": capability.target_scope,
                "evidence_path": capability.evidence_path,
            },
            "hmc_config": hmc_config.signature_payload(),
            "metadata": metadata,
            "diagnostics": diagnostics,
            "initial_state_shape": list(initial_state.shape.as_list()),
            "latent_sample_shape": list(samples.shape.as_list()),
            "position_sample_shape": list(position_samples.shape.as_list()),
            "retained_sample_count": int(flat_position.shape[0]),
            "retained_sample_count_per_chain": int(worker_input["num_results"]),
            "chain_count": chain_count,
            "sample_mean": _json_safe(sample_mean),
            "sample_covariance": _json_safe(sample_covariance),
            "finite_checks": {
                "latent_samples_finite": finite_latent_samples,
                "position_samples_finite": finite_position_samples,
                "target_log_prob_finite": finite_target_log_prob,
            },
            "acceptance_rate": acceptance_rate,
            "rhat": None,
            "ess": None,
            "rhat_ess_status": "not_computed_bounded_phase20_helper",
            "jit_compile": True,
            "jit_compile_false_runtime_executed": False,
            "training_executed": False,
            "gpu_sample_generation_executed": False,
            "posterior_validation_executed": True,
            "full_chain_hmc_executed": True,
            "hmc_tuning_executed": False,
            "elapsed_seconds": time.monotonic() - worker_start,
        }
    except Exception as exc:
        return {
            "worker_index": int(worker_input.get("worker_index", -1)),
            "pid": os.getpid(),
            "return_code": 1,
            "passed": False,
            "seed": int(worker_input.get("seed", -1)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "jit_compile": bool(worker_input.get("jit_compile", True)),
            "jit_compile_false_runtime_executed": False,
            "training_executed": False,
            "gpu_sample_generation_executed": False,
            "posterior_validation_executed": True,
            "full_chain_hmc_executed": False,
            "hmc_tuning_executed": False,
            "elapsed_seconds": time.monotonic() - worker_start,
        }


def _worker_initial_state(*, chain_count: int, dimension: int, seed: int, tf: Any) -> Any:
    base = tf.reshape(
        tf.cast(tf.range(chain_count * dimension), tf.float64),
        (chain_count, dimension),
    )
    centered = base - tf.reduce_mean(base, axis=0, keepdims=True)
    return 0.05 * centered + tf.cast(seed % 19, tf.float64) * 0.001


def _aggregate_worker_samples(workers: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    sample_counts = [
        int(row.get("retained_sample_count", 0))
        for row in workers
        if int(row.get("return_code", 1)) == 0
    ]
    if not sample_counts:
        return {
            "total_retained_sample_count": 0,
            "posterior_sample_summary": {
                "mean": None,
                "covariance": None,
                "sample_source": "no_successful_workers",
            },
        }
    means = []
    covs = []
    weights = []
    for row in workers:
        count = int(row.get("retained_sample_count", 0))
        if count <= 0:
            continue
        means.append(row.get("sample_mean"))
        covs.append(row.get("sample_covariance"))
        weights.append(count)
    total = float(sum(weights))
    mean = [
        sum(float(weight) * float(vector[index]) for weight, vector in zip(weights, means))
        / total
        for index in range(len(means[0]))
    ]
    covariance = [[0.0 for _ in mean] for _ in mean]
    for weight, worker_mean, worker_cov in zip(weights, means, covs):
        delta = [float(worker_mean[index]) - mean[index] for index in range(len(mean))]
        for i in range(len(mean)):
            for j in range(len(mean)):
                covariance[i][j] += float(weight) * (
                    float(worker_cov[i][j]) + delta[i] * delta[j]
                )
    covariance = [
        [entry / total for entry in row]
        for row in covariance
    ]
    return {
        "total_retained_sample_count": int(sum(weights)),
        "posterior_sample_summary": {
            "mean": mean,
            "covariance": covariance,
            "sample_source": "weighted_public_worker_summaries",
        },
    }


def _posterior_residuals(
    *,
    sample_summary: Mapping[str, Any],
    reference_posterior: Mapping[str, Any],
) -> Mapping[str, Any]:
    sample_mean = sample_summary.get("mean")
    sample_cov = sample_summary.get("covariance")
    reference_mean = reference_posterior.get("mean")
    reference_cov = reference_posterior.get("covariance")
    if sample_mean is None or sample_cov is None:
        return {
            "mean_residual": None,
            "cov_residual": None,
            "mean_max_abs_residual": math.inf,
            "cov_max_abs_residual": math.inf,
        }
    mean_residual = [
        float(sample_mean[index]) - float(reference_mean[index])
        for index in range(len(reference_mean))
    ]
    cov_residual = [
        [
            float(sample_cov[i][j]) - float(reference_cov[i][j])
            for j in range(len(reference_cov[i]))
        ]
        for i in range(len(reference_cov))
    ]
    return {
        "mean_residual": mean_residual,
        "cov_residual": cov_residual,
        "mean_max_abs_residual": max(abs(value) for value in mean_residual),
        "cov_max_abs_residual": max(
            abs(value)
            for row in cov_residual
            for value in row
        ),
    }


def _sample_diagnostics(
    *,
    workers: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    residuals: Mapping[str, Any],
    config: NeuTraLGSSMReferenceHMCValidationConfig,
) -> Mapping[str, Any]:
    acceptance_rates = [
        float(row["acceptance_rate"])
        for row in workers
        if row.get("acceptance_rate") is not None
        and math.isfinite(float(row["acceptance_rate"]))
    ]
    finite_samples = all(
        bool(row.get("finite_checks", {}).get("position_samples_finite"))
        for row in workers
        if int(row.get("return_code", 1)) == 0
    )
    finite_target = all(
        bool(row.get("finite_checks", {}).get("target_log_prob_finite"))
        for row in workers
        if int(row.get("return_code", 1)) == 0
    )
    if acceptance_rates:
        mean_acceptance = sum(acceptance_rates) / float(len(acceptance_rates))
        min_acceptance = min(acceptance_rates)
        max_acceptance = max(acceptance_rates)
    else:
        mean_acceptance = None
        min_acceptance = None
        max_acceptance = None
    acceptance_in_bounds = bool(
        acceptance_rates
        and min_acceptance >= float(config.min_acceptance_rate)
        and max_acceptance <= float(config.max_acceptance_rate)
    )
    return {
        "retained_sample_count_total": int(
            aggregate.get("total_retained_sample_count", 0)
        ),
        "posterior_sample_summary": aggregate.get("posterior_sample_summary"),
        "finite_samples": bool(finite_samples),
        "finite_target_log_prob": bool(finite_target),
        "acceptance_rates": acceptance_rates,
        "mean_acceptance_rate": mean_acceptance,
        "min_acceptance_rate": min_acceptance,
        "max_acceptance_rate": max_acceptance,
        "acceptance_rate_in_bounds": acceptance_in_bounds,
        "posterior_mean_abs_tolerance": float(config.posterior_mean_abs_tolerance),
        "posterior_cov_abs_tolerance": float(config.posterior_cov_abs_tolerance),
        "mean_max_abs_residual": residuals["mean_max_abs_residual"],
        "cov_max_abs_residual": residuals["cov_max_abs_residual"],
        "rhat": None,
        "ess": None,
        "rhat_ess_status": "not_computed_bounded_phase20_helper",
        "uncertainty_class": "descriptive_short_chain_gate",
    }


def _validate_config(config: NeuTraLGSSMReferenceHMCValidationConfig) -> None:
    if bool(config.require_cpu_hidden):
        try:
            assert_cpu_only_env()
        except RuntimeError as exc:
            raise NeuTraLGSSMReferenceHMCValidationError(str(exc)) from exc
    if not bool(config.jit_compile):
        raise NeuTraLGSSMReferenceHMCValidationError(
            "Phase 20 forbids jit_compile=false"
        )
    if int(config.seed) < 0:
        raise NeuTraLGSSMReferenceHMCValidationError("seed must be nonnegative")
    if int(config.worker_count) < 1 or int(config.worker_count) > 8:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "worker_count must be in 1..8 for Phase 20"
        )
    if int(config.chain_count) < 1 or int(config.chain_count) > 16:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "chain_count must be in 1..16 for Phase 20"
        )
    if int(config.worker_count) > int(config.chain_count):
        raise NeuTraLGSSMReferenceHMCValidationError(
            "worker_count must not exceed chain_count for Phase 20"
        )
    if int(config.num_results) <= 0 or int(config.num_results) > 512:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "num_results must be in 1..512 for Phase 20"
        )
    if int(config.num_burnin_steps) <= 0 or int(config.num_burnin_steps) > 512:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "num_burnin_steps must be in 1..512 for Phase 20"
        )
    if int(config.num_leapfrog_steps) <= 0 or int(config.num_leapfrog_steps) > 64:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "num_leapfrog_steps must be in 1..64 for Phase 20"
        )
    if float(config.step_size) <= 0.0:
        raise NeuTraLGSSMReferenceHMCValidationError("step_size must be positive")
    if int(config.quadrature_grid_size) < 21 or int(config.quadrature_grid_size) > 401:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "quadrature_grid_size must be in 21..401"
        )
    if int(config.quadrature_grid_size) % 2 != 1:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "quadrature_grid_size must be odd"
        )
    if float(config.quadrature_extent) <= 0.0:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "quadrature_extent must be positive"
        )


def _validate_provenance(
    config: NeuTraLGSSMReferenceHMCValidationConfig,
    *,
    payload: Mapping[str, Any],
    phase18: Mapping[str, Any],
    phase19: Mapping[str, Any],
) -> None:
    if payload.get("schema") != "bayesfilter.neutra.frozen_affine_diag.v1":
        raise NeuTraLGSSMReferenceHMCValidationError("Phase 17 payload schema mismatch")
    if payload.get("target_signature") != str(config.target_signature):
        raise NeuTraLGSSMReferenceHMCValidationError("payload target signature mismatch")
    if payload.get("source_adapter_signature") != str(config.adapter_signature):
        raise NeuTraLGSSMReferenceHMCValidationError("payload adapter signature mismatch")
    if bool(payload.get("hmc_executed")):
        raise NeuTraLGSSMReferenceHMCValidationError("payload records HMC execution")
    if phase18.get("decision") != "PASS_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE":
        raise NeuTraLGSSMReferenceHMCValidationError("Phase 18 diagnostic did not pass")
    if phase18.get("fixed_transport_adapter_signature") != str(
        config.fixed_transport_adapter_signature
    ):
        raise NeuTraLGSSMReferenceHMCValidationError(
            "Phase 18 fixed-transport adapter signature mismatch"
        )
    if phase18.get("jit_compile") is not True:
        raise NeuTraLGSSMReferenceHMCValidationError("Phase 18 was not JIT compiled")
    if phase18.get("jit_compile_false_runtime_executed") is not False:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "Phase 18 records jit_compile=false runtime"
        )
    if phase19.get("decision") != "PASS_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS":
        raise NeuTraLGSSMReferenceHMCValidationError("Phase 19 harness did not pass")
    if phase19.get("cuda_visible_devices") != "-1":
        raise NeuTraLGSSMReferenceHMCValidationError("Phase 19 was not CPU-hidden")
    if phase19.get("jit_compile") is not True:
        raise NeuTraLGSSMReferenceHMCValidationError("Phase 19 was not JIT compiled")
    if phase19.get("jit_compile_false_runtime_executed") is not False:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "Phase 19 records jit_compile=false runtime"
        )
    if phase19.get("posterior_validation_executed") is not False:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "Phase 19 should not have run posterior validation"
        )
    if phase19.get("gpu_sample_generation_executed") is not False:
        raise NeuTraLGSSMReferenceHMCValidationError(
            "Phase 19 records GPU sample generation"
        )


def _worker_chain_count(*, worker_index: int, worker_count: int, chain_count: int) -> int:
    base = chain_count // worker_count
    extra = chain_count % worker_count
    count = base + (1 if worker_index < extra else 0)
    return count


def _single_worker_field(workers: Sequence[Mapping[str, Any]], name: str) -> Any:
    values = sorted(
        {str(row[name]) for row in workers if row.get(name) is not None}
    )
    if len(values) == 1:
        return values[0]
    if not values:
        return None
    return values


def _finite_or_none(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise NeuTraLGSSMReferenceHMCValidationError(f"{label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise NeuTraLGSSMReferenceHMCValidationError(f"{label} must be a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    for key in (
        "artifact_hash",
        "artifact_hash_semantics",
        "reference_hash",
        "reference_hash_semantics",
    ):
        normalized.pop(key, None)
    blob = json.dumps(_json_safe(normalized), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tensorflow_imported() -> bool:
    import sys

    return "tensorflow" in sys.modules


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0.0 else "-inf"
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the Phase 20 LGSSM HMC validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-path", type=Path, default=DEFAULT_PHASE17_PAYLOAD_PATH)
    parser.add_argument(
        "--phase18-diagnostic-path",
        type=Path,
        default=DEFAULT_PHASE18_OUTPUT_PATH,
    )
    parser.add_argument(
        "--phase19-harness-path",
        type=Path,
        default=DEFAULT_PHASE19_OUTPUT_PATH,
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_PHASE20_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--chain-count", type=int, default=4)
    parser.add_argument("--num-results", type=int, default=64)
    parser.add_argument("--num-burnin-steps", type=int, default=64)
    parser.add_argument("--num-leapfrog-steps", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--jit-compile", choices=("true", "false"), default="true")
    args = parser.parse_args(argv)
    config = NeuTraLGSSMReferenceHMCValidationConfig(
        payload_path=args.payload_path,
        phase18_diagnostic_path=args.phase18_diagnostic_path,
        phase19_harness_path=args.phase19_harness_path,
        output_path=args.output_path,
        seed=args.seed,
        worker_count=args.worker_count,
        chain_count=args.chain_count,
        num_results=args.num_results,
        num_burnin_steps=args.num_burnin_steps,
        num_leapfrog_steps=args.num_leapfrog_steps,
        step_size=args.step_size,
        jit_compile=(args.jit_compile == "true"),
    )
    try:
        result = run_lgssm_reference_hmc_validation(config)
    except Exception as exc:
        result = phase20_error_payload(exc, config=config)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(config.output_path, result)
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "output_path": str(config.output_path),
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "passed": bool(result["passed"]),
                "decision": result["decision"],
                "output_path": str(config.output_path),
                "worker_count": result["worker_count"],
                "chain_count": result["chain_count"],
                "jit_compile": result["jit_compile"],
                "posterior_validation_executed": result[
                    "posterior_validation_executed"
                ],
                "mean_max_abs_residual": result["posterior_residuals"][
                    "mean_max_abs_residual"
                ],
                "cov_max_abs_residual": result["posterior_residuals"][
                    "cov_max_abs_residual"
                ],
                "nonclaims": NEUTRA_LGSSM_REFERENCE_HMC_VALIDATION_NONCLAIMS,
            },
            sort_keys=True,
        )
    )
    return 0 if bool(result["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
