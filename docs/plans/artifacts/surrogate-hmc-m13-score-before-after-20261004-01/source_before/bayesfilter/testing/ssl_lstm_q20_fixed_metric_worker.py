"""Spawn-safe q=20 SSL-LSTM fixed-metric HMC candidate callbacks.

TensorFlow is imported only inside the worker factory so a process-owning
parent can use ``spawn`` without initializing CUDA first.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping

from bayesfilter.inference.hmc_fixed_metric_grid_search import (
    CandidateScreenRejected,
    CandidateTuneRejected,
    FixedMetricCandidateRunners,
    FixedMetricScreenOutcome,
    FixedMetricTuneOutcome,
    GridSearchTargetVeto,
    run_fixed_metric_candidate,
)


Q = 20
TARGET_SIGNATURE = "302d50b16ac4804e1656527bbbfdb535ce46049536b3e7187fe5bd223e1cdb71"
PLAN = (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-process-grid-hmc-tuning-plan-2026-07-20.md"
)
TUNE_RESULTS = 1
TUNE_BURNIN = 64
TUNE_ADAPTATION_STEPS = 64
SCREEN_BURNIN = 1
TARGET_ACCEPTANCE = 0.70
HOST_RAM_CAP_BYTES = 64 * 1024**3
INITIAL_OFFSETS = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _signature(label: str, payload: Any) -> str:
    return hashlib.sha256(_canonical({"label": label, "payload": payload})).hexdigest()


def expected_lineage_payload() -> Mapping[str, str]:
    starts = {
        "center": (0.35, -0.08, 0.65, 0.05),
        "offsets": INITIAL_OFFSETS,
    }
    common_state = {
        "q": Q,
        "target_signature": TARGET_SIGNATURE,
        "coordinate": "plain_target_free_coordinates",
        "metric": "identity",
        "starts": starts,
        "dtype": "float64",
        "jit_compile": True,
    }
    return {
        "coordinate_signature": TARGET_SIGNATURE,
        "metric_signature": _signature(
            "bayesfilter.ssl_lstm_q20.identity_metric.v1",
            {"dimension": 4, "matrix": "identity"},
        ),
        "private_start_bank_content_signature": _signature(
            "bayesfilter.ssl_lstm_q20.start_bank.v1", starts
        ),
        "common_state_signature": _signature(
            "bayesfilter.ssl_lstm_q20.common_state.v1", common_state
        ),
    }


def _require_lineage(request: Any) -> None:
    expected = expected_lineage_payload()
    actual = request.lineage.payload()
    if actual != expected:
        raise GridSearchTargetVeto("q20 fixed-metric worker lineage mismatch")


def _configure_tensorflow() -> tuple[Any, Mapping[str, Any]]:
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise GridSearchTargetVeto("TF_FORCE_GPU_ALLOW_GROWTH=true is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise GridSearchTargetVeto("q20 HMC worker requires one visible GPU")

    import tensorflow as tf

    physical = tf.config.list_physical_devices("GPU")
    if len(physical) != 1:
        raise GridSearchTargetVeto(
            f"q20 HMC worker requires exactly one visible GPU, found {len(physical)}"
        )
    try:
        tf.config.experimental.set_memory_growth(physical[0], True)
    except RuntimeError as error:
        raise GridSearchTargetVeto(
            "GPU memory growth could not be set before initialization"
        ) from error
    if tf.config.experimental.get_memory_growth(physical[0]) is not True:
        raise GridSearchTargetVeto("GPU memory growth verification failed")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise GridSearchTargetVeto("q20 HMC worker logical GPU verification failed")
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except (RuntimeError, ValueError):
        pass
    return tf, {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "memory_growth": True,
        "logical_gpus": tuple(device.name for device in logical),
        "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
    }


class _Q20PlainHMCAdapter:
    def __init__(self, target: Any) -> None:
        self.target = target
        self.parameter_dim = 4
        self.parameter_names = tuple(target.parameter_names)
        self.target_scope = f"{target.target_scope}:plain_fixed_metric_grid"

    def adapter_signature(self) -> str:
        return _signature(
            "bayesfilter.ssl_lstm_q20.plain_hmc_adapter.v1",
            {
                "target_signature": self.target.target_signature(),
                "target_adapter_signature": self.target.adapter_signature(),
                "scope": self.target_scope,
            },
        )

    def target_signature(self) -> str:
        return self.target.target_signature()

    def value_score_capability(self) -> Any:
        from bayesfilter.inference.posterior_adapter import ValueScoreCapability

        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="ssl_lstm_q20_plain_fixed_metric_grid",
            evidence_path=PLAN,
            target_scope=self.target_scope,
            nonclaims=(
                "plain-target HMC tuning only",
                "no NeuTra transport",
                "no convergence or posterior correctness claim",
            ),
        )

    def log_prob_and_grad(self, state: Any) -> tuple[Any, Any]:
        import tensorflow as tf

        values = tf.convert_to_tensor(state, tf.float64)
        if values.shape.rank == 1:
            return self.target.value_and_score(values)
        if values.shape.rank == 2:
            return self.target.batch_value_and_score(values)
        raise ValueError("q20 HMC state must have rank one or two")


class _CandidateCallbacks:
    def __init__(self, request: Any, tf: Any, device: Mapping[str, Any]) -> None:
        from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (
            PRIOR_CENTER,
            complexity_posterior_target,
        )

        target = complexity_posterior_target(Q, jit_compile=True)
        if target.target_signature() != TARGET_SIGNATURE:
            raise GridSearchTargetVeto("q20 target semantic signature mismatch")
        self.request = request
        self.tf = tf
        self.device = dict(device)
        self.adapter = _Q20PlainHMCAdapter(target)
        center = tf.convert_to_tensor(PRIOR_CENTER, tf.float64)
        self.initial_state = center[tf.newaxis, :] + tf.constant(
            INITIAL_OFFSETS, tf.float64
        )
        if tuple(self.initial_state.shape) != (4, 4):
            raise GridSearchTargetVeto("q20 initial-state shape mismatch")
        self._runners: dict[tuple[int, int, int, bool], Any] = {}

    def _runner(
        self,
        *,
        num_results: int,
        num_burnin_steps: int,
        step_size: float,
        num_leapfrog_steps: int,
        seed: tuple[int, int],
        tuning: bool,
    ) -> Any:
        from bayesfilter.inference.hmc import (
            FullChainHMCConfig,
            build_reusable_full_chain_tfp_hmc_runner,
        )
        from bayesfilter.inference.hmc_tuning import HMCTuningPolicy

        key = (
            int(num_results),
            int(num_burnin_steps),
            int(num_leapfrog_steps),
            bool(tuning),
        )
        cached = self._runners.get(key)
        if cached is not None:
            return cached
        policy = (
            HMCTuningPolicy.fixed_mass_dual_averaging(
                num_adaptation_steps=TUNE_ADAPTATION_STEPS,
                target_accept_prob=TARGET_ACCEPTANCE,
                source=PLAN,
            )
            if tuning
            else None
        )
        config = FullChainHMCConfig(
            num_results=int(num_results),
            num_burnin_steps=int(num_burnin_steps),
            step_size=float(step_size),
            num_leapfrog_steps=int(num_leapfrog_steps),
            seed=tuple(int(item) for item in seed),
            use_xla=True,
            trace_policy="standard",
            tuning_policy=policy,
            target_scope=self.adapter.target_scope,
        )
        runner = build_reusable_full_chain_tfp_hmc_runner(
            self.adapter, self.initial_state, config
        )
        self._runners[key] = runner
        return runner

    def tune(self, request: Any) -> FixedMetricTuneOutcome:
        if request.lineage != self.request.lineage:
            raise GridSearchTargetVeto("q20 tune lineage changed inside worker")
        runner = self._runner(
            num_results=TUNE_RESULTS,
            num_burnin_steps=TUNE_BURNIN,
            step_size=request.initial_step_size,
            num_leapfrog_steps=request.num_leapfrog_steps,
            seed=request.seed,
            tuning=True,
        )
        result = runner.run(seed=request.seed, step_size=request.initial_step_size)
        diagnostics = result.diagnostics
        finite_step = diagnostics.get("final_step_size_finite")
        if finite_step is None or not bool(finite_step.numpy()):
            raise CandidateTuneRejected("nonfinite_adapted_step_size")
        step_trace = self.tf.convert_to_tensor(result.trace.get("step_size"), self.tf.float64)
        if step_trace.shape.rank not in {1, 2}:
            raise CandidateTuneRejected("nonfinite_adapted_step_size")
        final_step = step_trace[-1]
        if not bool(self.tf.reduce_all(self.tf.math.is_finite(final_step)).numpy()):
            raise CandidateTuneRejected("nonfinite_adapted_step_size")
        if step_trace.shape.rank == 2:
            if tuple(step_trace.shape[1:]) != (4,):
                raise CandidateTuneRejected("nonfinite_adapted_step_size")
            spread = float(
                (self.tf.reduce_max(final_step) - self.tf.reduce_min(final_step)).numpy()
            )
            scale = max(1.0, float(self.tf.reduce_max(self.tf.abs(final_step)).numpy()))
            if spread > 1.0e-10 * scale:
                raise CandidateTuneRejected("nonfinite_adapted_step_size")
        tuned = float(self.tf.reduce_mean(final_step).numpy())
        if not math.isfinite(tuned) or tuned <= 0.0:
            raise CandidateTuneRejected("nonfinite_adapted_step_size")
        return FixedMetricTuneOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            seed=request.seed,
            tuned_step_size=tuned,
            lineage=request.lineage,
        )

    def screen(self, request: Any) -> FixedMetricScreenOutcome:
        from bayesfilter.inference.hmc_verification import (
            evaluate_hmc_acceptance_evidence,
        )

        if request.lineage != self.request.lineage:
            raise GridSearchTargetVeto("q20 screen lineage changed inside worker")
        runner = self._runner(
            num_results=request.num_results,
            num_burnin_steps=SCREEN_BURNIN,
            step_size=request.tuned_step_size,
            num_leapfrog_steps=request.num_leapfrog_steps,
            seed=request.seed,
            tuning=False,
        )
        result = runner.run(seed=request.seed, step_size=request.tuned_step_size)
        required = {"is_accepted", "log_accept_ratio", "target_log_prob"}
        if not required.issubset(result.trace):
            raise CandidateScreenRejected("candidate_screen_execution_failed")
        divergence_status = str(
            result.diagnostics.get("native_divergence_status", "not_exposed_by_kernel")
        )
        divergence_count = result.diagnostics.get("divergence_count")
        if hasattr(divergence_count, "numpy"):
            divergence_count = int(divergence_count.numpy())
        evidence = evaluate_hmc_acceptance_evidence(
            samples=result.samples.numpy(),
            log_accept_ratio=result.trace["log_accept_ratio"].numpy(),
            is_accepted=result.trace["is_accepted"].numpy(),
            target_log_prob=result.trace["target_log_prob"].numpy(),
            policy=self.request.acceptance_policy,
            native_divergence_status=divergence_status,
            native_divergence_count=divergence_count,
        )
        return FixedMetricScreenOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            replication_index=request.replication_index,
            seed=request.seed,
            tuned_step_size=request.tuned_step_size,
            lineage=request.lineage,
            acceptance_evidence_payload=evidence.payload(),
        )


def q20_fixed_metric_worker_factory(request: Any) -> FixedMetricCandidateRunners:
    """Construct target-specific callbacks after spawn and GPU setup."""

    _require_lineage(request)
    started = time.perf_counter()
    tf, device = _configure_tensorflow()
    callbacks = _CandidateCallbacks(request, tf, device)
    host_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if host_rss > HOST_RAM_CAP_BYTES:
        raise GridSearchTargetVeto("q20 HMC worker exceeded 64 GiB host RSS")
    telemetry_path = os.environ.get("BAYESFILTER_Q20_WORKER_TELEMETRY_DIR")
    if telemetry_path:
        path = Path(telemetry_path)
        path.mkdir(parents=True, exist_ok=True)
        try:
            allocator = {
                key + "_bytes": int(value)
                for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
            }
        except (RuntimeError, ValueError):
            allocator = {"status": "unavailable"}
        receipt = {
            "schema": "bayesfilter.ssl_lstm_q20.fixed_metric_worker.v1",
            "pid": os.getpid(),
            "round_index": request.round_index,
            "num_leapfrog_steps": request.num_leapfrog_steps,
            "target_signature": callbacks.adapter.target_signature(),
            "adapter_signature": callbacks.adapter.adapter_signature(),
            "lineage": request.lineage.payload(),
            "device": device,
            "host_ru_maxrss_bytes": host_rss,
            "gpu_allocator": allocator,
            "factory_seconds": time.perf_counter() - started,
            "all_draws_discarded": True,
        }
        (path / f"worker-r{request.round_index}-l{request.num_leapfrog_steps}-p{os.getpid()}.json").write_bytes(
            _canonical(receipt)
        )
    return FixedMetricCandidateRunners(
        tune_runner=callbacks.tune,
        screen_runner=callbacks.screen,
    )


def run_q20_candidate_worker(request: Any) -> Any:
    """Run one complete public-API candidate inside a spawned child."""

    runners = q20_fixed_metric_worker_factory(request)
    return run_fixed_metric_candidate(
        round_index=request.round_index,
        num_leapfrog_steps=request.num_leapfrog_steps,
        config=request.config,
        lineage=request.lineage,
        acceptance_policy=request.acceptance_policy,
        tune_runner=runners.tune_runner,
        screen_runner=runners.screen_runner,
    )


def q20_hmc_rate_probe(seed: tuple[int, int]) -> Mapping[str, Any]:
    """Measure one current-source plain-target transition-leapfrog rate."""

    from bayesfilter.inference.hmc_fixed_metric_grid_search import (
        FixedMetricCandidateWorkerRequest,
        FixedMetricGridSearchConfig,
        FixedMetricSearchLineage,
    )
    from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy

    lineage = FixedMetricSearchLineage(**expected_lineage_payload())
    request = FixedMetricCandidateWorkerRequest(
        round_index=0,
        num_leapfrog_steps=3,
        config=FixedMetricGridSearchConfig(refinement_rounds=0),
        lineage=lineage,
        acceptance_policy=HMCAcceptancePolicy(),
    )
    _require_lineage(request)
    tf, device = _configure_tensorflow()
    callbacks = _CandidateCallbacks(request, tf, device)
    runner = callbacks._runner(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.01,
        num_leapfrog_steps=1,
        seed=seed,
        tuning=False,
    )
    first = runner.run(seed=seed, step_size=0.01)
    warm_rows = []
    for index in range(2):
        folded = (int(seed[0]), int(seed[1]) + index + 1)
        result = runner.run(seed=folded, step_size=0.01)
        if not bool(tf.reduce_all(tf.math.is_finite(result.samples)).numpy()):
            raise GridSearchTargetVeto("q20 HMC rate probe produced nonfinite samples")
        wall = float(result.metadata["sample_chain_call_s"])
        warm_rows.append(
            {
                "seed": folded,
                "sample_chain_seconds": wall,
                "seconds_per_transition_leapfrog": wall / 3.0,
                "sample_device": str(result.samples.device),
            }
        )
    try:
        allocator = {
            key + "_bytes": int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        }
    except (RuntimeError, ValueError):
        allocator = {"status": "unavailable"}
    return {
        "schema": "bayesfilter.ssl_lstm_q20.hmc_rate_probe.v1",
        "target_signature": callbacks.adapter.target_signature(),
        "adapter_signature": callbacks.adapter.adapter_signature(),
        "first_call_seconds": float(first.metadata["sample_chain_call_s"]),
        "warm_rows": tuple(warm_rows),
        "warm_seconds_per_transition_leapfrog_max": max(
            row["seconds_per_transition_leapfrog"] for row in warm_rows
        ),
        "device": device,
        "gpu_allocator": allocator,
        "host_ru_maxrss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }


def q20_fixed_kernel_hmc_test(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run one fresh non-retained mechanics test for a frozen survivor."""

    from bayesfilter.inference.hmc_fixed_metric_grid_search import (
        FixedMetricCandidateWorkerRequest,
        FixedMetricGridSearchConfig,
        FixedMetricSearchLineage,
    )
    from bayesfilter.inference.hmc_verification import (
        HMCAcceptancePolicy,
        evaluate_hmc_acceptance_evidence,
    )

    lineage = FixedMetricSearchLineage(**dict(payload["lineage"]))
    policy = HMCAcceptancePolicy()
    request = FixedMetricCandidateWorkerRequest(
        round_index=0,
        num_leapfrog_steps=int(payload["num_leapfrog_steps"]),
        config=FixedMetricGridSearchConfig(refinement_rounds=0),
        lineage=lineage,
        acceptance_policy=policy,
    )
    _require_lineage(request)
    tf, device = _configure_tensorflow()
    callbacks = _CandidateCallbacks(request, tf, device)
    seed = tuple(int(item) for item in payload["seed"])
    runner = callbacks._runner(
        num_results=64,
        num_burnin_steps=64,
        step_size=float(payload["step_size"]),
        num_leapfrog_steps=request.num_leapfrog_steps,
        seed=seed,
        tuning=False,
    )
    result = runner.run(seed=seed, step_size=float(payload["step_size"]))
    divergence_status = str(
        result.diagnostics.get("native_divergence_status", "not_exposed_by_kernel")
    )
    divergence_count = result.diagnostics.get("divergence_count")
    if hasattr(divergence_count, "numpy"):
        divergence_count = int(divergence_count.numpy())
    evidence = evaluate_hmc_acceptance_evidence(
        samples=result.samples.numpy(),
        log_accept_ratio=result.trace["log_accept_ratio"].numpy(),
        is_accepted=result.trace["is_accepted"].numpy(),
        target_log_prob=result.trace["target_log_prob"].numpy(),
        policy=policy,
        native_divergence_status=divergence_status,
        native_divergence_count=divergence_count,
    )
    return {
        "schema": "bayesfilter.ssl_lstm_q20.fixed_kernel_hmc_test.v1",
        "status": "PASSED" if evidence.promotion_eligible else "VETOED",
        "target_signature": callbacks.adapter.target_signature(),
        "adapter_signature": callbacks.adapter.adapter_signature(),
        "lineage": lineage.payload(),
        "num_leapfrog_steps": request.num_leapfrog_steps,
        "step_size": float(payload["step_size"]),
        "seed": seed,
        "num_results": 64,
        "num_burnin_steps": 64,
        "acceptance_evidence": evidence.payload(),
        "sample_chain_seconds": float(result.metadata["sample_chain_call_s"]),
        "device": device,
        "raw_samples_retained": False,
        "nonclaims": (
            "short fixed-kernel HMC mechanics test only",
            "no convergence or posterior correctness claim",
        ),
    }


__all__ = [
    "HOST_RAM_CAP_BYTES",
    "INITIAL_OFFSETS",
    "PLAN",
    "Q",
    "SCREEN_BURNIN",
    "TARGET_ACCEPTANCE",
    "TARGET_SIGNATURE",
    "TUNE_ADAPTATION_STEPS",
    "TUNE_BURNIN",
    "TUNE_RESULTS",
    "expected_lineage_payload",
    "q20_fixed_kernel_hmc_test",
    "q20_fixed_metric_worker_factory",
    "q20_hmc_rate_probe",
    "run_q20_candidate_worker",
]
