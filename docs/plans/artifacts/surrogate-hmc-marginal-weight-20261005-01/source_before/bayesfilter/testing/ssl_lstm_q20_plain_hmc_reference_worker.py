"""CPU/XLA q=20 plain-HMC callbacks for an independent reference campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
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
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
PLAN = "docs/plans/bayesfilter-ssl-lstm-q20-seed-b-plain-hmc-reference-plan-2026-08-07.md"
INITIAL_STATES = (
    (0.73311370, 0.17273238, 0.58942510, 0.15892059),
    (0.73311370, 0.17273238, 0.58942510, 0.15892059),
    (0.44667563, -0.24131804, -0.58769660, 0.11989041),
    (0.44667563, -0.24131804, -0.58769660, 0.11989041),
)


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _hash(label: str, payload: Any) -> str:
    return hashlib.sha256(_canonical({"label": label, "payload": payload})).hexdigest()


def expected_lineage_payload() -> Mapping[str, str]:
    return {
        "coordinate_signature": TARGET_SIGNATURE,
        "metric_signature": _hash("bayesfilter.ssl_lstm.q20.identity_metric.v2", {"dimension": 4, "matrix": "identity"}),
        "private_start_bank_content_signature": _hash("bayesfilter.ssl_lstm.q20.mode_start_bank.v1", {"states": INITIAL_STATES}),
        "common_state_signature": _hash("bayesfilter.ssl_lstm.q20.plain_reference_state.v1", {"q": Q, "target_signature": TARGET_SIGNATURE, "route": "plain_fixed_hmc_current_batch_native"}),
    }


def _configure_tensorflow() -> tuple[Any, Mapping[str, Any]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise GridSearchTargetVeto("plain HMC reference requires CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise GridSearchTargetVeto("plain HMC reference worker can see a GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf, {"cuda_visible_devices": "-1", "physical_gpus": [], "jit_compile": True, "dtype": "float64"}


class PlainQ20Adapter:
    """Identity-coordinate adapter over the current batch-native target."""

    def __init__(self, target: Any) -> None:
        self.target = target
        self.parameter_dim = 4
        self.parameter_names = tuple(target.parameter_names)
        self.target_scope = f"{target.target_scope}:plain_fixed_hmc_reference"
        self.target_status_invalid_rows_become_nonfinite = True

    def target_signature(self) -> str:
        return self.target.target_signature()

    def adapter_signature(self) -> str:
        return _hash("bayesfilter.ssl_lstm.q20.plain_hmc_adapter.v2", {"target_signature": self.target_signature(), "target_adapter_signature": self.target.adapter_signature(), "target_scope": self.target_scope, "coordinate": "identity"})

    def value_score_capability(self) -> Any:
        from bayesfilter.inference.posterior_adapter import ValueScoreCapability
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="bayesfilter.testing.ssl_lstm_q20_plain_hmc_reference_worker",
            evidence_path=PLAN,
            target_scope=self.target_scope,
            nonclaims=("plain fixed-HMC reference only", "no NeuTra transport", "no posterior claim without sequential screen"),
        )

    def latent_to_position(self, values: Any) -> Any:
        import tensorflow as tf
        return tf.convert_to_tensor(values, tf.float64)

    def log_prob_and_grad(self, values: Any) -> tuple[Any, Any]:
        import tensorflow as tf
        tensor = tf.convert_to_tensor(values, tf.float64)
        if tensor.shape.rank == 1:
            batch = tf.ensure_shape(tensor[tf.newaxis, :], [1, 4])
            value, score, _status = self.target.neutra_batch_log_prob_and_grad_status(batch)
            return value[0], score[0]
        if tensor.shape.rank == 2:
            value, score, _status = self.target.neutra_batch_log_prob_and_grad_status(tensor)
            return value, score
        raise ValueError("plain q20 state must have rank one or two")

    def log_prob_and_grad_status(self, values: Any) -> tuple[Any, Any, Mapping[str, Any]]:
        import tensorflow as tf

        def public_status(raw: Mapping[str, Any]) -> Mapping[str, Any]:
            status = {
                key: raw[key]
                for key in ("status_code", "valid_pre_regularized_score", "floor_count_value")
                if key in raw
            }
            if "min_innovation_eigenvalue" in raw and "innovation_condition_estimate" in raw:
                status["min_innovation_eigenvalue"] = raw["min_innovation_eigenvalue"]
                status["innovation_condition_estimate"] = raw["innovation_condition_estimate"]
            return status

        tensor = tf.convert_to_tensor(values, tf.float64)
        if tensor.shape.rank == 1:
            tensor = tf.ensure_shape(tensor[tf.newaxis, :], [1, 4])
            value, score, status = self.target.neutra_batch_log_prob_and_grad_status(tensor)
            filtered = public_status(status)
            return value[0], score[0], {key: tensor_value[0] for key, tensor_value in filtered.items()}
        value, score, status = self.target.neutra_batch_log_prob_and_grad_status(tensor)
        return value, score, public_status(status)

    def target_status_telemetry(self, values: Any) -> Mapping[str, Any]:
        return self.log_prob_and_grad_status(values)[2]


class _Callbacks:
    def __init__(self, request: Any, tf: Any) -> None:
        from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import batch_native_complexity_posterior_target
        target = batch_native_complexity_posterior_target(Q, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
        if target.target_signature() != TARGET_SIGNATURE:
            raise GridSearchTargetVeto("plain reference target signature mismatch")
        self.request = request
        self.tf = tf
        self.adapter = PlainQ20Adapter(target)
        self.initial_state = tf.constant(INITIAL_STATES, tf.float64)
        self._runners: dict[tuple[int, int, int, bool], Any] = {}

    def _runner(self, *, num_results: int, num_burnin_steps: int, step_size: float, leapfrog: int, seed: tuple[int, int], tuning: bool) -> Any:
        from bayesfilter.inference.hmc import FullChainHMCConfig, build_reusable_full_chain_tfp_hmc_runner
        from bayesfilter.inference.hmc_tuning import HMCTuningPolicy
        key = (int(num_results), int(num_burnin_steps), int(leapfrog), bool(tuning))
        if key not in self._runners:
            policy = HMCTuningPolicy.fixed_mass_dual_averaging(num_adaptation_steps=64, target_accept_prob=0.70, source=PLAN) if tuning else None
            config = FullChainHMCConfig(num_results=num_results, num_burnin_steps=num_burnin_steps, step_size=step_size, num_leapfrog_steps=leapfrog, seed=seed, use_xla=True, trace_policy="standard", target_status_trace_policy="per_chain_step", tuning_policy=policy, target_scope=self.adapter.target_scope)
            self._runners[key] = build_reusable_full_chain_tfp_hmc_runner(self.adapter, self.initial_state, config)
        return self._runners[key]

    def tune(self, request: Any) -> FixedMetricTuneOutcome:
        runner = self._runner(num_results=1, num_burnin_steps=64, step_size=request.initial_step_size, leapfrog=request.num_leapfrog_steps, seed=request.seed, tuning=True)
        result = runner.run(seed=request.seed, step_size=request.initial_step_size)
        step_trace = self.tf.convert_to_tensor(result.trace["step_size"], self.tf.float64)
        tuned = float(self.tf.reduce_mean(step_trace[-1]).numpy())
        if not math.isfinite(tuned) or tuned <= 0.0:
            raise CandidateTuneRejected("nonfinite_adapted_step_size")
        return FixedMetricTuneOutcome(num_leapfrog_steps=request.num_leapfrog_steps, seed=request.seed, tuned_step_size=tuned, lineage=request.lineage)

    def screen(self, request: Any) -> FixedMetricScreenOutcome:
        from bayesfilter.inference.hmc_verification import evaluate_hmc_acceptance_evidence
        runner = self._runner(num_results=request.num_results, num_burnin_steps=1, step_size=request.tuned_step_size, leapfrog=request.num_leapfrog_steps, seed=request.seed, tuning=False)
        result = runner.run(seed=request.seed, step_size=request.tuned_step_size)
        evidence = evaluate_hmc_acceptance_evidence(samples=result.samples.numpy(), log_accept_ratio=result.trace["log_accept_ratio"].numpy(), is_accepted=result.trace["is_accepted"].numpy(), target_log_prob=result.trace["target_log_prob"].numpy(), policy=self.request.acceptance_policy, native_divergence_status="not_exposed_by_kernel", native_divergence_count=None)
        return FixedMetricScreenOutcome(num_leapfrog_steps=request.num_leapfrog_steps, replication_index=request.replication_index, seed=request.seed, tuned_step_size=request.tuned_step_size, lineage=request.lineage, acceptance_evidence_payload=evidence.payload())


def q20_plain_hmc_worker_factory(request: Any) -> FixedMetricCandidateRunners:
    from bayesfilter.inference.hmc_fixed_metric_grid_search import FixedMetricSearchLineage
    expected = expected_lineage_payload()
    if request.lineage.payload() != expected:
        raise GridSearchTargetVeto("plain reference lineage mismatch")
    tf, _device = _configure_tensorflow()
    callbacks = _Callbacks(request, tf)
    return FixedMetricCandidateRunners(tune_runner=callbacks.tune, screen_runner=callbacks.screen)


def run_q20_plain_hmc_candidate(request: Any) -> Any:
    runners = q20_plain_hmc_worker_factory(request)
    return run_fixed_metric_candidate(round_index=request.round_index, num_leapfrog_steps=request.num_leapfrog_steps, config=request.config, lineage=request.lineage, acceptance_policy=request.acceptance_policy, tune_runner=runners.tune_runner, screen_runner=runners.screen_runner)


__all__ = ["INITIAL_STATES", "PLAN", "PlainQ20Adapter", "TARGET_SIGNATURE", "expected_lineage_payload", "q20_plain_hmc_worker_factory", "run_q20_plain_hmc_candidate"]
