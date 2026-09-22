"""Optional finite starting-step search before bootstrap, with no tuning authority.

The target, affine mass and TFP transition are unchanged. Four simultaneous
momentum probes nominate a startup pair; fresh bootstrap must still check it.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import time

POLICY = "bayesfilter.hmc_bootstrap_finite_initialization.v1"


class BootstrapMomentumProbe:
    """Stable batch graphs, cached by L; epsilon and seed are runtime inputs."""

    def __init__(self, adapter, *, count=4, use_xla=True,
                 target_status_trace_policy="per_chain_step"):
        import tensorflow as tf
        from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
        self.adapter, self.count, self.use_xla = adapter, count, use_xla
        if target_status_trace_policy not in {"none", "per_chain_step"}:
            raise ValueError("unknown bootstrap probe target status policy")
        self.status_required = target_status_trace_policy == "per_chain_step"
        self.shape = (count, adapter.parameter_dim)
        self.target = reviewed_value_score_target_fn(adapter, require_batched=True)
        self.graphs = {}

        @tf.function(input_signature=[tf.TensorSpec(self.shape, tf.float64)],
                     jit_compile=use_xla, autograph=False)
        def initial(state):
            if getattr(adapter, "supports_retained_value_score_status", False):
                value, score, raw_status = adapter.log_prob_and_grad_status(state)
            else:
                value, score = adapter.log_prob_and_grad(state)
                raw_status = {}
            # Combined value/score diagnostics can be richer than the public
            # HMC status schema (q20 has a lone optional eigenvalue). Use the
            # adapter's public telemetry boundary, and preserve the raw data.
            status = self._status(state)
            endpoint = self._endpoint(state, value, score, status)
            endpoint["raw_status"] = raw_status
            return endpoint
        self.initial = initial

    def _status(self, state):
        # An absent telemetry policy is explicit; never manufacture valid bits.
        return self.adapter.target_status_telemetry(state) if self.status_required else {}

    def _endpoint(self, state, value, score, status):
        import tensorflow as tf
        from bayesfilter.inference.hmc_verification import (
            TARGET_STATUS_TELEMETRY_CORE_FIELDS,
            TARGET_STATUS_TELEMETRY_OPTIONAL_CONDITIONING_FIELDS,
        )
        if self.status_required and any(key not in status for key in TARGET_STATUS_TELEMETRY_CORE_FIELDS):
            raise ValueError("bootstrap initialization requires complete target status")
        status = {key: tf.convert_to_tensor(item) for key, item in status.items()}
        if any(item.shape != (self.count,) for item in status.values()):
            raise ValueError("bootstrap probe status must preserve its batch dimension")
        if self.status_required and (not status["status_code"].dtype.is_integer or
                not status["valid_pre_regularized_score"].dtype.is_bool or
                not status["floor_count_value"].dtype.is_integer):
            raise TypeError("bootstrap probe status has invalid dtypes")
        optional = [key in status for key in TARGET_STATUS_TELEMETRY_OPTIONAL_CONDITIONING_FIELDS]
        if any(optional) and not all(optional):
            raise ValueError("bootstrap probe conditioning status is incomplete")
        if state.shape != self.shape or score.shape != self.shape or value.shape != (self.count,):
            raise ValueError("bootstrap probe value/score must preserve its batch dimension")
        flags = {
            "state_finite": tf.reduce_all(tf.math.is_finite(state), axis=-1),
            "value_finite": tf.math.is_finite(value),
            "score_finite": tf.reduce_all(tf.math.is_finite(score), axis=-1),
        }
        if self.status_required:
            flags["status_valid"] = ((status["status_code"] == 0) & status["valid_pre_regularized_score"]
                                     & (status["floor_count_value"] >= 0))
        for key in TARGET_STATUS_TELEMETRY_OPTIONAL_CONDITIONING_FIELDS:
            if key in status:
                flags[key + "_finite"] = tf.math.is_finite(status[key])
        healthy = tf.reduce_all(tf.stack(list(flags.values())), axis=0)
        return {"state": state, "physical_state": self.adapter.latent_to_position(state),
                "value": value, "score": score, "status": status,
                "flags": flags, "healthy": healthy}

    def __call__(self, state, epsilon, leapfrogs, seed):
        import tensorflow as tf
        import tensorflow_probability as tfp
        if leapfrogs not in self.graphs:
            @tf.function(input_signature=[tf.TensorSpec(self.shape, tf.float64),
                         tf.TensorSpec([], tf.float64), tf.TensorSpec([2], tf.int32)],
                         jit_compile=self.use_xla, autograph=False)
            def probe(points, step, stream):
                kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=self.target,
                    step_size=step, num_leapfrog_steps=leapfrogs, state_gradients_are_stopped=True)
                initial = kernel.bootstrap_results(points)
                retained, result = kernel.one_step(points, initial, seed=stream)
                initial_status = self._status(points)
                proposed_status = self._status(result.proposed_state)
                retained_status = {key: tf.where(result.is_accepted, proposed_status[key], item)
                                   for key, item in initial_status.items()}
                return {
                    "retained": self._endpoint(retained, result.accepted_results.target_log_prob,
                        result.accepted_results.grads_target_log_prob[0], retained_status),
                    "proposed": self._endpoint(result.proposed_state, result.proposed_results.target_log_prob,
                        result.proposed_results.grads_target_log_prob[0], proposed_status),
                    "log_accept_ratio": result.log_accept_ratio,
                    "is_accepted": result.is_accepted,
                }
            self.graphs[leapfrogs] = probe
        return self.graphs[leapfrogs](state, tf.constant(epsilon, tf.float64), tf.constant(seed, tf.int32))


def initialize_bootstrap_step(*, adapter, geometry, config, progress_callback=None,
                              probe_count=4, max_rounds=20):
    """Shrink unsafe startup proposals, preserving the first failure and seeds.

    Counts are bounded engineering hypotheses, documented in the q20 repair
    plan. Acceptance only nominates startup; no tuning/posterior claim is made.
    Initial/retained invalidity or a runtime exception stops the scope.
    """
    import tensorflow as tf
    from bayesfilter.inference.hmc_bootstrap import build_bootstrap_fixed_mass_adapter
    # Compatibility alias resolves to the same implementation after extraction.
    from bayesfilter.inference.hmc_kernel_tuning import _bootstrap_leapfrog_payload
    from bayesfilter.inference.hmc_preparation import HMCPreparationFailure, _progress_json_value
    if type(probe_count) is not int or probe_count <= 1 or type(max_rounds) is not int or max_rounds <= 0:
        raise ValueError("bootstrap initialization needs multiple probes and a positive round cap")
    latent = build_bootstrap_fixed_mass_adapter(adapter=adapter, mass_artifact=geometry.mass_artifact,
        mass_signature=geometry.mass_artifact_signature, target_scope=config.target_scope or adapter.target_scope)
    if config.use_xla and not latent.value_score_capability().is_accepted_full_chain_xla_diagnostic_authority:
        raise ValueError("bootstrap initialization requires qualified full-chain XLA authority")
    points = tf.broadcast_to(latent.initial_position(), (probe_count, geometry.target_dimension))
    probe = BootstrapMomentumProbe(latent, count=probe_count, use_xla=config.use_xla,
                                   target_status_trace_policy=config.target_status_trace_policy)
    record = {"policy": POLICY, "original_geometry_hash": geometry.artifact_hash,
              "original_epsilon": geometry.initial_step_size, "probe_count": probe_count,
              "max_rounds": max_rounds, "shrink_factor": config.step_repair_factor,
              "acceptance_floor": config.repair_band[0], "use_xla": config.use_xla,
              "target_status_trace_policy": config.target_status_trace_policy,
              "bootstrap_seed": list(config.seed), "artifact_authority": False,
              "rounds": [], "first_invalid_proposal": None}

    def host(value):
        return _progress_json_value(tf.nest.map_structure(lambda x: x.numpy().tolist(), value))

    def emit(stage, details):
        if progress_callback is not None:
            progress_callback("bootstrap_initialization." + stage, details)

    def fail(reason):
        record["status"] = reason
        emit("failed", record)
        raise HMCPreparationFailure(reason, details={"stage": "bootstrap_initialization", **record})

    emit("started", record)
    initial = probe.initial(points)
    record["initial"] = host(initial)
    if not bool(tf.reduce_all(initial["healthy"]).numpy()):
        fail("invalid_initial_target")
    emit("initial_checked", {"initial": record["initial"]})
    epsilon = geometry.initial_step_size
    for index in range(max_rounds):
        if not math.isfinite(epsilon) or epsilon <= 0.:
            fail("bootstrap_initialization_step_underflow")
        pair = _bootstrap_leapfrog_payload(epsilon, geometry.target_trajectory_length,
                                          max_leapfrog_steps=config.max_leapfrog_steps)
        raw = hashlib.sha256(json.dumps([POLICY, list(config.seed), index]).encode()).digest()
        seed = [int.from_bytes(raw[i:i+4], "big") & 0x7fffffff for i in (0, 4)]
        row = {"round": index, "epsilon": epsilon, **pair, "seed": seed}
        emit("probe_started", row)
        began = time.monotonic()
        result = probe(points, epsilon, pair["num_leapfrog_steps"], seed)
        proposed_good = result["proposed"]["healthy"] & tf.math.is_finite(result["log_accept_ratio"])
        retained_good = bool(tf.reduce_all(result["retained"]["healthy"]).numpy())
        all_good = bool(tf.reduce_all(proposed_good).numpy())
        acceptance = float(tf.reduce_mean(tf.exp(tf.minimum(result["log_accept_ratio"], 0.))).numpy())
        row.update(wall_seconds=time.monotonic()-began, retained_valid=retained_good,
            proposal_valid=all_good, mean_acceptance_probability=_progress_json_value(acceptance),
            log_accept_ratio=host(result["log_accept_ratio"]),
            proposal_flags=host(result["proposed"]["flags"]), is_accepted=host(result["is_accepted"]))
        if not all_good and record["first_invalid_proposal"] is None:
            record["first_invalid_proposal"] = {**row, **host(result["proposed"])}
            emit("first_invalid_proposal", record["first_invalid_proposal"])
        record["rounds"].append(row)
        emit("probe_completed", row)
        if not retained_good:
            record["invalid_retained"] = host(result["retained"])
            fail("invalid_retained_target")
        if all_good and acceptance >= config.repair_band[0]:
            record.update(status="startup_nominated", selected_epsilon=epsilon,
                          selected_num_leapfrog_steps=pair["num_leapfrog_steps"],
                          graph_traces={str(k): g.experimental_get_tracing_count() for k,g in probe.graphs.items()})
            revised = replace(geometry, initial_step_size=epsilon,
                initial_num_leapfrog_steps=pair["num_leapfrog_steps"],
                unclamped_num_leapfrog_steps=pair["unclamped_num_leapfrog_steps"],
                formula_report={**geometry.formula_report, "bootstrap_initialization": record})
            emit("completed", {"geometry_artifact_hash": revised.artifact_hash, **record})
            return revised
        epsilon /= config.step_repair_factor
    fail("bootstrap_initialization_exhausted")
