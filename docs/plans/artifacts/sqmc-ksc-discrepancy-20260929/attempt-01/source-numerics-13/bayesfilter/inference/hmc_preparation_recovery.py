"""Bounded discarded-preparation recovery; never candidate admission.

Only explicitly rejected nonfinite proposals can request recovery. Retained
state, target, telemetry and execution failures cannot enter this controller.
TensorFlow owns numerical checks; Python owns attempts and evidence.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
import math
import time
from typing import Any, Mapping


def validate_preparation_max_restarts(value: Any, *, mass_policy: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("preparation_max_restarts must be a nonnegative integer")
    if value and mass_policy != "windowed_adaptive":
        raise ValueError("preparation recovery requires windowed adaptation")
    return value


def _tensor_record(value, *, include_data):
    import tensorflow as tf
    tensor = tf.convert_to_tensor(value)
    data = bytes(tf.io.serialize_tensor(tensor).numpy())
    record = {"dtype": tensor.dtype.name, "shape": tensor.shape.as_list(),
              "sha256": hashlib.sha256(data).hexdigest()}
    if include_data:
        record["tensor_base64"] = base64.b64encode(data).decode("ascii")
    return record


@dataclass(frozen=True)
class DiscardedPreparationAttempt:
    seed: tuple[int, int]
    initial_coordinate_signature: str
    checkpoint: Any
    completed_windows: tuple[Any, ...]
    failed_window: Any
    latent_draws: Any
    trace: Mapping[str, Any]
    failing_step: float
    initial_probe: Any

    @property
    def transition_count(self):
        return sum(w.window.length for w in self.completed_windows) + self.failed_window.length

    def payload(self, *, include_data=False):
        import tensorflow as tf
        tensors = {"checkpoint_theta": self.checkpoint.canonical_theta,
                   "checkpoint_latent": self.checkpoint.active_latent,
                   "checkpoint_center": self.checkpoint.transform.center,
                   "checkpoint_factor": self.checkpoint.transform.factor,
                   "failed_latent_draws": self.latent_draws}
        for i, window in enumerate(self.completed_windows):
            for name in ("adaptation_latent_states", "adaptation_canonical_states",
                         "log_accept_ratio", "is_accepted", "target_log_prob",
                         "step_size_trace", "proposed_step_size_trace", "consumed_step_size_trace"):
                tensors[f"completed_{i}_{name}"] = getattr(window, name)
        def visit(prefix, value):
            if isinstance(value, Mapping):
                for key, child in value.items():
                    visit(prefix + "_" + key, child)
            else:
                tensors[prefix] = value
        visit("trace", self.trace)
        return {
            "schema": "bayesfilter.discarded_preparation_attempt.v1",
            "reason": "rejected_nonfinite_proposal", "seed": self.seed,
            "initial_coordinate_signature": self.initial_coordinate_signature,
            "checkpoint_coordinate_signature": self.checkpoint.transform.signature,
            "checkpoint_metric_signature": self.checkpoint.momentum_metric.signature,
            "checkpoint_covariance_signature": self.checkpoint.transform.covariance_signature,
            "failed_window_index": self.failed_window.index,
            "discarded_transition_count": self.transition_count,
            "failed_window_length": self.failed_window.length,
            "failing_step": self.failing_step,
            "completed_windows": [w.public_payload() for w in self.completed_windows],
            "initial_probe": self.initial_probe.payload(),
            "tensors": {name: _tensor_record(value, include_data=include_data)
                        for name, value in tensors.items()},
            "failed_window_used_for_covariance": False, "discarded_draws_reused": False,
            "used_for_posterior": False,
        }


class RejectedPreparationProposal(ValueError):
    def __init__(self, attempt):
        super().__init__("operational warmup rejected a nonfinite proposal; entire attempt discarded")
        self.attempt = attempt


def rejected_proposal_failure(*, trace, latent_draws, checkpoint, validate_trace,
                             expected_draw_count, step_size_upper_bound,
                             target_status_trace_policy, **context):
    """Check every independent invariant before classifying proposal-only failure."""
    import tensorflow as tf
    accepted = tf.convert_to_tensor(trace["is_accepted"])
    log_accept = tf.convert_to_tensor(trace["log_accept_ratio"])
    finite_proposals = tf.convert_to_tensor(trace["recovery_proposal_finite"])
    score_finite = tf.convert_to_tensor(trace["recovery_retained_score_finite"])
    proposed = tf.convert_to_tensor(trace["recovery_proposed_state"])
    shape = (expected_draw_count,)
    if (accepted.dtype != tf.bool or finite_proposals.dtype != tf.bool or score_finite.dtype != tf.bool
            or accepted.shape != shape or log_accept.shape != shape
            or finite_proposals.shape != shape or score_finite.shape != shape
            or proposed.shape != latent_draws.shape):
        raise ValueError("preparation recovery telemetry is malformed")
    if not bool(tf.reduce_all(score_finite)):
        raise ValueError("preparation recovery found nonfinite retained scores")
    before = tf.concat([tf.convert_to_tensor(checkpoint.active_latent)[None], latent_draws[:-1]], axis=0)
    expected = tf.where(accepted[..., None], proposed, before)
    if not bool(tf.reduce_all(tf.equal(latent_draws, expected))):
        raise ValueError("preparation recovery accepted-state consistency failed")
    bad = ~tf.math.is_finite(log_accept)
    if not bool(tf.reduce_any(bad)):
        if not bool(tf.reduce_all(finite_proposals)):
            raise ValueError("nonfinite preparation proposal has inconsistent acceptance evidence")
        return None
    eligible = tf.math.is_inf(log_accept) & (log_accept < 0) & ~accepted & ~finite_proposals
    if not bool(tf.reduce_all(~bad | eligible)):
        raise ValueError("preparation failure is not a rejected nonfinite proposal")
    if not bool(tf.reduce_all(bad | finite_proposals)):
        raise ValueError("nonfinite preparation proposal has inconsistent acceptance evidence")
    # This copy is for failure classification only. It is never returned as
    # usable trace evidence or passed to covariance/adaptation/posterior code.
    classification_trace = dict(trace, log_accept_ratio=tf.where(bad, tf.zeros_like(log_accept), log_accept))
    health = validate_trace(trace=classification_trace, expected_draw_count=expected_draw_count,
                            step_size_upper_bound=step_size_upper_bound,
                            target_status_trace_policy=target_status_trace_policy)
    if health["divergence_count"] not in (None, 0):
        raise ValueError("preparation recovery cannot reclassify an additional divergence veto")
    failing_step = float(tf.reduce_min(tf.boolean_mask(trace["consumed_step_size"], bad)))
    return RejectedPreparationProposal(DiscardedPreparationAttempt(
        checkpoint=checkpoint, latent_draws=latent_draws, trace=trace,
        failing_step=failing_step, **context))


def _probe_work(payload):
    work = 0
    for attempt in payload.get("attempts", ()):
        count = attempt.get("probe_count", 1)
        results = attempt.get("probe_num_results", 1)
        if type(count) is not int or type(results) is not int or min(count, results) < 1:
            raise ValueError("preparation probe work requires positive integer counts")
        work += count * results
    return work


def _payload_probe_work(initial_probe, windows):
    work = _probe_work(initial_probe)
    for window in windows:
        decision = window.get("metric_decision")
        probe = decision.get("report", {}).get("candidate_reasonable_epsilon") if decision else None
        if probe:
            work += _probe_work(probe)
    return work


def attempt_probe_work(initial_probe, windows):
    work = _probe_work(initial_probe.payload())
    for window in windows:
        if window.metric_decision is not None:
            probe = window.metric_decision.report.get("candidate_reasonable_epsilon")
            if probe:
                work += _probe_work(probe)
    return work


def validate_recovery_payload(summary, *, config, successful_seed, successful_coordinate,
                              successful_probe, successful_windows, validate_probe):
    """Validate the restart chain and accounting when reading saved preparation."""
    from bayesfilter.inference.hmc_warmup import _strict_seed, _seed, _is_sha256_hex, normalize_operational_warmup_config
    from bayesfilter.inference.hmc_tuning import build_windowed_warmup_schedule
    if not isinstance(summary, Mapping) or summary.get("schema") != "bayesfilter.preparation_recovery.v1":
        raise ValueError("preparation recovery summary is invalid")
    root = _strict_seed(summary.get("original_seed"), name="recovery original seed")
    cap = validate_preparation_max_restarts(summary.get("restart_cap"), mass_policy=config.mass_policy)
    attempts, repairs = summary.get("attempts"), summary.get("repairs")
    if (summary.get("status") != "completed" or cap != config.preparation_max_restarts
            or not isinstance(attempts, (tuple, list)) or not isinstance(repairs, (tuple, list))
            or type(summary.get("restart_count")) is not int or summary["restart_count"] != len(attempts)
            or len(attempts) != len(repairs) or len(attempts) > cap):
        raise ValueError("preparation recovery restart count is inconsistent")
    coordinate = summary.get("original_coordinate_signature")
    expected_seed = root
    discarded_count = 0
    expected_probe_work = 0
    for index, (attempt, repair) in enumerate(zip(attempts, repairs)):
        if (attempt.get("schema") != "bayesfilter.discarded_preparation_attempt.v1"
                or attempt.get("reason") != "rejected_nonfinite_proposal"
                or _strict_seed(attempt.get("seed"), name="discarded seed") != expected_seed
                or attempt.get("initial_coordinate_signature") != coordinate
                or attempt.get("failed_window_used_for_covariance") is not False
                or attempt.get("discarded_draws_reused") is not False
                or attempt.get("used_for_posterior") is not False):
            raise ValueError("discarded preparation lineage or exclusion is invalid")
        windows = attempt.get("completed_windows")
        if not isinstance(windows, (tuple, list)) or attempt.get("failed_window_index") != len(windows):
            raise ValueError("discarded preparation window sequence is invalid")
        for window in windows:
            if window.get("coordinate_signature_used") != coordinate:
                raise ValueError("discarded preparation coordinate chain is broken")
            coordinate = window.get("next_coordinate_signature") or coordinate
        if coordinate != attempt.get("checkpoint_coordinate_signature"):
            raise ValueError("preparation checkpoint is not the preceding valid coordinate")
        count = attempt.get("discarded_transition_count")
        length = attempt.get("failed_window_length")
        if (type(count) is not int or type(length) is not int or length < 1
                or count != sum(w["window"]["end"] - w["window"]["start"] for w in windows) + length
                or count > config.warmup_steps):
            raise ValueError("discarded preparation work count is invalid")
        discarded_count += count
        expected_probe_work += _payload_probe_work(attempt.get("initial_probe", {}), windows)
        tensors = attempt.get("tensors")
        if (not isinstance(tensors, Mapping) or not {"checkpoint_theta", "checkpoint_latent", "failed_latent_draws",
                "trace_log_accept_ratio", "trace_recovery_proposed_state"} <= set(tensors)
                or any(not _is_sha256_hex(record.get("sha256")) for record in tensors.values())):
            raise ValueError("discarded preparation tensor evidence is missing")
        failed_step, ceiling = attempt.get("failing_step"), repair.get("step_ceiling")
        if (type(failed_step) not in (int, float) or not math.isfinite(failed_step) or failed_step <= 0
                or repair.get("failed_step") != failed_step or ceiling != .5 * failed_step
                or repair.get("checkpoint_coordinate_signature") != coordinate
                or repair.get("discarded_attempt_index") != index
                or repair.get("covariance_statistics_reset") is not True
                or repair.get("dual_averaging_reset") is not True):
            raise ValueError("preparation restart contraction or reset evidence is invalid")
        expected_seed = _seed(root, index + 1, lane=81)
        if (_strict_seed(repair.get("restart_seed"), name="restart seed") != expected_seed
                or _strict_seed(repair.get("probe_seed"), name="restart probe seed") != _seed(root, index + 1, lane=82)):
            raise ValueError("preparation restart streams are invalid")
        selected = validate_probe(repair.get("probe"), name="contracted preparation probe")
        expected_probe_work += _probe_work(repair["probe"])
        if selected > ceiling or any(a["step_size"] > ceiling for a in repair["probe"]["attempts"]):
            raise ValueError("preparation probe exceeded the contracted step ceiling")
    if (tuple(successful_seed) != expected_seed or tuple(summary.get("successful_seed", ())) != expected_seed
            or successful_coordinate != coordinate or summary.get("successful_initial_coordinate_signature") != coordinate
            or summary.get("discarded_transition_count") != discarded_count
            or summary.get("successful_transition_count") != config.warmup_steps):
        raise ValueError("successful preparation does not continue the restart lineage")
    charged = summary.get("charged_probe_transition_bound")
    expected_probe_work += _payload_probe_work(successful_probe, successful_windows)
    if type(charged) is not int or charged != expected_probe_work:
        raise ValueError("preparation recovery probe work differs from its saved probes")
    limit = summary.get("transition_work_limit")
    window_count = len(build_windowed_warmup_schedule(normalize_operational_warmup_config(config)))
    probe_cap = 20 * 4 * max(2, config.metric_probe_num_results)
    expected_limit = (cap + 1) * (config.warmup_steps + window_count * probe_cap) + cap * probe_cap
    if (type(charged) is not int or charged < 0 or type(limit) is not int or limit <= 0
            or limit != expected_limit
            or discarded_count + config.warmup_steps + charged > limit):
        raise ValueError("preparation recovery work budget is invalid")


def run_with_preparation_recovery(attempt_fn, arguments):
    from bayesfilter.inference import hmc_warmup as warmup
    from bayesfilter.inference.hmc_tuning import build_windowed_warmup_schedule
    import tensorflow as tf

    options = dict(arguments)
    callback = options.pop("recovery_callback", None)
    config = options["config"]
    cap = config.preparation_max_restarts
    if options.get("engineering_probe_config") is not None or options.get("_g2_seed_use_registry") is not None:
        raise ValueError("preparation recovery is not qualified for the G2 engineering route")
    root_seed = warmup._strict_seed(options["seed"], name="seed")
    initial_signature = options["initial_transform"].signature
    discarded = []
    repairs = []
    transition_count = 0
    probe_work = 0
    window_count = len(build_windowed_warmup_schedule(warmup.normalize_operational_warmup_config(config)))
    probe_cap = 20 * 4 * max(2, config.metric_probe_num_results)
    work_cap = (cap + 1) * (config.warmup_steps + window_count * probe_cap) + cap * probe_cap
    started = time.perf_counter()

    def emit(event, payload):
        if callback is not None:
            callback(event, payload)

    for index in range(cap + 1):
        emit("attempt_start", {"attempt_index": index, "restart_cap": cap,
             "discarded_transitions": transition_count, "charged_probe_transition_bound": probe_work,
             "transition_work_limit": work_cap, "seed": options["seed"],
             "coordinate_signature": options["initial_transform"].signature})
        try:
            result = attempt_fn(**options)
        except RejectedPreparationProposal as exc:
            failed = exc.attempt
            discarded.append(failed)
            transition_count += failed.transition_count
            probe_work += attempt_probe_work(failed.initial_probe, failed.completed_windows)
            emit("attempt_discarded", {"attempt_index": index, **failed.payload(include_data=True)})
            if index == cap:
                exc.discarded_attempts = tuple(discarded)
                exc.recovery_summary = {"status": "restart_cap_exhausted", "restart_cap": cap,
                    "discarded_transition_count": transition_count, "charged_probe_transition_bound": probe_work,
                    "transition_work_limit": work_cap, "attempts": [a.payload() for a in discarded]}
                emit("exhausted", exc.recovery_summary)
                raise
            checkpoint = failed.checkpoint
            affine = warmup._AffineWarmupAdapter(base_adapter=options["adapter"],
                transform=checkpoint.transform, target_scope=options["target_scope"])
            health = warmup._evaluate_retained_target_health(adapter=affine,
                samples=tf.convert_to_tensor(checkpoint.active_latent)[None],
                target_status_trace_policy=options["target_status_trace_policy"])
            if health["shared_invalidity_reasons"] or health["candidate_data_invalidity_reasons"]:
                raise ValueError("preparation restart checkpoint target is invalid") from exc
            if not warmup._all_close(checkpoint.transform.latent_to_theta(checkpoint.active_latent),
                                     checkpoint.canonical_theta, rtol=1e-12, atol=1e-12):
                raise ValueError("preparation restart checkpoint coordinate lineage is invalid") from exc
            ceiling = failed.failing_step * .5
            if not math.isfinite(ceiling) or ceiling <= 0:
                raise ValueError("preparation contracted step is not representable") from exc
            if transition_count + probe_work + probe_cap + config.warmup_steps > work_cap:
                raise ValueError("preparation cumulative transition budget exhausted") from exc
            restart_seed = warmup._seed(root_seed, index + 1, lane=81)
            probe_seed = warmup._seed(root_seed, index + 1, lane=82)
            probe = warmup.find_reasonable_epsilon(adapter=affine, current_state=checkpoint.active_latent,
                initial_step_size=ceiling, seed=probe_seed, num_leapfrog_steps=options["trajectory_policy"].num_leapfrog_steps,
                momentum_probe_count=4, probe_num_results=max(2, config.metric_probe_num_results),
                target_status_trace_policy=options["target_status_trace_policy"],
                jit_compile=options["jit_compile"], preparation_step_ceiling=ceiling)
            probe_work += _probe_work(probe.payload())
            repair = {"discarded_attempt_index": index, "checkpoint_coordinate_signature": checkpoint.transform.signature,
                "failed_step": failed.failing_step, "step_ceiling": ceiling,
                "restart_seed": restart_seed, "probe_seed": probe_seed, "probe": probe.payload(),
                "covariance_statistics_reset": True, "dual_averaging_reset": True}
            repairs.append(repair)
            emit("restart_probe", repair)
            if not probe.passed or probe.selected_step_size is None:
                raise ValueError("preparation contracted probe was inconclusive") from exc
            options.update(initial_transform=checkpoint.transform,
                initial_canonical_theta=checkpoint.canonical_theta,
                initial_position_covariance_estimate_signature=checkpoint.transform.covariance_signature,
                initial_step_size=probe.selected_step_size, initial_step_size_upper_bound=probe.selected_step_size,
                initial_step_qualification_source="bayesfilter_contracted_preparation_restart_v1",
                seed=restart_seed)
            continue
        if isinstance(result, warmup.OperationalWindowedWarmupCloseout):
            summary = {"status": "partial_timeout", "discarded_transition_count": transition_count,
                "restart_count": len(discarded), "restart_cap": cap,
                "charged_probe_transition_bound": probe_work, "transition_work_limit": work_cap,
                "completed_transition_count": result.completed_transition_count,
                "attempts": [a.payload() for a in discarded], "repairs": repairs}
            emit("timeout", summary)
            return replace(result, boundary_payload={**result.boundary_payload,
                "preparation_recovery": summary}, discarded_attempts=tuple(discarded))
        probe_work += attempt_probe_work(result.reasonable_epsilon, result.windows)
        summary = {"schema": "bayesfilter.preparation_recovery.v1", "status": "completed",
            "restart_cap": cap, "restart_count": len(discarded), "original_seed": root_seed,
            "original_coordinate_signature": initial_signature,
            "successful_seed": result.seed_root, "successful_initial_coordinate_signature": result.initial_coordinate_signature,
            "discarded_transition_count": transition_count, "successful_transition_count": result.config.warmup_steps,
            "charged_probe_transition_bound": probe_work, "transition_work_limit": work_cap,
            "attempts": [a.payload() for a in discarded], "repairs": repairs,
            "elapsed_seconds": time.perf_counter() - started}
        if transition_count + result.config.warmup_steps + probe_work > work_cap:
            raise ValueError("preparation cumulative work accounting exceeded its derived bound")
        emit("completed", summary)
        return replace(result, preparation_recovery=summary, discarded_attempts=tuple(discarded),
                       elapsed_s=time.perf_counter() - started)
    raise AssertionError("unreachable preparation recovery state")
