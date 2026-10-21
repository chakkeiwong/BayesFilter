"""Bounded discarded proposals for the fully refreshed MAMS transition.

The paper's autocorrelation length update is a *proposal heuristic*. Its
average-preserving count law and preconditioning differ from BayesFilter's
uniform-integer law and repository-issued prepared coordinates. This module
uses the actual frozen transition, measures fixed-policy windows separately,
and leaves all numerical qualification to the existing public tuner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

from .hmc_trajectory_duration import compile_mean_duration


class _PilotPolicyFailure(ValueError):
    """Numerical proposal failed; the shared state/identity invariants held."""


class MAMSPilotBudgetExceeded(RuntimeError):
    """The declared pilot allocation ended without a complete proposal set."""


class MAMSPilotUnavailable(ValueError):
    """No policy survived the declared pilot domain and numerical checks."""


def _positive(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class MAMSPilotConfig:
    """Every numerical choice is explicit; no transferred epsilon/L default."""

    initial_epsilon: float
    initial_mean_duration: float
    epsilon_domain: tuple[float, float]
    max_steps_cap: int
    epsilon_adaptation_windows: int
    probe_transitions: int
    fixed_settling_transitions: int
    fixed_window_transitions: int
    fixed_window_count: int
    max_autocorrelation_lag: int
    target_accept_probability: float
    log_epsilon_gain: float
    source_length_coefficient: float
    window_agreement_factor: float
    duration_factors: tuple[float, ...]
    epsilon_factors: tuple[float, ...]
    max_requested_duration: float
    max_candidates: int
    max_pilot_gradient_work: int
    max_pilot_wall_seconds: float
    frozen_coordinate_scale: tuple[float, ...]
    frozen_projection_directions: tuple[tuple[float, ...], ...]
    seed: tuple[int, int]

    def __post_init__(self) -> None:
        for name in ("initial_epsilon", "initial_mean_duration", "log_epsilon_gain",
                     "source_length_coefficient", "window_agreement_factor",
                     "max_requested_duration", "max_pilot_wall_seconds"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.window_agreement_factor < 1:
            raise ValueError("window_agreement_factor must be at least one")
        domain = tuple(_positive(v, "epsilon_domain") for v in self.epsilon_domain)
        if len(domain) != 2 or domain[0] >= domain[1]:
            raise ValueError("epsilon_domain must be an increasing pair")
        object.__setattr__(self, "epsilon_domain", domain)
        if not domain[0] <= self.initial_epsilon <= domain[1]:
            raise ValueError("initial epsilon is outside the declared domain")
        for name in ("max_steps_cap", "epsilon_adaptation_windows", "probe_transitions",
                     "fixed_settling_transitions", "fixed_window_transitions", "fixed_window_count",
                     "max_autocorrelation_lag", "max_candidates", "max_pilot_gradient_work"):
            value = getattr(self, name)
            minimum = 0 if name in {"epsilon_adaptation_windows", "fixed_settling_transitions"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.max_steps_cap >= 2**31 - 1:
            raise ValueError("max_steps_cap exceeds frozen transition support")
        if self.fixed_window_count < 2:
            raise ValueError("two separate fixed-policy windows are required")
        if self.fixed_window_transitions < 4 * self.max_autocorrelation_lag:
            raise ValueError("fixed windows need at least four draws per declared lag")
        if self.max_autocorrelation_lag < 2 or self.max_autocorrelation_lag % 2:
            raise ValueError("max_autocorrelation_lag must be an even number >= 2")
        if not 0 < self.target_accept_probability < 1:
            raise ValueError("adaptation target must lie strictly inside (0,1)")
        for name in ("duration_factors", "epsilon_factors", "frozen_coordinate_scale"):
            values = tuple(_positive(v, name) for v in getattr(self, name))
            if not values or (name != "frozen_coordinate_scale" and len(set(values)) != len(values)):
                raise ValueError(f"{name} must be nonempty with distinct proposal factors")
            object.__setattr__(self, name, values)
        directions = tuple(tuple(float(v) for v in row) for row in self.frozen_projection_directions)
        if any(len(row) != len(self.frozen_coordinate_scale)
               or any(not math.isfinite(v) for v in row)
               or not any(v != 0 for v in row) for row in directions):
            raise ValueError("frozen projection directions must match prepared dimension and be finite/nonzero")
        object.__setattr__(self, "frozen_projection_directions", directions)
        if 1. not in self.duration_factors or 1. not in self.epsilon_factors:
            raise ValueError("both factor neighborhoods must include one")
        if len(self.seed) != 2 or any(type(v) is not int or not 0 <= v < 2**31 for v in self.seed):
            raise ValueError("seed must have two nonnegative int32 components")
        if self.initial_mean_duration > self.max_requested_duration:
            raise ValueError("initial duration is outside the declared duration domain")
        if any(not math.isfinite(self.initial_mean_duration * f) for f in self.duration_factors):
            raise ValueError("duration factor overflows the finite proposal domain")


def _autocorrelation_summary(samples: Any, *, frozen_scale: tuple[float, ...],
                             max_lag: int, projections=(), use_xla=True) -> dict[str, Any]:
    from .hmc_pilot_diagnostics import summarize_pilot_window
    return summarize_pilot_window(samples, frozen_scale=frozen_scale,
        max_lag=max_lag, projections=projections, use_xla=use_xla)


def _bracketed_epsilon(epsilon: float, acceptance: float, target: float,
                       lower_accepting: float | None, upper_rejecting: float | None,
                       log_gain: float, domain: tuple[float, float]
                       ) -> tuple[float, float | None, float | None, float | None, bool]:
    """A bounded proposal bracket, not a monotonic-acceptance assertion."""
    if acceptance >= target:
        lower_accepting = max(epsilon, lower_accepting or epsilon)
    else:
        upper_rejecting = min(epsilon, upper_rejecting or epsilon)
    if (lower_accepting is not None and upper_rejecting is not None
            and lower_accepting < upper_rejecting):
        proposed = math.exp((math.log(lower_accepting) + math.log(upper_rejecting))/2)
    else:
        log_proposed = math.log(epsilon) + log_gain * (1 if acceptance >= target else -1)
        # Bound in log space so a wide but explicit gain does not overflow.
        bounded = min(domain[1], max(domain[0], math.exp(
            min(math.log(domain[1]), max(math.log(domain[0]), log_proposed)))))
        try:
            proposed = math.exp(log_proposed)
        except OverflowError:
            proposed = None  # The raw proposal is outside finite representation.
        return bounded, lower_accepting, upper_rejecting, proposed, (
            log_proposed < math.log(domain[0]) or log_proposed > math.log(domain[1]))
    if not math.isfinite(proposed):
        raise ValueError("epsilon bracket update overflow")
    bounded = min(domain[1], max(domain[0], proposed))
    return bounded, lower_accepting, upper_rejecting, proposed, bounded != proposed


def propose_mams_trajectory_candidates(binding: Any, config: MAMSPilotConfig,
                                       *, output_dir: str | Path | None = None,
                                       resume_from: str | Path | None = None) -> dict[str, Any]:
    """Discarded proposals with optional recovery into a new output directory.

    Completed numerical windows replay from checksummed tensor checkpoints.
    The identical policy and source identity are required; failed attempts
    remain charged. The returned result is never a tuning artifact.
    """
    from contextlib import ExitStack
    existed = output_dir is not None and Path(output_dir).exists()
    with ExitStack() as stack:
        try:
            return _propose(binding, config, output_dir=output_dir,
                            resume_from=resume_from, stack=stack)
        except Exception as exc:
            # Preserve a terminal explanation for diagnostic/serialization
            # failures as well as failures inside the numerical call.
            progress = Path(output_dir)/"pilot-progress.json" if output_dir else None
            if not existed and progress is not None and progress.is_file():
                record = json.loads(progress.read_text())
                if record.get("status") == "running":
                    record.update(status="failed_processing", reason=str(exc),
                                  error_type=type(exc).__name__)
                    temporary = progress.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
                    temporary.replace(progress)
            raise


def _propose(binding, config, *, output_dir, resume_from, stack):
    """Return frozen-policy hypotheses; fresh public-tuner evidence is required.

    The proposer performs no acceptance qualification. Each duration has its
    own bounded epsilon adaptation and two continuous, separately diagnosed
    fixed-policy windows. No changing-policy sequence enters autocorrelation.
    """
    import tensorflow as tf
    from .hmc_candidate_set_execution import HMCCandidateExecutionBinding
    from .hmc_candidate_set_tuning import HMCTuningCandidateRecord

    if not isinstance(binding, HMCCandidateExecutionBinding) or not isinstance(config, MAMSPilotConfig):
        raise TypeError("repository binding and typed MAMS pilot config required")
    binding.validate()
    profile = binding.config.transition_profile
    if profile is None or (profile.family, profile.trajectory) != ("mams", "uniform_duration"):
        raise ValueError("MAMS pilot requires the prepared full-refresh uniform-duration profile")
    if len(config.frozen_coordinate_scale) != int(binding.initial_active_state.shape[-1]):
        raise ValueError("frozen scale dimension differs from prepared coordinates")
    if config.epsilon_domain != binding.scope.epsilon_domain:
        raise ValueError("pilot epsilon domain differs from the bound tuning scope")
    compile_mean_duration(profile=profile, epsilon=config.initial_epsilon,
        requested_mean_duration=config.initial_mean_duration,
        max_steps_cap=config.max_steps_cap)

    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("hmc_mams_trajectory_proposals.py", "hmc_trajectory_duration.py",
                                  "hmc_pilot_diagnostics.py")}
    root = hashlib.sha256(json.dumps({"binding": binding.binding_hash,
        "config": asdict(config), "sources": source_hashes,
        "purpose": "discarded_mams_pilot_v1"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    directory = Path(output_dir) if output_dir is not None else None
    previous = None
    if resume_from is not None:
        if directory is None:
            raise ValueError("pilot recovery requires a fresh output directory")
        previous = json.loads((Path(resume_from) / "pilot-progress.json").read_text())
        if previous.get("purpose_digest") != root:
            raise ValueError("pilot recovery source/configuration/binding identity mismatch")
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    store = None
    if directory is not None:
        from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
        if previous is not None:
            shutil.copytree(Path(resume_from) / "windows", directory / "windows")
        store = stack.enter_context(DurableTensorCheckpoint(directory / "windows",
            {"purpose_digest": root, "role": "discarded_mams_pilot_only"}))
    started = time.monotonic()
    prior_elapsed = float(previous["elapsed_seconds"]) if previous else 0.
    consumed_work = int(previous["reserved_gradient_work"]) if previous else 0
    actual_work = int(previous["actual_gradient_work"]) if previous else 0
    candidate_ordinal = 0
    seed_log: list[dict[str, Any]] = []
    duration_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = list(previous["attempts"]) if previous else []
    active_duration: dict[str, Any] | None = None
    pairs: set[tuple[int, float]] = set()

    def snapshot(status: str, *, reason: str | None = None) -> None:
        if directory is None:
            return
        progress = {"schema": "bayesfilter.discarded_mams_pilot_progress.v1",
            "status": status, "reason": reason, "binding_hash": binding.binding_hash,
            "purpose_digest": root, "config": asdict(config), "source_hashes": source_hashes,
            "duration_rows": duration_rows, "active_duration": active_duration,
            "candidate_rows": candidate_rows, "omitted_candidates": omitted,
            "attempts": attempts, "seeds": seed_log,
            "reserved_gradient_work": consumed_work, "actual_gradient_work": actual_work,
            "elapsed_seconds": prior_elapsed + time.monotonic() - started,
            "resume_from": str(Path(resume_from).resolve()) if resume_from else None,
            "tuning_artifact_authority": False, "posterior_convergence_authority": False}
        temporary = directory / "pilot-progress.json.tmp"
        temporary.write_text(json.dumps(progress, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(directory / "pilot-progress.json")

    snapshot("running")

    def seed_for(label: str, duration_index: int, call_index: int) -> tuple[int, int]:
        raw = hashlib.sha256(json.dumps((root, label, duration_index, call_index)).encode()).digest()
        result = tuple(int.from_bytes(raw[i:i+4], "big") & 0x7fffffff for i in (0, 4))
        seed_log.append({"stage": label, "duration_index": duration_index,
                         "call_index": call_index, "seed": result})
        return result

    def run_window(*, ell: float, epsilon: float, transitions: int,
                   state: Any, label: str, duration_index: int, call_index: int):
        nonlocal consumed_work, actual_work, candidate_ordinal
        from bayesfilter.runtime.execution_budget import execution_budget_available
        compiled = compile_mean_duration(profile=profile, epsilon=epsilon,
            requested_mean_duration=ell, max_steps_cap=config.max_steps_cap)
        required = (transitions * int(state.shape[0])
                    * compiled.frozen.conservative_gradient_work_per_transition)
        key = f"{label}-{duration_index}-{call_index}"
        replayed = store is not None and store.contains(key)
        if not replayed and consumed_work + required > config.max_pilot_gradient_work:
            snapshot("gradient_budget_exhausted", reason="next pilot window exceeds the declared work cap")
            raise MAMSPilotBudgetExceeded("declared MAMS pilot gradient-work budget exhausted")
        if not execution_budget_available(
                elapsed_seconds=prior_elapsed + time.monotonic() - started,
                wall_time_limit_seconds=config.max_pilot_wall_seconds):
            snapshot("wall_budget_exhausted", reason="next pilot window starts beyond the declared wall cap")
            raise MAMSPilotBudgetExceeded("declared MAMS pilot wall budget exhausted")
        candidate_ordinal += 1
        candidate = HMCTuningCandidateRecord.create(binding.scope,
            leapfrog_steps=compiled.frozen.max_steps, epsilon=epsilon,
            creation_ordinal=candidate_ordinal)
        seed = seed_for(label, duration_index, call_index)
        consumed_work += 0 if replayed else required
        attempt = {"stage": label, "duration_index": duration_index,
            "call_index": call_index, "requested_mean_duration": ell,
            "epsilon": epsilon, "max_steps": compiled.frozen.max_steps,
            "seed": seed, "transitions_per_chain": transitions,
            "reserved_gradient_work": 0 if replayed else required,
            "replayed": replayed, "status": "running"}
        attempts.append(attempt)
        snapshot("running")
        try:
            if store is None:
                result = binding._run(candidate, state, transitions, seed)
            else:
                from types import SimpleNamespace
                def compute():
                    computed = binding._run(candidate, state, transitions, seed)
                    return dict(samples=computed.samples, trace=computed.trace,
                                metadata=computed.metadata)
                result = SimpleNamespace(**store.run(key,
                    dict(candidate=candidate.payload(), seed=seed, transitions=transitions,
                         initial_state_sha256=store.tensor_hash(state)), compute))
        except Exception as exc:
            attempt.update(status="failed_call", error_type=type(exc).__name__, error=str(exc))
            snapshot("failed_call", reason=str(exc))
            raise
        used = (int(result.metadata["value_score_row_evaluations"])
                + int(result.metadata["target_status_row_evaluations"]))
        if replayed:
            # A crash can occur after durable numerical commit but before
            # progress accounting. Recover the actual work once, retaining
            # the original reserve and separating replay from new work.
            for prior in reversed(attempts[:-1]):
                if (not prior["replayed"] and prior["stage"] == label
                        and prior["duration_index"] == duration_index
                        and prior["call_index"] == call_index):
                    if prior["status"] == "running" and "actual_gradient_work" not in prior:
                        prior.update(status="recovered_committed_call", actual_gradient_work=used)
                        actual_work += used
                    break
        actual_work += 0 if replayed else used
        attempt.update(status="completed", actual_gradient_work=0 if replayed else used,
                       wall_seconds=result.metadata["sample_chain_call_s"])
        if used > required:
            attempt["status"] = "work_accounting_invalid"
            snapshot("work_accounting_invalid", reason="actual numerical work exceeded the conservative reserve")
            raise ValueError("MAMS pilot work exceeded its conservative reserve")
        failures = binding.health_failures(state, result.samples, result.trace)
        if failures:
            attempt.update(status="numerical_veto", health_failures=failures)
            snapshot("numerical_veto", reason=", ".join(failures))
            proposal_failures = {"nonfinite_proposal", "nonfinite_proposal_displacement",
                "nonfinite_log_accept_ratio", "nonfinite_proposed_target_log_prob",
                "nonfinite_final_momentum", "nonfinite_log_acceptance_correction",
                "nonfinite_target_score", "native_divergence_positive",
                "proposed_target_status_telemetry_failed"}
            error = _PilotPolicyFailure if set(failures) <= proposal_failures else ValueError
            raise error("discarded MAMS pilot numerical health failed: " + ", ".join(failures))
        acceptance = tf.exp(tf.minimum(result.trace["log_accept_ratio"], 0.))
        if not bool(tf.reduce_all(tf.math.is_finite(acceptance))):
            attempt["status"] = "nonfinite_acceptance"
            snapshot("numerical_veto", reason="nonfinite acceptance")
            raise ValueError("discarded MAMS pilot has nonfinite acceptance")
        attempt["acceptance"] = float(tf.reduce_mean(acceptance))
        snapshot("running")
        return result, compiled, attempt["acceptance"]

    def feasible_epsilon_interval(ell: float) -> tuple[float, float]:
        low = max(config.epsilon_domain[0], 2*ell/(config.max_steps_cap + 1))
        high = min(config.epsilon_domain[1], ell)
        if not low <= high:
            raise ValueError("duration has no epsilon inside declared count/domain support")
        return low, high

    def evaluate_duration(ell: float, index: int, epsilon_seed: float) -> dict[str, Any]:
        nonlocal active_duration
        low, high = feasible_epsilon_interval(ell)
        epsilon = min(high, max(low, epsilon_seed))
        row = {"requested_mean_duration": ell, "epsilon_seed": epsilon_seed,
               "adjusted_epsilon_seed": epsilon, "epsilon_seed_adjusted": epsilon != epsilon_seed,
               "epsilon_feasible_interval": (low, high), "adaptation": [], "fixed_windows": []}
        active_duration = row
        snapshot("running")
        state = tf.identity(binding.initial_active_state)
        lower_accepting = None
        upper_rejecting = None
        for iteration in range(config.epsilon_adaptation_windows):
            try:
                result, compiled, acceptance = run_window(ell=ell, epsilon=epsilon,
                    transitions=config.probe_transitions, state=state,
                    label="epsilon_adaptation", duration_index=index, call_index=iteration)
                row["adaptation"].append({"epsilon": epsilon, "max_steps": compiled.frozen.max_steps,
                    "acceptance": acceptance, "actual_work": attempts[-1]["actual_gradient_work"],
                    "next_state_discarded": True})
                state = result.samples[-1]
            except _PilotPolicyFailure as exc:
                # A failed proposal nominates smaller resolution. Zero here is
                # a search direction only and is never acceptance evidence.
                acceptance = 0.
                row["adaptation"].append({"epsilon": epsilon,
                    "acceptance": None, "proposal_failure": str(exc),
                    "repair_direction": "smaller_epsilon", "state_reused": True,
                    "actual_work": attempts[-1].get("actual_gradient_work")})
            bounded, lower_accepting, upper_rejecting, update, hit = _bracketed_epsilon(
                epsilon, acceptance, config.target_accept_probability,
                lower_accepting, upper_rejecting, config.log_epsilon_gain, (low, high))
            row["adaptation"][-1].update(unbounded_next_epsilon=update,
                bounded_next_epsilon=bounded, domain_or_count_boundary_hit=hit,
                lower_accepting_epsilon=lower_accepting,
                upper_rejecting_epsilon=upper_rejecting,
                bracket_is_heuristic_across_count_changes=True)
            epsilon = bounded
            snapshot("running")
        if config.fixed_settling_transitions:
            result, compiled, acceptance = run_window(ell=ell, epsilon=epsilon,
                transitions=config.fixed_settling_transitions, state=state,
                label="fixed_policy_settling", duration_index=index, call_index=0)
            row["fixed_settling"] = {"epsilon": epsilon,
                "max_steps": compiled.frozen.max_steps, "acceptance": acceptance,
                "actual_work": (result.metadata["value_score_row_evaluations"]
                                + result.metadata["target_status_row_evaluations"]),
                "samples_discarded": int(result.samples.shape[0]),
                "excluded_from_autocorrelation": True}
            state = result.samples[-1]
            snapshot("running")
        for window in range(config.fixed_window_count):
            result, compiled, acceptance = run_window(ell=ell, epsilon=epsilon,
                transitions=config.fixed_window_transitions, state=state,
                label="fixed_policy_window", duration_index=index, call_index=window)
            diagnostic = _autocorrelation_summary(result.samples,
                frozen_scale=config.frozen_coordinate_scale,
                max_lag=config.max_autocorrelation_lag,
                projections=config.frozen_projection_directions,
                use_xla=binding.config.use_xla)
            from .hmc_pilot_diagnostics import summarize_pilot_displacement
            diagnostic.update(summarize_pilot_displacement(
                state, result.samples, result.trace["proposed_state"],
                frozen_scale=config.frozen_coordinate_scale, use_xla=binding.config.use_xla))
            used = (result.metadata["value_score_row_evaluations"]
                    + result.metadata["target_status_row_evaluations"])
            row["fixed_windows"].append({"epsilon": epsilon,
                "max_steps": compiled.frozen.max_steps,
                "acceptance": acceptance, "actual_work": used,
                "diagnostics": diagnostic, "samples_discarded": int(result.samples.shape[0])})
            state = result.samples[-1]
            snapshot("running")
        estimates = [w["diagnostics"]["source_harmonic_coordinate_iact"]
                     for w in row["fixed_windows"]]
        compatible = (all(v is not None and math.isfinite(v) for v in estimates)
                      and max(estimates)/min(estimates) <= config.window_agreement_factor)
        row.update(epsilon=epsilon, compiled_duration=compiled.payload(),
                   fixed_policy_estimates_compatible=compatible)
        active_duration = None
        return row

    def try_duration(ell, index, seed):
        nonlocal active_duration
        try:
            return evaluate_duration(ell, index, seed)
        except _PilotPolicyFailure as exc:
            omitted.append({"requested_mean_duration": ell,
                "reason": "numerically_failed_fixed_policy_window",
                "detail": str(exc), "partial_duration": active_duration})
            active_duration = None
            snapshot("running")
            return None

    initial = try_duration(config.initial_mean_duration, 0, config.initial_epsilon)
    if initial is not None:
        duration_rows.append(initial)
    snapshot("running")
    first_estimates = [w["diagnostics"]["source_harmonic_coordinate_iact"]
                       for w in initial["fixed_windows"]] if initial else []
    usable = initial is not None and initial["fixed_policy_estimates_compatible"]
    source_duration = (config.source_length_coefficient * config.initial_mean_duration
                       * math.fsum(first_estimates)/len(first_estimates) if usable else None)
    worst_estimates = [w["diagnostics"]["worst_observable_iact"]
                       for w in initial["fixed_windows"]] if initial else []
    slow_usable = (bool(worst_estimates)
                   and all(all(w["diagnostics"]["usable_observable_iact"]) for w in initial["fixed_windows"])
                   and all(v is not None and math.isfinite(v) and v > 0 for v in worst_estimates)
                   and max(worst_estimates)/min(worst_estimates) <= config.window_agreement_factor)
    slow_duration = (config.source_length_coefficient * config.initial_mean_duration
                     * math.fsum(worst_estimates)/len(worst_estimates) if slow_usable else None)
    requested_durations = [(config.initial_mean_duration, "initial")]
    # The initial neighborhood is always retained; a noisy source estimate
    # cannot erase broad predeclared exploration.
    requested_durations.extend((config.initial_mean_duration * f, "declared_neighborhood")
                               for f in config.duration_factors if f != 1.)
    if source_duration is not None:
        requested_durations.append((source_duration, "paper_harmonic_coordinate_rule"))
    if slow_duration is not None:
        requested_durations.append((slow_duration, "local_slow_observable_extension"))
    seen_duration: set[float] = {config.initial_mean_duration}
    for duration_index, (ell, origin) in enumerate(requested_durations[1:], start=1):
        if ell in seen_duration:
            omitted.append({"requested_mean_duration": ell, "origin": origin,
                            "reason": "duplicate_duration_hypothesis"})
            snapshot("running")
            continue
        seen_duration.add(ell)
        if not math.isfinite(ell) or not 0 < ell <= config.max_requested_duration:
            omitted.append({"requested_mean_duration": ell, "origin": origin,
                            "reason": "outside_declared_duration_domain"})
            snapshot("running")
            continue
        try:
            row = try_duration(ell, duration_index, initial["epsilon"] if initial else config.initial_epsilon)
        except ValueError as exc:
            # Unsupported duration is omitted; shared invariant failures escape.
            if "duration has no epsilon" not in str(exc):
                raise
            omitted.append({"requested_mean_duration": ell, "origin": origin,
                            "reason": "infeasible_epsilon_count_domain"})
            snapshot("running")
            continue
        if row is None:
            continue
        row["proposal_origin"] = origin
        duration_rows.append(row)
        snapshot("running")

    for row in duration_rows:
        ell = row["requested_mean_duration"]
        for factor in config.epsilon_factors:
            epsilon = row["epsilon"] * factor
            if not config.epsilon_domain[0] <= epsilon <= config.epsilon_domain[1]:
                omitted.append({"requested_mean_duration": ell, "epsilon": epsilon,
                                "reason": "outside_declared_epsilon_domain"})
                snapshot("running")
                continue
            try:
                compiled = compile_mean_duration(profile=profile, epsilon=epsilon,
                    requested_mean_duration=ell, max_steps_cap=config.max_steps_cap)
            except ValueError as exc:
                omitted.append({"requested_mean_duration": ell, "epsilon": epsilon,
                                "reason": "infeasible_epsilon_count_pair", "detail": str(exc)})
                snapshot("running")
                continue
            pair = (compiled.frozen.max_steps, epsilon)
            if pair in pairs:
                omitted.append({"requested_mean_duration": ell, "epsilon": epsilon,
                                "reason": "duplicate_executed_policy"})
                snapshot("running")
                continue
            if len(pairs) >= config.max_candidates:
                omitted.append({"requested_mean_duration": ell, "epsilon": epsilon,
                                "reason": "declared_candidate_cap"})
                snapshot("running")
                continue
            pairs.add(pair)
            candidate_rows.append({"compiled_duration": compiled.payload(),
                                   "fresh_measurement_and_verification_required": True})
            snapshot("running")
    if not pairs:
        snapshot("no_candidate", reason="pilot yielded no candidate inside declared support")
        raise MAMSPilotUnavailable("MAMS pilot yielded no candidate inside declared support")
    ordered = sorted(pairs)
    for name, expected in source_hashes.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            snapshot("source_changed", reason=name)
            raise ValueError("MAMS pilot source changed during execution")
    result = {
        "schema": "bayesfilter.discarded_mams_trajectory_proposals.v1",
        "binding_hash": binding.binding_hash, "config": asdict(config),
        "profile": profile.payload(), "purpose_digest": root,
        "proposal_source_sha256": source_hashes["hmc_mams_trajectory_proposals.py"],
        "duration_compiler_sha256": source_hashes["hmc_trajectory_duration.py"],
        "pilot_diagnostics_sha256": source_hashes["hmc_pilot_diagnostics.py"],
        "source_rule": "mams_paper_section_6_eq_12_to_14_hypothesis_with_local_safeguards",
        "executed_duration_law": "uniform_integer_count_1_through_K_not_paper_average_preserving_law",
        "source_length_update_usable": usable,
        "source_length_hypothesis": source_duration,
        "slow_observable_length_update_usable": slow_usable,
        "slow_observable_length_hypothesis": slow_duration,
        "duration_hypotheses": requested_durations,
        "duration_rows": duration_rows, "candidate_rows": candidate_rows,
        "epsilon_by_l": tuple((count, tuple(eps for k, eps in ordered if k == count))
                              for count in sorted({k for k, _ in ordered})),
        "candidates": [profile.freeze(epsilon=eps, max_steps=count).payload()
                       for count, eps in ordered],
        "omitted_candidates": omitted, "seeds": seed_log,
        "reserved_gradient_work": consumed_work,
        "actual_gradient_work": actual_work,
        "elapsed_seconds": prior_elapsed + time.monotonic() - started,
        "adaptation_and_pilot_draws_discarded": True,
        "tuning_artifact_authority": False,
        "posterior_convergence_authority": False,
        "fresh_measurement_and_verification_required": True,
    }
    snapshot("completed")
    if directory is not None:
        (directory / "pilot-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result
