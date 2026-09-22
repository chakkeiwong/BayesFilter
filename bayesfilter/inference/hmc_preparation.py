"""Active HMC geometry preparation and cooperative phase-level deadlines.

Numerical primitives keep their existing implementations. Preparation progress
is diagnostic evidence; it cannot grant candidate or retained-sampling authority.
"""
from __future__ import annotations

import json
import math
import operator
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Callable, Mapping


class HMCPreparationBudgetExceeded(TimeoutError):
    """Preparation reached its cooperative deadline before a frozen scope existed."""


class HMCPreparationFailure(RuntimeError):
    """Preparation failed with structured causes retained outside live tensors."""

    def __init__(self, message, *, details):
        super().__init__(message)
        self.details = dict(details)


def expanded_preparation_bound(upper, *, factor, steps):
    """A finite exploration cap, never stability or acceptance qualification."""
    if type(steps) is not int or steps < 0:
        raise ValueError("preparation bound expansion steps must be a nonnegative integer")
    if not math.isfinite(upper) or upper <= 0 or not math.isfinite(factor) or factor <= 1:
        raise ValueError("preparation bound and repair factor must be finite and positive with factor > 1")
    try:
        expanded = upper * factor ** steps
    except OverflowError as exc:
        raise ValueError("expanded preparation bound must be finite") from exc
    if not math.isfinite(expanded):
        raise ValueError("expanded preparation bound must be finite")
    return expanded


def _failure_details(stage, record):
    details = {"stage": stage, "final_status": record.final_status}
    # These are the stage's existing public diagnostics, including captured
    # exception type/message; no private warmup arrays enter the progress file.
    details["diagnostics"] = dict(getattr(record, "diagnostics", {}))
    for name in ("hard_vetoes", "continuation_vetoes", "repair_triggers"):
        details[name] = tuple(getattr(record, name, ()))
    details["rounds"] = [
        {"round_index": index,
         "final_status": getattr(row, "final_status", getattr(row, "classification", None)),
         "hard_vetoes": tuple(getattr(row, "hard_vetoes", ())),
         "diagnostics": dict(getattr(row, "diagnostics", {})),
         "repair_triggers": tuple(getattr(row, "repair_triggers", ()))}
        for index, row in enumerate(getattr(record, "rounds", ()))
    ]
    return details


def _progress_json_value(value):
    """Keep failed numerical diagnostics explicit without nonstandard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, Mapping):
        return {key: _progress_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_progress_json_value(item) for item in value]
    return value


class HMCPreparationProgress:
    def __init__(self, output_dir, *, max_wall_time_seconds=None):
        if max_wall_time_seconds is not None and (
                not math.isfinite(max_wall_time_seconds) or max_wall_time_seconds <= 0):
            raise ValueError("preparation wall-time limit must be finite and positive")
        self.path = None if output_dir is None else Path(output_dir) / "preparation_progress.json"
        self.max_wall_time_seconds = max_wall_time_seconds
        self.started = time.monotonic()
        self.events = []
        self.status = "running"
        self.failure = None

    @property
    def elapsed_seconds(self):
        return time.monotonic() - self.started

    def _write(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "bayesfilter.hmc_preparation_progress.v1",
            "status": self.status, "elapsed_seconds": self.elapsed_seconds,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "events": self.events, "failure": self.failure,
            "artifact_authority": False,
            "resume_policy": "fresh output directory; bootstrap chunk resume requires matching sources, target, starts and configuration; mass warmup is not resumable here"}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(_progress_json_value(payload), indent=2,
                                        default=str, allow_nan=False) + "\n")
        temporary.replace(self.path)

    def phase(self, stage, payload=None):
        self.events.append({"phase": stage, "elapsed_seconds": self.elapsed_seconds,
                            "details": dict(payload or {})})
        if self.max_wall_time_seconds is not None and self.elapsed_seconds >= self.max_wall_time_seconds:
            reason = "preparation deadline reached at " + stage
            self.status = "failed"
            self.failure = {"type": "HMCPreparationBudgetExceeded", "reason": reason}
            self._write()
            raise HMCPreparationBudgetExceeded(reason)
        self._write()

    def admit_work(self, stage, *, estimated_seconds, details=None):
        if not math.isfinite(estimated_seconds) or estimated_seconds <= 0:
            raise ValueError("work forecast must be positive and finite")
        remaining = (None if self.max_wall_time_seconds is None else
                     self.max_wall_time_seconds-self.elapsed_seconds)
        payload = {**dict(details or {}), "estimated_seconds": estimated_seconds,
                   "remaining_seconds": remaining}
        if remaining is not None and estimated_seconds > remaining:
            self.events.append({"phase": stage + ".deferred", "elapsed_seconds": self.elapsed_seconds,
                                "details": payload})
            self.status = "deferred"
            self.failure = {"type": "HMCPreparationBudgetExceeded",
                            "reason": "predicted work exceeds remaining preparation allowance"}
            self._write()
            raise HMCPreparationBudgetExceeded(self.failure["reason"] + ": " + stage)
        self.phase(stage, payload)

    def __enter__(self):
        self.phase("preparation_started")
        return self

    def __exit__(self, kind, exc, traceback):
        self.status = ("prepared" if exc is None else "deferred"
                       if isinstance(exc, HMCPreparationBudgetExceeded) else "failed")
        if exc is not None:
            self.failure = {"type": type(exc).__name__, "reason": str(exc)}
            if isinstance(exc, HMCPreparationFailure):
                self.failure["details"] = exc.details
        self._write()
        return False


def prepare_operational_windowed_mass_handoff(
    *,
    adapter: Any,
    initial_position: Any,
    config: HMCKernelTuningConfig | None = None,
    negative_hessian: Any | None = None,
    initial_covariance: Any | None = None,
    parameter_scales: Any | None = None,
    metric_window_size: int | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    initialize_bootstrap: bool = False,
    bootstrap_execution: Any | None = None,
) -> Mapping[str, Any]:
    """Prepare one public operational mass and post-warmup start-bank handoff.

    This is the bounded public prefix of ordinary HMC tuning: BayesFilter owns
    geometry initialization, bootstrap repair, the dimension-scaled warmup
    budget, operational windowed adaptation, and the final affine/start-bank
    lineage checks.  It intentionally stops before fixed-step or trajectory
    selection so callers that already own a reviewed fixed-kernel comparison
    do not have to run a second candidate campaign.

    ``metric_window_size`` is an explicit experimental one-window schedule:
    keep the preset's fast buffers and collect exactly this many consecutive
    positions in one slow window. It changes the allocation, not covariance
    adequacy gates or defaults. The schedule and all decisions are recorded.

    All warmup draws are discarded.  The returned mapping contains live
    BayesFilter adapters plus the public geometry, bootstrap, windowed-stage,
    and budget records needed to audit the handoff.  It is not posterior or
    convergence evidence.
    """
    from bayesfilter.inference.hmc_geometry import (
        _validate_position, initialize_hmc_kernel_geometry,
    )
    from bayesfilter.inference.hmc_bootstrap import run_hmc_bootstrap_screen
    from bayesfilter.inference.hmc_mass_adaptation import (
        _bootstrap_preflight_passed, run_hmc_windowed_mass_stage,
        build_operational_fixed_mass_hmc_adapter, _windowed_mass_stage_internal_config,
    )
    from bayesfilter.inference.hmc_configuration import (
        HMCKernelTuningConfig,
        _public_geometry_config, _public_bootstrap_config,
        _public_budget_policy_factory, _HMCAttemptBudgetPolicy,
        _public_loop_config, _phase7_windowed_stage_config,
    )
    from bayesfilter.runtime import stable_config_hash


    cfg = HMCKernelTuningConfig.standard() if config is None else config
    if not isinstance(cfg, HMCKernelTuningConfig):
        raise TypeError("config must be HMCKernelTuningConfig")
    if cfg.mass_policy != "windowed_adaptive":
        raise ValueError(
            "operational windowed mass preparation requires mass_policy='windowed_adaptive'"
        )
    if metric_window_size is not None:
        if isinstance(metric_window_size, bool):
            raise TypeError("metric_window_size must be an integer count")
        metric_window_size = operator.index(metric_window_size)
        if metric_window_size < 2:
            raise ValueError("metric_window_size must be at least two")
    position = _validate_position(initial_position)
    scope = cfg.target_scope or str(getattr(adapter, "target_scope", ""))
    if not scope:
        raise ValueError("target_scope must be supplied by config or adapter")

    def progress(stage: str, payload: Mapping[str, Any] | None = None) -> None:
        if progress_callback is not None:
            progress_callback(
                stage,
                {
                    "schema": "bayesfilter.operational_windowed_mass_preparation_progress.v1",
                    "stage": stage,
                    "reports_posterior_convergence": False,
                    "raw_samples_retained": False,
                    **({} if payload is None else dict(payload)),
                },
            )

    progress("geometry_started")
    geometry = initialize_hmc_kernel_geometry(
        adapter=adapter,
        initial_position=position,
        config=_public_geometry_config(cfg),
        negative_hessian=negative_hessian,
        initial_covariance=initial_covariance,
        parameter_scales=parameter_scales,
    )
    progress(
        "geometry_completed",
        {"geometry_artifact_hash": geometry.artifact_hash},
    )
    if type(initialize_bootstrap) is not bool:
        raise TypeError("initialize_bootstrap must be boolean")
    if initialize_bootstrap or cfg.bootstrap_initialization_rounds:
        from bayesfilter.inference.hmc_bootstrap_initialization import initialize_bootstrap_step
        geometry = initialize_bootstrap_step(adapter=adapter, geometry=geometry,
            config=_public_bootstrap_config(cfg, geometry=geometry), progress_callback=progress,
            max_rounds=cfg.bootstrap_initialization_rounds or 20)
    bootstrap_config = _public_bootstrap_config(cfg, geometry=geometry)
    execution_kwargs = {}
    if bootstrap_execution is not None:
        from bayesfilter.inference.hmc_bootstrap_checkpoint import CheckpointedBootstrapRunner
        if not isinstance(bootstrap_execution, CheckpointedBootstrapRunner):
            raise TypeError("bootstrap execution must use the repository checkpointed runner")
        bootstrap_execution.observe_initialization(geometry)
        bootstrap_config = replace(bootstrap_config, acceptance_role="warmup_startup_only")
        execution_kwargs["run_full_chain"] = bootstrap_execution
    bootstrap = run_hmc_bootstrap_screen(
        adapter=adapter,
        geometry=geometry,
        config=bootstrap_config,
        progress_callback=lambda stage, payload: progress("bootstrap." + stage, payload),
        **execution_kwargs,
    )
    # Keep round diagnostics and any first-failure record before the veto gate.
    progress("bootstrap_result", {"result": bootstrap.payload()})
    progress(
        "bootstrap_completed",
        {
            "bootstrap_artifact_hash": bootstrap.artifact_hash,
            "final_status": bootstrap.final_status,
        },
    )
    if not _bootstrap_preflight_passed(bootstrap):
        details = _failure_details("bootstrap", bootstrap)
        progress("bootstrap_failed", details)
        raise HMCPreparationFailure(
            "bootstrap contained a hard-vetoed round and cannot seed operational warmup",
            details=details)
    if bootstrap_execution is not None:
        from bayesfilter.inference.hmc_kernel_tuning import _json_ready
        # Preserve the startup handoff even if the full mass stage is unaffordable.
        (bootstrap_execution.root / "bootstrap-result.json").write_text(
            json.dumps(_progress_json_value(_json_ready(bootstrap.payload())),
                       indent=2, allow_nan=False) + "\n")
    if (cfg.bootstrap_initialization_rounds or bootstrap_execution is not None) and not bootstrap.passed:
        raise HMCPreparationFailure("startup floor was not reached",
                                    details=_failure_details("bootstrap", bootstrap))

    budget_factory = _public_budget_policy_factory(cfg, geometry=geometry)
    if budget_factory is None:
        raise RuntimeError("public operational warmup budget policy is unavailable")
    budget_policy = budget_factory(geometry.target_dimension, 0)
    if not isinstance(budget_policy, _HMCAttemptBudgetPolicy):
        raise TypeError("public operational warmup budget policy is invalid")
    schedule_options = {}
    if metric_window_size is not None:
        # Preserve all canonical numerical settings. Only consolidate the slow
        # draws into one explicit window; later fast draws must use any update.
        standard = _windowed_mass_stage_internal_config(
            budget_policy, mass_policy=cfg.mass_policy,
            metric_evidence_policy=cfg.metric_evidence_policy,
            metric_probe_num_results=cfg.metric_probe_num_results,
            preparation_max_restarts=cfg.preparation_max_restarts)
        schedule = replace(
            standard, first_window_size=metric_window_size,
            warmup_steps=standard.initial_buffer + metric_window_size + standard.final_buffer,
        )
        budget_policy = replace(budget_policy, phase4_warmup_steps=schedule.warmup_steps)
        schedule_options["_windowed_config"] = schedule
        progress("explicit_metric_window_schedule", {"schedule": schedule.payload()})
    loop_config = _public_loop_config(cfg)
    windowed_config = _phase7_windowed_stage_config(loop_config, attempt_index=0)
    if bootstrap_execution is not None:
        bootstrap_execution.admit("windowed_mass_full_work", budget_policy.phase4_warmup_steps)
    progress(
        "windowed_mass_started",
        {
            "warmup_steps": budget_policy.phase4_warmup_steps,
            "budget_policy_hash": stable_config_hash(budget_policy.payload()),
        },
    )
    windowed = run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=windowed_config,
        _attempt_budget_policy=budget_policy,
        _progress_callback=(
            None
            if progress_callback is None
            else lambda stage, payload: progress(f"windowed_mass.{stage}", payload)
        ),
        **schedule_options,
    )
    # Persist the public decisions before either rejection gate raises. A
    # completed but unaccepted window is evidence, not an absent result.
    progress("windowed_mass_result", {"result": windowed.payload()})
    operational = windowed.operational_warmup_result
    if not windowed.passed or operational is None:
        details = _failure_details("windowed_mass", windowed)
        progress("windowed_mass_failed", details)
        raise HMCPreparationFailure(
            f"operational windowed mass stage did not pass: {windowed.final_status}",
            details=details)
    if (
        cfg.metric_update_requirement == "require_operational_update"
        and operational.operational_metric_update_count <= 0
    ):
        raise RuntimeError("required operational metric update was not observed")
    progress(
        "windowed_mass_completed",
        {
            "final_status": windowed.final_status,
            "operational_metric_update_count": (
                operational.operational_metric_update_count
            ),
            "metric_adaptation_status": operational.metric_adaptation_status,
        },
    )
    handoff = build_operational_fixed_mass_hmc_adapter(
        adapter=adapter,
        geometry=geometry,
        windowed_stage=windowed,
        target_scope=scope,
    )
    progress(
        "handoff_completed",
        {
            "final_adapter_signature": handoff["final_adapter_signature"],
            "adapted_mass_artifact_signature": (
                handoff["adapted_mass_artifact_signature"]
            ),
        },
    )
    return {
        **handoff,
        "config": cfg,
        "geometry": geometry,
        "bootstrap": bootstrap,
        "windowed_stage": windowed,
        "budget_policy_payload": budget_policy.payload(),
        "budget_policy_hash": stable_config_hash(budget_policy.payload()),
        # This prefix invokes warmup preparation only.  Preserve the handoff's
        # explicit nonclaim that fixed-kernel HMC/tuning was run.
        "hmc_or_tuning_invoked": handoff["hmc_or_tuning_invoked"],
        "warmup_invoked": True,
        "warmup_draws_discarded": True,
        "reports_posterior_convergence": False,
    }
