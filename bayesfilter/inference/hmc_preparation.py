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


def _diagnostic_json(value):
    """Preserve rejected-window NaN/inf as named JSON diagnostics, never numerics."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_diagnostic_json(item) for item in value]
    return value


class HMCPreparationBudgetExceeded(TimeoutError):
    """Preparation reached its cooperative deadline before a frozen scope existed."""


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
            "resume_policy": "retry failed preparation in a fresh directory; numerical resume requires a frozen scope"}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(_diagnostic_json(payload), indent=2, default=str, allow_nan=False) + "\n")
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

    def __enter__(self):
        self.phase("preparation_started")
        return self

    def __exit__(self, kind, exc, traceback):
        self.status = "prepared" if exc is None else "failed"
        if exc is not None:
            self.failure = {"type": type(exc).__name__, "reason": str(exc)}
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
    from bayesfilter.inference.hmc_kernel_tuning import (
        HMCKernelTuningConfig, _validate_position, initialize_hmc_kernel_geometry,
        _public_geometry_config, run_hmc_bootstrap_screen, _public_bootstrap_config,
        _bootstrap_preflight_passed, _public_budget_policy_factory, _HMCAttemptBudgetPolicy,
        _public_loop_config, _phase7_windowed_stage_config, stable_config_hash,
        run_hmc_windowed_mass_stage, build_operational_fixed_mass_hmc_adapter,
        _windowed_mass_stage_internal_config,
    )


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
    bootstrap = run_hmc_bootstrap_screen(
        adapter=adapter,
        geometry=geometry,
        config=_public_bootstrap_config(cfg, geometry=geometry),
        progress_callback=(
            None if progress_callback is None
            else lambda stage, payload: progress(f"bootstrap.{stage}", payload)
        ),
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
        raise RuntimeError(
            "bootstrap contained a hard-vetoed round and cannot seed operational warmup"
        )

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
        standard = _windowed_mass_stage_internal_config(budget_policy)
        schedule = replace(
            standard, first_window_size=metric_window_size,
            warmup_steps=standard.initial_buffer + metric_window_size + standard.final_buffer,
        )
        budget_policy = replace(budget_policy, phase4_warmup_steps=schedule.warmup_steps)
        schedule_options["_windowed_config"] = schedule
        progress("explicit_metric_window_schedule", {"schedule": schedule.payload()})
    loop_config = _public_loop_config(cfg)
    windowed_config = _phase7_windowed_stage_config(loop_config, attempt_index=0)
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
        raise RuntimeError(
            f"operational windowed mass stage did not pass: {windowed.final_status}"
        )
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
