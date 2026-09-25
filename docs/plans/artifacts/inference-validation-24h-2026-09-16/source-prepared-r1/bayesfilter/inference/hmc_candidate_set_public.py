"""Preparation translations for the single public candidate-set scheduler.

Legacy configuration classes describe preparation and initial hypotheses only.
The common controller owns every subsequent candidate decision.
"""
from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import time
from typing import Any

from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig, _sha256
from bayesfilter.inference.hmc_candidate_set_execution import (
    HMCCandidateExecutionConfig, bind_hmc_candidate_set_execution,
    bind_hmc_candidate_set_execution_from_preparation,
)
from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy
from bayesfilter.inference.hmc_preparation import HMCPreparationProgress


def _preflight_output(output_dir):
    if output_dir is not None:
        root = Path(output_dir)
        if any((root / name).exists() for name in (
                "candidate_set_result.json", "tuning_checkpoint.json", "execution_spec.json",
                "controller_checkpoint.json", "position_field_preparation.json", "preparation_progress.json")):
            raise FileExistsError("existing tuning run; resume it or use a fresh output directory")


def _starts(initial_position):
    import tensorflow as tf
    starts = tf.convert_to_tensor(initial_position, tf.float64)
    if starts.shape.rank == 1:
        starts = tf.repeat(starts[None], 4, axis=0)
    if starts.shape.rank != 2 or starts.shape[0] != 4:
        raise ValueError("supply one position or a four-chain initial bank")
    return starts


def _provenance(adapter, target_lineage, source_paths):
    from bayesfilter.inference.hmc import stable_adapter_signature
    path = inspect.getsourcefile(type(adapter))
    paths = tuple(source_paths or ()) or ((path,) if path else ())
    lineage = target_lineage or getattr(adapter, "target_lineage", None) or {
        "adapter_signature": stable_adapter_signature(adapter),
        "coverage": "adapter-declared identity; data/prior completeness requires consumer review",
    }
    return lineage, paths


def _reject_changed_retired_options(config, names):
    reference = replace(config, **{name: config.__dataclass_fields__[name].default for name in names})
    changed = [name for name in names if getattr(config, name) != getattr(reference, name)]
    if changed:
        raise ValueError("retired tuning options require migration to search_config/execution_config: " + ", ".join(changed))


def _execution(config, *, seed, policy, status, use_xla, warmup=0, preparation_elapsed=0.):
    if config is not None:
        if not isinstance(config, HMCCandidateExecutionConfig):
            raise TypeError("execution_config must be HMCCandidateExecutionConfig")
        if config.use_xla != use_xla or config.target_status_trace_policy != status:
            raise ValueError("preparation and candidate execution policies disagree")
        return replace(config, preparation_elapsed_seconds=preparation_elapsed)
    # The evaluator's existing minimum defines the first evidence allocation;
    # the frozen controller rungs supply further evidence when inconclusive.
    return HMCCandidateExecutionConfig(
        measurement_num_results=policy.min_decisions_per_chain,
        verification_num_results=policy.min_decisions_per_chain,
        num_warmup_steps=warmup, seed=seed, acceptance_policy=policy,
        target_status_trace_policy=status, use_xla=use_xla,
        non_xla_reason=None if use_xla else "explicit public preparation CPU/reference exception",
        preparation_elapsed_seconds=preparation_elapsed)


def _search(config, *, epsilon, grid, max_candidates=100):
    if config is not None:
        if not isinstance(config, HMCControllerConfig):
            raise TypeError("search_config must be HMCControllerConfig")
        return config
    # One pilot for every L, then fixed measurement and fresh verification.
    # Geometric epsilon refinement is a bounded hypothesis, never a ranking.
    return HMCControllerConfig(primary_l_grid=tuple(grid), initial_epsilon=epsilon,
        pilot_enabled=True, refinement_rounds=1,
        refinement_l_grid=tuple((a+b)//2 for a, b in zip(grid[:-1], grid[1:]) if b-a > 1),
        max_candidates=max_candidates, total_budget_units=3 * max_candidates,
        repair_reserve_units=min(20, 3 * max_candidates))


def _preflight_search(config, *, max_leapfrog_steps=None, max_work_items=None):
    if config is not None and not isinstance(config, HMCControllerConfig):
        raise TypeError("search_config must be HMCControllerConfig")
    if max_work_items is not None and (type(max_work_items) is not int or max_work_items < 0):
        raise ValueError("max_work_items must be an integer >= 0")
    if config is not None and max_leapfrog_steps is not None:
        steps = (*config.primary_l_grid, *config.refinement_l_grid, *config.expansion_l_grid)
        if any(value > max_leapfrog_steps for value in steps):
            raise ValueError("all search, refinement and expansion L values must obey max_leapfrog_steps")
    if config is not None:
        if sum(len(values) for _, values in config.epsilon_by_l) * config.candidate_reserve_units > config.total_budget_units:
            raise ValueError("declared primary candidate cohort does not fit total budget")


def _preflight_exact_target(adapter, *, use_xla, target_scope):
    from bayesfilter.inference.posterior_adapter import value_score_capability
    capability = value_score_capability(adapter)
    if capability.value_score_authority not in {
        "graph_native", "analytical_manual", "reviewed_gradient_tape_xla_exception",
    }:
        raise ValueError("candidate execution requires an exact TensorFlow value/score authority")
    if target_scope and capability.target_scope is not None and capability.target_scope != target_scope:
        raise ValueError("target scope mismatch")
    if use_xla and not capability.is_accepted_full_chain_xla_diagnostic_authority:
        raise ValueError("target lacks full-chain XLA qualification")


def _wall_limit(search_config, preparation_limit=None):
    values = [value for value in (preparation_limit,
        None if search_config is None else search_config.max_wall_time_seconds) if value is not None]
    return min(values) if values else None


def _domain(search, factor, repairs, upper=None):
    proposals = [epsilon for _, values in search.epsilon_by_l for epsilon in values]
    low = min(proposals) / factor ** (repairs + 1)
    high = max(proposals) * factor ** (repairs + 1) if upper is None else upper
    if high < max(proposals):
        raise ValueError("epsilon proposal exceeds the declared final-metric safety bound")
    return low, high


def run_shared_ordinary_tuning(*, adapter: Any, initial_position: Any, config: Any,
        output_dir=None, negative_hessian=None, initial_covariance=None, parameter_scales=None,
        search_config=None, execution_config=None, target_lineage=None, source_paths=(),
        max_work_items=None):
    import tensorflow as tf
    from bayesfilter.inference.hmc import PrecomputedMassArtifact, stable_adapter_signature
    from bayesfilter.inference.hmc_kernel_tuning import (
        HMCKernelTuningConfig, prepare_operational_windowed_mass_handoff,
        initialize_hmc_kernel_geometry, _public_geometry_config, _fixed_mass_step_upper_bound,
    )
    from bayesfilter.hmc_ordinary_selection_policy import ORDINARY_BROAD_PRIMARY_L_GRID

    _preflight_output(output_dir)
    cfg = HMCKernelTuningConfig.standard() if config is None else config
    if not isinstance(cfg, HMCKernelTuningConfig):
        raise TypeError("ordinary preparation requires HMCKernelTuningConfig")
    _preflight_search(search_config, max_leapfrog_steps=cfg.max_leapfrog_steps,
                      max_work_items=max_work_items)
    if cfg.chain_execution_mode != "tf_function":
        raise ValueError("shared tuning requires tf_function; historical eager execution is diagnostic-only")
    if cfg.staged_timeout_policy is not None or cfg.terminal_phase6_repair_extra_attempts:
        raise ValueError("legacy stage retry policies are retired; declare shared search budgets and evidence_rungs")
    _reject_changed_retired_options(cfg, (
        "step_repair_min_directional_factor", "step_repair_high_acceptance_directional_factor",
        "step_repair_high_acceptance_ladder_max_factor", "repair_nonfinite_proposal_screen",
        "trajectory_window_lower_multiplier", "trajectory_window_upper_multiplier",
        "operational_candidate_handoff_policy", "operational_verification_bracket_policy",
        "staged_timeout_global_started_perf_counter_s", "staged_timeout_stage_started_perf_counter_s",
        "staged_timeout_enlargement_rounds", "incall_progress_heartbeat_s"))
    started = time.monotonic()
    lineage, paths = _provenance(adapter, target_lineage, source_paths)
    policy = HMCAcceptancePolicy(target=cfg.target_accept_prob, practical_region=cfg.acceptance_band, repair_region=cfg.repair_band)
    execution = _execution(execution_config, seed=cfg.seed, policy=policy,
        status=cfg.target_status_trace_policy, use_xla=cfg.use_xla)
    _preflight_exact_target(adapter, use_xla=cfg.use_xla, target_scope=cfg.target_scope)
    progress = HMCPreparationProgress(output_dir,
        max_wall_time_seconds=_wall_limit(search_config, cfg.public_timeout_budget_s))
    with progress:
        preparation = None
        upper = None
        if cfg.mass_policy == "windowed_adaptive":
            preparation = prepare_operational_windowed_mass_handoff(adapter=adapter,
                initial_position=initial_position, config=cfg, negative_hessian=negative_hessian,
                initial_covariance=initial_covariance, parameter_scales=parameter_scales,
                progress_callback=progress.phase)
            epsilon = preparation["windowed_stage"].operational_warmup_result.final_kernel_state.epsilon
            upper = _fixed_mass_step_upper_bound(preparation["windowed_stage"])
            scope = preparation["target_scope"]
        else:
            progress.phase("identity_geometry_started")
            starts = _starts(initial_position)
            geometry = initialize_hmc_kernel_geometry(adapter=adapter, initial_position=starts[0],
                config=_public_geometry_config(cfg))
            epsilon = geometry.initial_step_size
            from bayesfilter.inference.posterior_adapter import value_score_capability
            scope = cfg.target_scope or value_score_capability(adapter).target_scope
            if not scope:
                raise ValueError("ordinary tuning requires an explicit target_scope")
            mass = PrecomputedMassArtifact(position=tf.zeros(starts.shape[-1], tf.float64),
                factor=tf.eye(starts.shape[-1], dtype=tf.float64),
                covariance=tf.eye(starts.shape[-1], dtype=tf.float64),
                adapter_signature=stable_adapter_signature(adapter),
                position_role="fixed_identity", covariance_source="explicit fixed_identity preparation policy")
            progress.phase("identity_geometry_completed")
        grid = tuple(l for l in ORDINARY_BROAD_PRIMARY_L_GRID if l <= cfg.max_leapfrog_steps)
        if not grid:
            raise ValueError("max_leapfrog_steps excludes the declared broad L grid")
        search = _search(search_config, epsilon=epsilon, grid=grid)
        search = replace(search, max_wall_time_seconds=progress.max_wall_time_seconds)
        execution = replace(execution, preparation_elapsed_seconds=time.monotonic()-started)
        if execution_config is None:
            execution = replace(execution,
                chunk_max_results=cfg.verification_chunk_max_results or execution.chunk_max_results,
                verification_num_results=max(execution.verification_num_results,
                                             cfg.verification_min_retained_results_for_pass or 0))
        common = dict(adapter=adapter, target_lineage=lineage, config=execution,
            source_paths=paths, scope_id=scope, search_id=_sha256(search.payload())[:20],
            epsilon_domain=_domain(search, cfg.step_repair_factor, cfg.max_attempts, upper),
            repair_factor=cfg.step_repair_factor, max_repairs_per_family=cfg.max_attempts)
        if preparation is not None:
            binding = bind_hmc_candidate_set_execution_from_preparation(preparation=preparation, **common)
        else:
            binding = bind_hmc_candidate_set_execution(initial_position=starts, mass_artifact=mass,
                                                      target_scope=scope, **common)
        progress.phase("frozen_execution_ready")
    return run_typed_hmc_candidate_set(binding.typed_adapter, search, output_dir=output_dir,
        max_work_items=max_work_items, _preparation_elapsed_seconds=progress.elapsed_seconds)


def run_shared_fixed_transport_tuning(*, base_adapter, fixed_transport, initial_position,
        config, frozen_transport_payload, output_dir=None, search_config=None,
        execution_config=None, target_lineage=None, source_paths=(), max_work_items=None):
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import FixedTransportHMCKernelTuningConfig
    _preflight_output(output_dir)
    _preflight_search(search_config, max_work_items=max_work_items)
    if frozen_transport_payload is None:
        raise ValueError("fixed-transport migration requires frozen_transport_payload or a numerical candidate_set_adapter")
    cfg = config or FixedTransportHMCKernelTuningConfig(initial_step_size=.1,
                                                       leapfrog_grid=(3, 5, 9, 13, 18, 25))
    if not isinstance(cfg, FixedTransportHMCKernelTuningConfig):
        raise TypeError("fixed-transport preparation requires FixedTransportHMCKernelTuningConfig")
    if cfg.chain_count != 4:
        raise ValueError("shared fixed-transport tuning requires exactly four chains")
    if cfg.chain_execution_mode != "tf_function":
        raise ValueError("shared fixed-transport tuning requires tf_function execution")
    if cfg.require_modern_rank_normalized_verification:
        raise ValueError("R-hat tuning admission is retired; use reporting and separate posterior assessment")
    if cfg.tuning_policy != "measured_joint_grid_v1" or cfg.fixed_grid_scale_candidates:
        raise ValueError("historical directional/scale selection is retired; supply shared epsilon_by_l proposals")
    _reject_changed_retired_options(cfg, (
        "selection_policy", "selection_replications", "selection_num_results", "selection_num_burnin_steps",
        "selection_seed_base", "budget_schedule", "tune_num_results",
        "screen_seed_base", "verification_seed_base", "fixed_grid_base_step_size_candidates",
        "fixed_grid_num_leapfrog_steps", "fixed_grid_fallback_acceptance_max", "output_filename",
        "report_modern_rank_normalized_verification", "verification_min_retained_results_per_chain",
        "verification_rhat_max", "verification_coordinate_system"))
    if cfg.selection_acceptance_band != cfg.acceptance_band:
        raise ValueError("retired selection_acceptance_band must migrate to execution_config.acceptance_policy")
    lineage, paths = _provenance(base_adapter, target_lineage, source_paths)
    policy = HMCAcceptancePolicy(target=cfg.target_accept_prob, practical_region=cfg.acceptance_band, repair_region=cfg.repair_band,
        max_abs_log_accept_energy_proxy=cfg.maximum_absolute_energy_error)
    execution = _execution(execution_config, seed=cfg.tune_seed_base, policy=policy,
        status=cfg.target_status_trace_policy, use_xla=cfg.use_xla,
        warmup=cfg.screen_num_burnin_steps)
    _preflight_exact_target(base_adapter, use_xla=cfg.use_xla, target_scope=cfg.target_scope)
    if execution_config is None:
        if cfg.screen_num_burnin_steps != cfg.verification_num_burnin_steps:
            raise ValueError("shared stages use one discarded warmup count; supply execution_config")
        execution = replace(execution,
            measurement_num_results=max(policy.min_decisions_per_chain, cfg.screen_num_results),
            verification_num_results=max(policy.min_decisions_per_chain, cfg.verification_num_results))
    search = _search(search_config, epsilon=cfg.initial_step_size, grid=cfg.leapfrog_grid,
                     max_candidates=cfg.max_joint_candidate_count)
    if search_config is None and cfg.step_size_candidates:
        search = replace(search, epsilon_by_l=tuple((l, cfg.step_size_candidates) for l in cfg.leapfrog_grid))
    scope = cfg.target_scope or str(lineage.get("adapter_signature", "fixed_transport"))
    with HMCPreparationProgress(output_dir, max_wall_time_seconds=search.max_wall_time_seconds) as progress:
        progress.phase("frozen_transport_binding_started")
        binding = bind_hmc_candidate_set_execution(adapter=base_adapter,
            initial_position=_starts(cfg.initial_state_bank or initial_position),
            target_scope=scope, target_lineage=lineage, source_paths=paths, config=execution,
            scope_id=scope, search_id=_sha256(search.payload())[:20],
            epsilon_domain=_domain(search, cfg.step_repair_factor, cfg.fixed_grid_max_attempts,
                                   cfg.maximum_candidate_step_size),
            repair_factor=cfg.step_repair_factor, max_repairs_per_family=cfg.fixed_grid_max_attempts,
            frozen_transport_payload=frozen_transport_payload, start_coordinates="active")
        binding.validate_transport(fixed_transport)
        progress.phase("frozen_transport_binding_completed")
    return run_typed_hmc_candidate_set(binding.typed_adapter, search,
        output_dir=output_dir, max_work_items=max_work_items,
        _preparation_elapsed_seconds=progress.elapsed_seconds)
