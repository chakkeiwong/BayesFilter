"""BayesFilter-owned robust fixed-mass HMC broad-grid tuning.

This module closes the generic orchestration gap between the existing geometry
and mass stages and the existing callback-only broad-grid contracts.  For every
primary ``L`` it runs independent TFP dual averaging, repairs epsilon with
fresh fixed-kernel screens, qualifies every surviving pair with a reviewed
fresh-transition rung,
transitions, and selects by minimum bulk ESS after hard validity, symmetric
acceptance, and R-hat suitability gates.

All draws are discarded after diagnostics.  A selected candidate is a tuning
handoff only; this module does not claim posterior convergence or sampler
superiority.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from bayesfilter.inference.hmc import (
    FullChainHMCConfig,
    build_reusable_full_chain_tfp_hmc_runner,
    run_full_chain_tfp_hmc,
    stable_adapter_signature,
)
from bayesfilter.inference.hmc_convergence import (
    RankNormalizedHMCThresholds,
    rank_normalized_hmc_diagnostics,
)
from bayesfilter.inference.hmc_kernel_tuning import (
    HMCKernelTuningConfig,
    prepare_operational_windowed_mass_handoff,
)
from bayesfilter.inference.hmc_tuning import HMCTuningPolicy
from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    evaluate_hmc_acceptance_evidence,
)
from bayesfilter.runtime import stable_config_hash


DEFAULT_L_GRID = (3, 5, 9, 13, 18, 25)
NONCLAIMS = (
    "discarded robust broad-grid tuning evidence only",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no empirical validity claim",
)


def _seed(root: tuple[int, int], domain: str, l: int, index: int = 0) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"robust-broad-grid:{root[0]}:{root[1]}:{domain}:{int(l)}:{int(index)}".encode("ascii")
    ).digest()
    modulus = 2**31 - 1
    value = (
        (int(root[0]) + int.from_bytes(digest[:8], "big")) % modulus,
        (int(root[1]) + int.from_bytes(digest[8:16], "big")) % modulus,
    )
    return (0, 1) if value == (0, 0) else value


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class RobustBroadGridConfig:
    """Reviewed controls for the generic five-stage tuning campaign."""

    root_seed: tuple[int, int] = (20260814, 7401)
    mass_preparation_seed: tuple[int, int] = (2026, 728)
    l_grid: tuple[int, ...] = DEFAULT_L_GRID
    target_accept_prob: float = 0.70
    acceptance_band: tuple[float, float] = (0.65, 0.75)
    repair_band: tuple[float, float] = (0.55, 0.85)
    dual_averaging_steps: int = 128
    tune_num_results: int = 8
    tune_burnin_steps: int = 128
    repair_screen_results: int = 64
    repair_screen_burnin_steps: int = 16
    qualification_results: int = 500
    qualification_burnin_steps: int = 125
    max_epsilon_repairs: int = 5
    epsilon_repair_factor: float = 1.25
    target_scope: str | None = None
    chain_execution_mode: str = "tf_function"
    use_xla: bool = True
    l_grid_provenance: str = (
        "inherited_robust_broad_grid_baseline_not_universal_default"
    )
    qualification_rung_provenance: str = (
        "inherited_500_transition_screen_not_posterior_verification"
    )

    def __post_init__(self) -> None:
        seed = tuple(int(item) for item in self.root_seed)
        if len(seed) != 2 or any(item < 0 for item in seed):
            raise ValueError("root_seed must contain two nonnegative integers")
        preparation_seed = tuple(int(item) for item in self.mass_preparation_seed)
        if len(preparation_seed) != 2 or any(item < 0 for item in preparation_seed):
            raise ValueError("mass_preparation_seed must contain two nonnegative integers")
        grid = tuple(int(item) for item in self.l_grid)
        if not grid or len(set(grid)) != len(grid):
            raise ValueError("l_grid must be non-empty and contain unique values")
        if any(item < 2 for item in grid):
            raise ValueError("l_grid values must be at least 2")
        target = float(self.target_accept_prob)
        practical = tuple(float(item) for item in self.acceptance_band)
        repair = tuple(float(item) for item in self.repair_band)
        if not (0.0 < practical[0] < target < practical[1] < 1.0):
            raise ValueError("target must lie inside acceptance_band")
        if not (0.0 < repair[0] <= practical[0] <= practical[1] <= repair[1] < 1.0):
            raise ValueError("repair_band must contain acceptance_band")
        if not (abs(target - sum(practical) / 2.0) < 1e-12):
            raise ValueError("acceptance_band must be symmetric around target")
        for name in (
            "dual_averaging_steps",
            "tune_num_results",
            "tune_burnin_steps",
            "repair_screen_results",
            "repair_screen_burnin_steps",
            "qualification_results",
            "qualification_burnin_steps",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.repair_screen_results < 64:
            raise ValueError("repair_screen_results must support acceptance evidence")
        repairs = int(self.max_epsilon_repairs)
        if repairs < 0 or repairs > 5:
            raise ValueError("max_epsilon_repairs must lie in [0, 5]")
        factor = float(self.epsilon_repair_factor)
        if not (math.isfinite(factor) and 1.0 < factor <= 2.0):
            raise ValueError("epsilon_repair_factor must lie in (1, 2]")
        mode = str(self.chain_execution_mode)
        if mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be tf_function or eager")
        if self.use_xla and mode != "tf_function":
            raise ValueError("XLA requires tf_function execution")
        scope = None if self.target_scope is None else str(self.target_scope)
        if scope == "":
            raise ValueError("target_scope must be non-empty")
        object.__setattr__(self, "root_seed", seed)
        object.__setattr__(self, "mass_preparation_seed", preparation_seed)
        object.__setattr__(self, "l_grid", grid)
        object.__setattr__(self, "target_accept_prob", target)
        object.__setattr__(self, "acceptance_band", practical)
        object.__setattr__(self, "repair_band", repair)
        object.__setattr__(self, "max_epsilon_repairs", repairs)
        object.__setattr__(self, "epsilon_repair_factor", factor)
        object.__setattr__(self, "target_scope", scope)
        object.__setattr__(self, "chain_execution_mode", mode)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        for name in ("l_grid_provenance", "qualification_rung_provenance"):
            provenance = str(getattr(self, name))
            if not provenance:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, provenance)

    def payload(self) -> Mapping[str, Any]:
        return {
            "route": "bayesfilter.robust_fixed_mass_broad_grid_v1",
            "root_seed": self.root_seed,
            "mass_preparation_seed": self.mass_preparation_seed,
            "mass_preparation_seed_role": "replay of prior passed serious mass-preparation baseline",
            "l_grid": self.l_grid,
            "l_grid_provenance": self.l_grid_provenance,
            "target_accept_prob": self.target_accept_prob,
            "acceptance_band": self.acceptance_band,
            "repair_band": self.repair_band,
            "dual_averaging_steps": self.dual_averaging_steps,
            "tune_num_results": self.tune_num_results,
            "tune_burnin_steps": self.tune_burnin_steps,
            "repair_screen_results": self.repair_screen_results,
            "repair_screen_burnin_steps": self.repair_screen_burnin_steps,
            "qualification_results": self.qualification_results,
            "qualification_rung_provenance": self.qualification_rung_provenance,
            "qualification_burnin_steps": self.qualification_burnin_steps,
            "max_epsilon_repairs": self.max_epsilon_repairs,
            "epsilon_repair_factor": self.epsilon_repair_factor,
            "target_scope": self.target_scope,
            "chain_execution_mode": self.chain_execution_mode,
            "use_xla": self.use_xla,
            "non_xla_role": (
                None if self.use_xla else "explicit_reference_or_debug_exception"
            ),
            "nonclaims": NONCLAIMS,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "numpy"):
        return _jsonable(value.numpy())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _trace_tensors(run: Any) -> tuple[Any, Any, Any, Any, Any | None]:
    import tensorflow as tf

    samples = tf.cast(tf.convert_to_tensor(run.samples), tf.float64)
    trace = dict(run.trace) if isinstance(run.trace, Mapping) else {}
    required = ("log_accept_ratio", "is_accepted", "target_log_prob")
    if samples.shape.rank != 3 or any(key not in trace for key in required):
        raise ValueError("HMC run did not return the standard three-dimensional trace")
    log_accept = tf.cast(tf.convert_to_tensor(trace["log_accept_ratio"]), tf.float64)
    accepted = tf.cast(tf.convert_to_tensor(trace["is_accepted"]), tf.bool)
    target = tf.cast(tf.convert_to_tensor(trace["target_log_prob"]), tf.float64)
    divergence = trace.get("divergence")
    divergence_tensor = None if divergence is None else tf.cast(tf.convert_to_tensor(divergence), tf.bool)
    for tensor in (samples, log_accept, target):
        if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
            raise ValueError("HMC run returned a non-finite tensor")
    return samples, log_accept, accepted, target, divergence_tensor


def _evidence(run: Any, policy: HMCAcceptancePolicy) -> Any:
    import tensorflow as tf

    samples, log_accept, accepted, target, divergence = _trace_tensors(run)
    return evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=log_accept,
        is_accepted=accepted,
        target_log_prob=target,
        policy=policy,
        native_divergence_status="available" if divergence is not None else "not_exposed_by_kernel",
        native_divergence_count=None if divergence is None else int(tf.reduce_sum(tf.cast(divergence, tf.int32)).numpy()),
    )


def _fixed_config(*, step: float, l: int, seed: tuple[int, int], cfg: RobustBroadGridConfig, results: int, burnin: int) -> FullChainHMCConfig:
    return FullChainHMCConfig(
        num_results=results,
        num_burnin_steps=burnin,
        step_size=step,
        num_leapfrog_steps=l,
        seed=seed,
        use_xla=cfg.use_xla,
        trace_policy="standard",
        target_status_trace_policy="none",
        target_scope=cfg.target_scope,
        chain_execution_mode=cfg.chain_execution_mode,
    )


def _tune_one_l(adapter: Any, starts: Any, l: int, initial_step: float, cfg: RobustBroadGridConfig, policy: HMCAcceptancePolicy) -> Mapping[str, Any]:
    import tensorflow as tf

    tune_config = FullChainHMCConfig(
        num_results=cfg.tune_num_results,
        num_burnin_steps=cfg.tune_burnin_steps,
        step_size=initial_step,
        num_leapfrog_steps=l,
        seed=_seed(cfg.root_seed, "dual_averaging", l),
        use_xla=cfg.use_xla,
        trace_policy="standard",
        target_status_trace_policy="none",
        tuning_policy=HMCTuningPolicy.fixed_mass_dual_averaging(
            num_adaptation_steps=cfg.dual_averaging_steps,
            target_accept_prob=cfg.target_accept_prob,
            source="bayesfilter.inference.hmc_robust_broad_grid.dual_averaging",
        ),
        target_scope=cfg.target_scope,
        chain_execution_mode=cfg.chain_execution_mode,
    )
    runner = build_reusable_full_chain_tfp_hmc_runner(
        adapter,
        starts,
        tune_config,
        dynamic_num_leapfrog_steps=True,
    )
    run = runner.run(current_state=starts, seed=tune_config.seed, step_size=initial_step, num_leapfrog_steps=l)
    step_trace = run.trace.get("step_size")
    if step_trace is None:
        raise ValueError("dual averaging did not return step_size trace")
    step_array = tf.reshape(tf.cast(tf.convert_to_tensor(step_trace), tf.float64), [-1])
    if int(step_array.shape[0]) == 0 or not bool(tf.reduce_all(tf.math.is_finite(step_array)).numpy()) or float(step_array[-1].numpy()) <= 0.0:
        raise ValueError("dual averaging returned a non-finite epsilon")
    return {"step_size": float(step_array[-1].numpy()), "steps": tuple(float(item) for item in step_array.numpy().tolist()), "tune_run": run}


def _repair_one_l(adapter: Any, starts: Any, l: int, step: float, cfg: RobustBroadGridConfig, policy: HMCAcceptancePolicy) -> Mapping[str, Any]:
    history = []
    current = _finite_positive(step, "step_size")
    runner = build_reusable_full_chain_tfp_hmc_runner(
        adapter,
        starts,
        _fixed_config(step=current, l=l, seed=_seed(cfg.root_seed, "repair", l), cfg=cfg, results=cfg.repair_screen_results, burnin=cfg.repair_screen_burnin_steps),
        dynamic_num_leapfrog_steps=True,
    )
    final_evidence = None
    for repair_index in range(cfg.max_epsilon_repairs + 1):
        config = _fixed_config(step=current, l=l, seed=_seed(cfg.root_seed, "repair", l, repair_index), cfg=cfg, results=cfg.repair_screen_results, burnin=cfg.repair_screen_burnin_steps)
        run = runner.run(current_state=starts, seed=config.seed, step_size=current, num_leapfrog_steps=l)
        evidence = _evidence(run, policy)
        payload = evidence.payload()
        history.append({"repair_index": repair_index, "step_size": current, "evidence": payload})
        final_evidence = evidence
        if evidence.acceptance_decision == "passed":
            break
        if evidence.acceptance_decision == "repair_step_lower":
            current /= cfg.epsilon_repair_factor
        elif evidence.acceptance_decision == "repair_step_higher":
            current *= cfg.epsilon_repair_factor
        else:
            break
    return {"step_size": current, "history": tuple(history), "evidence": final_evidence}


def _qualification(
    adapter: Any,
    starts: Any,
    l: int,
    step: float,
    cfg: RobustBroadGridConfig,
    policy: HMCAcceptancePolicy,
    *,
    phase4_adapter: Any,
    parameter_names: Sequence[str],
) -> Mapping[str, Any]:
    import tensorflow as tf

    config = _fixed_config(step=step, l=l, seed=_seed(cfg.root_seed, "qualification", l), cfg=cfg, results=cfg.qualification_results, burnin=cfg.qualification_burnin_steps)
    run = run_full_chain_tfp_hmc(adapter, starts, config)
    evidence = _evidence(run, policy)
    samples, _, _, _, divergence = _trace_tensors(run)
    phase4_samples = adapter.latent_to_position(samples)
    model_samples = phase4_adapter.latent_to_position(phase4_samples)
    convergence = rank_normalized_hmc_diagnostics(
        model_samples,
        parameter_names=tuple(str(name) for name in parameter_names),
        thresholds=RankNormalizedHMCThresholds(rhat_max=1.05, bulk_ess_min=1.0, tail_ess_min=1.0),
    )
    return {
        "step_size": step,
        "acceptance": evidence.payload(),
        "convergence": convergence,
        "native_divergence_count": None if divergence is None else int(tf.reduce_sum(tf.cast(divergence, tf.int32)).numpy()),
        "run_metadata": dict(run.metadata),
    }


def _suitable(row: Mapping[str, Any]) -> bool:
    acceptance = row["qualification"]["acceptance"]
    convergence = row["qualification"]["convergence"]
    divergence = row["qualification"].get("native_divergence_count")
    return bool(
        acceptance.get("evidence_validity") == "valid"
        and acceptance.get("acceptance_decision")
        in {"passed", "inconclusive_evidence"}
        and not acceptance.get("candidate_promotion_vetoes")
        and not acceptance.get("cost_stop_reasons")
        and (divergence is None or divergence == 0)
        and convergence.get("diagnostics_all_finite") is True
        and float(convergence.get("max_rhat", float("inf"))) <= 1.05
    )


def select_robust_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Select highest minimum bulk ESS only among fully suitable candidates."""

    suitable = [row for row in rows if _suitable(row)]
    if not suitable:
        return None
    return sorted(
        suitable,
        key=lambda row: (
            -float(row["qualification"]["convergence"]["min_bulk_ess"]),
            int(row["l"]),
            str(row.get("candidate_signature", "")),
        ),
    )[0]


def tune_hmc_kernel_robust_broad_grid(
    *,
    adapter: Any,
    initial_position: Any,
    config: RobustBroadGridConfig | None = None,
    initial_covariance: Any | None = None,
    negative_hessian: Any | None = None,
    parameter_scales: Any | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    """Run the five-stage robust synthetic broad-grid tuning campaign."""

    cfg = RobustBroadGridConfig() if config is None else config
    if not isinstance(cfg, RobustBroadGridConfig):
        raise TypeError("config must be RobustBroadGridConfig")
    started = time.perf_counter()
    base_signature = stable_adapter_signature(adapter)

    def progress(stage: str, **payload: Any) -> None:
        if progress_callback is None:
            return
        progress_callback(
            stage,
            {
                "schema": "bayesfilter.robust_fixed_mass_broad_grid_progress.v1",
                "stage": stage,
                "elapsed_s": time.perf_counter() - started,
                "candidate_count_completed": len(payload.get("rows", ())),
                "active_l": payload.get("active_l"),
                "status": payload.get("status"),
                "reports_posterior_convergence": False,
                "raw_samples_retained": False,
            },
        )

    progress("campaign_started")
    # Reuse the bounded public mass-preparation prefix. The old narrow Phase-5
    # selector is deliberately bypassed.
    serious_cfg = HMCKernelTuningConfig.serious(
        target_accept_prob=cfg.target_accept_prob,
        acceptance_band=cfg.acceptance_band,
        repair_band=cfg.repair_band,
        target_scope=cfg.target_scope,
        chain_execution_mode=cfg.chain_execution_mode,
        use_xla=cfg.use_xla,
        seed=cfg.mass_preparation_seed,
    )
    try:
        preparation = prepare_operational_windowed_mass_handoff(
            adapter=adapter,
            initial_position=initial_position,
            config=serious_cfg,
            negative_hessian=negative_hessian,
            initial_covariance=initial_covariance,
            parameter_scales=parameter_scales,
            progress_callback=lambda stage, payload: progress(
                stage,
                status=payload.get("final_status"),
            ),
        )
    except Exception as exc:
        return {
            "status": "mass_preparation_failed",
            "base_adapter_signature": base_signature,
            "prepared_status": "operational_mass_preparation_exception",
            "reason": "operational mass preparation raised an exception",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "nonclaims": NONCLAIMS,
        }
    geometry = preparation["geometry"]
    bootstrap = preparation["bootstrap"]
    windowed = preparation["windowed_stage"]
    handoff = preparation
    mass = handoff["adapted_mass_artifact"]
    final_adapter = handoff["final_adapter"]
    phase4_adapter = handoff["phase4_adapter"]
    starts = handoff["initial_position"]
    parameter_names = (
        tuple(str(name) for name in adapter.parameter_names())
        if callable(getattr(adapter, "parameter_names", None))
        else tuple(f"q{index}" for index in range(int(getattr(adapter, "parameter_dim"))))
    )
    policy = HMCAcceptancePolicy(target=cfg.target_accept_prob, practical_region=cfg.acceptance_band, repair_region=cfg.repair_band)
    initial_step = float(windowed.operational_warmup_result.final_kernel_state.epsilon)
    rows = []
    for l in cfg.l_grid:
        progress("candidate_started", active_l=int(l), rows=rows)
        candidate = {"l": int(l), "candidate_signature": stable_config_hash({"l": int(l), "adapter": stable_adapter_signature(final_adapter), "seed": _seed(cfg.root_seed, "candidate", int(l))})}
        try:
            tuned = _tune_one_l(final_adapter, starts, int(l), initial_step, cfg, policy)
            repaired = _repair_one_l(final_adapter, starts, int(l), tuned["step_size"], cfg, policy)
            candidate.update({"dual_averaging": {"step_size": tuned["step_size"], "step_trace": tuned["steps"]}, "repair": {"step_size": repaired["step_size"], "history": repaired["history"]}})
            if (
                repaired["evidence"] is not None
                and repaired["evidence"].evidence_validity == "valid"
                and repaired["evidence"].acceptance_decision
                in {"passed", "inconclusive_evidence"}
                and not repaired["evidence"].candidate_promotion_vetoes
                and not repaired["evidence"].cost_stop_reasons
            ):
                candidate["qualification"] = _qualification(
                    final_adapter,
                    starts,
                    int(l),
                    repaired["step_size"],
                    cfg,
                    policy,
                    phase4_adapter=phase4_adapter,
                    parameter_names=parameter_names,
                )
            else:
                candidate["qualification"] = None
            rows.append(candidate)
        except Exception as exc:  # candidate-local failure; continue the grid
            candidate.update({"status": "candidate_failed", "error_type": type(exc).__name__, "error_message": str(exc), "qualification": None})
            rows.append(candidate)
        progress(
            "candidate_completed",
            active_l=int(l),
            rows=rows,
            status=candidate.get("status", "qualified" if candidate.get("qualification") else "not_qualified"),
        )
    completed = [row for row in rows if row.get("qualification") is not None]
    selected = select_robust_candidate(completed)
    status = "passed" if selected is not None else "no_suitable_candidate"
    result = _jsonable({
        "schema": "bayesfilter.robust_fixed_mass_broad_grid_result.v1",
        "status": status,
        "synthetic_only": True,
        "config": cfg.payload(),
        "base_adapter_signature": base_signature,
        "final_adapter_signature": stable_adapter_signature(final_adapter),
        "mass_artifact_signature": stable_config_hash(mass.signature_payload()),
        "adapted_mass_artifact_payload": mass.to_payload(include_arrays=True),
        "operational_start_lineage": handoff["start_lineage"],
        "candidate_count": len(rows),
        "qualification_count": len(completed),
        "candidates": rows,
        "selected_candidate": selected,
        "selected_kernel": None if selected is None else {
            "step_size": selected["qualification"]["step_size"],
            "num_leapfrog_steps": selected["l"],
            "final_adapter_signature": stable_adapter_signature(final_adapter),
            "mass_artifact_signature": stable_config_hash(mass.signature_payload()),
        },
        "selection_rule": "highest minimum bulk ESS among finite, zero-divergence, R-hat-suitable candidates with passed or valid inconclusive tuning acceptance evidence; tie lower L then signature",
        "elapsed_s": time.perf_counter() - started,
        "prepared_status": windowed.final_status,
        "geometry": geometry.payload(include_mass_arrays=True),
        "bootstrap": bootstrap.payload(),
        "windowed_mass": windowed.payload(),
        "nonclaims": NONCLAIMS,
    })
    progress("campaign_completed", rows=rows, status=status)
    return result


__all__ = [
    "DEFAULT_L_GRID",
    "NONCLAIMS",
    "RobustBroadGridConfig",
    "select_robust_candidate",
    "tune_hmc_kernel_robust_broad_grid",
]
