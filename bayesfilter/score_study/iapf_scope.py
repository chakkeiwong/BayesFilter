"""Validation and accounting for adaptive iAPF realized-particle scopes.

These helpers describe the finite procedure; they do not differentiate its
stopping rule or turn a realized-N comparison into a fixed-cost claim.
"""
from __future__ import annotations

import math


def validate_adaptive_ledger(ledger, *, initial_particles, max_particles, k, tau):
    """Validate the count/action history emitted by an adaptive iAPF run."""
    if type(initial_particles) is not int or type(max_particles) is not int:
        raise ValueError("particle limits must be integers")
    if initial_particles < 2 or max_particles < initial_particles:
        raise ValueError("invalid particle limits")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError("nonempty adaptive ledger required")
    from .iapf_adapter import iteration_decision
    counts, log_values = [], []
    for index, item in enumerate(ledger):
        if not isinstance(item, dict) or item.get("iteration") != index:
            raise ValueError("ledger iterations must be contiguous")
        n = item.get("particles")
        if type(n) is not int or not initial_particles <= n <= max_particles:
            raise ValueError("ledger particle count outside declared bounds")
        if (index == 0 and n != initial_particles) or (counts and n != ledger[index-1]["next_particles"]):
            raise ValueError("particle count differs from the preceding controller decision")
        action = item.get("action")
        if action not in {"fit", "final", "capacity_veto"}:
            raise ValueError("unknown adaptive action")
        if action == "final" and index != len(ledger) - 1:
            raise ValueError("final action must terminate the ledger")
        if action == "capacity_veto" and index != len(ledger) - 1:
            raise ValueError("capacity veto must terminate the ledger")
        value = item.get("log_value")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("finite log value required")
        counts.append(n)
        log_values.append(value)
        expected = iteration_decision(log_values, counts, k=k, tau=tau, max_particles=max_particles)
        if any(item.get(key) != expected_value for key, expected_value in expected.items()):
            raise ValueError("adaptive history disagrees with the declared controller")
    if ledger[-1]["action"] != "final":
        raise ValueError("successful adaptive run must end with final action")
    return {"initial_particle_count": initial_particles,
            "realized_particle_count": counts[-1],
            "maximum_particle_count": max_particles,
            "iterations": len(ledger), "count_history": counts}


def cost_record(*, ledger, fit_calls, fit_optimizer_steps, final_particle_count,
                horizon, offline_seconds,
                final_seconds):
    """Return explicit work accounting for an adaptive run."""
    if any(type(x) is not int or x < 0 for x in
           (fit_calls, fit_optimizer_steps, horizon)) or horizon < 1:
        raise ValueError("work dimensions must be nonnegative integers")
    if type(final_particle_count) is not int or final_particle_count < 2:
        raise ValueError("invalid final particle count")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
               for x in (offline_seconds, final_seconds)):
        raise ValueError("finite nonnegative elapsed times required")
    offline_points = sum(item["particles"] for item in ledger) * horizon
    final_points = final_particle_count * horizon
    if fit_calls != len(ledger)-1 or final_particle_count != ledger[-1]["particles"]:
        raise ValueError("fit/final work does not agree with the adaptive ledger")
    return {"schema": "iapf_work_accounting_v1",
            "offline_filter_passes": len(ledger),
            "offline_fit_calls": fit_calls,
            "offline_optimizer_steps": fit_optimizer_steps,
            "offline_particle_time_points": offline_points,
            "offline_fit_input_particle_time_points": sum(item["particles"] for item in ledger[:-1])*horizon,
            "final_particle_time_points": final_points,
            "offline_wall_seconds": float(offline_seconds),
            "final_wall_seconds": float(final_seconds),
            "total_wall_seconds": float(offline_seconds + final_seconds),
            "fit_and_final_cost_separated": True,
            "fixed_final_count_for_score": True,
            "timing_includes_compilation": True,
            "cost_unit": "particle_time_points_not_FLOPs",
            "fit_cost_reuse": "none_actual_recomputed_work"}


def validate_result_accounting(result, row, settings):
    """Bind counts, fitting data and work to the executed result before selection."""
    from .contracts import digest
    diag, config = result["diagnostics"], row["iapf"]
    from .iapf_adapter import FIT_DIAGNOSTIC_COLUMNS
    if diag.get("fit_objective") != config.get("fit_objective", "density_l2"):
        raise ValueError("fit objective differs from the declared configuration")
    if diag.get("fit_diagnostic_columns") != FIT_DIAGNOSTIC_COLUMNS:
        raise ValueError("fit diagnostic schema mismatch")
    ledger = diag["fit_iterations"]
    counts = validate_adaptive_ledger(ledger, initial_particles=settings["particles"],
        max_particles=config["max_particles"], k=config["k"], tau=config["tau"])
    if len(ledger) > config["max_iterations"]:
        raise ValueError("fit exceeds declared iteration budget")
    for rec in ledger[:-1]:
        if not all(rec.get(key) is True for key in
                   ("density_fit_valid", "density_fit_converged", "coefficient_cast_valid")):
            raise ValueError("invalid or unconverged offline fit in successful evidence")
        details = rec.get("density_fit_diagnostics", [])
        if len(details) != settings["horizon"] or any(len(step) != len(FIT_DIAGNOSTIC_COLUMNS) for step in details):
            raise ValueError("missing per-time fit diagnostics")
        if any(not all(math.isfinite(v) for v in step) or not 0 <= step[4] <= config["max_fit_steps"] or int(step[4]) != step[4]
               for step in details):
            raise ValueError("invalid fit diagnostic or optimizer count")
    if diag["adaptive_count_ledger"] != counts or diag["actual_particle_count"] != counts["realized_particle_count"]:
        raise ValueError("realized particle scope differs from count history")
    if diag["fit"]["particles"] != counts["realized_particle_count"] or diag["fit_digest"] != digest(diag["fit"]):
        raise ValueError("frozen fit/count evidence mismatch")
    expected_fit = {**ledger[-2]["density_fit_parameters"],
                    "fit_theta": config["fit_theta"], "particles": counts["realized_particle_count"]}
    # Nominal parameters are cast to the filter dtype by the executed fitter.
    import tensorflow as tf
    expected_fit["fit_theta"] = tf.constant(config["fit_theta"], tf.as_dtype(settings["dtype"])).numpy().tolist()
    if diag["fit"] != expected_fit or diag["candidate_configuration"] != {"iapf": config}:
        raise ValueError("frozen fit differs from the last fitted coefficients or selected configuration")
    from .conditional_means_tf import model_curves
    if diag["fit_model"] != model_curves(row,settings):
        raise ValueError("fitted and evaluated model coefficients differ")
    expected_data=diag["data_version"]
    if row["model"] == "nonlinear_scalar":
        physical=diag["physical_observations"]
        if digest(physical) != expected_data:
            raise ValueError("physical observation identity mismatch")
        executed=tf.constant(physical,tf.as_dtype(settings["dtype"]))
        if executed.shape != (settings["horizon"],settings["observation_dimension"]):
            raise ValueError("executed observation shape mismatch")
        expected_data=digest(executed.numpy().tolist())
        if diag["executed_observation_digest"] != expected_data:
            raise ValueError("executed observation cast identity mismatch")
    if diag["fit_observation_digest"] != expected_data:
        raise ValueError("fitted and evaluated observations differ")
    seeds = diag["fit_seed_records"]
    final_seeds = {tuple(v) for k, v in seeds.items() if k.startswith("iapf_final")}
    fit_seeds = {tuple(v) for k, v in seeds.items() if k.startswith("iapf_fit")}
    if len(final_seeds) != 4 or len(fit_seeds) != 4*len(ledger) or final_seeds & fit_seeds:
        raise ValueError("fitting/final streams are missing, duplicated or overlap")
    work = diag["work_accounting"]
    expected = cost_record(ledger=ledger, fit_calls=len(ledger)-1,
        fit_optimizer_steps=sum(int(step[4]) for rec in ledger for step in rec.get("density_fit_diagnostics", [])),
        final_particle_count=counts["realized_particle_count"], horizon=settings["horizon"],
        offline_seconds=work["offline_wall_seconds"], final_seconds=work["final_wall_seconds"])
    if work != expected:
        raise ValueError("work accounting differs from the executed count/fit ledger")
    return {**counts, "work_accounting": work, "fit_digest": diag["fit_digest"]}
