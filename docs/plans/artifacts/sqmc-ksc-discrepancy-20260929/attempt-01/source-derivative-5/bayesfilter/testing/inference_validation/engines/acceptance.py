"""Diagnostic operating characteristics of the actual candidate evidence screen.

Independent Beta marks or stationary persistent-refresh Beta marks have a known
mean. Synthetic moving states exercise evidence arithmetic only. The frozen
route measures stationary HMC windows; the prepared route uses the public
tuner and an independent stationary reference. These roles remain distinct.
"""
from __future__ import annotations

from collections import Counter
import time
import numpy as np

from ..designs import seed_for
from ..storage import write_json
from .statistics import binomial_interval
from ..targets import ValidationTarget
from ..references import analytic


def acceptance_marks(rng, draws, mean, concentration, persistence):
    values = rng.beta(mean * concentration, (1-mean) * concentration, size=(draws, 4))
    refresh = rng.random((draws, 4)) >= persistence
    for i in range(1, draws):
        values[i] = np.where(refresh[i], values[i], values[i-1])
    return values


def run(design, root, deadline=None):
    if design.scenario.route == "prepared":
        from .acceptance_hmc import run as run_public_search
        return run_public_search(design, root, deadline)
    if design.scenario.route == "frozen":
        return run_actual_hmc(design, root, deadline)
    from bayesfilter.inference.hmc_candidate_set_tuning import (
        HMCCandidateSetScope, HMCControllerConfig, HMCTuningCandidateSetController,
    )
    from bayesfilter.inference.hmc_candidate_decisions import HMCCandidateDecision
    from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy, evaluate_hmc_acceptance_evidence

    mean = design.options["acceptance_mean"]
    concentration = design.options["acceptance_concentration"]
    persistence = design.options["acceptance_persistence"]
    policy = HMCAcceptancePolicy()
    rows = []
    for replication in range(design.replications):
        if deadline is not None and time.monotonic() >= deadline:
            break
        looks = []
        scope = HMCCandidateSetScope(scope_id="acceptance-calibration", search_id=str(replication),
            target_signature="synthetic-marks", mass_signature="not-numerical",
            coordinate_system="diagnostic", start_bank_signature="synthetic-states",
            warmup_protocol="no-warmup", max_repairs_per_family=0)
        config = HMCControllerConfig(primary_l_grid=(3,), initial_epsilon=.5,
            total_budget_units=20, repair_reserve_units=1,
            evidence_rungs=tuple(design.options["evidence_rungs"]))

        def observe(work, candidate):
            rng = np.random.default_rng(seed_for(design.seed, design.design_id, replication, work.work_item_id))
            n = design.measurement_draws * work.evidence_multiplier
            marks = acceptance_marks(rng, n, mean, concentration, persistence)
            accepted = rng.random((n,4)) < marks
            proposals = rng.normal(size=(n,4,2))
            samples = np.cumsum(proposals * accepted[...,None], axis=0)
            evidence = evaluate_hmc_acceptance_evidence(samples=samples,
                log_accept_ratio=np.log(marks), is_accepted=accepted, policy=policy)
            decision = HMCCandidateDecision.from_evidence(evidence)
            looks.append({"stage":work.stage,"rung":work.evidence_rung,
                "evidence":evidence.payload(),"decision":decision.payload()})
            return {**decision.payload(),"acceptance":evidence.pooled_mean,
                    "stream_id":work.work_item_id}

        result = HMCTuningCandidateSetController(scope, config).run(observe)
        row = {"replication":replication,"verified":bool(result.verified_candidate_ids),
               "terminal_state":next(iter(result.candidate_states.values())),"looks":looks}
        write_json(root / f"replication-{replication:04d}.json", row)
        rows.append(row)
    states = Counter(r["terminal_state"] for r in rows)
    passed = sum(r["verified"] for r in rows)
    result = {"finding":"acceptance_rates_estimated" if len(rows)==design.replications else "incomplete",
        "planned":design.replications,"completed":len(rows),"verified":passed,
        "qualification_interval":binomial_interval(passed,len(rows)) if rows else None,
        "terminal_states":dict(states),"true_mean":mean,"persistence":persistence,
        "concentration":concentration,"practical_band":policy.practical_region,
        "true_mean_in_practical_band":policy.practical_region[0] <= mean <= policy.practical_region[1],
        "independent_unit":"entire single-pair controller search with fresh measurement and verification rungs",
        "interpretation":"operational compatibility-screen rates, not a test of strict mean-band membership",
        "numerical_hmc":False,"nominal_sequential_coverage_established":False,
        "all_member_retention_unchanged":True,"rows":rows}
    write_json(root / "acceptance.json", result)
    return result


def run_actual_hmc(design, root, deadline=None):
    """Measure the public HMC transition from exact stationary Gaussian starts.

    This deliberately bypasses preparation and candidate admission. It answers
    how the compatibility statistic behaves for a known stationary target;
    it cannot validate automatic geometry or posterior convergence.
    """
    import tensorflow as tf
    from bayesfilter.testing.inference_validation.procedures import FrozenTransition
    from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy, evaluate_hmc_acceptance_evidence
    target = ValidationTarget("gaussian", {"scale": design.scenario.parameters.get("scale", 1.)}, jit_compile=design.device == "gpu")
    chains = 4
    steps = design.draws
    policy = HMCAcceptancePolicy()
    transition = FrozenTransition(target, chains=chains, step_size=design.step_size,
        leapfrog_steps=design.leapfrog_steps, control=design.scenario.control,
        jit_compile=design.device == "gpu")
    rows = []
    for replication in range(design.replications):
        if deadline is not None and time.monotonic() >= deadline:
            break
        q = tf.constant(analytic.draw("gaussian", chains,
            seed_for(design.seed, design.design_id, replication, "stationary-start"), target.parameters), tf.float64)
        states, ratios, accepted = [], [], []
        healthy = True
        for index in range(steps):
            if deadline is not None and time.monotonic() >= deadline:
                break
            observed = transition.audit_step(q, tf.constant(
                seed_for(design.seed, design.design_id, replication, index, "transition"), tf.int32))
            q = observed["state"]
            ratio = observed["log_accept_ratio"]
            state = observed["state"]
            accepted_step = observed["is_accepted"]
            healthy = healthy and bool(tf.reduce_all(tf.math.is_finite(state))) and bool(tf.reduce_all(tf.math.is_finite(ratio)))
            if not healthy:
                break
            states.append(state)
            ratios.append(ratio)
            accepted.append(accepted_step)
        if len(states) != steps or not healthy:
            rows.append({"replication": replication, "status": "invalid", "completed_steps": len(states),
                         "reason": "deadline_or_nonfinite_transition"})
            continue
        samples = tf.stack(states, axis=0)
        log_ratios = tf.stack(ratios, axis=0)
        is_accepted = tf.stack(accepted, axis=0)
        evidence = evaluate_hmc_acceptance_evidence(samples=samples, log_accept_ratio=log_ratios,
            is_accepted=is_accepted, policy=policy)
        rows.append({"replication": replication, "status": "assessed",
                     "evidence": evidence.payload(),
                     "acceptance": evidence.pooled_mean,
                     "realized_acceptance": float(tf.reduce_mean(tf.cast(is_accepted, tf.float64))),
                     "steps": steps, "chains": chains})
    assessed = [r for r in rows if r["status"] == "assessed"]
    result = {"finding": "actual_hmc_acceptance_measured" if len(assessed) == design.replications else "incomplete",
        "planned": design.replications, "completed": len(rows), "assessed": len(assessed), "rows": rows,
        "step_size": design.step_size, "leapfrog_steps": design.leapfrog_steps,
        "target": {"law": "gaussian", "parameters": target.parameters},
        "stationary_initialization": "independent exact analytic draws",
        "independent_unit": "complete stationary four-chain HMC experiment",
        "numerical_hmc": True, "automatic_preparation": False,
        "rhat_or_ess_used_for_tuning": False, "posterior_convergence_claim": False,
        "screen_role": "descriptive calibration of the actual acceptance evidence arithmetic",
        "interpretation": "stationary acceptance is separate from fixed-start candidate-screen behavior"}
    write_json(root / "acceptance.json", result)
    return result
