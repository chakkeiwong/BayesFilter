"""Diagnostic calibration of public numerical single-pair tuning decisions.

NumPy/SciPy are independent reference/reporting tools in this module. The
public tuner owns all runtime decisions and qualification receipts.
"""
from collections import Counter
from pathlib import Path
import time

import numpy as np
from scipy import stats
import tensorflow as tf

from ..designs import seed_for
from ..procedures import FrozenTransition, initial_starts
from ..references import analytic
from ..storage import read_json, write_json, write_tensor
from ..targets import ValidationTarget
from .pipeline import check_inventory
from .statistics import binomial_interval


def stationary_reference(design, target, root):
    """One actual transition at each iid anchor; independent of tuning streams."""
    n = design.options["reference_anchors"]
    q = analytic.draw("gaussian", n, seed_for(design.seed, design.design_id, "reference-anchors"), target.parameters)
    step = FrozenTransition(target, chains=n, step_size=design.step_size,
        leapfrog_steps=design.leapfrog_steps, jit_compile=design.device == "gpu")
    observed = step.audit_step(tf.constant(q, tf.float64), tf.constant(
        seed_for(design.seed, design.design_id, "reference-momentum"), tf.int32))
    if not all(bool(tf.reduce_all(tf.math.is_finite(v))) for k, v in observed.items() if k != "is_accepted"):
        raise FloatingPointError("invalid stationary acceptance reference")
    proposal = observed["proposed_state"].numpy()
    p0, p1 = observed["initial_momentum"].numpy(), observed["final_momentum"].numpy()
    expected = analytic.log_density("gaussian", proposal, target.parameters) - analytic.log_density(
        "gaussian", q, target.parameters) + .5*np.sum(p0*p0-p1*p1, axis=1)
    ratios = observed["log_accept_ratio"].numpy()
    tolerance = 1.e-10*(1 + np.abs(expected) + np.sum(p0*p0+p1*p1, axis=1))
    if not np.all(np.abs(ratios-expected) <= tolerance):
        raise ValueError("actual MH ratio disagrees with independent endpoint energies")
    marks = np.exp(np.minimum(ratios, 0.))
    mean = float(marks.mean())
    se = float(marks.std(ddof=1)/np.sqrt(n))
    interval = [max(0., mean-stats.norm.ppf(.975)*se), min(1., mean+stats.norm.ppf(.975)*se)]
    write_tensor(root/"reference_acceptance.tensor", marks)
    result = {"mean": mean, "standard_error": se, "interval": interval, "anchors": n,
        "method": "iid one-transition marks from exact stationary Gaussian positions; normal MC interval",
        "independent_energy_check_passed": True, "reference_used_for_tuning": False,
        "stationary_does_not_equal_fixed_start_window": True}
    write_json(root/"reference.json", result)
    return result


def run(design, root, deadline=None):
    from bayesfilter.inference import (HMCControllerConfig, HMCCandidateExecutionConfig,
        HMCAcceptancePolicy, PrecomputedMassArtifact, bind_hmc_candidate_set_execution, tune_hmc_kernel)
    root = Path(root)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("acceptance calibration deadline")
    target = ValidationTarget("gaussian", design.scenario.parameters, jit_compile=design.device == "gpu")
    # Reference uses a separate adapter instance: FrozenTransition changes its
    # own batch contract to rank2, which must not affect the public scalar lane.
    reference = stationary_reference(design, ValidationTarget("gaussian", design.scenario.parameters,
        jit_compile=design.device == "gpu"), root)
    starts = initial_starts(target, design.scenario.start)
    mass = PrecomputedMassArtifact(position=[0., 0.], covariance=tf.eye(2, dtype=tf.float64),
        factor=tf.eye(2, dtype=tf.float64), adapter_signature=target.adapter_signature(),
        position_role="calibration_origin", covariance_source="declared identity metric; no adaptation")
    rungs = tuple(design.options["evidence_rungs"])
    policy = HMCAcceptancePolicy()
    rows = []
    for replication in range(design.replications):
        if deadline is not None and time.monotonic() >= deadline:
            break
        directory = root/f"replication-{replication:04d}"
        execution = HMCCandidateExecutionConfig(measurement_num_results=design.measurement_draws,
            verification_num_results=design.measurement_draws, num_warmup_steps=8,
            seed=seed_for(design.seed, design.design_id, replication, "tuning"), acceptance_policy=policy,
            target_status_trace_policy="none", use_xla=design.device == "gpu",
            non_xla_reason="explicit CPU acceptance reference" if design.device != "gpu" else None)
        binding = bind_hmc_candidate_set_execution(adapter=target, initial_position=starts,
            mass_artifact=mass, target_scope="inference_validation", scope_id="hmc-acceptance-calibration",
            search_id=f"{design.design_id}-{replication}", epsilon_domain=(design.step_size/2, design.step_size*2),
            repair_factor=2., max_repairs_per_family=0, config=execution,
            target_lineage={"model":"gaussian", "data":None, "prior":target.parameters, "control":"baseline"},
            source_paths=[__file__, str(Path(__file__).parents[1]/"targets.py")])
        search = HMCControllerConfig(primary_l_grid=(design.leapfrog_steps,), initial_epsilon=design.step_size,
            pilot_enabled=False, refinement_rounds=0, max_candidates=1,
            total_budget_units=2*sum(rungs)+1, repair_reserve_units=1, evidence_rungs=rungs,
            max_wall_time_seconds=max(.001, deadline-time.monotonic()) if deadline else design.budget_seconds)
        result = tune_hmc_kernel(adapter=target, initial_position=starts, config=search,
            candidate_set_adapter=binding.typed_adapter, output_dir=directory/"tuning").result
        payload = read_json(directory/"tuning"/"candidate_set_result.json")
        inventory = check_inventory(payload)
        if inventory["failures"]:
            raise ValueError("public search inventory mismatch: " + str(inventory["failures"]))
        row = {"replication":replication, "verified":bool(result.verified_candidate_ids),
            "terminal_state":next(iter(result.candidate_states.values())),
            "completion":result.completion_status, "inventory":inventory,
            "tuning_result":str(directory/"tuning"/"candidate_set_result.json"),
            "looks":[{"stage":o["stage"], "decision":o["observation"].get("decision"),
                "evidence":o["observation"].get("acceptance_evidence"),
                "hard_vetoes":o["observation"].get("hard_vetoes", [])} for o in payload["observations"]]}
        write_json(directory/"assessment.json", row)
        rows.append(row)
    passed = sum(r["verified"] for r in rows)
    complete = len(rows) == design.replications and all(r["completion"] == "complete" for r in rows)
    result = {"finding":"acceptance_rates_estimated" if complete else "incomplete", "rows":rows,
        "planned":design.replications, "completed":len(rows), "verified":passed,
        "qualification_interval":binomial_interval(passed,len(rows)) if rows else None,
        "terminal_states":dict(Counter(r["terminal_state"] for r in rows)), "reference":reference,
        "practical_band":policy.practical_region, "numerical_hmc":True,
        "public_tuner":"tune_hmc_kernel", "automatic_preparation":False,
        "independent_unit":"entire fixed-start public single-pair search; fresh measurement and verification",
        "rhat_or_ess_used_for_tuning":False, "nominal_sequential_coverage_established":False,
        "interpretation":"operational compatibility-screen rates conditional on declared metric and starts"}
    write_json(root/"acceptance.json", result)
    return result
