"""Diagnostic known-law calibration through the real candidate-set controller.

These synthetic trial vectors have exact finite-trial means. They cannot issue
numerical artifacts or validate posterior inference on an actual model.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
from bayesfilter.inference.hmc_acceptance_statistics import replicated_acceptance_statistics
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCControllerConfig, HMCCandidateSetScope, HMCTuningCandidateSetController,
)


CELLS = ("interior", "heterogeneous_valid", "misleading_pooled", "preferred_lower",
         "preferred_upper", "qualification_lower", "outside_qualification", "rare_extremes",
         "opposite_l_repairs", "persistent_0995", "initial_transient",
         "qualification_upper", "inside_lower", "inside_upper", "outside_upper",
         "persistent_0", "persistent_08", "persistent_095", "persistent_negative_08",
         "shared_starts", "opposed_starts", "rare_two_sided", "constant_law",
         "temporal_stationary", "temporal_opposing", "temporal_single_start", "temporal_nonlinear")

_RHO = {"persistent_0": 0., "persistent_08": .8, "persistent_095": .95,
        "persistent_0995": .995, "persistent_negative_08": -.8, "initial_transient": .995}
_EVENTS = ("false_verified", "search_direction_error", "verification_assertion_error",
           "useful_delivery", "false_preparation", "false_temporal_alert", "temporal_detection")


def window_law(cell):
    """Exact means of four partition windows, indexed by start then window."""
    flat, rise, fall = (.7,)*4, (.5,.5,.9,.9), (.9,.9,.5,.5)
    return {"temporal_stationary": (flat,)*4,
            "temporal_opposing": (rise, fall, rise, fall),
            "temporal_single_start": (rise, flat, flat, flat),
            "temporal_nonlinear": ((.5,.9,.9,.5),)*4}.get(cell)


def truth(cell, epsilon, steps, horizon, discarded_prefix=0):
    if cell == "heterogeneous_valid": return (.60, .68, .72, .80)
    if cell == "misleading_pooled": return (.40, .80, .80, .80)
    if cell == "preferred_lower": return (.65,)*4
    if cell == "preferred_upper": return (.75,)*4
    if cell == "qualification_lower": return (.55, .75, .75, .75)
    if cell == "qualification_upper": return (.85, .65, .65, .65)
    if cell == "inside_lower": return (.56, .75, .75, .75)
    if cell == "inside_upper": return (.84, .65, .65, .65)
    if cell == "outside_upper": return (.86, .65, .65, .65)
    if cell == "outside_qualification": return (.54, .75, .75, .75)
    if cell == "rare_extremes": return (.703,)*4
    if cell == "opposite_l_repairs":
        # Low/high acceptance parents both move toward epsilon=.65 or2.6.
        center = .65 if steps == 1 else 2.6
        value = max(.1, min(.95, .7-.2*math.log2(epsilon/center)))
        return (value,)*4
    if cell == "initial_transient":
        return (.7+.25*sum(.995**i for i in range(discarded_prefix, discarded_prefix+horizon))/horizon,)*4
    windows = window_law(cell)
    if windows is not None:
        counts = [horizon*(i+1)//4-horizon*i//4 for i in range(4)]
        return tuple(math.fsum(n*m for n,m in zip(counts,row))/horizon for row in windows)
    if cell not in CELLS: raise ValueError("unknown calibration cell")
    return (.7,)*4


@lru_cache(maxsize=16)
def _generator(repetitions, horizon, cell, discarded_prefix):
    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32), tf.TensorSpec([4], tf.float64)],
                 autograph=False, jit_compile=True)
    def generate(seed, means):
        if cell in _RHO:
            uniforms = tf.random.stateless_uniform([repetitions, horizon+discarded_prefix, 4], seed, dtype=tf.float64)
            flips = tf.where(uniforms < (1.-_RHO[cell])/2., -tf.ones_like(uniforms), tf.ones_like(uniforms))
            if cell == "initial_transient":
                starts = tf.ones([repetitions, 1, 4], tf.float64)
            else:
                u = tf.random.stateless_uniform([repetitions, 1, 4], tf.random.experimental.stateless_fold_in(seed, 1), dtype=tf.float64)
                starts = tf.where(u < .5, -tf.ones_like(u), tf.ones_like(u))
            # First observed state is the declared initial sign; rho^(t) is the
            # exact transient expectation, rather than stationary .70.
            signs = starts*tf.math.cumprod(tf.concat([tf.ones_like(flips[:, :1]), flips[:, 1:]], axis=1), axis=1)
            return .7 + .25*tf.reduce_mean(signs[:, discarded_prefix:], axis=1)
        uniforms = tf.random.stateless_uniform([repetitions, 4], seed, dtype=tf.float64)
        if cell == "rare_extremes":
            return tf.where(uniforms < .01, tf.ones_like(uniforms), .7*tf.ones_like(uniforms))
        if cell == "rare_two_sided":
            # P(0)=.003 and P(1)=.007 preserve mean .70 exactly.
            return tf.where(uniforms < .003, tf.zeros_like(uniforms),
                            tf.where(uniforms < .01, tf.ones_like(uniforms), .7*tf.ones_like(uniforms)))
        if cell == "constant_law":
            return tf.broadcast_to(means[None, :], [repetitions, 4])
        if cell in {"shared_starts", "opposed_starts"}:
            shared = tf.where(uniforms[:, :1] < .5, -.1, .1)
            signs = tf.constant([1., -1., 1., -1.] if cell == "opposed_starts" else [1.]*4, tf.float64)
            return means[None, :] + tf.cast(shared, tf.float64)*signs
        amplitude = tf.minimum(tf.minimum(means, 1.-means), .10)
        return means[None, :] + amplitude[None, :]*tf.where(uniforms < .5, -tf.ones_like(uniforms), tf.ones_like(uniforms))
    return generate


@lru_cache(maxsize=16)
def _window_generator(repetitions, horizon, cell):
    means = window_law(cell)
    counts = [horizon*(i+1)//4-horizon*i//4 for i in range(4)]
    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], autograph=False, jit_compile=True)
    def generate(seed):
        u = tf.random.stateless_uniform([repetitions, 4, 4], seed, dtype=tf.float64)
        noise = tf.where(u < .5, -tf.ones_like(u), tf.ones_like(u))*.02
        windows = tf.constant(means, tf.float64)[None, :, :] + noise
        scores = tf.reduce_sum(windows*tf.constant(counts, tf.float64), axis=2)/float(horizon)
        return scores, windows
    return generate


def _seed(namespace, cell, search, candidate, stage):
    encoded = json.dumps([namespace, cell, search, candidate, stage]).encode()
    digest = hashlib.sha256(encoded).digest()
    return tuple(int.from_bytes(digest[i:i+4], "big") & 0x7fffffff for i in (0, 4))


def search_once(*, cell, index, namespace, policy, rungs, initial_candidate_count=2):
    if type(initial_candidate_count) is not int or not 2 <= initial_candidate_count <= policy.max_candidates:
        raise ValueError("actual cohort must contain between two and the declared candidate cap")
    if cell == "opposite_l_repairs" and initial_candidate_count != 2:
        raise ValueError("opposite-L repair law has exactly two declared initial families")
    scope = HMCCandidateSetScope(scope_id=f"calibration-{cell}-{index}", search_id="known-law-v1",
        target_signature="finite-trial-"+cell, mass_signature="not_applicable",
        coordinate_system="ordinary", start_bank_signature="four_labeled_starts",
        warmup_protocol=policy.identity, epsilon_domain=(.05, 5.), repair_factor=2., max_repairs_per_family=2)
    ls = (1, 3) if initial_candidate_count == 2 else tuple(range(1, initial_candidate_count+1))
    search = HMCControllerConfig(primary_l_grid=ls, epsilon_by_l=tuple((l, (1.3,)) for l in ls),
        max_candidates=policy.max_candidates, total_budget_units=policy.max_candidates*12,
        repair_reserve_units=policy.max_candidates*2, evidence_rungs=rungs,
        replicated_acceptance_policy=policy)
    observations, streams = [], {}
    def observe(work, candidate):
        key = candidate.candidate_id, work.stage
        means = truth(cell, candidate.epsilon, candidate.leapfrog_steps, policy.trial_num_results, policy.discarded_prefix)
        if key not in streams:
            seed = _seed(namespace, cell, index, candidate.candidate_id, work.stage)
            if window_law(cell) is not None:
                scores, windows = _window_generator(policy.max_repetitions, policy.trial_num_results, cell)(tf.constant(seed, tf.int32))
            else:
                scores = _generator(policy.max_repetitions, policy.trial_num_results, cell, policy.discarded_prefix)(
                    tf.constant(seed, tf.int32), tf.constant(means, tf.float64))
                windows = None
            streams[key] = seed, scores, windows
        seed, scores, windows = streams[key]
        result = replicated_acceptance_statistics(scores[:work.trial_range[1]], policy=policy,
            stage=work.stage, evidence_rungs=rungs,
            window_scores=windows[:work.trial_range[1]] if windows is not None else None)
        decision = result["acceptance_decision"]
        pooled = math.fsum(w*m for w, m in zip(policy.start_weights, means))
        wrong_direction = ((decision == "repair_step_lower" and pooled >= policy.preferred_region[0])
                           or (decision == "repair_step_higher" and pooled <= policy.preferred_region[1]))
        qualified = all(policy.qualification_region[0] <= m <= policy.qualification_region[1] for m in means)
        temporal = result.get("temporal")
        expected_alerts = ([any(abs(row[b]-row[a]) > policy.temporal_tolerance
                               for a in range(4) for b in range(a+1,4)) for row in window_law(cell)]
                           if temporal is not None else None)
        alerts = temporal["supported_material_change_by_start"] if temporal else None
        observations.append({"stage": work.stage, "decision": decision, "candidate_id": candidate.candidate_id,
            "epsilon": candidate.epsilon, "l": candidate.leapfrog_steps, "truth": means,
            "qualified_truth": qualified, "wrong_direction": wrong_direction, "trial_range": work.trial_range,
            "mean": result["pooled_mean"], "seed": seed,
            "false_preparation": decision == "inconclusive_preparation" and qualified,
            "temporal_alerts": alerts, "expected_temporal_alerts": expected_alerts})
        return {"decision": decision, "acceptance": result["pooled_mean"],
                "trial_range": work.trial_range, "draw_range": (0, 0),
                "evidence_unit": "independent_fixed_horizon_trial", "seed_lineage": seed,
                "stream_id": work.work_item_id}
    result = HMCTuningCandidateSetController(scope, search).run(observe)
    verified = set(result.verified_candidate_ids)
    false_verified = any(o["candidate_id"] in verified and not o["qualified_truth"] for o in observations)
    wrong_measurement = any(o["wrong_direction"] and o["stage"] != "verification" for o in observations)
    wrong_verification = any(o["wrong_direction"] and o["stage"] == "verification" for o in observations)
    initial = {c.candidate_id for c in result.candidates if c.parent_candidate_id is None}
    easy = cell in {"interior", "heterogeneous_valid", "preferred_lower", "preferred_upper", "rare_extremes",
                   "rare_two_sided", "shared_starts", "opposed_starts", "constant_law"} or cell.startswith("persistent_") or window_law(cell) is not None
    # All repaired L families must survive, not just one favorable child.
    delivered = (initial <= verified if easy else
                 {o["l"] for o in observations if o["candidate_id"] in verified} == {1, 3}
                 if cell == "opposite_l_repairs" else None)
    temporal_observations = [o for o in observations if o["temporal_alerts"] is not None]
    false_temporal = (any(any(a and not e for a,e in zip(o["temporal_alerts"],o["expected_temporal_alerts"]))
                          for o in temporal_observations) if temporal_observations else None)
    temporal_detection = (all(any(all(a or not e for a,e in zip(o["temporal_alerts"],o["expected_temporal_alerts"]))
                                  for o in temporal_observations if o["candidate_id"] == cid)
                              for cid in initial)
                          if temporal_observations and any(temporal_observations[0]["expected_temporal_alerts"]) else None)
    observed_vectors = sum(b-a for o in observations for a,b in [o["trial_range"]])
    return {"index": index, "false_verified": false_verified,
        "search_direction_error": wrong_measurement,
        "verification_assertion_error": false_verified or wrong_verification,
        "useful_delivery": delivered, "verified_count": len(verified),
        "false_preparation": any(o["false_preparation"] for o in observations),
        "false_temporal_alert": false_temporal, "temporal_detection": temporal_detection,
        "initial_candidate_count": initial_candidate_count,
        "candidate_count": len(result.candidates), "candidate_states": result.candidate_states,
        "complete_trials": observed_vectors, "observed_trial_vectors": observed_vectors,
        "generated_trial_vectors": len(streams)*policy.max_repetitions,
        "observations": observations, "artifact_authority": False}


def _binomial_cdf(count, total, probability):
    if count < 0: return 0.
    if count >= total or probability == 0: return 1.
    if probability == 1: return 0.
    logs = [math.lgamma(total+1)-math.lgamma(k+1)-math.lgamma(total-k+1)
            +k*math.log(probability)+(total-k)*math.log1p(-probability) for k in range(count+1)]
    maximum = max(logs)
    return min(1., math.exp(maximum)*math.fsum(math.exp(x-maximum) for x in logs))


def binomial_interval(count, total, alpha):
    """Equal-tail Clopper--Pearson by direct binomial inversion (diagnostic)."""
    if not total: return (0., 1.)
    def root(k, target):
        low, high = 0., 1.
        for _ in range(56):
            middle = (low+high)/2
            if _binomial_cdf(k, total, middle) > target: low = middle
            else: high = middle
        return low, high
    # Invert small tails, avoiding loss of precision from a CDF near one.
    lower = (0. if count == 0 else math.exp(math.log(alpha/2)/total)
             if count == total else 1.-root(total-count, alpha/2)[1])
    upper = (1. if count == total else -math.expm1(math.log(alpha/2)/total)
             if count == 0 else root(count, alpha/2)[1])
    return lower, upper


def run_calibration(configuration, output, *, manifest=None):
    if configuration.get("schema") not in {"bayesfilter.acceptance_decision_calibration_config.v1",
                                           "bayesfilter.acceptance_decision_calibration_config.v2"}:
        raise ValueError("unsupported calibration configuration")
    policy = HMCReplicatedAcceptancePolicy.from_payload(configuration["policy"])
    cells, methods = tuple(configuration["cells"]), tuple(configuration["methods"])
    if not cells or len(set(cells)) != len(cells) or any(c not in CELLS for c in cells):
        raise ValueError("invalid calibration cells")
    repetitions = configuration["searches_per_cell"]
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("positive searches_per_cell required")
    if (not methods or len(set(methods)) != len(methods)
            or any(m not in {"bounded_betting_mixture_v1", "bounded_hoeffding_rungs_v1"} for m in methods)):
        raise ValueError("invalid calibration methods")
    if not 0 < configuration["rate_family_alpha"] < 1 or not 0 < configuration["wall_seconds"] <= 7200:
        raise ValueError("explicit bounded error and wall allocations required")
    rungs = tuple(configuration["evidence_rungs"])
    initial_candidates = configuration.get("initial_candidate_count", 2)
    if type(initial_candidates) is not int or not 2 <= initial_candidates <= policy.max_candidates:
        raise ValueError("invalid actual candidate cohort")
    if "opposite_l_repairs" in cells and initial_candidates != 2:
        raise ValueError("opposite-L fixture requires two initial candidates")
    from bayesfilter.inference.hmc_acceptance_protocol import replicated_evidence_preflight
    replicated_evidence_preflight(policy, evidence_rungs=rungs, candidate_cap=policy.max_candidates,
                                 leapfrog_steps=(1, initial_candidates))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    if manifest is not None:
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (root/"configuration.json").write_text(json.dumps(configuration, indent=2)+"\n")
    started = time.monotonic()
    summaries = []
    stopped = False
    # Simultaneous rate uncertainty across every predeclared method/cell/event.
    events = _EVENTS if configuration["schema"].endswith(".v2") else _EVENTS[:4]
    rate_alpha = configuration["rate_family_alpha"]/(len(cells)*len(methods)*len(events))
    for method in methods:
        declared = replace(policy, method=method)
        for cell in cells:
            cell_started = time.monotonic()
            rows = []
            path = root/f"{method}-{cell}.jsonl"
            with path.open("x") as stream:
                for index in range(repetitions):
                    if time.monotonic()-started >= configuration["wall_seconds"]:
                        stopped = True
                        break
                    row = search_once(cell=cell, index=index, namespace=configuration["seed_namespace"],
                                      policy=declared, rungs=rungs, initial_candidate_count=initial_candidates)
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                    stream.flush()
                    rows.append(row)
            rates = {}
            for event in events:
                selected = [r[event] for r in rows if r[event] is not None]
                count = sum(selected)
                rates[event] = {"count": count, "observed_denominator": len(selected),
                    "planned_denominator": repetitions,
                    "simultaneous_interval": binomial_interval(count, len(selected), rate_alpha) if selected else None}
            summaries.append({"method": method, "cell": cell, "completed": len(rows),
                "planned": repetitions, "rates": rates, "raw": str(path),
                "wall_seconds": time.monotonic()-cell_started,
                "complete_trials": sum(r["complete_trials"] for r in rows),
                "observed_trial_vectors": sum(r["observed_trial_vectors"] for r in rows),
                "generated_trial_vectors": sum(r["generated_trial_vectors"] for r in rows)})
            (root/"progress.json").write_text(json.dumps({"cells": summaries, "wall_seconds": time.monotonic()-started}, indent=2)+"\n")
            if stopped: break
        if stopped: break
    result = {"schema": "bayesfilter.replicated_acceptance_calibration.v2", "cells": summaries,
        "wall_seconds": time.monotonic()-started, "budget_stopped": stopped,
        "planned_cells": len(cells)*len(methods), "rate_one_interval_alpha": rate_alpha,
        "rate_events": events, "initial_candidate_count": initial_candidates,
        "cost_semantics": "observed vectors and eagerly generated vectors are distinct; neither measures HMC runtime",
        "artifact_authority": False, "posterior_correctness": "not_tested",
        "independent_unit": "one complete tuning search; all candidate outcomes counted",
        "numerical_guard_status": "experimental_reference_tested_not_formal_hardware_bound"}
    (root/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result
