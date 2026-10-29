"""Independent nested M13 reference refinement; no HMC endpoint admission."""
import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing import surrogate_hmc_m13_posterior_pilot as pilot
from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_importance_reference

PLAN = "docs/plans/surrogate-hmc-m13-posterior-refine-2026-09-24.md"
BASE = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-03/m13-posterior-pilot"
ORDERS = (65, 129)


def load_baseline(path):
    path = Path(path)
    report = json.loads((path/"report.json").read_text())
    for name, digest in report["source_sha256"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError("baseline source changed: "+name)
    if file_hash(path/"data.json") != report["data_sha256"]:
        raise ValueError("baseline data changed")
    if file_hash(path/"grid.json") != report["grid_sha256"]:
        raise ValueError("baseline grid changed")
    grid = json.loads((path/"grid.json").read_text())
    if (report["status"] != "M13_POSTERIOR_REFERENCE_PILOT_COMPLETED"
        or report["valid_node_count"] != pilot.ORDER**2
        or report["particles"] != pilot.PARTICLES or report["replicates"] != pilot.REPLICATES
        or grid["seed"] != list(pilot.SEED)
        or len(grid["axis"]) != pilot.ORDER):
        raise ValueError("baseline scope mismatch")
    logs = tf.constant(grid["log_likelihood"], tf.float64)
    valid = tf.constant(grid["valid"], tf.bool)
    if logs.shape != (pilot.ORDER, pilot.ORDER, pilot.REPLICATES) or valid.shape != logs.shape:
        raise ValueError("baseline grid shape mismatch")
    if not bool(tf.reduce_all(valid & tf.math.is_finite(logs)).numpy()):
        raise ValueError("baseline contains invalid likelihood cells")
    return report, grid


def nested_grid(grid, order):
    old_order = len(grid["axis"])
    if order != 2*old_order-1:
        raise ValueError("only successive nested grids are allowed")
    nodes = []
    for left, right in zip(grid["axis"][:-1], grid["axis"][1:]):
        nodes.extend([left, (left+right)/2])
    nodes.append(grid["axis"][-1])
    logs = [[[None]*pilot.REPLICATES for _ in nodes] for _ in nodes]
    valid = [[[False]*pilot.REPLICATES for _ in nodes] for _ in nodes]
    for i in range(old_order):
        for j in range(old_order):
            logs[2*i][2*j] = list(grid["log_likelihood"][i][j])
            valid[2*i][2*j] = list(grid["valid"][i][j])
    return dict(axis=nodes, log_likelihood=logs, valid=valid, seed=list(pilot.SEED))


def refinement_stage(output):
    from bayesfilter.testing.surrogate_hmc_m13_controls import partition_fixture
    baseline, grid = load_baseline(BASE)
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    fixture = partition_fixture("calibration")
    stored = json.loads((BASE/"data.json").read_text())
    if fixture.observations.numpy().tolist() != stored["observations"] or fixture.model_definition() != stored["model"]:
        raise ValueError("regenerated reference data/model mismatch")
    evaluate = trajectory_importance_reference(fixture, pilot.PARTICLES, pilot.REPLICATES, full_support=True)
    seed = tf.constant(pilot.SEED, tf.int32)
    sources = {str(Path(__file__)): file_hash(Path(__file__)), str(ROOT/PLAN): file_hash(ROOT/PLAN),
               **baseline["source_sha256"]}
    summaries = {}
    new_evaluations, reused = 0, 0
    start = time.monotonic()
    def finite_list(value):
        return [float(v) if math.isfinite(float(v)) else None for v in value.numpy().tolist()]
    for order in ORDERS:
        reused += len(grid["axis"])**2
        grid = nested_grid(grid, order)
        folder = out/f"grid-{order}"
        folder.mkdir()
        invalid = []
        nodes = grid["axis"]
        for i in sorted(range(order), key=lambda k: abs(nodes[k])):
            records = []
            for j in sorted(range(order), key=lambda k: abs(nodes[k])):
                if i % 2 == 0 and j % 2 == 0:
                    continue
                result = evaluate(tf.constant([nodes[i], nodes[j]], tf.float64), seed)
                valid = result["valid"] & tf.math.is_finite(result["log_likelihood"])
                grid["log_likelihood"][i][j] = finite_list(result["log_likelihood"])
                grid["valid"][i][j] = valid.numpy().tolist()
                if not all(grid["valid"][i][j]):
                    invalid.append([i, j])
                records.append(dict(index=[i, j], theta=[nodes[i], nodes[j]],
                    log_likelihood=grid["log_likelihood"][i][j], valid=grid["valid"][i][j],
                    minimum_ess=finite_list(result["minimum_ess"]),
                    maximum_weight=finite_list(result["proposal_maximum_normalized_weight"]),
                    fit_first=finite_list(result["proposal_first_diagnostics"]),
                    fit_full=finite_list(result["proposal_full_diagnostics"])))
                new_evaluations += 1
            write_json(folder/f"row-{i:03}.json", records)
            if i % 8 == 0:
                print("M13 refine", order, "row", i, "new evaluations", new_evaluations,
                      "invalid", len(invalid), "wall", time.monotonic()-start, flush=True)
        write_json(folder/"grid.json", grid)
        summary = (dict(status="INVALID_GRID_NO_POSTERIOR_SUMMARY", complete_valid_grid=False,
                        posterior_reference_qualified=False) if invalid else
                   pilot.summarize_grid(grid["log_likelihood"], grid["valid"]))
        summary["invalid_nodes"] = invalid
        if not invalid:
            changes, metrics = summary["fine_minus_coarse"], summary["fine"]["metrics"]
            summary["mean_changes_in_posterior_sd"] = [changes[k]/metrics[2+k] for k in range(2)]
            summary["relative_sd_changes"] = [changes[2+k]/metrics[2+k] for k in range(2)]
        summaries[str(order)] = summary
        write_json(folder/"summary.json", summary)
        if invalid:
            break
    if any(file_hash(Path(name)) != digest for name, digest in sources.items()):
        raise ValueError("source changed during nested reference refinement")
    return dict(status=("M13_POSTERIOR_REFINEMENT_DIAGNOSTIC_COMPLETED"
        if all(s["complete_valid_grid"] for s in summaries.values()) else
        "M13_POSTERIOR_REFINEMENT_REPAIR_REQUIRED"), summaries=summaries,
        qualified=False, posterior_reference_qualified=False, endpoint_qualified=False,
        inference_launched=False, all_models_validated=False, continuation_veto=False,
        baseline_report=str(BASE/"report.json"), baseline_report_sha256=file_hash(BASE/"report.json"),
        source_sha256=sources, data_sha256=baseline["data_sha256"],
        seed=list(pilot.SEED), particles=pilot.PARTICLES, replicates=pilot.REPLICATES,
        new_evaluations=new_evaluations, reused_nodes_across_levels=reused,
        dtype="float64", independent_reference_exception=True, jit_compile=True, plan=PLAN,
        posterior_comparisons="NOT_RUN", ranking_supported=False,
        next_action="Existing master: inspect resolution, qualify independent seed/sample-size and tails, then endpoint and surrogate-HMC tests.")
