"""Reuse checked M13 data1704 grids for a finite-box resolution diagnostic."""
from __future__ import annotations
import json
import math
from pathlib import Path
import time

import tensorflow as tf
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_m13_endpoint_materiality import _load, OLD_SEED, FRESH_SEEDS
from bayesfilter.testing.surrogate_hmc_m13_endpoint_mechanics import serial
from bayesfilter.testing.surrogate_hmc_m13_same_data_posterior import (
    F64, NAMES, PRIOR_SD, RADIUS, precision_screen, summarize,
)
from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_importance_reference

PLAN = "docs/plans/surrogate-hmc-m13-same-data-refine-2026-09-25.md"
BASE = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-same-data-posterior-01"
DEPENDENCIES = (
    "bayesfilter/testing/surrogate_hmc_m13_same_data_posterior.py",
    "bayesfilter/testing/surrogate_hmc_m13_endpoint_materiality.py",
    "bayesfilter/testing/surrogate_hmc_m13_trajectory_reference.py",
    "bayesfilter/testing/surrogate_hmc_m13_wrapped_reference.py",
    "bayesfilter/testing/surrogate_hmc_m13_reference.py",
    "bayesfilter/testing/surrogate_hmc_m13_controls.py",
    "experiments/dpf_implementation/tf_tfp/fixtures/range_bearing_tf.py",
)


def checked_grid(path, *, expected_hash, checksum, particles, seed, order=65):
    if file_hash(path) != expected_hash:
        raise ValueError("saved grid checksum changed")
    grid = json.loads(path.read_text())
    expected = dict(complete=True, invalid_nodes=[], data_seed=1704,
                    observation_checksum=checksum, particles=particles,
                    seed=list(seed), replicates=8)
    if any(grid.get(key) != value for key, value in expected.items()):
        raise ValueError("saved grid scope or validity mismatch")
    axis, logs = grid.get("axis", []), grid.get("log_likelihood", [])
    if len(axis) != order or len(logs) != order:
        raise ValueError("saved grid shape mismatch")
    for i, x in enumerate(axis):
        if not math.isclose(x, -RADIUS+2*RADIUS*i/(order-1), rel_tol=0., abs_tol=4e-15):
            raise ValueError("saved grid nodes mismatch")
    for row in logs:
        if len(row) != order or any(len(cell) != 8 for cell in row):
            raise ValueError("saved grid replicate shape mismatch")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for cell in row for x in cell):
            raise ValueError("saved grid contains invalid likelihood")
    return grid


def refine_grid(grid, evaluate, save_row):
    """Evaluate only newly inserted nodes; retain every prior value exactly."""
    old = grid["axis"]
    nodes = [value for a, b in zip(old[:-1], old[1:]) for value in (a, (a+b)/2)]+[old[-1]]
    order = len(nodes)
    logs = [[None for _ in nodes] for _ in nodes]
    for i, row in enumerate(grid["log_likelihood"]):
        for j, value in enumerate(row):
            logs[2*i][2*j] = list(value)
    sequence = sorted(range(order), key=lambda i: abs(nodes[i]))
    invalid = []
    regions = {name: dict(count=0, minimum_ess=float("inf"), maximum_weight=0.)
               for name in ("nominal", "intermediate", "outer")}
    calls = 0
    for i in sequence:
        records = []
        for j in sequence:
            if logs[i][j] is not None:
                continue
            result = evaluate(tf.constant([nodes[i], nodes[j]], F64), tf.constant(grid["seed"], tf.int32))
            calls += 1
            values = result["log_likelihood"]
            valid = result["valid"] & tf.math.is_finite(values)
            logs[i][j] = serial(values)
            if not bool(tf.reduce_all(valid)):
                invalid.append([i, j])
            extent = max(abs(nodes[i]), abs(nodes[j]))/PRIOR_SD
            region = "nominal" if extent <= 1. else "intermediate" if extent < 3. else "outer"
            ess = float(tf.reduce_min(result["minimum_ess"]))
            weight = float(tf.reduce_max(result["proposal_maximum_normalized_weight"]))
            regions[region]["count"] += 1
            if math.isfinite(ess):
                regions[region]["minimum_ess"] = min(regions[region]["minimum_ess"], ess)
            if math.isfinite(weight):
                regions[region]["maximum_weight"] = max(regions[region]["maximum_weight"], weight)
            records.append(dict(index=[i, j], theta=[nodes[i], nodes[j]],
                log_likelihood=logs[i][j], valid=serial(valid), region=region,
                minimum_ess=serial(result["minimum_ess"]),
                maximum_weight=serial(result["proposal_maximum_normalized_weight"])))
        save_row(i, records, len(invalid))
    return dict(axis=nodes, log_likelihood=logs, invalid_nodes=invalid,
                complete=True, new_evaluations=calls, reused_evaluations=len(old)**2,
                new_node_regions=regions)


def refine_stage(output):
    start = time.monotonic()
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    _, fixture, _ = _load()
    parent = json.loads((BASE/"report.json").read_text())
    snapshot = json.loads((BASE/"run-manifest.json").read_text())
    if not parent.get("source_unchanged"):
        raise ValueError("saved grid run source changed")
    for name in DEPENDENCIES:
        if snapshot["source_hashes"].get(name) != file_hash(ROOT/name):
            raise ValueError("saved grid numerical dependency changed: "+name)
    previous_path = BASE/"m13-same-data-posterior/report.json"
    previous = json.loads(previous_path.read_text())
    data = json.loads((previous_path.parent/"data.json").read_text())
    if (data["data_seed"] != 1704 or data["observation_checksum"] != fixture.observation_checksum
            or data["observations"] != fixture.observations.numpy().tolist()
            or previous["invalid_nodes"] or previous["order"] != 65):
        raise ValueError("saved reference data or validity mismatch")
    write_json(out/"data.json", data)
    specs = [(4096, OLD_SEED)]+[(16384, seed) for seed in FRESH_SEEDS]
    saved = {}
    for n, seed in specs:
        key = f"n{n}-s{seed[0]}"
        saved[key] = checked_grid(previous_path.parent/key/"grid.json",
            expected_hash=previous["diagnostics"][key]["grid_sha256"],
            checksum=fixture.observation_checksum, particles=n, seed=seed)
    evaluators = {n: trajectory_importance_reference(fixture, n, 8, full_support=True)
                  for n in (4096, 16384)}
    grids, diagnostics, all_invalid = {}, {}, []
    for n, seed in specs:
        key = f"n{n}-s{seed[0]}"
        directory = out/key
        directory.mkdir(exist_ok=False)

        def save_row(i, records, invalid_count):
            write_json(directory/f"row-{i:03}.json", records)
            print(key, "row", i, "new", len(records), "invalid", invalid_count,
                  "wall", round(time.monotonic()-start, 2), flush=True)

        refined = refine_grid(saved[key], evaluators[n], save_row)
        path = directory/"grid.json"
        write_json(path, {**saved[key], **refined, "reused_grid_sha256":
                         previous["diagnostics"][key]["grid_sha256"]})
        diagnostics[key] = dict(grid_sha256=file_hash(path),
            reused_evaluations=refined["reused_evaluations"],
            new_evaluations=refined["new_evaluations"],
            new_node_regions=refined["new_node_regions"],
            prior_node_regions=previous["diagnostics"][key]["regions"])
        all_invalid += [dict(group=key, index=ij) for ij in refined["invalid_nodes"]]
        if not refined["invalid_nodes"]:
            grids[key] = tf.constant(refined["log_likelihood"], F64)
    report = dict(plan=PLAN, qualification_scope="finite_box_endpoint_precision_only",
        endpoint_qualified=False, posterior_reference_qualified=False, inference_launched=False,
        data_seed=1704, observation_checksum=fixture.observation_checksum, order=129,
        radius=RADIUS, prior_support="R^2", tail_bound="NOT_ESTABLISHED", dtype="float64",
        jit_compile=True, independent_reference_exception=True,
        baseline_report=str(previous_path), baseline_sha256=file_hash(previous_path),
        metric_names=list(NAMES), diagnostics=diagnostics, invalid_nodes=all_invalid,
        wall_seconds=time.monotonic()-start, ranking_supported=False,
        heuristic_dominance=previous["heuristic_dominance"])
    if all_invalid:
        return report | dict(status="M13_SAME_DATA_REFINED_REFERENCE_INVALID", qualified=False,
            next_action="Existing master: localize invalid new reference nodes.")
    baseline = summarize(grids[f"n4096-s{OLD_SEED[0]}"])
    reference = summarize(tf.concat([grids[f"n16384-s{s[0]}"] for s in FRESH_SEEDS], axis=2))
    screen = precision_screen(baseline[0], reference[0], [baseline, reference])
    report.update(status="M13_SAME_DATA_REFINED_PRECISION_PASSED" if screen["passed"] else
        "M13_SAME_DATA_REFINED_PRECISION_REPAIR_REQUIRED", qualified=screen["passed"], screen=screen,
        baseline=serial(baseline[0]), reference=serial(reference[0]),
        baseline_grid_difference=serial(baseline[0]["metrics"]-baseline[1]["metrics"]),
        reference_grid_difference=serial(reference[0]["metrics"]-reference[1]["metrics"]),
        baseline_grid_difference_mcse=serial(baseline[2]),
        reference_grid_difference_mcse=serial(reference[2]),
        next_action="Existing master: domain sensitivity, scoped controls and public HMC tuning; no admission from this finite-box screen.")
    write_json(out/"summary.json", report)
    return report
