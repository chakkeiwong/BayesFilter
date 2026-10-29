"""Complete the preserved M13 posterior grid; diagnostic, never HMC admission."""
import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing import surrogate_hmc_m13_posterior_pilot as pilot
from bayesfilter.testing import surrogate_hmc_m13_posterior_refine as refine
from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_importance_reference

PLAN = "docs/plans/surrogate-hmc-m13-posterior-completion-2026-09-24.md"
PARTIAL = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-extension-04/m13-posterior-refine"
ORDER = 129
MISSING_ROWS = (0, 1, 127, 128)


def missing_cells():
    return {(i, j) for i in MISSING_ROWS for j in range(ORDER)
            if i % 2 or j % 2}


def merge_rows(grid, rows):
    """Validate each coordinate, replicate and unique index before reuse."""
    expected = {(i, j) for i in range(ORDER) for j in range(ORDER)
                if i % 2 or j % 2} - missing_cells()
    seen = set()
    for records in rows:
        for record in records:
            index = tuple(record["index"])
            if index in seen or index not in expected:
                raise ValueError("duplicate or unexpected preserved cell")
            i, j = index
            if record["theta"] != [grid["axis"][i], grid["axis"][j]]:
                raise ValueError("preserved coordinate mismatch")
            values = record["log_likelihood"]
            if (len(values) != pilot.REPLICATES or
                    record["valid"] != [True]*pilot.REPLICATES or
                    any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
                raise ValueError("invalid preserved cell")
            grid["log_likelihood"][i][j] = list(values)
            grid["valid"][i][j] = list(record["valid"])
            seen.add(index)
    if seen != expected:
        raise ValueError("incomplete preserved cells")
    return seen


def load_partial():
    baseline, grid33 = refine.load_baseline(refine.BASE)
    previous = PARTIAL.parent
    report = json.loads((previous/"report.json").read_text())
    manifest = json.loads((previous/"run-manifest.json").read_text())
    if (not report["source_unchanged"] or
            report["stages"]["m13-posterior-refine"]["execution"] != "JOB_CPU_BUDGET_LIMIT"):
        raise ValueError("unexpected partial-run status")
    numerical_sources = [str(Path(refine.__file__).relative_to(ROOT))]
    numerical_sources += [str(Path(name).relative_to(ROOT)) for name in baseline["source_sha256"]]
    for name in numerical_sources:
        if file_hash(ROOT/name) != manifest["source_hashes"][name]:
            raise ValueError("partial numerical source changed: "+name)
    path = PARTIAL/"grid-65/grid.json"
    grid65 = json.loads(path.read_text())
    expected = refine.nested_grid(grid33, 65)
    logs = tf.constant(grid65["log_likelihood"], tf.float64)
    valid = tf.constant(grid65["valid"], tf.bool)
    if (logs.shape != (65, 65, pilot.REPLICATES) or valid.shape != logs.shape
            or grid65["seed"] != list(pilot.SEED) or grid65["axis"] != expected["axis"]
            or not bool(tf.reduce_all(valid & tf.math.is_finite(logs)))):
        raise ValueError("invalid preserved 65-grid")
    for i in range(33):
        for j in range(33):
            if grid65["log_likelihood"][2*i][2*j] != grid33["log_likelihood"][i][j]:
                raise ValueError("preserved nested value changed")
    grid = refine.nested_grid(grid65, ORDER)
    paths = sorted((PARTIAL/"grid-129").glob("row-*.json"))
    if {p.name for p in paths} != {f"row-{i:03d}.json" for i in range(2, 127)}:
        raise ValueError("preserved row set mismatch")
    hashes = {str(p): file_hash(p) for p in [path, *paths, previous/"run-manifest.json", previous/"report.json"]}
    seen = merge_rows(grid, (json.loads(p.read_text()) for p in paths))
    return baseline, grid, hashes, seen


def completion_stage(output):
    from bayesfilter.testing.surrogate_hmc_m13_controls import partition_fixture
    baseline, grid, saved_hashes, seen = load_partial()
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    fixture = partition_fixture("calibration")
    stored = json.loads((refine.BASE/"data.json").read_text())
    if (fixture.observations.numpy().tolist() != stored["observations"] or
            fixture.model_definition() != stored["model"]):
        raise ValueError("reference data/model changed")
    sources = {**baseline["source_sha256"],
        str(Path(__file__)): file_hash(Path(__file__)),
        str(ROOT/PLAN): file_hash(ROOT/PLAN),
        str(Path(refine.__file__)): file_hash(Path(refine.__file__))}
    write_json(out/"reuse-manifest.json", dict(preserved_sha256=saved_hashes,
        source_sha256=sources, data_sha256=baseline["data_sha256"],
        seed=list(pilot.SEED), particles=pilot.PARTICLES, replicates=pilot.REPLICATES))
    evaluate = trajectory_importance_reference(fixture, pilot.PARTICLES, pilot.REPLICATES, full_support=True)
    seed = tf.constant(pilot.SEED, tf.int32)
    start = time.monotonic()
    invalid, minimum_ess, maximum_weight = [], float("inf"), 0.
    def finite_list(value):
        return [float(v) if math.isfinite(float(v)) else None for v in value.numpy().tolist()]
    for count, (i, j) in enumerate(sorted(missing_cells()), 1):
        result = evaluate(tf.constant([grid["axis"][i], grid["axis"][j]], tf.float64), seed)
        values = finite_list(result["log_likelihood"])
        validity = (result["valid"] & tf.math.is_finite(result["log_likelihood"])).numpy().tolist()
        grid["log_likelihood"][i][j], grid["valid"][i][j] = values, validity
        if not all(validity):
            invalid.append([i, j])
        minimum_ess = min(minimum_ess, float(tf.reduce_min(result["minimum_ess"])))
        maximum_weight = max(maximum_weight, float(tf.reduce_max(result["proposal_maximum_normalized_weight"])))
        write_json(out/"completed-cells"/f"cell-{i:03d}-{j:03d}.json",
            dict(index=[i, j], theta=[grid["axis"][i], grid["axis"][j]],
                log_likelihood=values, valid=validity, minimum_ess=finite_list(result["minimum_ess"]),
                maximum_weight=finite_list(result["proposal_maximum_normalized_weight"]),
                fit_first=finite_list(result["proposal_first_diagnostics"]),
                fit_full=finite_list(result["proposal_full_diagnostics"])))
        if count % 64 == 0:
            print("M13 completion", count, "/386", "invalid", len(invalid),
                  "wall", time.monotonic()-start, flush=True)
    if any(file_hash(Path(name)) != digest for name, digest in {**sources, **saved_hashes}.items()):
        raise ValueError("source or preserved artifact changed during completion")
    write_json(out/"grid-129.json", grid)
    summary = (dict(status="INVALID_GRID_NO_POSTERIOR_SUMMARY", complete_valid_grid=False,
                    posterior_reference_qualified=False) if invalid else
               pilot.summarize_grid(grid["log_likelihood"], grid["valid"]))
    if not invalid:
        changes, metrics = summary["fine_minus_coarse"], summary["fine"]["metrics"]
        summary["mean_changes_in_posterior_sd"] = [changes[k]/metrics[2+k] for k in range(2)]
        summary["relative_sd_changes"] = [changes[2+k]/metrics[2+k] for k in range(2)]
    write_json(out/"summary.json", summary)
    return dict(status="M13_POSTERIOR_REFERENCE_COMPLETION_COMPLETED" if not invalid else
        "M13_POSTERIOR_REFERENCE_COMPLETION_REPAIR_REQUIRED", qualified=False,
        posterior_reference_qualified=False, endpoint_qualified=False, inference_launched=False,
        all_models_validated=False, continuation_veto=False, invalid_nodes=invalid,
        valid_node_count=ORDER**2-len(invalid), new_evaluations=len(missing_cells()),
        preserved_evaluations=len(seen), reused_coarse_nodes=65**2,
        minimum_ess_new_cells=minimum_ess, maximum_weight_new_cells=maximum_weight,
        summary=summary, source_sha256=sources, preserved_sha256=saved_hashes,
        grid_sha256=file_hash(out/"grid-129.json"), data_sha256=baseline["data_sha256"],
        seed=list(pilot.SEED), particles=pilot.PARTICLES, replicates=pilot.REPLICATES,
        dtype="float64", independent_reference_exception=True, jit_compile=True, plan=PLAN,
        posterior_comparisons="NOT_RUN", ranking_supported=False,
        next_action="Existing master: inspect reference resolution and localize the M13 force correction failure.")
