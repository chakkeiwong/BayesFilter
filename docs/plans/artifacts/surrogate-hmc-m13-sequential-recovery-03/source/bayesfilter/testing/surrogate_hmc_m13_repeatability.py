"""Diagnostic comparison of saved and current M13 code in fresh processes.

This module issues no tuning, posterior, or runtime-admission artifact.
Saved source is loaded only in the archived diagnostic subprocess; the live
checkout and its numerical defaults are never overwritten.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-consumer-03"
REPLAY = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-repair-01/m13-consumer-repair/report.json"
PLAN = "docs/plans/surrogate-hmc-m13-repeatability-2026-09-24.md"
PREFIX = "m13-repeatability"
CASES = (("archived", 1), ("current", 1), ("archived", 2), ("current", 2))
CASE_NAMES = tuple(f"{PREFIX}-{mode}-{replicate}" for mode, replicate in CASES)
SUMMARY_NAME = PREFIX + "-summary"
THRESHOLD = 3e-5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline():
    replay = json.loads(REPLAY.read_text())
    path = BASE / "m13-consumer-localize/report.json"
    if digest(path) != replay["previous_localization_sha256"]:
        raise ValueError("saved M13 localization changed")
    saved = json.loads(path.read_text())
    arm = next(a for a in saved["arms"] if a["particle_count"] == 256 and a["dtype"] == "float32")
    reference_path = ROOT / saved["reference_path"]
    if digest(reference_path) != saved["reference_sha256"]:
        raise ValueError("saved M13 reference changed")
    reference = json.loads(reference_path.read_text())
    if not arm["all_points_valid"] or not reference["likelihood_reference_qualified"]:
        raise ValueError("valid saved finite program and qualified reference required")
    return saved, arm, reference


class SavedSourceFinder(importlib.abc.MetaPathFinder):
    """Load only the recorded dependency files, with their recorded digests."""

    def __init__(self, sources, source_root):
        self.source_root = Path(source_root).resolve()
        self.files = {}
        for relative, expected in sources.items():
            path = Path(source_root) / relative
            if digest(path) != expected:
                raise ValueError(f"saved source changed: {relative}")
            module = relative.removesuffix(".py").replace("/", ".")
            self.files[module] = path

    def find_spec(self, fullname, path=None, target=None):
        source = self.files.get(fullname)
        return None if source is None else importlib.util.spec_from_file_location(fullname, source)


def materialize_saved_sources(sources, destination):
    """Copy the recorded closure; recover a missing file only by identical hash."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("fresh archived-source directory required")
    origins = {}
    for relative, expected in sources.items():
        source = BASE / "source" / relative
        recovered = not source.is_file()
        if recovered:
            source = ROOT / relative
        if digest(source) != expected:
            raise ValueError(f"recorded source hash mismatch: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        origins[relative] = dict(source=str(source), sha256=expected,
                                 recovered_identical_missing_snapshot_file=recovered)
    (destination / "source-origins.json").write_text(json.dumps(origins, indent=2, sort_keys=True)+"\n")
    return destination


def load_target(mode, arm, archive_root=None):
    if mode not in ("archived", "current"):
        raise ValueError("unknown source mode")
    sources = arm["manifest"]["source_sha256"]
    modules = {name.removesuffix(".py").replace("/", ".") for name in sources}
    if any(name in sys.modules for name in modules):
        raise ValueError("source comparison requires a fresh worker before numerical dependency imports")
    finder = None
    if mode == "archived":
        if archive_root is None:
            raise ValueError("archived comparison requires a fresh diagnostic source directory")
        source_root = materialize_saved_sources(sources, archive_root)
        finder = SavedSourceFinder(sources, source_root)
        # Preserve the archived source directory; do not create cache files there.
        sys.dont_write_bytecode = True
        sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module("bayesfilter.highdim.ledh_range_bearing_target_tf")
    except BaseException:
        if finder is not None:
            sys.meta_path.remove(finder)
        raise
    return module, finder


def scope_without_source(manifest):
    # Factory manifests have integer seed keys; JSON persistence makes them strings.
    # Compare the same serialization without dropping any scientific scope field.
    return json.loads(json.dumps({key: value for key, value in manifest.items() if key != "source_sha256"}))


def source_provenance(mode, arm, archive_root=None):
    if mode == "archived" and archive_root is None:
        raise ValueError("archived source provenance requires its actual diagnostic directory")
    result = {}
    for relative, expected in arm["manifest"]["source_sha256"].items():
        name = relative.removesuffix(".py").replace("/", ".")
        module = sys.modules.get(name)
        if module is None:
            result[relative] = {"loaded": False}
            continue
        path = Path(module.__file__).resolve()
        actual = digest(path)
        desired = (Path(archive_root) / relative if mode == "archived" else ROOT / relative).resolve()
        if path != desired or (mode == "archived" and actual != expected):
            raise ValueError(f"unexpected callable source: {relative}")
        result[relative] = {"loaded": True, "path": str(path), "sha256": actual}
    required = ("ledh_range_bearing_target_tf.py", "ledh_canonical_batch_fused_tf.py",
                "ledh_canonical_score_tf.py", "ledh_unified_correction_tf.py")
    if any(not result["bayesfilter/highdim/" + name]["loaded"] for name in required):
        raise ValueError("required canonical call-chain module was not loaded")
    return result


def verify_call_chain(target_module):
    """Check that the consumer's actual globals resolve to the loaded authority."""
    batch = sys.modules["bayesfilter.highdim.ledh_canonical_batch_fused_tf"]
    score = sys.modules["bayesfilter.highdim.ledh_canonical_score_tf"]
    links = (
        (target_module.M13LikelihoodTarget.evaluate, "canonical_batch_fused_value_score",
         batch.canonical_batch_fused_value_score),
        (batch.canonical_batch_fused_value_score, "canonical_value_and_analytical_score",
         score.canonical_value_and_analytical_score))
    result = []
    for consumer, name, producer in links:
        if consumer.__globals__.get(name) is not producer:
            raise ValueError(f"consumer resolves to another implementation: {name}")
        result.append(dict(consumer=consumer.__qualname__, dependency=name,
                           producer_source=producer.__code__.co_filename))
    return result


def serial(value):
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    return value.numpy().tolist()


def prepare_target(mode, *, archive_root=None):
    """Prepare the real frozen consumer, also usable in a CPU-only import smoke."""
    from dataclasses import replace

    saved, arm, reference = baseline()
    target_module, finder = load_target(mode, arm, archive_root)
    try:
        import tensorflow as tf
        from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import build_range_bearing_fixture_tf

        if tf.__version__ != saved["tensorflow"]:
            raise ValueError("TensorFlow version differs from the archived baseline")
        tf.config.experimental.enable_tensor_float_32_execution(arm["manifest"]["tf32"])
        fixture = build_range_bearing_fixture_tf()
        for key in ("A", "Q", "R", "m0", "P0"):
            if serial(getattr(fixture, key)) != arm["manifest"]["fixture"][key]:
                raise ValueError("M13 fixture law changed")
        fixture = replace(fixture,
            observations=tf.constant(reference["scope"]["observations"], tf.float64),
            observation_checksum=reference["scope"]["fixture"]["observation_checksum"])
        target = target_module.make_range_bearing_target(
            fixture, particle_count=256, controls=arm["manifest"]["settings"], dtype=tf.float32)
        manifest = dict(target.manifest)
        if scope_without_source(manifest) != scope_without_source(arm["manifest"]):
            actual_scope, saved_scope = scope_without_source(manifest), scope_without_source(arm["manifest"])
            changed = [key for key in sorted(actual_scope.keys() | saved_scope.keys())
                       if actual_scope.get(key) != saved_scope.get(key)]
            raise ValueError("M13 input, control or execution scope differs from baseline: "+str(changed))
        if mode == "archived" and manifest["source_sha256"] != arm["manifest"]["source_sha256"]:
            raise ValueError("archived target factory did not retain its source closure")
        return target, target_module, arm, saved, finder
    except BaseException:
        if finder is not None:
            sys.meta_path.remove(finder)
        raise


def variant_stage(output, mode, replicate):
    """One fresh GPU process, one compilation, two calls to the same graph."""
    import time
    from bayesfilter.inference.surrogate_hmc_program import write_json

    if (mode, replicate) not in CASES:
        raise ValueError("undeclared comparison arm")
    archive_root = Path(output).parent / "archived-source" if mode == "archived" else None
    target, target_module, arm, saved, finder = prepare_target(mode, archive_root=archive_root)
    try:
        import tensorflow as tf
        manifest = dict(target.manifest)
        points = tf.constant([row["theta"] for row in arm["rows"]], tf.float32)
        evaluate = target.compile(9, filter_moments=True)
        started = time.monotonic()
        values, scores, status = evaluate(points)
        repeated_values, repeated_scores, repeated_status = evaluate(points)
        valid = bool(tf.reduce_all(status["program_valid"] & repeated_status["program_valid"])
                     & tf.reduce_all(tf.math.is_finite(values)) & tf.reduce_all(tf.math.is_finite(scores))
                     & tf.reduce_all(tf.math.is_finite(repeated_values))
                     & tf.reduce_all(tf.math.is_finite(repeated_scores)))
        if not valid or "GPU" not in values.device or values.device != arm["kernel_device"]:
            raise ValueError("invalid finite program or missing GPU placement")
        same = bool(tf.reduce_all(values == repeated_values) & tf.reduce_all(scores == repeated_scores)
                    & tf.reduce_all(status["program_valid"] == repeated_status["program_valid"]))
        provenance = source_provenance(mode, arm, archive_root)
        call_chain = verify_call_chain(target_module)
        trace_path = Path(output).with_name("status-tensors.json")
        write_json(trace_path, {"first": serial(status), "repeat": serial(repeated_status)})
        result = dict(status="M13_REPEATABILITY_ARM_COMPLETE", qualified=False,
            diagnostic_only=True, inference_launched=False, posterior_qualified=False,
            scope_controls_qualified=False, all_models_validated=False, prior=None,
            plan=PLAN, source_mode=mode, replicate=replicate, manifest=manifest,
            actual_loaded_sources=provenance, verified_call_chain=call_chain, parameter_points=[row["theta"] for row in arm["rows"]],
            baseline_sha256=digest(BASE / "m13-consumer-localize/report.json"),
            reference_sha256=saved["reference_sha256"],
            value=serial(values), score=serial(scores),
            repeated_value=serial(repeated_values), repeated_score=serial(repeated_scores),
            checks=dict(all_points_valid=valid, within_process_exact_repeat=same),
            kernel_device=values.device, numerical_wall_seconds=time.monotonic()-started,
            status_tensors_artifact=str(trace_path), status_tensors_sha256=digest(trace_path),
            nonclaims=["two fresh processes do not certify general repeatability",
                       "no source variant or numerical default is promoted",
                       "no likelihood-accuracy or posterior claim"])
        return result
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)


def compare_values(reference, candidate):
    """Post-run diagnostic arithmetic; no inference or admission decision."""
    if len(reference["value"]) != 9 or len(candidate["value"]) != 9:
        raise ValueError("nine matched points required")
    rows = []
    for index, (rv, cv, rs, cs) in enumerate(zip(
            reference["value"], candidate["value"], reference["score"], candidate["score"], strict=True)):
        if len(rs) != 2 or len(cs) != 2 or not all(math.isfinite(x) for x in (rv, cv, *rs, *cs)):
            raise ValueError("finite values and two score components required")
        value_error = abs(cv-rv)/(1+abs(rv))
        score_error = [abs(c-r)/(1+abs(r)) for r, c in zip(rs, cs)]
        rows.append(dict(index=index, value_absolute_difference=abs(cv-rv),
                         value_scaled_error=value_error, score_scaled_errors=score_error))
    maximum_value = max(row["value_scaled_error"] for row in rows)
    maximum_score = max(max(row["score_scaled_errors"]) for row in rows)
    return dict(rows=rows, value_max_scaled_error=maximum_value, score_max_scaled_error=maximum_score,
                within_original_tolerance=maximum_value <= THRESHOLD and maximum_score <= THRESHOLD)


def classify(comparisons, within_process):
    if not within_process:
        return "WITHIN_PROCESS_VARIABILITY_OBSERVED"
    if not comparisons["archived_fresh_repeat"]["within_original_tolerance"]:
        return "ARCHIVED_FRESH_PROCESS_VARIABILITY_OBSERVED"
    if not comparisons["current_fresh_repeat"]["within_original_tolerance"]:
        return "CURRENT_FRESH_PROCESS_VARIABILITY_OBSERVED"
    if not all(comparisons[name]["within_original_tolerance"] for name in
               ("matched_source_pair_1", "matched_source_pair_2")):
        return "SOURCE_ASSOCIATED_FP32_DIFFERENCE_OBSERVED"
    if not all(row["within_original_tolerance"] for name, row in comparisons.items()
               if name.startswith("saved_vs_")):
        return "SAVED_BASELINE_NOT_REPRODUCED"
    return "FROZEN_SCOPE_REPLAYS_MATCH_SAVED_BASELINE"


def summary_stage(output):
    saved, arm, _ = baseline()
    root = Path(output).parent.parent
    reports = {name: json.loads((root / name / "report.json").read_text()) for name in CASE_NAMES}
    for name, (mode, replicate) in zip(CASE_NAMES, CASES):
        report = reports[name]
        if (report["status"] != "M13_REPEATABILITY_ARM_COMPLETE"
                or report["source_mode"] != mode or report["replicate"] != replicate
                or not report["checks"]["all_points_valid"]
                or scope_without_source(report["manifest"]) != scope_without_source(arm["manifest"])
                or report["parameter_points"] != [row["theta"] for row in arm["rows"]]
                or report["baseline_sha256"] != digest(BASE / "m13-consumer-localize/report.json")
                or report["reference_sha256"] != saved["reference_sha256"]):
            raise ValueError("incompatible or invalid repeatability arm")
    a1, c1, a2, c2 = (reports[name] for name in CASE_NAMES)
    old = dict(value=[row["log_likelihood"] for row in arm["rows"]],
               score=[row["score"] for row in arm["rows"]])
    comparisons = {
        "archived_fresh_repeat": compare_values(a1, a2),
        "current_fresh_repeat": compare_values(c1, c2),
        "matched_source_pair_1": compare_values(a1, c1),
        "matched_source_pair_2": compare_values(a2, c2)}
    for name, report in reports.items():
        comparisons["saved_vs_" + name] = compare_values(old, report)
    within_process = all(report["checks"]["within_process_exact_repeat"] for report in reports.values())
    finding = classify(comparisons, within_process)
    return dict(status="M13_REPEATABILITY_COMPARISON_COMPLETE", finding=finding, qualified=False,
        diagnostic_only=True, plan=PLAN, inference_launched=False, posterior_qualified=False,
        all_models_validated=False, comparisons=comparisons, within_process_exact_repeat=within_process,
        original_scaled_tolerance=THRESHOLD,
        next_action="Existing master: use the classified source/repeatability finding to design the specific repair; no automatic tuning or HMC.",
        nonclaims=["no statistical ranking or general reproducibility guarantee",
                   "no threshold relaxation, default promotion or posterior qualification"])
