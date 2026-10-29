"""Regression checks for source isolation and bounded M13 diagnostic execution."""
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from bayesfilter.testing import surrogate_hmc_m13_repeatability as repeat


def test_saved_source_importer_executes_saved_dependency_chain(tmp_path, monkeypatch):
    live = tmp_path / "live" / "m13_source_probe"
    saved = tmp_path / "saved"
    live.mkdir(parents=True)
    (live / "__init__.py").write_text("")
    bodies = {
        "m13_source_probe/consumer.py": "from .engine import value\ndef evaluate(): return value()\n",
        "m13_source_probe/engine.py": "def value(): return 17\n"}
    sources = {}
    for name, body in bodies.items():
        path = saved / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        sources[name] = repeat.digest(path)
        (live / Path(name).name).write_text(body.replace("17", "99"))
    monkeypatch.syspath_prepend(str(live.parent))
    finder = repeat.SavedSourceFinder(sources, saved)
    sys.meta_path.insert(0, finder)
    try:
        consumer = importlib.import_module("m13_source_probe.consumer")
        assert consumer.evaluate() == 17
        assert Path(consumer.__file__) == saved / "m13_source_probe/consumer.py"
        assert Path(consumer.evaluate.__globals__["value"].__code__.co_filename) == saved / "m13_source_probe/engine.py"
    finally:
        sys.meta_path.remove(finder)
        for name in ("m13_source_probe.consumer", "m13_source_probe.engine", "m13_source_probe"):
            sys.modules.pop(name, None)
    (saved / "m13_source_probe/engine.py").write_text("def value(): return 18\n")
    with pytest.raises(ValueError, match="saved source changed"):
        repeat.SavedSourceFinder(sources, saved)


def test_archive_loader_rejects_preloaded_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "m13_source_conflict", SimpleNamespace())
    with pytest.raises(ValueError, match="fresh worker"):
        repeat.load_target("archived", {"manifest": {"source_sha256": {"m13_source_conflict.py": "unused"}}})


@pytest.mark.parametrize("mode", ["archived", "current"])
def test_real_frozen_target_prepares_and_wires_in_fresh_cpu_worker(mode):
    # Import and fixed-input smoke only; no numerical filter or GPU is run.
    source = """
import importlib
import json
import sys
from bayesfilter.testing import surrogate_hmc_m13_repeatability as m
target, module, arm, saved, finder = m.prepare_target(sys.argv[1])
import tensorflow as tf
assert not tf.config.list_physical_devices("GPU")
importlib.import_module("bayesfilter.highdim.ledh_unified_correction_tf")
provenance = m.source_provenance(sys.argv[1], arm)
links = m.verify_call_chain(module)
evaluate = target.compile(9, filter_moments=True)
assert len(evaluate.input_signature) == 1
assert evaluate.input_signature[0].shape.as_list() == [9, 2]
assert len(links) == 2
assert provenance["bayesfilter/highdim/ledh_range_bearing_target_tf.py"]["loaded"]
print(json.dumps({"cpu_only_smoke": True, "gpu_intentionally_hidden": True, "mode": sys.argv[1]}))
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_FORCE_GPU_ALLOW_GROWTH="true",
                       TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-c", source, mode], cwd=repeat.ROOT,
                            env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "cpu_only_smoke": True, "gpu_intentionally_hidden": True, "mode": mode}


def test_actual_callable_binding_detects_mixed_consumer(monkeypatch):
    def authority():
        return 7
    batch_globals = {"canonical_value_and_analytical_score": authority}
    exec("def batch(): return canonical_value_and_analytical_score()", batch_globals)
    batch = batch_globals["batch"]
    target_globals = {"canonical_batch_fused_value_score": batch}
    exec("def evaluate(): return canonical_batch_fused_value_score()", target_globals)
    target = SimpleNamespace(M13LikelihoodTarget=SimpleNamespace(evaluate=target_globals["evaluate"]))
    monkeypatch.setitem(sys.modules, "bayesfilter.highdim.ledh_canonical_batch_fused_tf",
                        SimpleNamespace(canonical_batch_fused_value_score=batch))
    monkeypatch.setitem(sys.modules, "bayesfilter.highdim.ledh_canonical_score_tf",
                        SimpleNamespace(canonical_value_and_analytical_score=authority))
    assert len(repeat.verify_call_chain(target)) == 2
    target_globals["canonical_batch_fused_value_score"] = lambda: 7
    with pytest.raises(ValueError, match="another implementation"):
        repeat.verify_call_chain(target)


def values():
    return dict(value=[0.] * 9, score=[[0., 1.] for _ in range(9)])


def test_comparison_checks_each_score_and_point_without_averaging():
    base, candidate = values(), values()
    candidate["score"][8][1] += 8e-5
    result = repeat.compare_values(base, candidate)
    assert result["score_max_scaled_error"] == pytest.approx(4e-5)
    assert result["value_max_scaled_error"] == 0.
    assert not result["within_original_tolerance"]
    candidate["score"][8][1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        repeat.compare_values(base, candidate)
    with pytest.raises(ValueError, match="nine matched"):
        repeat.compare_values(base, {"value": [0.], "score": [[0., 1.]]})


@pytest.mark.parametrize("failed,within,expected", [
    (None, True, "FROZEN_SCOPE_REPLAYS_MATCH_SAVED_BASELINE"),
    (None, False, "WITHIN_PROCESS_VARIABILITY_OBSERVED"),
    ("archived_fresh_repeat", True, "ARCHIVED_FRESH_PROCESS_VARIABILITY_OBSERVED"),
    ("current_fresh_repeat", True, "CURRENT_FRESH_PROCESS_VARIABILITY_OBSERVED"),
    ("matched_source_pair_1", True, "SOURCE_ASSOCIATED_FP32_DIFFERENCE_OBSERVED"),
    ("saved_vs_case", True, "SAVED_BASELINE_NOT_REPRODUCED")])
def test_failure_classification_does_not_misattribute_old_variability(failed, within, expected):
    names = ("archived_fresh_repeat", "current_fresh_repeat", "matched_source_pair_1",
             "matched_source_pair_2", "saved_vs_case")
    comparisons = {name: {"within_original_tolerance": name != failed} for name in names}
    assert repeat.classify(comparisons, within) == expected


def test_summary_rejects_changed_scope_and_keeps_all_comparisons(tmp_path, monkeypatch):
    monkeypatch.setattr(repeat, "BASE", tmp_path / "baseline")
    baseline_path = repeat.BASE / "m13-consumer-localize/report.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text("{}")
    arm = dict(manifest=dict(dtype="float32", particle_count=256, source_sha256={"x.py": "old"}),
               rows=[dict(theta=[float(i), 0.], log_likelihood=0., score=[0., 1.]) for i in range(9)])
    saved = dict(reference_sha256="reference")
    monkeypatch.setattr(repeat, "baseline", lambda: (saved, arm, {}))
    root = tmp_path / "comparison"
    for name, (mode, replicate) in zip(repeat.CASE_NAMES, repeat.CASES):
        path = root / name / "report.json"
        path.parent.mkdir(parents=True)
        record = dict(status="M13_REPEATABILITY_ARM_COMPLETE", source_mode=mode, replicate=replicate,
                      checks=dict(all_points_valid=True, within_process_exact_repeat=True),
                      manifest=deepcopy(arm["manifest"]), parameter_points=[r["theta"] for r in arm["rows"]],
                      baseline_sha256=repeat.digest(baseline_path), reference_sha256="reference", **values())
        if mode == "current":
            record["value"][0] = 0.1
        path.write_text(json.dumps(record))
    output = root / repeat.SUMMARY_NAME / "report.json"
    result = repeat.summary_stage(output)
    assert result["finding"] == "SOURCE_ASSOCIATED_FP32_DIFFERENCE_OBSERVED"
    assert len(result["comparisons"]) == 8
    assert not result["qualified"] and not result["inference_launched"]
    bad = root / repeat.CASE_NAMES[-1] / "report.json"
    record = json.loads(bad.read_text())
    record["manifest"]["particle_count"] = 1024
    bad.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="incompatible"):
        repeat.summary_stage(output)


def test_master_reserves_complete_batch_with_no_partial_or_mixed_selection():
    from bayesfilter.inference.surrogate_hmc_closure_program import jobs, PASS_CAPS
    rows = jobs("m13-repeatability")
    assert [row["name"] for row in rows] == ["review", *repeat.CASE_NAMES, repeat.SUMMARY_NAME]
    assert sum(row["cpu"] for row in rows) + 120 == PASS_CAPS["m13-repeatability"]["cpu_worker_seconds"]
    assert sum(row["timeout"] for row in rows if row["device"] == "gpu") == PASS_CAPS["m13-repeatability"]["gpu_seconds"]
    for name in (*repeat.CASE_NAMES, repeat.SUMMARY_NAME):
        with pytest.raises(ValueError, match="complete four-process"):
            jobs(name)
    with pytest.raises(ValueError):
        jobs("m13-repeatability,ar2-controls")


@pytest.mark.parametrize("fail_first", [False, True])
def test_master_executes_ordered_batch_once_and_blocks_failed_dependencies(monkeypatch, tmp_path, fail_first):
    from bayesfilter.inference import surrogate_hmc_closure_program as closure
    monkeypatch.setattr(closure, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(closure, "snapshot_sources", lambda output: {})
    monkeypatch.setattr(closure, "archived_judgments", lambda: {})
    seen = []
    def run(command, directory, ledger, **options):
        seen.append(directory.name)
        directory.mkdir()
        status = "FAILED" if fail_first and directory.name == repeat.CASE_NAMES[0] else "COMPLETED"
        record = dict(status=status, cpu_worker_seconds=.01, gpu_seconds=0.)
        ledger["jobs"].append(record)
        closure.write_json(directory / "run-manifest.json", {"command": command})
        closure.write_json(directory / "report.json", dict(status="FIXTURE", qualified=False))
        return record
    monkeypatch.setattr(closure, "run_job", run)
    output = tmp_path / "surrogate-hmc-closure-20260922-m13-repeatability-01"
    with pytest.raises(SystemExit):
        closure.main(["--output", str(tmp_path / "surrogate-hmc-closure-20260922-m13-repair-02"),
                      "--stages", "m13-repeatability", "--dry-run"])
    assert closure.main(["--output", str(output), "--stages", "m13-repeatability"]) == int(fail_first)
    assert seen == (["review", repeat.CASE_NAMES[0]] if fail_first else
                    ["review", *repeat.CASE_NAMES, repeat.SUMMARY_NAME])
    result = json.loads((output / "report.json").read_text())
    assert not result["all_models_validated"] and not result["inference_launched"]
    if fail_first:
        assert result["stages"][repeat.SUMMARY_NAME]["execution"] == "BLOCKED_DEPENDENCY"
    else:
        seen.clear()
        second = tmp_path / "surrogate-hmc-closure-20260922-m13-repeatability-02"
        assert closure.main(["--output", str(second), "--stages", "m13-repeatability"]) == 1
        assert seen == ["review"]
        retry = json.loads((second / "report.json").read_text())
        assert all(retry["stages"][name]["execution"] == "ATTEMPT_LIMIT" for name in repeat.CASE_NAMES)
