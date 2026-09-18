"""The campaign must fail closed on incomplete or numerically unequal evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unfinished_attempt_consumes_reserved_budget(tmp_path, monkeypatch):
    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver,"OUTPUT",tmp_path)
    rows = [{"device": "GPU", "timeout_seconds": 300}, {"device": "GPU", "timeout_seconds": 300, "elapsed_seconds": 17.5}, {"device": "CPU", "timeout_seconds": 900}]
    assert driver.charged_seconds(rows, "GPU") == 317.5
    (tmp_path / "supplemental-compute-0001.json").write_text(json.dumps({"device":"CPU","charged_seconds":1800}))
    assert driver.charged_seconds(rows,"CPU") == 2700


def test_linked_worktrees_share_campaign_budget_and_artifact_root(tmp_path, monkeypatch):
    driver = load("run_filter_repair_campaign")
    primary = tmp_path / "primary"
    validation = tmp_path / "validation"
    calls = []

    def git_common(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return str(primary / ".git") + "\n"

    monkeypatch.setattr(driver.subprocess, "check_output", git_common)
    expected = primary / "docs/plans/artifacts/filter-gradient-repair-20260917"
    assert driver.campaign_output_root(primary) == expected
    assert driver.campaign_output_root(validation) == expected
    assert [root for _, root in calls] == [primary, validation]
    assert all(command == ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"]
               for command, _ in calls)


def test_interrupt_stops_worker_and_finalizes_attempt(tmp_path, monkeypatch):
    import argparse

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "records", list)
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "git", lambda *_: "test")
    signals = []
    monkeypatch.setattr(driver.os, "killpg", lambda pid, signal: signals.append((pid, signal)))

    class Worker:
        pid = 123
        waits = 0

        def wait(self, **_):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return -15

    worker = Worker()
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *_, **kwargs: worker)
    args = argparse.Namespace(action="audit", device="CPU", group="policy", arm="after",
        fixture="dns", jit="on", size=1, repeat=0)
    assert driver.run_job(args) == 130
    run = json.loads((tmp_path / "run-00001/run.json").read_text())
    assert run["state"] == "failed" and run["exit_code"] == 130
    assert run["elapsed_seconds"] >= 0
    assert signals == [(123, driver.signal.SIGTERM)] and worker.waits == 2


def test_baseline_parent_packages_are_pinned_and_legacy_snapshot_is_recovered(tmp_path, monkeypatch):
    import subprocess

    driver = load("run_filter_repair_campaign")
    snapshot = tmp_path / "fresh"
    monkeypatch.setattr(driver, "BASELINE_ROOT", snapshot)
    driver.ensure_baseline()
    manifest = json.loads((snapshot / "source-manifest.json").read_text())
    assert all(path in manifest["files"] for path in driver.BASELINE_PARENT_PACKAGES)
    package = "experiments/dpf_implementation/tf_tfp/__init__.py"
    legacy = tmp_path / "legacy"
    (legacy / package).parent.mkdir(parents=True)
    (legacy / package).write_bytes((snapshot / package).read_bytes())
    (legacy / "source-manifest.json").write_text(json.dumps({
        "commit": driver.BASELINE, "files": {package: manifest["files"][package]},
    }))
    monkeypatch.setattr(driver, "BASELINE_ROOT", legacy)
    driver.ensure_baseline()
    driver.ensure_baseline()
    recovered = json.loads((legacy / "source-manifest.json").read_text())
    assert recovered["files"] == {path: manifest["files"][path]
                                  for path in (*driver.BASELINE_PARENT_PACKAGES, package)}
    program = """
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import experiments
import experiments.dpf_implementation
import experiments.dpf_implementation.tf_tfp
print(json.dumps([str(pathlib.Path(module.__file__).resolve()) for module in (
    experiments, experiments.dpf_implementation, experiments.dpf_implementation.tf_tfp)]))
"""
    imported = json.loads(subprocess.check_output([sys.executable, "-c", program, str(legacy)],
                                                  cwd=driver.ROOT, text=True, timeout=30))
    assert all(Path(path).is_relative_to(legacy) for path in imported)


def test_gate_rejects_empty_or_unsupported_closure(tmp_path, monkeypatch, capsys):
    driver = load("run_filter_repair_campaign")
    ledger = tmp_path / "docs/plans/filter_gradient_repair_ledger_20260917.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"findings": [{"id": f"F{i:02d}", "status": "closed"} for i in range(1, 21)]}))
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "records", list)
    monkeypatch.setattr(driver, "source_hashes", dict)
    assert driver.gate() == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["merge_allowed"] is False
    assert len(verdict["open_findings"]) == 20


def test_parity_rejects_nonfinite_and_discrete_mismatch():
    comparison = load("compare_filter_repair_campaign")
    for before, after in (([float("nan")], [0.]), ([1], [2]), ([0.5], [0.6]), ([True], [False])):
        with pytest.raises(ValueError):
            comparison.compare_values(before, after)
    assert comparison.compare_values([[1.0, 2.0]], [[1.0, 2.0 + 1e-12]]) < 2e-12
    for before,after in (([[1.,2.]],[1.,2.]),([1.],[True]),([1.],[1])):
        with pytest.raises(ValueError):
            comparison.compare_values(before,after)


def test_additional_harness_keeps_original_measurements_and_binds_extension():
    driver = load("run_filter_repair_campaign")
    comparison = load("compare_filter_repair_campaign")
    original = driver.measurement_harness("covariance")
    additional = driver.measurement_harness("latent_sir")
    assert set(original) == {
        "filter_repair_benchmark_worker.py", "measure_filter_xla_memory.py",
        "filter_repair_endpoint_fixtures.py",
    }
    assert {name: additional[name] for name in original} == original
    assert set(additional) - set(original) == {
        "filter_repair_additional_worker.py", "filter_repair_additional_fixtures.py",
    }
    measurement = {"schema": "filter_repair_measurement.v2", "harness_sha256": original}
    with pytest.raises(ValueError, match="Stale measurement harness"):
        comparison.current_provenance({}, measurement, "before", additional, {})
    forecast = driver.measurement_harness("ssl_forecast")
    assert {name: forecast[name] for name in original} == original
    assert set(forecast) - set(original) == {
        "filter_repair_forecast_worker.py", "filter_repair_forecast_fixtures.py",
    }
    with pytest.raises(ValueError, match="Stale measurement harness"):
        comparison.current_provenance({}, measurement, "before", forecast, {})
    preparation = driver.measurement_harness("ukf_initializer")
    assert {name: preparation[name] for name in original} == original
    assert set(preparation) - set(original) == {
        "filter_repair_preparation_worker.py", "filter_repair_preparation_fixtures.py",
    }
    with pytest.raises(ValueError, match="Stale measurement harness"):
        comparison.current_provenance({}, measurement, "before", preparation, {})
    centered = driver.measurement_harness("centered_child")
    assert {name: centered[name] for name in original} == original
    assert set(centered) - set(original) == {
        "filter_repair_centered_worker.py", "filter_repair_centered_fixtures.py",
    }
    with pytest.raises(ValueError, match="Stale measurement harness"):
        comparison.current_provenance({}, measurement, "before", centered, {})


def test_source_guard_blocks_otherwise_complete_gate(tmp_path, monkeypatch, capsys):
    driver = load("run_filter_repair_campaign")
    ledger = tmp_path / "docs/plans/filter_gradient_repair_ledger_20260917.json"
    ledger.parent.mkdir(parents=True)
    (tmp_path / "evidence.json").write_text("{}")
    ledger.write_text(json.dumps({"findings": [{"id": f"F{i:02d}",
        "status": "closed", "evidence": ["evidence.json"]} for i in range(1, 21)]}))
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuites><testsuite><testcase name="executed"/></testsuite></testsuites>')
    rows = [{"key": ["test", group, "after"], "state": "passed", "source_sha256": {},
             "device": driver.TEST_DEVICES.get(group, "CPU"), "result": str(tmp_path / "result.json")}
            for group in driver.TEST_GROUPS]
    rows.append({"key": ["compare"], "state": "passed", "source_sha256": {}})
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver, "records", lambda: rows)
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "verify_source_policy", lambda *_: {"passed": False, "violations": ["loop"]})
    assert driver.gate() == 1
    verdict = json.loads(capsys.readouterr().out)
    assert not verdict["open_findings"] and not verdict["missing_current_tests"]
    assert verdict["current_comparison"] and not verdict["source_policy"]["passed"]


@pytest.mark.parametrize("body,passed", [
    ('<testcase name="executed"/>', True),
    ("", False),
    ('<testcase><skipped message="unavailable GPU"/></testcase>', False),
    ('<testcase><failure/></testcase>', False),
    ('<testcase><error/></testcase>', False),
])
def test_required_test_evidence_cannot_be_empty_or_skipped(tmp_path, body, passed):
    driver = load("run_filter_repair_campaign")
    (tmp_path / "junit.xml").write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    assert driver.test_evidence({"result": str(tmp_path / "result.json")})["passed"] is passed


def test_required_test_evidence_requires_readable_junit(tmp_path):
    driver = load("run_filter_repair_campaign")
    run = {"result": str(tmp_path / "result.json")}
    assert not driver.test_evidence(run)["passed"]
    (tmp_path / "junit.xml").write_text("broken")
    assert not driver.test_evidence(run)["passed"]


def test_baseline_failure_cannot_hide_numerical_or_resource_errors():
    comparison = load("compare_filter_repair_campaign")
    assert comparison.baseline_compilation_failure({"phase": "trace","error_type": "AttributeError",
        "error": "SymbolicTensor has no attribute numpy"}) == "baseline_host_operation_during_trace"
    assert comparison.baseline_compilation_failure({"phase": "first_execution","error_type": "InvalidArgumentError",
        "error": "Detected unsupported operations on XLA_CPU_JIT"}) == "baseline_xla_compilation_failure"
    assert comparison.baseline_compilation_failure({"phase": "trace", "error_type": "TypeError",
        "error": "batch_finite_value_score tf.ensure_shape: Could not generate a generic TraceType"}) == "baseline_forward_accumulator_shape_tracing_failure"
    generator_failure = {"phase": "trace", "error_type": "ValueError", "fixture": "simulation_sv",
        "error": "Generator.from_seed: tf.function only supports singleton tf.Variables"}
    assert comparison.baseline_compilation_failure(generator_failure) == "baseline_generator_variable_created_during_trace"
    assert comparison.baseline_compilation_failure({**generator_failure, "fixture": "tt"}) is None
    assert comparison.baseline_compilation_failure({**generator_failure, "phase": "warm"}) is None
    nested = {"phase": "trace", "jit": "off", "error_type": "RuntimeError",
        "error": "Invalid callback/compilation boundary", "graph": {"callbacks": [], "nested_xla": ["inner"]}}
    assert comparison.baseline_compilation_failure(nested) == "baseline_forced_nested_xla_requires_eager_reference"
    assert comparison.baseline_compilation_failure({**nested, "graph": {"callbacks": ["PyFunc"], "nested_xla": ["inner"]}}) is None
    for phase,error in (("warm","NaN in XLA"),("first_execution","XLA out of memory"),
                         ("preparation","SymbolicTensor")):
        assert comparison.baseline_compilation_failure({"phase": phase,"error": error,"error_type": "RuntimeError"}) is None


def test_matrix_test_failure_stops_remaining_groups(tmp_path, monkeypatch):
    import argparse
    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "TEST_GROUPS", {"first": (), "second": ()})
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "records", list)
    executed = []
    def fail(job):
        executed.append(job.group)
        return 7
    monkeypatch.setattr(driver, "run_job", fail)
    assert driver.run_matrix(argparse.Namespace(stage="tests")) == 7
    assert executed == ["first"]


def test_matrix_stops_when_source_changes(tmp_path, monkeypatch):
    import argparse
    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "TEST_GROUPS", {"first": ()})
    source = iter(({"kernel": "original"}, {"kernel": "changed"}))
    monkeypatch.setattr(driver, "source_hashes", lambda: next(source))
    with pytest.raises(RuntimeError, match="Source changed"):
        driver.run_matrix(argparse.Namespace(stage="tests"))


def test_pause_request_stops_between_workers_without_taking_active_lock(tmp_path, monkeypatch):
    import argparse

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver.fcntl, "flock", lambda *_: pytest.fail("Active matrix owns the lock"))
    monkeypatch.setattr(sys, "argv", ["run_filter_repair_campaign.py", "pause"])
    assert driver.main() == 0
    assert json.loads((tmp_path / "pause-request.json").read_text())["requested_utc"]
    monkeypatch.setattr(driver, "TEST_GROUPS", {"next": ()})
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "run_job", lambda _: pytest.fail("Pause must precede next worker"))
    with pytest.raises(RuntimeError, match="paused between workers"):
        driver.run_matrix(argparse.Namespace(stage="tests"))


@pytest.mark.parametrize("graph_passes", [False, True])
def test_matrix_prefers_valid_graph_reference_before_eager(tmp_path, monkeypatch, graph_passes):
    import argparse

    import compare_filter_repair_campaign as comparison

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    (tmp_path / "source-manifest.json").write_text(json.dumps({"files": {}}))
    monkeypatch.setattr(driver, "BASELINE_ROOT", tmp_path)
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "ensure_baseline", lambda: None)
    monkeypatch.setattr(driver, "sha", lambda _: "fixed")
    rows = []
    for arm, mode in (("before", "off"), ("before", "on"), ("before", "eager"),
                      ("after", "off"), ("after", "on")):
        failed = arm == "before" and (mode == "on" or mode == "off" and not graph_passes)
        value = {"status": "failed" if failed else "passed", "jit": mode}
        if failed:
            value.update(phase="first_execution", error_type="InvalidArgumentError",
                         error="Detected unsupported operations on XLA_GPU_JIT")
        path = tmp_path / f"{arm}-{mode}.json"
        path.write_text(json.dumps(value))
        rows.append({"key": ["measure", "policy", arm, "rectangular", mode, 1, 0, "GPU"],
                         "result": str(path), "state": value["status"]})
    monkeypatch.setattr(driver, "records", lambda: rows)
    monkeypatch.setattr(driver, "run_job", lambda _: pytest.fail("Valid reference already exists"))
    monkeypatch.setattr(comparison, "current_provenance", lambda *_: None)
    pairs = []
    monkeypatch.setattr(comparison, "compare_pair",
                        lambda before, after: pairs.append((before["jit"], after["jit"])) or 0.0)
    args = argparse.Namespace(stage="qualify", selection="fixture", fixture="rectangular")
    assert driver.run_matrix(args) == 0
    expected = "off" if graph_passes else "eager"
    assert pairs == [(expected, "off"), (expected, "on")]


@pytest.mark.parametrize("gpu_index", (2, 3))
def test_gpu_test_group_uses_gpu_and_preflight(tmp_path, monkeypatch, gpu_index):
    import argparse
    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "TEST_GROUPS", {"random_gpu": ()})
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "records", list)
    checked = []
    monkeypatch.setattr(driver, "check_gpu_idle", lambda index: checked.append(index) or ["idle"])
    jobs = []
    monkeypatch.setattr(driver, "run_job", lambda job: jobs.append(job) or 0)
    assert driver.run_matrix(argparse.Namespace(stage="tests", test_gpu_index=gpu_index)) == 0
    assert jobs[0].device == "GPU" and jobs[0].gpu_preflight == ["idle"]
    assert checked == [gpu_index] and jobs[0].test_gpu_index == gpu_index


def test_alternate_gpu_option_cannot_change_measurement_device(monkeypatch):
    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(sys, "argv", ["driver", "measure", "--test-gpu-index", "3"])
    monkeypatch.setattr(driver, "run_job", lambda _: pytest.fail("Invalid device selection must not launch"))
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2


def test_gpu_preflight_queries_selected_device(monkeypatch):
    driver = load("run_filter_repair_campaign")
    commands = []
    monkeypatch.setattr(driver.subprocess, "check_output", lambda command, **_: commands.append(command) or "18, 0")
    monkeypatch.setattr(driver.time, "sleep", lambda _: None)
    driver.check_gpu_idle(3)
    assert len(commands) == 2 and all(command[1:3] == ["-i", "3"] for command in commands)


def test_gpu_idle_rechecks_recent_utilization_and_records_samples(monkeypatch):
    driver = load("run_filter_repair_campaign")
    samples = iter(("18, 7", "18, 0", "18, 0"))
    monkeypatch.setattr(driver.subprocess, "check_output", lambda *a, **k: next(samples))
    sleeps = []
    monkeypatch.setattr(driver.time, "sleep", sleeps.append)
    assert driver.check_gpu_idle() == [
        {"memory_mib": 18, "utilization_percent": 7},
        {"memory_mib": 18, "utilization_percent": 0},
        {"memory_mib": 18, "utilization_percent": 0},
    ]
    assert sleeps == [2, 2]


@pytest.mark.parametrize("reading", ("101, 0", "18, 6"))
def test_gpu_idle_keeps_original_contention_thresholds(monkeypatch, reading):
    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver.subprocess, "check_output", lambda *a, **k: reading)
    sleeps = []
    monkeypatch.setattr(driver.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="contention veto after bounded recheck"):
        driver.check_gpu_idle()
    assert sleeps == [2] * 5


def test_regression_thresholds_require_investigation():
    comparison = load("compare_filter_repair_campaign")
    before = {"warm_median_seconds": 1.,"device_peak_bytes": 1024,"host_peak_bytes": 2048,
        "late_device_growth_bytes": 0,"late_host_growth_bytes": 0}
    assert comparison.regression_reasons(before,before) == []
    for field,value,reason in (("warm_median_seconds",1.21,"warm_time_over_20_percent"),
        ("device_peak_bytes",2049,"device_peak_over_2x"),
        ("host_peak_bytes",2049+256*2**20,"host_peak_over_256_MiB"),
        ("late_device_growth_bytes",256,"continuing_device_allocation_growth")):
        assert reason in comparison.regression_reasons(before,{**before,field:value})


def test_candidate_measurement_source_freshness(tmp_path,monkeypatch):
    import hashlib
    comparison = load("compare_filter_repair_campaign")
    monkeypatch.setattr(comparison,"ROOT",tmp_path)
    module = tmp_path / "kernel.py"
    module.write_text("x=1\n")
    measurement = {"schema": "filter_repair_measurement.v2","harness_sha256": {},
        "source_root": str(tmp_path),"imported_source_sha256": {"kernel.py":hashlib.sha256(module.read_bytes()).hexdigest()},"status": "failed"}
    run = {"state": "failed", "cwd": str(tmp_path),
           "source_sha256": dict(measurement["imported_source_sha256"])}
    comparison.current_provenance(run,measurement,"after",{}, {})
    module.write_text("x=2\n")
    with pytest.raises(ValueError,match="Stale candidate source"):
        comparison.current_provenance(run,measurement,"after",{}, {})
    measurement["imported_source_sha256"]["kernel.py"] = hashlib.sha256(module.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="changed during measurement"):
        comparison.current_provenance(run,measurement,"after",{}, {})


def test_legacy_parent_marker_provenance_requires_pinned_source(tmp_path, monkeypatch):
    import hashlib
    comparison = load("compare_filter_repair_campaign")
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    relative = comparison.BASELINE_PARENT_PACKAGES[0]
    module = tmp_path / relative
    module.parent.mkdir(parents=True)
    module.write_text("# Package marker\n")
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    measurement = {"schema": "filter_repair_measurement.v2", "harness_sha256": {},
        "source_root": str(tmp_path), "imported_source_sha256": {relative: digest}, "status": "failed"}
    run = {"state": "failed", "cwd": str(tmp_path), "source_sha256": {}}
    comparison.current_provenance(run, measurement, "after", {}, {relative: digest})
    for baseline in ({}, {relative: "different"}):
        with pytest.raises(ValueError, match="changed during measurement"):
            comparison.current_provenance(run, measurement, "after", {}, baseline)


def test_candidate_evidence_reuse_requires_shared_repository_and_exact_sources(tmp_path, monkeypatch):
    import hashlib

    comparison = load("compare_filter_repair_campaign")
    primary, validation = tmp_path / "primary", tmp_path / "validation"
    primary.mkdir()
    validation.mkdir()
    monkeypatch.setattr(comparison, "ROOT", validation)
    monkeypatch.setattr(comparison, "campaign_output_root", lambda _: primary / "artifacts")
    module = validation / "kernel.py"
    module.write_text("x=1\n")
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    measurement = {"schema": "filter_repair_measurement.v2", "harness_sha256": {},
        "source_root": str(primary), "imported_source_sha256": {"kernel.py": digest}, "status": "failed"}
    run = {"state": "failed", "cwd": str(primary), "source_sha256": {"kernel.py": digest}}
    comparison.current_provenance(run, measurement, "after", {}, {})
    with pytest.raises(ValueError, match="source root mismatch"):
        comparison.current_provenance({**run, "cwd": str(validation)}, measurement, "after", {}, {})
    with pytest.raises(ValueError, match="changed during measurement"):
        comparison.current_provenance({**run, "source_sha256": {}}, measurement, "after", {}, {})
    module.write_text("x=2\n")
    with pytest.raises(ValueError, match="Stale candidate source"):
        comparison.current_provenance(run, measurement, "after", {}, {})
    monkeypatch.setattr(comparison, "campaign_output_root", lambda root: root / "artifacts")
    with pytest.raises(ValueError, match="different campaign repository"):
        comparison.current_provenance(run, measurement, "after", {}, {})


def test_investigation_review_requires_current_runs_and_cannot_waive_growth(tmp_path):
    comparison = load("compare_filter_repair_campaign")
    (tmp_path / "review.md").write_text("Mechanism and alternatives checked on current repeated runs.\n")
    finding = {"fixture": "tt","size": 2,"jit": "on","reasons": ["warm_time_over_20_percent"],"runs": ["run-a", "run-b", "run-c"]}
    review = {**finding,"evidence":["review.md"],"disposition":"accept_documented_tradeoff",
        "mechanism":"Native recurrence overhead measured for the tiny fixture.",
        "alternatives_checked":"Unrolling violates the fixed-graph contract.",
        "reason_for_acceptance":"The larger fixture remains within the declared operating budget.",
        "limitations":"No speedup claim for this fixture."}
    pending,resolved = comparison.reviewed_investigations([finding],[review],tmp_path)
    assert not pending and len(resolved) == 1
    for bad in ({**review,"runs":["stale"]},{**review,"evidence":[]},
                {**review,"mechanism":""},{**review,"evidence":["../review.md"]}):
        assert comparison.reviewed_investigations([finding],[bad],tmp_path)[0] == [finding]
    growth = {**finding,"reasons":["continuing_device_allocation_growth"]}
    assert comparison.reviewed_investigations([growth],[{**review,**growth}],tmp_path)[0] == [growth]
