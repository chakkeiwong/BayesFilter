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


def gpu_preflight(index):
    return [{"selected_gpu_index": index, "selected_uuid": f"GPU-test-{index}",
             "performance_preflight_uncontended": True}] * 2


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


def test_small_test_reserves_and_enforces_its_timeout_without_expanding_budget(tmp_path, monkeypatch):
    import argparse

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "records", list)
    monkeypatch.setattr(driver, "charged_seconds", lambda *_: 239.)
    monkeypatch.setattr(driver, "BUDGET_SECONDS", {"CPU": 300})
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "git", lambda *_: "test")
    monkeypatch.setattr(driver, "test_evidence", lambda *_: {"passed": True})
    waits = []

    class Worker:
        def wait(self, *, timeout):
            waits.append(timeout)
            return 0

    monkeypatch.setattr(driver.subprocess, "Popen", lambda *_, **kwargs: Worker())
    args = argparse.Namespace(action="test", device="CPU", group="policy", arm="after",
        fixture="dns", jit="on", size=1, repeat=0, test_timeout_seconds=60)
    assert driver.run_job(args) == 0
    run = json.loads((tmp_path / "run-00001/run.json").read_text())
    assert run["timeout_seconds"] == 60 and waits == [60]
    monkeypatch.setattr(driver, "charged_seconds", lambda *_: 241.)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        driver.run_job(args)
    args.test_timeout_seconds = 901
    with pytest.raises(ValueError, match="bounded registered limits"):
        driver.run_job(args)


@pytest.mark.parametrize("fixture,ceiling", [("fixed_fitting", 900), ("fixed_selection", 300)])
def test_full_fit_measurement_ceiling_preserves_campaign_budget(tmp_path, monkeypatch, fixture, ceiling):
    import argparse

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "records", list)
    monkeypatch.setattr(driver, "charged_seconds", lambda *_: 100.)
    monkeypatch.setattr(driver, "BUDGET_SECONDS", {"CPU": 1000})
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "git", lambda *_: "test")
    monkeypatch.setattr(driver, "ensure_baseline", lambda: None)
    waits = []

    class Worker:
        def wait(self, *, timeout):
            waits.append(timeout)
            return 0

    monkeypatch.setattr(driver.subprocess, "Popen", lambda *_, **kwargs: Worker())
    args = argparse.Namespace(action="measure", device="CPU", group="policy", arm="after",
        fixture=fixture, jit="on", size=2, repeat=0)
    assert driver.run_job(args) == 0
    run = json.loads((tmp_path / "run-00001/run.json").read_text())
    assert run["timeout_seconds"] == ceiling and waits == [ceiling]
    monkeypatch.setattr(driver, "charged_seconds", lambda *_: 1001. - ceiling)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        driver.run_job(args)


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
    pool = driver.measurement_harness("cpu_forecast_shard")
    assert {name: pool[name] for name in original} == original
    assert set(pool) - set(original) == {
        "filter_repair_forecast_pool_worker.py", "filter_repair_forecast_pool_fixtures.py",
    }
    assert driver.measurement_device("cpu_forecast_shard") == "CPU"
    assert driver.measurement_device("cpu_forecast_pool") == "CPU"
    assert driver.measurement_device("complexity_forecast") == "GPU"
    assert driver.measurement_modes("cpu_forecast_shard") == ("off", "on", "eager")
    assert driver.measurement_modes("cpu_forecast_pool") == ("eager",)
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


def test_test_roles_exclude_only_reviewed_explanatory_jobs():
    driver = load("run_filter_repair_campaign")
    assert set(driver.EXPLANATORY_TEST_GROUPS) <= set(driver.TEST_GROUPS)
    assert all(isinstance(reason, str) and reason.strip()
        for reason in driver.EXPLANATORY_TEST_GROUPS.values())
    required = set(driver.mandatory_test_groups())
    assert not required.intersection(driver.EXPLANATORY_TEST_GROUPS)
    assert required | set(driver.EXPLANATORY_TEST_GROUPS) == set(driver.TEST_GROUPS)
    assert "initializer_native_residual_cpu" in driver.EXPLANATORY_TEST_GROUPS
    assert set(driver.TEST_BATCHES["initializer_native_cpu"]) <= required
    assert set(driver.TEST_BATCHES["initializer_native_gpu"]) <= required
    assert set(driver.TEST_BATCHES["posterior_public_cpu"]) <= required
    assert set(driver.TEST_BATCHES["posterior_public_gpu"]) <= required
    assert set(driver.TEST_BATCHES["sequential_controller_cpu"]) <= required
    assert set(driver.TEST_BATCHES["sequential_controller_gpu"]) <= required
    assert set(driver.TEST_BATCHES["sequential_public_cpu"]) <= required
    assert set(driver.TEST_BATCHES["sequential_public_gpu"]) <= required
    assert set(driver.TEST_BATCHES["sequential_public_consumers_cpu"]) <= required
    assert set(driver.TEST_BATCHES["sequential_public_consumers_gpu"]) <= required
    for device in ("cpu", "gpu"):
        boundary = driver.TEST_GROUPS[f"sequential_public_consumers_boundary_{device}"]
        assert "tests/test_filter_repair_lifecycle_original.py::test_original_full_lifecycle_records_and_target_order[symmetric]" in boundary
        assert not any("test_filter_repair_lifecycle_actual.py" in target for target in boundary)
    assert {"factor_domain_runtime", "factor_guard_qualification", "factor_guard_cpu_fixed",
        "factor_guard_cpu_padded", "factor_guard_cpu_domain", "factor_guard_resource_lifetime",
        "factor_guard_gpu_lifetime", "active_cod_runtime", "padded_factor", "factor_runtime_inputs",
        "fixed_fitting", "fixed_fitting_consumers", "source_guard_localization",
        "source_preparation_localization", "quadratic_initializer_localization"} <= required
    for batch in driver.TEST_BATCHES.values():
        assert set(batch) <= set(driver.TEST_GROUPS)
        assert len(batch) == len(set(batch))
    assert set(driver.TEST_BATCHES["factor_guard"]) <= required
    assert driver.TEST_DEVICES["factor_guard_qualification"] == "GPU"
    assert all(driver.TEST_DEVICES.get(group, "CPU") == "CPU" for group in
        ("factor_guard_cpu_fixed", "factor_guard_cpu_padded", "factor_guard_cpu_domain"))


def test_public_cost_preflight_declines_shared_gpu_before_worker_launch(tmp_path, monkeypatch):
    import argparse

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    args = argparse.Namespace(action="test", device="GPU",
        group="posterior_public_memory_xla_3_gpu", gpu_uuid="GPU-test-2",
        gpu_preflight=[{"performance_preflight_uncontended": False}] * 2)
    with pytest.raises(RuntimeError, match="declined before launch"):
        driver.require_unshared_cost_preflight(args)
    record, = tmp_path.glob("cost-preflight-declined-*.json")
    assert json.loads(record.read_text())["worker_launched"] is False
    args.gpu_preflight = [{"performance_preflight_uncontended": True}] * 2
    driver.require_unshared_cost_preflight(args)
    args.group = "posterior_public_3_gpu"
    args.gpu_preflight = [{"performance_preflight_uncontended": False}] * 2
    driver.require_unshared_cost_preflight(args)


def test_intermediate_authority_disposition_keeps_original_numerical_vetoes(monkeypatch):
    driver = load("run_filter_repair_campaign")
    mandatory = set(driver.mandatory_test_groups())
    replacements = driver.ORIGINAL_AUTHORITY_REPLACEMENTS
    assert len(replacements) == 16
    assert set(replacements.values()) <= mandatory
    assert set(replacements) <= set(driver.EXPLANATORY_TEST_GROUPS)
    assert {"lifecycle_original_runtime_5_cpu", "lifecycle_original_runtime_gpu",
            "terminal_original_cpu", "terminal_original_gpu", "refinement_original_gpu"} <= mandatory
    group, original = next(iter(replacements.items()))
    monkeypatch.setattr(driver, "TEST_GROUPS", {group: ()})
    with pytest.raises(ValueError, match="mandatory original-source"):
        driver.mandatory_test_groups()
    monkeypatch.setattr(driver, "TEST_GROUPS", {group: (), original: ()})
    monkeypatch.setattr(driver, "EXPLANATORY_TEST_GROUPS", {group: "historical", original: "waived"})
    with pytest.raises(ValueError, match="mandatory original-source"):
        driver.mandatory_test_groups()


def test_test_matrix_skips_exact_explanations_but_fails_new_runtime_group(tmp_path, monkeypatch):
    import argparse

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "TEST_GROUPS", {"factor_domain_guard_trial": (), "new_runtime": ()})
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "records", list)
    executed = []

    def execute(job):
        executed.append(job.group)
        return 4

    monkeypatch.setattr(driver, "run_job", execute)
    assert driver.run_matrix(argparse.Namespace(stage="tests")) == 4
    assert executed == ["new_runtime"]


def test_explicit_test_batch_executes_registered_diagnostics(tmp_path, monkeypatch):
    import argparse

    driver = load("run_filter_repair_campaign")
    group = "factor_domain_guard_trial"
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "TEST_BATCHES", {"diagnostic": (group,)})
    monkeypatch.setattr(driver, "TEST_DEVICES", {})
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "records", list)
    executed = []
    monkeypatch.setattr(driver, "run_job", lambda job: executed.append(job.group) or 0)
    assert driver.run_matrix(argparse.Namespace(stage="tests", test_batch="diagnostic")) == 0
    assert executed == [group]


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
@pytest.mark.parametrize("gpu_index", [0, 1, 2, 3])
def test_matrix_prefers_valid_graph_reference_before_eager(tmp_path, monkeypatch, graph_passes, gpu_index):
    import argparse

    import compare_filter_repair_campaign as comparison

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    (tmp_path / "source-manifest.json").write_text(json.dumps({"files": {}}))
    monkeypatch.setattr(driver, "BASELINE_ROOT", tmp_path)
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "ensure_baseline", lambda: None)
    monkeypatch.setattr(driver, "sha", lambda _: "fixed")
    monkeypatch.setattr(driver, "check_gpu_available", lambda _: gpu_preflight(gpu_index))
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
                         "device": "GPU", "gpu_uuid": f"GPU-test-{gpu_index}",
                         "environment": {"CUDA_VISIBLE_DEVICES": f"GPU-test-{gpu_index}"},
                         "result": str(path), "state": value["status"]})
    monkeypatch.setattr(driver, "records", lambda: rows)
    monkeypatch.setattr(driver, "run_job", lambda _: pytest.fail("Valid reference already exists"))
    monkeypatch.setattr(comparison, "current_provenance", lambda *_: None)
    pairs = []
    monkeypatch.setattr(comparison, "compare_pair",
                        lambda before, after: pairs.append((before["jit"], after["jit"])) or 0.0)
    args = argparse.Namespace(stage="qualify", selection="fixture", fixture="rectangular", measurement_gpu_index=gpu_index)
    assert driver.run_matrix(args) == 0
    expected = "off" if graph_passes else "eager"
    assert pairs == [(expected, "off"), (expected, "on")]


@pytest.mark.parametrize("prior_index,stage,uncontended", ((2, "qualify", True), (3, "repeat", False), (3, "repeat", None)))
def test_measurement_matrix_rejects_other_or_shared_gpu_evidence(tmp_path, monkeypatch, prior_index, stage, uncontended):
    import argparse
    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "BASELINE_ROOT", tmp_path)
    (tmp_path / "source-manifest.json").write_text('{"files": {}}')
    result = tmp_path / "prior.json"
    result.write_text('{"status": "passed"}')
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "ensure_baseline", lambda: None)
    monkeypatch.setattr(driver, "records", lambda: [{"key": ["measure", "policy", "before", "rectangular", "off", 1, 0, "GPU"],
        "device": "GPU", "gpu_uuid": f"GPU-test-{prior_index}",
        "gpu_performance_preflight_uncontended": uncontended,
        "environment": {"CUDA_VISIBLE_DEVICES": f"GPU-test-{prior_index}"}, "result": str(result)}])
    selected = []
    monkeypatch.setattr(driver, "check_gpu_available", lambda index: selected.append(index) or gpu_preflight(index))

    def launch(job):
        assert job.measurement_gpu_index == 3 and job.arm == "before"
        raise RuntimeError("fresh_worker_required")

    monkeypatch.setattr(driver, "run_job", launch)
    with pytest.raises(RuntimeError, match="fresh_worker_required"):
        driver.run_matrix(argparse.Namespace(stage=stage, selection="fixture", fixture="rectangular", measurement_gpu_index=3))
    assert selected == [3, 3]


@pytest.mark.parametrize("gpu_index", (0, 1, 2, 3))
def test_gpu_test_group_uses_gpu_and_preflight(tmp_path, monkeypatch, gpu_index):
    import argparse
    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "TEST_GROUPS", {"random_gpu": ()})
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "records", list)
    checked = []
    monkeypatch.setattr(driver, "check_gpu_available", lambda index: checked.append(index) or gpu_preflight(index))
    jobs = []
    monkeypatch.setattr(driver, "run_job", lambda job: jobs.append(job) or 0)
    assert driver.run_matrix(argparse.Namespace(stage="tests", test_gpu_index=gpu_index)) == 0
    assert jobs[0].device == "GPU" and jobs[0].gpu_preflight == gpu_preflight(gpu_index)
    assert checked == [gpu_index, gpu_index] and jobs[0].test_gpu_index == gpu_index


def test_alternate_gpu_option_cannot_change_measurement_device(monkeypatch):
    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(sys, "argv", ["driver", "measure", "--test-gpu-index", "3"])
    monkeypatch.setattr(driver, "run_job", lambda _: pytest.fail("Invalid device selection must not launch"))
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2


@pytest.mark.parametrize("prior_uuid", (None, "GPU-test-2", "GPU-replaced-3", "GPU-test-3"))
def test_test_matrix_pins_auto_selection_and_rejects_other_gpu_evidence(tmp_path, monkeypatch, prior_uuid):
    import argparse

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "TEST_GROUPS", {"first_gpu": (), "second_gpu": ()})
    monkeypatch.setattr(driver, "TEST_DEVICES", {"first_gpu": "GPU", "second_gpu": "GPU"})
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "test_evidence", lambda _: {"passed": True})
    rows = [{"key": ["test", group, "after", "covariance", "on", 1, 0, "GPU"],
             "state": "passed", "source_sha256": {}, "device": "GPU", "gpu_uuid": prior_uuid}
            for group in driver.TEST_GROUPS]
    monkeypatch.setattr(driver, "records", lambda: rows)
    checked, jobs = [], []
    monkeypatch.setattr(driver, "check_gpu_available", lambda index: checked.append(index) or gpu_preflight(3))
    monkeypatch.setattr(driver, "run_job", lambda job: jobs.append(job) or 0)
    assert driver.run_matrix(argparse.Namespace(stage="tests")) == 0
    if prior_uuid == "GPU-test-3":
        assert checked == [None] and jobs == []
        return
    assert checked == [None, 3, 3]
    assert [job.group for job in jobs] == ["first_gpu", "second_gpu"]
    assert all(job.gpu_uuid == "GPU-test-3" for job in jobs)


def test_matrix_stops_if_pinned_index_changes_physical_identity(monkeypatch):
    import argparse

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "check_gpu_available", lambda _: gpu_preflight(2))
    args = argparse.Namespace(test_gpu_index=2, gpu_uuid="GPU-replaced-2")
    with pytest.raises(RuntimeError, match="Physical GPU identity changed"):
        driver.prepare_gpu(args, "test_gpu_index")


def test_auto_gpu_worker_uses_and_records_uuid(tmp_path, monkeypatch):
    import argparse

    driver = load("run_filter_repair_campaign")
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "records", list)
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "git", lambda *_: "test")
    monkeypatch.setattr(driver, "test_evidence", lambda *_: {"passed": True})
    monkeypatch.setattr(driver, "check_gpu_available", lambda _: gpu_preflight(2))
    launches = []

    class Worker:
        def wait(self, *, timeout):
            return 0

    def launch(command, **kwargs):
        launches.append(kwargs["env"])
        return Worker()

    monkeypatch.setattr(driver.subprocess, "Popen", launch)
    args = argparse.Namespace(action="test", device="GPU", group="policy", arm="after",
        fixture="dns", jit="on", size=1, repeat=0, test_timeout_seconds=60)
    assert driver.run_job(args) == 0
    run = json.loads((tmp_path / "run-00001/run.json").read_text())
    assert launches[0]["CUDA_VISIBLE_DEVICES"] == run["gpu_uuid"] == "GPU-test-2"
    assert run["gpu_preflight"] == gpu_preflight(2)
    assert run["gpu_performance_preflight_uncontended"]


def test_repeat_aggregation_rejects_mixed_physical_devices():
    comparison = load("compare_filter_repair_campaign")
    original = {"hardware_scope": {"environment": {"CUDA_VISIBLE_DEVICES": "2"}}}
    alternate = {"hardware_scope": {"environment": {"CUDA_VISIBLE_DEVICES": "3"}}}
    comparison.validate_repeat_hardware([original] * 3)
    comparison.validate_repeat_hardware([alternate] * 3)
    with pytest.raises(ValueError, match="same physical GPU"):
        comparison.validate_repeat_hardware([original, alternate, alternate])


@pytest.mark.parametrize("uncontended", (None, False, True))
def test_shared_gpu_cannot_qualify_terminal_performance(uncontended):
    comparison = load("compare_filter_repair_campaign")
    run = {"device": "GPU", "gpu_uuid": "GPU-test-2", "gpu_performance_preflight_uncontended": uncontended}
    if uncontended:
        comparison.validate_performance_preflight(run)
    else:
        with pytest.raises(ValueError, match="terminal performance"):
            comparison.validate_performance_preflight(run)
    comparison.validate_performance_preflight({**run, "device": "CPU"})


def test_baseline_python_tensor_condition_is_a_tracing_failure_only():
    comparison = load("compare_filter_repair_campaign")
    failure = {"phase": "trace", "error_type": "OperatorNotAllowedInGraphError",
               "error": "Using a symbolic `tf.Tensor` as a Python `bool` is not allowed."}
    assert comparison.baseline_compilation_failure(failure) == "baseline_host_operation_during_trace"
    assert comparison.baseline_compilation_failure({**failure, "phase": "first_execution"}) is None
    assert comparison.baseline_compilation_failure({**failure, "error_type": "RuntimeError"}) is None


def test_whole_endpoint_and_kernel_timings_cannot_issue_a_speed_ratio():
    from compare_filter_repair_campaign import performance_comparison

    driver = load("run_filter_repair_campaign")
    before = dict(warm_median_seconds=2., device_peak_bytes=100, host_peak_bytes=100,
                  late_device_growth_bytes=0, late_host_growth_bytes=0)
    after = {**before, "warm_median_seconds": .01, "device_peak_bytes": 1000,
             "late_device_growth_bytes": 1}
    ratio, reasons = performance_comparison(before, after, same_timing_scope=False)
    assert ratio is None
    assert reasons == ["continuing_device_allocation_growth"]
    ratio, reasons = performance_comparison(before, after, same_timing_scope=True)
    assert ratio == .005
    assert "device_peak_over_2x" in reasons
    assert driver.measurement_modes("source_route_sequence") == ("off", "on", "eager")


def test_pool_comparison_rejects_missing_or_contaminated_child_sources():
    comparison = load("compare_filter_repair_campaign")
    value = {"schema": "filter_repair_measurement.v2", "harness_sha256": {},
        "fixture": "cpu_forecast_pool", "status": "passed", "source_root": "/baseline",
        "imported_source_sha256": {"bayesfilter/worker.py": "before"},
        "pool_calls": [{"configured_worker_count": 2, "startup_worker_pids": [10, 20]}]}
    with pytest.raises(ValueError, match="Missing process-pool child"):
        comparison.current_provenance({}, value, "before", {}, {})
    value["worker_sources"] = [dict(pid=pid, source_root="/baseline",
        imported_source_sha256={"bayesfilter/worker.py": "after"}) for pid in [10, 20]]
    with pytest.raises(ValueError, match="Process-pool source contamination"):
        comparison.current_provenance({}, value, "before", {}, {})
    value["worker_sources"][0]["source_root"] = "/candidate"
    with pytest.raises(ValueError, match="Invalid process-pool child source root"):
        comparison.current_provenance({}, value, "before", {}, {})


def test_pool_summary_requires_complete_consistent_child_memory():
    comparison = load("compare_filter_repair_campaign")
    value = {"fixture": "cpu_forecast_pool", "preparation_seconds": 0., "trace_seconds": 0.,
        "cold": {"synchronized_seconds": 1.}, "graph": {"nodes": 0}, "stages": {},
        "warm": [{"VmRSS": 10, "VmHWM": 20, "synchronized_seconds": 1., "output_copy_seconds": 0.}]*20}
    with pytest.raises(ValueError, match="Missing per-call"):
        comparison.summarize(value)
    value["pool_calls"] = [{"aggregate_parent_worker_ru_maxrss_bytes": 60,
        "parent_ru_maxrss_bytes": 20, "worker_ru_maxrss_sum_bytes": 40}]*21
    with pytest.raises(ValueError, match="final process-pool memory"):
        comparison.summarize(value)
    value["pool_final_memory"] = {"aggregate_parent_worker_ru_maxrss_bytes": 80,
        "parent_ru_maxrss_bytes": 20, "worker_ru_maxrss_sum_bytes": 60}
    value["worker_sources"] = [{"numerical_ru_maxrss_bytes": 25}, {"numerical_ru_maxrss_bytes": 35}]
    assert comparison.summarize(value)["pool_rss_peak_sum_bytes"] == 80
    value["pool_calls"][-1] = {**value["pool_calls"][-1], "worker_ru_maxrss_sum_bytes": 0}
    with pytest.raises(ValueError, match="Invalid process-pool peak sum"):
        comparison.summarize(value)


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


def test_batch_group_names_validate_before_any_worker_launch(monkeypatch):
    import argparse

    driver = load("run_filter_repair_campaign")
    for groups in driver.TEST_BATCHES.values():
        assert set(groups) <= driver.TEST_GROUPS.keys()
    monkeypatch.setitem(driver.TEST_BATCHES, "invalid", ("policy", "misspelled"))
    monkeypatch.setattr(driver, "source_hashes", dict)
    launched = []
    monkeypatch.setattr(driver, "run_job", lambda args: launched.append(args))
    with pytest.raises(ValueError, match="Unknown test groups"):
        driver.run_matrix(argparse.Namespace(stage="tests", test_batch="invalid"))
    assert launched == []

@pytest.mark.parametrize("recorded,requested,launch", [(0, 0, False), (0, 1, True), (1, 1, False), (1, 2, True)])
def test_test_matrix_resumes_each_requested_repeat_independently(monkeypatch, recorded, requested, launch):
    from types import SimpleNamespace

    driver = load("run_filter_repair_campaign")
    rows = [{"key": ["test", "fixture", "after", "dns", "on", 1, recorded, "CPU"],
        "state": "passed", "source_sha256": {}, "device": "CPU"}]
    monkeypatch.setattr(driver, "TEST_BATCHES", {"fixture": ("fixture",)})
    monkeypatch.setattr(driver, "TEST_GROUPS", {"fixture": ()})
    monkeypatch.setattr(driver, "TEST_DEVICES", {"fixture": "CPU"})
    monkeypatch.setattr(driver, "source_hashes", dict)
    monkeypatch.setattr(driver, "check_matrix_state", lambda _: None)
    monkeypatch.setattr(driver, "records", lambda: rows)
    monkeypatch.setattr(driver, "test_evidence", lambda _: {"passed": True})
    launched = []

    def execute(job):
        launched.append(job.repeat)
        return 0

    monkeypatch.setattr(driver, "run_job", execute)
    assert driver.run_matrix(SimpleNamespace(stage="tests", test_batch="fixture", repeat=requested)) == 0
    assert launched == ([requested] if launch else [])


def test_sequential_public_consumer_partition_covers_every_existing_case_once():
    import ast
    from pathlib import Path

    driver = load("run_filter_repair_campaign")
    source = Path(driver.ROOT, "tests/test_sequential_map_covariance.py")
    expected = {"tests/test_sequential_map_covariance.py::" + node.name
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}
    registered = [case for group in driver.SEQUENTIAL_PUBLIC_CONSUMERS for case in group]
    assert set(registered) == expected
    assert len(registered) == len(set(registered))
    for device in ("cpu", "gpu"):
        assert set(driver.TEST_BATCHES[f"program_ownership_{device}"]) <= set(driver.mandatory_test_groups())
