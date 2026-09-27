"""Independent corruption checks for complete-initializer capacity evidence."""

import json
from copy import deepcopy

import pytest

from scripts.analyze_filter_repair_posterior_initializer_capacity import (
    group_identity,
    memory_trajectory,
    source_differences,
    validate_capacity,
    validate_measurements,
    worker_vetoes,
)


def fixture(kind="reuse", arm="xla"):
    payload = {"accepted": True, "center": [.13], "covariance_theta": [[.64]],
        "status": "accepted", "condition_number": 3.}
    result = {"schema": "filter_posterior_initializer_capacity.v1", "reference_revision": "031692a0b",
        "dimension": 1, "kind": kind, "arm": arm, "gpu": False,
        "original_source_sha256": {"frozen-reference.py": "original"},
        "independent_gaussian": {"mean": [.13], "covariance": [[.64]]}}
    if kind == "reuse":
        result.update(warm_calls=20, records=[deepcopy(payload) for _ in range(22)],
            original_records=[deepcopy(payload) for _ in range(2)], elapsed_seconds=[.1] * 22,
            stages={key: {} for key in ("prepared", "cold", "changed", "warm_1", "warm_5", "warm_10", "warm_15", "warm_20")},
            program=None if arm == "prior" else {"jit_compile": arm == "xla", "trace_count": 1,
                "hlo_unchanged": True, "hlo_bytes": 100, "host_callback_ops": [], "nested_xla_function_count": 1},
            reuse=[] if arm == "prior" else [{"owner_reused": True, "traces": 1, "dependency_count": 10} for _ in range(20)])
    else:
        result.update(owner_count=4, records=[deepcopy(payload) for _ in range(4)],
            original_records=[deepcopy(payload)], elapsed_seconds=[.1] * 4,
            superseded_collection=[[[True] * 4 for _ in range(i)] for i in range(4)],
            final_collection=[[True] * 4 for _ in range(4)],
            programs=[{"jit_compile": True, "traces": 1, "dependency_count": 10} for _ in range(4)])
    memory = {"rollup": {"Rss": 1024**2}, "map_count": 10, "gpu": None}
    if kind == "reuse":
        result['stages'] = {name: deepcopy(memory) for name in result['stages']}
        result['call_memory'] = [deepcopy(memory) for _ in range(22)]
        result['owner_returns'] = [] if arm == "prior" else [
            {"memory": deepcopy(memory), "output_bytes": 512} for _ in range(21)]
    else:
        result['stages'] = {name: deepcopy(memory) for name in (
            'prepared', 'all_python_owners_released',
            *(f'owner_{n}_{label}' for n in range(1, 5) for label in ('completed', 'collected')))}
    return result


@pytest.mark.parametrize("kind,arm", [("reuse", "prior"), ("reuse", "graph"), ("reuse", "xla"), ("owner_replacement", "xla")])
def test_valid_capacity_records(kind, arm):
    validate_capacity(fixture(kind, arm))
    validate_measurements(fixture(kind, arm))


@pytest.mark.parametrize("failure", ["short", "replay", "original", "gaussian", "trace", "callback", "hlo", "owner", "dependencies", "timing"])
def test_reuse_corruption_is_rejected(failure):
    result = fixture()
    if failure == "short":
        result["records"].pop()
    elif failure == "replay":
        result["records"][10]["center"] = [.14]
    elif failure == "original":
        result["original_records"][0]["condition_number"] = 4.
    elif failure == "gaussian":
        result["independent_gaussian"]["covariance"] = [[.63]]
    elif failure == "trace":
        result["reuse"][3]["traces"] = 2
    elif failure == "callback":
        result["program"]["host_callback_ops"] = ["PyFunc"]
    elif failure == "hlo":
        result["program"]["hlo_unchanged"] = False
    elif failure == "owner":
        result["reuse"][4]["owner_reused"] = False
    elif failure == "dependencies":
        result["reuse"][7]["dependency_count"] += 1
    else:
        result["elapsed_seconds"][0] = float("nan")
    with pytest.raises(AssertionError):
        validate_capacity(result)


@pytest.mark.parametrize("failure", ["retained", "short", "superseded", "trace"])
def test_successful_owner_corruption_is_rejected(failure):
    result = fixture("owner_replacement")
    if failure == "retained":
        result["final_collection"][1][0] = False
    elif failure == "short":
        result["superseded_collection"][3].pop()
    elif failure == "superseded":
        result["superseded_collection"][2][0][2] = False
    else:
        result["programs"][3]["traces"] = 2
    with pytest.raises(AssertionError):
        validate_capacity(result)


def test_capacity_memory_units_and_unavailable_allocator():
    result = {"stages": {"cpu": {"rollup": {"Rss": 2 * 1024**2}, "map_count": 5, "gpu": None},
        "gpu": {"rollup": {"Rss": 3 * 1024**2}, "map_count": 6, "gpu": {"current": 10, "peak": 20}}}}
    rows = memory_trajectory(result)
    assert rows[0]["rss_mib"] == 2 and rows[0]["gpu_current_bytes"] is None
    assert rows[1]["rss_mib"] == 3 and rows[1]["gpu_current_bytes"] == 10
    assert rows[1]["gpu_peak_bytes"] == 20


@pytest.mark.parametrize("failure", ['missing_call', 'missing_return', 'device', 'size', 'owner_stage', 'allocator'])
def test_incomplete_or_invalid_memory_observations_are_rejected(failure):
    result = fixture('owner_replacement' if failure == 'owner_stage' else 'reuse')
    if failure == 'missing_call':
        result['call_memory'].pop()
    elif failure == 'missing_return':
        result['owner_returns'].pop()
    elif failure == 'device':
        result['call_memory'][5]['gpu'] = {'current': 10, 'peak': 10}
    elif failure == 'size':
        result['owner_returns'][2]['output_bytes'] = 0
    elif failure == 'owner_stage':
        del result['stages']['owner_4_collected']
    else:
        result['stages']['cold']['gpu'] = {'current': 20, 'peak': 10}
    with pytest.raises(AssertionError):
        validate_measurements(result)


def test_numerical_source_drift_is_never_a_reporting_exception():
    first = {"bayesfilter/runtime.py": "original", "scripts/run_filter_repair_campaign.py": "old"}
    changed = {**first, "scripts/run_filter_repair_campaign.py": "new",
        "tests/test_filter_repair_posterior_capacity_crash_diagnostic.py": "new"}
    assert set(source_differences(first, changed)) == {
        "scripts/run_filter_repair_campaign.py", "tests/test_filter_repair_posterior_capacity_crash_diagnostic.py"}
    for path in ("bayesfilter/runtime.py", "tests/test_filter_repair_posterior_initializer_capacity.py",
            "tests/filter_repair_frozen_checkpoint.py"):
        with pytest.raises(AssertionError, match="numerical source drift"):
            source_differences(first, {**changed, path: "modified"})


def test_native_failure_without_junit_is_preserved(tmp_path):
    assert worker_vetoes(tmp_path, {"state": "failed", "exit_code": -11}) == [
        "worker_failed", "missing_or_invalid_junit"]
    assert worker_vetoes(tmp_path, {"state": "passed", "exit_code": 0}) == ["missing_or_invalid_junit"]
    (tmp_path / 'junit.xml').write_text('<testsuite><testcase><failure /></testcase></testsuite>')
    assert worker_vetoes(tmp_path, {"state": "passed", "exit_code": 0}) == ["junit_failed"]


def test_capacity_cell_identity_is_not_caller_invented():
    assert group_identity("posterior_initializer_capacity_reuse_prior_3_cpu", "CPU") == {
        "kind": "reuse", "arm": "prior", "dimension": 3, "device": "CPU"}
    assert group_identity("posterior_initializer_capacity_owner_1_gpu", "GPU")["kind"] == "owner_replacement"
    for group, device in (("unrecognized", "CPU"), ("posterior_initializer_capacity_reuse_prior_3_cpu", "GPU")):
        with pytest.raises(AssertionError):
            group_identity(group, device)


def test_failed_baseline_remains_in_complete_capacity_matrix(tmp_path, monkeypatch):
    from scripts import analyze_filter_repair_posterior_initializer_capacity as analyzer

    monkeypatch.setattr(analyzer, "validate_cost_device", lambda *args: None)
    memory = {"rollup": {"Rss": 1024**2}, "map_count": 10, "gpu": None}
    number = 0
    for dimension in (1, 3):
        for kind, arm in (("reuse", "prior"), ("reuse", "graph"), ("reuse", "xla"), ("owner_replacement", "xla")):
            number += 1
            directory = tmp_path / f'run-{number:05d}'
            directory.mkdir()
            failed = dimension == 3 and arm == "prior"
            group = f'posterior_initializer_capacity_reuse_{arm}_{dimension}_cpu' if kind == "reuse" else f'posterior_initializer_capacity_owner_{dimension}_cpu'
            run = {"device": "CPU", "key": ["test", group], "state": "failed" if failed else "passed",
                "exit_code": -11 if failed else 0, "source_sha256": {"bayesfilter/runtime.py": "fixed"}, "environment": {}}
            (directory / 'run.json').write_text(json.dumps(run))
            (directory / 'process.log').write_text('{"tensorflow_version": "fixture", "tf32_enabled": true}\n')
            if failed:
                continue
            (directory / 'junit.xml').write_text('<testsuite><testcase /></testsuite>')
            record = fixture(kind, arm)
            record.update(dimension=dimension, gpu=False, hardware={}, gpu_process_observation={}, fixture={"dimension": dimension})
            (directory / 'posterior-initializer-capacity.json').write_text(json.dumps(record))
    number += 1
    directory = tmp_path / f'run-{number:05d}'
    directory.mkdir()
    run = {"device": "CPU", "key": ["test", "posterior_initializer_capacity_observer_cpu"],
        "state": "passed", "exit_code": 0, "source_sha256": {"bayesfilter/runtime.py": "fixed"}, "environment": {}}
    (directory / 'run.json').write_text(json.dumps(run))
    (directory / 'process.log').write_text('{"tensorflow_version": "fixture", "tf32_enabled": true}\n')
    (directory / 'junit.xml').write_text('<testsuite><testcase /></testsuite>')
    (directory / 'posterior-initializer-capacity-observer.json').write_text(json.dumps({
        "schema": "filter_posterior_initializer_capacity_observer.v1",
        "gpu": False, "hardware": {}, "gpu_process_observation": {}, "calls_between_snapshots": 0,
        "completed_records_retained": 22, "owner_records_retained": 4,
        "stages": {"prepared": memory}, "owner_stages": {"prepared": memory}}))
    report = analyzer.analyze(tmp_path, 1, number, ["CPU"])
    failed, = [row for row in report['results'] if not row['valid']]
    assert failed['arm'] == 'prior' and failed['dimension'] == 3
    assert failed['memory'] is None and failed['warm_median_seconds'] is None
    assert failed['vetoes'] == ['worker_failed', 'missing_or_invalid_junit', 'missing_complete_capacity_record']
    assert len([row for row in report['results'] if row['valid']]) == 7
    path = tmp_path / 'run-00002/posterior-initializer-capacity.json'
    modified = json.loads(path.read_text())
    modified['original_source_sha256']['frozen-reference.py'] = 'changed'
    path.write_text(json.dumps(modified))
    with pytest.raises(AssertionError, match='original source drift'):
        analyzer.analyze(tmp_path, 1, number, ['CPU'])
