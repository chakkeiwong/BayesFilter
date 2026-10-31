"""Diagnostic numerical veto and threshold checks for initializer cost reports."""

from copy import deepcopy

import pytest

from scripts.analyze_filter_repair_posterior_initializer_costs import (
    compare_records,
    summarize_pair,
    validate_numerics,
)


def record(arm="xla"):
    payload = {"accepted": True, "center": [.13], "covariance_theta": [[.64]],
        "diagnostics": {"config": {"locator_config": {"jit_compile": arm != "graph"}}},
        "condition_number": 3., "ledger": [{"position": [.13], "count": 7}]}
    original = deepcopy(payload)
    original["diagnostics"]["config"]["locator_config"]["jit_compile"] = True
    return {"schema": "filter_posterior_initializer_cost.v1", "arm": arm, "dimension": 1,
        "reference_revision": "031692a0b", "reference_entire_module": True,
        "jit_compile": arm != "graph", "samples": [{}, {}, {}],
        "result": payload, "changed_result": deepcopy(payload),
        "warm_records": [deepcopy(payload) for _ in range(3)],
        "original_records": [original, deepcopy(original)],
        "independent_gaussian": {"mean": [.13], "covariance": [[.64]]},
        "program": None if arm == "prior" else {"trace_count": 1, "jit_compile": arm == "xla",
            "host_callback_ops": [], "hlo_bytes": 10, "hlo_unchanged": True}}


@pytest.mark.parametrize("arm", ["prior", "graph", "xla"])
def test_valid_cost_numerics(arm):
    validate_numerics(record(arm))


@pytest.mark.parametrize("failure", ["replay", "condition", "gaussian", "trace", "callback", "compile"])
def test_cost_numerical_vetoes(failure):
    value = record()
    if failure == "replay":
        value["warm_records"][2]["center"] = [.14]
    elif failure == "condition":
        value["original_records"][0]["condition_number"] = 4.
    elif failure == "gaussian":
        value["independent_gaussian"]["mean"] = [4.]
    elif failure == "trace":
        value["program"]["trace_count"] = 2
    elif failure == "callback":
        value["program"]["host_callback_ops"] = ["PyFunc"]
    elif failure == "compile":
        value["program"]["jit_compile"] = False
    with pytest.raises(AssertionError):
        validate_numerics(value)


def test_only_boolean_execution_metadata_can_differ():
    compare_records({"jit_compile": False, "value": 1.}, {"jit_compile": True, "value": 1.})
    with pytest.raises(AssertionError):
        compare_records({"jit_compile": "false"}, {"jit_compile": True})
    with pytest.raises(AssertionError):
        compare_records({"value": 2.}, {"value": 1.})
    with pytest.raises(AssertionError):
        compare_records({"value": float("nan")}, {"value": float("nan")})


def test_cost_triggers_are_not_acceptance():
    prior = [{"cold_seconds": 1., "warm_seconds": 1., "observed_rss_mib": 200.,
        "gpu_allocator_peak_bytes": 100} for _ in range(3)]
    candidate = [{"cold_seconds": 2.1, "warm_seconds": 1.21, "observed_rss_mib": 500.,
        "gpu_allocator_peak_bytes": 201} for _ in range(3)]
    assert summarize_pair(prior, candidate)["investigation_triggers"] == ["cold_above_2x",
        "warm_above_20_percent", "host_rss_above_256MiB_or_2x", "gpu_allocator_peak_above_2x"]
    assert not summarize_pair(prior, prior)["investigation_triggers"]


def cohort(tmp_path, monkeypatch, *, bad_graph=False):
    import json

    from scripts import analyze_filter_repair_posterior_initializer_costs as analyzer

    monkeypatch.setattr(analyzer, "validate_cost_device", lambda *_: None)
    memory = {"rollup": {"Rss": 100 * 1024**2}, "ru_maxrss_bytes": 100 * 1024**2,
        "map_count": 100, "gpu": None}
    number = 0
    for dimension in (1, 3):
        for repeat in (0, 1, 2):
            for arm in ("prior", "graph", "xla"):
                number += 1
                directory = tmp_path / f"run-{number:05d}"
                directory.mkdir()
                result = record(arm)
                result.update(dimension=dimension, gpu=False, hardware={"fixture": True},
                    original_source_sha256={"reference": "pinned"},
                    input_sha256=["input"], changed_input_sha256=["changed"], target_sha256=["target"],
                    config={}, movement_config={}, thresholds={},
                    cold={"seconds": 1., "memory": memory},
                    changed_cost={"seconds": .1, "memory": memory},
                    samples=[{"seconds": .1, "memory": memory} for _ in range(3)],
                    stages={stage: memory for stage in ("prepared", "cold", "warm", "changed")},
                    gpu_process_observation={})
                failure = arm == "graph" and bad_graph
                if failure:
                    result["original_records"][0]["condition_number"] = 5.
                run = {"device": "CPU", "key": ["test", f"posterior_initializer_cost_{arm}_{dimension}_cpu",
                    "after", None, None, None, repeat], "source_sha256": {"runtime": "frozen"},
                    "environment": {}, "state": "failed" if failure else "passed", "exit_code": int(failure)}
                (directory / "run.json").write_text(json.dumps(run))
                (directory / "posterior-initializer-cost.json").write_text(json.dumps(result))
                (directory / "junit.xml").write_text('<testsuite><testcase>'
                    + ('<failure />' if failure else '') + '</testcase></testsuite>')
                (directory / "process.log").write_text(json.dumps({"tensorflow_version": "test", "tf32_enabled": True}) + '\n')
    return analyzer, number


def test_failed_graph_is_preserved_without_speed_ratios(tmp_path, monkeypatch):
    analyzer, last = cohort(tmp_path, monkeypatch, bad_graph=True)
    report = analyzer.analyze(tmp_path, 1, last, ["CPU"])
    for row in report["comparisons"]:
        assert row["prior_pairs"]["graph"] is None
        assert row["prior_pairs"]["xla"] is not None
        assert all(not item["valid"] and item["vetoes"] for item in row["arms"]["graph"])


def test_mixed_sources_cannot_be_paired(tmp_path, monkeypatch):
    import json
    analyzer, last = cohort(tmp_path, monkeypatch)
    path = tmp_path / 'run-00003/run.json'
    run = json.loads(path.read_text())
    run['source_sha256']['runtime'] = 'changed'
    path.write_text(json.dumps(run))
    with pytest.raises(AssertionError, match='source drift'):
        analyzer.analyze(tmp_path, 1, last, ['CPU'])


def test_missing_process_repeat_blocks_summary(tmp_path, monkeypatch):
    analyzer, last = cohort(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match='Three fresh processes'):
        analyzer.analyze(tmp_path, 1, last - 1, ['CPU'])
