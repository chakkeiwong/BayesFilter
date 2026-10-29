"""KSC master wiring and complete-schedule budget checks."""
import json
from pathlib import Path

import pytest

from bayesfilter.inference import surrogate_hmc_program as master
from scripts import run_surrogate_hmc_ksc as ksc
from scripts import prepare_surrogate_hmc_ksc as preparation
from scripts import prepare_surrogate_hmc_ksc_finite as finite_preparation


def test_master_dispatches_finite_reference_with_independent_endpoint(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(finite_preparation, "main", lambda args: seen.extend(args) or 3)
    code = master.main(["--stage", "P3-KSC-finite-reference", "--scope", "KSC", "--output", str(tmp_path),
        "--qualification-root", "/q", "--selection", "/s/selection.json", "--posterior-root", "/r",
        "--endpoint-root", "/endpoint", "--previous-accounting", "/extra.json", "--dry-run"])
    assert code == 3
    assert seen[seen.index("--stage")+1] == "reference"
    assert seen[seen.index("--endpoint-root")+1] == "/endpoint"
    assert seen[seen.index("--previous-accounting")+1] == "/extra.json"
    assert "--cost-root" not in seen


@pytest.mark.parametrize("extra", [[], ["--endpoint-root", "/endpoint", "--cost-root", "/cost"]])
def test_finite_reference_rejects_missing_or_conflicting_binding(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(finite_preparation, "main", lambda args: pytest.fail("invalid preparation dispatched"))
    with pytest.raises(SystemExit):
        master.main(["--stage", "P3-KSC-finite-reference", "--scope", "KSC", "--output", str(tmp_path),
            "--qualification-root", "/q", "--selection", "/s/selection.json", "--posterior-root", "/r", *extra])


@pytest.mark.parametrize("substitute_check", [False, True])
def test_finite_endpoint_requires_exact_qualification_checks(tmp_path, monkeypatch, substitute_check):
    monkeypatch.setattr(finite_preparation, "ROOT", tmp_path)
    checks = {name: True for name in finite_preparation.ENDPOINT_CHECKS}
    if substitute_check:
        del checks["canonical_value_call_chain_parity"]
        checks["unrelated_check"] = True
    master.write_json(tmp_path / "data.json", {"observation": 1.})
    master.write_json(tmp_path / "report.json", dict(status="QUALIFIED_FINITE_ENDPOINT", qualified=True,
        checks=checks, endpoint_identity="D", forecast={key: {metric: 1. for metric in
        ("reference", "C_inference", "grid_batch_max", "transition_L2_max")} for key in
        ("cpu_worker_seconds", "gpu_seconds")}))
    master.write_json(tmp_path / "run-manifest.json", dict(exit_code=0, source_hashes={},
        selection_sha256="selection", posterior_report_sha256="R", endpoint_identity="D",
        data_sha256=master.file_hash(tmp_path / "data.json")))
    if substitute_check:
        with pytest.raises(ValueError, match="qualified finite endpoint"):
            finite_preparation.check_endpoint(tmp_path, selection_sha="selection", posterior_sha="R")
    else:
        report, _ = finite_preparation.check_endpoint(tmp_path, selection_sha="selection", posterior_sha="R")
        assert report["qualified"]


def test_public_master_dispatches_checked_ksc_consumer(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ksc, "main", lambda args: seen.extend(args) or 7)
    code = master.main(["--stage", "P3-KSC-inference", "--scope", "KSC", "--output", str(tmp_path),
        "--qualification-root", "/q", "--selection", "/s/selection.json", "--posterior-root", "/r",
        "--cost-root", "/c", "--max-cpu-worker-seconds", "20000", "--dry-run"])
    assert code == 7
    assert seen[seen.index("--controls-root")+1] == "/s"
    assert seen[seen.index("--max-cpu-worker-seconds")+1] == "20000.0"
    assert "--dry-run" in seen


def test_master_rejects_wrong_scope_before_ksc_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(ksc, "main", lambda args: pytest.fail("wrong scope dispatched"))
    with pytest.raises(SystemExit):
        master.main(["--stage", "P3-KSC-inference", "--scope", "LG1", "--output", str(tmp_path)])


def test_complete_forecast_reserves_maximum_trajectory_and_both_looks(tmp_path):
    report = dict(status="COMPLETE_TRANSITION_COST_MEASURED", complete_transition_costs=[
        dict(compilation_included=i == 0, cpu_worker_seconds=20 if i == 0 else i/10,
             wall_seconds=20 if i == 0 else i/10) for i in range(5)])
    (tmp_path / "report.json").write_text(json.dumps(report))
    result = ksc.complete_forecast(tmp_path)
    assert result["required"]["cpu_worker_seconds"] == pytest.approx(1.5 * (7172*.4+330))
    assert result["equivalent_L1_transitions"] == 96+192+192*2+(250+1000+2000)*2


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_forecast_rejects_invalid_cost(tmp_path, bad):
    report = dict(status="COMPLETE_TRANSITION_COST_MEASURED", complete_transition_costs=[
        dict(compilation_included=i == 0, cpu_worker_seconds=bad, wall_seconds=.2) for i in range(5)])
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="invalid measured cost"):
        ksc.complete_forecast(tmp_path)


def test_qualified_reference_rejects_missing_precision(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps(dict(status="REFERENCE_PRECISION_FAILED", qualified=False)))
    (tmp_path / "run-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="qualified same-law"):
        ksc.check_reference(tmp_path)


def test_ksc_forwards_extra_accounting(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ksc, "main", lambda args: seen.extend(args) or 0)
    master.main(["--stage", "P3-KSC-inference", "--scope", "KSC", "--output", str(tmp_path),
        "--qualification-root", "/q", "--selection", "/s/selection.json", "--posterior-root", "/r",
        "--cost-root", "/c", "--previous-accounting", "/extra/accounting.json", "--dry-run"])
    assert seen[seen.index("--previous-accounting") + 1] == "/extra/accounting.json"


@pytest.mark.parametrize("overrides", [["--selection", "/s/other.json"], ["--sampling-timeout=1"],
                                       ["--tuning-root", "/old-tuning"]])
def test_ksc_never_silently_ignores_an_override(tmp_path, monkeypatch, overrides):
    monkeypatch.setattr(ksc, "main", lambda args: pytest.fail("incompatible override launched"))
    with pytest.raises(SystemExit):
        master.main(["--stage", "P3-KSC-inference", "--scope", "KSC", "--output", str(tmp_path),
            "--qualification-root", "/q", "--selection", "/s/selection.json", "--posterior-root", "/r",
            "--cost-root", "/c", *overrides])


def test_extra_accounting_is_charged_once(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    path = tmp_path / "docs/plans/artifacts/surrogate-hmc-all-models-test-ksc-t2-tests-01/accounting.json"
    master.write_json(path, dict(charged_this_run=dict(cpu_worker_seconds=7., gpu_seconds=3.)))
    extra = tmp_path / "extra.json"
    master.write_json(extra, dict(charged=dict(cpu_worker_seconds=5., gpu_seconds=2.)))
    total, family, _ = preparation.budgets([path, extra, extra])
    assert total == dict(cpu_worker_seconds=12., gpu_seconds=5.)
    assert family == dict(cpu_worker_seconds=7., gpu_seconds=3.)


def completed_history(tmp_path):
    directory = tmp_path / "docs/plans/artifacts/surrogate-hmc-all-models-test-ksc-t2-inference-01"
    master.write_json(directory / "data.json", dict(observations=[1., 2.]))
    source = directory / "executed-source/numerical.py"
    source.parent.mkdir()
    source.write_text("archived numerical implementation\n")
    # Today's implementation may differ: reporting must preserve the old source.
    (tmp_path / "numerical.py").write_text("later implementation\n")
    master.write_json(directory / "run-manifest.json", dict(scope="M03-KSC-T2", exit_code=0,
        source_hashes={"numerical.py": master.file_hash(source)}, data_sha256=master.file_hash(directory / "data.json")))
    master.write_json(directory / "accounting.json", dict(exit_code=0))
    report = dict(status="COMPLETED_BOUNDED_INFERENCE", model_id="M03-KSC-T2", arms={},
        posterior_comparisons={"B/R": "DISCREPANCY", "C/D": "NOT_RUN", "D/R": "NOT_RUN", "C/R": "NOT_RUN"},
        heuristic_dominance="NOT_RUN")
    for arm in ("A", "B"):
        tuning = directory / arm / "tuning/manifest.json"
        master.write_json(tuning, dict(passed=True, checks=dict(fresh_rhat=True)))
        chunks = []
        for i in range(4):
            archive = directory / arm / f"chunk-{i+1:02d}/manifest.json"
            archive.parent.mkdir(parents=True)
            sample = archive.parent / "samples.tfbin"
            sample.write_bytes(b"preserved test samples")
            master.write_json(archive, dict(raw_samples=dict(path=sample.name, sha256=master.file_hash(sample)),
                tuning_manifest_sha256=master.file_hash(tuning), health_passed=True,
                continuation_manifest=chunks[-1]["manifest"] if chunks else None))
            chunks.append(dict(manifest=str(archive), health_passed=True))
        report["arms"][arm] = dict(status="DISCREPANCY", tuning=dict(passed=True, manifest=str(tuning)),
            chunks=chunks, retained_draws=1000,
            looks=[dict(draws_per_chain=1000, status="DISCREPANCY", diagnostic_valid=True)])
    master.write_json(directory / "report.json", report)
    return directory


def test_history_preserves_closed_comparisons_without_resampling(tmp_path):
    completed_history(tmp_path)
    state = master.ksc_inference_history(tmp_path)
    assert state["artifact_valid"] and state["closed"] and state["retained_surrogate_hmc_tested"]
    assert state["comparisons"] == {"B/R": "DISCREPANCY", "C/D": "NOT_RUN", "D/R": "NOT_RUN", "C/R": "NOT_RUN"}
    assert state["authorizes_new_inference"] is False


@pytest.mark.parametrize("broken", ["sample", "source", "continuation", "comparison", "diagnostic", "partial_look"])
def test_invalid_history_never_becomes_inference_evidence(tmp_path, broken):
    directory = completed_history(tmp_path)
    if broken in ("sample", "source"):
        path = directory / ("B/chunk-01/samples.tfbin" if broken == "sample" else "executed-source/numerical.py")
        path.write_text("changed")
    elif broken == "continuation":
        path = directory / "B/chunk-02/manifest.json"
        payload = json.loads(path.read_text())
        payload["continuation_manifest"] = None
        master.write_json(path, payload)
    else:
        path = directory / "report.json"
        payload = json.loads(path.read_text())
        if broken == "comparison":
            payload["posterior_comparisons"]["C/D"] = "EQUIVALENT"
        elif broken == "diagnostic":
            payload["arms"]["B"]["looks"][-1]["diagnostic_valid"] = False
        else:
            payload["arms"]["B"]["looks"][-1]["status"] = "INCONCLUSIVE"
            payload["arms"]["B"]["status"] = "INCONCLUSIVE"
        master.write_json(path, payload)
    state = master.ksc_inference_history(tmp_path)
    assert state["status"] == "ARTIFACT_INVALID"
    assert not state["surrogate_hmc_tested"] and not state["authorizes_new_inference"]


def completed_finite_history(tmp_path):
    """Independent C/D can pass even when D/R and C/R show a discrepancy."""
    exact = completed_history(tmp_path)
    directory = exact.with_name(exact.name.replace("-inference-", "-finite-inference-"))
    reference_root = tmp_path / "finite-reference"
    endpoint_root, posterior_root, controls = (tmp_path / name for name in ("endpoint", "posterior", "controls"))
    for root in (endpoint_root, posterior_root):
        master.write_json(root / "report.json", dict(status="qualified test reference"))
    master.write_json(controls / "selection.json", dict(multiplier=.1))
    prior = json.loads((exact / "run-manifest.json").read_text())
    prior.update(controls_root=str(controls), posterior_root=str(posterior_root))
    master.write_json(exact / "run-manifest.json", prior)
    binding = dict(data_sha256=prior["data_sha256"], selection_sha256=master.file_hash(controls / "selection.json"),
        endpoint_identity="independently frozen endpoint", endpoint_report_sha256=master.file_hash(endpoint_root / "report.json"),
        posterior_report_sha256=master.file_hash(posterior_root / "report.json"))
    for root in (directory, reference_root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "data.json").write_bytes((exact / "data.json").read_bytes())
        source = root / "executed-source/numerical.py"
        source.parent.mkdir()
        source.write_text("preserved executed source\n")
        master.write_json(root / "run-manifest.json", dict(scope="M03-KSC-T2", exit_code=0,
            source_hashes={"numerical.py": master.file_hash(source)}, **binding))
        master.write_json(root / "accounting.json", dict(exit_code=0))
    tensor = reference_root / "grid.tensor"
    tensor.write_bytes(b"preserved reference grid")
    item = dict(path=str(tensor), sha256=master.file_hash(tensor))
    reference = dict(status="QUALIFIED_FINITE_POSTERIOR_REFERENCE", qualified=True, full_prior_integrated=True,
        model_id="M03-KSC-T2", endpoint_identity=binding["endpoint_identity"],
        log_density=item, grids=[dict(density=item)], D_vs_R=dict(status="DISCREPANCY"))
    master.write_json(reference_root / "report.json", reference)
    run = json.loads((directory / "run-manifest.json").read_text())
    run.update(finite_reference_root=str(reference_root), endpoint_root=str(endpoint_root), posterior_root=str(posterior_root),
               finite_reference_report_sha256=master.file_hash(reference_root / "report.json"))
    master.write_json(directory / "run-manifest.json", run)
    tuning = directory / "C/tuning/manifest.json"
    master.write_json(tuning, dict(passed=True, checks=dict(fresh_rhat=True)))
    chunks = []
    for i in range(4):
        archive = directory / "C" / f"chunk-{i+1:02d}/manifest.json"
        archive.parent.mkdir(parents=True)
        sample = archive.parent / "samples.tfbin"
        sample.write_bytes(b"independent C samples")
        master.write_json(archive, dict(raw_samples=dict(path=sample.name, sha256=master.file_hash(sample)),
            tuning_manifest_sha256=master.file_hash(tuning), health_passed=True,
            continuation_manifest=chunks[-1]["manifest"] if chunks else None))
        chunks.append(dict(manifest=str(archive), health_passed=True))
    comparisons = {"C/D": dict(status="EQUIVALENT", diagnostic_valid=True),
                   "C/R": dict(status="DISCREPANCY", diagnostic_valid=True)}
    row = dict(status="EQUIVALENT", tuning=dict(passed=True, manifest=str(tuning)), chunks=chunks,
               retained_draws=1000, looks=[dict(draws_per_chain=1000, comparisons=comparisons)])
    master.write_json(directory / "report.json", dict(status="COMPLETED_BOUNDED_FINITE_INFERENCE",
        model_id="M03-KSC-T2", endpoint_identity=binding["endpoint_identity"], arms=dict(C=row),
        D_vs_R=reference["D_vs_R"], posterior_comparisons={"B/R": "NOT_RUN", "C/D": "EQUIVALENT",
                                                          "C/R": "DISCREPANCY", "D/R": "DISCREPANCY"}))
    return directory


def test_history_merges_independent_finite_target_without_vetoing_on_target_bias(tmp_path):
    completed_finite_history(tmp_path)
    state = master.ksc_inference_history(tmp_path)
    assert state["artifact_valid"] and state["closed"]
    assert state["comparisons"] == {"B/R": "DISCREPANCY", "C/D": "EQUIVALENT",
                                     "D/R": "DISCREPANCY", "C/R": "DISCREPANCY"}
    assert set(state["arms"]) == {"A", "B", "C"}
    assert not state["authorizes_new_inference"] and not state["all_models_validated"]


@pytest.mark.parametrize("corruption", ["reference_grid", "source", "samples", "uncertainty", "repeated_look", "binding"])
def test_finite_history_rejects_invalid_evidence_preserving_closed_b(tmp_path, corruption):
    directory = completed_finite_history(tmp_path)
    path = directory / "report.json"
    report = json.loads(path.read_text())
    if corruption == "reference_grid":
        (tmp_path / "finite-reference/grid.tensor").write_bytes(b"corrupt")
    elif corruption == "source":
        (directory / "executed-source/numerical.py").write_text("changed source")
    elif corruption == "samples":
        (directory / "C/chunk-01/samples.tfbin").write_bytes(b"corrupt")
    elif corruption == "uncertainty":
        report["arms"]["C"]["looks"][0]["comparisons"]["C/D"]["diagnostic_valid"] = False
    elif corruption == "repeated_look":
        report["arms"]["C"]["looks"].append(dict(draws_per_chain=2000,
            comparisons=report["arms"]["C"]["looks"][0]["comparisons"]))
    else:
        manifest = directory / "run-manifest.json"
        run = json.loads(manifest.read_text())
        run["endpoint_identity"] = "different finite target"
        master.write_json(manifest, run)
    master.write_json(path, report)
    state = master.ksc_inference_history(tmp_path)
    assert state["status"] == "ARTIFACT_INVALID" and not state["artifact_valid"]
    assert state["comparisons"]["B/R"] == "DISCREPANCY"
    assert state["comparisons"]["C/D"] == "NOT_RUN"
    assert not state["authorizes_new_inference"]


def qualified_reference(tmp_path):
    master.write_json(tmp_path / "data.json", dict(observations=[1., 2.]))
    data_hash = master.file_hash(tmp_path / "data.json")
    grids = []
    for i in range(4):
        grid = {}
        for name in ("gamma", "beta", "density", "predictive"):
            path = tmp_path / f"{i}-{name}.tfbin"
            path.write_bytes(b"preserved independent grid")
            grid[name] = dict(path=str(path), sha256=master.file_hash(path))
        grids.append(grid)
    report = dict(status="QUALIFIED_MODEL_POSTERIOR_REFERENCE", qualified=True,
        model_id="M03-KSC-T2", full_prior_integrated=True, data_sha256=data_hash,
        parameter_intervals=[32, 64, 128, 256], grids=grids, allowance=[1e-5]*13,
        precision=[True]*13, contraction=[True]*13, margins=[.1]*5+[.02]*6+[.03]*2,
        expectations=[[0.]*13]*4, means=[.5,.5], scales=[.2,.2], cuts=[[.2,.5,.8]]*2,
        function_names=[f"function-{i}" for i in range(13)])
    master.write_json(tmp_path / "report.json", report)
    master.write_json(tmp_path / "run-manifest.json", dict(data_sha256=data_hash, source_hashes={}))
    return report


def test_reference_guard_accepts_complete_precision_record(tmp_path):
    qualified_reference(tmp_path)
    assert ksc.check_reference(tmp_path)["qualified"]


@pytest.mark.parametrize("key,value", [("precision", []), ("contraction", [True]),
    ("allowance", [.02]*13), ("expectations", [[0.]*13]), ("scales", [0., .2]),
    ("cuts", [[.1,.5,.8]]*2), ("margins", [1.]*13)])
def test_reference_guard_rejects_incomplete_or_unqualified_functionals(tmp_path, key, value):
    report = qualified_reference(tmp_path)
    report[key] = value
    master.write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="functional precision"):
        ksc.check_reference(tmp_path)
