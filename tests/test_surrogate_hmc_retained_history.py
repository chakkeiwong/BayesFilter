"""A restart must not create unplanned posterior looks or new tuning authority."""
import json
from pathlib import Path

import pytest

from bayesfilter.inference import surrogate_hmc_program as program


def prior_report(root, stage, *, attempt=4, model="LG1", draws=2000,
                 status="INCONCLUSIVE", with_looks=True, suffix="01"):
    arms = ("C",) if stage == "P2-C-sample" else ("A", "B")
    comparisons = ({"C/D": status, "C/R": status, "D/R": "DISCREPANCY", "B/R": "NOT_RUN"}
                   if "C" in arms else {"B/R": status, "C/D": "NOT_RUN"})
    report = dict(inference_attempt=attempt, provenance={"scope": {"model": model, "attempt": attempt}},
                  comparisons=comparisons, arms={})
    for arm in arms:
        look = ({"C/D": {"status": status}, "C/R": {"status": status}}
                if arm == "C" else {"status": status})
        report["arms"][arm] = dict(status=status, retained_draws=draws,
                                   looks=[look] if with_looks else [])
    path = root / f"docs/plans/artifacts/surrogate-hmc-all-models-test-{suffix}/{stage}/report.json"
    program.write_json(path, report)
    return path


def test_discovery_distinguishes_scope_partial_look_and_repairable_failure(tmp_path):
    prior_report(tmp_path, "P2-sample", attempt=3, suffix="old-attempt")
    prior_report(tmp_path, "P2-sample", model="LG2", suffix="other-model")
    prior_report(tmp_path, "P2-sample", status="FAILED", with_looks=False, draws=0, suffix="failed-worker")
    assert program.retained_attempt_history(tmp_path, 4) == {}
    path = prior_report(tmp_path, "P2-sample", draws=1000)
    entry = program.retained_attempt_history(tmp_path, 4)["P2-sample-job"]
    assert not entry["closed"] and not entry["authorizes_new_inference"]
    assert entry["report_sha256"] == program.file_hash(path)
    assert program.retained_attempt_history(tmp_path, 5) == {}


def test_health_veto_closes_attempt_even_before_first_look(tmp_path):
    prior_report(tmp_path, "P2-sample", status="HEALTH_VETO", with_looks=False, draws=0)
    assert program.retained_attempt_history(tmp_path, 4)["P2-sample-job"]["closed"]


def test_C_early_closure_requires_both_comparisons(tmp_path):
    path = prior_report(tmp_path, "P2-C-sample", status="EQUIVALENT", draws=1000)
    assert program.retained_attempt_history(tmp_path, 4)["P2-C-sample-job"]["closed"]
    report = json.loads(path.read_text())
    report["arms"]["C"]["looks"][-1]["C/R"]["status"] = "INCONCLUSIVE"
    program.write_json(path, report)
    assert not program.retained_attempt_history(tmp_path, 4)["P2-C-sample-job"]["closed"]


def test_conflicting_reports_fail_closed_instead_of_selecting_favorable_result(tmp_path):
    prior_report(tmp_path, "P2-sample")
    prior_report(tmp_path, "P2-sample", suffix="duplicate")
    with pytest.raises(ValueError, match="multiple retained reports"):
        program.retained_attempt_history(tmp_path, 4)


def test_historical_result_cannot_authorize_new_dependent_job(tmp_path, monkeypatch):
    prior_report(tmp_path, "P2-sample", draws=1000)
    history = program.retained_attempt_history(tmp_path, 4)
    jobs = [dict(name=name, depends_on=deps, command=[name], device="cpu", timeout=1)
            for name, deps in (("P2-sample-job", []), ("fresh-dependent-job", ["P2-sample-job"]),
                               ("independent-job", []))]
    launched = []
    monkeypatch.setattr(program, "run_job", lambda command, *a, **k:
                        (launched.append(command[0]) or {"status": "COMPLETED"}))
    jobs = program.preserve_retained_attempts(jobs, history, all_stages=False)
    outcomes = program.execute_job_graph(jobs, tmp_path / "out", {})
    assert launched == ["independent-job"]
    assert outcomes["P2-sample-job"]["status"] == "HISTORICAL_RESULT_REPORTED"
    assert not outcomes["P2-sample-job"]["new_sampling"]
    assert outcomes["fresh-dependent-job"]["status"] == "SKIPPED_PREREQUISITE"


def test_all_reports_closed_results_and_only_launches_other_model(tmp_path, monkeypatch, capsys):
    prior_report(tmp_path, "P2-sample")
    prior_report(tmp_path, "P2-C-sample")
    history = program.retained_attempt_history(tmp_path, 4)
    monkeypatch.setattr(program, "retained_attempt_history", lambda root, attempt: history)
    monkeypatch.setattr(program, "gaussian_attempt_history", lambda root: {})
    monkeypatch.setattr(program, "gaussian_qualification_forecast", lambda *args: {"control_forecast": True})
    monkeypatch.setattr(program, "source_inventory", lambda: {})
    monkeypatch.setattr(program, "retained_forecast", lambda *a: pytest.fail("closed attempt needs no new forecast"))
    launched = []

    def fake_run(command, directory, ledger, **kwargs):
        launched.append(directory.name)
        return dict(status="COMPLETED", exit_code=0)

    monkeypatch.setattr(program, "run_job", fake_run)
    output = tmp_path / "out"
    assert program.main(["--stage", "all", "--output", str(output)]) == 2
    capsys.readouterr()
    assert launched == ["P3-LG2-reference-job", "P3-LG2-qualify-job", "P3-LG2-controls-job",
                        "P3-SV20-reference-job", "P3-SV20-posterior-job", "P3-SV20-qualify-job", "P3-SV20-controls-job", "P3-C01-qualify-job", "P3-C01-inference-job"]
    decision = json.loads((output / "decision.json").read_text())
    assert decision["comparisons"] == {"B/R": "INCONCLUSIVE", "C/D": "INCONCLUSIVE",
                                        "C/R": "INCONCLUSIVE", "D/R": "DISCREPANCY"}
    assert decision["execution_status"] == "COMPLETED"
    assert not decision["all_models_validated"]
    manifest = json.loads((output / "run-manifest.json").read_text())
    assert manifest["inference_ready"]  # New C01 branch, never the closed LG1 attempt.
    assert manifest["inference_forecast"] == {"control_forecast": True}
    assert manifest["prior_retained_results"] == history


def test_changed_historical_report_is_not_reused(tmp_path, monkeypatch):
    path = prior_report(tmp_path, "P2-sample")
    history = program.retained_attempt_history(tmp_path, 4)
    path.write_text(path.read_text() + "\n")
    jobs = program.preserve_retained_attempts(
        [dict(name="P2-sample-job", depends_on=[])], history, all_stages=False)
    monkeypatch.setattr(program, "run_job", lambda *a, **k: pytest.fail("must not sample"))
    outcomes = program.execute_job_graph(jobs, tmp_path / "out", {})
    assert outcomes["P2-sample-job"]["status"] == "BLOCKED_PREFLIGHT"


def test_P0_reports_closed_LG1_and_C01_without_launching(tmp_path, monkeypatch, capsys):
    prior_report(tmp_path, "P2-sample")
    c_path = prior_report(tmp_path, "P2-C-sample")
    finite_path = tmp_path / "finite-reference/report.json"
    program.write_json(finite_path, {"D_vs_R": {"status": "DISCREPANCY"}})
    c_report = json.loads(c_path.read_text())
    c_report["provenance"].update(finite_reference_path=str(finite_path),
                                finite_reference_sha256=program.file_hash(finite_path))
    program.write_json(c_path, c_report)
    gaussian_path = tmp_path / "docs/plans/artifacts/surrogate-hmc-all-models-test-C01/P3-C01-inference/report.json"
    program.write_json(gaussian_path, dict(
        provenance={"scope": "C01-anisotropic-gaussian-d3-attempt1"}, closed=True,
        arms={"exact": {"status": "INCONCLUSIVE"}, "zero": {"status": "TUNING_REJECTED"}}))
    history = program.retained_attempt_history(tmp_path, 4)
    gaussian_history = program.gaussian_attempt_history(tmp_path)
    monkeypatch.setattr(program, "retained_attempt_history", lambda *a: history)
    monkeypatch.setattr(program, "gaussian_attempt_history", lambda *a: gaussian_history)
    monkeypatch.setattr(program, "source_inventory", lambda: {})
    monkeypatch.setattr(program, "run_job", lambda *a, **k: pytest.fail("P0 must not run inference"))
    output = tmp_path / "out"
    assert program.main(["--stage", "P0", "--scope", "all", "--output", str(output)]) == 0
    capsys.readouterr()
    decision = json.loads((output / "decision.json").read_text())
    assert decision["comparisons"] == {"B/R": "INCONCLUSIVE", "C/D": "INCONCLUSIVE",
                                       "C/R": "INCONCLUSIVE", "D/R": "DISCREPANCY"}
    assert decision["finite_posterior_reference"]["D_vs_R"]["status"] == "DISCREPANCY"
    assert decision["scope_results"]["C01"]["comparisons"]["zero/analytic"] == "TUNING_REJECTED"
    assert decision["job_outcomes"] == {}
    assert not decision["all_models_validated"]


def test_report_only_reader_rejects_changed_history(tmp_path):
    path = prior_report(tmp_path, "P2-sample")
    history = program.retained_attempt_history(tmp_path, 4)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="recorded retained report changed"):
        program.read_retained_report(tmp_path / "none.json", "P2-sample-job", history, {}, report_only=True)


def test_P0_rejects_changed_linked_finite_reference(tmp_path, monkeypatch, capsys):
    c_path = prior_report(tmp_path, "P2-C-sample")
    finite_path = tmp_path / "finite-reference/report.json"
    program.write_json(finite_path, {"D_vs_R": {"status": "DISCREPANCY"}})
    report = json.loads(c_path.read_text())
    report["provenance"].update(finite_reference_path=str(finite_path),
                              finite_reference_sha256=program.file_hash(finite_path))
    program.write_json(c_path, report)
    finite_path.write_text(finite_path.read_text() + "\n")
    history = program.retained_attempt_history(tmp_path, 4)
    monkeypatch.setattr(program, "retained_attempt_history", lambda *a: history)
    monkeypatch.setattr(program, "source_inventory", lambda: {})
    output = tmp_path / "out"
    assert program.main(["--stage", "P0", "--scope", "LG1", "--output", str(output)]) == 1
    capsys.readouterr()
    decision = json.loads((output / "decision.json").read_text())
    assert decision["execution_status"] == "FAILED"
    assert "finite posterior reference changed" in decision["error"]
    assert decision["comparisons"]["D/R"] == "NOT_RUN"
    assert decision["comparisons"]["C/D"] == "NOT_RUN"
