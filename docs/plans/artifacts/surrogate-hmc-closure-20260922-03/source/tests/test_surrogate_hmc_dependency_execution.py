"""Failed candidates must not veto scientifically independent comparisons."""
import json
from pathlib import Path

import pytest

from bayesfilter.inference import surrogate_hmc_program as program


@pytest.fixture(autouse=True)
def isolate_campaign_history(monkeypatch, tmp_path):
    # These tests exercise new attempts; real local results have separate tests.
    monkeypatch.setattr(program, "retained_attempt_history", lambda root, attempt: {})
    monkeypatch.setattr(program, "gaussian_attempt_history", lambda root: {})
    monkeypatch.setattr(program, "gaussian_qualification_forecast", lambda *args: {"control_forecast": True})
    # The SV20 readers are imported locally by the master. Exercise their real
    # empty-history behavior instead of inheriting this checkout's later pilots.
    from bayesfilter.inference import surrogate_hmc_sv_program as sv_program
    read_history = sv_program.sv20_inference_history
    read_endpoint = sv_program.sv20_endpoint_state
    monkeypatch.setattr(sv_program, "sv20_inference_history", lambda root: read_history(tmp_path))
    monkeypatch.setattr(sv_program, "sv20_endpoint_state", lambda root: read_endpoint(tmp_path))


def job(name, *dependencies):
    return dict(name=name, depends_on=list(dependencies), command=[name],
                device="cpu", timeout=1, cpu_allowance=1)


def test_failed_B_tuning_preserves_independent_C_execution(tmp_path, monkeypatch):
    launched = []

    def fake_run(command, directory, ledger, **kwargs):
        launched.append(command[0])
        return dict(status="FAILED" if command[0] == "B-tune" else "COMPLETED")

    monkeypatch.setattr(program, "run_job", fake_run)
    jobs = [job("reference"), job("controls", "reference"),
            job("B-tune", "controls"), job("B-sample", "B-tune"),
            job("endpoint", "controls"), job("D", "endpoint"),
            job("C-tune", "D"), job("C-sample", "C-tune")]
    outcomes = program.execute_job_graph(jobs, tmp_path, {})
    assert "B-sample" not in launched
    assert launched[-1] == "C-sample"
    assert outcomes["B-tune"]["status"] == "FAILED"
    assert outcomes["B-sample"]["status"] == "SKIPPED_PREREQUISITE"
    assert outcomes["C-sample"]["status"] == "COMPLETED"
    assert json.loads((tmp_path / "execution-results.json").read_text()) == outcomes


def test_shared_prerequisite_failure_blocks_both_descendants(tmp_path, monkeypatch):
    launched = []

    def fake_run(command, directory, ledger, **kwargs):
        launched.append(command[0])
        return dict(status="FAILED")

    monkeypatch.setattr(program, "run_job", fake_run)
    results = program.execute_job_graph(
        [job("controls"), job("B", "controls"), job("endpoint", "controls"),
         job("C", "endpoint")], tmp_path, {})
    assert launched == ["controls"]
    assert results["C"]["status"] == "SKIPPED_PREREQUISITE"


def test_local_forecast_failure_does_not_cancel_other_comparison(tmp_path, monkeypatch):
    launched = []

    def fake_run(command, directory, ledger, **kwargs):
        launched.append(command[0])
        return dict(status="COMPLETED")

    def before_job(specification):
        if specification["name"] == "B-sample":
            raise ValueError("retained forecast exceeds the bounded job budget")

    monkeypatch.setattr(program, "run_job", fake_run)
    results = program.execute_job_graph(
        [job("B-sample"), job("C-sample")], tmp_path, {}, before_job=before_job)
    assert launched == ["C-sample"]
    assert results["B-sample"]["status"] == "BLOCKED_PREFLIGHT"
    assert "forecast" in results["B-sample"]["error"]


def test_interrupt_never_launches_another_branch(tmp_path, monkeypatch):
    launched = []

    def fake_run(command, directory, ledger, **kwargs):
        launched.append(command[0])
        raise KeyboardInterrupt

    monkeypatch.setattr(program, "run_job", fake_run)
    with pytest.raises(KeyboardInterrupt):
        program.execute_job_graph([job("B"), job("C")], tmp_path, {})
    assert launched == ["B"]
    assert json.loads((tmp_path / "execution-results.json").read_text())["B"]["status"] == "INTERRUPTED"


@pytest.mark.parametrize("jobs", [
    [job("B", "missing")],
    [job("B", "C"), job("C")],
    [job("B"), job("B")],
])
def test_invalid_dependency_graph_rejected_before_launch(tmp_path, monkeypatch, jobs):
    monkeypatch.setattr(program, "run_job", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="dependency graph"):
        program.execute_job_graph(jobs, tmp_path, {})


def test_master_all_continues_C_after_B_failure(tmp_path, monkeypatch, capsys):
    launched = []

    def fake_run(command, directory, ledger, **kwargs):
        launched.append(directory.name)
        record = dict(status="FAILED" if directory.name == "P2-tune-job" else "COMPLETED")
        ledger["jobs"].append(record)
        return record

    monkeypatch.setattr(program, "run_job", fake_run)
    monkeypatch.setattr(program, "retained_forecast", lambda *args: dict(
        gpu_seconds=1, cpu_worker_seconds=1, eligible_arms=["C"]))
    monkeypatch.setattr(program, "source_inventory", lambda: {})
    output = tmp_path / "campaign"
    assert program.main(["--stage", "all", "--output", str(output)]) == 1
    capsys.readouterr()
    assert "P2-sample-job" not in launched
    assert "P2-C-sample-job" in launched
    assert "P3-SV20-reference-job" in launched
    decision = json.loads((output / "decision.json").read_text())
    assert decision["execution_status"] == "PARTIAL"
    assert decision["job_outcomes"]["P2-tune-job"]["status"] == "FAILED"
    assert not decision["all_models_validated"]


def test_standalone_forecast_failure_preserves_a_reviewable_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(program, "run_job", lambda *args, **kwargs: pytest.fail("must not launch"))
    monkeypatch.setattr(program, "source_inventory", lambda: {})
    monkeypatch.setattr(program, "retained_forecast", lambda *args: dict(
        gpu_seconds=4000, cpu_worker_seconds=4800, eligible_arms=["C"]))
    output = tmp_path / "blocked"
    args = ["--stage", "P2-C-sample", "--output", str(output),
            "--reference-root", str(tmp_path / "reference"),
            "--selection", str(tmp_path / "controls/selection.json"),
            "--endpoint-root", str(tmp_path / "endpoint"),
            "--finite-reference-root", str(tmp_path / "D"),
            "--tuning-root", str(tmp_path / "C-tuning")]
    assert program.main(args) == 1
    capsys.readouterr()
    decision = json.loads((output / "decision.json").read_text())
    result = decision["job_outcomes"]["P2-C-sample-job"]
    assert result["status"] == "BLOCKED_PREFLIGHT"
    assert "forecast" in result["error"]
    assert (output / "accounting.json").is_file()
    assert decision["comparisons"]["C/D"] == "NOT_RUN"


def test_LG2_all_executes_its_own_reference_and_consumer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(program, "ROOT", tmp_path)
    assert program.main(["--stage", "all", "--scope", "LG2", "--dry-run",
                         "--output", str(tmp_path / "unused")]) == 0
    launch = json.loads(capsys.readouterr().out)
    assert set(launch["scope_cards"]) == {"LG2"}
    assert launch["scope_card"]["parameters"] == ["a", "b"]
    assert launch["scope_card"]["inference_attempt"] == 1
    assert [j["name"] for j in launch["jobs"]] == ["P3-LG2-reference-job", "P3-LG2-qualify-job", "P3-LG2-controls-job"]
    assert launch["jobs"][1]["depends_on"] == ["P3-LG2-reference-job"]
    assert launch["jobs"][2]["depends_on"] == ["P3-LG2-qualify-job"]
    assert launch["jobs"][0]["device"] == "cpu"
    assert launch["jobs"][1]["device"] == "gpu"
    assert not launch["inference_ready"]


@pytest.mark.parametrize("stage,scope", [("P2-tune", "LG2"), ("P3-LG2-reference", "LG1"), ("P1", "LG2"),
                                       ("P3-SV20-reference", "LG2"), ("P2-tune", "SV20")])
def test_wrong_stage_scope_rejected(stage, scope, tmp_path):
    with pytest.raises(SystemExit) as exc:
        program.main(["--stage", stage, "--scope", scope, "--output", str(tmp_path)])
    assert exc.value.code == 2


@pytest.mark.parametrize("stage,status,posterior_qualified", [
    ("reference", "QUALIFIED_LIKELIHOOD_PROBES", False),
    ("posterior", "QUALIFIED_MODEL_POSTERIOR_REFERENCE", True)])
def test_sv20_reference_result_does_not_promote_hmc(tmp_path, monkeypatch, capsys, stage, status, posterior_qualified):
    launched = []

    def fake_run(command, directory, ledger, **kwargs):
        launched.append(command)
        target = Path(command[command.index("--output")+1])
        target.mkdir(parents=True)
        (target/"report.json").write_text(json.dumps(dict(
            status=status, qualified=True)))
        return dict(status="COMPLETED")

    monkeypatch.setattr(program, "run_job", fake_run)
    monkeypatch.setattr(program, "source_inventory", lambda: {})
    output = tmp_path/"sv20"
    args = ["--stage", "P3-SV20-"+stage, "--scope", "SV20", "--output", str(output)]
    if stage == "posterior":
        args += ["--reference-root", str(tmp_path/"prior-likelihood")]
    assert program.main(args) == 2
    capsys.readouterr()
    assert len(launched) == 1
    assert Path(launched[0][1]).name == f"prepare_surrogate_hmc_sv_{stage}.py"
    decision = json.loads((output/"decision.json").read_text())
    assert decision["reference_quality"] == status
    assert decision["execution_status"] == "COMPLETED"
    assert not decision["all_models_validated"]
    assert not decision["scope_results"]["SV20"]["surrogate_hmc_tested"]
    assert decision["scope_results"]["SV20"]["posterior_reference_qualified"] is posterior_qualified
    assert all(v == "NOT_RUN" for v in decision["comparisons"].values())
    register = json.loads((output/"model-register.json").read_text())
    actual_sv = next(row for row in register["families"] if row["id"] == "M02")
    assert actual_sv["inference_status"] == "NOT_RUN"
    assert actual_sv["variant_status"]["SV40"] == "SCOPE_UNRESOLVED"


def test_sv20_retry_consumes_original_allocation(tmp_path):
    path = tmp_path/"docs/plans/artifacts/surrogate-hmc-all-models-test/accounting.json"
    path.parent.mkdir(parents=True)
    job = dict(command=["python", "scripts/prepare_surrogate_hmc_sv_reference.py"],
               gpu_seconds=100., cpu_worker_seconds=200.)
    path.write_text(json.dumps(dict(jobs=[job])))
    assert program.sv20_reference_remaining_allocation(tmp_path) == (350., 500.)
    path.write_text(json.dumps(dict(jobs=[job, job])))
    assert program.sv20_reference_remaining_allocation(tmp_path) == (0., 0.)
    assert program.sv20_reference_remaining_allocation(tmp_path, "posterior") == (600., 900.)
    assert program.sv20_reference_remaining_allocation(tmp_path, "qualify") == (600., 450.)
    assert program.sv20_reference_remaining_allocation(tmp_path, "controls") == (600., 600.)
    job['command'][1] = 'scripts/qualify_surrogate_hmc_sv_consumer.py'
    path.write_text(json.dumps(dict(jobs=[job])))
    assert program.sv20_reference_remaining_allocation(tmp_path, "qualify") == (500., 250.)


def test_sv20_family_budget_counts_tests_mixed_runs_and_current_jobs(tmp_path):
    def worker(script, gpu, cpu):
        return dict(command=["python", script], gpu_seconds=gpu, cpu_worker_seconds=cpu)

    ledger = program.validated_budget(program.GPU_CAP, program.CPU_CAP)
    tests = tmp_path/"surrogate-hmc-all-models-sv20-tests-01/accounting.json"
    tests.parent.mkdir()
    tests.write_text(json.dumps(dict(charged_this_run=dict(gpu_seconds=0., cpu_worker_seconds=40.))))
    mixed = tmp_path/"surrogate-hmc-all-models-mixed/accounting.json"
    mixed.parent.mkdir()
    mixed.write_text(json.dumps(dict(charged_this_run=dict(gpu_seconds=900., cpu_worker_seconds=450.),
        jobs=[worker("prepare_surrogate_hmc_sv_reference.py", 100., 150.),
              worker("another_family.py", 800., 280.)])))
    program.charge_previous_runs(ledger, [tests, mixed])
    ledger["jobs"].append(worker("prepare_surrogate_hmc_sv_posterior.py", 300., 400.))
    remaining = program.sv20_family_remaining_allocation(ledger, parent_cpu_seconds=3.)
    # The September 22 inference plan increased the cumulative family ceiling;
    # these charges still include dedicated tests, mixed jobs and parent CPU.
    assert remaining == dict(gpu_seconds=14400. - 400., cpu_worker_seconds=13000. - 613.)


def test_sv20_family_budget_blocks_cross_stage_overrun(tmp_path, monkeypatch, capsys):
    real_charge = program.charge_previous_runs
    path = tmp_path/"surrogate-hmc-all-models-sv20-earlier/accounting.json"
    path.parent.mkdir()
    path.write_text(json.dumps(dict(charged_this_run=dict(gpu_seconds=50., cpu_worker_seconds=12600.))))
    monkeypatch.setattr(program, "charge_previous_runs", lambda ledger, paths: real_charge(ledger, [path]))
    launched = []
    monkeypatch.setattr(program, "run_job", lambda *args, **kwargs: launched.append(args))
    output = tmp_path/"blocked"
    assert program.main(["--stage", "P3-SV20-reference", "--scope", "SV20", "--output", str(output)]) == 1
    capsys.readouterr()
    assert not launched
    outcome = json.loads((output/"execution-results.json").read_text())["P3-SV20-reference-job"]
    assert outcome["status"] == "BLOCKED_PREFLIGHT"
    assert "SV20 family reservation: cpu_worker_seconds" in outcome["error"]


def test_sv20_failed_reference_blocks_only_its_posterior(tmp_path, monkeypatch, capsys):
    launched = []
    def fake_run(command, directory, ledger, **kwargs):
        launched.append(directory.name)
        return dict(status="FAILED")
    monkeypatch.setattr(program, "run_job", fake_run)
    monkeypatch.setattr(program, "source_inventory", lambda: {})
    output = tmp_path/"failed-sv20"
    assert program.main(["--stage", "all", "--scope", "SV20", "--output", str(output)]) == 1
    capsys.readouterr()
    assert launched == ["P3-SV20-reference-job"]
    outcomes = json.loads((output/"execution-results.json").read_text())
    assert outcomes["P3-SV20-posterior-job"]["status"] == "SKIPPED_PREREQUISITE"
    assert outcomes["P3-SV20-qualify-job"]["status"] == "SKIPPED_PREREQUISITE"
    assert outcomes["P3-SV20-controls-job"]["status"] == "SKIPPED_PREREQUISITE"


def test_sv20_consumer_mechanics_never_promotes_retained_inference(tmp_path, monkeypatch, capsys):
    launched = []

    def fake_run(command, directory, ledger, **kwargs):
        launched.append(command)
        target = Path(command[command.index('--output')+1])
        target.mkdir(parents=True)
        (target/'report.json').write_text(json.dumps(dict(
            status='QUALIFIED_CONSUMER_MECHANICS', qualified=True, original_force_healthy=False)))
        return dict(status='COMPLETED')

    monkeypatch.setattr(program, 'run_job', fake_run)
    monkeypatch.setattr(program, 'source_inventory', lambda: {})
    posterior = tmp_path/'posterior'
    posterior.mkdir()
    (posterior/'report.json').write_text(json.dumps(dict(status='QUALIFIED_MODEL_POSTERIOR_REFERENCE', qualified=True)))
    output = tmp_path/'consumer'
    assert program.main(['--stage', 'P3-SV20-qualify', '--scope', 'SV20', '--output', str(output),
                         '--reference-root', str(tmp_path/'reference'), '--posterior-root', str(posterior)]) == 2
    capsys.readouterr()
    assert Path(launched[0][1]).name == 'qualify_surrogate_hmc_sv_consumer.py'
    decision = json.loads((output/'decision.json').read_text())
    scope = decision['scope_results']['SV20']
    assert scope['posterior_reference_qualified'] and scope['consumer_mechanics_qualified']
    assert not scope['surrogate_hmc_tested'] and not scope['consumer']['original_force_healthy']
    assert set(scope['comparisons'].values()) == {'NOT_RUN'}
    assert not decision['all_models_validated']


def test_sv20_control_cost_deferral_preserves_qualified_references(tmp_path, monkeypatch, capsys):
    def fake_run(command, directory, ledger, **kwargs):
        assert Path(command[1]).name == 'calibrate_surrogate_hmc_sv_controls.py'
        target = Path(command[command.index('--output')+1])
        target.mkdir(parents=True)
        (target/'report.json').write_text(json.dumps(dict(verdict='SELECTED_FOR_PROPOSAL_EXPERIMENT',
            forecast=dict(status='UNDER_BUDGETED_INITIAL_ALLOCATION', inference_authorized=False))))
        return dict(status='COMPLETED')

    monkeypatch.setattr(program, 'run_job', fake_run)
    monkeypatch.setattr(program, 'source_inventory', lambda: {})
    posterior = tmp_path/'posterior'
    posterior.mkdir()
    (posterior/'report.json').write_text(json.dumps(dict(status='QUALIFIED_MODEL_POSTERIOR_REFERENCE', qualified=True)))
    qualification = tmp_path/'qualification'
    qualification.mkdir()
    (qualification/'report.json').write_text(json.dumps(dict(status='QUALIFIED_CONSUMER_MECHANICS', qualified=True)))
    (qualification/'run-manifest.json').write_text(json.dumps(dict(posterior_reference=str(posterior))))
    output = tmp_path/'controls'
    assert program.main(['--stage', 'P3-SV20-controls', '--scope', 'SV20', '--output', str(output),
        '--reference-root', str(tmp_path/'reference'), '--qualification-root', str(qualification)]) == 2
    capsys.readouterr()
    decision = json.loads((output/'decision.json').read_text())
    scope = decision['scope_results']['SV20']
    assert scope['posterior_reference_qualified'] and scope['consumer_mechanics_qualified']
    assert not scope['surrogate_hmc_tested']
    assert decision['inference_blocker'] == 'UNDER_BUDGETED_INITIAL_ALLOCATION'
    assert set(scope['comparisons'].values()) == {'NOT_RUN'}
    assert not decision['all_models_validated']


def test_failed_LG1_does_not_cancel_LG2_or_promote_qualification(tmp_path, monkeypatch, capsys):
    launched = []

    def fake_run(command, directory, ledger, **kwargs):
        launched.append(directory.name)
        target = Path(command[command.index("--output") + 1])
        program.write_json(target / "report.json", {"qualified": True})
        return dict(status="FAILED" if directory.name == "P1-LG1" else "COMPLETED",
                    exit_code=1 if directory.name == "P1-LG1" else 0, job_name=directory.name)

    monkeypatch.setattr(program, "run_job", fake_run)
    monkeypatch.setattr(program, "source_inventory", lambda: {})
    output = tmp_path / "campaign"
    assert program.main(["--stage", "all", "--output", str(output)]) == 1
    capsys.readouterr()
    assert launched == ["P1-LG1", "P2-reference-job", "P3-LG2-reference-job", "P3-LG2-qualify-job", "P3-LG2-controls-job",
                        "P3-SV20-reference-job", "P3-SV20-posterior-job", "P3-SV20-qualify-job", "P3-SV20-controls-job", "P3-C01-qualify-job", "P3-C01-inference-job"]
    decision = json.loads((output / "decision.json").read_text())
    assert decision["probe"]["job_name"] == launched[-1]
    assert decision["scope_results"]["LG2"]["consumer_qualification"]["qualified"]
    assert set(decision["scope_results"]["LG2"]["comparisons"].values()) == {"NOT_RUN"}
    assert not decision["all_models_validated"] and not decision["default_readiness"]


def test_all_merges_C_outcomes_without_erasing_B_result(tmp_path, monkeypatch, capsys):
    def fake_run(command, directory, ledger, **kwargs):
        target = Path(command[command.index("--output") + 1])
        if directory.name == "P2-sample-job":
            program.write_json(target / "report.json", {"comparisons": {"B/R": "INCONCLUSIVE", "C/D": "NOT_RUN"}})
        if directory.name == "P2-C-sample-job":
            program.write_json(target / "report.json", {"comparisons": {"B/R": "NOT_RUN", "C/D": "EQUIVALENT", "C/R": "INCONCLUSIVE"}})
        return dict(status="COMPLETED", exit_code=0, job_name=directory.name)

    monkeypatch.setattr(program, "run_job", fake_run)
    monkeypatch.setattr(program, "source_inventory", lambda: {})
    monkeypatch.setattr(program, "retained_forecast", lambda *args: dict(
        gpu_seconds=1, cpu_worker_seconds=1, eligible_arms=["B", "C"]))
    output = tmp_path / "campaign"
    assert program.main(["--stage", "all", "--scope", "LG1", "--output", str(output)]) == 2
    capsys.readouterr()
    decision = json.loads((output / "decision.json").read_text())
    assert decision["comparisons"]["B/R"] == "INCONCLUSIVE"
    assert decision["comparisons"]["C/D"] == "EQUIVALENT"
    assert decision["comparisons"]["C/R"] == "INCONCLUSIVE"
    assert decision["probe"]["job_name"] == "P2-C-sample-job"
