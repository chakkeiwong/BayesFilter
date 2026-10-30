"""Additional allocation: durable costs, deadlines and planned scientific retry."""
import json
from pathlib import Path
import pytest
from bayesfilter.inference import surrogate_hmc_recovery_program as master


def test_deadline_and_both_compute_reserves_bound_tuning(monkeypatch):
    monkeypatch.setattr(master, "seconds_to_deadline", lambda p: 60000.)
    available = dict(cpu_worker_seconds=100000., gpu_seconds=110000.)
    device, wall, cpu = master.phase_budget("m13-public-tuning", available,
        allocation="48h", attempt=1)
    assert device == "gpu" and wall == 23980. and cpu == 63980.
    device, wall, cpu = master.phase_budget("m13-public-tuning", available,
        allocation="48h", attempt=2,
        tuning_charged=dict(cpu_worker_seconds=100000., gpu_seconds=105000.))
    assert wall == 10200. and cpu == 15200.
    monkeypatch.setattr(master, "seconds_to_deadline", lambda p: 36050.)
    with pytest.raises(ValueError, match="deadline"):
        master.phase_budget("m13-public-tuning", available, allocation="48h")


def test_healthy_inconclusive_attempt_retries_but_health_veto_does_not(monkeypatch):
    monkeypatch.setattr(master, "seconds_to_deadline", lambda p: 170000.)
    available = dict(cpu_worker_seconds=150000., gpu_seconds=150000.)
    for cause in ("WARMUP_MAXIMUM_INCONCLUSIVE", "VERIFICATION_MAXIMUM_INCONCLUSIVE", "BUDGET_EXHAUSTED"):
        result = dict(qualified=False, tuning=dict(sequential_status=cause), status=cause)
        decision = master.phase_decision("m13-public-tuning", "COMPLETED", result,
            available, allocation="48h", attempts=1)
        assert decision["automatic"] and not decision["terminal"]
        assert master.phase_decision("m13-public-tuning", "COMPLETED", result,
            available, allocation="48h", attempts=2)["terminal"]
    for cause in ("HEALTH_VETO", "missing"):
        decision = master.phase_decision("m13-public-tuning", "COMPLETED",
            dict(qualified=False, tuning=dict(sequential_status=cause), status=cause),
            available, allocation="48h", attempts=1)
        assert decision["terminal"] and not decision["automatic"]


def make_receipt(root, name, cpu, gpu, *, started=False):
    worker = root / name
    worker.mkdir(parents=True)
    (worker / "run-manifest.json").write_text("{}")
    (worker / "guard-receipt.json").write_text(json.dumps(dict(
        accounting_complete=True, cpu_worker_seconds=cpu, gpu_seconds=gpu)))
    if started:
        (worker / "sampling-state.json").write_text(json.dumps(dict(
            status="SAMPLING_STARTED", inference_launched=True)))
    return worker


def test_durable_receipts_carry_costs_and_attempts_not_other_allocations(tmp_path, monkeypatch):
    monkeypatch.setattr(master, "ARTIFACTS", tmp_path)
    development = tmp_path / "surrogate-hmc-m13-sequential-48h-development-01"
    make_receipt(development, "probe", 11., 7.)
    prior = tmp_path / "surrogate-hmc-m13-sequential-48h-01"
    make_receipt(prior, "m13-public-tuning", 101., 97., started=True)
    (prior / "run-manifest.json").write_text(json.dumps(dict(allocation_id=master.CONTINUATION_ID)))
    # A stale summary must not erase the guard receipt.
    (prior / "accounting.json").write_text(json.dumps(dict(charged_this_run=dict(cpu_worker_seconds=0., gpu_seconds=0.))))
    old = tmp_path / "surrogate-hmc-m13-sequential-old"
    make_receipt(old, "m13-public-tuning", 50000., 50000., started=True)
    (old / "run-manifest.json").write_text(json.dumps(dict(allocation_id=master.ALLOCATION_ID)))
    charged, receipts = master.continuation_charges(tmp_path / "new")
    assert charged == dict(cpu_worker_seconds=112., gpu_seconds=104.)
    history = receipts[-1]
    assert history["scientific_attempts_started"] == 1
    assert history["tuning_charged"] == dict(cpu_worker_seconds=101., gpu_seconds=97.)
    make_receipt(prior, "m13-public-tuning-02", 3., 2., started=True)
    with pytest.raises(ValueError, match="both scientific attempts"):
        master.continuation_charges(tmp_path / "new")


def test_unresolved_guard_blocks_new_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(master, "ARTIFACTS", tmp_path)
    worker = tmp_path / "surrogate-hmc-m13-sequential-48h-development-01/probe"
    worker.mkdir(parents=True)
    (worker / "run-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="unresolved"):
        master.continuation_charges(tmp_path / "new")


def test_preflight_worker_uses_valid_attempt_argument(tmp_path):
    command = master.phase_command("m13-preflight", tmp_path, allocation="48h", attempt=0)
    assert command[command.index("--attempt")+1] == "1"


@pytest.mark.parametrize("prior_attempts", [0, 1])
def test_master_retries_then_hands_qualified_second_attempt_to_retention(tmp_path, monkeypatch, prior_attempts):
    monkeypatch.setattr(master, "seconds_to_deadline", lambda p: 172000.)
    monkeypatch.setattr(master, "prior_recovery_charges",
        lambda *a, **k: (dict(cpu_worker_seconds=4., gpu_seconds=3.),
            [dict(kind="allocation history", scientific_attempts_started=prior_attempts)]))
    monkeypatch.setattr(master, "prerequisite_check", lambda allocation="legacy": {})
    monkeypatch.setattr(master, "snapshot", lambda output: {})
    monkeypatch.setattr(master, "check_snapshot", lambda hashes: None)
    monkeypatch.setattr(master, "gpu_qualification_passed", lambda result: True)
    launched = []

    def run(command, directory, ledger, **kwargs):
        from bayesfilter.inference.surrogate_hmc_program import reserve_job
        reserve_job(ledger, kwargs["device"], kwargs["timeout"],
            kwargs["cpu_allowance"], max_timeout_seconds=kwargs["max_timeout_seconds"])
        directory.mkdir()
        launched.append(directory.name)
        result = dict(status="PASS")
        if directory.name == "m13-preflight":
            assert command[command.index("--attempt")+1] == str(prior_attempts+1)
        path = directory / "report.json"
        if directory.name == "gpu-smoke":
            path = directory / "qualification/report.json"
        if directory.name == "m13-public-tuning":
            result = dict(status="INCONCLUSIVE", qualified=False,
                tuning=dict(sequential_status="WARMUP_MAXIMUM_INCONCLUSIVE"))
        if directory.name == "m13-public-tuning-02":
            assert command[command.index("--attempt")+1] == "2"
            result = dict(status="QUALIFIED", qualified=True, source_attempt=2)
        if directory.name == "m13-retained":
            parent = master.read(directory.parent / "report.json")
            assert parent["stages"]["m13-public-tuning"]["result"]["source_attempt"] == 2
            assert len(parent["attempts"]) == 2-prior_attempts
            result = dict(status="POSTERIOR_PASS", posterior_qualified=True)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result))
        job = dict(status="COMPLETED", cpu_worker_seconds=1., gpu_seconds=1.)
        ledger["jobs"].append(job)
        for key in ledger["charged"]:
            ledger["charged"][key] += 1.
        return job

    monkeypatch.setattr(master, "run_job", run)
    output = tmp_path / "campaign"
    assert master.execute(output, research=True, allocation="48h",
        limits=dict(cpu_worker_seconds=172800., gpu_seconds=172800.)) == 0
    assert launched == (["tests", "gpu-smoke", "m13-preflight"] +
                        ([] if prior_attempts else ["m13-public-tuning"]) +
                        ["m13-public-tuning-02", "m13-retained"])
    assert master.read(output / "accounting.json")["cumulative_recovery_charged"] == dict(
        cpu_worker_seconds=10.-prior_attempts, gpu_seconds=9.-prior_attempts)
    next_plan = master.read(output / "next-phase-plan.json")
    assert next_plan["part_of_existing_master"]
    assert "1000/2000/4000/6000/8000/10000" in next_plan["retained_policy"]


def test_continuation_requires_qualified_extended_calibration(tmp_path, monkeypatch):
    import json
    def saved(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    reference = saved(tmp_path/"reference.json", dict(data_seed=1704, reference_sha256="fixture"))
    baseline = saved(tmp_path/"baseline.json", dict(health_passed=True, reference_sha256="fixture"))
    coverage = saved(tmp_path/"coverage"/"coverage.json",
        dict(qualified=True, looks=[1000, 2000], reference_sha256="fixture"))
    monkeypatch.setattr(master, "REFERENCE", reference)
    monkeypatch.setattr(master, "BASELINE", baseline)
    monkeypatch.setattr(master, "EXTENDED_COVERAGE", coverage)
    monkeypatch.setattr(master, "LEGACY", tmp_path/"legacy")
    monkeypatch.setattr(master, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="coverage"):
        master.prerequisite_check("48h")
    saved(coverage, dict(qualified=True, looks=list(master.EXTENDED_LOOKS),
        reference_sha256="fixture", method="m13_ten_functions_guarded_lugsail_six_looks_v2"))
    saved(coverage.with_name("guard-receipt.json"), dict(status="COMPLETED", accounting_complete=True))
    source = tmp_path/"estimator.py"
    source.write_text("original")
    saved(tmp_path/"coverage-extended-plan.json", dict(result=str(coverage),
        sources={"estimator.py": master.file_hash(source)}))
    assert master.prerequisite_check("48h")["retained_looks"] == list(master.EXTENDED_LOOKS)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        master.prerequisite_check("48h")


@pytest.mark.parametrize("phase", ["tests", "gpu-smoke", "m13-preflight", "m13-public-tuning", "m13-retained"])
def test_each_worker_reserves_with_real_guard_under_48h_allocation(monkeypatch, phase):
    from bayesfilter.inference.surrogate_hmc_program import reserve_job
    monkeypatch.setattr(master, "seconds_to_deadline", lambda _: 172800.)
    limits = dict(cpu_worker_seconds=172800., gpu_seconds=172800.)
    device, wall, cpu = master.phase_budget(phase, limits, allocation="48h", attempt=1)
    ceiling = min(24 * 3600., max(master.allocation_profile("48h")["limits"].values()))
    reserve_job(dict(limits=limits, charged=dict(cpu_worker_seconds=0., gpu_seconds=0.)),
        device, wall, cpu, max_timeout_seconds=ceiling)
    assert wall <= ceiling <= 86400.
