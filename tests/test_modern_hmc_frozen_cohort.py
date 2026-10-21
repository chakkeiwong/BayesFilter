"""Orchestration/accounting regressions; no GPU or sampler-quality evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import continue_modern_hmc_frozen_cohort as cohort
from scripts.settle_modern_hmc_frozen_run import settle_budget
from tests.test_modern_hmc_comparison_prices import design


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def designs(tmp_path):
    paths = []
    for method, roots in zip(cohort.METHODS, (8, 8, 6, 2)):
        value = design()
        value.update(method=method, target="NL-SIR", seed=72410)
        value["search"].update(epsilon_by_l=[[i+1, [.5]] for i in range(roots)],
                               max_candidates=2*roots)
        value["execution"].update(use_xla=True)
        value["execution"]["acceptance_policy"]["max_candidates"] = 2*roots
        paths.append(save(tmp_path / f"{method}.json", value))
    return paths


def price_report(designs):
    rows = cohort.inventory(designs)
    for row, envelope in zip(rows, (1600., 1400., 1200., 1000.)):
        row.update(root_count=len(row.pop("roots")), scenarios=[dict(
            load_multiplier=load, maximum_count_scenario_seconds=envelope*load/4,
            minimum_path_seconds=envelope*load/4) for load in (1., 2., 4.)])
    return dict(status="audited_cost_diagnostic", actual_price_worker_seconds=40., rows=rows)


def preparation(tmp_path):
    receipt = save(tmp_path / "old-preparation.json", dict(status="passed"))
    return dict(unallocated_after_preparation_and_pricing_hold_seconds=9500.,
        pricing_reserved_seconds=500., unallocated_before_preparation_seconds=10010.,
        preparation_actual_seconds=10., preparation_charged_once=True,
        existing_held_reserve_seconds=3600., prior_balance_carried_separately_seconds=17.,
        preparation_receipts=[dict(path=str(receipt), sha256=cohort.digest(receipt), seconds=10.)])


def test_all_four_methods_and_two_root_mams_reserve_the_whole_denominator(designs):
    allocation = cohort.allocate_cohort(price_report(designs), designs,
        after_preparation=1000., engineering_hold=20.)
    assert allocation["available_numerical_seconds"] == 940.
    assert [r["root_count"] for r in allocation["rows"]] == [8, 8, 6, 2]
    assert [r["reserved_member_count"] for r in allocation["rows"]] == [16, 16, 12, 4]
    assert sum(r["initial_cap_seconds"] for r in allocation["rows"]) == pytest.approx(940.)
    assert all(r["initial_cap_seconds"] > 0 for r in allocation["rows"])
    assert not any(r["minimum_path_fits_cap"] for r in allocation["rows"])
    assert allocation["method_ranking"] is None


@pytest.mark.parametrize("mutation", ["omission", "order", "seed", "repairs", "xla", "duplicate"])
def test_incomplete_or_changed_design_cohorts_rejected(designs, mutation):
    if mutation == "omission": designs.pop()
    elif mutation == "order": designs.reverse()
    elif mutation == "duplicate": designs[1] = designs[0]
    else:
        value = cohort.read(designs[0])
        if mutation == "seed": value["seed"] += 1
        elif mutation == "repairs": value["search"]["max_candidates"] -= 1
        else: value["execution"]["use_xla"] = False
        save(designs[0], value)
    with pytest.raises(ValueError):
        cohort.inventory(designs)


@pytest.mark.parametrize("mutation", ["partial", "omission", "order", "hash", "roots",
    "missing_load", "duplicate_load", "negative_cost", "no_balance", "nan_hold"])
def test_no_allocation_from_invalid_prices_or_funds(designs, mutation):
    report, balance, hold = price_report(designs), 1000., 20.
    if mutation == "partial": report["status"] = "running"
    elif mutation == "omission": report["rows"].pop()
    elif mutation == "order": report["rows"].reverse()
    elif mutation == "hash": report["rows"][0]["design_sha256"] = "changed"
    elif mutation == "roots": report["rows"][0]["root_count"] -= 1
    elif mutation == "missing_load": report["rows"][0]["scenarios"].pop()
    elif mutation == "duplicate_load": report["rows"][0]["scenarios"][0]["load_multiplier"] = 2.
    elif mutation == "negative_cost": report["actual_price_worker_seconds"] = -1.
    elif mutation == "no_balance": balance = 50.
    else: hold = float("nan")
    with pytest.raises(ValueError):
        cohort.allocate_cohort(report, designs, after_preparation=balance, engineering_hold=hold)


def test_releases_do_not_spend_future_shares_and_overruns_shrink_them(designs):
    allocation = cohort.allocate_cohort(price_report(designs), designs,
        after_preparation=1000., engineering_hold=20.)
    rows = allocation["rows"]
    first = cohort.next_cap(allocation, 0, 940.)
    balance = 940.-first/2
    second = cohort.next_cap(allocation, 1, balance)
    assert second == pytest.approx(rows[1]["initial_cap_seconds"]+first/2)
    assert balance-second >= sum(r["initial_cap_seconds"] for r in rows[2:])-1e-10
    reduced = cohort.next_cap(allocation, 1, 100.)
    assert reduced == pytest.approx(100.*rows[1]["initial_cap_seconds"] /
                                  sum(r["initial_cap_seconds"] for r in rows[1:]))
    with pytest.raises(ValueError):
        cohort.next_cap(allocation, 2, -1.)


@pytest.mark.parametrize("mutation", [None, "double_charge", "changed_receipt", "duplicate_receipt"])
def test_preparation_reconciliation_releases_old_hold_once(tmp_path, mutation):
    value = preparation(tmp_path)
    if mutation == "double_charge": value["preparation_actual_seconds"] += 10.
    elif mutation == "changed_receipt": Path(value["preparation_receipts"][0]["path"]).write_text("changed")
    elif mutation == "duplicate_receipt": value["preparation_receipts"] *= 2
    if mutation is None:
        assert cohort.reconcile_preparation(value) == 10000.
    else:
        with pytest.raises(ValueError): cohort.reconcile_preparation(value)


@pytest.mark.parametrize("status,allowed", [("completed", True), ("resource_stopped", True),
    ("supervisor_budget_exhausted", True), ("repair_required", False),
    ("worker_interrupted", False), ("supervisor_interrupted", False)])
def test_scientific_failure_or_censoring_can_continue_infrastructure_stops(status, allowed):
    settlement = dict(status="settled", evidence_audit_passed=True, cohort_status=status,
        disposition=dict(failed_assessment_ids=["failed-member"]))
    assert cohort.continue_after(settlement) is allowed
    settlement["evidence_audit_passed"] = False
    with pytest.raises(ValueError): cohort.continue_after(settlement)


def test_observer_error_waits_for_bounded_child_and_does_not_orphan_it(tmp_path, monkeypatch):
    receipt = save(tmp_path / "checks.json", dict(status="passed", elapsed_seconds=.1))
    ledger = cohort.EngineeringLedger(tmp_path / "engineering.json", cap=10., initial_receipt=receipt)
    ticks = iter((0., 1.))
    monkeypatch.setattr(cohort, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    marker = tmp_path / "finished"
    program = "import pathlib,sys,time; time.sleep(.1); pathlib.Path(sys.argv[1]).write_text('done')"
    def failed_observer(pid):
        raise ValueError("broken telemetry")
    with pytest.raises(ValueError, match="broken telemetry"):
        cohort.child([sys.executable, "-c", program, str(marker)], output=tmp_path / "process",
            env=os.environ.copy(), cwd=tmp_path, progress=failed_observer, ledger=ledger, poll_seconds=.01)
    assert marker.read_text() == "done"
    assert cohort.read(tmp_path / "process/observation-error.json")["action"] == "wait_for_existing_bounded_supervisor"
    assert ledger.spent == 1.1  # Only the controlled observation interval was charged.


def run_fixture(tmp_path, designs, monkeypatch):
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    source = tmp_path / "source"
    provenance = dict(git_commit="e"*40)
    save(source / ".validation-source.json", provenance)
    rows = cohort.inventory(designs)
    for row in rows:
        row.update(binding_hash="binding", data_sha256="data", geometry_sha256="geometry")
    prices = dict(status="completed", source_root=str(source), numerical_source=provenance,
        rows=rows, supervisor_elapsed_seconds=40.)
    prices_path = save(tmp_path / "pricing/result.json", prices)
    args = SimpleNamespace(output=tmp_path / "continuation", source_root=source, prices=prices_path,
        design=designs, preparation_accounting=save(tmp_path / "preparation.json", preparation(tmp_path)),
        engineering_receipt=save(tmp_path / "checks.json", dict(status="passed", elapsed_seconds=1.)),
        deployment_audit=save(tmp_path / "deployment.json", dict(status="passed")), engineering_cap_seconds=100.,
        recovery_cap_seconds=500., poll_seconds=1., capacity_wait_seconds=10., gpu="0",
        price_unit="original-prices", plan="scientific-plan.md", recovery_namespace=20261019)
    monkeypatch.setattr(cohort, "measured_overhead", lambda path: 2.)
    monkeypatch.setattr(cohort, "wait_prices", lambda path, **kw: cohort.read(path))
    monkeypatch.setattr(cohort, "audit_attempts", lambda *a: [])
    monkeypatch.setattr(cohort, "capacity", lambda *a, **kw: None)
    return args, prices


@pytest.mark.parametrize("price_timeout,recovery_cap", [(False, 500.), (True, 500.), (False, 0.)])
def test_pipeline_waits_audits_launches_all_methods_and_settles_costs_once(tmp_path, designs, monkeypatch, price_timeout, recovery_cap):
    args, prices = run_fixture(tmp_path, designs, monkeypatch)
    args.recovery_cap_seconds = recovery_cap
    events = []
    pricing_charge = 70. if price_timeout else 40.
    if price_timeout:
        prices["status"] = "supervisor_budget_exhausted"
        save(args.prices, prices)
        def validate_resume(prior, paths, **kw):
            assert prior["status"] == "supervisor_budget_exhausted"
            assert paths == designs
            assert kw["seed_namespace"] == 20261019
            events.append("timeout_validation")
        monkeypatch.setattr(cohort, "resume_rows", validate_resume)
    def report(path, paths, **kw):
        assert kw["available_seconds"] == 9900.-pricing_charge
        events.append("price_audit")
        value = price_report(paths)
        value["actual_price_worker_seconds"] = pricing_charge
        return value
    monkeypatch.setattr(cohort, "build_report", report)
    def child(command, **kw):
        phase = Path(command[command.index("--output")+1]).parent
        if "--resume-prices" in command:
            assert price_timeout
            assert command[command.index("--resume-prices")+1] == str(args.prices)
            assert command[command.index("--seed-namespace")+1] == "20261019"
            assert command[command.index("--max-wall-seconds")+1] == "500.0"
            assert events == ["timeout_validation"]
            events.append("price_recovery")
            recovered = dict(prices, status="completed", supervisor_elapsed_seconds=pricing_charge)
            save(phase / "run/result.json", recovered)
            return 0
        events.append(phase.name)
        launch = cohort.read(phase / "launch.json")
        assert launch["source_commit"] == "e"*40
        save(phase / "run/manifest.json", dict(status="qualifying"))
        kw["progress"](123)
        observed = cohort.read(args.output / "status.json")
        assert observed["status"] == "running_" + phase.name
        current = next(row for row in observed["methods"] if row["method"] == phase.name)
        assert current["status"] == "running" and current["child_pid"] == 123
        assert current["child_progress"]["manifest.json"]["status"] == "qualifying"
        method_index = cohort.METHODS.index(phase.name)
        assert all(row["status"] == "settled" for row in observed["methods"][:method_index])
        assert all(row["status"] == "not_started" for row in observed["methods"][method_index+1:])
        save(phase / "run/supervisor.json", dict(exit_code=0, elapsed_seconds=10.,
            max_wall_seconds=float(command[-1]), status="completed"))
        return 0
    monkeypatch.setattr(cohort, "child", child)
    def audit(phase):
        settlement = settle_budget(cohort.read(phase / "allocation.json"),
                                  cohort.read(phase / "run/supervisor.json"))
        settlement.update(status="settled", evidence_audit_passed=True, cohort_status="completed",
            disposition=dict(failed_assessment_ids=["posterior-failure"]))
        return settlement, {}
    monkeypatch.setattr(cohort, "audit", audit)
    assert cohort.run(args) == 0
    assert events == (["timeout_validation", "price_recovery"] if price_timeout else []) + ["price_audit", *cohort.METHODS]
    accounting = cohort.read(args.output / "accounting.json")
    assert accounting["status"] == "settled"
    assert accounting["pricing_inclusive_seconds"] == pricing_charge
    assert accounting["numerical_inclusive_seconds"] == 40.
    assert accounting["signed_balance_seconds"] == pytest.approx(
        10000.-pricing_charge-40.-accounting["exclusive_engineering_seconds"])
    assert cohort.read(args.output / "status.json")["release_readiness"] is False
    with pytest.raises(FileExistsError): cohort.run(args)


@pytest.mark.parametrize("audit_failure", [False, True])
def test_infrastructure_failure_stops_later_methods_and_withholds_unaudited_balance(tmp_path, designs, monkeypatch, audit_failure):
    args, _ = run_fixture(tmp_path, designs, monkeypatch)
    monkeypatch.setattr(cohort, "build_report", lambda path, paths, **kw: price_report(paths))
    launched = []
    def child(command, **kw):
        phase = Path(command[command.index("--output")+1]).parent
        launched.append(phase.name)
        save(phase / "run/supervisor.json", dict(exit_code=1, elapsed_seconds=10.,
            max_wall_seconds=float(command[-1]), status="completed"))
        return 1
    monkeypatch.setattr(cohort, "child", child)
    def audit(phase):
        if audit_failure:
            raise ValueError("invalid source or tensor evidence")
        settlement = settle_budget(cohort.read(phase / "allocation.json"),
                                  cohort.read(phase / "run/supervisor.json"))
        settlement.update(status="settled", evidence_audit_passed=True,
                          cohort_status="repair_required", disposition={})
        return settlement, {}
    monkeypatch.setattr(cohort, "audit", audit)
    assert cohort.run(args) == 1
    assert launched == ["fixed_hmc"]
    assert cohort.read(args.output / "status.json")["status"] == "requires_inspection"
    accounting = cohort.read(args.output / "accounting.json")
    if audit_failure:
        assert accounting["status"] == "requires_terminal_child_accounting"
        assert accounting["signed_balance_seconds"] is None
        assert accounting["unused_engineering_hold_released_seconds"] == 0.
    else:
        assert accounting["status"] == "settled"
        assert accounting["numerical_inclusive_seconds"] == 10.


@pytest.mark.parametrize("failure", ["price_audit", "price_incomplete", "source", "engineering", "gpu"])
def test_no_numerical_launch_before_audited_prices_and_valid_environment(tmp_path, designs, monkeypatch, failure):
    args, prices = run_fixture(tmp_path, designs, monkeypatch)
    def never_launch(*a, **kw):
        pytest.fail("a numerical child launched before preconditions passed")
    monkeypatch.setattr(cohort, "child", never_launch)
    def bad_report(*a, **kw):
        raise ValueError("incomplete or corrupt price receipts")
    monkeypatch.setattr(cohort, "build_report", bad_report)
    if failure == "source":
        prices["source_root"] = "wrong"
        save(args.prices, prices)
    elif failure == "price_incomplete":
        prices["status"] = "running"
        save(args.prices, prices)
    elif failure == "engineering":
        save(args.engineering_receipt, dict(status="failed", elapsed_seconds=1.))
    elif failure == "gpu":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        with pytest.raises(ValueError, match="visible GPU"): cohort.run(args)
        return
    assert cohort.run(args) == 1
    assert cohort.read(args.output / "status.json")["status"] == "requires_inspection"


def test_live_original_receipt_never_permits_launch_or_false_settlement(tmp_path, monkeypatch):
    path = save(tmp_path / "prices.json", dict(status="running", max_wall_seconds=7200.,
        started_utc=datetime.now(timezone.utc).isoformat(), supervisor_elapsed_seconds=1.))
    receipt = save(tmp_path / "checks.json", dict(status="passed", elapsed_seconds=1.))
    ledger = cohort.EngineeringLedger(tmp_path / "engineering.json", cap=100., initial_receipt=receipt)
    monkeypatch.setattr(cohort, "service_state", lambda unit: dict(ActiveState="active"))
    def stop_wait(seconds):
        raise RuntimeError("test ended waiting without launching")
    monkeypatch.setattr(cohort.time, "sleep", stop_wait)
    with pytest.raises(RuntimeError, match="without launching"):
        cohort.wait_prices(path, unit="prices", progress=dict(status_file=str(tmp_path / "status.json")),
            ledger=ledger, poll_seconds=1., grace_seconds=10.)


def test_exhausted_price_allowance_settles_timeout_without_retry_or_allocation(tmp_path, designs, monkeypatch):
    args, prices = run_fixture(tmp_path, designs, monkeypatch)
    args.recovery_cap_seconds = 0.
    prices["status"] = "supervisor_budget_exhausted"
    save(args.prices, prices)
    def forbidden(*a, **kw):
        pytest.fail("an exhausted price allowance triggered more work")
    monkeypatch.setattr(cohort, "child", forbidden)
    monkeypatch.setattr(cohort, "resume_rows", forbidden)
    monkeypatch.setattr(cohort, "build_report", forbidden)
    assert cohort.run(args) == 1
    status = cohort.read(args.output / "status.json")
    assert "no recovery allowance" in status["error"]
    assert all(row["status"] == "not_started" for row in status["methods"])
    accounting = cohort.read(args.output / "accounting.json")
    assert accounting["status"] == "settled"
    assert accounting["pricing_inclusive_seconds"] == 40.
    assert accounting["numerical_inclusive_seconds"] == 0.
    assert accounting["signed_balance_seconds"] == pytest.approx(
        10000.-40.-accounting["exclusive_engineering_seconds"])


@pytest.mark.parametrize("cap", [-1., float("nan"), float("inf"), True])
def test_invalid_recovery_allowance_cannot_launch(tmp_path, designs, monkeypatch, cap):
    args, _ = run_fixture(tmp_path, designs, monkeypatch)
    args.recovery_cap_seconds = cap
    with pytest.raises(ValueError, match="nonnegative"):
        cohort.run(args)
    assert not args.output.exists()



@pytest.mark.parametrize('worker_code', [0, 1, -9, -15])
def test_launcher_exit_matches_real_posix_process(worker_code):
    import subprocess
    result = subprocess.run([sys.executable, '-c', f'import sys; sys.exit({worker_code})'])
    checked = cohort.validate_launcher_exit(result.returncode, dict(exit_code=worker_code))
    assert checked['worker_exit_code'] == worker_code
    assert checked['launcher_exit_code'] == worker_code % 256


@pytest.mark.parametrize('observed,recorded', [(-9,-9), (0,-9), (1,-9), (247,0), (True,1), (0,None), (0,256)])
def test_launcher_exit_rejects_mismatch_or_signalled_launcher(observed, recorded):
    with pytest.raises(ValueError, match='launcher exit'):
        cohort.validate_launcher_exit(observed, dict(exit_code=recorded))


def install_timeout_pipeline(monkeypatch):
    launched = []
    monkeypatch.setattr(cohort, 'build_report', lambda path, paths, **kw: price_report(paths))
    def child(command, **kw):
        phase = Path(command[command.index('--output')+1]).parent
        launched.append(phase.name)
        timeout = phase.name == 'fixed_hmc'
        save(phase / 'run/supervisor.json', dict(exit_code=-9 if timeout else 0,
            elapsed_seconds=10., max_wall_seconds=float(command[-1]),
            status='supervisor_budget_exhausted' if timeout else 'completed'))
        return 247 if timeout else 0
    monkeypatch.setattr(cohort, 'child', child)
    def audit(phase):
        supervisor = cohort.read(phase / 'run/supervisor.json')
        settlement = settle_budget(cohort.read(phase / 'allocation.json'), supervisor)
        settlement.update(status='settled', evidence_audit_passed=True,
            cohort_status=supervisor['status'], disposition=dict(unfinished_verified_ids=['censored']))
        return settlement, {}
    monkeypatch.setattr(cohort, 'audit', audit)
    return launched


def test_audited_timeout_settles_and_all_remaining_methods_run(tmp_path, designs, monkeypatch):
    args, _ = run_fixture(tmp_path, designs, monkeypatch)
    launched = install_timeout_pipeline(monkeypatch)
    assert cohort.run(args) == 0
    assert launched == list(cohort.METHODS)
    assert cohort.read(args.output/'fixed_hmc/settlement.json')['cohort_status'] == 'supervisor_budget_exhausted'
    assert cohort.read(args.output/'accounting.json')['numerical_inclusive_seconds'] == 40.


@pytest.mark.parametrize('damage', [None, 'live', 'allocation', 'environment', 'prices', 'hole', 'engineering', 'missing_child', 'exit'])
def test_recovery_reuses_terminal_prefix_and_rejects_changed_evidence(tmp_path, designs, monkeypatch, damage):
    args, _ = run_fixture(tmp_path, designs, monkeypatch)
    launched = install_timeout_pipeline(monkeypatch)
    correct_exit_check = cohort.validate_launcher_exit
    def old_bug(*a):
        raise ValueError('historical launcher comparison failed')
    monkeypatch.setattr(cohort, 'validate_launcher_exit', old_bug)
    assert cohort.run(args) == 1
    assert launched == ['fixed_hmc']
    prior = args.output
    prior_engineering = cohort.read(prior/'engineering.json')['charged_seconds']
    monkeypatch.setattr(cohort, 'validate_launcher_exit', correct_exit_check)
    monkeypatch.setattr(cohort, 'service_state', lambda unit: dict(ActiveState='active' if damage == 'live' else 'failed'))
    if damage in {'allocation','environment','prices'}:
        path = prior / ('cohort-allocation.json' if damage == 'allocation' else 'status.json' if damage == 'environment' else 'fixed_hmc/launch.json')
        value = cohort.read(path)
        if damage == 'allocation': value['available_numerical_seconds'] += 1
        elif damage == 'environment': value['environment']['CUDA_VISIBLE_DEVICES'] = '9'
        else: value['prices_sha256'] = 'corrupt'
        save(path, value)
    elif damage == 'hole':
        (prior/'fixed_hmc').rename(prior/'snaper')
    elif damage == 'engineering':
        value = cohort.read(prior/'engineering.json'); value['charged_seconds'] += 1
        save(prior/'engineering.json', value)
    elif damage == 'missing_child':
        (prior/'fixed_hmc/run/supervisor.json').unlink()
    elif damage == 'exit':
        save(prior/'fixed_hmc/process/returncode.json', dict(launcher_exit_code=1))
    before = {str(p):cohort.digest(p) for p in prior.rglob('*') if p.is_file()}
    args.output = tmp_path/'recovery'
    args.resume_from, args.resume_unit = prior, 'stopped-controller'
    args.engineering_receipt = save(tmp_path/'repair-checks.json', dict(status='passed', elapsed_seconds=2.))
    launched.clear()
    code = cohort.run(args)
    assert {str(p):cohort.digest(p) for p in prior.rglob('*') if p.is_file()} == before
    if damage is not None:
        assert code == 1
        assert launched == []
        assert cohort.read(args.output/'status.json')['status'] == 'requires_inspection'
    else:
        assert code == 0
        assert launched == list(cohort.METHODS[1:])
        ledger = cohort.read(args.output/'engineering.json')
        assert ledger['entries'][0]['seconds'] == prior_engineering
        assert ledger['entries'][1]['seconds'] == 2.
        accounting = cohort.read(args.output/'accounting.json')
        assert accounting['numerical_inclusive_seconds'] == 40.
        assert accounting['pricing_inclusive_seconds'] == 40.
        assert accounting['signed_balance_seconds'] == pytest.approx(10000.-80.-ledger['charged_seconds'])
        recovery = cohort.read(args.output/'recovery.json')
        assert recovery['recovered_methods'] == ['fixed_hmc']
        assert recovery['numerical_charged_once_seconds'] == 10.
        assert cohort.read(args.output/'status.json')['methods'][0]['recovered_without_rerun']
