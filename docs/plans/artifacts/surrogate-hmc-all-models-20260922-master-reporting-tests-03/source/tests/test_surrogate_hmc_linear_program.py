"""LG2 preparation remains separate from inference and preserves cumulative costs."""
import hashlib
import json

import pytest

from bayesfilter.inference.surrogate_hmc_linear_program import (
    CHECK_NAMES, INPUT_NAMES, lg2_endpoint_history, lg2_family_budget,
)
from scripts.qualify_surrogate_hmc_linear_endpoint import complete_forecast, remaining


@pytest.mark.parametrize("dry_run", [False, True])
def test_master_dispatches_lg2_endpoint_without_inference(tmp_path, monkeypatch, dry_run):
    from bayesfilter.inference.surrogate_hmc_program import main
    from scripts import qualify_surrogate_hmc_linear_endpoint as endpoint
    calls = []
    monkeypatch.setattr(endpoint, "main", lambda args: calls.append(args) or 0)
    extra = ["--dry-run"] if dry_run else []
    assert main(["--stage", "P3-LG2-endpoint", "--scope", "LG2",
                 "--output", str(tmp_path), *extra]) == 0
    assert calls == [["--output", str(tmp_path), *extra]]


@pytest.mark.parametrize("extra", [["--scope", "LG1"], ["--reference-root", "/tmp/other"],
                                    ["--sampling-timeout", "1"]])
def test_master_rejects_lg2_endpoint_scope_or_contract_overrides(tmp_path, monkeypatch, extra):
    from bayesfilter.inference.surrogate_hmc_program import main
    from scripts import qualify_surrogate_hmc_linear_endpoint as endpoint
    def forbidden(args):
        pytest.fail("invalid scope or ignored override reached LG2 execution")
    monkeypatch.setattr(endpoint, "main", forbidden)
    with pytest.raises(SystemExit):
        main(["--stage", "P3-LG2-endpoint", "--scope", "LG2",
              "--output", str(tmp_path), *extra])


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def record(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def preparation(root, cost=1.):
    out = root/'docs/plans/artifacts/surrogate-hmc-all-models-test-lg2-endpoint-01'
    inputs = {name: record(out/'inputs'/name, '{}') for name in INPUT_NAMES}
    name = 'bayesfilter/testing/linear_probe.py'
    source = record(out/'source'/name, 'fixed source')
    record(root/name, 'fixed source')
    timings = [dict(healthy=True, compilation_included=i == 0, cpu_worker_seconds=cost,
                    wall_seconds=cost) for i in range(5)]
    save(out/'report.json', dict(status='QUALIFIED_INFERENCE_ENDPOINT', qualified=True,
        retained_inference=False, surrogate_hmc_tested=False, checks=dict.fromkeys(CHECK_NAMES, True),
        posterior_comparisons=dict.fromkeys(('B/R', 'C/D', 'D/R', 'C/R'), 'NOT_RUN'),
        complete_transition_costs=timings, diagnostic_costs=timings,
        forecast=complete_forecast(timings, timings)))
    save(out/'run-manifest.json', dict(schema='surrogate_hmc_lg2_endpoint_v1', exit_code=0,
        inputs=inputs, source_hashes={name: source['sha256']}, source_drift=[]))
    save(out/'accounting.json', dict(exit_code=0,
        charged_this_run=dict(cpu_worker_seconds=100., gpu_seconds=100.)))
    return out


def test_qualified_preparation_does_not_claim_retained_inference(tmp_path):
    preparation(tmp_path)
    result = lg2_endpoint_history(tmp_path)
    assert result['qualified']
    assert result['inference_readiness'] == 'DEFERRED_COMPLETE_BUDGET'
    assert result['remaining_family'] == dict(cpu_worker_seconds=1700., gpu_seconds=7100.)
    assert not result['surrogate_hmc_tested']
    assert not result['authorizes_new_inference']
    assert set(result['comparisons'].values()) == {'NOT_RUN'}
    # The closed preparation attempt does not erase the remaining family balance.
    assert remaining(tmp_path) == (0., 0.)


def test_affordable_preparation_still_requires_a_sampling_plan(tmp_path):
    preparation(tmp_path, .01)
    result = lg2_endpoint_history(tmp_path)
    assert result['qualified']
    assert result['inference_readiness'] == 'REQUIRES_INFERENCE_PLAN_AND_PUBLIC_TUNING'
    assert not result['authorizes_new_inference']


@pytest.mark.parametrize('change', ['source', 'current_source', 'input', 'check', 'binding_check', 'forecast', 'inference'])
def test_corrupted_or_misclassified_preparation_is_not_qualified(tmp_path, change):
    out = preparation(tmp_path)
    if change == 'source':
        (out/'source/bayesfilter/testing/linear_probe.py').write_text('changed')
    elif change == 'current_source':
        (tmp_path/'bayesfilter/testing/linear_probe.py').write_text('changed')
    elif change == 'input':
        (out/'inputs/claim').write_text('changed')
    else:
        report = json.loads((out/'report.json').read_text())
        if change == 'check':
            del report['checks']['canonical_call_chain']
        elif change == 'binding_check':
            del report['checks']['independent_typed_bindings']
        elif change == 'forecast':
            report['forecast']['required']['cpu_worker_seconds'] = 1.
        else:
            report['surrogate_hmc_tested'] = True
        save(out/'report.json', report)
    result = lg2_endpoint_history(tmp_path)
    assert not result['qualified']
    assert result['status'] == 'RECONCILIATION_REQUIRED'


def test_running_or_failed_preparation_remains_visible(tmp_path):
    out = preparation(tmp_path)
    run = json.loads((out/'run-manifest.json').read_text())
    del run['exit_code']
    save(out/'run-manifest.json', run)
    assert lg2_endpoint_history(tmp_path)['status'] == 'RUNNING'
    run['exit_code'] = 2
    save(out/'run-manifest.json', run)
    assert lg2_endpoint_history(tmp_path)['status'] == 'ENDPOINT_QUALIFICATION_FAILED'


def test_family_budget_counts_mixed_jobs_once_without_charging_lg4(tmp_path):
    base = tmp_path/'docs/plans/artifacts'
    save(base/'surrogate-hmc-all-models-test-mixed/accounting.json', dict(
        charged_this_run=dict(cpu_worker_seconds=150., gpu_seconds=80.), jobs=[
            dict(command=['python', 'prepare_surrogate_hmc_linear_reference.py', '--model', 'LG2'],
                 cpu_worker_seconds=30., gpu_seconds=0.),
            dict(command=['python', 'qualify_surrogate_hmc_linear_consumer.py', '--model', 'LG4'],
                 cpu_worker_seconds=100., gpu_seconds=80.)]))
    dedicated = base/'surrogate-hmc-all-models-test-lg2-controls-01'
    save(dedicated/'accounting.json', dict(charged_this_run=dict(cpu_worker_seconds=40., gpu_seconds=50.)))
    # Child accounting is already included in the top-level charge.
    save(dedicated/'worker/accounting.json', dict(charged_this_run=dict(cpu_worker_seconds=39., gpu_seconds=50.)))
    balance, attempts = lg2_family_budget(tmp_path)
    assert balance == dict(cpu_worker_seconds=1710., gpu_seconds=7150.)
    assert attempts == 0
    assert remaining(tmp_path) == (350., 450.)
