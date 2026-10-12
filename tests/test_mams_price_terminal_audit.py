"""Read-only interrupted price audits; no sampler or posterior evidence."""
import json

import pytest

from scripts import audit_mams_full_model_pricing as audit
from scripts.price_mams_adaptation_full_models import finalize_price_manifest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_price_supervisor_finalizes_manifest_without_losing_provenance(tmp_path):
    write(tmp_path/"manifest.json", dict(status="running", git_commit="original"))
    write(tmp_path/"result.json", dict(status="supervisor_budget_exhausted"))
    finalize_price_manifest(tmp_path, elapsed_seconds=12.)
    result=json.loads((tmp_path/"manifest.json").read_text())
    assert result["status"] == "supervisor_budget_exhausted"
    assert result["git_commit"] == "original"
    assert result["supervisor_elapsed_seconds"] == 12.


def test_archived_deadline_reconciliation_is_explicit_and_read_only():
    manifest=dict(status="running")
    result=dict(status="supervisor_budget_exhausted")
    supervisor=dict(exit_code=-9, elapsed_seconds=12., max_wall_seconds=10.)
    record=audit.terminal_status(manifest,result,supervisor)
    assert record["original_artifacts_unchanged"] is True
    assert manifest == dict(status="running")
    assert audit.terminal_status(result,result,supervisor) is None


@pytest.mark.parametrize("override", [dict(exit_code=0),dict(elapsed_seconds=5.)])
def test_unexplained_manifest_disagreement_is_not_reconciled(override):
    with pytest.raises(ValueError,match="status mismatch"):
        audit.terminal_status(dict(status="running"),dict(status="supervisor_budget_exhausted"),
            dict(dict(exit_code=-9,elapsed_seconds=12.,max_wall_seconds=10.),**override))
    with pytest.raises(ValueError,match="status mismatch"):
        audit.terminal_status(dict(status="running"),dict(status="completed"),
            dict(exit_code=0,elapsed_seconds=12.,max_wall_seconds=10.))


def pilot_fixture(tmp_path):
    attempt=dict(status="completed",stage="fixed_policy_window",duration_index=0,call_index=0,
        seed=[19,27],transitions_per_chain=8,epsilon=.7,max_steps=3,
        actual_gradient_work=128,reserved_gradient_work=256)
    pilot=dict(purpose_digest="scope",attempts=[attempt])
    identity=dict(purpose_digest="scope",role="discarded_mams_pilot_only")
    write(tmp_path/"pilot/windows/identity.json",identity)
    candidate=dict(epsilon=.7,leapfrog_steps=3,transition_spec={"epsilon":.7,"max_steps":3})
    inputs=dict(seed=[19,27],transitions=8,candidate=candidate,initial_state_sha256="start")
    call=tmp_path/"pilot/windows/attempts/window"
    write(call/"call-start.json",dict(inputs=inputs))
    directory=tmp_path/"pilot/windows/committed/fixed_policy_window-0-0"
    directory.mkdir(parents=True)
    tensor=directory/"samples.tensor"
    tensor.write_bytes(b"diagnostic-checksum-fixture")
    native=dict(transition_spec=candidate["transition_spec"],use_xla=True,
        samples_device="GPU:0",value_score_row_evaluations=64,
        target_status_row_evaluations=64,sample_chain_call_s=2.)
    bundle=dict(schema="bayesfilter.durable_tensor_checkpoint.v1",
        identity_hash=audit.payload_digest(identity),inputs_hash=audit.payload_digest(inputs),
        metadata=dict(attempt=str(call)),tree=dict(mapping=dict(
            samples=dict(tensor=dict(file=tensor.name,sha256=audit.digest(tensor),shape=[8,4,5],dtype="float64")),
            metadata=dict(scalar=native))))
    path=directory/"bundle.json"
    write(path,bundle)
    path.with_name("bundle.sha256").write_text(audit.digest(path))
    return pilot,path,tensor


def test_pilot_audit_preserves_completed_work_and_unreturned_attempt(tmp_path):
    pilot,_,_=pilot_fixture(tmp_path)
    pilot["attempts"].append(dict(status="running"))
    rows=audit.audit_pilot_bundles(tmp_path,pilot,5)
    assert len(rows)==1 and rows[0]["actual_gradient_work"]==128
    assert rows[0]["tensor_receipts"]==1


@pytest.mark.parametrize("mutation", ["tensor","bundle","seed","work","dimension"])
def test_partial_pilot_corruption_fails_the_audit(tmp_path,mutation):
    pilot,path,tensor=pilot_fixture(tmp_path)
    if mutation=="tensor":
        tensor.write_bytes(b"changed")
    elif mutation=="bundle":
        path.write_text("{}")
    elif mutation=="seed":
        pilot["attempts"][0]["seed"]=[19,28]
    elif mutation=="work":
        pilot["attempts"][0]["actual_gradient_work"]=16
    with pytest.raises(ValueError):
        audit.audit_pilot_bundles(tmp_path,pilot,6 if mutation=="dimension" else 5)
