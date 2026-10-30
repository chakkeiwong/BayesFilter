"""Master phase/cap tests use synthetic archives, not inference evidence."""
import json
from types import SimpleNamespace
import pytest
import tensorflow as tf

from bayesfilter.inference import surrogate_hmc_resume_program as master
from bayesfilter.inference import hmc_tensorflow_tuning as stage
from bayesfilter.inference.surrogate_hmc_program import write_json
from bayesfilter.testing import surrogate_hmc_m13_retained as retained
from bayesfilter.testing.surrogate_hmc_m13_staged_assessment import assessment_scope


@pytest.mark.parametrize("cap,expected_looks", [(750, []),(1250,[1000])])
def test_retained_cap_preserves_full_calibration_without_extra_look(tmp_path, monkeypatch, cap, expected_looks):
    calls, looks_seen = [], []
    def run(config):
        root = config.output_dir
        root.mkdir()
        calls.append(config)
        raw = stage._write_tensor(root/"raw.tfbin", tf.zeros([250,4,2],tf.float64))
        chunks = {}
        for key, tensor in dict(force_fallback=tf.zeros([250,4],tf.bool),
                                is_accepted=tf.ones([250,4],tf.bool),
                                log_accept_ratio=tf.zeros([250,4],tf.float64),
                                delta_h=tf.zeros([250,4],tf.float64)).items():
            chunks[key] = dict(stage._write_tensor(root/(key+".tfbin"),tensor), dtype=tensor.dtype.name)
        manifest = root/"manifest.json"
        write_json(manifest,dict(raw_samples=raw, chunk=chunks,
            checks=dict(finite=True,endpoint_valid=True),chunk_graph_trace_count=1))
        return dict(archive_manifest_path=str(manifest),health_passed=True)
    monkeypatch.setattr(retained,"build_retained_bound_hmc_archive_runner_from_tuning_result",
                        lambda **kwargs:SimpleNamespace(run=run))
    from bayesfilter.inference import hmc_sequential_verification as sequential
    monkeypatch.setattr(sequential,"runtime_budget_guard",lambda:None)
    def assess(points, reference, *, calibration, looks):
        assert tuple(looks)==master.EXTENDED_LOOKS
        looks_seen.append(int(points.shape[1]))
        return dict(status="INCONCLUSIVE",rhat=[1.02,1.02])
    monkeypatch.setattr(retained,"assess",assess)
    tuning=SimpleNamespace(config=SimpleNamespace(nomination=SimpleNamespace(target_scope="mechanics")),
        artifact_manifest_path="diagnostic-only")
    calibration=dict(qualified=True,method=assessment_scope(master.EXTENDED_LOOKS),
        looks=list(master.EXTENDED_LOOKS),reference_sha256="fixture")
    result=retained.consume_chunks(tuning,SimpleNamespace(binding_hash="fixture"),
        tmp_path/"retained",dict(reference_sha256="fixture"),calibration,
        looks=master.EXTENDED_LOOKS,max_results=cap)
    assert result["draws_per_chain"]==cap
    assert result["status"]=="M13_RETAINED_SAMPLE_CAP_INCONCLUSIVE"
    assert result["planned_looks"]==list(master.EXTENDED_LOOKS)
    assert looks_seen==expected_looks
    assert len(calls)==cap//250
    assert calls[0].continuation_manifest is None
    assert all(c.continuation_manifest is not None for c in calls[1:])


def test_phase_handoff_needs_capacity_for_one_chunk(monkeypatch):
    monkeypatch.setenv("BAYESFILTER_WORKER_WALL_DEADLINE",str(master.time.monotonic()+100))
    monkeypatch.setenv("BAYESFILTER_WORKER_CPU_ALLOWANCE","999999")
    assert not master.first_retained_chunk_fits(100)
    assert master.first_retained_chunk_fits(10)


@pytest.mark.parametrize("valid", [True,False])
def test_supervisor_distinguishes_worker_exit_from_scientific_outcome(tmp_path,monkeypatch,valid):
    from datetime import datetime,timedelta,timezone
    now=datetime.now(timezone.utc)
    monkeypatch.setattr(master,"START",(now-timedelta(seconds=60)).isoformat())
    monkeypatch.setattr(master,"DEADLINE",(now+timedelta(seconds=3600)).isoformat())
    monkeypatch.setattr(master,"ARTIFACTS",tmp_path)
    parent=tmp_path/"parent.json"
    parent.write_text(json.dumps(dict(config=dict(fresh_seed=[1,2]))))
    monkeypatch.setattr(master,"PARENT",parent)
    monkeypatch.setattr(master,"SOURCES",())
    monkeypatch.setattr(master,"update_checkpoint",lambda *args:None)
    def job(command,directory,ledger,**kwargs):
        directory.mkdir()
        write_json(directory/"report.json",dict(status="M13_POSTERIOR_SAMPLING_COMPLETE",qualified=True))
        record=dict(status="COMPLETED" if valid else "FAILED",exit_code=0 if valid else 1,
                    accounting_complete=True,cpu_worker_seconds=5.,gpu_seconds=5.)
        for key in ledger["charged"]:
            ledger["charged"][key]+=record[key]
        ledger["jobs"].append(record)
        return record
    monkeypatch.setattr(master,"run_job",job)
    out=tmp_path/"run"
    assert master.execute(out)==0
    result=json.loads((out/"report.json").read_text())
    assert result["qualified"] is valid
    assert result["status"]==("M13_POSTERIOR_SAMPLING_COMPLETE" if valid else "WORKER_RESOURCE_OR_INFRASTRUCTURE_STOP")
