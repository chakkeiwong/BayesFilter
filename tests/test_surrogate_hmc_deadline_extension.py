"""CPU-only deadline/continuation mechanics; these are not M13 inference claims."""
from dataclasses import replace
from datetime import datetime, timezone
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import surrogate_hmc_deadline_extension as master
from bayesfilter.inference.surrogate_hmc_program import write_json


@pytest.fixture(autouse=True)
def isolated_active_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(master.previous, "CHECKPOINT", tmp_path/"checkpoint.md")

def read(path):
    return json.loads(Path(path).read_text())


def stopped_parent(status="M13_RESUMED_BUDGET_EXHAUSTED", qualified=False):
    return dict(status=status, qualified=qualified, source_unchanged=True,
        execution=dict(status="COMPLETED", exit_code=0, accounting_complete=True),
        worker=dict(post_warmup_draws_per_chain=8500, verification_passed=True),
        ledger=dict(charged=dict(cpu_worker_seconds=50000., gpu_seconds=49000.)))


def test_deadline_and_cumulative_allowance_are_hong_kong_midnight():
    end = datetime.fromisoformat(master.DEADLINE)
    start = datetime.fromisoformat(master.previous.START)
    assert end.isoformat() == "2026-10-01T16:00:00+00:00"
    assert (end-start).total_seconds() == master.ALLOWANCE == 73338
    assert master.POST_WARMUP_MAXIMUM == 20000


@pytest.mark.parametrize("status,action", [
    ("M13_RESUMED_BUDGET_EXHAUSTED", "resume_verification"),
    ("M13_RETAINED_BUDGET_EXHAUSTED", "resume_retained"),
    ("M13_RETAINED_MATERIAL_DISCREPANCY", "preserve_terminal_result"),
    ("M13_RETAINED_HEALTH_REPAIR_REQUIRED", "preserve_terminal_result"),
    ("M13_RETAINED_PRECISION_OR_MIXING_INCONCLUSIVE", "preserve_terminal_result"),
])
def test_only_healthy_resource_stops_continue(status, action):
    assert master.continuation_action(stopped_parent(status)) == action


@pytest.mark.parametrize("field,value", [("exit_code", 1), ("accounting_complete", False), ("status", "FAILED")])
def test_bad_execution_never_qualifies_or_restarts(field, value):
    parent = stopped_parent("M13_POSTERIOR_SAMPLING_COMPLETE", True)
    parent["execution"][field] = value
    assert master.continuation_action(parent) == "stop_invalid_parent"


def test_sample_cap_and_verified_retained_requirement():
    parent = stopped_parent()
    parent["worker"]["post_warmup_draws_per_chain"] = 20000
    assert master.continuation_action(parent) == "stop_sample_cap"
    parent = stopped_parent("M13_RETAINED_BUDGET_EXHAUSTED")
    parent["worker"]["verification_passed"] = False
    assert master.continuation_action(parent) == "preserve_terminal_result"


def test_all_prior_work_and_preparation_remain_charged(tmp_path, monkeypatch):
    monkeypatch.setattr(master, "DEVELOPMENT", tmp_path)
    monkeypatch.setattr(master, "utc_now", lambda: datetime(2026,10,1,10,tzinfo=timezone.utc))
    monkeypatch.setattr(master.resource, "getrusage", lambda _: SimpleNamespace(ru_utime=2., ru_stime=3.))
    write_json(tmp_path/"test/guard-receipt.json", dict(accounting_complete=True,
        cpu_worker_seconds=10., gpu_seconds=0.))
    ledger = master.resource_ledger(stopped_parent())
    elapsed = (master.utc_now()-datetime.fromisoformat(master.previous.START)).total_seconds()
    assert ledger["charged"]["cpu_worker_seconds"] == max(50015., elapsed)
    assert ledger["charged"]["gpu_seconds"] == max(49000., elapsed)
    assert ledger["limits"]["cpu_worker_seconds"] == 73338.


@pytest.mark.parametrize("parent_status,new_exit", [("M13_RESUMED_BUDGET_EXHAUSTED",0),
    ("M13_RESUMED_BUDGET_EXHAUSTED",1), ("M13_RETAINED_MATERIAL_DISCREPANCY",0)])
def test_supervisor_waits_then_launches_once_with_remaining_limits(tmp_path, monkeypatch, parent_status, new_exit):
    parent = tmp_path/"parent"
    owner = dict(pid=17,start="test")
    write_json(parent/"run-manifest.json", dict(status="RUNNING", owner=owner))
    write_json(parent/"report.json", stopped_parent(parent_status))
    old_tuning = tmp_path/"tuning.json"
    write_json(old_tuning, dict(config=dict(fresh_seed=[1,2])))
    monkeypatch.setattr(master,"PARENT",parent)
    monkeypatch.setattr(master.previous,"ARTIFACTS",tmp_path)
    monkeypatch.setattr(master.previous,"PARENT",old_tuning)
    monkeypatch.setattr(master,"source_hashes",lambda _: {})
    monkeypatch.setattr(master,"process_identity",lambda _:owner)
    monkeypatch.setattr(master,"remaining_seconds",lambda:7200.)
    monkeypatch.setattr(master,"resource_ledger",lambda _:dict(limits=dict(cpu_worker_seconds=73338.,gpu_seconds=73338.),
        charged=dict(cpu_worker_seconds=67000.,gpu_seconds=65000.),jobs=[]))
    waited, calls = [], []
    def wait(seconds):
        waited.append(seconds)
        write_json(parent/"run-manifest.json",dict(status="TERMINAL",owner=owner))
    monkeypatch.setattr(master.time,"sleep",wait)
    def job(command,directory,ledger,**kwargs):
        assert waited and read(parent/"run-manifest.json")["status"] == "TERMINAL"
        calls.append(kwargs)
        write_json(directory/"report.json",dict(status="M13_POSTERIOR_SAMPLING_COMPLETE",qualified=True))
        return dict(status="COMPLETED" if new_exit == 0 else "FAILED",exit_code=new_exit,accounting_complete=True)
    monkeypatch.setattr(master,"run_job",job)
    output=tmp_path/"extension"
    assert master.execute(output) == 0
    result=read(output/"report.json")
    if parent_status == "M13_RETAINED_MATERIAL_DISCREPANCY":
        assert not calls and not result["qualified"]
    else:
        assert len(calls) == 1
        assert calls[0]["timeout"] == 7080.
        assert calls[0]["cpu_allowance"] == 6218.
        assert result["qualified"] is (new_exit == 0)


def test_real_retained_continuation_matches_uninterrupted_chain_and_replays_looks(tmp_path, monkeypatch):
    from bayesfilter.inference import tune_hmc_kernel
    from bayesfilter.inference.neural_force_hmc import PositionOnlyForceEvaluation, TargetPotentialEvaluation
    from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import bind_components
    from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config, read_tensor
    from bayesfilter.testing import surrogate_hmc_m13_retained as fresh
    from bayesfilter.testing import surrogate_hmc_m13_retained_resume as resumed
    from bayesfilter.inference import hmc_sequential_verification as sequential
    variance=math.log(2.)**2
    @tf.function(input_signature=[tf.TensorSpec([None,2],tf.float64)],jit_compile=True)
    def force(x):
        return PositionOnlyForceEvaluation(x/variance,tf.ones(tf.shape(x)[:-1],tf.bool))
    @tf.function(input_signature=[tf.TensorSpec([None,2],tf.float64)],jit_compile=True)
    def endpoint(x):
        valid=tf.ones(tf.shape(x)[:-1],tf.bool)
        return TargetPotentialEvaluation(tf.reduce_sum(x*x,-1)/(2*variance),
            dict(value_valid=valid,force_valid=valid,numerical_failure=~valid,outside_declared_support=~valid))
    scope="Gaussian-midnight-retained-resume-mechanics-only"
    binding=bind_components(force,endpoint,scope)
    adapter=SimpleNamespace(target_scope=scope,adapter_signature=lambda:scope+":"+binding.binding_hash)
    cfg=tuning_config(scope)
    cfg=replace(cfg,nomination=replace(cfg.nomination,
        mass_window_results=(16,32),step_adaptation_results=64,verification_results=64),warmup_results=2000)
    monkeypatch.setattr(sequential,"runtime_budget_guard",lambda:None)
    tuning=tune_hmc_kernel(adapter=adapter,initial_position=(0.,0.),config=cfg,
        parameter_scales=(math.log(2.),)*2,output_dir=tmp_path/"tuning",runner_binding=binding)
    assert tuning.passed
    looks=master.previous.EXTENDED_LOOKS
    reference=dict(reference_sha256="diagnostic-reference")
    calibration=dict(qualified=True,method=fresh.assessment_scope(looks),looks=list(looks),**reference)
    def compare(points, reference, **kwargs):
        assert kwargs["looks"] == looks
        return dict(status="EQUIVALENT" if points.shape[1]==2000 else "INCONCLUSIVE",
            rhat=[1.,1.], means=tf.reduce_mean(points,[0,1]).numpy().tolist(), diagnostic_only=True)
    monkeypatch.setattr(fresh,"assess",compare)
    monkeypatch.setattr(resumed,"assess",compare)
    full=fresh.consume_chunks(tuning,binding,tmp_path/"full",reference,calibration,looks=looks,max_results=2000)
    count=[0]
    def permit(phase,n):
        count[0]+=n
        return count[0]<=1250
    monkeypatch.setattr(sequential,"runtime_budget_guard",lambda:permit)
    prefix=fresh.consume_chunks(tuning,binding,tmp_path/"prefix",reference,calibration,looks=looks,max_results=2000)
    assert prefix["status"] == "M13_RETAINED_BUDGET_EXHAUSTED" and prefix["draws_per_chain"] == 1250
    monkeypatch.setattr(sequential,"runtime_budget_guard",lambda:None)
    real_builder=resumed.build_retained_bound_hmc_archive_runner_from_tuning_result
    generated=[]
    def builder(**kwargs):
        actual=real_builder(**kwargs)
        def run(config):
            generated.append(config.seed)
            return actual.run(config)
        return SimpleNamespace(run=run)
    monkeypatch.setattr(resumed,"build_retained_bound_hmc_archive_runner_from_tuning_result",builder)
    result=resumed.consume_chunks(tuning,binding,tmp_path/"resumed",reference,calibration,
        looks=looks,max_results=2000,resume_report=tmp_path/"prefix/report.json")
    assert generated == [(fresh.SEED[0],fresh.SEED[1]+i) for i in range(5,8)]
    assert result["draws_per_chain"] == 2000 and result["qualified"]
    assert result["assessments"] == full["assessments"]
    def samples(report):
        arrays=[]
        for row in report["chunks"]:
            path=Path(row["manifest"])
            arrays.append(read_tensor(path.parent,read(path)["raw_samples"]))
        return tf.concat(arrays,0)
    tf.debugging.assert_equal(samples(result),samples(full))
    assert len(result["chunks"]) == 8
    corrupt=copy.deepcopy(prefix)
    corrupt["chunks"][2]["sha256"]="wrong"
    write_json(tmp_path/"corrupt.json",corrupt)
    with pytest.raises(ValueError,match="checksum"):
        resumed.consume_chunks(tuning,binding,tmp_path/"rejected",reference,calibration,
            looks=looks,max_results=2000,resume_report=tmp_path/"corrupt.json")
    assert len(generated) == 3


@pytest.mark.parametrize("phase", ["verification", "retained"])
def test_worker_uses_public_authority_and_cumulative_counts(tmp_path, monkeypatch, phase):
    from bayesfilter import inference
    from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config
    from bayesfilter.testing import surrogate_hmc_m13_cumulative as retained
    parent=tmp_path/"parent"
    manifest=parent/"worker/tuning/manifest.json"
    config=tuning_config("mock-worker-wiring")
    sequence=dict(status="PASSED" if phase=="retained" else "BUDGET_EXHAUSTED",
        counts=dict(warmup=9000,verification=9000),events=[dict(rhat=dict(maximum=[1.,1.]))],
        resume_timing=dict(last_chunk_wall_seconds=10.))
    write_json(manifest,dict(config=config.payload(),sequence=sequence,checks={}))
    previous=stopped_parent("M13_RETAINED_BUDGET_EXHAUSTED" if phase=="retained"
                            else "M13_RESUMED_BUDGET_EXHAUSTED")
    previous["worker"].update(tuning_manifest=str(manifest),
        tuning_manifest_sha256=master.file_hash(manifest))
    if phase=="retained":
        prefix=dict(draws_per_chain=1250,status="M13_RETAINED_BUDGET_EXHAUSTED")
        previous["worker"]["posterior"]=prefix
        write_json(parent/"worker/posterior/report.json",prefix)
    write_json(parent/"report.json",previous)
    adapter,binding=object(),object()
    monkeypatch.setattr(master,"PARENT",parent)
    monkeypatch.setattr(master.previous,"components",lambda:(adapter,binding,None,{}))
    monkeypatch.setattr(master.previous,"prerequisites",lambda _:({},{}))
    monkeypatch.setattr(master.previous,"first_retained_chunk_fits",lambda _:True)
    for name in ("REFERENCE","EXTENDED_COVERAGE","BASELINE"):
        monkeypatch.setattr(master.previous,name,manifest)
    calls=[]
    returned=SimpleNamespace(passed=phase=="retained",artifact_manifest_path=str(manifest))
    def tune(**kwargs):
        calls.append(("tune",kwargs))
        assert kwargs["adapter"] is adapter and kwargs["runner_binding"] is binding
        assert kwargs["config"].resume_manifest == str(manifest)
        assert kwargs["config"].verification_max_results == 20000
        assert kwargs["config"].nomination == config.nomination
        return returned
    def load(path,**kwargs):
        calls.append(("load",kwargs))
        assert Path(path)==manifest and kwargs["runner_binding"] is binding
        return returned
    def consume(tuning,actual_binding,out,**kwargs):
        calls.append(("consume",kwargs))
        assert actual_binding is binding and tuning is returned
        assert kwargs["resume_report"]==(parent/"worker/posterior/report.json" if phase=="retained" else None)
        assert kwargs["max_results"]==20000
        assert kwargs["allow_sampling"] is (phase=="retained")
        return dict(status="M13_POSTERIOR_MIXING_INCONCLUSIVE",qualified=False,sampling_qualified=False,
            qualification_scope="cumulative sampling",draws_per_chain=10500 if phase=="retained" else 9000,
            continuation_draws_per_chain=1500 if phase=="retained" else 0,next_phase_plan={})
    monkeypatch.setattr(inference,"tune_hmc_kernel",tune)
    monkeypatch.setattr(inference,"load_verified_tensorflow_hmc_result",load)
    monkeypatch.setattr(retained,"consume_chunks",consume)
    output=tmp_path/"new"
    output.mkdir()
    assert master.worker(output)==0
    report=read(output/"report.json")
    assert [kind for kind,_ in calls]==(["tune","consume"] if phase=="verification" else ["load","consume"])
    assert report["post_warmup_draws_per_chain"]==(9000 if phase=="verification" else 10500)


@pytest.mark.parametrize("verified", [False,True])
def test_known_reporting_failure_recovers_checkpoint_without_forging_success(tmp_path, verified):
    parent=stopped_parent()
    parent["status"]="WORKER_RESOURCE_OR_INFRASTRUCTURE_STOP"
    parent["execution"].update(status="FAILED",exit_code=1)
    write_json(tmp_path/"report.json",parent)
    manifest=tmp_path/"worker/tuning/manifest.json"
    write_json(manifest,dict(artifact_authority=True,passed=verified,handoff_eligible=verified,
        sequence=dict(status="VERIFIED" if verified else "BUDGET_EXHAUSTED",warmup_ready=True,
            events=[dict(checks=dict(finite=True,all_chains_moved=True))],chunks=[{}],
            counts=dict(verification=9000,warmup=9000))))
    log=tmp_path/"worker/command.log"
    log.write_text("surrogate_hmc_resume_program.py\n"
        "    tuning_manifest_sha256=file_hash(tuning.artifact_manifest_path),\n"
        "AttributeError: 'str' object has no attribute 'read_bytes'\n")
    result=master.parent_input(tmp_path)
    assert result["execution"]["exit_code"]==1
    assert result["status"]=="WORKER_RESOURCE_OR_INFRASTRUCTURE_STOP"
    assert result["infrastructure_recovery"]["full_public_artifact_validation_required"]
    assert master.continuation_action(result)==("resume_retained" if verified else "resume_verification")
    assert read(tmp_path/"report.json")==parent
    log.write_text("Some unrelated worker failure")
    assert master.continuation_action(master.parent_input(tmp_path))=="stop_invalid_parent"


@pytest.mark.parametrize("count", [8500, 20000])
def test_legacy_pass_must_be_reassessed_cumulatively(count):
    parent = stopped_parent("M13_RETAINED_QUALIFIED", True)
    parent["worker"]["post_warmup_draws_per_chain"] = count
    assert master.continuation_action(parent) == "resume_retained"
    parent["status"] = "M13_POSTERIOR_SAMPLING_COMPLETE"
    parent["worker"]["posterior"] = dict(schema="bayesfilter.m13.cumulative_posterior.v1", sampling_qualified=True)
    assert master.continuation_action(parent) == "already_qualified"


def test_source_migration_allows_only_reviewed_orchestration_with_intact_snapshot(tmp_path, monkeypatch):
    root, parent = tmp_path/"checkout", tmp_path/"parent"
    orchestration = "bayesfilter/testing/surrogate_hmc_m13_retained.py"
    numerical = "bayesfilter/inference/neural_force_hmc.py"
    expected = {}
    for name in (orchestration, numerical):
        for base in (root, parent/"source"):
            path = base/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original")
        expected[name] = master.file_hash(root/name)
    write_json(parent/"run-manifest.json", dict(source_hashes=expected))
    monkeypatch.setattr(master, "ROOT", root)
    monkeypatch.setattr(master, "EXTRA_SOURCES", ())
    (root/orchestration).write_text("reviewed accounting repair")
    assert master.source_hashes(parent)[orchestration] != expected[orchestration]
    (parent/"source"/orchestration).write_text("damaged saved source")
    with pytest.raises(ValueError, match="source closure changed"):
        master.source_hashes(parent)
    (parent/"source"/orchestration).write_text("original")
    (root/numerical).write_text("changed kernel")
    with pytest.raises(ValueError, match="source closure changed"):
        master.source_hashes(parent)
