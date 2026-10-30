"""CPU-only diagnostic fixtures for campaign recovery and numerical wiring."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time

import pytest
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sir_training_master", ROOT/"scripts/run_sir_neutra_training.py")
master = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master)


def test_budget_counts_live_and_completed_attempts():
    state = dict(budgets_seconds=dict(training=100.,hmc=200.), attempts=[
        dict(budget="training",wall_seconds=30.),
        dict(budget="hmc",wall_seconds=20.),
        dict(budget="training",started=time.time()-10,cap_seconds=40)])
    assert 59. <= master.remaining(state,"training") <= 60.
    assert master.remaining(state,"hmc") == 180.


def test_crash_recovery_charges_and_reuses_complete_result(tmp_path):
    campaign=master.Campaign(tmp_path)
    out=tmp_path/"old"
    master.save(out/"result.json",dict(valid=True,answer=42))
    master.save(out/"manifest.json",dict(status="FINISHED",wall_seconds=123.))
    campaign.state["attempts"].append(dict(key="x",budget="training",status="RUNNING",
        pid=99999999,output=str(out),started=time.time()-130,cap_seconds=500))
    campaign.reconcile()
    assert campaign.state["attempts"][0]["wall_seconds"]==123.
    campaign.state["budgets_seconds"]["training"]=123.
    assert campaign.completed("x")
    result,path=campaign.task("x","train")
    assert result["answer"]==42 and path==out
    with pytest.raises(RuntimeError,match="live supervisor"):
        master.Campaign(tmp_path)


def test_next_phase_preserves_separate_budgets(tmp_path):
    campaign=master.Campaign(tmp_path)
    campaign.next("training","fresh draws")
    result=master.read(tmp_path/"next-phase.json")
    assert result["training_seconds_remaining"]==16*3600
    assert result["hmc_seconds_remaining"]==24*3600
    assert master.latent_seed(201,3)!=master.latent_seed(201,4)
    assert master.latent_seed(201,3,2)!=master.latent_seed(201,3,1)
    assert [master.learning_rate(.01,i) for i in (127,128,383,384)]==[.01,.005,.005,.0025]


def test_repair_preserves_budget_cached_results_and_parent_evidence(tmp_path,monkeypatch):
    parent=tmp_path/"parent"
    original=master.Campaign(parent)
    out=parent/"001-completed"
    master.save(out/"result.json",dict(valid=True,checkpoint="unchanged"))
    original.state["attempts"].append(dict(key="completed",budget="training",
        status="FINISHED",exit_code=0,output=str(out),wall_seconds=400.))
    original.persist()
    previous=(parent/"campaign.json").read_bytes()
    old_hashes=original.state["source_sha256"]
    original.lock.close()
    monkeypatch.setattr(master,"hashes",lambda:{"repaired":"source"})
    repaired=master.Campaign(tmp_path/"repair",resume_from=parent,repair_note=master.PLAN)
    assert (parent/"campaign.json").read_bytes()==previous
    assert repaired.state["repairs"][0]["parent_source_sha256"]==old_hashes
    assert master.remaining(repaired.state,"training")==16*3600-400
    assert master.remaining(repaired.state,"hmc")==24*3600
    result,path=repaired.task("completed","train")
    assert result["checkpoint"]=="unchanged" and path==out
    assert len(repaired.state["attempts"])==1


def test_repair_refuses_live_parent(tmp_path):
    parent=master.Campaign(tmp_path/"parent")
    with pytest.raises(RuntimeError,match="live supervisor"):
        master.Campaign(tmp_path/"repair",resume_from=parent.root,repair_note=master.PLAN)
    parent.lock.close()
    master.save(parent.root/"campaign.json",{**parent.state,"attempts":[dict(status="RUNNING")]})
    with pytest.raises(RuntimeError,match="reconcile parent workers"):
        master.Campaign(tmp_path/"repair2",resume_from=parent.root,repair_note=master.PLAN)


def test_xla_logabsdet_signed_nondiagonal_singular_fixtures():
    matrices=tf.constant([[[2.,1.,3.],[4.,-2.,1.],[0.,5.,2.]],
                          [[-2.,0.,0.],[0.,.25,0.],[0.,0.,3.]],
                          [[.01,2.,3.],[0.,.02,2.],[0.,0.,.03]],
                          [[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]]],tf.float64)
    fn=tf.function(lambda x:master.logabsdet3(tf,x),
        input_signature=[tf.TensorSpec([4,3,3],tf.float64)],jit_compile=True)
    result=fn(matrices)
    reference=tf.linalg.slogdet(matrices)[1]
    tf.debugging.assert_near(result[:3],reference[:3],atol=1e-12,rtol=1e-12)
    assert float(result[-1])==float("-inf")
    assert "LogMatrixDeterminant" not in {op.type for op in fn.get_concrete_function().graph.get_operations()}
    with pytest.raises(ValueError,match="3x3"):
        master.logabsdet3(tf,tf.eye(2,dtype=tf.float64))


def quadratic_trainer(tf,job):
    from dataclasses import replace
    from bayesfilter.inference.neutra_transport import (
        NeuTraTransport,NeuTraTransportConfig,NeuTraOptimizerConfig,NeuTraTransportTrainer)
    @tf.function(input_signature=[tf.TensorSpec([4,3],tf.float32)],jit_compile=True)
    def target(q):
        return -tf.reduce_sum(q*q,axis=-1)/2,-q,tf.reduce_all(tf.math.is_finite(q),axis=-1)
    scope=dict(test="quadratic CPU diagnostic")
    transport=NeuTraTransport(replace(NeuTraTransportConfig.hoffman_author_iaf(
        3,conditional_scale_cap=2.,seed=(261021,job["seed"]),dtype="float32"),
        hidden_layers=(3,3)))
    trainer=NeuTraTransportTrainer(transport,target,NeuTraOptimizerConfig(
        batch_size=8,shard_size=4,learning_rate=job["rate"],target_dtype="float32",
        estimator="standard",beta1=.9,beta2=.999,epsilon=1e-7,
        gradient_clip_norm=1.,jit_compile=True),
        target_signature="a"*64)
    return None,SimpleNamespace(value_score=target,scope=scope,target_signature="a"*64),trainer


def test_training_resume_restores_adam_and_random_stream(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"target_and_trainer",quadratic_trainer)
    monkeypatch.setenv("BAYESFILTER_WORKER_WALL_DEADLINE",str(time.monotonic()+600))
    job=dict(key="quadratic",seed=201,rate=.001,width=3,scale=1.,updates=4)
    full=master.train(tf,job,tmp_path/"full")
    first=master.train(tf,{**job,"updates":2},tmp_path/"first")
    resumed=master.train(tf,{**job,"resume":first["checkpoint"]},tmp_path/"resumed")
    a,b=map(master.read,(full["checkpoint"],resumed["checkpoint"]))
    assert a["trainer"]==b["trainer"]
    assert a["attempted"]==b["attempted"]==4
    assert a["next_latent_seed"]==b["next_latent_seed"]
    assert full["valid"] and resumed["valid"]
    # A saved failed monitor must not be erased by a healthy final selection.
    b["monitor_failed"]=True
    master.save(tmp_path/"failed.json",b)
    failed=master.train(tf,{**job,"updates":5,"resume":str(tmp_path/"failed.json")},tmp_path/"failed")
    assert not failed["valid"] and failed["attempted"]==4


def test_invalid_batch_consumes_seed_without_adam_update(tmp_path,monkeypatch):
    seeds=[]
    def factory(tf,job):
        case,target,trainer=quadratic_trainer(tf,job)
        original=trainer.train_step
        def step(z):
            seeds.append(z.numpy().tolist())
            if len(seeds)==1:
                result=trainer.evaluate(z)
                return {**result,"valid":tf.constant(False),
                    "gradient_norm":tf.constant(0.),"clipped_gradient_norm":tf.constant(0.)}
            return original(z)
        trainer.train_step=step
        return case,target,trainer
    monkeypatch.setattr(master,"target_and_trainer",factory)
    monkeypatch.setenv("BAYESFILTER_WORKER_WALL_DEADLINE",str(time.monotonic()+600))
    result=master.train(tf,dict(key="reject",seed=201,rate=.001,updates=1),tmp_path)
    assert result["accepted"]==1 and result["attempted"]==2 and seeds[0]!=seeds[1]


def test_public_callback_reaches_existing_sequential_controller(monkeypatch):
    from bayesfilter.inference import neutra_hmc as shared
    from bayesfilter.inference import hmc_sequential_verification as sequence
    from bayesfilter.inference.hmc_tensorflow_verified import VerifiedTensorFlowHMCConfig
    calls=[]
    monkeypatch.setattr(sequence,"run_sequential_verification",lambda **kw:calls.append(kw) or "ok")
    config=object.__new__(VerifiedTensorFlowHMCConfig)
    chunk=lambda *args:None
    assert shared.run_sequential_neutra_hmc(config=config,run_chunk=chunk)=="ok"
    assert calls[0]["config"] is config and calls[0]["run_chunk"] is chunk
    with pytest.raises(shared.NeuTraHMCError):
        shared.run_sequential_neutra_hmc(config=config,run_chunk=None)


def test_public_fresh_and_resume_consumers_call_shared_entry():
    import ast
    for name,entry in (("hmc_tensorflow_verified.py","_fresh_sequence"),
                       ("hmc_tensorflow_resume.py","replay_then_continue")):
        tree=ast.parse((ROOT/"bayesfilter/inference"/name).read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==entry)
        calls=[n.func.id for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        assert "run_sequential_neutra_hmc" in calls
        assert "run_sequential_verification" not in calls


def qualified_result():
    diagnostic=dict(rank_normalized_split_rhat=dict(maximum=[1.001]*3),
        rank_normalized_ess=dict(bulk=[500.]*3,tail=[450.]*3),
        mean=dict(mcse_sd_ratio=[.05]*3,pooled_mean=[0.]*3,mean_mcse=[.01]*3))
    return dict(status="VERIFIED",posterior_sampling_complete=True,diagnostics=diagnostic)


def test_downstream_report_needs_only_neutra_rhat_and_completed_sampling():
    result=master.posterior_assessment(qualified_result())
    assert result["primary_status"]=="NEUTRA_POSTERIOR_CRITERIA_MET"
    assert result["qualified"]
    assert not result["default_ready"] and not result["comparative_claim"]
    assert not master.posterior_assessment(None)["qualified"]
    arm=qualified_result()
    arm["posterior_sampling_complete"]=False
    assert not master.posterior_assessment(arm)["qualified"]
    arm=qualified_result()
    arm["status"]="BUDGET_EXHAUSTED"
    assert not master.posterior_assessment(arm)["qualified"]


@pytest.mark.parametrize("values",[
    [1.02]*3, [1.001], None, [], [float("nan")]*3, [float("inf")]*3,
    ["nan",1.,1.], [True]*3, [0.]*3, [-1.]*3,
])
def test_single_method_report_requires_valid_rhat(values):
    arm=qualified_result()
    arm["diagnostics"]["rank_normalized_split_rhat"]["maximum"]=values
    assert not master.posterior_assessment(arm)["qualified"]


@pytest.mark.parametrize("section,field,values",[
    ("rank_normalized_ess","bulk",[399.]*3),
    ("rank_normalized_ess","tail",[199.29,1129.09,769.04]),
    ("rank_normalized_ess","tail",["nan",450.,450.]),
    ("rank_normalized_ess","tail",[]),
    ("mean","mcse_sd_ratio",[.11]*3),
    ("mean","mcse_sd_ratio",None),
])
def test_precision_diagnostics_never_veto_completion(section,field,values):
    arm=qualified_result()
    arm["diagnostics"][section][field]=values
    result=master.posterior_assessment(arm)
    assert result["qualified"]
    assert result["precision_diagnostic_role"]=="diagnostic_only"
    key={"bulk":"bulk_ess","tail":"tail_ess","mcse_sd_ratio":"mcse_sd_ratio"}[field]
    assert result["precision_diagnostics"][key]==values
    assert key not in result["checks"]


def test_reassessment_cli_preserves_sampler_results_budget_and_provenance(tmp_path):
    arm=qualified_result()
    arm["diagnostics"]["rank_normalized_ess"]["tail"]=[199.29,1129.09,769.04]
    previous=dict(result=arm,assessment=dict(qualified=False))
    state=dict(status="NEUTRA_HMC_INCOMPLETE",source_sha256={"worker":"original"},
        budgets_seconds=dict(training=16*3600,hmc=24*3600),attempts=[
            dict(status="FINISHED",budget="hmc",wall_seconds=1234.)])
    master.save(tmp_path/"campaign.json",state)
    master.save(tmp_path/"hmc-result.json",previous)
    master.save(tmp_path/"next-phase.json",dict(phase=state["status"]))
    subprocess.run([sys.executable,str(ROOT/"scripts/run_sir_neutra_training.py"),
        "reassess","--output",str(tmp_path)],check=True,capture_output=True,text=True)
    report=master.read(tmp_path/"hmc-result.json")
    revised=master.read(tmp_path/"campaign.json")
    record=master.read(report["assessment_record"])
    assert report["result"]==previous["result"]
    assert report["assessment"]["qualified"]
    assert revised["status"]=="NEUTRA_POSTERIOR_CRITERIA_MET"
    assert record["previous_report"]==previous
    assert record["workers_launched"]==0 and not record["tensorflow_imported"]
    for key in ("source_sha256","attempts","budgets_seconds"):
        assert revised[key]==state[key]
    assert master.read(tmp_path/"next-phase.json")["hmc_seconds_remaining"]==24*3600-1234


def test_reassessment_refuses_unfinished_worker(tmp_path):
    master.save(tmp_path/"campaign.json",dict(status="NEUTRA_HMC_INCOMPLETE",
        attempts=[dict(status="RUNNING")]))
    with pytest.raises(RuntimeError,match="active campaign"):
        master.reassess_saved_result(tmp_path)
    assert not (tmp_path/"reassessments").exists()


def test_master_runs_only_neutra_with_all_remaining_hmc_budget(tmp_path,monkeypatch):
    campaign=master.Campaign(tmp_path)
    campaign.state["attempts"].append(dict(key="hmc-dense_affine",budget="hmc",
        status="FINISHED",exit_code=-15,wall_seconds=1234.))
    calls=[]
    def task(key,kind,**kwargs):
        calls.append(dict(key=key,kind=kind,**kwargs))
        return qualified_result(),tmp_path/"neutra"
    monkeypatch.setattr(campaign,"task",task)
    assert campaign.sample_neutra("tuning","frozen")=="NEUTRA_POSTERIOR_CRITERIA_MET"
    assert len(calls)==1
    assert calls[0]["key"]=="hmc-neutra" and calls[0]["arm"]=="neutra"
    assert calls[0]["allowance"]==24*3600-1234
    assert calls[0]["frozen"]=="frozen"
    assert master.read(tmp_path/"hmc-result.json")["assessment"]["qualified"]
    assert not (tmp_path/"hmc-comparison.json").exists()


def test_master_continues_neutra_saved_state_without_comparator(tmp_path,monkeypatch):
    campaign=master.Campaign(tmp_path)
    calls=[]
    def task(key,kind,**kwargs):
        calls.append(dict(key=key,kind=kind,**kwargs))
        if len(calls)==1:
            campaign.state["attempts"].append(dict(key=key,budget="hmc",wall_seconds=321.))
            return dict(status="BUDGET_EXHAUSTED",manifest="saved-manifest"),tmp_path
        return qualified_result(),tmp_path
    monkeypatch.setattr(campaign,"task",task)
    campaign.sample_neutra("tuning","frozen")
    assert [c["key"] for c in calls]==["hmc-neutra","hmc-neutra-continue"]
    assert calls[1]["resume_manifest"]=="saved-manifest"
    assert calls[1]["allowance"]==24*3600-321


def test_master_hmc_worker_binds_frozen_map_for_fresh_and_resume(tmp_path,monkeypatch):
    from bayesfilter.testing import surrogate_hmc_five_model as five
    payload=tmp_path/"map.json"
    frozen=tmp_path/"frozen.json"
    master.save(payload,dict(marker="trained-map"))
    master.save(frozen,dict(frozen_map=str(payload),target_signature="signature",target_scope="scope"))
    monkeypatch.setattr(master,"make_case",lambda *args:"sir-case")
    calls=[]
    monkeypatch.setattr(five,"inference",lambda *args,**kwargs:calls.append((args,kwargs)) or {})
    job=dict(arm="neutra",tuning="tuning",frozen=str(frozen))
    master.hmc(tf,job,tmp_path)
    master.hmc(tf,{**job,"resume_manifest":"saved-manifest"},tmp_path)
    for args,kw in calls:
        assert args[0]=="sir-case" and kw["neutra_payload"]==dict(marker="trained-map")
        assert kw["training_target_signature"]=="signature" and kw["training_target_scope"]=="scope"
    assert calls[0][1]["target_accept_prob"]==.7
    assert calls[1][1]["resume_manifest"]=="saved-manifest"
    assert "target_accept_prob" not in calls[1][1]
    with pytest.raises(ValueError,match="only frozen NeuTra"):
        master.hmc(tf,{**job,"arm":"dense_affine"},tmp_path)
    assert len(calls)==2


@pytest.mark.parametrize("warning,healthy,expected",[(True,True,True),(False,True,False),(True,False,False)])
def test_joint_scope_selection_precision_warning_preserves_health_veto(tmp_path,monkeypatch,warning,healthy,expected):
    from bayesfilter.testing import surrogate_hmc_scope_tuning as tuning
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
    case=SimpleNamespace(name="fixture",horizon=2,p=3,force_dtype=tf.float32,
        precision_policy="uniform",tuning_seed_offset=0,theta0=tf.zeros([3]),
        controls=ProposalControls(),spec=SimpleNamespace(dimension=3),
        chart=lambda q:(q,None),scope=lambda:dict(force_batch_size=2,test="FP32 fixture"))
    monkeypatch.setattr(runtime,"N",4)
    monkeypatch.setattr(runtime,"fresh_observations",lambda *a:tf.zeros([2,3]))
    monkeypatch.setattr(runtime,"design",lambda *a:(tf.zeros([4,3]),))
    monkeypatch.setattr(tuning,"independent_references",lambda *a:{})
    monkeypatch.setattr(tuning,"trajectory_diagnostics",lambda r:{"valid":r["valid"]})
    monkeypatch.setattr(tuning,"control_candidates",lambda base,**kw:("fixture",[base]))
    monkeypatch.setattr(tuning,"make_context",lambda *a,**kw:
        lambda *args:(tf.zeros([4,3]),tf.zeros([4]),tf.constant(True)))
    monkeypatch.setattr(tuning,"make_beta_solver",lambda *a:lambda *args:
        dict(beta=tf.constant([1/3]*3),objective=0.,gap=0.,iterations=1,converged=True))
    monkeypatch.setattr(tuning,"pilot_ratios",lambda *a:(tf.ones([4,3]),tf.zeros([4])))
    monkeypatch.setattr(tuning,"pilot_objective",lambda *a:(tf.constant(1.),))
    calls=[]
    def program(spec,n,horizon,controls,dtype,*,batch_size,record_history=False,**kw):
        def execute(theta,*args):
            calls.append((batch_size,dtype.name))
            result=dict(value=tf.fill([batch_size],tf.constant(8. if dtype==tf.float64 else 10.,dtype)),
                score=tf.ones([batch_size,3],dtype),steps_completed=tf.fill([batch_size],horizon),
                valid=tf.fill([batch_size],healthy if batch_size==2 else True))
            if record_history:result["ancestor_history"]=tf.zeros([batch_size,horizon,n,3],dtype)
            return result
        return execute
    monkeypatch.setattr(tuning,"make_batched_score_filter",program)
    result=tuning.tune(case,tmp_path,native_batch_sizes=(4,2),precision_warning_only=warning)
    assert (result["selected"] is not None)==expected
    assert (4,"float32") in calls and (2,"float32") in calls and (4,"float64") in calls
    b2,b4=(master.read(tmp_path/f"batch-{b}/filter-selection.json") for b in (2,4))
    assert b2["scope"]["force_batch_size"]==2 and b4["scope"]["force_batch_size"]==4
    assert b2["selection_hash"]!=b4["selection_hash"]
    if expected:
        assert b2["selected"]==b4["selected"]
        assert not b2["selected"]["precision_pass"]
        assert runtime.load_selection(case,tmp_path/"batch-2/filter-selection.json")["selected"]["valid"]
        with pytest.raises(ValueError,match="mismatched"):
            runtime.load_selection(case,tmp_path/"batch-4/filter-selection.json")
