"""Real filter repeated trials survive a new process at two durable boundaries.

CPU/XLA engineering regression. Two trials cannot certify delivery or posterior
inference; the oracle is uninterrupted execution with identical native streams.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import tensorflow as tf

from bayesfilter.inference import (
    HMCCandidateExecutionConfig, PrecomputedMassArtifact,
    bind_hmc_candidate_set_execution, tune_hmc_kernel,
)
from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
from bayesfilter.testing.acceptance_decision_models import evidence_accounting
from bayesfilter.testing.acceptance_validation_inventory import model_configuration
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from tests.test_hmc_acceptance_protocol import policy


def setup(case):
    p = policy(base_repetitions=1, max_repetitions=2, max_candidates=2)
    cfg = model_configuration(case, policy_payload=p.payload(), evidence_rungs=(1,2),
        seed=(20261002,913), wall_seconds=120, classification="regression",
        expected_outcome="inconclusive_at_cap")
    target = ValidationTarget(cfg["target"], cfg["parameters"], cfg["data"])
    g = cfg["geometry"]
    factor = tf.linalg.diag(tf.constant(g["scale"],tf.float64))
    mass = PrecomputedMassArtifact(position=g["center"], factor=factor,
        covariance=tf.matmul(factor,factor,transpose_b=True),
        adapter_signature=target.adapter_signature(), position_role="recovery_fixture",
        covariance_source="explicit model-specific diagnostic geometry; no quality claim")
    starts = tf.constant(cfg["active_starts"],tf.float64)
    execution = HMCCandidateExecutionConfig(measurement_num_results=65,verification_num_results=65,
        num_warmup_steps=3,seed=tuple(cfg["seed"]),acceptance_policy=p,
        target_status_trace_policy="per_chain_step",use_xla=True,chain_mode="batched",
        reuse_leapfrog_graphs=True,chunk_max_results=34)
    binding = bind_hmc_candidate_set_execution(adapter=target,initial_position=starts,
        start_coordinates="active",target_scope="inference_validation",mass_artifact=mass,
        target_lineage={"model":cfg["target"],"data":cfg["data"]},
        source_paths=[str(Path(__file__).resolve().parents[1]/"bayesfilter/testing/inference_validation/targets.py")],
        config=execution,scope_id="replicated-ssm-recovery-"+case,search_id="same-stream",
        epsilon_domain=(.0001,1.95),repair_factor=1.3,max_repairs_per_family=0)
    search = HMCControllerConfig(primary_l_grid=(1,),epsilon_by_l=((1,(cfg["epsilon_by_l"][0][1][0],)),),
        total_budget_units=10,repair_reserve_units=1,max_candidates=2,evidence_rungs=(1,2),
        replicated_acceptance_policy=p)
    return target,binding,search


def numerical_summary(binding, result):
    trials = {}
    for row in binding._evidence.values():
        for trial in row.get("trials",()):
            key = row["work"]["candidate_id"]+":"+row["work"]["stage"]+":"+str(trial["ordinal"])
            trials[key] = {k:trial[k] for k in ("ordinal","seed","samples","trace","health","scores")}
    return {"states":result.candidate_states,"trials":trials,
            "decisions":[[r.candidate_id,r.stage,r.decision] for r in result.verification_receipts],
            "accounting":evidence_accounting(binding,result)}


@pytest.mark.parametrize("case",["lgssm_qr","nonlinear"])
@pytest.mark.parametrize("boundary",["partial_trial","evidence_before_receipt"])
def test_filter_trials_recover_in_new_process_against_uninterrupted(case,boundary,tmp_path,monkeypatch):
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateSetController, HMCInfrastructureFailure
    target,baseline,search = setup(case)
    complete = tune_hmc_kernel(adapter=target,initial_position=baseline.initial_active_state,
        config=search,candidate_set_adapter=baseline.typed_adapter)
    expected = numerical_summary(baseline,complete.result)
    assert expected["accounting"]["unique_complete_trials"] == 2
    assert set(expected["states"].values()) == {"inconclusive_at_cap"}
    _,binding,_ = setup(case)
    with monkeypatch.context() as patch:
        if boundary == "partial_trial":
            original = binding._run
            calls = []
            def interrupt(*args):
                calls.append(1)
                if len(calls) == 2:
                    raise tf.errors.UnavailableError(None,None,"injected after first native chunk")
                return original(*args)
            patch.setattr(binding,"_run",interrupt)
        else:
            original = HMCTuningCandidateSetController._apply_observation
            def interrupt(self,work,observation):
                raise HMCInfrastructureFailure("injected after evidence before receipt")
            patch.setattr(HMCTuningCandidateSetController,"_apply_observation",interrupt)
        paused = tune_hmc_kernel(adapter=target,initial_position=binding.initial_active_state,
            config=search,candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path/"tuning")
    assert paused.result.completion_status == "paused_infrastructure"
    code = """
import json, sys
from pathlib import Path
from tests.test_hmc_acceptance_ssm_recovery import setup, numerical_summary
from bayesfilter.inference import resume_hmc_candidate_set_tuning
target, _, _ = setup(sys.argv[1])
run = resume_hmc_candidate_set_tuning(Path(sys.argv[2])/'tuning/tuning_checkpoint.json', adapter=target)
summary = numerical_summary(run.adapter._execution_binding, run.result)
Path(sys.argv[2], 'resumed.json').write_text(json.dumps(summary,allow_nan=False))
"""
    env = {k:os.environ[k] for k in ("PATH","LANG","LD_LIBRARY_PATH") if k in os.environ}
    env.update(CUDA_VISIBLE_DEVICES="-1",TF_FORCE_GPU_ALLOW_GROWTH="true",
        TF_NUM_INTRAOP_THREADS="2",TF_NUM_INTEROP_THREADS="1",OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",TF_CPP_MIN_LOG_LEVEL="2",BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    with (tmp_path/"child.log").open("w") as log:
        child = subprocess.run([sys.executable,"-c",code,case,str(tmp_path)],env=env,
            stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert child.returncode == 0, (tmp_path/"child.log").read_text()[-5000:]
    actual = json.loads((tmp_path/"resumed.json").read_text())
    expected = json.loads(json.dumps(expected))
    for field in ("states","trials","decisions"):
        assert actual[field] == expected[field]
    cost = actual["accounting"]
    assert cost["unique_complete_trials"] == cost["independent_trial_streams"] == 2
    assert cost["completed_trial_transitions"] == 2*68*4
    extra = 34*4*2 if boundary == "partial_trial" else 0
    assert cost["gradient_work"] == expected["accounting"]["gradient_work"]+extra
    assert cost["attempted_work_outside_complete_trials"] == extra
