"""Full-target preparation and posterior-consumer controls; CPU diagnostics."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.testing.inference_validation import modern_full_model_consumer as consumer
from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
from tests.test_modern_hmc_pipeline import setup
from tests.test_mams_full_model_pricing import geometry_fixture


@pytest.mark.parametrize("target,method", [
    ("NL-PP-SGQF", "mams"), ("NL-PP-UKF", "snaper"),
    ("NL-SIR", "randomized_hmc"), ("NL-STR", "fixed_hmc")])
def test_full_data_binding_keeps_prepared_starts_and_explicit_policy(tmp_path, target, method):
    from bayesfilter.testing.inference_validation.modern_nonlinear import make_nonlinear_comparison_target
    fixture=make_nonlinear_comparison_target(target,jit_compile=False)
    directory=tmp_path/"geometry"
    geometry_fixture(directory,fixture,stale=True)
    base,_=setup(family="mams" if method=="mams" else "hmc")
    execution=replace(base.config,transition_profile=HMCTransitionProfile(*consumer.METHOD_PROFILES[method]),
                      seed=(20261012,6000))
    binding,actual,metadata=consumer.full_model_binding(target=target,geometry_directory=directory,
        method=method,seed=6001,execution=execution,epsilon_domain=(.001,4.),repair_factor=1.5,
        max_repairs_per_family=1,source_paths=(__file__,))
    assert binding.config==execution
    assert binding.scope.epsilon_domain==(.001,4.)
    assert binding.scope.max_repairs_per_family==1
    assert binding.scope.repair_factor==1.5
    assert metadata["horizon"]==(100 if target=="NL-STR" else 20)
    np.testing.assert_array_equal(actual.observations,fixture.observations)
    np.testing.assert_allclose(binding.initial_active_state,metadata["contrasts"],atol=1e-12)
    assert metadata["global_coverage_established"] is False


@pytest.mark.parametrize("method,profile", [("nuts",("hmc","fixed")),("mams",("hmc","uniform_duration"))])
def test_wrong_method_label_fails_before_target_preparation(method,profile):
    binding,_=setup()
    with pytest.raises(ValueError,match="method label"):
        consumer.full_model_binding(target="NL-SIR",geometry_directory="nonexistent",method=method,
            seed=1,execution=replace(binding.config,transition_profile=HMCTransitionProfile(*profile)),
            epsilon_domain=(.001,4.),repair_factor=1.5,max_repairs_per_family=1,source_paths=())


def reference():
    from scripts.audit_mams_reference_compatibility import quantity_names
    names=quantity_names("NL-SIR")
    return dict(target="NL-SIR",quantities=names,mean=[0.]*len(names),sd=[1.]*len(names),
                uncertainty=[0.]*len(names))


def precision_policy(**overrides):
    from bayesfilter.inference import HMCPrecisionPolicy, HMCPrecisionTarget
    return HMCPrecisionPolicy(tuple(HMCPrecisionTarget(name, mcse_sd_ratio_max=.1)
        for name in reference()["quantities"]), method="lugsail", jit_compile=False, **overrides)


def test_reference_screen_detects_mean_shift_and_refuses_constant_uncertainty():
    values=tf.random.stateless_normal([2048,4,9],[281,821],dtype=tf.float64)
    exact=reference()
    policy = precision_policy()
    assert consumer.reference_screen(values,exact,mcse_multiplier=3.,normalized_tolerance=.15,precision_policy=policy)["passed"]
    assert not consumer.reference_screen(values+.5,exact,mcse_multiplier=3.,normalized_tolerance=.15,precision_policy=policy)["passed"]
    constant=consumer.reference_screen(tf.zeros_like(values),exact,
        mcse_multiplier=3.,normalized_tolerance=.15,precision_policy=policy)
    assert constant["status"]=="uncertainty_unavailable"
    assert constant["passed"] is False
    assert not consumer.reference_screen(None,exact,mcse_multiplier=3.,normalized_tolerance=.15,precision_policy=policy)["passed"]
    with pytest.raises(ValueError,match="dimensions"):
        consumer.reference_screen(values[:,:,:3],exact,mcse_multiplier=3.,normalized_tolerance=.15,precision_policy=policy)
    explicit = precision_policy(batch_size=128)
    limited = consumer.reference_screen(values, exact, mcse_multiplier=3., normalized_tolerance=.15,
                                       precision_policy=explicit)
    assert limited["status"] == "uncertainty_unavailable"
    assert limited["precision_policy"]["batch_size"] == 128


def posterior_fixture():
    candidates=[SimpleNamespace(candidate_id="root",parent_candidate_id=None),
                SimpleNamespace(candidate_id="child",parent_candidate_id="root"),
                SimpleNamespace(candidate_id="peer",parent_candidate_id=None)]
    result=SimpleNamespace(candidates=candidates,verified_candidate_ids=("child","peer"),
        candidate_states={"root":"repair_requested","child":"verified","peer":"verified"})
    members={name:SimpleNamespace(member_hash="hash-"+name,name=name) for name in ("child","peer")}
    factory=lambda member,index: SimpleNamespace(payload=lambda **kw: dict(jit_compile=False,index=index),
        assessment_policy=SimpleNamespace(precision=precision_policy()))
    return SimpleNamespace(result=result),members,factory


def test_member_veto_does_not_suppress_peer_or_change_tuning_membership(tmp_path,monkeypatch):
    import bayesfilter.inference as inference
    run,members,factory=posterior_fixture()
    calls=[]
    def posterior(**kwargs):
        name=kwargs["member"].name
        calls.append(name)
        if name=="child":
            return dict(passed=False,hard_vetoes=["numerical_veto"],private_retained_raw=None)
        return dict(passed=True,hard_vetoes=[],private_retained_raw=tf.random.stateless_normal(
            [2048,4,9],[281,821],dtype=tf.float64))
    monkeypatch.setattr(inference,"run_hmc_posterior",posterior)
    report=consumer.assess_full_model_members(run=run,members=members,output_dir=tmp_path/"assessment",
        target="NL-SIR",reference=reference(),posterior_config_factory=factory,
        mcse_multiplier=3.,normalized_tolerance=.15,use_xla=False,check_inputs=lambda:None)
    assert calls==["child","peer"]
    assert report["verified_candidate_ids"]==["child","peer"]
    assert len(report["rows"])==3
    assert report["rows"][1]["delivered"] is False
    assert report["rows"][2]["delivered"] is True
    assert report["delivered_count"]==1
    assert report["rows"][1]["parent_candidate_id"]=="root"


def test_resource_stop_keeps_all_posterior_members_unstarted(tmp_path):
    from bayesfilter.runtime.execution_budget import execution_budget
    run,members,factory=posterior_fixture()
    with execution_budget(check=lambda:False):
        report=consumer.assess_full_model_members(run=run,members=members,output_dir=tmp_path/"assessment",
            target="NL-SIR",reference=reference(),posterior_config_factory=factory,
            mcse_multiplier=3.,normalized_tolerance=.15,use_xla=False,check_inputs=lambda:None)
    assert report["status"]=="resource_stopped"
    assert report["rows"][0]["status"]=="not_verified"
    assert all(row["status"]=="posterior_not_started_resource_stopped" for row in report["rows"][1:])
    assert json.loads((tmp_path/"assessment/result.json").read_text())["verified_candidate_ids"]==["child","peer"]


def test_missing_member_and_wrong_reference_fail_before_assessment(tmp_path):
    run,members,factory=posterior_fixture()
    with pytest.raises(ValueError,match="member inventory"):
        consumer.assess_full_model_members(run=run,members={"child":members["child"]},
            output_dir=tmp_path/"missing",target="NL-SIR",reference=reference(),
            posterior_config_factory=factory,mcse_multiplier=3.,normalized_tolerance=.15,
            use_xla=False,check_inputs=lambda:None)
    with pytest.raises(ValueError,match="reference target"):
        consumer.assess_full_model_members(run=run,members=members,
            output_dir=tmp_path/"wrong",target="NL-STR",reference=reference(),
            posterior_config_factory=factory,mcse_multiplier=3.,normalized_tolerance=.15,
            use_xla=False,check_inputs=lambda:None)
    assert not (tmp_path/"missing").exists() and not (tmp_path/"wrong").exists()


def test_supervisor_preserves_completed_and_interrupted_posterior_members(tmp_path):
    from scripts.run_modern_hmc_frozen_full_model import finalize_interruption
    posterior=tmp_path/"posterior"
    posterior.mkdir()
    consumer.write_json(posterior/"result.json",dict(rows=[
        dict(candidate_id="one",status="posterior_checks_passed",delivered=True),
        dict(candidate_id="two",status="posterior_running"),
        dict(candidate_id="three",status="posterior_not_started")]))
    design=dict(target="NL-SIR",method="mams",search=dict(epsilon_by_l=[[3,[1.2]]]))
    result=finalize_interruption(tmp_path,design,status="worker_interrupted")
    assert len(result["rows"])==3
    assert result["rows"][0]["delivered"] is True
    assert result["rows"][0]["status"]=="posterior_checks_passed"
    assert result["rows"][1]["status"].startswith("posterior_interrupted")
    assert result["rows"][2]["status"].startswith("posterior_not_started")
    assert result["requested_primary_pairs"]==[(3,1.2)]


def posterior_settings():
    return dict(sequential=dict(warmup_chunk_results=64,warmup_min_results=128,
        warmup_check_window_results=64,warmup_max_results=256,warmup_rhat_max=1.05,
        retained_chunk_results=64,retained_min_results=128,retained_max_results=256,
        retained_rhat_max=1.01,minimum_chain_count=4,energy_error_log_accept_threshold=-1000.,
        max_results_per_chain=10000,count_budget_reason=None),
        precision=dict(method="lugsail",batch_size=None,min_batches=20,lugsail_r=3,lugsail_c=.5),
        assessment=dict(warmup_bulk_ess_min=0.,warmup_tail_ess_min=0.,warmup_consecutive_checks=1,
                        retained_bulk_ess_min=100.,retained_tail_ess_min=100.),
        seed=(912,111),mcse_sd_ratio_max=.1,reference_mcse_multiplier=3.,reference_normalized_tolerance=.1)


def test_posterior_factory_preserves_member_policy_and_declared_seeds():
    from scripts.run_modern_hmc_frozen_full_model import posterior_factory
    settings=posterior_settings()
    factory=posterior_factory(settings,target="NL-SIR",use_xla=False)
    member=SimpleNamespace(step_size=1.7,num_leapfrog_steps=8)
    config=factory(member,3)
    assert config.step_size==1.7 and config.num_leapfrog_steps==8
    assert config.warmup_seed==(912,117) and config.retained_seed==(912,118)
    assert len(config.assessment_policy.precision.targets)==9
    settings["sequential"]["step_size"]=.5
    with pytest.raises(ValueError,match="override member"):
        posterior_factory(settings,target="NL-SIR",use_xla=False)


@pytest.mark.parametrize("control", ["reference_mcse_multiplier", "reference_normalized_tolerance"])
def test_bad_reference_controls_fail_before_target_work(control):
    from scripts.run_modern_hmc_frozen_full_model import posterior_factory
    settings=posterior_settings()
    settings[control]=float("nan")
    with pytest.raises(ValueError,match="reference-screen"):
        posterior_factory(settings,target="NL-SIR",use_xla=False)


def test_source_inventory_includes_nonlibrary_reference_and_reporting_helpers():
    from scripts.run_modern_hmc_frozen_full_model import execution_source_paths, ROOT
    from bayesfilter.testing.inference_validation.modern_bridge_audit import source_dependencies
    closure=source_dependencies(execution_source_paths(),repository_root=ROOT)
    assert all(str(ROOT/path) in closure for path in (
        "scripts/audit_mams_reference_compatibility.py", "scripts/validate_mams_adaptation.py",
        "scripts/run_modern_hmc_frozen_full_model.py"))


def test_supervisor_timeout_finalizes_manifest_and_preserves_members(tmp_path, monkeypatch):
    from scripts import run_modern_hmc_frozen_full_model as driver
    output=tmp_path/"run"
    design=tmp_path/"design.json"
    consumer.write_json(design,dict(target="NL-SIR",method="mams",search=dict(epsilon_by_l=[[3,[1.2]]])))
    class Process:
        pid=123
        def __enter__(self):
            (output/"posterior").mkdir(parents=True)
            consumer.write_json(output/"manifest.json",dict(status="posterior"))
            consumer.write_json(output/"posterior/result.json",dict(rows=[
                dict(candidate_id="one",status="posterior_checks_passed",delivered=True),
                dict(candidate_id="two",status="posterior_running")]))
            return self
        def __exit__(self,*args):
            return False
        def wait(self,timeout=None):
            if timeout is not None:
                raise driver.subprocess.TimeoutExpired("worker",timeout)
            return -9
    monkeypatch.setattr(driver.subprocess,"Popen",lambda *a,**kw: Process())
    monkeypatch.setattr(driver.os,"killpg",lambda *a: None)
    args=SimpleNamespace(output=output,design=design,cpu_reference=True,max_wall_seconds=10.)
    assert driver.supervise(args)==-9
    result=json.loads((output/"result.json").read_text())
    assert result["rows"][0]["delivered"] is True
    assert result["rows"][1]["status"]=="posterior_interrupted_supervisor_budget_exhausted"
    assert result["posterior"]["status"]=="supervisor_budget_exhausted"
    assert json.loads((output/"manifest.json").read_text())["status"]=="supervisor_budget_exhausted"


def test_interrupted_tuning_keeps_child_parameters_and_uncreated_roots(tmp_path):
    from scripts.run_modern_hmc_frozen_full_model import finalize_interruption
    tuning=tmp_path/"search/tuning"
    tuning.mkdir(parents=True)
    root=dict(candidate_id="root",parent_candidate_id=None,candidate_family_id="family",
              epsilon=1.2,leapfrog_steps=3,transition_spec=None,creation_ordinal=1)
    child=dict(root,candidate_id="child",parent_candidate_id="root",epsilon=.8,
               leapfrog_steps=5,creation_ordinal=2)
    consumer.write_json(tuning/"tuning_checkpoint.json",dict(result=dict(candidates=[root,child],
        candidate_states={"root":"repair_requested","child":"verified"},
        verified_candidate_ids=["child"],completion_status="partial_budget",repair_actions=[{"id":"repair"}])))
    design=dict(target="NL-SIR",method="mams",search=dict(epsilon_by_l=[[3,[1.2]],[9,[1.7]]]))
    result=finalize_interruption(tmp_path,design,status="supervisor_budget_exhausted")
    assert result["uncreated_primary_pairs"]==[(9,1.7)]
    assert result["repair_descendant_count"]==1
    assert result["verified_candidate_ids"]==["child"]
    assert result["rows"][1]["parent_candidate_id"]=="root"
    assert result["rows"][1]["epsilon"]==.8
    assert result["rows"][1]["leapfrog_steps"]==5
    assert result["delivered_count"]==0


def test_cpu_reference_can_use_xla_and_gpu_cannot_silently_disable_it():
    from scripts.run_modern_hmc_frozen_full_model import validate_execution_mode
    validate_execution_mode(SimpleNamespace(use_xla=True), cpu_reference=True)
    validate_execution_mode(SimpleNamespace(use_xla=False), cpu_reference=True)
    validate_execution_mode(SimpleNamespace(use_xla=True), cpu_reference=False)
    with pytest.raises(ValueError, match="requires XLA"):
        validate_execution_mode(SimpleNamespace(use_xla=False), cpu_reference=False)
