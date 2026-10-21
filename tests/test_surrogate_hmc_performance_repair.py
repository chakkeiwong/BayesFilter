"""CPU-only orchestration regressions; no scientific or GPU evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import run_surrogate_hmc_performance_repair as master


def arguments(tmp_path, **changes):
    base=dict(output=tmp_path/"run", budget_seconds=30., per_model_timeout=10.,
        model="sir_d18", gpu="0", campaign_ledger=None, batch_size=2, paired_repeats=0, unit=False,
        unit_horizon=2, correction_steps=0, offset=.08, fd_ladder=False,
        fd_sir=False, score_only=False, profile=False, fp64_reference=False,debug_horizon=0,fd_widths=[.005,.001])
    base.update(changes)
    return argparse.Namespace(**base)


def test_failed_model_does_not_cancel_independent_model(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"MODELS",("sir_d18","ksc"))
    commands=[]
    def fake(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if len(commands)==1 else 0)
    monkeypatch.setattr(master.subprocess,"run",fake)
    args=arguments(tmp_path,model=None)
    assert master.run(args)==1
    assert len(commands)==2
    assert all(c[c.index("--offset")+1]=="0.08" for c in commands)
    state=json.loads((args.output/"next-phase-plan.json").read_text())
    assert state["status"]=="REPAIR_REQUIRED"
    assert len(state["completed"])==2


def test_fixed_ot_family_reaches_worker_command(tmp_path, monkeypatch):
    commands = []
    def fake(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(master.subprocess, "run", fake)
    assert master.run(arguments(tmp_path, offset=0., filter_tuning=True, fixed_ot_five=True,
                                controls_warm_start_root=tmp_path/"prior")) == 0
    assert "--fixed-ot-five" in commands[0]
    assert "--filter-tuning" in commands[0]
    assert commands[0][commands[0].index("--controls-warm-start-root")+1] == str(tmp_path/"prior")
    assert commands[0][commands[0].index("--priors")+1] == master.LITERATURE_PRIOR


def test_flow_steps_reaches_offline_worker(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(master.subprocess, "run", lambda command, **kwargs:
                        (commands.append(command) or SimpleNamespace(returncode=0)))
    assert master.run(arguments(tmp_path, model="range_bearing", filter_tuning=True,
                                fixed_ot_five=True, flow_steps=16)) == 0
    assert commands[0][commands[0].index("--flow-steps")+1] == "16"
    assert "--fixed-ot-five" in commands[0]


def test_flow_steps_only_overrides_tuning_in_advance(tmp_path, monkeypatch):
    evidence = tmp_path/"precision.json"
    master.save(evidence, dict(models=[dict(model="range_bearing", status="PRECISION_PASS")]))
    ledger = tmp_path/"budget.json"
    master.save(ledger, dict(authorized_gpu_wall_seconds=100., max_attempts=3, attempts=[]))
    calls = []
    def fake_run(child):
        calls.append(child)
        (child.output/child.model).mkdir(parents=True)
        master.save(child.output/child.model/"result.json", dict(status="PASS"))
        return 0
    monkeypatch.setattr(master, "run", fake_run)
    assert master.advance(arguments(tmp_path, model="range_bearing", command="advance",
        flow_steps=16, precision_results=evidence, campaign_ledger=ledger)) == 0
    assert calls[0].filter_tuning and calls[0].flow_steps == 16
    assert calls[1].hmc_mechanics and calls[1].flow_steps is None
    assert calls[1].selection == calls[0].output/"range_bearing"/"filter-selection.json"


def test_flow_step_cli_rejects_inference_override(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["repair", "run", "--model", "range_bearing",
        "--output", str(tmp_path/"run"), "--flow-steps", "16", "--hmc-mechanics"])
    with pytest.raises(SystemExit):
        master.main()


def test_explicit_benchmark_prior_reaches_diagnostic_worker(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(master.subprocess, "run", lambda command, **kwargs:
                        (commands.append(command) or SimpleNamespace(returncode=0)))
    assert master.run(arguments(tmp_path, priors=master.BENCHMARK_PRIOR)) == 0
    assert commands[0][commands[0].index("--priors")+1] == master.BENCHMARK_PRIOR


def test_preparation_cli_defaults_to_literature_prior(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(master, "run", lambda args: received.append(args) or 0)
    monkeypatch.setattr(sys, "argv", ["repair", "run", "--model", "lgssm", "--output", str(tmp_path/"run")])
    assert master.main() == 0
    assert received[0].priors == master.LITERATURE_PRIOR


def test_phase_refresh_reaches_campaign_without_erasing_reference(tmp_path):
    ledger = tmp_path / "budget.json"
    master.save(ledger, dict(authorized_gpu_wall_seconds=100., max_attempts=3,
                            attempts=[dict(wall_seconds=12.)]))
    master.save(tmp_path / "next-phase-plan.json", dict(reference_complete=True))
    args = arguments(tmp_path, campaign_ledger=ledger)
    args.output.mkdir()
    master.publish_advance_state(args, dict(status="RUNNING", model="ksc",
                                          phase="mechanics", next="consumer-cost"))
    campaign = json.loads((tmp_path / "next-phase-plan.json").read_text())
    phase = json.loads((args.output / "next-phase-plan.json").read_text())
    assert campaign["reference_complete"]
    assert campaign["active_execution"] == phase
    assert campaign["remaining_campaign_seconds"] == 88.
    assert campaign["remaining_campaign_attempts"] == 2
    master.publish_advance_state(args, dict(status="PASS", overall_status="REPAIR_OR_BUDGET_REQUIRED",
                                          unresolved_models=["sir_d18"]))
    campaign = json.loads((tmp_path / "next-phase-plan.json").read_text())
    assert campaign["status"] == "REPAIR_OR_BUDGET_REQUIRED"


def test_precision_diagnostic_is_not_a_parity_or_hmc_pass(tmp_path,monkeypatch):
    commands=[]
    def fake(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(master.subprocess,"run",fake)
    args=arguments(tmp_path,score_only=True,profile=True,fp64_reference=True,debug_horizon=10)
    assert master.run(args)==0
    assert "--score-only" in commands[0] and "--fp64-reference" in commands[0] and "--profile" in commands[0]
    assert commands[0][commands[0].index("--debug-horizon")+1]=="10"
    state=json.loads((args.output/"next-phase-plan.json").read_text())
    assert state["status"]=="DIAGNOSTIC_COMPLETE"


def test_timeout_keeps_partial_state_and_does_not_overwrite_evidence(tmp_path,monkeypatch):
    args=arguments(tmp_path)
    def timed(command,**kwargs):
        out=args.output/"sir_d18"
        out.mkdir()
        master.save(out/"result.json",dict(status="COMPILING",data_sha256="kept"))
        raise subprocess.TimeoutExpired(command,1)
    monkeypatch.setattr(master.subprocess,"run",timed)
    assert master.run(args)==1
    result=json.loads((args.output/"sir_d18/result.json").read_text())
    assert result["status"]=="TIMEOUT" and result["last_completed_status"]=="COMPILING"
    assert result["data_sha256"]=="kept"
    with pytest.raises(FileExistsError): master.run(args)


def test_nonfinite_failure_artifact_is_explicit_and_readable(tmp_path):
    path=tmp_path/"result.json"
    master.save(path,dict(status="INVALID",values=[float("nan"),float("inf"),-float("inf"),1.]))
    assert json.loads(path.read_text())["values"]==["nan","inf","-inf",1.]
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.parametrize("extra",[
    ["--profile"],["--budget-seconds","nan"],["--per-model-timeout","inf"],["--mechanics-timeout","nan"],["--mechanics-timeout","0"],
    ["--offset","nan"],["--fp64-reference"],["--fd-ladder"],
    ["--score-only","--unit"],["--correction-steps","-1"],
    ["--debug-horizon","10"],["--fd-widths","0","0.01"],["--fd-widths","0.01"]])
def test_invalid_configuration_fails_before_worker_launch(tmp_path,monkeypatch,extra):
    monkeypatch.setattr(sys,"argv",["master","run","--output",str(tmp_path/"run"),*extra])
    with pytest.raises(SystemExit) as exc: master.main()
    assert exc.value.code==2
    assert not (tmp_path/"run").exists()


def test_campaign_budget_caps_timeout_and_counts_failure(tmp_path,monkeypatch):
    budget=tmp_path/"budget.json"
    master.save(budget,dict(authorized_gpu_wall_seconds=100.,max_attempts=10,
        attempts=[dict(wall_seconds=98.)]))
    seen=[]
    def fake(command,**kwargs):
        seen.append(kwargs["timeout"])
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(master.subprocess,"run",fake)
    assert master.run(arguments(tmp_path,campaign_ledger=budget))==1
    assert seen==[2.]
    ledger=json.loads(budget.read_text())
    assert len(ledger["attempts"])==2
    assert ledger["used_seconds"]>=98.
    assert ledger["attempts"][-1]["wall_seconds"]>=0.


def test_campaign_exhaustion_prevents_launch(tmp_path,monkeypatch):
    budget=tmp_path/"budget.json"
    master.save(budget,dict(authorized_gpu_wall_seconds=100.,max_attempts=10,
        attempts=[dict(wall_seconds=100.)]))
    def forbidden(*args,**kwargs):
        raise AssertionError("exhausted budget launched a worker")
    monkeypatch.setattr(master.subprocess,"run",forbidden)
    assert master.run(arguments(tmp_path,campaign_ledger=budget))==1


def test_unfinished_attempt_conservatively_reserves_budget(tmp_path):
    budget=tmp_path/"budget.json"
    master.save(budget,dict(authorized_gpu_wall_seconds=100.,max_attempts=10,
        attempts=[dict(reserved_seconds=90.,status="RUNNING")]))
    _,remaining=master.campaign_remaining(budget)
    assert remaining==10.


@pytest.mark.parametrize("score,noise,expected",[
    (2.,0.,"AGREES_WITHIN_MATERIALITY"),
    (3.,0.,"SCORE_MISMATCH"),
    (2.,1.,"UNRESOLVED_WIDTH_CONVERGENCE")])
def test_fd_screen_requires_convergence_before_score_agreement(score,noise,expected):
    checks=[dict(width=h,finite_difference=2.+h*h+(noise if i==2 else 0.),
                 analytical_score=score,value_bracket=[-3.+2*h,-3.-2*h])
            for i,h in enumerate((.01,.003,.001))]
    assert master.assess_finite_difference(checks,2.**-52)["status"]==expected


def test_fd_screen_does_not_promote_roundoff_or_two_widths():
    checks=[dict(width=h,finite_difference=2.,analytical_score=2.,
                 value_bracket=[-3000.,-3000.])
            for h in (.001,.0003,.0001)]
    assert master.assess_finite_difference(checks,2.**-23)["status"]=="UNRESOLVED_WIDTH_CONVERGENCE"
    assert master.assess_finite_difference(checks[:2],2.**-52)["status"]=="UNRESOLVED_WIDTH_CONVERGENCE"


def precision_fixture(dtype):
    return dict(status="DIAGNOSTIC_FINISHED",dtype=dtype,model="lgssm",
        particles=4032,horizon=50,batch_size=2,controls={},beta=[.2,.3,.5],
        data_sha256="data",source_sha256={"source":"same"},
        source_dependency_sha256={"dependency":"same"},
        parameter_positions=[[1.,2.],[1.01,2.01]],
        candidate=dict(value=[-100.,-101.],score=[[.001,4.],[3.,5.]],
                       valid=[True,True],steps_completed=[50,50]))


def test_precision_screen_near_zero_absolute_scale():
    a=precision_fixture("float32"); b=precision_fixture("float64")
    a["candidate"]["score"][0][0]=.002
    result=master.compare_precision_results(a,b)
    assert result["status"]=="PRECISION_PASS"
    assert result["rows"][0]["absolute_score_error"][0]==.001
    a["candidate"]["score"][0][1]=4.1
    assert master.compare_precision_results(a,b)["status"]=="PRECISION_REPAIR_REQUIRED"


@pytest.mark.parametrize("mutation",["source","dependency","theta","unfinished","invalid_row","missing_steps","nonfinite"])
def test_precision_screen_rejects_invalid_or_mismatched_pair(mutation):
    a=precision_fixture("float32"); b=precision_fixture("float64")
    if mutation=="source": a["source_sha256"]={"source":"changed"}
    if mutation=="dependency": a["source_dependency_sha256"]={"dependency":"changed"}
    if mutation=="theta": a["parameter_positions"][1][0]=1.02
    if mutation=="unfinished": a["status"]="TIMEOUT"
    if mutation=="invalid_row": a["candidate"]["valid"][1]=False
    if mutation=="missing_steps": a["candidate"]["steps_completed"][1]=49
    if mutation=="nonfinite": a["candidate"]["score"][1][0]="nan"
    assert master.compare_precision_results(a,b)["status"]!="PRECISION_PASS"


def test_precision_driver_visits_both_modes_and_models(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"MODELS",("lgssm","ksc"))
    calls=[]
    def fake(args):
        calls.append((args.model,args.fp64_reference,args.score_only,args.campaign_ledger))
        out=args.output/args.model
        out.mkdir(parents=True)
        item=precision_fixture("float64" if args.fp64_reference else "float32")
        item["model"]=args.model
        master.save(out/"result.json",item)
        return 0
    monkeypatch.setattr(master,"run",fake)
    args=arguments(tmp_path,model=None)
    assert master.precision(args)==0
    assert [(c[0],c[1]) for c in calls]==[("lgssm",False),("lgssm",True),("ksc",False),("ksc",True)]
    assert all(c[2] for c in calls)
    report=json.loads((args.output/"precision.json").read_text())
    assert report["status"]=="PRECISION_PASS"
    assert "mechanics" in json.loads((args.output/"next-phase-plan.json").read_text())["next"]


@pytest.mark.parametrize("fail_tuning", [False, True])
def test_advance_retunes_failed_precision_and_continues_independent_scopes(tmp_path, monkeypatch, fail_tuning):
    monkeypatch.setattr(master, "MODELS", ("sir_d18", "ksc"))
    evidence = tmp_path/"precision.json"
    evidence.write_text(json.dumps(dict(models=[
        dict(model="sir_d18", status="PRECISION_REPAIR_REQUIRED"),
        dict(model="ksc", status="PRECISION_PASS")])))
    monkeypatch.setattr(master, "campaign_remaining", lambda path: (dict(max_attempts=10,attempts=[]), 1000.))
    calls = []
    def fake_run(child):
        calls.append(child)
        destination = child.output/child.model
        destination.mkdir(parents=True)
        failed = fail_tuning and child.model == "sir_d18" and child.filter_tuning
        status = "FILTER_SCOPE_REPAIR_REQUIRED" if failed else (
            "FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC" if child.filter_tuning else "PASS_HMC_MECHANICS_ONLY")
        (destination/"result.json").write_text(json.dumps(dict(status=status)))
        if child.filter_tuning:
            (destination/"filter-selection.json").write_text("{}")
        else:
            assert child.selection.is_file()
        return 1 if failed else 0
    monkeypatch.setattr(master, "run", fake_run)
    args = arguments(tmp_path, model=None, offset=0., command="advance",
        precision_results=evidence, campaign_ledger=tmp_path/"ledger.json")
    assert master.advance(args) == int(fail_tuning)
    assert len(calls) == (3 if fail_tuning else 4)
    assert calls[0].smoother_reset
    assert not next(c for c in calls if c.model=="ksc" and c.filter_tuning).smoother_reset
    assert all(c.per_model_timeout <= args.per_model_timeout if c.filter_tuning else c.per_model_timeout <= 1200 for c in calls)
    report = json.loads((args.output/"advance.json").read_text())
    assert report["models"][-1]["status"] == "HMC_MECHANICS_COMPLETE_POSTERIOR_NOT_RUN"
    assert "remaining_campaign_seconds" in report


def test_advance_subset_preserves_order_and_does_not_repeat_completed_model(tmp_path, monkeypatch):
    evidence = tmp_path / "precision.json"
    master.save(evidence, dict(models=[dict(model=m, status="PRECISION_PASS")
                                      for m in master.MODELS]))
    monkeypatch.setattr(master, "campaign_remaining",
                        lambda path: (dict(max_attempts=10, attempts=[]), 1000.))
    calls = []
    def fake_run(child):
        calls.append((child.model, child.filter_tuning))
        (child.output / child.model).mkdir(parents=True)
        master.save(child.output / child.model / "result.json", dict(
            status="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC" if child.filter_tuning
            else "PASS_HMC_MECHANICS_ONLY"))
        return 0
    monkeypatch.setattr(master, "run", fake_run)
    args = arguments(tmp_path, model=None, models=["ksc", "range_bearing"],
        command="advance", precision_results=evidence, campaign_ledger=tmp_path/"ledger.json")
    assert master.advance(args) == 0
    assert calls == [("ksc", True), ("ksc", False),
                     ("range_bearing", True), ("range_bearing", False)]
    assert [r["model"] for r in json.loads((args.output/"advance.json").read_text())["models"]] == args.models


def test_advance_missing_precision_cannot_launch_and_records_next_phase(tmp_path, monkeypatch):
    evidence = tmp_path/"precision.json"
    evidence.write_text('{"models":[]}')
    monkeypatch.setattr(master, "campaign_remaining", lambda path: (dict(max_attempts=10,attempts=[]), 100.))
    monkeypatch.setattr(master, "run", lambda child: pytest.fail("must not launch without precision evidence"))
    args = arguments(tmp_path, precision_results=evidence, campaign_ledger=tmp_path/"ledger.json")
    assert master.advance(args)==1
    report = json.loads((args.output/"advance.json").read_text())
    phase = json.loads((args.output/"next-phase-plan.json").read_text())
    assert report["models"][0]["status"]=="BASELINE_PRECISION_UNRESOLVED"
    assert phase["status"]=="BASELINE_PRECISION_UNRESOLVED"


@pytest.mark.parametrize("command,extra", [
    ("precision", ["--filter-tuning"]), ("regression", ["--hmc-mechanics", "--selection", "x"]),
    ("run", ["--smoother-reset"]), ("run", ["--hmc-mechanics"]),
    ("advance", []), ("advance", ["--filter-tuning"]),
    ("run", ["--filter-tuning", "--score-only"]),
    ("run", ["--filter-tuning", "--hmc-mechanics", "--selection", "x"])])
def test_invalid_continuation_flags_cannot_launch(tmp_path, monkeypatch, command, extra):
    monkeypatch.setattr(sys, "argv", ["master",command,"--output",str(tmp_path/"new"),*extra])
    with pytest.raises(SystemExit) as exc:
        master.main()
    assert exc.value.code==2
    assert not (tmp_path/"new").exists()

@pytest.mark.parametrize("bad",["unknown_model","unknown_mode","zero_time","nan_time"])
def test_execution_plan_rejects_unsafe_or_incomplete_scope_settings(tmp_path,bad):
    scopes={name:dict(contraction="direct",tuning_timeout_seconds=5.) for name in master.MODELS}
    if bad=="unknown_model": scopes["unrequested"] = scopes.pop("lgssm")
    if bad=="unknown_mode": scopes["lgssm"]["contraction"]="eager"
    if bad=="zero_time": scopes["lgssm"]["tuning_timeout_seconds"]=0.
    if bad=="nan_time": scopes["lgssm"]["tuning_timeout_seconds"]=float("nan")
    path=tmp_path/"execution.json"
    path.write_text(json.dumps(dict(schema="surrogate_hmc_scope_execution_v1",scopes=scopes)))
    with pytest.raises(ValueError):
        master.execution_scopes(path)


def test_advance_propagates_frozen_scope_controls_to_tuning_and_mechanics(tmp_path,monkeypatch):
    execution=tmp_path/"execution.json"
    scopes={name:dict(contraction="matrix" if name=="sir_d18" else "direct",
                      tuning_timeout_seconds=7.) for name in master.MODELS}
    execution.write_text(json.dumps(dict(schema="surrogate_hmc_scope_execution_v1",scopes=scopes)))
    evidence=tmp_path/"precision.json"
    evidence.write_text(json.dumps(dict(models=[dict(model="sir_d18",status="PRECISION_PASS")])))
    monkeypatch.setattr(master,"campaign_remaining",lambda path:(dict(max_attempts=10,attempts=[]),1000.))
    calls=[]
    def fake_run(child):
        calls.append(child)
        destination=child.output/child.model
        destination.mkdir(parents=True)
        (destination/"result.json").write_text(json.dumps(dict(status="PASS")))
        return 0
    monkeypatch.setattr(master,"run",fake_run)
    args=arguments(tmp_path,model="sir_d18",offset=0.,command="advance",
                   precision_results=evidence,campaign_ledger=tmp_path/"ledger.json",
                   execution_plan=execution,mechanics_timeout=3.)
    assert master.advance(args)==0
    assert len(calls)==2
    assert all(c.shared_gaussian_contraction=="matrix" for c in calls)
    assert calls[0].per_model_timeout==7.
    assert calls[1].per_model_timeout==3.


def test_saved_pilot_materializes_nested_tensors_without_numerical_changes(tmp_path):
    class HostArray:
        def tolist(self):
            return [[1.0, float("nan")], [2.0, 3.0]]
    class Tensor:
        def numpy(self):
            return HostArray()
    path = tmp_path / "result.json"
    master.save(path, {"status": "PASS_HMC_MECHANICS_ONLY",
                       "pilot": {"acceptance": Tensor()}})
    assert json.loads(path.read_text()) == {"status": "PASS_HMC_MECHANICS_ONLY",
        "pilot": {"acceptance": [[1.0, "nan"], [2.0, 3.0]]}}
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.parametrize("extra",[
    ["--consumer-cost"], ["--consumer-cost","--model","lgssm"],
    ["--consumer-cost","--unit","--model","lgssm","--selection","selection.json"]])
def test_consumer_cost_rejects_missing_or_mixed_scope(tmp_path,monkeypatch,extra):
    monkeypatch.setattr(sys,"argv",["master","run","--output",str(tmp_path/"run"),*extra])
    with pytest.raises(SystemExit) as exc:
        master.main()
    assert exc.value.code == 2
    assert not (tmp_path/"run").exists()


@pytest.mark.parametrize("extra",[
    ["--force-reuse"], ["--force-reuse", "--model", "lgssm"],
    ["--force-reuse", "--unit", "--model", "lgssm", "--selection", "selection.json"],
    ["--force-reuse", "--consumer-cost", "--model", "lgssm", "--selection", "selection.json"],
    ["--force-reuse", "--debug-horizon", "5", "--model", "lgssm", "--selection", "selection.json"]])
def test_force_reuse_rejects_missing_or_mixed_scope(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["master", "run", "--output", str(tmp_path/"run"), *extra])
    with pytest.raises(SystemExit) as exc:
        master.main()
    assert exc.value.code == 2
    assert not (tmp_path/"run").exists()


@pytest.mark.parametrize("command,extra", [
    ("run", []), ("run", ["--model","sir_d18"]),
    ("run", ["--model","sir_d18","--score-only"]),
    ("precision", ["--model","sir_d18","--score-only","--fp64-reference"]),
    ("run", ["--model","sir_d18","--score-only","--fp64-reference","--selection","x"]),
    ("run", ["--model","sir_d18","--score-only","--fp64-reference","--filter-tuning"])])
def test_precision_islands_cannot_enter_claim_or_incomplete_modes(tmp_path, monkeypatch, command, extra):
    monkeypatch.setattr(sys,"argv",["master",command,"--output",str(tmp_path/"run"),
        "--precision-islands","both_fp32",*extra])
    with pytest.raises(SystemExit) as exc:
        master.main()
    assert exc.value.code == 2
    assert not (tmp_path/"run").exists()


def test_precision_islands_forwards_explicit_mixed_precision_diagnostic(tmp_path, monkeypatch):
    commands=[]
    def fake(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(master.subprocess,"run",fake)
    args=arguments(tmp_path,score_only=True,fp64_reference=True,debug_horizon=10,
                   precision_islands="both_fp32")
    master.run(args)
    assert len(commands)==1
    assert commands[0][commands[0].index("--precision-islands")+1]=="both_fp32"
    assert "--score-only" in commands[0] and "--fp64-reference" in commands[0]
    assert "--filter-tuning" not in commands[0] and "--hmc-mechanics" not in commands[0]


def islands_fixture():
    common=dict(status="DIAGNOSTIC_FINISHED",dtype="float64",tf32=False,jit_compile=True,
        model="test",particles=4032,batch_size=1,controls={},beta=[1.],seed=1,
        data_sha256="data",parameter_positions=[[0.,0.]],source_sha256={},
        source_dependency_sha256={})
    reference=dict(common,horizon=2,trace_fields=["log_increment"],
        candidate=dict(value=[3.],score=[[.1,.3]],trace=[[[1.],[2.]]],
                       score_trace=[[[.2,.1],[-.1,.2]]],valid=[True],steps_completed=[2]))
    candidate=dict(common,horizon=1,
        precision_islands=dict(policy="both_fp32",runtime_module_patched=False),
        candidate=dict(value=[1.],score=[[.2,.1]],valid=[True],steps_completed=[1]))
    return candidate,reference


def test_mixed_prefix_uses_saved_increments_without_implying_hmc_admission():
    a,b=islands_fixture()
    result=master.compare_precision_islands(a,b)
    assert result["status"]=="PREFIX_PRECISION_PASS"
    assert result["rows"][0]["reference_score"]==[.2,.1]
    assert "no tuning or HMC admission" in result["evidence_role"]
    a["candidate"]["score"][0][1]+=.011
    assert master.compare_precision_islands(a,b)["status"]=="PRECISION_REPAIR_REQUIRED"


@pytest.mark.parametrize("mutation",["data","source","controls","unfinished","shape","sum","nan","not_mixed"])
def test_mixed_prefix_rejects_noncomparable_reference(mutation):
    a,b=islands_fixture()
    if mutation=="data": a["data_sha256"]="other"
    if mutation=="source": b["source_sha256"]={"x":"other"}
    if mutation=="controls": b["controls"]={"other":1}
    if mutation=="unfinished": a["status"]="RUNNING"
    if mutation=="shape": b["candidate"]["score_trace"][0][0]=[.2]
    if mutation=="sum": b["candidate"]["value"][0]=4.
    if mutation=="nan": a["candidate"]["score"][0][0]=float("nan")
    if mutation=="not_mixed": a.pop("precision_islands")
    assert master.compare_precision_islands(a,b)["status"]=="INVALID_COMPARISON"


@pytest.mark.parametrize("field,value", [
    ("runtime_precision", "tf32"), ("tuning_seed_offset", True),
    ("tuning_seed_offset", -1), ("tuning_seed_offset", 1.5),
    ("control_family", "unreviewed"), ("missing", None)])
def test_execution_plan_v2_rejects_invalid_scientific_scope(tmp_path, field, value):
    scopes = {name: dict(contraction="direct", tuning_timeout_seconds=7.,
        runtime_precision="uniform", tuning_seed_offset=0,
        control_family="baseline_more_steps_half_epsilon") for name in master.MODELS}
    if field == "missing":
        del scopes["sir_d18"]["runtime_precision"]
    else:
        scopes["sir_d18"][field] = value
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(dict(schema="surrogate_hmc_scope_execution_v2", scopes=scopes)))
    with pytest.raises(ValueError):
        master.execution_scopes(path)


@pytest.mark.parametrize("family,smoother", [
    ("baseline_more_steps_half_epsilon", False), ("smoother_reset_1_2_4_8", True),
    ("fixed_epsilon1_steps5", False)])
def test_advance_v2_binds_same_precision_and_fresh_data_for_both_phases(
        tmp_path, monkeypatch, family, smoother):
    scopes = {name: dict(contraction="direct", tuning_timeout_seconds=7.,
        runtime_precision="uniform", tuning_seed_offset=0,
        control_family="baseline_more_steps_half_epsilon") for name in master.MODELS}
    scopes["sir_d18"].update(runtime_precision="fp64_recursion_fp32_pairs",
                            tuning_seed_offset=20, control_family=family)
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps(dict(schema="surrogate_hmc_scope_execution_v2", scopes=scopes)))
    evidence = tmp_path / "precision.json"
    evidence.write_text(json.dumps(dict(models=[
        dict(model="sir_d18", status="PRECISION_REPAIR_REQUIRED")])))
    monkeypatch.setattr(master, "campaign_remaining", lambda path: (dict(max_attempts=10,attempts=[]), 1000.))
    calls = []
    def fake_run(child):
        calls.append(child)
        destination = child.output / child.model
        destination.mkdir(parents=True)
        (destination / "result.json").write_text(json.dumps(dict(status="PASS")))
        return 0
    monkeypatch.setattr(master, "run", fake_run)
    args = arguments(tmp_path, model="sir_d18", offset=0., command="advance",
        precision_results=evidence, campaign_ledger=tmp_path/"ledger.json", execution_plan=execution)
    assert master.advance(args) == 0
    assert len(calls) == 2
    assert all(c.runtime_precision == "fp64_recursion_fp32_pairs" for c in calls)
    assert all(c.tuning_seed_offset == 20 for c in calls)
    assert calls[0].smoother_reset is smoother
    assert not calls[1].smoother_reset
    assert calls[0].fixed_ot_five is (family == "fixed_epsilon1_steps5")
    assert not calls[1].fixed_ot_five
    assert calls[1].selection == calls[0].output / "sir_d18" / "filter-selection.json"


@pytest.mark.parametrize("phase", ["filter_tuning", "hmc_mechanics", "consumer_cost", "force_reuse"])
def test_worker_command_preserves_mixed_scope(tmp_path, monkeypatch, phase):
    commands = []
    def fake(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(master.subprocess, "run", fake)
    args = arguments(tmp_path, runtime_precision="fp64_recursion_fp32_pairs",
        tuning_seed_offset=20, selection=tmp_path/"selection.json", **{phase: True})
    assert master.run(args) == 0
    command = commands[0]
    assert command[command.index("--runtime-precision")+1] == "fp64_recursion_fp32_pairs"
    assert command[command.index("--tuning-seed-offset")+1] == "20"
    assert "--"+phase.replace("_", "-") in command


def test_execution_plan_cannot_be_silently_overridden_by_global_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["master", "advance", "--output", str(tmp_path/"new"),
        "--execution-plan", "unused.json", "--tuning-seed-offset", "20"])
    with pytest.raises(SystemExit) as error:
        master.main()
    assert error.value.code == 2
    assert not (tmp_path/"new").exists()


@pytest.mark.parametrize("extra", [
    ["--filter-tuning"], ["--hmc-mechanics","--selection","unused.json"],
    ["--unit"], ["--score-only"], ["--offset",".01"], ["--batch-size","1"],
    ["--selection","unused.json"], ["--consumer-cost"], ["--force-reuse"],
])
def test_component_cost_rejects_mixed_authority_or_incompatible_shapes(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["master", "run", "--output", str(tmp_path/"new"),
                                     "--model","lgssm","--component-cost", *extra])
    with pytest.raises(SystemExit) as error:
        master.main()
    assert error.value.code == 2
    assert not (tmp_path/"new").exists()


def test_component_cost_reaches_worker_and_refreshes_own_next_phase(tmp_path, monkeypatch):
    commands = []
    def fake(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(master.subprocess, "run", fake)
    args = arguments(tmp_path, component_cost=True, offset=0,
                     runtime_precision="fp64_recursion_fp32_pairs")
    assert master.run(args) == 0
    assert "--component-cost" in commands[0]
    assert commands[0][commands[0].index("--runtime-precision")+1] == "fp64_recursion_fp32_pairs"
    state = json.loads((args.output/"next-phase-plan.json").read_text())
    assert state["status"] == "DIAGNOSTIC_COMPLETE"
    assert "primitive timings" in state["next"]


@pytest.mark.parametrize("extra", [
    ["--filter-tuning"], ["--hmc-mechanics","--selection","unused.json"],
    ["--unit"], ["--component-cost"], ["--score-only"], ["--batch-size","1"],
    ["--selection","unused.json"], ["--profile"], ["--offset",".01"],
])
def test_covariance_comparison_rejects_mixed_authority(tmp_path, monkeypatch, extra):
    baseline=tmp_path/"baseline.py"
    baseline.write_text("# preserved diagnostic source")
    monkeypatch.setattr(sys,"argv",["master","run","--output",str(tmp_path/"new"),
        "--model","lgssm","--covariance-baseline",str(baseline),*extra])
    with pytest.raises(SystemExit) as error:
        master.main()
    assert error.value.code==2
    assert not (tmp_path/"new").exists()


def test_covariance_comparison_reaches_worker_and_refreshes_phase(tmp_path,monkeypatch):
    commands=[]
    def fake(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(master.subprocess,"run",fake)
    baseline=tmp_path/"baseline.py"
    args=arguments(tmp_path,covariance_baseline=baseline,offset=0,
                   runtime_precision="fp64_recursion_fp32_pairs")
    assert master.run(args)==0
    command=commands[0]
    assert command[command.index("--covariance-baseline")+1]==str(baseline)
    state=json.loads((args.output/"next-phase-plan.json").read_text())
    assert state["status"]=="DIAGNOSTIC_COMPLETE"
    assert "retune" in state["next"]

def test_force_health_replay_propagates_without_tuning_authority(tmp_path,monkeypatch):
    seen=[]
    replay=tmp_path/"old-input-selection.json"
    replay.write_text("{}")
    monkeypatch.setattr(master.subprocess,"run",lambda command,**kw:
        (seen.append(command) or SimpleNamespace(returncode=0)))
    args=arguments(tmp_path,force_health_replay=replay,runtime_precision="fp64_recursion_fp32_pairs")
    assert master.run(args)==0
    assert seen[0][seen[0].index("--force-health-replay")+1]==str(replay)
    assert "--selection" not in seen[0] and "--hmc-mechanics" not in seen[0]
    plan=json.loads((args.output/"next-phase-plan.json").read_text())
    assert plan["status"]=="DIAGNOSTIC_COMPLETE"
    assert "actual force consumer" in plan["next"]


@pytest.mark.parametrize("extra",[
    ["--selection","old.json"],["--hmc-mechanics"],["--filter-tuning"],
    ["--score-only"],["--debug-horizon","1"],["--offset",".1"]])
def test_force_health_replay_rejects_other_execution_modes(tmp_path,monkeypatch,extra):
    replay=tmp_path/"input.json"; replay.write_text("{}")
    monkeypatch.setattr(sys,"argv",["master","run","--model","range_bearing",
        "--output",str(tmp_path/"run"),"--force-health-replay",str(replay),*extra])
    with pytest.raises(SystemExit) as exc:
        master.main()
    assert exc.value.code==2 and not (tmp_path/"run").exists()


def test_attempt_cap_preserves_real_compute_remainder(tmp_path):
    path=tmp_path/"budget.json"
    master.save(path,dict(authorized_gpu_wall_seconds=100.,max_attempts=1,
        attempts=[dict(wall_seconds=25.)]))
    ledger,remaining=master.campaign_remaining(path)
    assert remaining==75.
    assert master.campaign_attempts_remaining(ledger)==0


@pytest.mark.parametrize("entry_point", ["run","advance"])
@pytest.mark.parametrize("limit", ["attempt","campaign_time","phase_time"])
def test_each_launch_boundary_enforces_and_reports_each_limit(tmp_path,monkeypatch,entry_point,limit):
    path=tmp_path/"budget.json"
    master.save(path,dict(authorized_gpu_wall_seconds=100.,
        max_attempts=1 if limit=="attempt" else 10,
        attempts=[dict(wall_seconds=100. if limit=="campaign_time" else 25.)]))
    evidence=tmp_path/"precision.json"
    master.save(evidence,dict(models=[dict(model="sir_d18",status="PRECISION_PASS")]))
    args=arguments(tmp_path,campaign_ledger=path,precision_results=evidence,
        budget_seconds=0. if limit=="phase_time" else 30.)
    def forbidden(*args,**kwargs):
        pytest.fail("a limit was exhausted but a worker launched")
    monkeypatch.setattr(master.subprocess,"run",forbidden)
    if entry_point=="advance":
        monkeypatch.setattr(master,"run",forbidden)
    assert getattr(master,entry_point)(args)==1
    phase=json.loads((args.output/"next-phase-plan.json").read_text())
    expected={"attempt":"ATTEMPT_LIMIT_REACHED",
              "campaign_time":"BUDGET_EXHAUSTED",
              "phase_time":"PHASE_BUDGET_EXHAUSTED"}[limit]
    assert phase["status"]==expected
    assert phase["remaining_campaign_seconds"]==(0. if limit=="campaign_time" else 75.)
    assert phase["remaining_campaign_attempts"]==(0 if limit=="attempt" else 9)
    assert len(json.loads(path.read_text())["attempts"])==1


def test_attempt_limit_between_tuning_and_mechanics_preserves_completed_phase(tmp_path,monkeypatch):
    path=tmp_path/"budget.json"
    master.save(path,dict(authorized_gpu_wall_seconds=100.,max_attempts=1,attempts=[]))
    evidence=tmp_path/"precision.json"
    master.save(evidence,dict(models=[dict(model="sir_d18",status="PRECISION_PASS")]))
    calls=[]
    def fake(child):
        calls.append(child)
        assert child.filter_tuning
        destination=child.output/child.model
        destination.mkdir(parents=True)
        master.save(destination/"result.json",dict(status="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC"))
        master.save(destination/"filter-selection.json",{})
        ledger=json.loads(path.read_text())
        ledger["attempts"].append(dict(wall_seconds=10.))
        master.save(path,ledger)
        return 0
    monkeypatch.setattr(master,"run",fake)
    args=arguments(tmp_path,campaign_ledger=path,precision_results=evidence)
    assert master.advance(args)==1
    assert len(calls)==1
    phase=json.loads((args.output/"next-phase-plan.json").read_text())
    row=phase["completed"][0]
    assert row["filter-tuning"]["status"]=="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC"
    assert row["status"]=="ATTEMPT_LIMIT_REACHED"
    assert row["stopped_phase"]=="mechanics"
    assert phase["remaining_campaign_seconds"]==90.
    assert phase["remaining_campaign_attempts"]==0


@pytest.mark.parametrize("partial_result", [False,True])
def test_advance_reports_timeout_as_time_limit_not_candidate_failure(tmp_path,monkeypatch,partial_result):
    path=tmp_path/"budget.json"
    master.save(path,dict(authorized_gpu_wall_seconds=100.,max_attempts=10,attempts=[]))
    evidence=tmp_path/"precision.json"
    master.save(evidence,dict(models=[dict(model="sir_d18",status="PRECISION_PASS")]))
    calls=[]
    def fake(child):
        calls.append(child)
        destination=child.output/child.model
        destination.mkdir(parents=True)
        if partial_result:
            master.save(destination/"result.json",dict(status="TIMEOUT"))
        master.save(child.output/"next-phase-plan.json",
            dict(completed=[dict(model=child.model,status="TIMEOUT")]))
        return 1
    monkeypatch.setattr(master,"run",fake)
    args=arguments(tmp_path,campaign_ledger=path,precision_results=evidence)
    assert master.advance(args)==1
    assert len(calls)==1
    phase=json.loads((args.output/"next-phase-plan.json").read_text())
    assert phase["status"]=="TUNING_TIME_LIMIT_REACHED"


@pytest.mark.parametrize("bad_limit", [-1,1.5,True])
def test_invalid_attempt_limit_fails_closed(tmp_path,bad_limit):
    path=tmp_path/"budget.json"
    master.save(path,dict(authorized_gpu_wall_seconds=100.,max_attempts=bad_limit,attempts=[]))
    with pytest.raises(ValueError,match="attempt limit"):
        master.campaign_remaining(path)


@pytest.mark.parametrize("charge", [-1.,float("nan"),float("inf")])
def test_invalid_time_charge_cannot_increase_available_budget(tmp_path,charge):
    path=tmp_path/"budget.json"
    path.write_text(json.dumps(dict(authorized_gpu_wall_seconds=100.,max_attempts=10,
        attempts=[dict(wall_seconds=charge)])))
    with pytest.raises(ValueError,match="finite and nonnegative"):
        master.campaign_remaining(path)


def test_full_sequence_flag_reaches_worker(tmp_path, monkeypatch):
    commands=[]
    monkeypatch.setattr(master.subprocess, "run", lambda command, **kwargs:
        (commands.append(command) or SimpleNamespace(returncode=0)))
    args=arguments(tmp_path,component_cost=True,component_sequence=True,offset=0)
    assert master.run(args)==0
    assert "--component-sequence" in commands[0]

def test_full_sequence_requires_component_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(sys,"argv",["master","run","--output",str(tmp_path/"bad"),"--component-sequence"])
    with pytest.raises(SystemExit) as error:master.main()
    assert error.value.code==2
    assert not (tmp_path/"bad").exists()


@pytest.mark.parametrize("mode", ["--consumer-cost", "--force-reuse"])
def test_cli_allows_declared_mixed_consumer_scope(tmp_path, monkeypatch, mode):
    captured=[]
    monkeypatch.setattr(master,"run",lambda args: captured.append(args) or 0)
    monkeypatch.setattr(sys,"argv",["master","run","--output",str(tmp_path/"new"),
        "--model","sir_d18",mode,"--selection","unused.json",
        "--runtime-precision","fp64_recursion_fp32_pairs"])
    assert master.main()==0
    assert captured[0].runtime_precision=="fp64_recursion_fp32_pairs"


@pytest.mark.parametrize("failed_phase", [None, "mechanics", "consumer-cost"])
def test_advance_automatically_checks_consumer_with_fresh_scope_and_budget(
        tmp_path, monkeypatch, failed_phase):
    monkeypatch.setattr(master, "MODELS", ("sir_d18", "ksc"))
    evidence = tmp_path/"precision.json"
    evidence.write_text(json.dumps(dict(models=[
        dict(model=name, status="PRECISION_PASS") for name in master.MODELS])))
    monkeypatch.setattr(master, "campaign_remaining",
                        lambda _: (dict(max_attempts=20, attempts=[]), 1000.))
    calls = []
    def fake_run(child):
        phase = child.output.name
        calls.append((child.model, phase))
        destination = child.output/child.model
        destination.mkdir(parents=True)
        if child.filter_tuning:
            (destination/"filter-selection.json").write_text("{}")
        else:
            assert child.selection.is_file()
            assert child.selection == args.output/child.model/"filter-tuning"/child.model/"filter-selection.json"
        assert sum((child.filter_tuning, child.hmc_mechanics, child.consumer_cost)) == 1
        assert child.per_model_timeout <= child.budget_seconds
        if child.consumer_cost:
            assert child.per_model_timeout <= 7.
        failed = child.model == "sir_d18" and phase == failed_phase
        (destination/"result.json").write_text(json.dumps(dict(
            status="REPAIR_REQUIRED" if failed else "PASS")))
        return int(failed)
    monkeypatch.setattr(master, "run", fake_run)
    args = arguments(tmp_path, model=None, offset=0., command="advance",
        precision_results=evidence, campaign_ledger=tmp_path/"budget.json",
        consumer_cost_after_mechanics=True, consumer_timeout=7.)
    assert master.advance(args) == int(failed_phase is not None)
    assert ("sir_d18", "consumer-cost") in calls if failed_phase != "mechanics" else ("sir_d18", "consumer-cost") not in calls
    assert calls[-1] == ("ksc", "consumer-cost")
    report = json.loads((args.output/"advance.json").read_text())
    assert report["models"][-1]["status"] == "COMPUTATION_CHECKS_COMPLETE_POSTERIOR_NOT_RUN"
    if failed_phase == "consumer-cost":
        assert report["models"][0]["status"] == "CONSUMER_COST_REPAIR_REQUIRED"
    next_state=json.loads((args.output/"next-phase-plan.json").read_text())
    assert next_state["overall_status"] == report["status"]
    assert next_state["unresolved_models"] == (["sir_d18"] if failed_phase else [])


def test_consumer_continuation_flag_is_advance_only(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["master", "run", "--output", str(tmp_path/"new"),
                                     "--consumer-cost-after-mechanics"])
    with pytest.raises(SystemExit):
        master.main()


@pytest.mark.parametrize("extra",[["--consumer-baseline","missing.py"],
    ["--consumer-cost","--model","lgssm","--selection","selection.json","--consumer-baseline","missing.py"]])
def test_consumer_baseline_rejects_missing_context_or_source(tmp_path,monkeypatch,extra):
    monkeypatch.setattr(sys,"argv",["master","run","--output",str(tmp_path/"run"),*extra])
    with pytest.raises(SystemExit) as exc:
        master.main()
    assert exc.value.code==2
    assert not (tmp_path/"run").exists()


def test_consumer_baseline_propagates_to_worker(tmp_path,monkeypatch):
    commands=[]
    baseline=tmp_path/"baseline.py";baseline.write_text("preserved diagnostic source")
    def fake(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(master.subprocess,"run",fake)
    args=arguments(tmp_path,consumer_cost=True,selection=tmp_path/"selection.json",
                   consumer_baseline=baseline,offset=0)
    assert master.run(args)==0
    assert len(commands)==1
    command=commands[0]
    assert command[command.index("--consumer-baseline")+1]==str(baseline)
    assert command[command.index("--selection")+1]==str(args.selection)


@pytest.mark.parametrize("extra",[["--consumer-paired-only"],
    ["--consumer-paired-only","--consumer-cost","--model","lgssm","--selection","selection.json"]])
def test_paired_only_requires_consumer_and_baseline(tmp_path,monkeypatch,extra):
    monkeypatch.setattr(sys,"argv",["master","run","--output",str(tmp_path/"run"),*extra])
    with pytest.raises(SystemExit) as exc: master.main()
    assert exc.value.code==2


def test_paired_only_propagates_to_worker(tmp_path,monkeypatch):
    commands=[]
    baseline=tmp_path/"baseline.py";baseline.write_text("preserved diagnostic source")
    monkeypatch.setattr(master.subprocess,"run",lambda command,**kwargs:
                        (commands.append(command) or SimpleNamespace(returncode=0)))
    args=arguments(tmp_path,consumer_cost=True,selection=tmp_path/"selection.json",
                   consumer_baseline=baseline,consumer_paired_only=True,offset=0)
    assert master.run(args)==0
    assert "--consumer-paired-only" in commands[0]


def test_stage_arithmetic_is_forwarded_and_keeps_diagnostic_status(tmp_path, monkeypatch):
    commands=[]
    def fake(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(master.subprocess,"run",fake)
    args=arguments(tmp_path,stage_arithmetic=True,offset=0.)
    assert master.run(args)==0
    assert "--stage-arithmetic" in commands[0]
    state=json.loads((args.output/"next-phase-plan.json").read_text())
    assert state["status"]=="DIAGNOSTIC_COMPLETE"
    assert "identical-input" in state["next"]


def test_reset_arithmetic_forwarding_and_diagnostic_status(tmp_path,monkeypatch):
    commands=[]
    monkeypatch.setattr(master.subprocess,'run',lambda command,**kwargs:
                        (commands.append(command) or SimpleNamespace(returncode=0)))
    args=arguments(tmp_path,reset_arithmetic_inputs=tmp_path/'saved',offset=0.)
    assert master.run(args)==0
    command=commands[0]
    assert command[command.index('--reset-arithmetic-inputs')+1]==str(args.reset_arithmetic_inputs)
    state=json.loads((args.output/'next-phase-plan.json').read_text())
    assert state['status']=='DIAGNOSTIC_COMPLETE'
    assert 'reset arithmetic' in state['next']


def test_transport_arithmetic_forwarding_and_diagnostic_status(tmp_path,monkeypatch):
    commands=[]
    monkeypatch.setattr(master.subprocess,'run',lambda command,**kwargs:
                        (commands.append(command) or SimpleNamespace(returncode=0)))
    args=arguments(tmp_path,transport_arithmetic_inputs=tmp_path/'saved',offset=0.)
    assert master.run(args)==0
    command=commands[0]
    assert command[command.index('--transport-arithmetic-inputs')+1]==str(args.transport_arithmetic_inputs)
    state=json.loads((args.output/'next-phase-plan.json').read_text())
    assert state['status']=='DIAGNOSTIC_COMPLETE'
    assert 'transport candidates' in state['next']
