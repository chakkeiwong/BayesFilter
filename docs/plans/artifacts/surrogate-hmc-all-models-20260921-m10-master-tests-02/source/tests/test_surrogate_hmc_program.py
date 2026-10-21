"""Scientific consumer checks and bounded orchestration regression tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from bayesfilter.inference.surrogate_hmc_program import (
    CPU_CAP, GPU_CAP, ROOT, lg1_scope, model_register, reserve_job, run_job,
    validated_budget,
    charge_previous_runs,
    retained_forecast,
    validate_retained_forecast,
)


@pytest.mark.parametrize("gpu,cpu", [(float("nan"), 1), (1, float("inf")), (-1, 2), (GPU_CAP+1, 2), (1, CPU_CAP+1)])
def test_invalid_or_expanded_budget_rejected(gpu,cpu):
    with pytest.raises(ValueError):
        validated_budget(gpu,cpu)


def test_budget_reservation_prevents_launch():
    budget=validated_budget(5,20)
    reserve_job(budget,"gpu",5)
    budget["charged"]["gpu_seconds"]=.1
    with pytest.raises(ValueError,match="insufficient"):
        reserve_job(budget,"gpu",5)


def test_measured_cpu_allowance_has_its_own_watchdog(tmp_path):
    ledger = validated_budget(10, 1)
    reserve_job(ledger, "cpu", 2, cpu_allowance=.05)
    result = run_job([sys.executable, "-c", "while True: pass"], tmp_path/"cpu",
                     ledger, device="cpu", timeout=2, cpu_allowance=.05)
    assert result["status"] == "JOB_CPU_BUDGET_LIMIT"
    assert result["cpu_worker_seconds"] > 0
    with pytest.raises(ValueError, match="CPU allowance"):
        reserve_job(ledger, "cpu", 2, cpu_allowance=float("nan"))


def test_retained_launch_requires_verified_B_but_A_is_optional(tmp_path):
    report = {"arms": {"A": {"passed": True, "wall_seconds": 25},
                       "B": {"passed": True, "wall_seconds": 1100}}}
    path = tmp_path/"report.json"
    path.write_text(json.dumps(report))
    forecast = retained_forecast(tmp_path)
    assert forecast["gpu_seconds"] == pytest.approx(3332.)
    assert forecast["cpu_worker_seconds"] == pytest.approx(3998.4)
    assert forecast["retained_per_chain_max"] == 2000
    report["arms"]["A"] = {"passed": False}
    path.write_text(json.dumps(report))
    assert retained_forecast(tmp_path)["eligible_arms"] == ["B"]
    report["arms"]["B"]["passed"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="verified B"):
        retained_forecast(tmp_path)
    report["arms"]["B"] = {"passed": True, "wall_seconds": float("nan")}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="invalid scoped tuning cost"):
        retained_forecast(tmp_path)


def test_retained_forecast_limits_are_checked_before_dry_run_or_execution():
    with pytest.raises(ValueError, match="exceeds"):
        validate_retained_forecast({"gpu_seconds": 200, "cpu_worker_seconds": 250},
                                  {"timeout": 190, "cpu_allowance": 500})
    with pytest.raises(ValueError, match="exceeds"):
        validate_retained_forecast({"gpu_seconds": 200, "cpu_worker_seconds": 250},
                                  {"timeout": 300, "cpu_allowance": 240})


def test_finite_target_requires_its_own_tuning_and_forecast(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"arms": {"B": {"passed": True, "wall_seconds": 1.}}}))
    with pytest.raises(ValueError, match="verified C"):
        retained_forecast(tmp_path, "C")
    path.write_text(json.dumps({"arms": {"C": {"passed": True, "wall_seconds": 1000.}}}))
    forecast = retained_forecast(tmp_path, "C")
    assert forecast["eligible_arms"] == ["C"]
    assert forecast["gpu_seconds"] == pytest.approx(2776.)
    for stage in ("P2-finite-reference", "P2-C-tune", "P2-C-sample"):
        scope = lg1_scope(stage)
        assert scope["data_partition"] == "claim"
        assert scope["endpoint"].startswith("lg1_dctr_physical_fp32_chart_fp64_v1")


def test_C_dry_run_wires_frozen_references_and_own_tuning(tmp_path, capsys, monkeypatch):
    import bayesfilter.inference.surrogate_hmc_program as program
    monkeypatch.setattr(program, "ROOT", tmp_path)
    tuning = tmp_path / "tuning"
    tuning.mkdir()
    (tuning / "report.json").write_text(json.dumps({"arms": {"C": {"passed": True, "wall_seconds": 1000.}}}))
    args = ["--stage", "P2-C-sample", "--scope", "LG1", "--output", str(tmp_path / "output"),
            "--reference-root", str(tmp_path / "R"), "--selection", str(tmp_path / "selection.json"),
            "--endpoint-root", str(tmp_path / "endpoint"), "--finite-reference-root", str(tmp_path / "D"),
            "--tuning-root", str(tuning), "--dry-run"]
    assert program.main(args) == 0
    launch = json.loads(capsys.readouterr().out)
    command = launch["jobs"][0]["command"]
    assert command[command.index("--stage") + 1] == "sample"
    assert command[command.index("--tuning-root") + 1] == str(tuning)
    assert command[command.index("--finite-reference-root") + 1] == str(tmp_path / "D")
    assert launch["inference_forecast"]["eligible_arms"] == ["C"]
    assert not (tmp_path / "output").exists()


def test_posterior_scope_distinguishes_attempt_from_replicate_and_mechanics():
    scope = lg1_scope("P2-sample")
    assert scope["stage"] == "P2-sample" and scope["mechanics_only"] is False
    assert scope["inference_attempt"] == 4 and scope["data_replicate"] == 0
    assert scope["data_partition"] == "claim"


def test_cumulative_budget_uses_incremental_charges_once(tmp_path):
    first = tmp_path / "first.json"
    first.write_text(json.dumps({"charged": {"gpu_seconds": 10., "cpu_worker_seconds": 20.}}))
    second = tmp_path / "second.json"
    second.write_text(json.dumps({"charged": {"gpu_seconds": 15., "cpu_worker_seconds": 27.},
                                  "charged_this_run": {"gpu_seconds": 5., "cpu_worker_seconds": 7.}}))
    ledger = validated_budget(100, 100)
    charge_previous_runs(ledger, [first, second, first])
    assert ledger["charged"] == {"gpu_seconds": 15., "cpu_worker_seconds": 27.}
    incremental_only = tmp_path / "incremental.json"
    incremental_only.write_text(json.dumps({"charged_this_run": {"gpu_seconds": 1., "cpu_worker_seconds": 2.}}))
    fresh = validated_budget(100, 100)
    charge_previous_runs(fresh, [incremental_only])
    assert fresh["charged"] == {"gpu_seconds": 1., "cpu_worker_seconds": 2.}
    with pytest.raises(ValueError, match="exceeds"):
        charge_previous_runs(validated_budget(14, 100), [first, second])
    second.write_text(json.dumps({"charged": {"gpu_seconds": -1, "cpu_worker_seconds": 0}}))
    with pytest.raises(ValueError, match="invalid prior"):
        charge_previous_runs(validated_budget(100, 100), [second])


def test_launch_error_preserves_terminal_manifest_and_charge(tmp_path):
    ledger = validated_budget(10, 30)
    with pytest.raises(FileNotFoundError):
        run_job([str(tmp_path / "missing-executable")], tmp_path / "job", ledger,
                device="cpu", timeout=1)
    manifest = json.loads((tmp_path / "job/run-manifest.json").read_text())
    assert manifest["status"] == "LAUNCH_ERROR"
    assert manifest["exit_code"] is None
    assert len(ledger["jobs"]) == 1
    assert manifest["wall_seconds"] >= 0


def test_failed_and_timed_out_work_is_charged_and_preserved(tmp_path):
    ledger=validated_budget(10,30)
    failure=run_job([sys.executable,"-c","print('preserved'); raise SystemExit(7)"],tmp_path/"failure",ledger,device="cpu",timeout=2)
    assert failure["exit_code"]==7 and failure["status"]=="FAILED"
    assert "preserved" in (tmp_path/"failure/command.log").read_text()
    timeout=run_job([sys.executable,"-c","import time; time.sleep(10)"],tmp_path/"timeout",ledger,device="cpu",timeout=.3)
    assert timeout["status"]=="WALL_TIME_LIMIT"
    assert ledger["charged"]["cpu_worker_seconds"]>0
    assert ledger["charged"]["gpu_seconds"]==0
    with pytest.raises(FileExistsError):
        run_job([sys.executable,"-c","pass"],tmp_path/"failure",ledger,device="cpu",timeout=1)


def test_dry_run_has_no_framework_import_or_files(tmp_path):
    output=tmp_path/"uncreated"
    code="""import sys, runpy
sys.argv=['scripts/run_surrogate_hmc_program.py','--output',sys.argv[1],'--dry-run']
try:
    runpy.run_path('scripts/run_surrogate_hmc_program.py',run_name='__main__')
except SystemExit as exc:
    assert exc.code==0
assert 'tensorflow' not in sys.modules
assert 'numpy' not in sys.modules
"""
    result=subprocess.run([sys.executable,"-c",code,str(output)],cwd=ROOT,text=True,capture_output=True,check=True)
    payload=json.loads(result.stdout)
    assert payload["inference_ready"] is False
    assert not output.exists()
    assert set(payload["scope_cards"]) == {"LG1", "LG2", "C01", "SV20", "M10"}
    assert all(payload["scope_cards"][name]["comparisons"]["B_vs_R"] == "NOT_RUN" for name in ("LG1", "LG2"))
    assert payload["scope_cards"]["C01"]["role"] == "mechanism_control"


def test_all_minimum_families_remain_visible():
    register=model_register()
    assert len(register["families"])==17
    assert not register["all_models_validated"]
    assert all(row["inference_status"]=="NOT_RUN" for row in register["families"])
    sv = next(row for row in register['families'] if row['id'] == 'M02')
    assert sv['variant_status']['SV20'] == 'REFERENCE_CONSUMER_CALIBRATION_AVAILABLE'
    assert 'affordable complete inference forecast' in sv['missing']
    assert lg1_scope()["parameter_dimension"]==1


def test_reference_stage_is_cpu_only_and_cannot_launch_sampling(tmp_path):
    command = [sys.executable, "scripts/run_surrogate_hmc_program.py", "--stage", "P2-reference",
               "--scope", "LG1", "--output", str(tmp_path / "uncreated"), "--dry-run"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert len(payload["jobs"]) == 1
    assert payload["jobs"][0]["device"] == "cpu"
    assert payload["jobs"][0]["timeout"] <= 300
    assert payload["inference_ready"] is False
    assert not (tmp_path / "uncreated").exists()


def test_stationary_kalman_matches_dense_observation_law_and_total_gradient():
    import tensorflow as tf
    from bayesfilter.testing.surrogate_hmc_lg1_tf import kalman_physical
    y=tf.constant([[.3],[-.7],[1.2],[.5]],tf.float64)
    phi=tf.constant([-.8,.1,.8],tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(phi)
        indices=tf.range(4)
        distance=tf.abs(indices[:,None]-indices[None,:])
        covariance=tf.pow(phi[:,None,None],tf.cast(distance[None],tf.float64))/(1-phi[:,None,None]**2)+.5*tf.eye(4,dtype=tf.float64)[None]
        chol=tf.linalg.cholesky(covariance)
        residual=tf.broadcast_to(y[None],[3,4,1])
        solved=tf.linalg.triangular_solve(chol,residual)
        dense=-.5*(4*tf.math.log(tf.constant(2*3.141592653589793,tf.float64))+2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)),axis=-1)+tf.reduce_sum(solved**2,axis=[1,2]))
    derivative=tape.gradient(dense,phi)
    value,score,_=kalman_physical(phi,y)
    tf.debugging.assert_near(value,dense,atol=1e-12,rtol=1e-12)
    tf.debugging.assert_near(score,derivative,atol=1e-11,rtol=1e-11)


def test_bounded_prior_chart_and_predictive_probability_are_same_law():
    import tensorflow as tf
    from bayesfilter.testing.surrogate_hmc_lg1_tf import kalman_physical,kalman_unconstrained
    y=tf.constant([[.3],[-.7],[1.2],[.5]],tf.float64)
    z=tf.constant([[-2.],[0.],[2.]],tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(z)
        logp,gradient,predictive=kalman_unconstrained(z,y)
    tf.debugging.assert_near(gradient,tape.gradient(logp,z),atol=1e-11,rtol=1e-11)
    sig=tf.math.sigmoid(z[:,0]);phi=-.95+1.9*sig
    likelihood,_,_=kalman_physical(phi,y)
    explicit=likelihood-tf.math.log(tf.constant(1.9,tf.float64))+tf.math.log(1.9*sig*(1-sig))
    tf.debugging.assert_near(logp,explicit,atol=1e-12,rtol=1e-12)
    distance=tf.abs(tf.range(4)[:,None]-tf.range(4)[None,:])
    cov=tf.pow(phi[:,None,None],tf.cast(distance[None],tf.float64))/(1-phi[:,None,None]**2)+.5*tf.eye(4,dtype=tf.float64)[None]
    cross=tf.pow(phi[:,None],tf.cast(tf.range(4,0,-1)[None],tf.float64))/(1-phi[:,None]**2)
    solved=tf.linalg.solve(cov,tf.broadcast_to(y[None],[3,4,1]))[:,:,0]
    mean=tf.reduce_sum(cross*solved,axis=1)
    variance=1/(1-phi**2)+.5-tf.reduce_sum(cross*tf.linalg.solve(cov,cross[:,:,None])[:,:,0],axis=1)
    exact=.5*tf.math.erfc(-mean/tf.sqrt(2*variance))
    tf.debugging.assert_near(predictive,exact,atol=1e-12,rtol=1e-12)


def test_consumer_resolves_to_complete_canonical_callable(monkeypatch):
    import tensorflow as tf
    import bayesfilter.highdim.ledh_program_fixtures_tf as fixtures
    from bayesfilter.testing.surrogate_hmc_lg1_tf import make_diagnostic_dataset,make_lg1_consumer,endpoint_status
    calls=[]
    def canonical(model,theta,directions,initial,covs,noises,obs,**controls):
        calls.append(controls)
        batch=tf.shape(theta)[0]
        value=-tf.reduce_sum(theta**2,axis=-1)
        status=endpoint_status(value)
        status.update(program_valid=status['value_valid'],direction_value_valid=status['value_valid'][:,None],
                      direction_force_valid=status['force_valid'][:,None])
        return value,-2*theta,status
    monkeypatch.setattr(fixtures,'canonical_batch_fused_value_score',canonical)
    consumer=make_lg1_consumer(make_diagnostic_dataset(),batch_size=4,particle_count=16,jit_compile=False)
    z=tf.constant([[-1.],[0.],[1.],[2.]],tf.float64)
    before=consumer.endpoint.function(z).potential
    field=consumer.force.function(z)
    assert len(calls)==1
    assert calls[0]['reset_policy']=='contract_e'
    assert calls[0]['correction_steps']>0 and calls[0]['pairwise_steps']>0
    assert calls[0]['coordinate_cap']>0 and calls[0]['reset_design'].shape==(16,1)
    assert field.force.dtype==tf.float64
    assert consumer.force.identity!=consumer.endpoint.identity
    assert consumer.force.semantics=='deterministic_position_only_proposal_field'
    tf.debugging.assert_equal(before,consumer.endpoint.function(z).potential)
