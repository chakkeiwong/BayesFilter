"""Actual master endpoints and failure-closed reference/model dispatch."""
import os
from pathlib import Path
import subprocess
import sys


def test_actual_nonlinear_registry_consumers_and_reference_veto():
    code='''
from bayesfilter.score_study.registry import default_registry
from bayesfilter.score_study.nonlinear_adapter import evaluate_nonlinear
from bayesfilter.score_study.contracts import validate_result
from tests.highdim.test_younis_score_master_provider_consumers_tf import fixture
row,context=fixture("kdm_covariance")
row.update(model="nonlinear_scalar",estimator="nonlinear_analytical",sgqf_level=2)
context["study"]["settings"].update(transition_curve=.2,observation_curve=.12,
    reference=dict(points=401,radius=9.,relative_tolerance=1e-7,tail_tolerance=1e-9))
versions=set()
for proposal in ("grid_reference","ekf","ukf","bootstrap","local_linear","ledh","sgqf","kdm_covariance"):
    current={**row,"proposal":proposal,"estimator":"nonlinear_reference" if proposal=="grid_reference" else "nonlinear_analytical"}
    result=evaluate_nonlinear(current,context)
    validate_result(result,current,default_registry())
    assert result["diagnostics"]["oracle_kind"]=="refined_numerical_grid_not_exact"
    versions.add(result["diagnostics"]["data_version"])
assert len(versions)==1
context["study"]["settings"]["reference"]["radius"]=.5
try:evaluate_nonlinear(row,context)
except ValueError as error:assert "reference refinement veto" in str(error)
else:raise AssertionError("truncated reference accepted")
'''
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1","BAYESFILTER_PRELOAD_CUSTOM_OP":"0",
         "TF_NUM_INTRAOP_THREADS":"1","TF_NUM_INTEROP_THREADS":"1","OMP_NUM_THREADS":"1"}
    result=subprocess.run([sys.executable,"-c",code],cwd=Path(__file__).resolve().parents[2],
        env=env,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
