"""CPU-only tests distinguishing finite-difference defects from branch changes."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('ksc_discrepancy_runner',Path(__file__).resolve().parents[2]/'docs/benchmarks/run_sqmc_ksc_discrepancy.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

@pytest.mark.parametrize('branch,slopes,expected',[
    (False,(2.00001,2.000001),'pass'),
    (False,(3.,3.000001),'stable_derivative_mismatch'),
    (True,(3.,3.000001),'unresolved_branch_crossing_or_fd_instability'),
    (False,(2.,3.),'unresolved_branch_crossing_or_fd_instability'),
])
def test_fd_disposition_requires_stability_and_unchanged_ancestry(branch,slopes,expected):
    rows=[dict(step=h,centered=s,branch_changes=branch) for h,s in zip((1e-4,1e-5),slopes)]
    assert runner.fd_classify(2.,rows)['status']==expected


def test_one_stable_looking_step_cannot_establish_derivative_correctness():
    assert runner.fd_classify(2.,[dict(step=1e-5,centered=2.,branch_changes=False)])['status']=='unresolved_branch_crossing_or_fd_instability'
