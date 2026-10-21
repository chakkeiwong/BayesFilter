"""CPU-only diagnostic staging and corruption checks; no model admission."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
os.environ.setdefault('BAYESFILTER_PRELOAD_CUSTOM_OP','0')
from dataclasses import replace
import json
import pytest
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from bayesfilter.testing import surrogate_hmc_stage_arithmetic_diagnostic as stage
from bayesfilter.testing import surrogate_hmc_reset_arithmetic_diagnostic as reset


def test_saved_reset_decomposition_and_corruption(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'N',16)
    case=runtime.load_case('lgssm')
    case.controls=replace(case.controls,flow_steps=2,reset_steps=2,correction_steps=0)
    stage.run_diagnostic(case,tmp_path/'stage',horizon=1)
    actual=reset.run_diagnostic(case,tmp_path/'reset',tmp_path/'stage',observations=(1,))
    assert actual['status']=='RESET_ARITHMETIC_LOCALIZED'
    assert actual['cpu_only'] and not actual['posterior_complete']
    assert actual['trace_counts']=={'float32':[1]*6,'float64':[1]*6}
    row=actual['stages'][0]
    assert row['healthy']
    assert set(row['local'])=={'weighted_moments','transport','transported_moments','recolour','restore'}
    for field in ('value','tangent'):
        assert float(row['staged_vs_fused'][field]['maximum_scaled'])<1e-4
        assert row['full_reset'][field]['finite']
    path=tmp_path/'stage/stage-arithmetic.json'
    saved=json.loads(path.read_text())
    saved['stages'][0]['saved_reset_inputs']['child']['sha256']='wrong'
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='checksum'):
        reset.run_diagnostic(case,tmp_path/'corrupt',tmp_path/'stage',observations=(1,))
