"""Independent native CmdStan versus TF/XLA density and gradient reference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
import benchmark_nuts_xla as base
from benchmark_stan_transplant import inputs,density_check
from nuts_benchmark_targets import make_target,NAMES

def test_native_target_density_and_gradient(tmp_path):
    exe=ROOT/'docs/benchmarks/stan_targets'
    if not exe.exists():pytest.skip('isolated native Stan build is required')
    conf=base.parser().parse_args(['--device','cpu','--output',str(tmp_path)])
    tf,_,_=base.setup(conf)
    for name in NAMES:
        target=make_target(tf,name);q,data=inputs(tf,target,20261020)
        folder=tmp_path/name;folder.mkdir()
        report=density_check(tf,target,q,data,SimpleNamespace(stan=exe),folder)
        assert report['value_relative_error']<1e-10
        assert report['gradient_relative_error']<1e-9
