#!/usr/bin/env python3
"""Compare model definitions: P44 and frozen 3D are different targets.

Same-target P44/Kalman replication is tested in test_sqmc_campaign_repairs.py.
This diagnostic never subtracts scores in different parameter coordinates.
"""
import os,sys,json
from pathlib import Path
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH','true')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
if tf.config.list_physical_devices('GPU'):
    configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec

def main():
    models=[]
    for family in ('p44','frozen_3d'):
        spec=LGSSMSpec(family,3); theta=spec.default_theta()
        models.append(dict(target=spec.target_id,theta=theta.numpy().tolist(),
                           parameters={k:v.numpy().tolist() for k,v in spec.kalman_parameters(theta).items()}))
    print(json.dumps(dict(decision='different_targets; not a replication',models=models,
                          same_target_test='tests/highdim/test_sqmc_campaign_repairs.py'),indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
