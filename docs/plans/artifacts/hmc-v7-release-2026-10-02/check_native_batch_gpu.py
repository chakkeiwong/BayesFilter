"""Trusted native batching parity with explicit device and memory verification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def worker(args):
    sys.path.insert(0,str(args.source));os.chdir(args.source)
    from scripts.run_hmc_v7_release_prices import check_source
    signature=check_source(args.source)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    from bayesfilter.inference.hmc_replicated_batch import ReplicatedTrialBatchRunner
    import pytest
    import tensorflow_probability as tfp
    from tensorflow_probability.python.internal import samplers
    from tensorflow_probability.python.mcmc import hmc, sample, metropolis_hastings
    from tensorflow_probability.python.mcmc.internal import leapfrog_integrator
    dependencies={str(Path(module.__file__)):hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                  for module in (samplers,hmc,sample,metropolis_hastings,leapfrog_integrator)}
    write(args.output/'worker-manifest.json',dict(source_manifest_sha256=signature,memory_policy=memory,
        tensorflow_version=tf.__version__,tfp_version=tfp.__version__,tfp_sources=dependencies,
        gpu_uuid=args.gpu,jit_compile=True,tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
        environment={key:os.environ[key] for key in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH',
            'BAYESFILTER_TEST_DEVICE_SCOPE','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')},
        scope='native parity only, no tuning or release evidence'))
    rows=[]
    original=ReplicatedTrialBatchRunner.run
    def observed(self, **kwargs):
        result=original(self,**kwargs)
        rows.append(dict(shape=result[0].shape.as_list(),device=result[0].device,
                         runtime=result[2],seeds=tf.convert_to_tensor(kwargs['seeds']).numpy().tolist()))
        write(args.output/'calls.json',rows)
        assert 'GPU:0' in result[0].device, result[0].device
        return result
    # Host observation only: the numerical callable and its inputs are unchanged.
    ReplicatedTrialBatchRunner.run=observed
    with tf.device('/GPU:0'):
        code=pytest.main(['-q','--disable-warnings','tests/test_hmc_replicated_batch.py',
                         '-k',args.selection,'--junitxml='+str(args.output/'results.xml')])
    write(args.output/'result.json',dict(exit_code=int(code),calls=len(rows),
        devices=sorted({row['device'] for row in rows}),
        allocator=tf.config.experimental.get_memory_info('GPU:0'),release_ready=False))
    return int(code)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--selection',default='not maximum_batch')
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,TF_FORCE_GPU_ALLOW_GROWTH='true',
        BAYESFILTER_TEST_DEVICE_SCOPE='visible',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    os.sched_setaffinity(0,{24,25,26,27})
    if args.worker:return worker(args)
    args.output.mkdir(parents=True,exist_ok=False)
    runner=args.output/'runner.py';runner.write_bytes(Path(__file__).read_bytes())
    command=[sys.executable,str(runner),*sys.argv[1:],'--worker'];started=time.monotonic()
    record=dict(command=command,environment=sys.executable,resource='gpu',
        started_utc=datetime.now(timezone.utc).isoformat(),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        git_commit=json.loads((args.source.parent/'assembly.json').read_text())['git_commit'],
        plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md')
    write(args.output/'manifest.json',record)
    with (args.output/'run.log').open('x') as log:
        try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
        except subprocess.TimeoutExpired:code=124
    record.update(exit_code=code,wall_seconds=time.monotonic()-started)
    write(args.output/'receipt.json',record);print(json.dumps(record));return code


if __name__=='__main__':raise SystemExit(main())
