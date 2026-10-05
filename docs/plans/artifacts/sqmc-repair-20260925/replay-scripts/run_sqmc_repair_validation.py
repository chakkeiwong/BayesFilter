"""Bounded validation runner; preserves complete logs and command manifests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/home/chakwong/BayesFilter-SQMC')
out = root/'docs/plans/artifacts/sqmc-repair-20260925'
label, seconds, *command = sys.argv[1:]
path = out/label
path.mkdir(exist_ok=False)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', BAYESFILTER_TEST_DEVICE_SCOPE='cpu',
           TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2', OMP_NUM_THREADS='2',
           PYTHONPATH=str(root)+':/home/chakwong/python/src', TF_FORCE_GPU_ALLOW_GROWTH='true')
started = time.perf_counter()
with (path/'run.log').open('w') as log:
    try:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=float(seconds))
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
elapsed = time.perf_counter()-started
manifest = dict(command=command, cwd=str(root), git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','BAYESFILTER_TEST_DEVICE_SCOPE','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS','PYTHONPATH','TF_FORCE_GPU_ALLOW_GROWTH']},
                execution_role='CPU reference diagnostic; GPUs intentionally hidden', conda_env='tftwogpu', seeds='Declared in referenced test/probe',
                plan='docs/plans/sqmc-repair-master-program-20260925.md', wall_seconds=elapsed, exit_code=code, log=str(path/'run.log'))
(path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(exit_code=code,wall_seconds=elapsed,log=str(path/'run.log'))))
print('\n'.join((path/'run.log').read_text().splitlines()[-18:]))
