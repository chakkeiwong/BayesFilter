"""Finite pilot coordinator; one thread per CPU worker, at most two workers."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
assert os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')=='true'
out=ROOT/'pilot-cpu-r1';out.mkdir(exist_ok=False)
def run(case):
 command=[sys.executable,str(ROOT/'run_controller_probe.py'),'--source',str(ROOT/'source-r1'),
   '--output',str(out/case),'--case',case,'--replications','2','--seed','2026092211',
   '--seconds','290','--device','cpu_reference']
 start=time.monotonic()
 with (out/(case+'.log')).open('x') as log:
  try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
  except subprocess.TimeoutExpired:code=124
 row={'case':case,'command':command,'returncode':code,'elapsed_seconds':time.monotonic()-start,'device':'cpu_reference','timeout_seconds':300}
 (out/(case+'-attempt.json')).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
 return row
with ThreadPoolExecutor(max_workers=2) as pool:
 rows=list(pool.map(run,('iid','correlated','slow_stationary','slow_shifted','dispersed','common_shift')))
(out/'execution.json').write_text(json.dumps({'attempts':rows,'charged_worker_seconds':sum(x['elapsed_seconds'] for x in rows)},indent=2)+'\n')
