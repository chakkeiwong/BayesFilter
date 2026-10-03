"""Execute only the frozen M21 independent confirmation inventory."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
assert os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')=='true'
inventory=json.loads((ROOT/'confirmation-inventory.json').read_text())
out=ROOT/'confirmation-cpu-r1';out.mkdir(exist_ok=False)
(out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
def run(case):
 cap=inventory['maximum_seconds_per_case']
 command=[sys.executable,str(ROOT/'run_controller_probe.py'),'--source',str(ROOT/'source-r1'),
  '--output',str(out/case),'--case',case,'--replications',str(inventory['replications_per_case']),
  '--seed',str(inventory['root_seed']),'--seconds',str(cap-10),'--device','cpu_reference']
 start=time.monotonic()
 with (out/(case+'.log')).open('x') as log:
  try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=cap).returncode
  except subprocess.TimeoutExpired:code=124
 row={'case':case,'command':command,'returncode':code,'elapsed_seconds':time.monotonic()-start,'device':'cpu_reference','timeout_seconds':cap,
  'inventory_sha256':hashlib.sha256((out/'inventory.json').read_bytes()).hexdigest()}
 (out/(case+'-attempt.json')).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
 return row
with ThreadPoolExecutor(max_workers=inventory['worker_limit']) as pool:
 rows=list(pool.map(run,inventory['cases']))
(out/'execution.json').write_text(json.dumps({'attempts':rows,'charged_worker_seconds':sum(r['elapsed_seconds'] for r in rows)},indent=2)+'\n')
