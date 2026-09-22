"""Respect the shared two-worker limit while queuing reviewed M21 controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[4]
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
assert os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')=='true'
waiting=time.monotonic()
while not (ROOT/'confirmation-cpu-r1/execution.json').is_file():
 if time.monotonic()-waiting>10800:raise TimeoutError('controller coordinator did not complete within its declared queue ceiling')
 time.sleep(5)
records=[]
def run(name,command,cwd,timeout):
 started=time.monotonic()
 with (ROOT/(name+'.log')).open('x') as log:
  try:code=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
  except subprocess.TimeoutExpired:code=124
 row={'name':name,'command':command,'cwd':str(cwd),'returncode':code,'elapsed_seconds':time.monotonic()-started,'timeout_seconds':timeout,'device':'cpu_reference'}
 (ROOT/(name+'-attempt.json')).write_text(json.dumps(row,indent=2)+'\n');records.append(row);print(json.dumps(row),flush=True)
 return code
code=run('mutation-tests-r1',[sys.executable,'-m','pytest','-q','tests/inference_validation/test_full_fit_mutations.py','tests/inference_validation/test_definitions.py','tests/inference_validation/test_controller_stopping.py',
 '--junitxml='+str(ROOT/'mutation-tests-r1.xml')],REPO,180)
if code:raise SystemExit(code)
code=run('public-controls-pilot-r1',[sys.executable,'-m','bayesfilter.testing.inference_validation','run',str(ROOT/'public-controls-pilot.json'),
 '--output',str(ROOT/'public-controls-cpu-r1'),'--max-workers','2'],ROOT/'source-r1',1500)
(ROOT/'queued-controls-execution.json').write_text(json.dumps({'steps':records,'waiting_not_numerical_work_seconds':time.monotonic()-waiting-sum(r['elapsed_seconds'] for r in records)},indent=2)+'\n')
raise SystemExit(code)
