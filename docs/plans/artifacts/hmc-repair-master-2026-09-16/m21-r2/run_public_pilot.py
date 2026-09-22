from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
assert os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')=='true'
cmd=[sys.executable,'-m','bayesfilter.testing.inference_validation','run',str(ROOT/'public-pilot.json'),'--output',str(ROOT/'public-pilot-cpu-r1'),'--max-workers','2']
s=time.monotonic()
with (ROOT/'public-pilot-coordinator.log').open('x') as log:
 result=subprocess.run(cmd,cwd=ROOT/'source-r1',stdout=log,stderr=subprocess.STDOUT,timeout=750)
index=json.loads((ROOT/'public-pilot-cpu-r1/run_index.json').read_text())
record={'command':cmd,'returncode':result.returncode,'coordinator_elapsed_seconds':time.monotonic()-s,'cpu_worker_seconds':sum(a['elapsed_seconds'] for j in index['jobs'].values() for a in j.get('attempts',[])),'worker_charges_replace_coordinator_time':True,'plan_file':'docs/plans/bayesfilter-hmc-merged-source-continuation-2026-09-22.md','gpu_intentionally_hidden':True}
(ROOT/'public-pilot-execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
raise SystemExit(result.returncode)
