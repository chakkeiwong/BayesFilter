from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
command=[sys.executable,str(ROOT/'run_estimator_confirmation.py')]
started=time.monotonic()
with (ROOT/'estimator-confirmation.log').open('x') as log:
    try:
        result=subprocess.run(command,cwd=ROOT/'source-r1',stdout=log,stderr=subprocess.STDOUT,timeout=900)
        code=result.returncode
    except subprocess.TimeoutExpired:
        code=124
record={'command':command,'returncode':code,'cpu_worker_seconds':time.monotonic()-started,'gpu_intentionally_hidden':True,'plan_file':'docs/plans/bayesfilter-hmc-merged-source-continuation-2026-09-22.md'}
(ROOT/'estimator-confirmation-execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
raise SystemExit(code)
