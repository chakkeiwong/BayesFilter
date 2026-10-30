import os
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
import sys, json, subprocess, resource, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from bayesfilter.inference.surrogate_hmc_program import validated_budget,charge_previous_runs,run_job,write_json
parent=Path(__file__).resolve().parent
runroot=ROOT/'docs/plans/artifacts'/('surrogate-hmc-all-models-20260919-tuner-repair-'+sys.argv[1])
runroot.mkdir(exist_ok=False)
ledger=validated_budget(3716.533096578,3828.164398)
charge_previous_runs(ledger,ROOT.glob('docs/plans/artifacts/surrogate-hmc-all-models-*/accounting.json'))
prior=dict(ledger['charged'])
program=sys.argv[2] if len(sys.argv)>2 else 'scripts/qualify_hmc_tensorflow_verified.py'
command=[sys.executable,str(ROOT/program),'--output',str(runroot/'qualification')]
start=time.monotonic()
try:
    outcome=run_job(command,runroot/'job',ledger,device='gpu',timeout=600)
finally:
    ledger['charged_this_run']={k:ledger['charged'][k]-prior[k] for k in prior}
    write_json(runroot/'accounting.json',ledger)
    write_json(runroot/'run-manifest.json',dict(command=command,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.executable,plan='docs/plans/surrogate-hmc-tf-tuner-repair-2026-09-19.md',result='qualification/report.json',wall_seconds=time.monotonic()-start,cpu_only=False))
print(json.dumps({'status':outcome['status'],'exit_code':outcome['exit_code'],'charged':ledger['charged']}))
sys.exit(outcome['exit_code'] or (1 if outcome['status']!='COMPLETED' else 0))
