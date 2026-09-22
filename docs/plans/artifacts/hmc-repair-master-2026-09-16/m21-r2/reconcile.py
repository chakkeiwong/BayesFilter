"""Reconcile measured completed work; retain live reservations separately."""
from pathlib import Path
import datetime,json
ROOT=Path(__file__).resolve().parents[1]
records=[]
def load(path): return json.loads((ROOT/path).read_text())
def charge(phase,path,seconds,note='measured worker wall time'):
    records.append(dict(phase=phase,path=path,cpu_seconds=seconds,gpu_seconds=0,note=note))
def attempt(phase,path,key='elapsed_seconds'):
    if (ROOT/path).exists(): charge(phase,path,load(path)[key])
def index(phase,path):
    if (ROOT/path).exists():
        v=load(path)
        for name,job in v['jobs'].items():
            for a in job.get('attempts',[]):
                if 'elapsed_seconds' in a:
                    charge(phase,path+'#'+name+':'+str(a.get('attempt',1)),a['elapsed_seconds'])
charge('M19','m19-r1/reconciliation-terminal.json',load('m19-r1/reconciliation-terminal.json')['budget_seconds']['used']['cpu_reference'])
for name in ['late-merge-tests-r1.json','late-merge-tests-r2.json']:
    attempt('M19','m19-r1/'+name,'wall_seconds')
attempt('M19','m19-r1/rank-failure-diagnosis.json')
charge('M20','m20-r1/reconciliation-cpu-tranche.json',load('m20-r1/reconciliation-cpu-tranche.json')['budget_seconds']['charged']['cpu_reference'])
for name in ['pilot-cpu-r1','confirmation-cpu-r1']:
    p='m21-r1/'+name+'/execution.json'
    charge('M21',p,load(p)['charged_worker_seconds'])
charge('M21','m21-r1/tests-r1.xml',30.,'conservative accounting allowance; original 11-test command lacked an external timer, pytest reported 1.40 seconds excluding startup')
index('M21','m21-r1/public-controls-cpu-r1/run_index.json')
for p in ['m22-r1/m21-report-r1-attempt.json','m22-r1/m21-estimator-diagnosis-r1-attempt.json','m22-r1/m21-report-tests-r1-attempt.json']:
    attempt('M21',p)
attempt('M22','m21-r1/mutation-tests-r1-attempt.json')
index('M22','m22-r1/null-pilot-cpu-r1/run_index.json')
index('M22','m22-r1/full-fit-pilot-cpu-r1/run_index.json')
for p in ['engineering-tests-r1-attempt.json','engineering-tests-r2-attempt.json','isolated-continuation-tests-r3.json']:
    attempt('M23','m23-r1/'+p)
index('M21','m21-r2/public-pilot-cpu-r1/run_index.json')
attempt('M21','m21-r2/estimator-confirmation-execution.json','cpu_worker_seconds')
for receipt in sorted((ROOT/'m24-r2').glob('tests-*/result.json')):
    attempt('M24',str(receipt.relative_to(ROOT)),'wall_seconds')
queue_path=ROOT/'m21-r2/confirmation-queue-progress.json'
reservations={}
if queue_path.exists():
    q=json.loads(queue_path.read_text())
    for name,task in q.get('tasks',{}).items():
        if task.get('cpu_worker_seconds') is not None:
            charge(task['phase'],'m21-r2/confirmation-queue-progress.json#'+name,task['cpu_worker_seconds'])
    for task in load('m21-r2/confirmation-queue.json')['tasks']:
        if q.get('tasks',{}).get(task['name'],{}).get('cpu_worker_seconds') is None:
            reservations[task['name']]=task['worker_seconds']
spent=sum(r['cpu_seconds'] for r in records)
phase_totals={p:sum(r['cpu_seconds'] for r in records if r['phase']==p) for p in ['M19','M20','M21','M22','M23','M24']}
result={'schema':'bayesfilter.hmc_campaign_reconciliation.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'terminal':False,'invalid_artifacts':[],'records':records,'phase_charges_cpu_seconds':phase_totals,'budget_seconds':{'opening':{'cpu_reference':172800,'gpu':86400},'charged':{'cpu_reference':spent,'gpu':0},'remaining':{'cpu_reference':172800-spent,'gpu':86400},'exceeded':spent>172800},'remaining_excludes_future_reservations':True,'uncompleted_queue_reservations_seconds':reservations,'non_numerical_git_document_build_waiting_not_charged_as_numerical_workers':True,'source':'completed per-worker records, never coordinator plus worker'}
assert not result['budget_seconds']['exceeded']
out=ROOT/'m21-r2/reconciliation-progress.json'
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out)
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
