"""Run the fixed M21/M22 inventory with two numerical workers at most."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
assert os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')=='true'
assert all(os.environ.get(k)=='1' for k in ('TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'))
queue_path=ROOT/'confirmation-queue.json';queue=json.loads(queue_path.read_text())
assert queue['max_workers']==2 and queue['numerical_worker_ceiling']==79800
assert json.loads((ROOT/'public-pilot-execution.json').read_text())['returncode']==0
assert json.loads((ROOT/'estimator-confirmation-execution.json').read_text())['returncode']==0
progress=ROOT/'confirmation-queue-progress.json'
if progress.exists(): raise ValueError('fresh coordinator launch required; preserve prior attempts')
state={'queue_sha256':hashlib.sha256(queue_path.read_bytes()).hexdigest(),'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'source_commit':queue['source_commit'],'tasks':{},'reserved_cpu_seconds':79800,'gpu_seconds':0,'max_numerical_workers':2,'scientific_gaps_closed':False}
lock=threading.Lock()
def save():
    temporary=progress.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(progress)
def reconcile():
    with (ROOT/'live-reconciliation.log').open('a') as log:
        subprocess.run([sys.executable,str(ROOT/'reconcile.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
def execute(task):
    output=Path(task['output']);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise ValueError('output root already exists: '+str(output))
    command=[sys.executable,'-m','bayesfilter.testing.inference_validation','run',task['suite'],'--output',task['output'],'--max-workers','1']
    start=time.monotonic()
    with lock:
        state['tasks'][task['name']]={'status':'running','phase':task['phase'],'command':command,'source':task['source'],'output':task['output'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};save()
    # The repository supervisor owns and enforces each numerical worker's
    # ceiling. The service also has a 48000-second outer wall-time bound.
    with (ROOT/(task['name']+'-coordinator.log')).open('x') as log:
        code=subprocess.run(command,cwd=task['source'],stdout=log,stderr=subprocess.STDOUT).returncode
    index_path=output/'run_index.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else {}
    charges=[a['elapsed_seconds'] for j in index.get('jobs',{}).values() for a in j.get('attempts',()) if 'elapsed_seconds' in a]
    with lock:
        state['tasks'][task['name']].update(status='complete' if code==0 else 'needs_repair',returncode=code,coordinator_elapsed_seconds=time.monotonic()-start,cpu_worker_seconds=sum(charges) if charges else None,worker_charges_replace_coordinator_time=True,index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None)
        save();reconcile()
    print(json.dumps({'task':task['name'],**state['tasks'][task['name']]}),flush=True)
save();reconcile()
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(execute,task) for task in queue['tasks']]
    for f in as_completed(futures): f.result()
state.update(status='execution_finished_review_required',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
save();reconcile()
program_path=ROOT.parent/'program-progress.json'
program=json.loads(program_path.read_text())
for phase in ('M21','M22'):
    program['phases'][phase].update(status='confirmation_finished_result_review_required',confirmation_queue=str(progress))
program['next_action']='Inspect all confirmation outcomes; reconcile failures and caps; repair affected mechanisms and refresh the next funded design. Command completion does not close scientific gaps.'
temporary=program_path.with_suffix('.tmp');temporary.write_text(json.dumps(program,indent=2)+'\n');temporary.replace(program_path)
