"""Standard-library campaign accounting and externally bounded launches."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import run_sqmc_expanded_comparison as campaign

BASE=campaign.ROOT/'docs/plans/artifacts/sqmc-expanded-20260928/renewal-01'
LEDGER=BASE/'budget.json'


def remaining(ledger):
    charged=ledger['prior_gpu_seconds_charged']+sum(r.get('wall_seconds',0.)
        for r in ledger['launches'] if r['resource']=='gpu')
    return charged,max(0.,43200-charged)


def run(action):
    if action.startswith('repair-'):
        import run_sqmc_expanded_repair as repair
        repair.configure_profile()
    BASE.mkdir(parents=True,exist_ok=True)
    if LEDGER.exists():
        ledger=json.loads(LEDGER.read_text())
    else:
        if action!='cpu':
            raise ValueError('start the renewed numerical window with the focused CPU checks')
        start=datetime.now(timezone.utc)
        ledger=dict(schema='sqmc_expanded_budget.v1',first_numerical_check_utc=start.isoformat(),
            elapsed_deadline_utc=(start+timedelta(hours=14)).isoformat(),
            total_gpu_seconds=43200,prior_gpu_seconds_charged=1918,
            prior_accounting='Entire interval from pilot 01 start to pilot 04 finish, rounded up; failed pilot end times unavailable. Conservative upper-bound charge, not measured device utilization.',
            prior_interval_utc=['2026-09-26T17:49:50.685987+00:00','2026-09-26T18:21:47.759585+00:00'],
            launches=[])
        campaign.dump(LEDGER,ledger)
    if any(r.get('status')=='running' for r in ledger['launches']):
        raise RuntimeError('existing launch has unresolved accounting; reconcile before continuing')
    elapsed_left=datetime.fromisoformat(ledger['elapsed_deadline_utc']).timestamp()-time.time()
    if elapsed_left<=0:
        raise RuntimeError('renewed elapsed budget exhausted')
    count=1+sum(r['action']==action for r in ledger['launches'])
    infrastructure_failures=sum(r['action']==action and r.get('return_code',0)!=0
        and r.get('failure_class') not in ('numerical_candidate_rejected','planned_numerical_repair_stop')
        for r in ledger['launches'])
    if action in ('cpu','gpu-check','repair-check') and infrastructure_failures>=3:
        raise RuntimeError('check infrastructure retry limit')
    charged,left=remaining(ledger)
    label=f'{action}-{count:02d}'
    output=BASE/label
    env=os.environ.copy()
    if action=='cpu':
        env.update(CUDA_VISIBLE_DEVICES='-1',TF_CPP_MIN_LOG_LEVEL='2',
                   TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2')
        command=[sys.executable,'-m','pytest','-q',
            'tests/highdim/test_sqmc_full_lgssm.py',
            'tests/highdim/test_sqmc_expanded_execution.py',
            'tests/highdim/test_sqmc_campaign_repairs.py',
            'tests/highdim/test_sqmc_expanded_control.py']
        resource,limit='cpu',180
    elif action in ('gpu-check','repair-check'):
        check_script='check_sqmc_expanded_repair_gpu.py' if action=='repair-check' else 'check_sqmc_expanded_gpu.py'
        command=[sys.executable,str(campaign.ROOT/'docs/benchmarks'/check_script),
                 '--output',str(output)]
        resource,limit='gpu',min(1200,left)
    else:
        check_action='repair-check' if action=='repair-run' else 'gpu-check'
        checks=[r for r in ledger['launches'] if r['action']==check_action and r.get('return_code')==0]
        if not checks:
            raise RuntimeError('bounded GPU check must pass before full execution')
        check=json.loads((Path(checks[-1]['output'])/'result.json').read_text())
        if check['status']!='passed':
            raise RuntimeError('bounded GPU check did not pass')
        prior_paths=[r['output'] for r in ledger['launches'] if r['action']==action and (Path(r['output'])/'manifest.json').exists()]
        reused,_=campaign.reusable_units(prior_paths,campaign.source_hashes())
        projected=1.5*sum(r['estimated_seconds']/4 for r in check['projection']['scope_estimates'] for route in campaign.ROUTES if r['scope']+'__'+route not in reused)
        if projected>min(left,elapsed_left):
            ledger['continuation_status']='cost_projection_exceeds_remaining_budget'
            ledger['projection']=check['projection']
            campaign.dump(LEDGER,ledger)
            raise RuntimeError('complete ladder cost projection exceeds remaining budget')
        command=[sys.executable,str(campaign.ENTRY_POINT),
            '--output',str(output),'--budget-seconds',str(left),
            '--prior-gpu-seconds',str(charged),'--elapsed-deadline',ledger['elapsed_deadline_utc']]
        for previous in ledger['launches']:
            if previous['action']==action and (Path(previous['output'])/'manifest.json').exists():
                command+=['--reuse-from',previous['output']]
        resource,limit='gpu',min(left,elapsed_left)
    limit=min(limit,elapsed_left)-15
    if limit<=0:
        raise RuntimeError('no remaining execution budget')
    logpath=BASE/f'{label}.log'
    record=dict(action=action,resource=resource,status='running',started_utc=campaign.now(),
        command=command,output=str(output),log=str(logpath),timeout_seconds=limit,
        gpu_seconds_remaining_before=left,prior_gpu_seconds_charged=charged,
        cpu_gpu_status='GPUs intentionally hidden with CUDA_VISIBLE_DEVICES=-1' if action=='cpu' else 'escalated GPU; worker verifies growth and device')
    ledger['launches'].append(record)
    campaign.dump(LEDGER,ledger)
    tick=time.perf_counter()
    try:
        with logpath.open('x') as log:
            process=subprocess.Popen(command,cwd=campaign.ROOT,env=env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=process.pid
            campaign.dump(LEDGER,ledger)
            try:
                code=process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                code=process.returncode
                record['failure_class']='external_timeout'
        record.update(status='finished',return_code=code)
        if code and (output/'result.json').exists():
            result=json.loads((output/'result.json').read_text())
            if result.get('failure_class')=='numerical_candidate_rejected':
                record['failure_class']='numerical_candidate_rejected'
    except BaseException as error:
        record.update(status='launcher_failure',error=repr(error))
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-tick,finished_utc=campaign.now())
        ledger['gpu_seconds_charged'],ledger['gpu_seconds_remaining']=remaining(ledger)
        campaign.dump(LEDGER,ledger)
        print(json.dumps(record),flush=True)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['cpu','gpu-check','run','repair-check','repair-run'])
    sys.exit(run(parser.parse_args().action))
