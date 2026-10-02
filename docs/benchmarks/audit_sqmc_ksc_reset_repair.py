#!/usr/bin/env python3
"""Read-only evidence audit with a new aggregate manifest; no framework import."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CAMPAIGN=ROOT/'docs/plans/artifacts/sqmc-ksc-reset-repair-20260929'
BASE=CAMPAIGN/'attempt-01'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def run(output=None):
    budget=read(CAMPAIGN/'budget.json');checks=[];units=[]
    def check(name,ok): checks.append(dict(check=name,pass_check=bool(ok)))
    check('prior ledger unchanged',sha(ROOT/budget['prior_ledger'])==budget['prior_sha256'])
    charge=budget['prior_charged_seconds']+sum(a['wall_seconds'] for a in budget['attempts'])
    check('charge reconciliation',abs(charge-budget['charged_seconds'])<1e-7)
    check('aggregate compute cap',charge<=budget['cap_seconds']==43200.)
    check('remaining allowance',abs(43200.-charge-budget['remaining_gpu_seconds'])<1e-7)
    data_hash=sha(BASE/'data.json');selections={}
    for p in BASE.glob('calibration*-attempt-*/result.json'):
        r=read(p)
        if r['status']=='finished':selections[r['unit']]=r
    seen={};total_rows=0
    for a in budget['attempts']:
        name=a['unit'];seen[name]=seen.get(name,0)+1
        folder=ROOT/a['output'];r=read(folder/'result.json');m=read(folder/'manifest.json')
        check(name+' complete',a['status']==r['status']==m['status']=='finished' and a['returncode']==0)
        check(name+' before deadline',dt.datetime.fromisoformat(a['finished_utc'])<=dt.datetime.fromisoformat(budget['deadline_utc']))
        check(name+' full lifetime charged',a['wall_seconds']>=m['wall_seconds'])
        check(name+' GPU/XLA/growth',m['jit_compile'] and not m['tf32'] and
              m['gpu_memory_policy']['all_physical_devices_memory_growth'] and
              'GPU:' in m['framework_gpu_probe']['device'])
        check(name+' particle/chunk policy',m['n']==m['chunk']==1008)
        source=read(folder/'source-sha256.json')
        check(name+' snapshot hashes',source['files']==m['source_sha256'] and
              all(sha(ROOT/source['snapshot']/p)==h for p,h in source['files'].items()))
        rows=[x for x in r['rows'] if 'arm' in x];total_rows+=len(rows)
        check(name+' finite valid evaluations',all(x['valid'] and math.isfinite(x['value']) and
              len(x['score'])==2 and all(math.isfinite(v) for v in x['score']) for x in rows))
        if name!='reference':check(name+' data identity',m['data_sha256']==data_hash)
        phase=name.split('__')[0]
        if phase in ('validation','evaluation','extension'):
            _,route,horizon=name.split('__');cal=selections[f'calibration__{route}__{horizon}']
            selected=cal['selection']['arm']
            check(name+' frozen before evaluation',dt.datetime.fromisoformat(cal['selection']['frozen_utc'])<dt.datetime.fromisoformat(m['started_utc']) and r['frozen_selection']==selected)
            controls={x['arm']:(x['controls'],x['reset_design']) for x in cal['rows']}
            check(name+' frozen controls',all((x['controls'],x['reset_design'])==controls[x['arm']] for x in rows))
            allowed_data={242001} if phase=='validation' else {243001,243002}
            allowed_design=({252001,252002} if phase=='validation' else
                            set(range(253001,253005)) if phase=='evaluation' else set(range(253005,253009)))
            check(name+' prescribed partitions',set(x['data_seed'] for x in rows)==allowed_data and
                  set(x['design_seed'] for x in rows)==allowed_design)
        units.append(dict(unit=name,command=a['command'],result_json=str((folder/'result.json').relative_to(ROOT)),
          manifest=str((folder/'manifest.json').relative_to(ROOT)),source_snapshot=source['snapshot'],
          data_seeds=sorted({x['data_seed'] for x in r['rows']}),
          design_seeds=sorted({x['design_seed'] for x in rows}),
          wall_seconds=a['wall_seconds'],git_commit=m['git_commit'],environment=m['environment']))
    check('at most two infrastructure retries',all(v<=3 for v in seen.values()))
    expected={'reference':1,'checks':4,'stage':1,'calibration':8,'validation':8,'evaluation':8,'extension':8}
    counts={phase:sum(u.split('__')[0]==phase for u in seen) for phase in expected}
    check('full balanced campaign completed',counts==expected)
    check('expected valid numerical evaluation count',total_rows==525)
    summary=read(BASE/'analysis-01/summary.json')
    check('report consistency checks',summary['all_evidence_checks_pass'] and summary['evaluations']==total_rows)
    result=dict(schema='sqmc_ksc_reset_repair_terminal_audit_v1',cpu_only=True,gpu_intentionally_unused=True,
      plan='docs/plans/sqmc-ksc-reset-repair-plan-20260929.md',
      result='docs/benchmarks/sqmc-ksc-reset-repair-results-20260929.md',
      data_sha256=data_hash,budget=budget,units=units,checks=checks,
      all_pass=all(c['pass_check'] for c in checks),
      limitation='Engineering/evidence checks do not establish score accuracy or promotion.')
    out=Path(output) if output is not None else BASE/'analysis-01/terminal-audit.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(all_pass=result['all_pass'],checks=len(checks),units=len(units),
      evaluations=total_rows,failed=[c['check'] for c in checks if not c['pass_check']],output=str(out))))
    if not result['all_pass']:raise SystemExit(1)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Fresh audit path; existing files are never overwritten')
    run(parser.parse_args().output)
