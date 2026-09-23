"""Reconcile M28 outer receipts once, including setup failures and document builds."""
import argparse
import datetime
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--terminal',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parent
opening_file=root.parent/'m27-r1/reconciliation-terminal.json'
opening=json.loads(opening_file.read_text())['remaining_budget_seconds']
rows=[]
for path in sorted(root.glob('*/execution.json')):
    value=json.loads(path.read_text())
    rows.append({'attempt':path.parent.name,'receipt':str(path),'cpu_reference':value['cpu_worker_seconds'],
                 'gpu':value['gpu_worker_seconds'],'exit_code':value['exit_code']})
guide=json.loads((root/'guide-build.json').read_text())
extra_cpu={'document_builds':guide['wall_seconds']+guide['failed_first_attempt']['wall_seconds']+guide['bibliography_repair']['wall_seconds'],
           'short_planning_rendering_and_inspection_conservative_allowance':30.}
manual=json.loads((root/'manual-cost-adjustments.json').read_text())
charged={k:sum(r[k] for r in rows) for k in ('cpu_reference','gpu')}
charged['cpu_reference']+=sum(extra_cpu.values());charged['gpu']+=manual['gpu_worker_seconds']
limits={'cpu_reference':1800.,'gpu':4800.}
assert all(charged[k]<=limits[k] for k in limits),charged
pending=[str(p.parent) for p in root.glob('*/manifest.json') if not (p.parent/'execution.json').exists()]
if args.terminal: assert not pending,pending
value={'phase':'M28','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'opening_ledger':str(opening_file),'opening_remaining_seconds':opening,'phase_budget_seconds':limits,
       'attempts':rows,'additional_cpu_seconds':extra_cpu,'additional_gpu_seconds':manual,
       'charged_seconds':charged,'remaining_phase_seconds':{k:limits[k]-charged[k] for k in limits},
       'remaining_budget_seconds':{k:opening[k]-charged[k] for k in limits},'pending_attempt_receipts':pending,
       'all_phase_workers_terminal':args.terminal and not pending,
       'double_count_rule':'Outer execution receipts only; nested sequential children already included.',
       'failures_included':True}
(root/('reconciliation-terminal.json' if args.terminal else 'reconciliation-live.json')).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:value[k] for k in ('charged_seconds','remaining_budget_seconds','pending_attempt_receipts')}))
