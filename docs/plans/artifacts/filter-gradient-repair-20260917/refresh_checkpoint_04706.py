"""Refresh streaming-memory unit without resetting cumulative campaign charges."""
import json
from pathlib import Path

work = Path('/tmp/bayesfilter-filter-gradient-xla-validation-20260918/docs/plans')
raw = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
p = work/'filter_gradient_repair_ledger_20260917.json'
d=json.loads(p.read_text()); c=d['current_checkpoint']; a=c['ledh_streaming_memory_allocation']
base=a.setdefault('base_charges_through04687', c['charged_seconds'].copy())
rows=[(int(p.parent.name[4:]),json.loads(p.read_text())) for p in sorted(raw.glob('run-*/run.json')) if int(p.parent.name[4:])>4687]
used={device:sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in rows if r['device']==device) for device in base}
charges={device:base[device]+used[device] for device in base}
active=[n for n,r in rows if r['state']=='running']
through=max([4687]+[n for n,r in rows if r['state']!='running'])
a.update(used_workers=len(rows),used_seconds=used,failed_runs=[n for n,r in rows if r['state']=='failed'])
c.update(through_run=through, charged_seconds=charges, remaining_hours={k:cap-charges[k]/3600 for k,cap in {'CPU':56,'GPU':52}.items()},active_job=active or None,active_session=None)
notes=a.get('progress_notes','Baseline04688 CPU and04689 GPU each pass3 freeze checks; streaming CPU04690 passes9 checks. GPU qualification and costs follow. No scientific admission or full-program completion.')
next_action=a.get('next_action','Finish CPU/GPU streaming qualification, then run isolated owner-memory/cost arms and the predeclared GPU ladder. Preserve strong invalidity diagnostics and exact original RNG scheduling.')
c['status']=notes;c['next']=next_action
p.write_text(json.dumps(d,indent=2)+'\n')
text=f'''Branch: repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Last pushed checkpoint
c7c0b88c2; streaming work is uncommitted. Main remains unmerged.

Active question: remove O(T*N*d) seeded process-noise storage and qualify
reusable versus fresh-owner memory. Active plan:
`filter_gradient_ledh_streaming_memory_20260929.md`.
Through {through:05d}; active workers: {active or 'none'}.
Charged/reserved CPU {charges['CPU']:.6f}s / GPU {charges['GPU']:.6f}s.
Remaining CPU {c['remaining_hours']['CPU']:.6f}h / GPU {c['remaining_hours']['GPU']:.6f}h.
Global caps56 CPU/52 GPU process-hours include the extra24 CPU hours.
Streaming allocation24 workers/3600 CPU/2400 GPU seconds; used/reserved
{len(rows)} workers/{used['CPU']:.6f} CPU/{used['GPU']:.6f} GPU seconds.
One numerical worker at a time. CPU is explicit reference only. GPU requires
trusted non-display availability and verified memory growth. Timing requires
unshared device preflight. Do not stop other campaigns.

{notes}

Next: {next_action}

Preserve prior qualified evidence: seeded public/LM result through04687,
validity boundary04662--04668, actual DZ5 renewal04618--04628/import isolation
04629--04630/evidence index04631. Older current-source readbacks retain their
original source scope; do not silently relax them after runtime changes.

Other gates: F14 unapproved optional pfor, registered analytical-score migration
and costs, current-source timing/memory applicability, DZ5 locator121 strict
trajectory differences and unconverged optimizers,4539 strict fitted-geometry
CPU/GPU record differences/isotropic angles, and all F01--F20 dispositions.
Locator optimized-HLO invariant-copy lead is not a qualified causal repair.
See `filter_gradient_terminal_gap_queue_20260928.md` for the work order.

Keep callback configuration fixed per retained owner; one-shot convenience must
refresh mutable closures. No global callback cache or system/cache mutation.
LEDH SeedSequence/PCG64/Philox streams remain unchanged; geometry's approved
TensorFlow stream migration does not apply to them. No copied numerical kernels.
Preserve strong reset rejection: unusable public value is NaN with validity,
code/index and separate raw diagnostic outputs. No tolerance relaxation.
No live MacroFinance edits, subagents, training, HMC, package/environment change
or canonical LEDH rebuild. Preserve author-profile NeuTra IAF. Unsupported
scientific/default claims remain blocked. No main merge until every master gate.
'''
(work/'filter_gradient_repair_resume_20260919.md').write_text('# Filter and gradient repair resume checkpoint\n\n'+text)
p=work/'filter_gradient_repair_master_20260917.md'; old=p.read_text();p.write_text(old.split('\n',1)[0]+'\n\n'+text+'\n'+old[old.index('Older checkpoints below'):])
print(json.dumps({'through':through,'active':active,'used':used,'remaining_hours':c['remaining_hours']}))
