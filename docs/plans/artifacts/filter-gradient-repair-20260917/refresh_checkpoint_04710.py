"""Refresh the bounded F14 optional-route unit after the pushed streaming repair."""
import json
from pathlib import Path
work=Path('/tmp/bayesfilter-filter-gradient-xla-validation-20260918/docs/plans')
raw=Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
p=work/'filter_gradient_repair_ledger_20260917.json';d=json.loads(p.read_text());c=d['current_checkpoint'];a=c['ledh_pfor_disposition_allocation']
rows=[(int(p.parent.name[4:]),json.loads(p.read_text())) for p in sorted(raw.glob('run-*/run.json')) if int(p.parent.name[4:])>4706]
used={k:sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in rows if r['device']==k) for k in ('CPU','GPU')}
charges={k:a['base_charges_through04706'][k]+used[k] for k in used}
active=[n for n,r in rows if r['state']=='running'];through=max([4706]+[n for n,r in rows if r['state']!='running'])
a.update(used_workers=len(rows),used_seconds=used,failed_runs=[n for n,r in rows if r['state']=='failed'])
notes=a.get('progress_notes','Streaming repair020d794be is pushed. Optional batch pfor is removed/rejected; two historical exploratory harnesses stop before framework import. Default sequential call path is unchanged. CPU qualification is active. Additional unguarded pfor sites are recorded and F14 remains open.')
next_action=a.get('next_action','Finish optional-branch CPU/GPU qualification and source-bound readback/policy, then qualify the remaining Contract E JVP, scalar-transport fallback and reference-scout pfor sites. Unshared GPU streaming costs/capacity and performance follow-up remain pending.')
c.update(through_run=through,charged_seconds=charges,remaining_hours={k:cap-charges[k]/3600 for k,cap in {'CPU':56,'GPU':52}.items()},active_job=active or None,active_session=None,status=notes,next=next_action)
p.write_text(json.dumps(d,indent=2)+'\n')
text=f'''Branch: repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Pushed streaming base
020d794be; current F14 unit follows that checkpoint. Main remains unmerged.

Active question: eliminate unapproved optional batch pfor without changing
sequential analytical scores. Plan: filter_gradient_ledh_pfor_disposition_20260929.md.
Through {through:05d}; active workers: {active or 'none'}.
Charged/reserved CPU {charges['CPU']:.6f}s / GPU {charges['GPU']:.6f}s.
Remaining CPU {c['remaining_hours']['CPU']:.6f}h / GPU {c['remaining_hours']['GPU']:.6f}h.
Global caps56 CPU/52 GPU process-hours include the extra24 CPU hours.
Active allocation8 workers/1800 CPU/1200 GPU seconds; used/reserved
{len(rows)} workers/{used['CPU']:.6f} CPU/{used['GPU']:.6f} GPU seconds.
One numerical worker at a time. CPU explicit reference; trusted GPU growth
required. Timing needs unshared non-display hardware. PID2260909 held GPU2/3
contexts at the latest streaming cost preflight; do not stop unrelated jobs.

{notes}

Next: {next_action}

Completed streaming unit04688--04706: exact original RNG scheduling and complete
buffered numerical records,9 qualification checks per device,43 numerical
regressions per device,8 healthy CPU capacity workers throughT128/N64,164 final
readback/policy checks. Removed full-horizon random storage. Fresh-owner native
RSS persists after Python GC; explicit retained-owner reuse is boundedly
qualified. CPU streaming warm times are descriptively slower at long horizons;
GPU uncontended timing/allocator/capacity remains pending. Archive/result:
filter_gradient_ledh_streaming_memory_result_20260929.md. Prior seeded LM,
validity-boundary and actual DZ5 evidence retain their documented source scope.

Other gates: additional F14 Contract E JVP/scalar-transport/reference scout
sites, registered analytical-score consumer migration/costs, current-source
measurement applicability, DZ5 locator121 strict trajectory differences with
unconverged optimizers,4539 strict fitted-geometry CPU/GPU record differences
and isotropic-angle reporting, all F01--F20 dispositions. Locator optimized-HLO
invariant-copy lead is not a causal repair. See terminal_gap_queue_20260928.

No live MacroFinance edits, subagents, training, HMC, package/environment or
system/cache changes, tolerance relaxation, or canonical LEDH rebuild. Preserve
canonical author-profile NeuTra IAF and original LEDH seed streams. Strong
reset rejection and unsupported-claim blocks remain. No main merge until all
master gates pass. Do not silently relax old source-bound readbacks.
'''
(work/'filter_gradient_repair_resume_20260919.md').write_text('# Filter and gradient repair resume checkpoint\n\n'+text)
p=work/'filter_gradient_repair_master_20260917.md';old=p.read_text();p.write_text(old.split('\n',1)[0]+'\n\n'+text+'\n'+old[old.index('Older checkpoints below'):])
print(json.dumps({'through':through,'active':active,'used':used,'remaining_hours':c['remaining_hours']}))
