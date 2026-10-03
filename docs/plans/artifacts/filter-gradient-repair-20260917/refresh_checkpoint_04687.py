"""Refresh registered LEDH continuation; count only charges after 04668."""
import json
from pathlib import Path

work = Path('/tmp/bayesfilter-filter-gradient-xla-validation-20260918/docs/plans')
raw = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
p = work / 'filter_gradient_repair_ledger_20260917.json'
d = json.loads(p.read_text())
c = d['current_checkpoint']
rows = [(int(p.parent.name[4:]), json.loads(p.read_text())) for p in sorted(raw.glob('run-*/run.json')) if int(p.parent.name[4:]) > 4668]
charges = dict(CPU=108237.06035735602, GPU=96144.14871976203)
for _, row in rows:
    charges[row['device']] += row.get('elapsed_seconds', row['timeout_seconds'])
active = [n for n, r in rows if r['state'] == 'running']
unit = [(n, r) for n, r in rows if r['key'][1].startswith('ledh_seeded_')]
c.update(through_run=max([4668] + [n for n,r in rows if r['state'] != 'running']),
    based_on_commit='aabd2b167', charged_seconds=charges,
    remaining_hours={k:cap-charges[k]/3600 for k,cap in dict(CPU=56,GPU=52).items()},
    active_job=active or None, active_session=None,
    status='Registered seeded value wrapper and LM precision repair pass CPU/GPU endpoint, primitive and numerical regressions. Preserved F14 static pfor failure04682, costs/capacity, score consumer and other master gates remain open.',
    next='Execute reviewed filter_gradient_ledh_streaming_memory_20260929.md: freeze the array-composed seeded authority, move seeded draws into the shared time loop, qualify memory/lifetime and renew GPU costs when unshared. Keep F14, score public migration and other master gaps open. No main merge.',
    active_allocations=[],
    ledh_seeded_public_plan='docs/plans/filter_gradient_ledh_seeded_public_20260929.md')
c['ledh_seeded_public_allocation'] = dict(workers=24, CPU_seconds=5400, GPU_seconds=3600,
    used_workers=len(unit), used_seconds={k:sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in unit if r['device']==k) for k in charges},
    failed_runs=[n for n,r in unit if r['state'] not in ('running','passed')], status='closed_bounded_numerics_CPU_costs_qualified_GPU_cost_capacity_open', global_caps_unchanged=True)
c['ledh_streaming_memory_plan']='docs/plans/filter_gradient_ledh_streaming_memory_20260929.md'
c['ledh_streaming_memory_allocation']={'workers':16,'CPU_seconds':3600,'GPU_seconds':2400,'status':'reviewed_not_started','global_caps_unchanged':True}
c['ledh_seeded_public_result']='docs/plans/filter_gradient_ledh_seeded_public_result_20260929.md'
p.write_text(json.dumps(d, indent=2)+'\n')
text = f'''Branch: repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Recovered committed,
pushed checkpoint aabd2b167. Main remains unmerged.

Active question: remove the registered seeded-owner process-noise buffer and
qualify reusable versus fresh-owner memory behavior. Prior value/LM repair is
boundedly qualified through04687. Next reviewed plan:
`filter_gradient_ledh_streaming_memory_20260929.md`.
Through {c['through_run']:05d}; active workers: {active or 'none'}.
Charged/reserved CPU {charges['CPU']:.6f}s / GPU {charges['GPU']:.6f}s.
Remaining CPU {c['remaining_hours']['CPU']:.6f}h / GPU {c['remaining_hours']['GPU']:.6f}h.
Global caps remain56 CPU/52 GPU process-hours; the extra24 CPU hours are included.
Closed seeded unit:19/24 workers,300.822211 CPU/456.624632 GPU seconds.
Next unit allocation:16 workers,3600 CPU/2400 GPU seconds; one numerical
worker at a time. CPU is an explicit reference. GPU requires trusted eligible hardware and
verified memory growth; recheck displays/occupancy before launch.

Current unit evidence:04669CPU RNG7,04670CPU endpoint8 and04671GPU RNG7 pass.
04672GPU endpoint has7 passes/1 healthy dual-trust ESS failure(1.31e-5).
04673 excludes RNG and seeded-owner wiring; normal inputs and native owners
match exactly.04674 diagnostic binding failure is preserved; one harness retry
uses an explicit factory closure.04675--04677 isolate inaccurate TF32 products
in the small shared LM solver. The explicit float32 product/JVP repair passes
CPU/GPU primitive/independent derivative checks04678/04679 and renewed CPU
endpoint04680; GPU endpoint04681 also passes8. CPU broad regression04682 has46 passes plus
one pre-existing static pfor failure, recorded under F14. GPU numerical-only
regressions04683 pass36 checks. CPU matched descriptive costs04684--04686 pass. Final current-source
readback/policy04687 passes162 checks. XLA cold5.03s/warm1.04ms; host RSS
increases521.86MiB after warm and993.70MiB after two extra one-shot calls.
Uncontended GPU cost preflight declined before launch; streaming seeded-memory
capacity and compiler/native owner lifetime remain open. Result:
filter_gradient_ledh_seeded_public_result_20260929.md.
Plan: filter_gradient_ledh_lm_precision_20260929.md. Costs remain held.

Completed evidence to reuse:
- Native LEDH rejection boundary04662--04668: three CPU/GPU guard cases and
  nine regressions per backend;162 final readback/policy checks. Failed resets
  yield NaN public value, false validity, code/index and retained raw diagnostics.
  Three CPU before/after fixtures preserve all reported raw fields exactly.
  Result: `filter_gradient_ledh_validity_boundary_result_20260928.md`.
- Locator optimized comparison04656--04661: all three arms reproduce saved
  short callbacks/records exactly;179 checks. Four candidate gradient fusions
  carry eight constant copies through loop entries143--150. Mechanism lead,
  not causal proof or runtime fix. Result: locator_optimized_hlo_result dated
  20260928. Full trajectories still have121 strict differences and unconverged
  optimizers; do not repeat unrelated dtype trials/full trajectories.
- Actual DZ5 source renewal04618--04628 and import isolation04629--04630 pass
  for their stated scopes. Evidence index04631 verifies282 saved runs.
- Precision symmetry and principal-angle repairs have component CPU/GPU
  evidence.4539 strict fitted-geometry record differences remain unresolved.

Next: {c['next']}
The seeded factory must have explicit fixed configuration and dynamic seed/
observation operands. The one-shot wrapper must refresh mutable Python callback
closures on each invocation; no identity-based global cache. The approved new
TensorFlow stream for geometry initializers does not authorize changing LEDH
SeedSequence/PCG64/Philox draws. No duplicated numerical kernels.

Other open gates and work order:
`filter_gradient_terminal_gap_queue_20260928.md`. Current-source timing/memory
applicability, registered analytical-score migration/costs, strict geometry/
isotropic reporting and all F01--F20 terminal dispositions remain open. Native
component passes are not registered-consumer or scientific admission.

Preserve live MacroFinance and other campaigns. No subagents, training, HMC,
package/environment changes, system-limit/cache changes or tolerance relaxation.
Preserve canonical author-profile NeuTra IAF. Canonical LEDH rebuild remains
excluded and unsupported claims blocked. No main merge until all master gates.
'''
(work/'filter_gradient_repair_resume_20260919.md').write_text('# Filter and gradient repair resume checkpoint\n\n'+text)
p = work/'filter_gradient_repair_master_20260917.md'
old = p.read_text()
p.write_text(old.split('\n',1)[0]+'\n\n'+text+'\n'+old[old.index('Older checkpoints below'):])
print(json.dumps({'through':c['through_run'], 'active':active, 'remaining_hours':c['remaining_hours'], 'unit':c['ledh_seeded_public_allocation']}))
