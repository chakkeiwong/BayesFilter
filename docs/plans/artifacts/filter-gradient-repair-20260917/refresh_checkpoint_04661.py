"""Refresh the SVD repair checkpoint; counts all prior charges once."""
import json
from pathlib import Path
work=Path('/tmp/bayesfilter-filter-gradient-xla-validation-20260918/docs/plans')
raw=Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
rows=[(int(p.parent.name[4:]),json.loads(p.read_text())) for p in sorted(raw.glob('run-*/run.json')) if int(p.parent.name[4:])>=4551]
charges=dict(CPU=95591.73871665674,GPU=90517.75145042063)
for _,row in rows:charges[row['device']]+=row.get('elapsed_seconds',row['timeout_seconds'])
static_rows=[json.loads(p.read_text()) for p in sorted(raw.glob('terminal-*-20260928-r*/run.json'))]
charges['CPU']+=sum(r.get('elapsed_seconds',r.get('timeout_seconds',0)) for r in static_rows)
phase=[(n,r) for n,r in rows if 4579<=n<=4605 and not r['key'][1].startswith(('factor_clipped_anchor_', 'dz5_locator_boundary_'))]
anchor_phase=[(n,r) for n,r in rows if n>=4579 and r['key'][1].startswith('factor_clipped_anchor_')]
boundary_phase=[(n,r) for n,r in rows if n>=4596 and r['key'][1].startswith('dz5_locator_boundary_')]
exact_phase=[(n,r) for n,r in rows if 4606<=n<=4610 and r['key'][1].startswith('dz5_exact_fit_')]
symmetry_phase=[(n,r) for n,r in rows if 4611<=n<=4617]
active=[{'run':n,'group':r['key'][1]} for n,r in rows if r['state']=='running']
failed=[{'run':n,'group':r['key'][1]} for n,r in phase if r['state'] not in ('running','passed')]
anchor_failed=[{'run':n,'group':r['key'][1]} for n,r in anchor_phase if r['state'] not in ('running','passed')]
p=work/'filter_gradient_repair_ledger_20260917.json';d=json.loads(p.read_text());c=d['current_checkpoint']
c.update(through_run=max(n for n,r in rows if r['state']!='running'),based_on_commit='4bf50d914',status='Final factor precision symmetry repair qualified CPU/GPU through04617; actual consumer source renewal and terminal master gaps remain.',charged_seconds=charges,remaining_hours={k:v-charges[k]/3600 for k,v in dict(CPU=56,GPU=52).items()},active_job=active or None,active_session=None)
c['principal_angle_allocation']={'first_run':4579,'workers':16,'combined_seconds':3600,'used_workers':len(phase),'used_combined_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in phase),'failed_runs':failed,'global_caps_unchanged':True}
c['factor_clipped_anchor_allocation']={'workers':8,'combined_seconds':1800,'used_workers':len(anchor_phase),'used_combined_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in anchor_phase),'failed_runs':anchor_failed,'global_caps_unchanged':True}
c['locator_boundary_allocation']={'workers':8,'combined_CPU_seconds':3600,'used_workers':len(boundary_phase),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in boundary_phase),'failed_runs':[n for n,r in boundary_phase if r['state'] not in ('running','passed')],'global_caps_unchanged':True}
c['exact_fit_allocation']={'workers':8,'combined_seconds':1800,'used_workers':len(exact_phase),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in exact_phase),'failed_runs':[n for n,r in exact_phase if r['state'] not in ('running','passed')],'global_caps_unchanged':True}
c['symmetry_allocation']={'workers':12,'combined_seconds':2400,'used_workers':len(symmetry_phase),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in symmetry_phase),'failed_runs':[n for n,r in symmetry_phase if r['state'] not in ('running','passed')],'global_caps_unchanged':True}
c['principal_angle_GPU_runs']=[4599,4600,4601]
c['principal_angle_GPU_warm_cost_status']='Observed warm time +21.87% in shared-device descriptive probes; defective baseline, no ranking. Matched uncontended cost attribution remains open. Allocator peak unchanged at70400bytes.'
c['locator_trajectory_runs']=[4584,4585,4586,4590,4593,4594,4595]
c['latest_completed_policy_run']=4617
c['latest_completed_policy_checks']=161
if any(n==4628 and r['state']=='passed' for n,r in rows):
    c['latest_completed_policy_run']=4628
    c['latest_completed_policy_checks']=181
if any(n==4630 and r['state']=='passed' for n,r in rows):
    c['latest_completed_policy_run']=4630
    c['latest_completed_policy_checks']=160
if any(n==4641 and r['state']=='passed' for n,r in rows):
    c['latest_completed_policy_run']=4641
    c['latest_completed_policy_checks']=162
c['exact_fit_result']='docs/plans/filter_gradient_dz5_exact_fit_result_20260928.md'
c['exact_fit_numerical_outcomes']={'4606_before_GPU':'rejected_stability_error1','4607_after_GPU':'rejected_stability_error2','4608_before_CPU':'usable','4609_after_CPU':'usable_exact_same_raw_public_as_before'}
c['exact_fit_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/exact-fit-inputs-04610-evidence.tar.gz','verified_members':39,'sha256':'dc44b1ea16c490b4f4f43f92464823af68b544cd373b66ece69ec6d5741af05c'}
c['locator_anchor_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/locator-anchor-cpu-04595-evidence.tar.gz','sha256':'a756686f6164ddeca63ac4e17f654a84d9d3540ad16bc7f8791c096baec293ff','verified_members':84}
c['factor_clipped_anchor_plan']='docs/plans/filter_gradient_factor_clipped_anchor_repair_20260928.md'
renewal=[(n,r) for n,r in rows if n>=4618 and r['key'][1].startswith('dz5_initializer_')]
adapter=[(n,r) for n,r in rows if n>=4627 and r['key'][1].startswith('terminal_adapter_')]
context=[(n,r) for n,r in rows if n>=4627 and r['key'][1].startswith('dz5_locator_first_context_')]
real_context=[(n,r) for n,r in rows if n>=4635 and r['key'][1].startswith('dz5_locator_one_iteration_')]
counter_context=[(n,r) for n,r in rows if n>=4638 and r['key'][1].startswith('dz5_locator_counter_')]
family_context=[(n,r) for n,r in rows if n>=4642 and r['key'][1].startswith('dz5_locator_family_')]
progress_context=[(n,r) for n,r in rows if n>=4646 and r['key'][1].startswith('dz5_locator_progress_')]
reporting_context=[(n,r) for n,r in rows if n>=4649 and r['key'][1].startswith('dz5_locator_reporting_')]
derived_context=[(n,r) for n,r in rows if n>=4654 and r['key'][1].startswith('dz5_locator_derived_replay')]
optimized_context=[(n,r) for n,r in rows if n>=4656 and r['key'][1].startswith('dz5_locator_optimized_')]
real_static=[r for r in static_rows if 'analyze_locator_hlo_context.py' in ' '.join(r.get('command',[]))]
indexed=[(n,r) for n,r in rows if n>=4627 and r['key'][1]=='terminal_endpoint_evidence_index_cpu']
def passed(group):
    return any(r['key'][1]==group and r['state']=='passed' for _,r in rows)
renewal_done=passed('dz5_initializer_renewal_readback_cpu')
adapter_done=passed('terminal_adapter_imports_cpu') and passed('terminal_adapter_policy_cpu')
context_done=passed('dz5_locator_first_context_readback_cpu')
real_context_done=passed('dz5_locator_one_iteration_readback_cpu')
counter_context_done=passed('dz5_locator_counter_reduction_readback_cpu')
family_context_done=passed('dz5_locator_family_readback_cpu')
progress_context_done=passed('dz5_locator_progress_readback_cpu')
reporting_context_done=passed('dz5_locator_reporting_readback_cpu')
derived_context_done=passed('dz5_locator_derived_replay_readback_cpu')
optimized_context_done=passed('dz5_locator_optimized_readback_cpu')
optimized_static=[r for r in static_rows if any(name in ' '.join(r.get('command',[])) for name in ('analyze_filter_repair_locator_optimized_hlo.py', 'inspect_filter_repair_locator_fusions'))]
optimized_closed=passed('dz5_locator_optimized_inspection_cpu') and any(n>=4661 and r['key'][1]=='dz5_locator_optimized_readback_cpu' and r['state']=='passed' for n,r in rows) and len(optimized_static)==2 and all(r['state']=='passed' for r in optimized_static)
c['locator_optimized_allocation']={'workers':6,'CPU_seconds':4200,'used_workers':len(optimized_context),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in optimized_context)+sum(r.get('elapsed_seconds',r['timeout_seconds']) for r in optimized_static),'failed_runs':[n for n,r in optimized_context if r['state'] not in ('running','passed')],'status':'closed_explanatory_comparison_passed' if optimized_closed else ('readback_passed_structural_review_pending' if optimized_context_done else 'active_allocated'),'global_caps_unchanged':True}
c['locator_optimized_plan']='docs/plans/filter_gradient_dz5_locator_optimized_hlo_20260928.md'
c['locator_derived_replay_allocation']={'workers':4,'CPU_seconds':1200,'used_workers':len(derived_context),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in derived_context),'failed_runs':[n for n,r in derived_context if r['state'] not in ('running','passed')],'status':'closed_diagnostic_readback_passed' if derived_context_done else 'active_allocated','global_caps_unchanged':True}
c['locator_derived_replay_plan']='docs/plans/filter_gradient_dz5_locator_derived_replay_20260928.md'
c['locator_reporting_result']='docs/plans/filter_gradient_dz5_locator_reporting_storage_result_20260928.md'
c['locator_reporting_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/locator-reporting-storage-04653-evidence.tar.gz','sha256':'bec562f76e0606e15fadd4fff57398b6ce2f276a7a9b5efb02419676ca58cf65','verified_members':71}
c['locator_reporting_allocation']={'workers':6,'CPU_seconds':1800,'used_workers':len(reporting_context),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in reporting_context),'failed_runs':[n for n,r in reporting_context if r['state'] not in ('running','passed')],'status':'closed_diagnostic_readback_passed' if reporting_context_done else 'active_allocated','global_caps_unchanged':True}
c['locator_reporting_plan']='docs/plans/filter_gradient_dz5_locator_reporting_storage_20260928.md'
c['locator_progress_result']='docs/plans/filter_gradient_dz5_locator_progress_split_result_20260928.md'
c['locator_progress_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/locator-progress-split-04648-evidence.tar.gz','sha256':'bacc315f6a47d11dca31ff2b6a5d060b34e4554a1d5cddaae075c44f2898ecb1','verified_members':38}
c['locator_progress_allocation']={'workers':5,'CPU_seconds':1500,'used_workers':len(progress_context),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in progress_context),'failed_runs':[n for n,r in progress_context if r['state'] not in ('running','passed')],'status':'closed_diagnostic_readback_passed' if progress_context_done else 'active_allocated','global_caps_unchanged':True}
c['locator_progress_plan']='docs/plans/filter_gradient_dz5_locator_progress_split_20260928.md'
c['locator_family_result']='docs/plans/filter_gradient_dz5_locator_accounting_families_result_20260928.md'
c['locator_family_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/locator-accounting-families-04645-evidence.tar.gz','sha256':'68d4dc56610cc97df0f43e4e19191ede7c325245ea6b2df8a8cbd64b5b28850f','verified_members':55}
if family_context_done:
    c['latest_completed_policy_run']=4645
    c['latest_completed_policy_checks']=164
c['locator_family_allocation']={'workers':6,'CPU_seconds':1800,'used_workers':len(family_context),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in family_context),'failed_runs':[n for n,r in family_context if r['state'] not in ('running','passed')],'status':'closed_diagnostic_readback_passed' if family_context_done else 'active_allocated','global_caps_unchanged':True}
c['locator_family_plan']='docs/plans/filter_gradient_dz5_locator_accounting_families_20260928.md'
c['terminal_gap_queue']='docs/plans/filter_gradient_terminal_gap_queue_20260928.md'
f18=next(item for item in d['findings'] if item['id']=='F18')
f18.setdefault('historical_remaining_call_chain_debt_04641',f18.get('remaining_call_chain_debt',[]))
f18['remaining_call_chain_debt']=[
    'DZ5 locator historical-context mismatch:121 strict record leaves; both optimizers unconverged. Through04661, optimized IR and exact controls isolate four floating gradient fusions that retain eight loop-invariant copies of printed2.4 in the candidate but embed constants in the original/replay-int32 arms. Targeted causal confirmation and a GPU-compatible runtime remedy remain open; no tolerance waiver.',
    'Geometry selected precision/covariance/center passes scoped bounds;4539 strict fitted-record differences and strict/isotropic angle reporting remain. Preserve exact-input and symmetrization evidence.',
    'Actual DZ5 r2 target/initializer and CPU/GPU process-lifetime renewal passed04618--04628; no unchanged renewal rerun is needed. Old r1 context probes are not current-source admission.',
    'Adapter import isolation passed04629--04630;43 exports/three aliases preserved,277-source guard, no new exception. Complete dynamic consumer/import coverage remains a terminal obligation.',
    'Endpoint-specific evidence index passes integrity tests04631; current dependency and measurement-scope applicability still needs contextual review. Do not launch the old1116 missing-pair matrix.',
    'Terminal F01--F20 consumer/evidence dispositions, numerical gaps, matched cost attribution and final remote integration remain. See filter_gradient_terminal_gap_queue_20260928.md.'
]
c['source_renewal_allocation']={'workers':14,'combined_seconds':15000,'used_workers':len(renewal),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in renewal),'failed_runs':[n for n,r in renewal if r['state'] not in ('running','passed')],'global_caps_unchanged':True}
c['terminal_adapter_allocation']={'workers':4,'CPU_seconds':1200,'used_workers':len(adapter),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in adapter),'failed_runs':[n for n,r in adapter if r['state'] not in ('running','passed')],'status':'active' if adapter else 'prepared_not_started','global_caps_unchanged':True}
c['locator_context_allocation']={'workers':6,'CPU_seconds':1800,'used_workers':len(context),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in context),'failed_runs':[n for n,r in context if r['state'] not in ('running','passed')],'status':'active' if context else 'prepared_not_started','global_caps_unchanged':True}
c['locator_real_optimizer_allocation']={'workers':6,'CPU_seconds':1800,'used_workers':len(real_context),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in real_context),'failed_runs':[n for n,r in real_context if r['state'] not in ('running','passed')],'status':'closed_diagnostic_readback_passed' if real_context_done else 'active_allocated','global_caps_unchanged':True}
c['locator_real_optimizer_allocation']['used_seconds']+=sum(r.get('elapsed_seconds',r['timeout_seconds']) for r in real_static)
c['locator_counter_allocation']={'workers':4,'CPU_seconds':1200,'used_workers':len(counter_context),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in counter_context),'failed_runs':[n for n,r in counter_context if r['state'] not in ('running','passed')],'status':'closed_diagnostic_readback_passed' if counter_context_done else 'active_allocated','global_caps_unchanged':True}
c['source_renewal_allocation']['status']='closed_renewed_evidence_passed' if renewal_done else 'active'
c['terminal_adapter_allocation']['status']='closed_passed' if adapter_done else ('active_allocated' if renewal_done else 'prepared_not_started')
c['locator_context_allocation']['status']='closed_diagnostic_readback_passed' if context_done else ('active_allocated' if adapter_done else 'prepared_not_started')
c['terminal_evidence_index_allocation']={'CPU_seconds':600,'test_workers':len(indexed),'used_seconds':sum(r.get('elapsed_seconds',r['timeout_seconds']) for _,r in indexed)+sum(r.get('elapsed_seconds',r.get('timeout_seconds',0)) for r in static_rows if 'index_filter_repair_endpoint_evidence.py' in ' '.join(r.get('command',[]))),'status':'readback_and_integrity_tests_passed' if passed('terminal_endpoint_evidence_index_cpu') else 'readback_complete_tests_pending','global_caps_unchanged':True}
c['active_allocations']=([name for name,done in [('source_renewal_allocation',renewal_done),('terminal_adapter_allocation',not renewal_done or adapter_done),('locator_context_allocation',not adapter_done or context_done)] if not done]+([] if passed('terminal_endpoint_evidence_index_cpu') else ['terminal_evidence_index_allocation']))
c['next']=('Finish active GPU rejection and renewed saved-evidence/policy readback; both CPU/GPU accepted lifetimes pass. Then qualify the adapter import repair.' if not renewal_done else 'Run terminal_adapter_imports_cpu and terminal_adapter_policy_cpu within the separate1200-second CPU allocation.' if not adapter_done else 'Execute original/candidate first-objective locator-context diagnostic and readback within its separate1800-second CPU allocation.' if not context_done else 'Continue endpoint-specific evidence/dependency reconciliation and bounded repairs for the unresolved strict numerical and registered LEDH consumer gaps. Main remains unmerged.')
if active:
    c['next']='Active: '+str(active)+'. One numerical worker at a time. '+c['next']
if context_done and not real_context_done:
    c['next']='The truncated dispatch removes the historical score discrepancy and is not a repair. Execute the reviewed genuine one-iteration optimizer control (original/candidate/readback), not another full trajectory. Active: '+str(active or 'none')+'.'
    c['active_allocations'].append('locator_real_optimizer_allocation')
if real_context_done and not counter_context_done:
    c['next']='Real one-iteration control reproduces the historical first-score difference exactly (04635--04637). Saved HLO identifies accounting-width differences as a lead. Execute the reviewed isolated CPU int32-accounting diagnostic and readback; no runtime counter change or full trajectory. Active: '+str(active or 'none')+'.'
    c['active_allocations'].append('locator_counter_allocation')
    if passed('dz5_locator_counter_readback_cpu'):
        c['next']='CPU accounting-width intervention restores the original first score exactly (04638--04639). Execute the remaining single-reduction/int64-resource diagnostic and readback within the same4-worker/1200-second unit; GPU qualification and runtime repair remain open. Active: '+str(active or 'none')+'.'
c['status']='Renewed actual CPU/GPU accepted initializer lifetimes pass; adapter reference-import repair and terminal evidence reconciliation continuing.'
if counter_context_done:
    c['status']='Source-renewal and adapter isolation pass; real one-iteration reproduction localizes the CPU score difference to accounting dtype/context. Single-reduction change is insufficient; no runtime counter repair or main merge.'
    c['next']='Prepare bounded resource/index-family or optimized-HLO localization using the positive one-iteration control. Preserve int64 GPU resources and the negative single-reduction result. No full trajectory or runtime change without a qualified smaller remedy. Registered LEDH consumer migration, strict fitted/isotropic records, current-source cost review and terminal F01--F20 dispositions remain open.'
c['terminal_cost_readback']={'root':'terminal-cost-readback-20260928-r1','status':'nonpassing legacy-only comparator; endpoint-specific cost evidence index required','pairs':0,'missing':1116,'excluded':770,'charged_CPU_seconds':4.774958212976344}
if not family_context_done:
    c['active_allocations'].append('locator_family_allocation')
    c['next']='Execute the reviewed isolated CPU index/calls, progress and invalid-row accounting-family probes, then saved-evidence/policy readback within the new6-worker/1800-second unit. Preserve int64 GPU runtime resources. Active: '+str(active or 'none')+'. Other master gaps remain open; no main merge.'
else:
    c['next']='Read the completed accounting-family result and prepare the next bounded intervention or optimized-HLO localization. No GPU/runtime remedy or full-trajectory qualification inferred. Registered LEDH migration, strict fitted/isotropic records, cost applicability and F01--F20 remain open.'
c['terminal_endpoint_index']={'root':'terminal-endpoint-index-20260928-r2','status':'22 reports;282 unique run manifest/payload identities verified; scoped dependency review still required','integrity_failures':0,'unrecorded_GenUT_hash_runs':list(range(4246,4260)),'no_numerical_reexecution':True}
if family_context_done and not progress_context_done:
    c['active_allocations'].append('locator_progress_allocation')
    c['next']='Progress-counter family alone restores the original first score (04643); index/calls and invalid-row families do not (04642/04644), with164 checks passing04645. Execute the reviewed round-budget versus reporting-counter split and readback within5 workers/1500 CPU seconds. Active: '+str(active or 'none')+'. No GPU/runtime remedy or full trajectory.'
    c['status']='Accounting-family unit closed; positive intervention isolated to four progress counters. Further control/reporting split active; master repairs and main merge remain open.'
c['terminal_source_audit']={'root':'terminal-source-audit-20260928-r2','status':'AST inventory parsed and gzip verified; contextual review open','charged_CPU_seconds':249.2124072649749}
if progress_context_done:
    c['latest_completed_policy_run']=4648
    c['latest_completed_policy_checks']=163
    c['next']='Reporting-only counters restore the original first score04647; round-budget storage does not04646. Execute individual reporting-counter and int64-storage/int32-arithmetic probes, then readback in the reviewed6-worker/1800-second CPU unit. Active: '+str(active or 'none')+'. No runtime repair or full trajectory.'
    c['status']='Progress split closed; positive CPU intervention isolated to reporting-only counters. Storage-compatible remedy still unqualified; master gaps remain.'
    if not reporting_context_done:
        c['active_allocations'].append('locator_reporting_allocation')
c['CPU_lifetime_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/dz5-cpu-lifetime-04625-evidence.tar.gz','sha256':'39a49c86c86d7ad034e08759d12e93efc346999dd320dd82fb0ca7a944bc8704','verified_members':38}
if reporting_context_done:
    c['latest_completed_policy_run']=4653
    c['latest_completed_policy_checks']=164
    c['next']='Replay counter storage alone restores original first score04651; attempts/optimizer counter changes and int64-storage/int32-increments do not04649/04650/04652. Execute reviewed derived-replay report probe and complete short-record readback within4 workers/1200 CPU seconds. Active: '+str(active or 'none')+'. No runtime remedy or full trajectory admitted.'
    c['status']='Reporting localization closed; replay resource is a sufficient CPU context intervention. Storage-preserving increments negative. Derived replay diagnostic active; master gaps remain.'
    if not derived_context_done:
        c['active_allocations'].append('locator_derived_replay_allocation')
if derived_context_done:
    c['latest_completed_policy_run']=4655
    c['latest_completed_policy_checks']=161
    c['next']='Prepare and skeptically review a bounded optimized-HLO/lowering comparison of original, candidate and positive replay-int32 one-iteration controls. Replay storage is a sufficient context intervention; int32 increments with int64 storage and removing the replay variable both preserve the candidate. Keep runtime unchanged, no full trajectory before a smaller qualified remedy. Follow the terminal gap queue for other repairs.'
    c['status']='Through04655 all four new diagnostic units closed with valid artifacts. Replay-counter storage localizes the CPU context effect; two storage-compatible probes are negative. No runtime remedy, whole-program completion or main merge.'
    c['locator_derived_replay_result']='docs/plans/filter_gradient_dz5_locator_derived_replay_result_20260928.md'
    c['locator_derived_replay_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/locator-derived-replay-04655-evidence.tar.gz','sha256':'8fbe0f59cd630a3f9e7c6dff64dd114f6794cd953d6a916a69b77d0fc98107d3','verified_members':23}
    c['checkpoint_updater_snapshot']='docs/plans/artifacts/filter-gradient-repair-20260917/refresh_checkpoint_04655.py'
c['GPU_renewal_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/dz5-renewal-gpu-04628-evidence.tar.gz','sha256':'475e4b2c4db2a61842b3164a5ed6b7f12c5ca2e1a7f0fd91db1357a520e82c63','verified_members':55}
if not optimized_closed:
    c['active_allocations'].append('locator_optimized_allocation')
c['next']='Qualify the compiler-constant lead with a bounded targeted intervention on the four gradient fusions; preserve exact frozen inputs, controls and all strict failures. Eight unchanged loop operands print2.4 in the candidate; original/replay-int32 embed them. No additional dtype search or full trajectory before a smaller qualified remedy. Continue the independent registered LEDH valid/rejected-reset consumer repair and terminal source applicability queue.' if optimized_closed else 'Finish optimized-HLO caller inspection and full short-record reporting within the existing6-worker/4200-second unit. Active: '+str(active or 'none')+'.'
c['status']='Optimized compiler comparison completed through04661 with exact control reproduction and a concrete invariant-constant/fusion lead; no causal compiler pass or runtime remedy qualified. Master gaps remain.' if optimized_closed else 'Optimized compiler comparison executing under explicit CPU diagnostic exception; runtime and all master gates unchanged.'
if optimized_closed:
    c['latest_completed_policy_run']=4661
    c['latest_completed_policy_checks']=179
    c['locator_optimized_result']='docs/plans/filter_gradient_dz5_locator_optimized_hlo_result_20260928.md'
    c['checkpoint_updater_snapshot']='docs/plans/artifacts/filter-gradient-repair-20260917/refresh_checkpoint_04661.py'
    receipt=work/'artifacts/filter-gradient-repair-20260917/locator-optimized-04661-verification.json'
    if receipt.exists():
        archive=json.loads(receipt.read_text())
        c['locator_optimized_archive']={'path':archive['archive'],'sha256':archive['sha256'],'verified_members':archive['members_reopened_and_verified']}
c['terminal_import_index_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/terminal-import-index-04631-evidence.tar.gz','sha256':'eb656ebb7844b5166e219058be8b59de19625ae2ed85131727e50cbb0d9c5399','verified_members':31}
c['source_renewal_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/dz5-source-renewal-target-04624-evidence.tar.gz','sha256':'828138a4beff6df9c408fbd3df3ddf6fffa93279f0fb89d9e57644e710cd3bca','verified_members':816}
c['terminal_adapter_repair']={'plan':'docs/plans/filter_gradient_terminal_source_review_20260928.md','status':'qualified_real_import_prior_and_policy_checks' if adapter_done else 'lazy export dispatch prepared; actual import and prior qualification pending','guarded_sources':277,'new_policy_exceptions':0,'groups':['terminal_adapter_imports_cpu','terminal_adapter_policy_cpu']}
c['locator_context_plan']='docs/plans/filter_gradient_dz5_locator_context_20260928.md'
c['locator_context_execution']='diagnostic_readback_complete_no_runtime_repair_inferred' if context_done else ('allocated_after_adapter_qualification' if adapter_done else 'prepared_awaiting_prior_units')
c['locator_context_result']='docs/plans/filter_gradient_dz5_locator_context_result_20260928.md'
c['locator_context_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/locator-first-context-04634-evidence.tar.gz','sha256':'f689fe29232c636ce1ba1da22078c1547530c901e27d4bb5bc5061733b4a754e','verified_members':34}
c['locator_real_optimizer_plan']='docs/plans/filter_gradient_dz5_locator_one_iteration_20260928.md'
c['locator_real_optimizer_result']='docs/plans/filter_gradient_dz5_locator_one_iteration_result_20260928.md'
c['locator_real_optimizer_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/locator-one-iteration-04637-evidence.tar.gz','sha256':'39a15575996963eeef0380d370dda075e71e0807ebfe8463421bc448d92422a9','verified_members':36}
c['locator_counter_plan']='docs/plans/filter_gradient_dz5_locator_counter_context_20260928.md'
c['locator_counter_result']='docs/plans/filter_gradient_dz5_locator_counter_context_result_20260928.md'
c['locator_HLO_structure_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/locator-hlo-structure-04637-evidence.tar.gz','sha256':'bf78ff80356ea4dbf65439ded96c9079d53fb88855ddaa82fb34574c0fd8c36c','verified_members':9}
c['locator_counter_archive']={'path':'docs/plans/artifacts/filter-gradient-repair-20260917/locator-counter-context-04641-evidence.tar.gz','sha256':'4edf35e22473a020869d6fbdb4651f3e3e07f5f42601272d2030dbbfe52995eb','verified_members':45}
c['source_renewal_plan']='docs/plans/filter_gradient_dz5_source_renewal_20260928.md'
c['symmetry_allocation']['status']='closed_passed_through04617'
c['symmetry_result']='docs/plans/filter_gradient_factor_precision_symmetry_result_20260928.md'
c['exact_fit_allocation']['status']='closed_valid_artifacts_rejected_GPU_outcomes'
c['principal_angle_allocation']['status']='closed_through04605'
c['factor_clipped_anchor_allocation']['status']='closed_with_preserved_failures04592_04604'
c['locator_boundary_allocation']['status']='closed_negative_barrier_diagnosis'
c['allocation_note']='Other allocation fields in this accumulated checkpoint are historical; the named active allocations share unchanged 56 CPU / 52 GPU hour global caps.'
p.write_text(json.dumps(d,indent=2)+'\n')
text=f"""Branch: repair/filter-gradient-xla-validation-20260918 in
/tmp/bayesfilter-filter-gradient-xla-validation-20260918. Qualified numerical source checkpoint:4bf50d914 (pushed). Main remains unmerged; origin/main is
integrated. Check git HEAD for subsequent documentation/source checkpoints.

Active question: localize the remaining locator compiler-context difference
and reconcile terminal endpoint evidence. Source renewal and adapter import
isolation are complete for their tested scopes.
Through {c['through_run']:05d}; active: {active or 'none'}.
Global charged/reserved CPU {charges['CPU']:.6f}s / GPU {charges['GPU']:.6f}s.
Remaining CPU {c['remaining_hours']['CPU']:.6f}h / GPU {c['remaining_hours']['GPU']:.6f}h.
Caps are56CPU/52GPU process-hours; extra24CPU hours are already counted.
Symmetry unit closed: {len(symmetry_phase)}/12 workers,
{c['symmetry_allocation']['used_seconds']:.6f}/2400 combined seconds, no failures.
One numerical worker at a time. CPU is an explicit reference; GPU requires
trusted eligible device and verified memory growth. GPU0 serves remote desktop
and GPU1 has active display. Recheck non-desktop occupancy before any GPU job.

Checked findings and evidence:
- Fresh source renewal: CPU04625 and GPU04626 each pass two complete accepted
  initializer workers; intended GPU rejection04627 and181 readback/policy
  checks04628 pass. Parent RSS growth is0.410MiB CPU and0.559MiB GPU. Child
  native/compiler RSS persists after Python release, with HLO inspection
  contributing to the final host sample; process exit contains observed growth.
- Adapter reference isolation: committed/pushed713df846c. All43 exports and
  three aliases preserved;12 fresh-import/prior checks and160 policy checks
  pass.277 sources guarded, no new exception. Evidence index verifies282
  distinct saved run identities; current-source dependency review remains.
- Symmetry result: filter_gradient_factor_precision_symmetry_result_20260928.md.
  CPU04611/GPU04612 trials and derivatives pass. Exact GPU04613 fit is usable
  (before04607 rejected with error2), CPU04614 remains unchanged. GPU04615 and
  CPU04616 each pass52 regressions; readback/policy04617 passes161 checks.
  Optimizer, covariance, anchors and validity fields are preserved; factor
  precisions are exactly symmetric. Strict CPU/GPU record differences4539
  remain recorded. GPU allocator peak2306304bytes is unchanged.
- SVD result: filter_gradient_principal_angle_precision_repair_20260928.md.
  CPU04579--04583 and GPU04599--04601 qualify accuracy; GPU error1.55e-15,
  48 regressions pass, peak70400bytes unchanged. Shared-device warm time+21.87%
  remains descriptive; uncontended cost attribution and strict angles are open.
- Anchor result: filter_gradient_factor_clipped_anchor_repair_20260928.md.
  Initialization/regressions pass both backends. Exact-input04606--04610
  separated the final precision defect from anchor changes; the earlier
  regenerated-cloud GPU failure04604 cannot attribute an anchor regression.
- Locator: filter_gradient_dz5_locator_trajectory_result_20260928.md and
  filter_gradient_dz5_callback_boundary_result_20260928.md. Original/current
  callbacks474/504, first score difference1.53e-13 at identical row1;121 strict
  record leaves differ. All14 saved-point graph/XLA target checks pass04590.
  Identical-output controllers reproduce the original exactly04593. Explicit
  input binding and external barriers do not resolve the difference. Both
  optimizers remain unconverged; no tolerance changes.
- Locator context: truncated dispatch04632--04634 removes the discrepancy;
  genuine one-iteration04635--04637 reproduces it exactly in three objectives.
  Accounting-family04642--04645 and progress split04646--04648 isolate the
  positive intervention to reporting counters. Replay storage alone04651
  restores the first score; attempts/optimizer counters04649/04650 do not.
  Int64-storage/int32-increments04652 and derived-count04654 are negative;
  the latter matches every candidate short callback/record exactly. Readbacks
  pass through04655. See locator_reporting_storage_result and
  locator_derived_replay_result dated20260928. Runtime remains unchanged;
  optimized comparison04656--04661 now preserves every saved callback/short
  record exactly.179 final checks pass. Original/candidate/replay-int32 have
  26/22/26 constant-embedded gradient fusions; four candidate fusions instead
  carry eight unchanged copies of printed2.4 through loop entries143--150.
  This supplies a targeted mechanism lead, not causal proof or a runtime fix.
  See filter_gradient_dz5_locator_optimized_hlo_result_20260928.md.

Next: {c['next']}
New closed diagnostic allocations (detailed plans/results linked in ledger):
- Accounting families: {len(family_context)}/6 workers, {c['locator_family_allocation']['used_seconds']:.6f}/1800 CPU seconds.
- Progress split: {len(progress_context)}/5 workers, {c['locator_progress_allocation']['used_seconds']:.6f}/1500 CPU seconds.
- Reporting storage/arithmetic: {len(reporting_context)}/6 workers, {c['locator_reporting_allocation']['used_seconds']:.6f}/1800 CPU seconds.
- Derived replay: {len(derived_context)}/4 workers, {c['locator_derived_replay_allocation']['used_seconds']:.6f}/1200 CPU seconds.
Earlier renewal, import isolation and diagnostic allocations remain closed.
Optimized compiler unit: {len(optimized_context)}/6 workers, {c['locator_optimized_allocation']['used_seconds']:.6f}/4200 CPU seconds.
Plan: filter_gradient_dz5_locator_optimized_hlo_20260928.md.
No unchanged renewal cohort or full optimizer trajectory is requested.
Remaining work order and exit gates: filter_gradient_terminal_gap_queue_20260928.md.
Old consumer snapshots qualify only their own bytes. Strict precision and
isotropic reporting, locator rounding, matched current-source cost attribution,
registered LEDH consumer migration and F01--F20 terminal dispositions remain
open. Do not reuse old admission or classify numerical mismatches as equivalence.

Preserve live MacroFinance files and other campaigns. No subagents, training,
HMC, package/environment mutation, global cache changes, system-limit changes
or tolerance relaxation. Canonical NeuTra remains author-profile IAF; unsupported
LEDH claims remain blocked. Do not merge main until all master gates pass.
"""
(work/'filter_gradient_repair_resume_20260919.md').write_text('# Filter and gradient repair resume checkpoint\n\n'+text)
p=work/'filter_gradient_repair_master_20260917.md';old=p.read_text();p.write_text(old.split('\n',1)[0]+'\n\n'+text+'\n'+old[old.index('Older checkpoints below'):])
print({'through':c['through_run'],'active':active,'remaining_hours':c['remaining_hours']})
