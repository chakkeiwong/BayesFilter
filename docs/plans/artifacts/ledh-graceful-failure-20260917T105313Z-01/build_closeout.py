"""Assemble evidence decisions without evaluating held-out data or running models."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[4]
relative = Path('docs/plans/artifacts/ledh-graceful-failure-20260917T105313Z-01')
campaign = root / relative
ledger = json.loads((campaign / 'campaign.json').read_text())
assert len(ledger['attempts']) == 3
assert not any(r['status'] == 'running' for r in ledger['runs'])
assert subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip() == ledger['branch']

def save(name, data):
    (campaign / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

attempts = []
for number, folders in enumerate([
    ['lg1-control-curves-01','lg1-n4096-curves-01'],
    ['lg1-halton-curves-01'], ['lg1-cap-repair-curves-01']], 1):
    grouped = {}
    for folder in folders:
        source = campaign / folder / 'curves.json'
        for group in json.loads(source.read_text())['curves']:
            maxima = {key:max(abs(row[key]) for row in group['records'])
                      for key in ('max_mean_error','max_cov_error','log_ratio_error')}
            valid = all(row['value_valid'] and row['force_valid'] for row in group['records'])
            passed = valid and maxima['max_mean_error'] <= .05 and maxima['max_cov_error'] <= .05 and maxima['log_ratio_error'] <= .02
            grouped.setdefault(group['name'], []).append(dict(
                replicate=group['replicate'], particle_count=group['n'],
                fixture_identity=group['target_identity'], status_valid=valid,
                maxima=maxima, calibration_screen_passed=passed,
                parameter_design=[row['phi'] for row in group['records']],
                source=str(source.relative_to(campaign))))
    candidates = [dict(name=name, groups=groups,
                       calibration_screen_passed=all(g['calibration_screen_passed'] for g in groups))
                  for name, groups in grouped.items()]
    assert not any(c['calibration_screen_passed'] for c in candidates)
    attempts.append(dict(attempt=number, candidates=candidates,
                         candidate_count=len(candidates),
                         dataset_candidate_groups=sum(len(c['groups']) for c in candidates),
                         passing_candidates=0))
save('calibration-summary.json', dict(
    classification='descriptive calibration screens, not statistical rankings or untouched validation',
    thresholds=dict(max_mean_error=.05,max_cov_error=.05,log_ratio_error=.02),
    attempts=attempts))

failure_reasons = {
    'containment-cpu-01':'Fixture supplied mismatched exact/biased configuration keys; repaired fixture.',
    'containment-cpu-02':'Test command used a nonexistent filename ending in _tf.py; corrected to the actual test path and reran.',
    'containment-cpu-03':'Existing nested annealing limitation reached by the full-feature test; implemented common tf.while_loop path.',
    'correction-cpu-01':'Pre-existing test imports deleted cubature_genut_batch_tf module. That obsolete module was not recreated; active shared-correction tests passed.',
    'annealing-cpu-01':'TensorFlow loop shape changed after one step; repaired shape invariant and reran actual compiled recursion.',
    'block-transport-gpu-01':'Local variable binding error in diagnostic block harness; fixed.',
    'block-transport-gpu-02':'XLA requires compile-time Range extent; block harness now uses fixed extent and variable offset.',
    'block-transport-gpu-03':'Single finite-difference step was insufficient under float32 rounding; used the master-prescribed step ladder with the unchanged 1e-2 criterion.',
    'public-tuner-cpu-01':'Numerics finished but JSON serializer rejected mappingproxy; converted host mappings and retried in a fresh directory.',
    'final-regressions-cpu-01':'141 passed; obsolete dual API fixtures and omitted optional fields caused five failures/four setup errors. Repaired real API tests passed.',
    'stale-fixture-repair-cpu-01':'12 passed; invalid process covariance was correctly flagged before time zero, while test expected time zero. Corrected to initial-stage time -1.',
}
accounted = []
for run in ledger['runs']:
    path = campaign / Path(run['manifest']).relative_to(Path(ledger['worktree']) / relative)
    manifest = json.loads(path.read_text())
    accounted.append(dict(label=run['label'],status=run['status'],
                         exit_code=manifest.get('exit_code'),
                         wall_seconds=manifest.get('wall_seconds'),
                         cpu_worker_seconds=manifest.get('cpu_worker_seconds'),
                         gpu_device_seconds=manifest.get('gpu_device_seconds'),
                         manifest=str(path.relative_to(campaign)),
                         failure_classification=failure_reasons.get(run['label'])))
save('run-accounting.json',dict(
    limits=ledger['limits'],charged=ledger['charged'],
    setup_cpu_reserve_seconds=ledger['setup_cpu_reserve_seconds'],
    setup_note=ledger['setup_note'], runs=accounted,
    note='Command success records harness completion; scientific screen decisions are separate. GPU charge is whole GPU-command wall time; CPU is child-process user+system time.'))

cases = {}
for name in ('LG1','LG2','LG4','LG20','SV20','SV40','PP20','SIR3'):
    reasons = ['LG1_FIRST_PREREQUISITE_NOT_PASSED']
    next_action = 'After a revised LG1 program passes, freeze and validate this exact model/data/tuning scope.'
    if name == 'LG1':
        reasons = ['CALIBRATION_SCREEN_FAILED','SUBSTANTIVE_ATTEMPT_LIMIT_REACHED',
                   'LEDH_TUNING_AUTHORITY_NOT_BOUND','TF_HMC_ADMISSION_UNAVAILABLE']
        next_action = 'Approve a revised bounded LG1 repair program; first localize first-step proposal/reset cloud error, then repeat permitted calibration before untouched validation and authority binding.'
    elif name.startswith('SV'):
        reasons.append('STATE_DEPENDENT_VARIANCE_PROPOSAL_CONFORMANCE_NOT_ESTABLISHED')
        next_action += ' Establish the true zero-mean, state-dependent-variance observation proposal without substituting a different likelihood.'
    elif name == 'PP20':
        reasons.append('CORRECTED_RK4_PARAMETER_TANGENT_ADAPTER_NOT_IMPLEMENTED')
    elif name == 'SIR3':
        reasons.append('FIXED_SIR_STATE_FILTER_ADAPTER_NOT_IMPLEMENTED')
        next_action += ' This is a fixed-parameter state-filtering scope, not a parameter-posterior study.'
    else:
        reasons.append('LINEAR_ADAPTER_SCOPE_NOT_VALIDATED')
    cases[name] = dict(
        engineering_containment='passed' if name=='LG1' else 'not_checked',
        engineering_scope='shared endpoint fixtures; not global or model-specific proof',
        healthy_equivalence='passed' if name=='LG1' else 'not_checked',
        healthy_equivalence_scope='saved full-feature shared fixture; not full LG1 scope validation',
        analytical_score='passed' if name=='LG1' else 'not_checked',
        analytical_scope='tested baseline LG1 configurations only; final cap16 scopes are not promoted',
        filtering_reference='failed' if name=='LG1' else 'not_checked',
        finite_target_posterior='not_checked',model_posterior='not_checked',
        posterior_applicable=name!='SIR3',numerical_exclusion='unknown',
        heuristic_dominance='not_checked',execution_status='blocked',
        reason_codes=reasons,next_justified_action=next_action,
        evidence=['calibration-summary.json','endpoints.md'] if name=='LG1' else ['endpoints.md'],
        claim_scope_identity=None, repository_tuning_artifact=None,
        conditional_heuristic_sample_counts=None,statistical_ranking='not_supported')
save('decision.json',dict(
    classification='investigation closeout at the declared stop condition; no admission authority',
    program=ledger['program'],program_sha256=ledger['program_sha256'],
    baseline_commit=ledger['git_commit'],branch=ledger['branch'],
    investigation_accounting='completed',validation_objective='not_achieved',
    stop_reason='LG1 substantive attempt ceiling reached; dependent expansion requires LG1 first',
    cases=cases, untouched_validation_and_claim_partitions=True,
    independent_terminal_review='not_performed; supervisor source/evidence review only',
    result_note='../../ledh-graceful-failure-execution-result-2026-09-18.md'))

changed = subprocess.check_output(['git','diff',ledger['git_commit'],'--name-only'],cwd=root,text=True).splitlines()
changed += subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines()
python_paths = sorted({name for name in changed if name.endswith('.py')})
for name in python_paths:
    ast.parse((root/name).read_text(),filename=name)
native=root/'bayesfilter/ops/_symmetric_sylvester_ops.so'
save('final-source-inventory.json',dict(
    baseline_commit=ledger['git_commit'],branch=ledger['branch'],
    python_syntax='passed',python_file_count=len(python_paths),
    changed_python_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in python_paths},
    native_library_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
    limitation='Fixture source closures omit some relative imports/native dependencies; these hashes are provenance, not canonical admission identity.'))
print(json.dumps(dict(attempts=[dict(attempt=a['attempt'],candidates=a['candidate_count'],groups=a['dataset_candidate_groups'],passed=a['passing_candidates']) for a in attempts],
                      runs=len(accounted),charged=ledger['charged'],python_syntax_files=len(python_paths))))
