"""Read-only post-run evidence audit; does not execute scientific kernels."""
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
sys.dont_write_bytecode = True

ROOT = Path('/home/ubuntu/python/BayesFilter')
CAMPAIGN = ROOT / 'docs/plans/artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    state = read(CAMPAIGN / 'forward-reverse-state.json')
    accounting = read(CAMPAIGN / 'state.json')
    assert state['status'] == 'complete' and accounting['active'] is None
    source = Path(accounting['source'])
    spec = importlib.util.spec_from_file_location('criteria', source / 'bayesfilter/testing/neutra_forward_reverse_criteria.py')
    criteria = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(criteria)
    errors, summaries, sources = [], [], set()
    hashes, probes, source_hashes = 0, 0, 0
    teacher_count = fit_count = clipped = 0
    costs = {role: dict(gpu_process_seconds=0., cpu_core_seconds=0., worker_wall_seconds=0.)
             for role in ('calibration', 'fixed', 'random')}

    def check(condition, message):
        if not condition:
            errors.append(message)

    for row in state['completed']:
        folder = Path(row['output'])
        manifest = read(folder / 'manifest.json')
        result = read(folder / 'result.json')
        label = row['job']
        check(row['status'] == manifest['status'] == 'complete', label + ': incomplete worker')
        for name, expected in manifest['artifact_sha256'].items():
            check(digest(folder / name) == expected, label + ': changed ' + name)
            hashes += 1
        sources.add(Path(manifest['source']))
        check(manifest['jit_compile'] is True, label + ': XLA absent')
        check(manifest['target_signature'] == result['target_signature'], label + ': target mismatch')
        role = row['role']
        if role == 'random' and row['phase'] == 'forward_reverse_fit':
            check(result.get('criterion') == criteria.PAIR_CRITERION,
                  label + ': random result did not use v2 prospectively')
        for key in ('gpu_process_seconds', 'cpu_core_seconds'):
            costs[role][key] += row[key]
        costs[role]['worker_wall_seconds'] += row['wall_seconds']
        if row['phase'] == 'forward_reverse_teacher':
            teacher_count += 1
            check(manifest['gpu_devices_intentionally_hidden'] is True and manifest['gpu'] == -1,
                  label + ': CPU teacher device provenance')
            check(result['reference_training_leakage'] is False, label + ': reference leakage')
            check(result['cpu_workers'] == 2, label + ': CPU worker count')
            for bank, screen in result.get('screens', {}).items():
                expected = bool(screen['finite'] and screen['ess_fraction'] >= .2 and
                    screen['summary_z_max'] <= 5 and screen['maximum_responsibility_discrepancy'] <= .15)
                check(screen['passed'] == expected, label + ': inconsistent teacher screen ' + bank)
            if result['status'] != 'teacher_passed':
                summaries.append(dict(target=row['target'], seed=row['seed'], role=role,
                    teacher_status=result['status'], warm_start_passed=None,
                    final_passed=None, pair_passed=False))
            continue
        fit_count += 1
        policy = manifest['memory_policy']
        check(manifest['TF_FORCE_GPU_ALLOW_GROWTH'] == 'true' and
              policy['all_physical_devices_memory_growth'] and
              policy['configured_before_logical_device_initialization'], label + ': memory growth')
        check(manifest['gpu'] == 2, label + ': physical GPU assignment')
        specification = read(folder / 'spec.json')['specification']
        view = criteria.assess_pair_criteria(result, len(specification['weights']))
        for block in result['training']:
            check(block['finite'] and block['batch_size'] > 1 and block['jit_compile'] and
                  'GPU' in block['device'] and not block['samplewise_loop'], label + ': training validity')
            clipped += block['clipped_updates']
        for endpoint in [result['forward']] + [b['endpoint'] for b in result['branches']]:
            probe = read(endpoint['probe_path'])
            probe_valid = (probe['complete'] and probe['finite'] and probe['rows'] ==
                           probe['valid_rows'] == 1000 and probe['jit_compile'])
            check(probe_valid, label + ': missing/invalid probe')
            probes += 1
            h = endpoint['heldout']
            expected = bool(h['finite'] and h['cross_entropy_delta'] <= .25 and
                h['summary_z_max'] is not None and h['summary_z_max'] <= 5 and
                h['maximum_responsibility_discrepancy'] <= .15 and probe_valid)
            check(expected == endpoint['passed'], label + ': inconsistent endpoint screen')
        selected = state['selection']['reverse']
        branch = next(b for b in view['branches'] if b['rate'] == selected['rate'] and
                      b['updates'] == selected['updates'])
        if role != 'calibration':
            check(len(view['branches']) == 1, label + ': holdout retuned')
            check(view['profile']['forward_updates'] == [8192, 8192] and
                  view['profile']['forward_rates'] == [.001, .0003], label + ': forward recipe changed')
        forward, final = view['forward'], branch['endpoint']
        metrics = {}
        for name, endpoint in [('forward', forward), ('final', final)]:
            probe = read(endpoint['probe_path'])
            h = endpoint['heldout']
            metrics[name] = dict(full_screen_passed=endpoint['passed'],
                feature_z=h['summary_z_max'], mass_error=h['maximum_responsibility_discrepancy'],
                forward_kl_estimate=h['forward_kl_estimate'],
                score_residual_norm={k:probe['score_residual_norm'][k]
                    for k in ('median', 'mean', 'p95', 'p99', 'max')})
        summaries.append(dict(target=row['target'], seed=row['seed'], role=role,
            warm_start_passed=forward['warm_start']['passed'],
            minimum_component_mass_ratio=min(forward['warm_start']['component_mass_ratios']),
            final_passed=final['passed'], pair_passed=branch['passed'], **metrics))

    for source_file in sources:
        for name, expected in read(source_file)['sha256'].items():
            check(digest(source_file.parent / name) == expected, str(source_file.parent) + ': changed ' + name)
            source_hashes += 1
    for phase in ('fixed', 'random'):
        recorded = {(r['target'], r['seed']): r['passed'] for r in state[phase + '_results']}
        actual = {(r['target'], r['seed']): r['pair_passed'] for r in summaries if r['role'] == phase}
        check(recorded == actual, phase + ': controller/result mismatch or skipped fit')
    revision = read(CAMPAIGN / 'forward-reverse-criterion-revision-v2.json')
    for original in revision['source_results']:
        check(digest(original['result_file']) == original['sha256'],
              original['job'] + ': original result changed after criterion revision')
    check(revision['retrospective'] is True, 'fixed reclassification lost retrospective label')
    check(revision['frozen_reverse_recipe'] == state['selection']['reverse'],
          'frozen reverse recipe changed after criterion revision')
    # Exercise the frozen controller on a copy of the completed state. Every
    # worker reference still points to its original, hashed evidence. Any
    # attempted execution or nonterminal state transition is a test failure.
    sys.path.insert(0, str(source))
    controller_spec = importlib.util.spec_from_file_location(
        'frozen_forward_reverse_controller', source / 'scripts/neutra_forward_reverse_campaign.py')
    controller_module = importlib.util.module_from_spec(controller_spec)
    controller_spec.loader.exec_module(controller_module)
    original_state_hash = digest(CAMPAIGN / 'forward-reverse-state.json')
    with tempfile.TemporaryDirectory(prefix='neutra-terminal-resume-') as temporary:
        temporary = Path(temporary)
        (temporary / 'forward-reverse-state.json').write_text(json.dumps(state))

        class TerminalController:
            def __init__(self):
                self.state = {'attempts': [{'output': str(temporary / 'unused-output')}]}
                self.execute_calls = 0

            def remaining(self):
                return accounting['remaining'].copy()

            def execute(self, *args, **kwargs):
                self.execute_calls += 1
                raise AssertionError('terminal resume attempted worker execution')

            def sync(self, *args, **kwargs):
                raise AssertionError('terminal resume attempted a state transition')

        controller = TerminalController()
        return_code = controller_module.run_campaign(controller)
        check(return_code == 0 and controller.execute_calls == 0, 'terminal resume not idle')
    check(digest(CAMPAIGN / 'forward-reverse-state.json') == original_state_hash,
          'terminal resume check changed original state')
    for key, value in accounting['remaining'].items():
        check(math.isfinite(value) and value >= 0, 'budget exhausted: ' + key)
    report = dict(status='passed' if not errors else 'failed',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        criterion=state['criterion'], workers=len(state['completed']),
        teachers=teacher_count, fits=fit_count, artifact_hashes_checked=hashes,
        source_hashes_checked=source_hashes, probes_checked=probes,
        preserved_original_result_hashes_checked=len(revision['source_results']),
        audit_command='python3 ' + str(Path(__file__).resolve()),
        audit_source_sha256=digest(Path(__file__)),
        terminal_resume_check={'scope': 'frozen controller with copied completed state and real saved evidence',
                               'return_code': return_code, 'worker_launches': controller.execute_calls,
                               'original_state_unchanged': digest(CAMPAIGN / 'forward-reverse-state.json') == original_state_hash},
        clipped_updates=clipped, errors=errors, outcomes=summaries, costs=costs,
        remaining=accounting['remaining'], scientific_promotion=False,
        interpretation='fixed reassessment retrospective; random v2 screen prospective; no HMC or method ranking',
        criterion_revision_path=str(CAMPAIGN / 'forward-reverse-criterion-revision-v2.json'),
        criterion_revision_sha256=digest(CAMPAIGN / 'forward-reverse-criterion-revision-v2.json'))
    output = CAMPAIGN / 'forward-reverse-terminal-audit-r2.json'
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('outcomes',)}, indent=2))
    for item in summaries:
        if item['role'] == 'random':
            print(json.dumps(item))
    assert not errors, errors


if __name__ == '__main__':
    main()
