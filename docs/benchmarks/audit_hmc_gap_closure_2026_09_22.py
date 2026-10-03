"""Read-only M25 and terminal M21 evidence audit; never modifies frozen runs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sources(root, manifest):
    payload = read(manifest)
    for relative, expected in payload['files'].items():
        assert digest(root / relative) == expected, relative
    return {'identity': payload['source_identity'], 'checked_files': len(payload['files']),
            'manifest_sha256': digest(manifest)}


def tensors(root):
    count = 0
    for meta_path in root.rglob('*.tensor.json'):
        meta = read(meta_path)
        assert digest(Path(str(meta_path)[:-5])) == meta['sha256'], str(meta_path)
        count += 1
    return count


def binomial(count, total):
    # Independent post-run reference; numerical implementation is not a tuner.
    from scipy.stats import beta
    return [float(beta.ppf(.025, count, total-count+1)) if count else 0.,
            float(beta.ppf(.975, count+1, total-count)) if count < total else 1.]


def controller(root, source_id, repair_source_id):
    all_streams = set()
    cases = {}
    cells = [('slow_stationary', 'controller-confirmation-slow-r1', 2026092262, source_id),
             ('dispersed', 'controller-confirmation-dispersed-r1', 2026092262, source_id),
             ('dispersed_repair', 'controller-repair-confirmation-r1', 2026092265, repair_source_id)]
    for case, name, seed_root, expected_source in cells:
        directory = root / name / 'worker'
        summary = read(directory / 'summary.json')
        manifest = summary['manifest']
        assert summary['complete'] and summary['planned_replications'] == 400
        assert manifest['source_identity'] == expected_source
        assert manifest['case'] == ('dispersed' if case == 'dispersed_repair' else case)
        assert manifest['method'] == 'autocorrelation'
        assert manifest['root_seed'] == seed_root
        fixed_count = manifest.get('counts', {}).get('fixed_retained', 10000)
        if case == 'dispersed_repair':
            assert manifest['counts'] == read(root/'controller-repair-counts.json')
            assert manifest['max_results_per_chain'] == 80000
        rows = summary['rows']
        assert [row['rep'] for row in rows] == list(range(400))
        delivered = warmup_caps = retained_caps = 0
        unavailable = covered = fixed_covered = 0
        retained_reasons = Counter()
        for row in rows:
            trial = directory / f"rep-{row['rep']:04d}"
            assert row == read(trial / 'result.json')
            assert row['status'] == 'complete' and not row['hard_vetoes']
            for seed in row['streams'].values():
                seed = tuple(seed)
                assert seed not in all_streams, ('duplicate_stream', seed)
                all_streams.add(seed)
            result = read(trial / 'controller.json')
            assert result['passed'] == row['passed']
            assert result['warmup_excluded_from_posterior']
            stopped = row['stopped']
            fixed = row['fixed']
            assert stopped['method'] == fixed['method'] == 'autocorrelation_normal_95'
            assert fixed['available']
            if stopped['available']:
                last = result['retained_checks'][-1]
                assessment = last[last['diagnostic_role']]
                target = assessment['precision']['targets'][0]
                for field in ('estimate', 'mcse', 'estimator'):
                    assert stopped[field] == target[field], (case, row['rep'], field)
                assert stopped['covered'] == (abs(stopped['estimate']) <= 1.959963984540054 * stopped['mcse'])
                if row['retained_cap']:
                    retained_reasons['rhat_failed'] += not assessment['modern_rhat']['passed']
                    retained_reasons['precision_failed'] += not assessment['precision']['passed']
            else:
                assert row['retained_count'] == 0
            assert read(trial / 'fixed-retained.tensor.json')['shape'] == [fixed_count, 4, 1]
            rho = manifest['rho']
            variance = (fixed_count + 2*math.fsum((fixed_count-k)*rho**k
                        for k in range(1,fixed_count))) / (4*fixed_count**2)
            assert math.isclose(row['fixed_exact_law']['variance'], variance, rel_tol=1e-12)
            delivered += row['passed']
            warmup_caps += row['warmup_cap']
            retained_caps += row['retained_cap']
            unavailable += not stopped['available']
            covered += stopped['available'] and stopped['covered']
            fixed_covered += fixed['covered']
        calibration = summary['calibration']
        assert delivered == calibration['posterior_checks_passed']['count']
        assert covered == calibration['intervals']['stopped']['unconditional']['count']
        assert fixed_covered == calibration['intervals']['fixed']['unconditional']['count']
        assert unavailable == calibration['intervals']['stopped']['unavailable']
        assert warmup_caps == calibration['warmup_cap_count']
        assert retained_caps == calibration['retained_cap_count']
        assert calibration['oracle']['screen_passed'], 'invalid exact-oracle screen'
        cases[case] = {'planned': 400, 'completed': len(rows), 'delivered': delivered,
            'delivery_interval': binomial(delivered, 400), 'warmup_caps': warmup_caps,
            'retained_caps': retained_caps, 'retained_cap_reasons_nonexclusive': dict(retained_reasons),
            'stopped_covered_all_planned': covered, 'stopped_unavailable': unavailable,
            'fixed_covered': fixed_covered, 'calibration': calibration,
            'summary_sha256': digest(directory/'summary.json'), 'tensor_checksums': tensors(directory)}
    return {'cases': cases, 'distinct_root_streams': len(all_streams)}


def terminal_public_groups(master):
    queue = read(master / 'm21-r2/confirmation-queue-progress.json')
    result = {}
    for key, task in queue['tasks'].items():
        if not key.startswith('m21-merged-confirmation-'):
            continue
        root = Path(task['output']) / key
        # A worker may finish every fit and persist its final output before
        # expensive framework teardown. Preserve process and evidence status
        # separately rather than ignoring the completed inventory.
        if not (root/'attempt-001-result.json').exists():
            continue
        final = read(root/'attempt-001-result.json')
        assert final['execution_status'] == 'complete'
        assert read(root/'result.json') == final
        assessment = read(root / 'assessment.json')
        assert assessment == final['assessment']
        planned = assessment['planned']
        completed = assessment['completed']
        assert completed <= planned
        assert [r['replication'] for r in assessment['replications']] == list(range(completed))
        verified = unassessed = delivered = warmup_caps = retained_caps = health_vetoes = 0
        missing_selected = 0
        receipt_count = evidence_count = 0
        selections = []
        for record in assessment['replications']:
            assert not record['inventory']['failures']
            pipeline = read(record['pipeline'])
            payload = read(pipeline['tuning_path'])
            body = dict(payload)
            result_hash = body.pop('result_hash')
            assert json_digest(body) == result_hash
            tuning = Path(record['pipeline']).parent / 'tuning'
            checkpoint = read(tuning/'tuning_checkpoint.json')
            checksum = checkpoint.pop('content_hash')
            assert json_digest(checkpoint) == checksum
            numerical_evidence = {}
            for expected in checkpoint['numerical_evidence_hashes']:
                evidence = read(tuning/'numerical_evidence'/f'{expected}.json')
                assert json_digest(evidence) == expected
                numerical_evidence[expected] = evidence
                evidence_count += 1
            candidates = {c['candidate_id']: c for c in payload['candidates']}
            ids = payload['verified_candidate_ids']
            assert payload['nominee_id'] is None
            assert set(ids) == {m['candidate_id'] for m in pipeline['members']}
            passed_receipts = {r['candidate_id'] for r in payload['verification_receipts']
                if r['stage'] == 'verification' and r['decision'] in ('passed', 'acceptance_in_band')
                and r.get('evidence_validity', 'valid') == 'valid'
                and not r.get('hard_vetoes') and not r.get('promotion_vetoes')}
            assert passed_receipts == set(ids)
            local_seeds = set()
            for rec in payload['verification_receipts']:
                candidate = candidates[rec['candidate_id']]
                assert rec['epsilon'] == candidate['epsilon']
                assert rec['exact_l'] == candidate['leapfrog_steps']
                evidence = numerical_evidence[rec['numerical_evidence_hash']]
                assert evidence['candidate'] == candidate and evidence['seed'] == rec['seed_lineage']
                assert evidence['work']['stage'] == rec['stage']
                seed = tuple(rec['seed_lineage'])
                assert seed not in local_seeds
                local_seeds.add(seed)
                receipt_count += 1
            members = [m for m in pipeline['members'] if m['status'] == 'assessed']
            verified += len(ids)
            unassessed += sum(m['status'] == 'unassessed_by_design' for m in pipeline['members'])
            if not members:
                missing_selected += 1
                selections.append({'replication': record['replication'], 'status': 'no_assessed_selected_member'})
                continue
            assert len(members) == 1
            member = members[0]
            selected = min(c for c in ids if candidates[c]['leapfrog_steps'] == 3)
            assert member['candidate_id'] == selected
            assert member['fixed_comparator']['status'] == 'assessed'
            assert member['warmup_exclusion_matches'] and member['duplicate_chains'] is not True
            post = member['posterior']
            assert post['warmup_excluded_from_posterior']
            assert read(member['draws_path']+'.json')['shape'][0] == post['retained_results_per_chain']
            delivered += post['passed']
            warmup_caps += post['warmup_cap_hit']
            retained_caps += post['retained_cap_hit']
            health_vetoes += bool(post['hard_vetoes'])
            selections.append({'replication': record['replication'], 'candidate_id': selected,
                'epsilon': member['epsilon'], 'L': member['L'], 'passed': post['passed'],
                'warmup_cap': post['warmup_cap_hit'], 'retained_cap': post['retained_cap_hit']})
        assert verified == assessment['verified_members']
        assert unassessed == assessment['unassessed_by_design_members']
        result[key] = {'planned': planned, 'completed': completed, 'verified': verified,
            'process_status_at_audit': task['status'],
            'complete_final_result_written': True,
            'complete_inventory': completed == planned, 'unstarted': planned-completed,
            'missing_selected_outputs': missing_selected,
            'requested': planned, 'unassessed_siblings': unassessed, 'delivered': delivered,
            'warmup_caps': warmup_caps, 'retained_caps': retained_caps, 'hard_health_vetoes': health_vetoes,
            'verification_receipts_checked': receipt_count, 'tensor_checksums': tensors(root),
            'numerical_evidence_checksums': evidence_count,
            'stopped_coverage': assessment['interval_coverage_at_stop'],
            'stopped_versus_fixed': assessment['stopped_versus_fixed'],
            'assessment_sha256': digest(root/'assessment.json'), 'selections': selections,
            'historical_unavailable_field_includes_unassessed_siblings': True}
    return result


def gpu_maps(root, source_id):
    """Audit saved GPU evidence without initializing a GPU or rerunning HMC."""
    rows, all_seeds = [], set()
    for kind in ('exact', 'partial', 'partial_half'):
        for seed in (2026092251, 2026092252):
            attempt = root / f'gpu-{kind}-{str(seed)[-4:]}-r1'
            execution = read(attempt/'execution.json')
            assert execution['exit_code'] == 0
            launch = read(attempt/'manifest.json')
            assert launch['source_identity'] == source_id and launch['trusted_escalated_launch']
            assert launch['environment']['TF_FORCE_GPU_ALLOW_GROWTH'] == 'true'
            assert digest(Path(launch['command'][1])) == launch['driver_sha256']
            worker = attempt/'worker'
            manifest, result = read(worker/'manifest.json'), read(worker/'result.json')
            assert manifest['device'] == 'gpu' and manifest['jit_compile']
            assert manifest['script_sha256'] == launch['driver_sha256']
            for relative, expected in manifest['source_hashes'].items():
                assert digest(root/'source-r2'/relative) == expected, relative
            memory = manifest['memory_policy']
            assert memory['mode'] == 'memory_growth'
            assert memory['all_physical_devices_memory_growth']
            assert memory['configured_before_logical_device_initialization']
            assert memory['physical_devices'] and all(d['memory_growth'] for d in memory['physical_devices'])
            assert result['status'] == 'complete' and not result['inventory']['failures']
            assert (result['map'], result['seed']) == (kind, seed)
            fit = worker/'fit'
            binding = read(fit/'tuning/execution_spec.json')['execution']
            assert binding['config']['use_xla'] and binding['runtime_policy']['device_type'] == 'GPU'
            assert binding['runtime_policy']['memory_policy'] == memory
            for path, expected in binding['source_closure'].items():
                assert digest(Path(path)) == expected, path
            starts = read(fit/'start_coordinates.json')
            assert starts['supplied_to_tuner'] == 'latent' and starts['forward_roundtrip_checked']
            spec = read(worker/'supplied_map.json')
            assert not spec['construction']['training_performed']
            strength = spec['construction']['strength']
            for z, model, x in zip(starts['latent_starts'], starts['model_starts'], (-1., -.3, .4, 1.)):
                v = 3*z[0]
                s = v/2 if strength is None else 6*math.tanh(strength*v/12)
                mapped = (v, math.exp(s)*z[1], math.exp(s)*z[2])
                assert all(math.isclose(a, x, rel_tol=2e-11, abs_tol=2e-11) for a in (*model, *mapped))
            tuning = read(fit/'tuning/candidate_set_result.json')
            body = dict(tuning)
            assert body.pop('result_hash') == json_digest(body)
            checkpoint = read(fit/'tuning/tuning_checkpoint.json')
            assert checkpoint.pop('content_hash') == json_digest(checkpoint)
            for field in ('scope', 'config', 'candidates', 'candidate_states', 'work_items',
                          'verification_receipts', 'verified_candidate_ids', 'completion_status'):
                assert tuning[field] == checkpoint['result'][field]
            candidates = {c['candidate_id']: c for c in tuning['candidates']}
            verified = set(tuning['verified_candidate_ids'])
            assert tuning['nominee_id'] is None
            assert verified == set(result['verified_candidate_ids'])
            assert verified == {cid for cid, state in tuning['candidate_states'].items() if state == 'verified'}
            evidence = {}
            for expected in checkpoint['numerical_evidence_hashes']:
                item = read(fit/'tuning/numerical_evidence'/f'{expected}.json')
                assert json_digest(item) == expected
                assert 'GPU' in item['samples_device']
                assert item['runtime'] and all(r['use_xla'] for r in item['runtime'])
                evidence[expected] = item
            passed = set()
            for receipt in tuning['verification_receipts']:
                candidate = candidates[receipt['candidate_id']]
                stream = tuple(receipt['seed_lineage'])
                assert stream not in all_seeds
                all_seeds.add(stream)
                assert receipt['epsilon'] == candidate['epsilon']
                assert receipt['exact_l'] == candidate['leapfrog_steps']
                item = evidence[receipt['numerical_evidence_hash']]
                assert item['candidate'] == candidate and item['seed'] == receipt['seed_lineage']
                if receipt['stage'] == 'verification' and receipt['promotion_eligible']:
                    assert receipt['decision'] == 'passed' and receipt['evidence_validity'] == 'valid'
                    assert not receipt['hard_vetoes'] and not receipt['promotion_vetoes']
                    passed.add(receipt['candidate_id'])
            assert passed == verified
            if kind == 'exact':
                assert verified, 'exact whitening positive-control tuning failed'
            pipeline = read(fit/'pipeline.json')
            assert {m['candidate_id'] for m in pipeline['members']} == verified
            assert pipeline['selection']['candidate_ids'] == sorted(verified)[:1]
            health = set()
            for member in pipeline['members']:
                if member['status'] != 'assessed':
                    assert member['status'] == 'unassessed_by_design'
                    continue
                post = member['posterior']
                assert member['warmup_exclusion_matches'] and post['warmup_excluded_from_posterior']
                assert post['assessment_role'] == 'posterior_only' and post['config']['jit_compile']
                assert post['config']['step_size'] == member['epsilon']
                assert post['config']['num_leapfrog_steps'] == member['L']
                for check in post['warmup_checks'] + post['retained_checks']:
                    health.update(check['health']['member_health_failures'])
            assert result['unassessed_members'] + len(result['posterior']) == len(verified)
            rows.append({'map': kind, 'seed': seed, 'search': result['search'],
                'candidate_count': len(candidates), 'verified_count': len(verified),
                'unassessed': result['unassessed_members'], 'posterior': result['posterior'],
                'health_failure_details': sorted(health), 'gpu_worker_seconds': execution['gpu_worker_seconds'],
                'numerical_evidence_checksums': len(evidence), 'tensor_checksums': tensors(worker),
                'result_sha256': digest(worker/'result.json')})
    return {'rows': rows, 'distinct_verification_streams': len(all_seeds),
            'role': 'supplied-map engineering; posterior failures preserved; no ranking'}


def tests_and_profiles(root):
    tests, suites = set(), []
    for path in sorted(root.glob('*tests*/tests.xml')):
        suite = ET.parse(path).getroot()
        cases = list(suite.iter('testcase'))
        assert not list(suite.iter('failure')) and not list(suite.iter('error')), str(path)
        for case in cases:
            assert case.find('skipped') is None, str(path)
            tests.add((case.attrib.get('classname'), case.attrib['name']))
        receipt = read(path.parent/'execution.json')
        assert receipt['exit_code'] == 0
        suites.append({'path': str(path), 'sha256': digest(path), 'passed': len(cases)})
    profiles = []
    for arm in ('baseline', 'noop', 'shift'):
        path = root/f'profile-{arm}-r1/worker/result.json'
        result = read(path)
        assert not result['inventory']['failures']
        assert not result['power_established'] and not result['ranking_supported']
        assert result['gpu_intentionally_hidden'] and not result['jit_compile']
        profiles.append({'arm': arm, 'elapsed_seconds': result['elapsed_seconds'],
            'inventory': result['inventory'], 'posterior': result['posterior'],
            'timing': result['timing'], 'unassessed_siblings': result['unassessed_siblings'],
            'result_sha256': digest(path)})
    return {'unique_tests_passed': len(tests), 'test_suites': suites, 'profiles': profiles}


def inactive_budget_parity(root):
    original = root/'controller-pilot-slow-r1/worker'
    repaired = root/'inactive-budget-parity-r1/worker'
    def numerical(value, directory):
        if isinstance(value, dict):
            return {k: numerical(v, directory) for k, v in value.items() if k != 'elapsed_seconds'}
        if isinstance(value, list):
            return [numerical(v, directory) for v in value]
        if isinstance(value, str) and '/worker/' in value:
            return '<worker>/' + value.split('/worker/', 1)[1]
        return value
    for rep in range(2):
        for filename in ('controller.json', 'result.json'):
            relative = Path(f'rep-{rep:04d}')/filename
            assert numerical(read(original/relative), original) == numerical(read(repaired/relative), repaired)
    left = {str(p.relative_to(original)): digest(p) for p in original.rglob('*.tensor')}
    right = {str(p.relative_to(repaired)): digest(p) for p in repaired.rglob('*.tensor')}
    assert left and left == right
    return {'reused_smoke_replications': 2, 'identical_tensor_files': len(left),
            'identical_numerical_reports': True, 'confirmation_evidence': False,
            'excluded_fields': ['elapsed_seconds', 'output_directory_prefix']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--public-only', action='store_true')
    parser.add_argument('--public-audit', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    master = Path(__file__).resolve().parents[1] / 'plans/artifacts/hmc-repair-master-2026-09-16'
    m25 = master / 'm25-r1'
    if args.public_only:
        result = {'m21_source': sources(master/'m21-r2/source-r1', master/'m21-r2/source-manifest-r1.json'),
            'terminal_public_groups': terminal_public_groups(master),
            'elapsed_seconds': time.monotonic()-started}
        assert len(result['terminal_public_groups']) == 4
        with args.output.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps({'elapsed_seconds': result['elapsed_seconds'],
            'terminal_public_groups': list(result['terminal_public_groups'])}))
        return
    if args.public_audit:
        prior = read(args.public_audit)
        public = prior['terminal_public_groups']
        queue = read(master/'m21-r2/confirmation-queue-progress.json')
        assert len(public) == 4
        for name, group in public.items():
            path = Path(queue['tasks'][name]['output'])/name/'assessment.json'
            assert digest(path) == group['assessment_sha256']
    else:
        public = terminal_public_groups(master)
    source = sources(m25/'source-r1', m25/'source-manifest-r1.json')
    repair_source = sources(m25/'source-r2', m25/'source-manifest-r2.json')
    result = {'m25_source': source,
        'm25_repair_source': repair_source,
        'm21_source': sources(master/'m21-r2/source-r1', master/'m21-r2/source-manifest-r1.json'),
        'controller': controller(m25, source['identity'], repair_source['identity']),
        'terminal_public_groups': public,
        'public_audit_sha256': digest(args.public_audit) if args.public_audit else None,
        'gpu_maps': gpu_maps(m25, repair_source['identity']),
        'tests_and_profiles': tests_and_profiles(m25),
        'inactive_count_budget_parity': inactive_budget_parity(m25),
        'scientific_gaps_closed': False, 'default_promoted': False,
        'ranking_supported': False, 'elapsed_seconds': time.monotonic()-started}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'elapsed_seconds': result['elapsed_seconds'],
        'controller_cases': list(result['controller']['cases']),
        'terminal_public_groups': list(result['terminal_public_groups'])}))


if __name__ == '__main__':
    main()
