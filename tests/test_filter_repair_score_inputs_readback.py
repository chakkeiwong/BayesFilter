"""Independent saved-evidence and deferred-scope checks for input execution."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from tests.test_filter_repair_gaussian_binding_readback import runtime_receipt
from tests.test_filter_repair_score_inputs import BASELINE, ROOT, compare, hashes, save

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def saved():
    found, failed = {}, []
    for path in sorted(RAW.glob('run-*/run.json')):
        if int(path.parent.name[4:]) <= 4896:
            continue
        manifest = json.loads(path.read_text())
        group = manifest['key'][1]
        if not group.startswith('score_inputs_') or group == 'score_inputs_readback_cpu':
            continue
        if manifest['state'] != 'passed':
            failed.append({'run': path.parent.name, 'group': group, 'state': manifest['state']})
            continue
        assert group not in found, 'Duplicate passed workers require a disposition'
        assert manifest['test_evidence']['passed']
        for source, digest in manifest['source_sha256'].items():
            if source.startswith('bayesfilter/'):
                assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
        found[group] = (path.parent, manifest)
    return found, failed


def read(directory, manifest, name):
    path = directory/name
    result = json.loads(path.read_text())
    assert result['baseline'] == BASELINE and result['source_sha256'] == hashes()
    assert all(manifest['source_sha256'][p] == h for p, h in hashes().items())
    return result, {'run': directory.name, 'artifact': name,
        'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256((directory/'run.json').read_bytes()).hexdigest(),
        'worker': runtime_receipt(directory, manifest)}


def test_deferred_input_branches_are_preserved_without_execution():
    """Compare the four eager assignments as ASTs; do not execute deferred filters."""
    for name, names in (('adapters', {'initial', 'process', 'uniforms', 'reset_design'}),
                        ('nonlinear_adapter', {'initial', 'process', 'uniforms', 'design'})):
        path = f'bayesfilter/score_study/{name}.py'
        old = ast.parse(subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT, text=True))
        new = ast.parse((ROOT/path).read_text())
        def assignments(tree, names=names):
            return [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in names and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func).startswith('tf.random.stateless_')]
        left, right = assignments(old), assignments(new)
        assert len(left) == len(right) == 4
        assert sorted(ast.dump(n, include_attributes=False) for n in left) == sorted(ast.dump(n, include_attributes=False) for n in right)
        branch = next(n for n in ast.walk(new) if isinstance(n, ast.If)
                      and all(item in n.body for item in right))
        condition = ast.unparse(branch.test)
        if name == 'adapters':
            assert condition == 'deferred_inputs'
            selection = next(n for n in ast.walk(new) if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'deferred_inputs')
            assert ast.unparse(selection.value) == "row['proposal'] in ('iapf', 'integrated_kdm', 'resampling_kdm', 'kdm_covariance') or row.get('collect_kdm_control', False)"
        else:
            assert condition == "proposal in ('iapf', 'kdm_covariance')"
        assert any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'make_score_inputs'
                   for node in branch.orelse for n in ast.walk(node))


def test_saved_input_execution_evidence(request):
    found, failed = saved()
    evidence, maxima, costs, callers = {}, {}, {}, {}
    for device in ('cpu', 'gpu'):
        for dtype in ('float64', 'float32'):
            group = f'score_inputs_primitives_{dtype}_{device}'
            directory, manifest = found[group]
            record, evidence[group] = read(directory, manifest, f'score-inputs-{dtype}.json')
            assert len(record['cases']) == 8
            assert record['explicit_nonjit_reference_checked'] and record['invalid_configuration_rejected']
            maxima[group] = max(case['errors'][i] for case in record['cases'] for i in (0, 1, 3))
            for case in record['cases']:
                assert case['raw_words_exact'] and case['uniforms_exact'] and case['errors'][2] == 0
                assert case['graph']['trace_count'] == 1 and case['graph']['jit_compile']
                assert case['graph']['no_host_callback_or_pfor']
                variant = case['seeds'][0][0]-137
                hlo = directory/f'inputs-{dtype}-{case["d"]}-{case["twist"]}-{variant}.hlo.txt'
                assert hashlib.sha256(hlo.read_bytes()).hexdigest() == case['graph']['hlo_sha256']
        for family in ('gaussian', 'nonlinear', 'gaussian_directions', 'nonlinear_directions'):
            group = f'score_inputs_{family}_{device}'
            directory, manifest = found[group]
            record, evidence[group] = read(directory, manifest, f'score-inputs-public-{family}.json')
            assert record['single_trace'] and record['invalidity_preserved'] and record['deferred_families_not_executed']
            assert len(record['cases']) == {'gaussian': 28, 'nonlinear': 20,
                'gaussian_directions': 4, 'nonlinear_directions': 4}[family]
            errors, refused, labels = [], 0, 0
            for case in record['cases']:
                tolerance = 1e-9 if case['dtype'] == 'float64' else 5e-5
                errors.append(compare(case['live_candidate'], case['live_original'], tolerance))
                assert case['frozen_inputs_exact']
                refused += case['live_candidate']['error'] is not None
                if 'discrete_witness' in case:
                    witness = case['discrete_witness']
                    assert witness['identical_labels'] and witness['instrumented_outputs_exact']
                    labels += 1
                if family.endswith('directions'):
                    assert case['healthy_frozen_record']['error'] is None
            callers[group] = {'cases': len(record['cases']), 'refused': refused,
                'exact_label_witnesses': labels, 'maximum_record_difference': max(errors)}
            expected_checks = 4 if family == 'gaussian_directions' else 5 if family == 'nonlinear_directions' else 1
            if family == 'gaussian_directions' and device == 'gpu':
                expected_checks += 2
            assert manifest['test_evidence']['tests'] == expected_checks
        arms = {}
        for arm in ('before', 'after'):
            group = f'score_inputs_cost_{arm}_{device}'
            directory, manifest = found[group]
            arms[arm], evidence[group] = read(directory, manifest, 'score-inputs-cost.json')
            assert arms[arm]['exact_replay'] and len(arms[arm]['warm_seconds']) == 30
            if device == 'gpu':
                assert manifest['gpu_performance_preflight_uncontended']
                assert 'GPU:0' in arms[arm]['runtime']['value_device']
        before, after = arms['before'], arms['after']
        for field in ('scope', 'row', 'settings', 'seed', 'baseline_adapter_sha256'):
            assert before[field] == after[field]
        error = compare(after['numerical_result'], before['numerical_result'], 1e-9)
        costs[device] = {'maximum_record_difference': error,
            'warm_median_before_ms': before['warm_median_seconds']*1000,
            'warm_median_after_ms': after['warm_median_seconds']*1000,
            'warm_ratio': after['warm_median_seconds']/before['warm_median_seconds'],
            'cold_ratio': after['cold_seconds']/before['cold_seconds'],
            'sampled_after_warm_rss_delta_mib': (after['memory']['after_warm']['smaps_rollup_Rss']-
                before['memory']['after_warm']['smaps_rollup_Rss'])/2**20,
            'before_memory': before['memory'], 'after_memory': after['memory'],
            'status': 'descriptive_only_no_statistical_or_terminal_capacity_acceptance'}
    # Fixture files were read from unchanged tracked baseline bytes. Bind them
    # here rather than claiming the worker manifest originally hashed them.
    fixture_bindings = {}
    for name in ('score_directions', 'nonlinear_untouched'):
        path = f'tests/fixtures/filter_repair_{name}_20260929.json'
        source = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], cwd=ROOT)
        assert source == (ROOT/path).read_bytes()
        assert all(path not in manifest['git_diff_stat'] for _, manifest in found.values())
        fixture_bindings[path] = {'sha256': hashlib.sha256(source).hexdigest(),
            'basis': 'unchanged tracked baseline bytes; bound during readback, not a direct worker hash'}
    siblings = {}
    for device, directory in (('cpu', Path(request.config.getoption('xmlpath')).parent),
                              ('gpu', found['score_inputs_gaussian_directions_gpu'][0])):
        for model in ('gaussian', 'nonlinear_scalar'):
            path = directory/f'fitted-apf-endpoint-{model}.json'
            record = json.loads(path.read_text())
            assert record['passed'] and record['factory_identity_verified']
            siblings[f'{model}_{device}'] = {'run': directory.name,
                'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'scope': record['scope']}
    save(request, 'score-inputs-readback.json', {'baseline': BASELINE, 'source_sha256': hashes(),
        'evidence': evidence, 'primitive_max_abs': maxima, 'callers': callers, 'costs': costs,
        'fixture_bindings': fixture_bindings, 'fixed_fitted_apf_siblings': siblings,
        'preserved_failed_workers': failed,
        'nonclaims': ['iAPF and KDM deferred; canonical LEDH admission excluded.',
                      'This unit does not close whole-program geometry, locator, memory or source-audit gates.']})
