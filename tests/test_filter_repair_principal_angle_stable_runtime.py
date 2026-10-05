"""Independent angle and full saved-fit checks of the installed shared repair."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as current
from tests.test_filter_repair_dz5_exact_fit import (
    test_exact_saved_fit_inputs as execute_saved_fit,
)
from tests.test_filter_repair_dz5_initializer_fit_localization import differences
from tests.test_filter_repair_gaussian_binding_readback import runtime_receipt
from tests.test_filter_repair_principal_angle_stable import (
    BASELINE,
    FIXTURE,
    FIXTURE_SHA,
    RAW,
    ROOT,
    D,
    graph,
    high_precision,
    program,
    save,
    synchronize,
)


def test_installed_angle_accuracy(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    fixture = json.loads(FIXTURE.read_text())
    first, second, rank = fixture['first_precision'], fixture['second_precision'], fixture['rank']
    authority = high_precision(first, second, rank)
    expected = np.asarray(authority['angles'])
    args = tf.constant(first, D), tf.constant(second, D), tf.constant(rank, tf.int32)
    records = {}
    for jit in (True, False):
        owner = program(current, len(first), jit=jit)
        output = synchronize(owner(*args))
        np.testing.assert_allclose(output[3][:rank], expected, atol=1e-10, rtol=1e-10)
        records[str(jit)] = {'angles': output[3][:rank].tolist(),
            'error': np.abs(output[3][:rank]-expected).tolist(), 'jit_compile': jit,
            'nonjit_reference_exception': not jit, 'output_device': owner(*args)[3].device}
        if jit:
            records[str(jit)]['graph'] = graph(owner, args, request, 'installed-angle')
    np.testing.assert_allclose(records['True']['angles'], records['False']['angles'], atol=1e-10, rtol=1e-10)
    save(request, 'installed-angle.json', {'baseline': BASELINE, 'fixture_sha256': FIXTURE_SHA,
        'source_sha256': hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        'authority': authority, 'records': records})


def test_frozen_diagnostic_builder(request):
    """Keep the pre-repair diagnostic reproducible after installing the repair."""
    from tests.test_filter_repair_principal_angle_stable import candidate_module
    candidate, source_hash, candidate_hash, baseline = candidate_module(request)
    for number in (4914, 4915):
        record = json.loads((RAW/f'run-{number:05d}/stable-angle-candidate.json').read_text())
        assert record['function_source_sha256'] == source_hash
        assert record['candidate_source_sha256'] == candidate_hash
    args = (tf.linalg.diag(tf.constant([1., 2., 4.], D)),
            tf.linalg.diag(tf.constant([1., 2., 4.], D)), tf.constant(2))
    actual = synchronize(program(current, 3)(*args))
    rebuilt = synchronize(program(candidate, 3)(*args))
    previous = synchronize(program(baseline, 3)(*args))
    for a, b, c in zip(actual, rebuilt, previous, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
    save(request, 'stable-angle-frozen-builder.json', {'baseline': BASELINE,
        'original_function_sha256': source_hash, 'candidate_function_sha256': candidate_hash,
        'matches_both_saved_probes': True, 'known_geometry_matches_installed_and_baseline': True})


def test_complete_saved_fit(request):
    execute_saved_fit('after', request)
    directory = Path(request.config.getoption('xmlpath')).parent
    payload = json.loads((directory/'dz5-exact-saved-fit.json').read_text())
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    original_path = RAW/f'run-{4613 if gpu else 4614:05d}/dz5-exact-saved-fit.json'
    original = json.loads(original_path.read_text())
    assert payload['fit_numerically_usable'] and payload['fit_error_code'] == 0
    assert payload['operand_sha256'] == original['operand_sha256']
    assert payload['archive_sha256'] == original['archive_sha256']
    assert payload['recipe_sha256'] == original['recipe_sha256']
    changed = differences(payload['result'], original['result'])
    nonangle = [x for x in changed if 'principal_angle' not in x['path']]
    records = []
    fits = payload['result']['fits']
    for family, stability in payload['result']['diagnostics']['selection']['stability'].items():
        for pair in stability['comparisons']:
            left = next(f for f in fits if f['family'] == family and f['replicate_index'] == pair['left_replicate'])
            right = next(f for f in fits if f['family'] == family and f['replicate_index'] == pair['right_replicate'])
            rank = pair['metrics']['positive_subspace_rank']
            # Comparison authority receives the same symmetric matrices as the
            # existing caller, not a separately regenerated fitted precision.
            a, b = np.asarray(left['precision_z']), np.asarray(right['precision_z'])
            authority = high_precision(((a+a.T)/2).tolist(), ((b+b.T)/2).tolist(), rank)
            actual = np.asarray(pair['metrics']['principal_angles_degrees'])
            expected = np.asarray(authority['angles'])
            bound = 1e-10+1e-10*np.abs(expected)
            error = np.abs(actual-expected)
            records.append({'family': family, 'left': pair['left_replicate'], 'right': pair['right_replicate'],
                'authority': authority, 'angles': actual.tolist(), 'errors': error.tolist(),
                'passed': bool(np.all(error <= bound)), 'checks': pair['checks']})
    save(request, 'installed-angle-complete-fit.json', {'baseline': BASELINE,
        'source_sha256': hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        'original_result_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
        'original_run': original_path.parent.name, 'same_operand_hashes': True,
        'strict_original_differences': changed, 'nonangle_differences': nonangle,
        'independent_pairs': records, 'selected_family': payload['result']['selected_family'],
        'accepted': payload['result']['accepted'], 'status': payload['result']['status']})
    assert not nonangle, nonangle[:5]
    assert records and all(r['passed'] for r in records)
    for key in ('selected_family', 'accepted', 'status'):
        assert payload['result'][key] == original['result'][key]


def test_saved_installed_evidence(request):
    found = {}
    for path in sorted(RAW.glob('run-*/run.json')):
        if int(path.parent.name[4:]) <= 4916:
            continue
        manifest = json.loads(path.read_text())
        group = manifest['key'][1]
        if not group.startswith(('principal_angle_stable_runtime_', 'principal_angle_stable_fit_')):
            continue
        assert manifest['state'] == 'passed' and manifest['test_evidence']['passed']
        assert group not in found
        found[group] = path.parent, manifest
    evidence = {}
    for device in ('cpu', 'gpu'):
        for part in ('runtime', 'fit'):
            group = f'principal_angle_stable_{part}_{device}'
            directory, manifest = found[group]
            name = 'installed-angle.json' if part == 'runtime' else 'installed-angle-complete-fit.json'
            path = directory/name
            record = json.loads(path.read_text())
            assert record['source_sha256'] == hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest()
            for source, digest in manifest['source_sha256'].items():
                if source.startswith('bayesfilter/'):
                    assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
            if part == 'runtime':
                assert manifest['test_evidence']['tests'] == 49
            else:
                assert record['same_operand_hashes'] and not record['nonangle_differences']
                assert all(pair['passed'] for pair in record['independent_pairs'])
            evidence[group] = {'run': directory.name, 'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'record': record, 'runtime': runtime_receipt(directory, manifest)}
    cpu = json.loads((found['principal_angle_stable_fit_cpu'][0]/'dz5-exact-saved-fit.json').read_text())
    gpu = json.loads((found['principal_angle_stable_fit_gpu'][0]/'dz5-exact-saved-fit.json').read_text())
    residual = differences(gpu['result'], cpu['result'])
    save(request, 'installed-angle-terminal.json', {'evidence': evidence,
        'remaining_CPU_GPU_record_differences': residual, 'remaining_count': len(residual),
        'remaining_angle_count': sum('principal_angle' in x['path'] for x in residual),
        'nonclaims': ['Unselected fit trajectories and full-record cross-backend equivalence remain open.',
            'No terminal cost/capacity or scientific admission is inferred.']})
