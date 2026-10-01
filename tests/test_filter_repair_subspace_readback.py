"""Diagnostic readback of source-bound subspace refusal and no-fire evidence."""

import hashlib
import json
from pathlib import Path

from tests.test_filter_repair_geometry_control import save

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
DEPENDENCIES = tuple(f'bayesfilter/inference/{name}.py' for name in (
    'fixed_center_curvature', 'fixed_center_stability_tf', 'fixed_center_fitting_tf',
    'fixed_center_selection_tf', 'dense_validated_fit_tf', 'factor_correlation_geometry',
    'mass_matrix_tf'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_subspace_current_source_evidence(request):
    groups = {f'subspace_identifiability_{part}_{device}'
        for part in ('controls', 'fit') for device in ('cpu', 'gpu')}
    found, failures = {}, []
    for manifest_path in sorted(RAW.glob('run-*/run.json')):
        if int(manifest_path.parent.name[4:]) <= 4930:
            continue
        manifest = json.loads(manifest_path.read_text())
        group = manifest['key'][1]
        if group not in groups:
            continue
        if manifest['state'] != 'passed':
            failures.append({'run': manifest_path.parent.name, 'state': manifest['state']})
            continue
        if all(manifest['source_sha256'][path] == digest(ROOT/path) for path in DEPENDENCIES):
            found[group] = (manifest_path.parent, manifest)
    assert set(found) == groups
    evidence = []
    for group in sorted(groups):
        directory, manifest = found[group]
        gpu = group.endswith('_gpu')
        assert manifest['device'] == ('GPU' if gpu else 'CPU')
        provenance = next(json.loads(line) for line in (directory/'process.log').read_text().splitlines()
            if line.startswith('{"tensorflow_version":'))
        assert provenance['gpu_memory_policy']['all_physical_devices_memory_growth']
        assert provenance['gpu_memory_policy']['configured_before_logical_device_initialization']
        if gpu:
            assert provenance['gpu_memory_policy']['physical_devices']
            assert manifest['environment']['CUDA_VISIBLE_DEVICES'] != '-1'
        else:
            assert manifest['environment']['CUDA_VISIBLE_DEVICES'] == '-1'
        if '_controls_' in group:
            names = ('subspace-identifiability-controls.json', 'subspace-identifiability-precedence.json',
                'subspace-identifiability-enclosing.json', 'installed-angle.json')
            controls = json.loads((directory/names[0]).read_text())
            assert len(controls['cases']) == 84
            assert all(row['nonangle_metrics_preserved'] for row in controls['cases'])
            enclosing = json.loads((directory/names[2]).read_text())
            assert enclosing['raw']['fit_error_code'] == 5 and not enclosing['raw']['usable']
            assert enclosing['public']['error']['message'] == (
                'principal subspace is not numerically resolved at the requested rank')
        else:
            names = ('subspace-identifiability-saved-fit.json', 'dz5-exact-saved-fit.json')
            result = json.loads((directory/names[0]).read_text())
            assert not result['complete_record_differences'] and not result['raw_differences']
            assert result['fit_error_code'] == 0
            assert result['selected_family'] == 'consensus_diagonal_consensus'
        evidence.append({'group': group, 'run': directory.name,
            'manifest_sha256': digest(directory/'run.json'),
            'artifact_sha256': {name: digest(directory/name) for name in names},
            'gpu_memory_policy': provenance['gpu_memory_policy'],
            'cuda_visible_devices': manifest['environment']['CUDA_VISIBLE_DEVICES']})
    save(request, 'subspace-identifiability-terminal.json', {'baseline': '0bfae62f2',
        'source_sha256': {path: digest(ROOT/path) for path in DEPENDENCIES},
        'evidence': evidence, 'preserved_failed_runs': failures,
        'refusal_and_healthy_preservation_qualified': True,
        'nonclaims': ['Resolution refusal does not establish exact eigenvalue multiplicity.',
            'Unselected optimizer differences and terminal memory/cost obligations remain.',
            'No canonical LEDH, HMC, posterior or whole-program admission.']})
