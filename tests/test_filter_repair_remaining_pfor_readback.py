"""Bind remaining-pfor results to current sources and preserve broader debt."""

import ast
import hashlib
import json
from pathlib import Path

from scripts.enforce_filter_gradient_policy import inspect_source
from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
SOURCES = ('bayesfilter/highdim/ledh_contract_e_streaming_tf.py',
           'bayesfilter/highdim/ledh_contract_e_reset_tf.py',
           'experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py',
           'bayesfilter/inference/batched_value_score.py',
           'bayesfilter/highdim/sir_latent_preclip_reference_tf.py',
           'bayesfilter/highdim/models.py', 'bayesfilter/highdim/sir_latent_preclip_tf.py')


def _write(request, name, result):
    path = Path(request.config.getoption('xmlpath')).parent / name
    path.write_text(json.dumps(result, indent=2) + '\n')


def test_current_repaired_sites_and_preserved_failures(request):
    for number, count in ((4711, 1), (4712, 1), (4715, 2)):
        run = _load(number, 'run.json')
        assert run['state'] == 'failed' and run['test_evidence']['failure'] == count
    reports = []
    for number, count in ((4713, 1), (4714, 1), (4716, 28), (4717, 2), (4718, 1)):
        run = _load(number, 'run.json')
        assert run['state'] == 'passed'
        assert run['test_evidence'] == {'passed': True, 'tests': count, 'failure': 0, 'error': 0, 'skipped': 0}
        for relative in SOURCES:
            assert run['source_sha256'][relative] == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), relative
        provenance = _provenance(number)
        assert provenance['tf32_enabled'] is True
        assert provenance['cuda_visible_devices'] == run['environment']['CUDA_VISIBLE_DEVICES']
        policy = provenance['gpu_memory_policy']
        assert policy['all_physical_devices_memory_growth'] and policy['configured_before_logical_device_initialization']
        if run['device'] == 'GPU':
            assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
            assert len(policy['physical_devices']) == 1
        else:
            assert provenance['cuda_visible_devices'] == '-1'
        reports.append({'run': number, 'tests': count, 'device': run['device']})
    derivative_errors = []
    for number in (4713, 4714):
        record = _load(number, 'remaining-pfor-contract-e.json')
        assert record['seed'] == 81101 and record['shape'] == {'B': 2, 'N': 8, 'd': 2, 'P': 3}
        assert record['chunk_policy'] == 'K=N=8'
        assert record['graph']['trace_count'] == 1 and record['graph']['no_pfor_or_host_callback']
        for row in record['records']:
            assert all(row['actual']['finite']) and all(row['actual']['factor_positive'])
            assert abs(row['actual']['source_direct_pair'][0]) > 1e-8
            assert abs(row['actual']['weight_direct_pair'][1]) > 1e-8
            derivative_errors.extend(item['max_absolute_error'] for item in row['finite_differences'])
        hlo = (RAW / f'run-{number:05d}/remaining-pfor-contract-e.hlo.txt').read_bytes()
        assert hashlib.sha256(hlo).hexdigest() == record['graph']['hlo_sha256']
    assert max(derivative_errors) < 2e-6
    for number in (4716, 4717):
        for rank in (2, 3):
            record = _load(number, f'remaining-pfor-transport-{rank}.json')
            assert record['default_batch_native_gate_preserved']
            assert record['graph']['trace_count'] == 1 and record['graph']['jit_compile']
            hlo = (RAW / f'run-{number:05d}/remaining-pfor-transport-{rank}.hlo.txt').read_bytes()
            assert hashlib.sha256(hlo).hexdigest() == record['graph']['hlo_sha256']
    scout = _load(4718, 'remaining-pfor-scout.json')
    assert len(scout['jacobians']) == 2 and scout['grid_count'] == 3
    _write(request, 'remaining-pfor-readback.json', {'runs': reports,
        'maximum_finite_difference_error': max(derivative_errors),
        'remaining_library_sites_repaired': 5, 'whole_F14_or_master_closed': False})


def test_pfor_discovery_has_no_unclassified_library_site(request):
    catalog = json.loads((ROOT / 'docs/plans/filter_gradient_remaining_pfor_discovery_20260929.json').read_text())
    current = []
    for relative in catalog['roots']:
        for path in sorted((ROOT / relative).rglob('*.py')):
            source = path.read_text()
            for row in inspect_source(path.relative_to(ROOT).as_posix(), source):
                if 'pfor' in row['kind']:
                    row['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
                    current.append(row)
    fields = ('path', 'function', 'kind', 'ast_sha256', 'source_sha256')
    dispositions = json.loads((ROOT / 'docs/plans/filter_gradient_pfor_runner_dispositions_20260929.json').read_text())
    expected = {tuple(row[field] for field in fields) for row in catalog['sites']
                if row['path'].startswith('bayesfilter/')}
    assert {tuple(row[field] for field in fields) for row in current} == expected
    assert dispositions['site_count'] == 28 and len(dispositions['rows']) == 17
    baseline = [site for row in dispositions['rows'] for site in row['baseline_sites']]
    assert {tuple(row[field] for field in fields) for row in baseline} == {
        tuple(row[field] for field in fields) for row in catalog['sites']
        if not row['path'].startswith('bayesfilter/')}
    for row in dispositions['rows']:
        assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['current_source_sha256']
    library = [row for row in current if row['path'].startswith('bayesfilter/')]
    assert len(library) == 1 and library[0]['function'] == 'ValidationTarget.log_density'
    path = ROOT / library[0]['path']
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ValidationTarget')
    custom = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'jacobian')
    # This is an explicit log-coordinate-Jacobian formula, not a TensorFlow
    # derivative API. Verify its implementation as well as the calling site.
    calls = [ast.unparse(node.func) for node in ast.walk(custom) if isinstance(node, ast.Call)]
    assert not any(name.endswith(('jacobian', 'batch_jacobian', 'GradientTape', 'vectorized_map')) for name in calls)
    assert 'tf.nn.log_softmax' in calls and 'tf.nn.softplus' in calls
    assert len(current) - len(library) == 0
    _write(request, 'remaining-pfor-discovery-readback.json', {
        'verified_sites': len(current), 'custom_library_log_jacobian': library,
        'runner_benchmark_sites_with_explicit_execution_disposition': 28,
        'runner_benchmark_sites_pending_individual_disposition': 0,
        'whole_F14_or_master_closed': False})
