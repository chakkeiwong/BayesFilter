"""Diagnostic import/fixture isolation for the preserved DZ5 candidate snapshot."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SNAPSHOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/'
    'filter-gradient-repair-20260917/dz5-candidate-source-03612')
BF = Path('/home/ubuntu/workspace/BayesFilter')
MF = Path('/home/ubuntu/workspace/MacroFinance-dz5-neutra')

CHILD = r'''
import hashlib
import importlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

BF = Path('/home/ubuntu/workspace/BayesFilter')
MF = Path('/home/ubuntu/workspace/MacroFinance-dz5-neutra')
OUT = Path('/tmp/dz5-output')
sys.path[:0] = [str(MF), str(BF)]
report = {'schema': 'filter_repair_dz5_snapshot_import.v1', 'passed': False,
    'role': 'isolated_CPU_import_and_fixture_reference_only',
    'target_evaluated': False, 'adapter_admitted': False,
    'nonclaims': ['No target, initializer, transition, timing or retained qualification.']}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def diagnostic_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json(item) for item in value]
    return value

try:
    manifest = json.loads(Path('/tmp/dz5-source/manifest.json').read_text())
    report['snapshot_manifest_sha256'] = sha('/tmp/dz5-source/manifest.json')
    expected = {path: entry['sha256'] for path, entry in manifest['sources'].items()}
    assert all(sha(path) == digest for path, digest in expected.items())
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    assert os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] == 'true'
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=False)
    assert not tf.config.list_physical_devices('GPU')
    report.update(gpu_memory_policy=memory, tensorflow=tf.__version__,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])

    from bayesfilter_estimation import audit_sources, declared_source_closure
    from two_currency_double_zlb_credit_estimation import (
        TFP_SPECIAL, TFP_SPECIAL_SHA, credit_fixture_identity)
    from two_currency_double_zlb_credit_fixtures import credit_fixture
    from two_currency_double_zlb_credit_target import CREDIT_FILTER_CONTRACT
    importlib.import_module('bayesfilter_estimation_initialization')
    importlib.import_module('two_currency_double_zlb_credit_recovery')
    importlib.import_module('bayesfilter.inference.dense_initializer_seeded_tf')
    importlib.import_module('bayesfilter.inference.tensor_npz_archive')
    assert sha(TFP_SPECIAL) == TFP_SPECIAL_SHA
    fixture = credit_fixture('CDF', n_steps=96)
    identity = credit_fixture_identity(fixture)
    assert identity == 'e116fe853c8579369036ab2ce57724ba07544524d0920fa0a706d76714f75d8a'
    assert len(fixture.parameter_names) == 23 and fixture.observations.shape[0] == 96
    assert CREDIT_FILTER_CONTRACT == 'rectangular_srukf_full_rank_identity_prepared_cir_v2'
    closure = audit_sources(declared_source_closure((
        MF / 'scripts/prepare_dz5_cdf_proposal.py',
        BF / 'bayesfilter/runtime/runner.py')))
    assert all(expected.get(path) == digest for path, digest in closure.items())
    loaded, unexpected = {}, {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        named = name.startswith(('bayesfilter', 'two_currency_double_zlb'))
        owned = path.is_relative_to(BF) or path.is_relative_to(MF)
        if named or owned:
            digest = sha(path)
            loaded[name] = {'path': str(path), 'sha256': digest}
            if not owned or expected.get(str(path)) != digest:
                unexpected[name] = loaded[name]
    report.update(loaded_modules=loaded, unexpected_modules=unexpected,
        source_closure=closure, fixture_identity=identity,
        parameter_names=fixture.parameter_names, observation_shape=list(fixture.observations.shape),
        filter_contract=CREDIT_FILTER_CONTRACT, tfp_special_sha256=TFP_SPECIAL_SHA)
    assert not unexpected, unexpected
    mounts = {}
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        if fields[4] in (str(BF), str(MF)):
            mounts[fields[4]] = fields[5].split(',')
    report['source_mount_options'] = mounts
    assert set(mounts) == {str(BF), str(MF)}
    assert all('ro' in options for options in mounts.values())
    assert all(sha(path) == digest for path, digest in expected.items())
    report['passed'] = True
except BaseException as error:
    report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
finally:
    (OUT / 'dz5-snapshot-import.json').write_text(json.dumps(diagnostic_json(report), indent=2, allow_nan=False)+'\n')
sys.exit(0 if report['passed'] else 1)
'''


def run_isolated_snapshot(request, *, snapshot, child, scope,
                          child_timeout_seconds=120):
    """Run a diagnostic against read-only project trees, preserving child logs."""
    directory = Path(request.config.getoption('xmlpath')).parent
    child_path = directory / 'isolated-dz5-import.py'
    with child_path.open('x') as stream:
        stream.write(child)
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    # CUDA names its helper threads through /proc/self/task/*/comm. A read-only
    # proc bind makes cuInit fail with EROFS before any target can execute.
    devices = ['--dev-bind', '/dev', '/dev', '--proc', '/proc'] if gpu else ['--dev', '/dev']
    command = ['bwrap', '--die-with-parent', '--ro-bind', '/', '/', *devices, '--tmpfs', '/tmp',
        '--ro-bind', str(snapshot), '/tmp/dz5-source',
        '--bind', str(directory), '/tmp/dz5-output',
        '--ro-bind', str(snapshot / str(MF).lstrip('/')), str(MF),
        '--ro-bind', str(snapshot / str(BF).lstrip('/')), str(BF),
        '--chdir', str(MF), sys.executable, '-B', '/tmp/dz5-output/isolated-dz5-import.py']
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(BF),
        'MPLCONFIGDIR': '/tmp/matplotlib', 'XDG_CACHE_HOME': '/tmp/cache'}
    with (directory / 'isolated-dz5-command.json').open('x') as stream:
        json.dump({'command': command, 'child_sha256': hashlib.sha256(child.encode()).hexdigest(),
            'snapshot_manifest_sha256': hashlib.sha256((snapshot / 'manifest.json').read_bytes()).hexdigest(),
            'scope': scope, 'child_timeout_seconds': child_timeout_seconds,
            'cuda_visible_devices': environment['CUDA_VISIBLE_DEVICES']}, stream, indent=2)
    with (directory / 'isolated-dz5-import.log').open('x') as log:
        process = subprocess.run(command, env=environment, stdout=log,
            stderr=subprocess.STDOUT, timeout=child_timeout_seconds, check=False)
    report_path = directory / 'dz5-snapshot-import.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    assert process.returncode == 0, {
        'error': report.get('error'), 'traceback': report.get('traceback'),
        'log': (directory / 'isolated-dz5-import.log').read_text()}
    assert report['passed'] and not report['unexpected_modules']
    return report


def test_dz5_candidate_snapshot_isolates_actual_imports(request):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    run_isolated_snapshot(request, snapshot=SNAPSHOT, child=CHILD,
        scope='import_and_fixture_CPU_reference_only')
