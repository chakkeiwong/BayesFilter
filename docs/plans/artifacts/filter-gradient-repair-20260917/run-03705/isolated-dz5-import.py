
import hashlib
import importlib
import json
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
    (OUT / 'dz5-snapshot-import.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
sys.exit(0 if report['passed'] else 1)
