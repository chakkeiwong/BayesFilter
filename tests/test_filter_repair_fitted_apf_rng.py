"""Diagnostic eager/XLA input-stream compatibility before fitted-APF repair."""

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def _draws(seed):
    return {
        'initial': tf.random.stateless_normal([16, 1], seed, dtype=tf.float64),
        'process': tf.random.stateless_normal([2, 16, 1], seed, dtype=tf.float64),
        'ancestors': tf.random.stateless_uniform([3, 16], seed, dtype=tf.float64),
        'mixture': tf.random.stateless_uniform([2, 16], seed, dtype=tf.float64),
    }


def test_fitted_apf_rng_compatibility_probe(request):
    owner = tf.function(_draws, input_signature=[tf.TensorSpec([2], tf.int32)],
                        jit_compile=True, autograph=False)
    rows = []
    for words in ([9296027, 1], [9296027, 2]):
        seed = tf.constant(words, tf.int32)
        reference, compiled = _draws(seed), owner(seed)
        fields = {}
        for name in reference:
            left, right = reference[name].numpy(), compiled[name].numpy()
            assert np.isfinite(left).all() and np.isfinite(right).all()
            difference = float(np.max(np.abs(left-right)))
            exact = bool(np.array_equal(left, right))
            fields[name] = {'shape': list(left.shape), 'reference': left.tolist(), 'compiled': right.tolist(),
                'maximum_absolute_difference': difference, 'exact': exact,
                'compatible': exact if name in ('ancestors', 'mixture') else difference <= 1e-12,
                'reference_sha256': hashlib.sha256(left.tobytes()).hexdigest(),
                'compiled_sha256': hashlib.sha256(right.tobytes()).hexdigest(),
                'reference_device': reference[name].device, 'compiled_device': compiled[name].device}
        rows.append({'seed': words, 'fields': fields})
    assert owner.experimental_get_tracing_count() == 1
    concrete = owner.get_concrete_function()
    assert concrete.function_def.attr['_XlaMustCompile'].b
    hlo = owner.experimental_get_compiler_ir(tf.constant([9296027, 1], tf.int32))(stage='hlo')
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory/'fitted-apf-rng.hlo.txt').write_text(hlo)
    report = {'schema': 'filter_repair_fitted_apf_rng_probe.v1', 'rows': rows,
        'simple_compilation_preserves_inputs': all(f['compatible'] for row in rows for f in row['fields'].values()),
        'trace_count': 1, 'jit_compile': True, 'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(),
        'scope': 'Eager independent input reference versus XLA; no algorithm, tuning, stream change or endpoint admission.'}
    (directory/'fitted-apf-rng.json').write_text(json.dumps(report, indent=2)+'\n')
