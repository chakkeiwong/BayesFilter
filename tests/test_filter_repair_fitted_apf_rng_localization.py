"""Inspect explicit Philox seed mapping and shared normal-transform compatibility."""

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import random_ops_util

from bayesfilter.ops.ledh_random_compat_tf import philox_box_muller_normal
from tests.test_filter_repair_fitted_apf_rng import _draws


def _explicit(seed):
    key, counter = random_ops_util._philox_scramble_seed(seed)
    state = tf.concat([counter, key], 0)
    return {
        'initial': tf.random.stateless_normal([16, 1], seed, dtype=tf.float64, alg='philox'),
        'process': tf.random.stateless_normal([2, 16, 1], seed, dtype=tf.float64, alg='philox'),
        'ancestors': tf.random.stateless_uniform([3, 16], seed, dtype=tf.float64, alg='philox'),
        'mixture': tf.random.stateless_uniform([2, 16], seed, dtype=tf.float64, alg='philox'),
        'shared_initial': philox_box_muller_normal(state, (16, 1), tf.float64)[0],
        'shared_process': philox_box_muller_normal(state, (2, 16, 1), tf.float64)[0],
        'key': key, 'counter': counter,
        'words': tf.raw_ops.StatelessRandomUniformFullIntV2(
            shape=[96], key=key, counter=counter, alg=1, dtype=tf.uint32),
    }


def test_explicit_philox_compatibility(request):
    owner = tf.function(_explicit, input_signature=[tf.TensorSpec([2], tf.int32)],
                        jit_compile=True, autograph=False)
    rows = []
    for words in ([9296027, 1], [9296027, 2]):
        seed = tf.constant(words, tf.int32)
        reference, explicit, compiled = _draws(seed), _explicit(seed), owner(seed)
        auto_key, auto_counter = tf.raw_ops.StatelessRandomGetKeyCounter(seed=seed)
        states = {}
        for name, old in (('key', auto_key), ('counter', auto_counter)):
            states[name] = {'auto': old.numpy().tolist(), 'explicit': explicit[name].numpy().tolist(),
                'compiled': compiled[name].numpy().tolist(),
                'all_exact': bool(np.array_equal(old, explicit[name]) and np.array_equal(old, compiled[name]))}
        states['words_exact'] = bool(np.array_equal(explicit['words'], compiled['words']))
        fields = {}
        for name in ('initial', 'process', 'ancestors', 'mixture'):
            left, eager, right = reference[name].numpy(), explicit[name].numpy(), compiled[name].numpy()
            fields[name] = {'reference': left.tolist(), 'eager_explicit': eager.tolist(), 'compiled_explicit': right.tolist(),
                'eager_explicit_exact': bool(np.array_equal(left, eager)),
                'compiled_explicit_error': float(np.max(np.abs(left-right))),
                'compiled_explicit_exact': bool(np.array_equal(left, right))}
            if name in ('initial', 'process'):
                shared = compiled['shared_'+name].numpy()
                fields[name].update(compiled_shared=shared.tolist(),
                    compiled_shared_error=float(np.max(np.abs(left-shared))),
                    compiled_shared_exact=bool(np.array_equal(left, shared)))
        rows.append({'seed': words, 'states': states, 'fields': fields,
                     'words_eager': explicit['words'].numpy().tolist(),
                     'words_compiled': compiled['words'].numpy().tolist()})
    assert owner.experimental_get_tracing_count() == 1
    hlo = owner.experimental_get_compiler_ir(tf.constant([9296027, 1], tf.int32))(stage='hlo')
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory/'fitted-apf-rng-explicit.hlo.txt').write_text(hlo)
    tf_source = Path(random_ops_util.__file__)
    (directory/'tensorflow-random_ops_util.py').write_bytes(tf_source.read_bytes())
    source = Path(__file__).resolve().parents[1]/'bayesfilter/ops/ledh_random_compat_tf.py'
    report = {'schema': 'filter_repair_fitted_apf_rng_explicit.v1', 'rows': rows,
        'trace_count': 1, 'jit_compile': True, 'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(),
        'tensorflow_seed_source_sha256': hashlib.sha256(tf_source.read_bytes()).hexdigest(),
        'shared_normal_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope': 'Private TensorFlow seed helper is diagnostic only; no runtime or RNG stream changes.'}
    (directory/'fitted-apf-rng-explicit.json').write_text(json.dumps(report, indent=2)+'\n')
