"""Independent NumPy/TF references for the native LEDH compatibility stream."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_filter_tf import _replication_generator
from bayesfilter.ops.ledh_random_compat_tf import (
    integer_seed_words,
    pcg64_initial_state,
    pcg64_next,
    philox_box_muller_normal,
    philox_replication_state,
    seed_sequence_words,
)


def _halves(integer):
    return [integer >> 64, integer & ((1 << 64) - 1)]


@pytest.mark.parametrize("seed", [0, 1, 42, 2**32 - 1, 2**32, 2**64 + 7, 2**160 - 1])
def test_seed_words_pcg_state_and_uniforms_exact(seed, request):
    words = tf.constant(integer_seed_words(seed), tf.uint32)

    def program(entropy):
        initial, increment = pcg64_initial_state(entropy)

        def body(index, state, raw, uniform):
            state, word, value = pcg64_next(state, increment)
            return (index + 1, state, tf.tensor_scatter_nd_update(raw, [[index]], [word]),
                    tf.tensor_scatter_nd_update(uniform, [[index]], [value]))

        _, final, raw, uniform = tf.while_loop(lambda index, *_: index < 32, body,
            (tf.constant(0), initial, tf.zeros([32], tf.uint64), tf.zeros([32], tf.float64)), parallel_iterations=1)
        return seed_sequence_words(entropy, 8), initial, increment, final, raw, uniform

    compiled = tf.function(program, input_signature=[tf.TensorSpec(words.shape, tf.uint32)],
                           jit_compile=True, autograph=False)
    material, initial, increment, final, raw, uniform = compiled(words)
    expected = np.random.PCG64(seed)
    np.testing.assert_array_equal(material, np.random.SeedSequence(seed).generate_state(8))
    np.testing.assert_array_equal(initial, _halves(expected.state["state"]["state"]))
    np.testing.assert_array_equal(increment, _halves(expected.state["state"]["inc"]))
    np.testing.assert_array_equal(raw, expected.random_raw(32))
    np.testing.assert_array_equal(final, _halves(expected.state["state"]["state"]))
    np.testing.assert_array_equal(uniform, np.random.default_rng(seed).uniform(size=32))
    for actual, original in zip(compiled(words), (material, initial, increment, final, raw, uniform), strict=True):
        np.testing.assert_array_equal(actual, original)
    changed = tf.tensor_scatter_nd_update(words, [[0]], [words[0] ^ tf.constant(17, tf.uint32)])
    np.testing.assert_array_equal(compiled(changed)[0],
        np.random.SeedSequence(changed.numpy()).generate_state(8))
    assert compiled.experimental_get_tracing_count() == 1
    report = {"seed": seed, "exact": True, "material": material.numpy().tolist(),
        "initial": initial.numpy().tolist(), "increment": increment.numpy().tolist(),
        "final": final.numpy().tolist(), "raw": raw.numpy().tolist(), "uniform": uniform.numpy().tolist()}
    directory = Path(request.config.getoption("xmlpath")).parent
    (directory / f"pcg-compat-{seed}.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize("dtype", [tf.float64, tf.float32])
@pytest.mark.parametrize("shape", [(2, 2), (6, 2), (17, 3)])
def test_native_box_muller_preserves_realization(dtype, shape, request):
    compiled = tf.function(lambda state: philox_box_muller_normal(state, shape, dtype),
        input_signature=[tf.TensorSpec([3], tf.uint64)], jit_compile=True, autograph=False)
    cases = []
    for seed in (0, 1, 42, 2**64 + 7):
        generator = _replication_generator(seed)
        # Seed mixing and draw transform are independently checked against old code.
        initial = tf.function(philox_replication_state,
            input_signature=[tf.TensorSpec([len(integer_seed_words(seed))], tf.uint32)],
            jit_compile=True, autograph=False)(tf.constant(integer_seed_words(seed), tf.uint32))
        np.testing.assert_array_equal(initial, tf.bitcast(generator.state, tf.uint64))
        state = initial
        for index in range(3):
            actual, following = compiled(state)
            expected = generator.normal(shape, dtype=dtype)
            np.testing.assert_array_equal(following, tf.bitcast(generator.state, tf.uint64))
            tolerance = 1e-12 if dtype == tf.float64 else 1e-5
            delta = np.abs(actual.numpy() - expected.numpy())
            scaled = delta / (tolerance * (1. + np.abs(expected.numpy())))
            cases.append({"seed": seed, "index": index, "max_absolute_error": float(delta.max()),
                "max_scaled_error": float(scaled.max()), "bitwise_equal": bool(np.array_equal(actual, expected)),
                "expected": expected.numpy().tolist(), "actual": actual.numpy().tolist()})
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
            np.testing.assert_array_equal(compiled(state)[0], actual)
            state = following
    assert compiled.experimental_get_tracing_count() == 1
    hlo = compiled.experimental_get_compiler_ir(initial)(stage="hlo")
    assert hlo == compiled.experimental_get_compiler_ir(state)(stage="hlo")
    definition = compiled.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in definition.node}
    operations.update(node.op for function in definition.library.function for node in function.node_def)
    assert not operations & {"PyFunc", "PyFuncStateless", "EagerPyFunc", "XlaHostCompute"}
    report = {"shape": list(shape), "dtype": dtype.name, "passed": True,
        "cases": cases, "trace_count": 1, "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "device": actual.device, "nonclaims": ["No complete filter or scientific admission."]}
    directory = Path(request.config.getoption("xmlpath")).parent
    stem = f"normal-compat-{shape[0]}-{shape[1]}-{dtype.name}"
    (directory / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / f"{stem}.hlo.txt").write_text(hlo)
