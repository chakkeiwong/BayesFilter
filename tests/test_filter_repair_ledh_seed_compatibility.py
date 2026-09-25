"""Diagnostic only: original LEDH random-input compatibility with native XLA."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_filter_tf import _replication_generator


@pytest.mark.parametrize("dtype", [tf.float64, tf.float32])
@pytest.mark.parametrize("shape", [(2, 2), (6, 2), (17, 3)])
def test_original_philox_key_counter_normal_compatibility(dtype, shape, request):
    signature = [tf.TensorSpec([1], tf.uint64), tf.TensorSpec([2], tf.uint64)]

    def draw(key, counter):
        return (
            tf.raw_ops.StatelessRandomNormalV2(shape=shape, key=key, counter=counter,
                                               alg=1, dtype=dtype),
            tf.raw_ops.StatelessRandomUniformFullIntV2(shape=shape, key=key,
                                                       counter=counter, alg=1, dtype=tf.uint64),
        )

    compiled = tf.function(draw, input_signature=signature, jit_compile=True, autograph=False)
    graph = tf.function(draw, input_signature=signature, jit_compile=False, autograph=False)
    cases = []
    for seed in (0, 1, 42, 2**64 + 7):
        generator = _replication_generator(seed)
        expected_material = np.random.SeedSequence(seed).generate_state(2, dtype=np.uint32)
        expected_seed = (int(expected_material[0]) << 31) ^ int(expected_material[1])
        assert generator.state.numpy().tolist() == [expected_seed, 0, 0]
        for call_index in range(3):
            state = tf.bitcast(generator.state.read_value(), tf.uint64)
            key, counter = state[2:], state[:2]
            actual, bits = compiled(key, counter)
            expected, expected_bits = graph(key, counter)
            original = generator.normal(shape, dtype=dtype)
            np.testing.assert_array_equal(expected, original)
            np.testing.assert_array_equal(bits, expected_bits)
            assert generator.state.numpy().tolist() == [expected_seed + (call_index + 1) * 256 * np.prod(shape), 0, 0]
            replay = compiled(key, counter)[0]
            np.testing.assert_array_equal(actual, replay)
            delta = np.abs(actual.numpy() - original.numpy())
            tolerance = 1e-12 if dtype == tf.float64 else 1e-5
            scaled = delta / (tolerance * (1. + np.abs(original.numpy())))
            cases.append({"seed": seed, "call_index": call_index,
                "seed_words": expected_material.tolist(), "state": state.numpy().tolist(),
                "exact_normal_equal": bool(np.array_equal(actual, original)),
                "normal_max_absolute_error": float(delta.max()),
                "normal_max_scaled_error": float(scaled.max()),
                "normal_matches_within_tolerance": bool(np.all(scaled <= 1.)),
                "raw_bits_exact": True, "reference": original.numpy().tolist(),
                "xla": actual.numpy().tolist()})
    assert compiled.experimental_get_tracing_count() == graph.experimental_get_tracing_count() == 1
    hlo = compiled.experimental_get_compiler_ir(key, counter)(stage="hlo")
    definition = compiled.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in definition.node}
    operations.update(node.op for function in definition.library.function for node in function.node_def)
    assert not operations & {"PyFunc", "PyFuncStateless", "EagerPyFunc", "XlaHostCompute"}
    report = {"schema": "filter_repair_ledh_seed_compatibility.v1", "diagnostic_complete": True,
        "dtype": dtype.name, "shape": list(shape), "cases": cases,
        "all_normals_exact": all(row["exact_normal_equal"] for row in cases),
        "all_normals_within_tolerance": all(row["normal_matches_within_tolerance"] for row in cases),
        "trace_count": 1, "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "device": actual.device, "nonclaims": ["No runtime stream migration or endpoint qualification."]}
    directory = Path(request.config.getoption("xmlpath")).parent
    stem = f"ledh-seed-{shape[0]}-{shape[1]}-{dtype.name}"
    (directory / f"{stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (directory / f"{stem}.hlo.txt").write_text(hlo)
