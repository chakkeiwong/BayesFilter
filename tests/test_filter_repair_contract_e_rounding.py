"""Diagnostic attribution of the CPU-derived FP32 ULP assertion on GPU.

This records the frozen/current finite-program diagnostic, not LEDH admission.
The original ULP limits remain unchanged and every result is retained.
"""

import hashlib
import types
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.highdim.test_ledh_contract_e_canonical_lgssm_phase5 import (
    SHORT_BALANCE_KWARGS,
    _fixture,
    _forward_parameter_jacobian,
    _max_ulp_distance,
    _prepared,
    canonical,
)
from tests.test_filter_repair_geometry_control import clean, save

BASELINE = "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf"
ROOT = Path(__file__).resolve().parents[1]


def frozen_authority():
    frozen = FrozenCheckpoint(BASELINE, "contract_e_rounding")
    # The pinned streaming implementation uses a package-form submodule import.
    # Keep this namespace empty so its attributes resolve only via the loader.
    namespace = types.ModuleType(frozen.prefix + ".bayesfilter.highdim")
    namespace.__path__ = []
    frozen.modules["bayesfilter.highdim"] = namespace
    return frozen, frozen.load("bayesfilter.highdim.ledh_contract_e_canonical_lgssm_tf")


@pytest.mark.parametrize("tf32", [True, False])
def test_original_and_current_fp32_rounding(tf32, request):
    tf.config.experimental.enable_tensor_float_32_execution(tf32)
    frozen, original = frozen_authority()
    theta = tf.constant(_fixture()["center_theta"], tf.float32)
    prepared = canonical._as_prepared_tensors(_prepared(), dtype=tf.float32)
    results = {}
    for label, module in (("original", original), ("current", canonical)):
        primal = module._canonical_primal_core(theta, prepared, **SHORT_BALANCE_KWARGS)
        manual = module._canonical_manual_jvp_core(theta, prepared, **SHORT_BALANCE_KWARGS)
        automatic = _forward_parameter_jacobian(theta, lambda value, authority=module:
            authority._canonical_primal_core(value, prepared, **SHORT_BALANCE_KWARGS)[
                "per_batch_log_likelihood"])
        aggregate = tf.reduce_mean(automatic, axis=0)
        per_batch_ulps = _max_ulp_distance(manual["per_batch_score"], automatic)
        aggregate_ulps = _max_ulp_distance(manual["score"], aggregate)
        results[label] = {
            "manual": clean(manual), "primal": clean(primal),
            "automatic_per_batch_score": clean(automatic),
            "automatic_score": clean(aggregate),
            "per_batch_ulps": per_batch_ulps, "aggregate_ulps": aggregate_ulps,
            "original_cpu_ulp_gate_passed": per_batch_ulps <= 3 and aggregate_ulps <= 2,
        }
    save(request, "contract-e-rounding.json", {
        "baseline": BASELINE, "baseline_sources": frozen.hashes(),
        "current_sources": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            for path in frozen.sources},
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "device": theta.device, "jit_compile": False,
        "role": "eager_reference_failure_attribution_not_default_qualification",
        "theta": clean(theta), "prepared": clean(prepared), "results": results,
        "nonclaims": ["The process succeeding means the diagnostic completed, not that a ULP gate passed.",
                      "No tolerance change, canonical admission or GPU/XLA qualification."],
    })
    assert tf.config.experimental.tensor_float_32_execution_enabled() == tf32
    assert all(bool(tf.reduce_all(tf.math.is_finite(tf.constant(row["automatic_score"]))))
               for row in results.values())


def test_default_factory_matches_frozen_program(request):
    """Exercise the actual default FP64 owner, dynamic inputs and status fields."""
    frozen, original = frozen_authority()
    theta = tf.constant(_fixture()["center_theta"], tf.float64)
    prepared = canonical._as_prepared_tensors(_prepared(), dtype=tf.float64)
    options = dict(batch_size=2, time_steps=2, num_particles=4, **SHORT_BALANCE_KWARGS)
    before = original.make_canonical_prepared_value_and_score_tf(**options)
    after = canonical.make_canonical_prepared_value_and_score_tf(**options)
    assert before._jit_compile and after._jit_compile
    records = []
    for parameters in (theta, theta + tf.constant([.001, 0., -.001, 0., 0.], tf.float64)):
        expected, actual = before(parameters, prepared), after(parameters, prepared)
        assert expected.keys() == actual.keys()
        errors = {}
        for key in expected:
            a, b = actual[key].numpy(), expected[key].numpy()
            if actual[key].dtype.is_floating:
                np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10, equal_nan=True, err_msg=key)
                finite = np.isfinite(a) & np.isfinite(b)
                errors[key] = float(np.max(np.abs(a-b)[finite], initial=0.))
            else:
                np.testing.assert_array_equal(a, b, err_msg=key)
        records.append({"theta": clean(parameters), "expected": clean(expected),
                        "actual": clean(actual), "maximum_absolute_errors": errors})
    graph = after.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in graph.node}
    operations.update(node.op for fn in graph.library.function for node in fn.node_def)
    assert not operations & {"PyFunc", "EagerPyFunc", "PyFuncStateless", "XlaHostCompute"}
    hlo = after.experimental_get_compiler_ir(theta, prepared)(stage="hlo")
    assert before.experimental_get_tracing_count() == after.experimental_get_tracing_count() == 1
    save(request, "contract-e-default-factory.json", {
        "baseline": BASELINE, "baseline_sources": frozen.hashes(),
        "current_sources": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            for path in frozen.sources}, "device": theta.device,
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "jit_compile": True, "trace_count": 1, "records": records,
        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "nonclaims": ["Diagnostic finite-program parity is not canonical LEDH admission.",
                      "FP64 factory comparison does not admit the TF32 eager AD score discrepancy."],
    })
