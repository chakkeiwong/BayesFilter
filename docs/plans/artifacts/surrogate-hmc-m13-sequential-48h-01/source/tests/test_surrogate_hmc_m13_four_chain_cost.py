"""CPU-only diagnostic wiring tests; no posterior or performance claims."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_closure_program import jobs, PASS_CAPS
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import (
    F32, F64, STARTS, batched_endpoint, bind_components, chunk_health,
    compiled_chunk, mechanics, native_force,
)


class CanonicalStandIn:
    def __init__(self):
        self.calls = []

    def compile(self, batch_size, **kwargs):
        self.calls.append((batch_size, kwargs))

        def evaluate(position):
            assert position.dtype == F32
            assert position.shape == (batch_size, 2)
            valid = position[:, 0] < 2.
            return -tf.reduce_sum(position**2, axis=-1), -2.*position, dict(program_valid=valid)
        return evaluate


def scalar_reference(position):
    return -tf.reduce_sum(position**2), position[0] < 2., {}


def test_native_batch_sign_dtype_and_invalid_status():
    target = CanonicalStandIn()
    force = native_force(target)
    points = tf.constant([[.25, -.5], [-.25, .5], [1., 1.], [3., 1.]], F64)
    result = force(points)
    expected = tf.cast((2.*tf.cast(points, F32) + tf.cast(points, F32)/
                       tf.cast(mechanics.PRIOR_SD**2, F32)), F64)
    tf.debugging.assert_near(result.force[:3], expected[:3], atol=1e-12)
    tf.debugging.assert_equal(result.force[3], tf.zeros([2], F64))
    assert result.force_valid.numpy().tolist() == [True, True, True, False]
    assert target.calls == [(4, dict(filter_moments=False, all_correction_diagnostics=False))]


def test_endpoint_preserves_rows_prior_and_failure_classification():
    endpoint = batched_endpoint(scalar_reference)
    points = tf.constant([[.1, .7], [-.2, -.1], [3., .2], [.3, -.8]], F64)
    result = endpoint(points)
    expected = tf.reduce_sum(points**2, axis=-1)-mechanics.prior_log_density(points)
    tf.debugging.assert_near(tf.gather(result.potential, [0, 1, 3]), tf.gather(expected, [0, 1, 3]), atol=1e-12)
    assert result.status["value_valid"].numpy().tolist() == [True, True, False, True]
    assert result.status["numerical_failure"].numpy().tolist() == [False, False, True, False]
    assert not bool(tf.reduce_any(result.status["outside_declared_support"]))
    assert bool(tf.math.is_inf(result.potential[2]))


def test_public_binding_uses_native_affine_transform_and_whole_chunk():
    force = native_force(CanonicalStandIn())
    endpoint = batched_endpoint(scalar_reference)
    binding = bind_components(force, endpoint, "test-m13-connection")
    center = tf.constant([.1, -.2], F64)
    factor = tf.constant([[.7, 0.], [.1, .8]], F64)
    run, kernel, adapter = compiled_chunk(binding, "test-m13-connection", 2,
        center=center, factor=factor, count=4)
    start = tf.constant(STARTS, F64)
    raw = adapter.latent_to_position(start)
    tf.debugging.assert_near(adapter.position_to_latent(raw), start, atol=1e-12)
    actual_force = kernel.parameters["force"].function(start)
    expected_force = tf.linalg.matmul(force(raw).force, factor)
    tf.debugging.assert_near(actual_force.force, expected_force, atol=1e-12)
    actual_endpoint = kernel.parameters["target"].function(start)
    tf.debugging.assert_near(actual_endpoint.potential, endpoint(raw).potential, atol=1e-12)
    assert kernel.parameters["force"].semantics == "deterministic_position_only_proposal_field"
    assert kernel.parameters["target"].includes_chart_log_jacobian
    chunk = run(start, tf.constant([123, 456], tf.int32))
    assert chunk.states.shape == (4, 4, 2)
    assert all(bool(x) for x in chunk_health(chunk, start).values())
    assert not bool(tf.reduce_any(chunk.force_fallback))


def test_master_registers_isolated_bounded_stage():
    selected = jobs("m13-four-chain-cost")
    assert [j["name"] for j in selected] == ["review", "m13-four-chain-cost"]
    assert selected[1]["cpu"] == 850.
    assert PASS_CAPS["m13-four-chain-cost"] == dict(cpu_worker_seconds=1000., gpu_seconds=1400.)
