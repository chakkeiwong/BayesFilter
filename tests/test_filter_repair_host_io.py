"""Independent NumPy reference bytes at the host serialization boundary."""

import hashlib

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_identity import CanonicalArrayIdentityV1
from bayesfilter.inference.target_failure_policy import (
    TargetFailurePolicy,
    evaluate_target_with_failure_policy,
)
from bayesfilter.ops.host_tensor_io import buffer_byte_regions, buffer_regions_overlap


def test_buffer_regions_match_independent_alias_authority():
    base = np.arange(2 * 9 * 6, dtype=np.float64).reshape(2, 9, 6)
    views = (
        base,
        base[:, ::2, :],
        base[:, 1::2, :],
        base[:, :, ::2],
        base[:, :, 1::2],
        base[::-1, ::-1, ::-1],
        base[:, :0],
        base[:, 2:7, 1:4],
        base.copy(),
        np.broadcast_to(base[0, 0], [4, 6]),
        base.view(np.uint8)[:, :, 1::3],
    )
    for left in views:
        for right in views:
            actual = buffer_regions_overlap(
                buffer_byte_regions(left)[0], buffer_byte_regions(right)[0]
            )
            assert actual == np.shares_memory(left, right)
    regions = buffer_byte_regions(base, split_first_axis=True)
    assert not buffer_regions_overlap(*regions)
    assert buffer_byte_regions([[1.0, 2.0]]) == ()


def test_partition_identity_checks_views_copies_and_signed_zero():
    from bayesfilter.inference.fixed_center_curvature import (
        _require_independent_partitions,
    )

    base = np.arange(40, dtype=np.float64).reshape(10, 4)
    _require_independent_partitions((("left", base[::2]), ("right", base[1::2])))
    with pytest.raises(ValueError, match="share memory"):
        _require_independent_partitions((("left", base), ("right", base[::-1])))
    copied = base[::2].copy()
    copied[0, 0] = -0.0
    with pytest.raises(ValueError, match="copied rows"):
        _require_independent_partitions((("left", base), ("right", copied)))


@pytest.mark.parametrize(
    "dtype",
    [
        "?",
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
@pytest.mark.parametrize("shape", [(), (0,), (2, 2)])
def test_identity_preserves_legacy_bytes(dtype, shape):
    size = int(np.prod(shape))
    base = np.arange(size).reshape(shape).astype(dtype)
    if base.dtype.kind in "fc" and size:
        base.flat[0] = -0.0
    expected = base.astype(base.dtype.newbyteorder(">"), copy=False).tobytes(order="C")
    for array in (
        base,
        np.asfortranarray(base).reshape(shape),
        base.astype(base.dtype.newbyteorder(">")),
    ):
        result = CanonicalArrayIdentityV1.from_array(array)
        assert result.shape == shape
        assert result.semantic_dtype == base.dtype.name
        assert result.byte_sha256 == hashlib.sha256(expected).hexdigest()


def test_failure_policy_preserves_finite_and_invalid_results():
    policy = TargetFailurePolicy("tensor-input")
    position = tf.constant([1.0, 2.0], tf.float64)
    result = evaluate_target_with_failure_policy(
        lambda x: (-tf.reduce_sum(x * x), -2 * x), position, policy
    )
    assert result.value == -5.0 and result.score_finite and not result.fallback_used
    tf.debugging.assert_equal(result.score, tf.constant([-2.0, -4.0], tf.float64))
    invalid = evaluate_target_with_failure_policy(
        lambda x: (tf.constant(float("nan"), tf.float64), x), position, policy
    )
    assert invalid.fallback_used and invalid.value_finite and invalid.score_finite
    tf.debugging.assert_equal(invalid.score, tf.zeros_like(position))
