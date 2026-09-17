"""TensorFlow numerical inputs and standard-library identity serialization.

These are host preparation/artifact boundaries, not XLA kernels. Buffer input
is supported for legacy callers without importing an array-computation backend.
"""

import sys
from itertools import product

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2


def numeric_tensor(value, dtype=None):
    """Convert numeric input, preserving legacy Python float/int precision.

    Buffer decoding also accepts non-native byte order and noncontiguous arrays.
    The buffer only transports bytes; all numeric conversions use TensorFlow.
    """
    if tf.is_tensor(value):
        return (
            tf.cast(value, dtype) if dtype is not None else tf.convert_to_tensor(value)
        )
    try:
        buffer = memoryview(value)
    except TypeError:
        buffer = None
    if buffer is not None:
        source_dtype = getattr(value, "dtype", None)
        if source_dtype is not None:
            source_dtype = tf.as_dtype(getattr(source_dtype, "name", source_dtype))
            if source_dtype.is_complex:
                component_dtype = source_dtype.real_dtype
            else:
                component_dtype = source_dtype
            little_endian = not buffer.format.startswith((">", "!"))
            if buffer.format.startswith(("@", "=")) or buffer.format[:1] not in "<>!":
                little_endian = sys.byteorder == "little"
            decode_dtype = {tf.uint32: tf.int32, tf.uint64: tf.int64}.get(
                component_dtype, component_dtype
            )
            raw = tf.io.decode_raw(
                buffer.tobytes(order="C"), decode_dtype, little_endian=little_endian
            )
            if decode_dtype != component_dtype:
                raw = tf.bitcast(raw, component_dtype)
            if source_dtype.is_complex:
                parts = tf.reshape(raw, [-1, 2])
                raw = tf.complex(parts[:, 0], parts[:, 1])
            result = tf.reshape(raw, buffer.shape)
            return tf.cast(result, dtype) if dtype is not None else result
    if dtype is None:
        leaves = tf.nest.flatten(value)
        if any(isinstance(item, complex) for item in leaves):
            dtype = tf.complex128
        elif not leaves or any(isinstance(item, float) for item in leaves):
            dtype = tf.float64
        elif all(isinstance(item, bool) for item in leaves):
            dtype = tf.bool
        elif all(isinstance(item, int) for item in leaves):
            dtype = tf.int64
    return tf.convert_to_tensor(value, dtype=dtype)


def numeric_dtype(dtype):
    result = tf.as_dtype(dtype)
    if not (
        result.is_integer
        or result.is_floating
        or result.is_complex
        or result == tf.bool
    ):
        raise ValueError("array dtype must be numeric or boolean")
    return result


def all_finite(value):
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype.is_complex:
        return tf.reduce_all(tf.math.is_finite(tf.math.real(tensor))) & tf.reduce_all(
            tf.math.is_finite(tf.math.imag(tensor))
        )
    if tensor.dtype.is_floating:
        return tf.reduce_all(tf.math.is_finite(tensor))
    return tf.constant(True)


def canonical_numeric_bytes(value, *, byteorder="little"):
    """Contiguous IEEE/integer bytes; preserve signed zero and scalar shape."""
    tensor = numeric_tensor(value)
    numeric_dtype(tensor.dtype)
    proto = tensor_pb2.TensorProto.FromString(tf.io.serialize_tensor(tensor).numpy())
    data = proto.tensor_content
    width = (
        tensor.dtype.real_dtype.size if tensor.dtype.is_complex else tensor.dtype.size
    )
    if byteorder not in ("little", "big"):
        raise ValueError("byteorder must be little or big")
    if byteorder == "big" and width > 1:
        # Byte serialization, never an array numerical computation.
        data = b"".join(
            data[offset : offset + width][::-1] for offset in range(0, len(data), width)
        )
    return tensor, data


def is_boolean_scalar(value):
    if isinstance(value, bool):
        return True
    dtype = getattr(value, "dtype", None)
    return (
        dtype is not None
        and tf.as_dtype(getattr(dtype, "name", dtype)) == tf.bool
        and tuple(value.shape) == ()
    )


def buffer_byte_regions(value, *, split_first_axis=False):
    """Describe host-buffer aliases without reading or calculating array values.

    Arrays expose their address/strides through the standard array-interface
    protocol. Immutable tensors and freshly converted Python sequences have no
    mutable input-buffer alias to reject. These intervals retain holes, negative
    strides and overlapping views; a bounding-box test alone would reject
    disjoint interleaved partitions. Iteration here describes storage, not a
    numerical kernel.
    """
    interface = getattr(value, "__array_interface__", None)
    if interface is None:
        return ()
    view = memoryview(value)
    if view.suboffsets and any(offset >= 0 for offset in view.suboffsets):
        raise ValueError("indirect host buffers are unsupported for alias validation")
    shape, strides = tuple(view.shape), tuple(view.strides)
    address = int(interface["data"][0])
    if split_first_axis:
        origins = tuple(address + index * strides[0] for index in range(shape[0]))
        shape, strides = shape[1:], strides[1:]
    else:
        origins = (address,)
    if not all(shape):
        return tuple(() for _ in origins)
    # Collapse contiguous dimensions, including reversed and broadcast views.
    axes = sorted(zip(shape, strides), key=lambda axis: abs(axis[1]))
    width, shift, sparse = view.itemsize, 0, []
    for extent, stride in axes:
        if abs(stride) <= width:
            shift += min(0, (extent - 1) * stride)
            width += (extent - 1) * abs(stride)
        else:
            sparse.append((extent, stride))
    result = []
    for origin in origins:
        ranges = sorted(
            (start, start + width)
            for index in product(*(range(extent) for extent, _ in sparse))
            for start in (
                origin + shift + sum(i * axis[1] for i, axis in zip(index, sparse)),
            )
        )
        merged = []
        for start, stop in ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
            else:
                merged.append((start, stop))
        result.append(tuple(merged))
    return tuple(result)


def buffer_regions_overlap(left, right):
    """Exact intersection of two sorted host-buffer interval descriptions."""
    first = second = 0
    while first < len(left) and second < len(right):
        a, b = left[first], right[second]
        if a[0] < b[1] and b[0] < a[1]:
            return True
        if a[1] <= b[0]:
            first += 1
        else:
            second += 1
    return False
