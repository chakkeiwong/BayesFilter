"""Independent NumPy reader verifies standard-library numeric NPZ serialization."""

import io
import zipfile

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.tensor_npz_archive import write_tensor_npz


@pytest.mark.parametrize('dtype', [tf.float64, tf.float32, tf.int64, tf.int32, tf.int16, tf.int8,
    tf.uint64, tf.uint32, tf.uint16, tf.uint8, tf.bool])
def test_npz_preserves_dtype_shape_and_reader_values(dtype):
    values = tf.constant([[0, 1], [1, 0]], dtype=dtype) if dtype != tf.bool else tf.constant([[False, True], [True, False]])
    fields = {'attempt_0_partition_0_scores': values, 'scalar': values[0, 0],
        'empty': tf.zeros([0, 2], dtype=dtype), 'vector': tf.reshape(values, [-1]),
        'large': tf.zeros([9000], dtype=dtype)}
    stream = io.BytesIO()
    write_tensor_npz(stream, fields)
    with np.load(io.BytesIO(stream.getvalue()), allow_pickle=False) as archive:
        assert archive.files == list(fields)
        for name, tensor in fields.items():
            expected = tensor.numpy()
            assert archive[name].shape == expected.shape
            assert archive[name].dtype == expected.dtype
            np.testing.assert_array_equal(archive[name], expected)
    original = io.BytesIO()
    np.savez_compressed(original, **{name: value.numpy() for name, value in fields.items()})
    with zipfile.ZipFile(stream) as actual_zip, zipfile.ZipFile(original) as original_zip:
        assert actual_zip.namelist() == original_zip.namelist()
        # NPY header padding and ZIP metadata need not be byte-identical;
        # readers must recover the same arrays from both archives.
        for name in actual_zip.namelist():
            actual = np.load(io.BytesIO(actual_zip.read(name)), allow_pickle=False)
            expected = np.load(io.BytesIO(original_zip.read(name)), allow_pickle=False)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()


def test_npz_float_special_values_and_integer_limits():
    fields = {'float64': tf.constant([0., -0., float('nan'), float('inf'), -float('inf'),
        float.fromhex('0x0.0000000000001p-1022')], tf.float64),
        'int64': tf.constant([-2**63, 2**63 - 1], tf.int64),
        'uint64': tf.constant([0, 2**64 - 1], tf.uint64)}
    output = io.BytesIO()
    write_tensor_npz(output, fields)
    with np.load(io.BytesIO(output.getvalue()), allow_pickle=False) as archive:
        for name, tensor in fields.items():
            assert archive[name].tobytes() == tensor.numpy().tobytes()
        assert np.signbit(archive['float64'][1])


@pytest.mark.parametrize('name', ['', '../scores', 'folder/scores', 'folder\\scores', '.', '..', 1])
def test_npz_rejects_nonflat_member_names(name):
    with pytest.raises(ValueError, match='member names'):
        write_tensor_npz(io.BytesIO(), {name: tf.constant([1.], tf.float64)})


@pytest.mark.parametrize('value', [tf.constant(['value']), tf.constant([1 + 2j], tf.complex128)])
def test_npz_rejects_unsupported_types(value):
    with pytest.raises(TypeError, match='unsupported'):
        write_tensor_npz(io.BytesIO(), {'value': value})
