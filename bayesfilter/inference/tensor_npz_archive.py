"""NumPy-compatible numeric NPZ archives using TensorFlow and the standard library.

File I/O and byte packing are host artifact-boundary work. TensorFlow supplies
flat numeric values and shapes; NumPy is neither imported nor used at runtime.
"""

import struct
import zipfile

import tensorflow as tf

_NUMERIC_DTYPES = {
    tf.float64: ('<f8', 'd'), tf.float32: ('<f4', 'f'),
    tf.int64: ('<i8', 'q'), tf.int32: ('<i4', 'i'),
    tf.int16: ('<i2', 'h'), tf.int8: ('|i1', 'b'),
    tf.uint64: ('<u8', 'Q'), tf.uint32: ('<u4', 'I'),
    tf.uint16: ('<u2', 'H'), tf.uint8: ('|u1', 'B'), tf.bool: ('|b1', '?'),
}


def write_tensor_npz(stream, tensors):
    """Write dense real numeric tensors without object or pickled members.

    Match np.savez_compressed member names, C order, shapes and dtype descriptors.
    The caller controls exclusive creation and artifact hashing. Byte-identical
    ZIP compression is not promised; standard NPZ readers observe equal arrays.
    """
    with zipfile.ZipFile(stream, mode='w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, value in tensors.items():
            if not isinstance(name, str) or not name or '/' in name or '\\' in name or name in ('.', '..'):
                raise ValueError('NPZ member names must be nonempty flat strings')
            tensor = tf.convert_to_tensor(value)
            if tensor.dtype not in _NUMERIC_DTYPES:
                raise TypeError(f'unsupported numeric NPZ dtype: {tensor.dtype.name}')
            descriptor, code = _NUMERIC_DTYPES[tensor.dtype]
            if not tensor.shape.is_fully_defined():
                raise ValueError('NPZ tensor shape must be fully defined at the artifact boundary')
            shape = tuple(tensor.shape.as_list())
            header = repr({'descr': descriptor, 'fortran_order': False, 'shape': shape}).encode('latin1')
            # NPY v1: magic(6), version(2), header length(2), then a newline-
            # terminated Python literal padded to a 64-byte boundary.
            padding = (-(10 + len(header) + 1)) % 64
            header += b' ' * padding + b'\n'
            if len(header) > 65535:
                raise ValueError('NPY v1 shape header exceeds 65535 bytes')
            values = tf.reshape(tensor, [-1]).numpy().tolist()
            with archive.open(name + '.npy', mode='w', force_zip64=True) as member:
                member.write(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header)
                for start in range(0, len(values), 8192):
                    chunk = values[start:start + 8192]
                    member.write(struct.pack('<' + str(len(chunk)) + code, *chunk))
