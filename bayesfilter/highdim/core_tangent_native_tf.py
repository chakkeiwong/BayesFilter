"""Compiled block encoding of the existing core-affine tangent extension.

The product rule for a finite TT is encoded by the block matrix
[[parent, tangent], [zero, parent]], with the original first/last blocks.
This is an execution repair of the local extension, not a source-route claim.
Python describes the heterogeneous tuple schema; tensor indices perform every
coefficient selection across all axes and parameter components at once.
"""

from collections import namedtuple
from functools import lru_cache
from itertools import accumulate
from math import prod

import tensorflow as tf

D = tf.float64
BlockProgram = namedtuple("BlockProgram", "encode decode mask shapes sizes")


@lru_cache(maxsize=16)
def block_program(shapes, component_count):
    if len(shapes) < 2 or component_count < 1:
        raise ValueError("at least two parent cores and one tangent component are required")
    if any(len(shape) != 3 or min(shape) < 1 for shape in shapes):
        raise ValueError("parent cores require positive fixed rank-three shapes")
    sizes = tuple(prod(shape) for shape in shapes)
    offsets = tuple(accumulate(sizes, initial=0))
    block_shapes = tuple((left if axis == 0 else 2 * left, width,
                          right if axis == len(shapes) - 1 else 2 * right)
                         for axis, (left, width, right) in enumerate(shapes))
    block_sizes = tuple(prod(shape) for shape in block_shapes)
    block_offsets = tuple(accumulate(block_sizes, initial=0))
    parent_size, block_size = offsets[-1], block_offsets[-1]

    def layout():
        axis = tf.repeat(tf.range(len(shapes)), block_sizes)
        local = tf.range(block_size) - tf.gather(block_offsets[:-1], axis)
        shape = tf.gather(tf.constant(shapes, tf.int32), axis)
        left, width, right = tf.unstack(shape, axis=1)
        output_right = tf.gather(tf.constant(block_shapes, tf.int32)[:, 2], axis)
        row, column = local // (width * output_right), local % output_right
        basis = (local // output_right) % width
        source = tf.gather(offsets[:-1], axis) + (row % left * width + basis) * right + column % right
        left_block = row // left
        right_block = tf.where(axis == len(shapes) - 1, 1, column // right)
        tangent = (left_block == 0) & (right_block == 1)
        indices = tf.where(tangent, parent_size + source,
                           tf.where(left_block == right_block, source, 2 * parent_size))
        return indices, tangent

    @tf.function(input_signature=[tf.TensorSpec([parent_size], D),
        tf.TensorSpec([component_count, parent_size], D)], jit_compile=True, autograph=False)
    def encode(parent, tangents):
        indices, _ = layout()
        sources = tf.concat([tf.broadcast_to(parent, [component_count, parent_size]),
                             tangents, tf.zeros([component_count, 1], D)], axis=1)
        return tf.gather(sources, indices, axis=1)

    @tf.function(input_signature=[tf.TensorSpec([parent_size], D),
        tf.TensorSpec([component_count, block_size], D)], jit_compile=True, autograph=False)
    def decode(parent, blocks):
        axis = tf.repeat(tf.range(len(shapes)), sizes)
        local = tf.range(parent_size) - tf.gather(offsets[:-1], axis)
        shape = tf.gather(tf.constant(shapes, tf.int32), axis)
        _left, width, right = tf.unstack(shape, axis=1)
        output_right = tf.gather(tf.constant(block_shapes, tf.int32)[:, 2], axis)
        row, basis, column = local // (width * right), (local // right) % width, local % right
        column += tf.where(axis == len(shapes) - 1, 0, right)
        indices = tf.gather(block_offsets[:-1], axis) + (row * width + basis) * output_right + column
        tangents = tf.gather(blocks, indices, axis=1)
        # Preserve rejection of an altered parent or nonzero lower-left block.
        valid = tf.reduce_all(encode.python_function(parent, tangents) == blocks)
        return tangents, valid

    @tf.function(input_signature=[], jit_compile=True, autograph=False)
    def mask():
        _, tangent = layout()
        return tf.tile(tf.logical_not(tangent), [component_count])

    return BlockProgram(encode, decode, mask, block_shapes, block_sizes)


def _parent_schema(parent_cores):
    parents = tuple(tf.convert_to_tensor(core, D) for core in parent_cores)
    shapes = tuple(tuple(core.shape) for core in parents)
    if len(shapes) < 2 or any(len(shape) != 3 or None in shape for shape in shapes):
        raise ValueError("parent cores require at least two fixed rank-three shapes")
    return parents, shapes


def _flatten(cores):
    return tf.concat(tuple(tf.reshape(core, [-1]) for core in cores), axis=0)


def _unpack(values, shapes, sizes):
    return tuple(tuple(tf.reshape(chunk, shape) for chunk, shape in zip(
        tf.split(row, sizes), shapes, strict=True)) for row in tf.unstack(values))


def encode_components(parent_cores, tangent_components):
    parents, shapes = _parent_schema(parent_cores)
    tangents = tuple(tuple(tf.convert_to_tensor(core, D) for core in component)
                     for component in tangent_components)
    if not tangents or any(tuple(tuple(core.shape) for core in row) != shapes for row in tangents):
        raise ValueError("core tangent shape mismatch")
    program = block_program(shapes, len(tangents))
    encode = program.encode.python_function if tf.inside_function() else program.encode
    encoded = encode(_flatten(parents), tf.stack(tuple(_flatten(row) for row in tangents)))
    return _unpack(encoded, program.shapes, program.sizes)


def decode_components(parent_cores, residual_components):
    parents, shapes = _parent_schema(parent_cores)
    components = tuple(tuple(tf.convert_to_tensor(core, D) for core in row)
                       for row in residual_components)
    program = block_program(shapes, len(components))
    if any(tuple(tuple(core.shape) for core in row) != program.shapes for row in components):
        raise ValueError("product-rule block has the wrong shape")
    decode = program.decode.python_function if tf.inside_function() else program.decode
    decoded, valid = decode(_flatten(parents), tf.stack(tuple(_flatten(row) for row in components)))
    tf.debugging.assert_equal(valid, True, message="product-rule parent or zero block mismatch")
    return _unpack(decoded, shapes, tuple(prod(shape) for shape in shapes))


def released_coordinate_mask(parent_cores, component_count):
    _, shapes = _parent_schema(parent_cores)
    mask = block_program(shapes, component_count).mask
    return mask.python_function() if tf.inside_function() else mask()
