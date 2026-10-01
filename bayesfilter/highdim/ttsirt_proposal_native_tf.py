"""XLA preparation of the existing frozen grid-CDF proposal branch.

The finite-grid method and reordered prefix conditioning remain an extension.
Python packs heterogeneous immutable schemas; the complete date recurrence,
including categorical selection and proposal correction, executes in tensors.
"""

import hashlib
import math
from collections import OrderedDict
from dataclasses import asdict
from itertools import accumulate

import tensorflow as tf

from bayesfilter.highdim.ttsirt_native_tf import transport_arguments, transport_program

_PROGRAMS = OrderedDict()


def _transport_schema(transport):
    from bayesfilter.highdim.filtering import _product_basis_payload
    from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import _update_hash

    density = transport.density
    digest = hashlib.sha256()
    _update_hash(digest, _product_basis_payload(density.sqrt_tt.product_basis))
    _update_hash(digest, density.defensive_density.manifest_payload())
    _update_hash(digest, asdict(transport.cdf_config))
    return (type(density.defensive_density),
            tuple(type(basis) for basis in density.sqrt_tt.product_basis.bases),
            tuple(tuple(core.values.shape) for core in density.sqrt_tt.cores),
            digest.digest())


def proposal_program(initial_transport, transports, coordinate_map, count, *, jit_compile=True):
    """Return a cached program and packed numerical inputs, never frozen values."""
    key = (id(initial_transport), tuple(id(item) for item in transports),
           id(coordinate_map), count, bool(jit_compile))
    if key in _PROGRAMS:
        _PROGRAMS.move_to_end(key)
        program, width = _PROGRAMS[key][3:]
    else:
        dimension, steps = initial_transport.dimension, len(transports)
        initial_inverse = transport_program(initial_transport, "inverse", count,
            jit_compile=jit_compile).python_function
        initial_pdf = transport_program(initial_transport, "pdf", count,
            jit_compile=jit_compile).python_function
        # Group by immutable numerical schema, not date or core coefficients.
        schemas, representatives, schema_indices = {}, [], []
        for transport in transports:
            schema = _transport_schema(transport)
            if schema not in schemas:
                schemas[schema] = len(representatives)
                representatives.append(transport)
            schema_indices.append(schemas[schema])
        width = max((sum(core.values.shape.num_elements() for core in item.density.sqrt_tt.cores)
                     for item in transports), default=0)

        def transition_kernel(transport):
            inverse = transport_program(transport, "inverse", count, dimension,
                jit_compile=jit_compile).python_function
            logpdf = transport_program(transport, "conditional_logpdf", count, dimension,
                jit_compile=jit_compile).python_function
            shapes = tuple(core.values.shape for core in transport.density.sqrt_tt.cores)
            sizes = tuple(math.prod(shape) for shape in shapes)
            offsets = (0, *tuple(accumulate(sizes)))

            def evaluate(flat, controls, parent, uniform):
                cores = tuple(tf.reshape(flat[offsets[index]:offsets[index+1]], shape)
                              for index, shape in enumerate(shapes))
                arguments = (cores, controls[0], controls[1], controls[2])
                local, code = inverse(*arguments, parent, uniform)
                log_q, pdf_code = logpdf(*arguments, parent, local)
                return local, log_q, tf.where(code == 0, pdf_code, code)
            return evaluate

        kernels = tuple(transition_kernel(item) for item in representatives)
        initial_specs = tuple(tf.TensorSpec(core.values.shape, tf.float64)
                              for core in initial_transport.density.sqrt_tt.cores)
        signature = (initial_specs, tf.TensorSpec([3], tf.float64),
            tf.TensorSpec([steps, width], tf.float64), tf.TensorSpec([steps, 3], tf.float64),
            tf.TensorSpec([dimension, count], tf.float64),
            tf.TensorSpec([steps, count], tf.float64), tf.TensorSpec([steps, count], tf.float64),
            tf.TensorSpec([steps, dimension, count], tf.float64))

        @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
        def program(initial_cores, initial_controls, packed, controls, initial_reference,
                    ancestor_uniforms, auxiliary_log, transition_reference):
            empty = tf.zeros([0, count], tf.float64)
            arguments = (initial_cores, initial_controls[0], initial_controls[1], initial_controls[2])
            initial_local, code = initial_inverse(*arguments, empty, initial_reference)
            initial_physical, initial_logdet = coordinate_map.forward(tf.transpose(initial_local))
            initial_density, pdf_code = initial_pdf(*arguments, empty, initial_local)
            code = tf.where(code == 0, pdf_code, code)
            initial_log_q = tf.math.log(initial_density) - initial_logdet
            states = tf.TensorArray(tf.float64, steps+1, element_shape=[count, dimension])
            states = states.write(0, initial_physical)
            ancestors = tf.TensorArray(tf.int32, steps, element_shape=[count])
            densities = tf.TensorArray(tf.float64, steps, element_shape=[count])
            if steps:
                indices = tf.constant(schema_indices, tf.int32)

                def step(index, previous, states, ancestors, densities, code):
                    cdf = tf.math.cumsum(tf.exp(auxiliary_log[index]))
                    cdf = tf.concat([cdf[:-1], tf.ones([1], tf.float64)], axis=0)
                    ancestor = tf.searchsorted(cdf, ancestor_uniforms[index], side="right", out_type=tf.int32)
                    parent_local, _ = coordinate_map.inverse(tf.gather(previous, ancestor))
                    branches = tuple(lambda fn=fn: fn(packed[index], controls[index],
                        tf.transpose(parent_local), transition_reference[index]) for fn in kernels)
                    local, log_q, next_code = tf.switch_case(indices[index], branches)
                    physical, logdet = coordinate_map.forward(tf.transpose(local))
                    return (index+1, physical, states.write(index+1, physical),
                            ancestors.write(index, ancestor), densities.write(index, log_q-logdet),
                            tf.where(code == 0, next_code, code))

                _, _, states, ancestors, densities, code = tf.while_loop(
                    lambda index, *_: index < steps, step,
                    (tf.constant(0), initial_physical, states, ancestors, densities, code),
                    maximum_iterations=steps, parallel_iterations=1)
            return states.stack(), initial_log_q, ancestors.stack(), densities.stack(), code

        _PROGRAMS[key] = (initial_transport, transports, coordinate_map, program, width)
        if len(_PROGRAMS) > 16:
            _PROGRAMS.popitem(last=False)
    initial = transport_arguments(initial_transport)
    packed = tuple(tf.concat(tuple(tf.reshape(core.values, [-1]) for core in item.density.sqrt_tt.cores), 0)
                   for item in transports)
    controls = tuple(tf.stack(transport_arguments(item)[1:]) for item in transports)
    arguments = (initial[0], tf.stack(initial[1:]),
        tf.stack(tuple(tf.pad(row, [[0, width-row.shape[0]]]) for row in packed))
            if packed else tf.zeros([0, width], tf.float64),
        tf.stack(controls) if controls else tf.zeros([0, 3], tf.float64))
    return program, arguments
