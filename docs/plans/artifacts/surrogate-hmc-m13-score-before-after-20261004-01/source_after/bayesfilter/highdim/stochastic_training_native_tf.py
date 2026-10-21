"""Compiled execution for the existing P75 stochastic density extension.

This preserves its finite objective and optimizer, without a Zhao-Cui
source-faithfulness claim. Records and heterogeneous variable schemas stay on
the host; core contractions and every numerical training step execute in XLA.
"""

from collections import OrderedDict
from dataclasses import fields, is_dataclass
from functools import lru_cache
from inspect import signature
from math import prod
from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.highdim import centered_tt_native_tf as tt
from bayesfilter.ops.stateless_random_tf import philox_normal_float64

D = tf.float64


def core_values(cores, basis, points):
    return tt._values(basis, tt.pack_components((cores,)), points)[:, 0]


def core_square_mass(cores, basis):
    packed = tt.pack_components((cores,))
    return tf.reshape(tt._cross(basis, packed, packed, tf.zeros([1, 0], D)), [])


def flat_cores(cores):
    return tf.concat(tuple(tf.reshape(core, [-1]) for core in cores), 0)


def core_penalties(cores):
    values = flat_cores(cores)
    return tf.reduce_sum(tf.abs(values)), tf.reduce_sum(tf.square(values))


@lru_cache(maxsize=16)
def random_core_program(shapes):
    distinct = tuple(dict.fromkeys(shapes))
    indices = tuple(distinct.index(shape) for shape in shapes)
    maximum = max(prod(shape) for shape in shapes)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)], jit_compile=True, autograph=False)
    def generate(seed):
        def at_axis(axis):
            def branch(shape):
                def draw():
                    values = .05 * philox_normal_float64(shape, tf.stack([seed, axis + 1]))
                    return tf.pad(tf.reshape(values, [-1]), [[0, maximum-prod(shape)]])
                return draw

            branches = tuple(branch(shape) for shape in distinct)
            return tf.switch_case(tf.gather(indices, axis), branches)

        return tf.map_fn(at_axis, tf.range(len(shapes)),
            fn_output_signature=tf.TensorSpec([maximum], D), parallel_iterations=1)

    return generate


def random_cores(shapes, seed):
    values = random_core_program(shapes)(tf.convert_to_tensor(seed, tf.int32))
    return tuple(tf.reshape(values[axis, :prod(shape)], shape)
                 for axis, shape in enumerate(shapes))


def _arguments(trainer, operation, arguments):
    bound = signature(getattr(trainer, operation)).bind(**arguments)
    bound.apply_defaults()
    values = dict(bound.arguments)
    if "batch" in values:
        batch = values["batch"]
        values["batch"] = {field.name: getattr(batch, field.name) for field in fields(batch)
                           if tf.is_tensor(getattr(batch, field.name))}
    values = tf.nest.map_structure(lambda value: None if value is None else tf.convert_to_tensor(value, D), values)
    flat = tf.nest.flatten(values)
    present = tuple(index for index, value in enumerate(flat) if value is not None)
    tensors = tuple(flat[index] for index in present)
    specs = tuple(tf.TensorSpec(value.shape, value.dtype) for value in tensors)
    structure = tf.nest.map_structure(lambda value: value is not None, values)
    layout = repr(structure)
    extent = len(flat)

    def restore(tensors):
        supplied = dict(zip(present, tensors, strict=True))
        result = tf.nest.pack_sequence_as(structure, tuple(supplied.get(index) for index in range(extent)))
        if "batch" in result:
            result["batch"] = SimpleNamespace(**result["batch"], role="", provenance_label="")
        return result

    return tensors, specs, layout, restore


def _outputs(record):
    if not is_dataclass(record):
        return record
    return {field.name: getattr(record, field.name) for field in fields(record)
            if tf.is_tensor(getattr(record, field.name))}


def _validity(trainer, operation, arguments, outputs):
    flat = flat_cores(tf.nest.flatten(outputs))
    valid = tf.reduce_all(tf.math.is_finite(flat))
    if operation in ("objective", "corrected_heldout_density_metric"):
        valid &= (outputs["normalizer"] > trainer.config.normalizer_floor) & (outputs["rho_min"] > 0.)
    elif operation == "log_density":
        valid &= trainer.normalizer() > trainer.config.normalizer_floor
    if operation == "square_root_prefit_objective":
        weight = arguments["reference_l2_weight"]
        floor = arguments["scale_floor"]
        if weight is not None:
            valid &= tf.math.is_finite(weight) & (weight >= 0.)
        if floor is not None:
            valid &= tf.math.is_finite(floor) & (floor > 0.)
    return valid


def call_method(trainer, operation, *, result_type=None, result_metadata=None,
                state_attribute="cores", validity_fn=None, **arguments):
    """Bounded explicit tensor signatures; variable values never enter the cache."""
    values, specs, layout, restore = _arguments(trainer, operation, arguments)
    state = getattr(trainer, state_attribute)
    cores = tuple(tf.convert_to_tensor(core, D) for core in tf.nest.flatten(state))
    core_specs = tuple(tf.TensorSpec(core.shape, D) for core in cores)
    key = (operation, state_attribute, validity_fn, core_specs, specs, layout)
    if not hasattr(trainer, "_execution_programs"):
        trainer._execution_programs = OrderedDict()
    cache = trainer._execution_programs
    if key not in cache:
        count = len(cores)

        def numerical(*tensors):
            proxy = object.__new__(type(trainer))
            proxy.__dict__.update(trainer.__dict__)
            setattr(proxy, state_attribute, tf.nest.pack_sequence_as(state, tensors[:count]))
            supplied = restore(tensors[count:])
            outputs = _outputs(getattr(proxy, operation)(**supplied))
            valid = (_validity if validity_fn is None else validity_fn)(proxy, operation, supplied, outputs)
            return outputs, valid

        forward = tf.function(numerical, input_signature=(*core_specs, *specs),
                              jit_compile=True, autograph=False)
        output_structure = forward.get_concrete_function().structured_outputs[0]
        output_specs = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, D), output_structure)
        input_count = count + len(specs)

        @tf.function(input_signature=(*core_specs, *specs, *tf.nest.flatten(output_specs)),
                     jit_compile=True, autograph=False)
        def backward(*tensors):
            query, cotangents = tensors[:input_count], tensors[input_count:]
            with tf.GradientTape() as tape:
                tape.watch(query)
                outputs, _valid = numerical(*query)
            return tape.gradient(tf.nest.flatten(outputs), query, output_gradients=cotangents,
                                 unconnected_gradients=tf.UnconnectedGradients.ZERO)

        @tf.custom_gradient
        def differentiable(*tensors):
            outputs, valid = forward(*tf.nest.map_structure(tf.stop_gradient, tensors))

            def pullback(*cotangents):
                return backward(*tensors, *cotangents[:-1])

            return (*tf.nest.flatten(outputs), valid), pullback

        differentiable.compiled = forward
        cache[key] = differentiable, output_structure
        if len(cache) > 16:
            cache.popitem(last=False)
    cache.move_to_end(key)
    evaluate, structure = cache[key]
    result = evaluate(*cores, *values)
    tf.debugging.assert_equal(result[-1], True, message="invalid compiled density calculation")
    output = tf.nest.pack_sequence_as(structure, result[:-1])
    return output if result_type is None else result_type(**output, **(result_metadata or {}))


def checked_optimizer_update(optimizer, variables, gradients, clip_norm, valid):
    """Guard assignments in both enclosing graphs and standalone XLA calls."""
    if any(gradient is None for gradient in gradients):
        raise ValueError("missing gradient for at least one trainable core")
    clipped, norm = tf.clip_by_global_norm(gradients, tf.constant(clip_norm, D))
    valid &= tf.reduce_all(tf.math.is_finite(flat_cores(gradients)))
    valid &= tf.reduce_all(tf.math.is_finite(flat_cores((*clipped, norm))))

    def apply():
        optimizer.apply_gradients(zip(clipped, variables, strict=True))
        return tf.reduce_all(tf.math.is_finite(flat_cores(variables)))

    valid = tf.cond(valid, apply, lambda: tf.constant(False))
    # Assertions can be removed by XLA. Preserve rejection in a returned field.
    return tf.where(valid, norm, tf.constant(float("nan"), D)), valid


def _optimizer_update(trainer, optimizer, operation, supplied):
    with tf.GradientTape() as tape:
        terms = getattr(trainer, operation)(**supplied)
    gradients = tape.gradient(terms.total_loss, trainer.variables)
    outputs = _outputs(terms)
    norm, valid = checked_optimizer_update(optimizer, trainer.variables, gradients,
        trainer.config.gradient_clip_norm, _validity(trainer, operation, supplied, outputs))
    return outputs, norm, valid


def optimizer_step(trainer, batch, optimizer, *, prefit=False, **options):
    from bayesfilter.highdim.stochastic_density_training import (
        P75ObjectiveTerms,
        P75PrefitTerms,
    )

    operation = "square_root_prefit_objective" if prefit else "objective"
    result_type = P75PrefitTerms if prefit else P75ObjectiveTerms
    if tf.inside_function():
        outputs, norm, valid = _optimizer_update(trainer, optimizer, operation, {"batch": batch, **options})
        tf.debugging.assert_equal(valid, True, message="invalid stochastic density training step")
        return result_type(**outputs, gradient_norm=norm)
    values, specs, layout, restore = _arguments(trainer, operation, {"batch": batch, **options})
    if hasattr(optimizer, "build"):
        optimizer.build(trainer.variables)
    key = (id(optimizer), operation, specs, layout)
    cache = trainer._optimizer_programs
    if key not in cache:
        @tf.function(input_signature=specs, jit_compile=True, autograph=False)
        def update(*tensors):
            return _optimizer_update(trainer, optimizer, operation, restore(tensors))

        cache[key] = optimizer, update
        if len(cache) > 16:
            cache.popitem(last=False)
    cache.move_to_end(key)
    outputs, norm, valid = cache[key][1](*values)
    tf.debugging.assert_equal(valid, True, message="invalid stochastic density training step")
    return result_type(**outputs, gradient_norm=norm)


def program_cache():
    return OrderedDict()


def calibrate_normalizer(trainer, target_log_normalizer):
    """Compile the existing one-core amplitude rescale for a fixed trainer."""
    if not hasattr(trainer, "_calibration_program"):
        @tf.function(input_signature=[tf.TensorSpec([], D)], jit_compile=True, autograph=False)
        def calibrate(log_target):
            target = tf.exp(log_target)
            square_mass = trainer.sqrt_square_normalizer()
            defensive_mass = trainer.defensive_density.normalizer(
                trainer.config.product_basis.convention.mass_measure)
            defensive = trainer.config.tau * defensive_mass
            valid = tf.math.is_finite(target) & tf.math.is_finite(square_mass)
            valid &= (target > defensive) & (square_mass > 0.)

            def rescale():
                scale = tf.sqrt((target - defensive) / square_mass)
                trainer.variables[0].assign(trainer.variables[0] * scale)
                return scale, trainer.normalizer()

            scale, realized = tf.cond(valid, rescale,
                lambda: (tf.constant(float("nan"), D), tf.constant(float("nan"), D)))
            return scale, realized, target, valid

        trainer._calibration_program = calibrate
    scale, realized, target, valid = trainer._calibration_program(
        tf.reshape(tf.convert_to_tensor(target_log_normalizer, D), []))
    tf.debugging.assert_equal(valid, True, message="invalid normalizer target or square-root mass")
    tf.debugging.assert_near(realized, target, atol=tf.constant(1e-12, D) * (1. + tf.abs(target)))
    return scale
