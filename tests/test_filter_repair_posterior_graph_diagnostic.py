"""Diagnostic localization of an empty result from the non-default graph path."""

import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import posterior_initializer_controller_tf as native
from bayesfilter.inference.joint_center_tf import make_joint_center_program
from bayesfilter.inference.posterior_initializer_controller_tf import (
    PreparedPosteriorInitializer,
)
from bayesfilter.inference.posterior_local_initializer import (
    _EligibilityTrackingEvaluator,
)
from bayesfilter.inference.posterior_movement_tf import make_posterior_movement_program
from bayesfilter.inference.program_cache_scope import ProgramCacheScope
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_posterior_initializer_controller import original_module
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_posterior_local_initializer import (
    _gaussian_callbacks,
    _initializer_config,
    _movement_config,
    _thresholds,
)

D = tf.float64


@pytest.mark.parametrize("mode", ["full", "no_meta", "locator", "movement", "full_xla", "no_function", "no_inline_locator", "no_inline_movement", "no_inline_curvature", "no_dependency", "no_pruning", "no_arithmetic", "no_constant", "scalar_dependencies", "curvature", "fitter", "raw_fitter", "functional_control", "scoped_executor", "functional_fit_ops", "functional_owner_ops"])
def test_empty_graph_result_localization(mode, request, monkeypatch):
    marked_ops = []
    if mode in ("functional_fit_ops", "functional_owner_ops"):
        from tensorflow.core.framework.attr_value_pb2 import AttrValue

        from bayesfilter.inference import (
            fixed_center_fitting_tf,
            fixed_center_selection_tf,
            fixed_center_stability_tf,
        )

        def mark(outputs):
            if not tf.executing_eagerly():
                op = tf.nest.flatten(outputs)[0].op
                while op.type in ("Identity", "IdentityN"):
                    op = op.inputs[0].op
                assert op.type in ("If", "StatelessIf", "While", "StatelessWhile"), op.type
                op._set_attr("_lower_using_switch_merge", AttrValue(b=False))
                marked_ops.append({"graph": op.graph.name, "name": op.name, "type": op.type})
            return outputs

        class GraphOperations:
            def __getattr__(self, name):
                return getattr(tf, name)

            def cond(self, *args, **kwargs):
                return mark(tf.cond(*args, **kwargs))

            def while_loop(self, *args, **kwargs):
                return mark(tf.while_loop(*args, **kwargs))

        for module in (fixed_center_fitting_tf, fixed_center_selection_tf, fixed_center_stability_tf):
            monkeypatch.setattr(module, "tf", GraphOperations())
        if mode == "functional_owner_ops":
            monkeypatch.setattr(native, "tf", GraphOperations())
    # Optimizer options are diagnostic in this fresh process, never a runtime repair.
    if mode == "no_meta":
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    if mode == "functional_control":
        # Diagnostic only: ask TF to retain functional If/While, as XLA does.
        from tensorflow.python.ops import control_flow_util_v2
        monkeypatch.setattr(control_flow_util_v2, "_DISABLE_LOWER_USING_SWITCH_MERGE", True)
    if mode == "no_function":
        tf.config.optimizer.set_experimental_options({"function_optimization": False})
    if mode in ("no_dependency", "no_pruning", "no_arithmetic", "no_constant"):
        key = {"no_dependency": "dependency_optimization", "no_pruning": "disable_model_pruning",
            "no_arithmetic": "arithmetic_optimization", "no_constant": "constant_folding"}[mode]
        tf.config.optimizer.set_experimental_options({key: mode == "no_pruning"})
    if mode.startswith("no_inline_"):
        name = {"no_inline_locator": "make_joint_center_program",
            "no_inline_movement": "make_posterior_movement_program",
            "no_inline_curvature": "make_posterior_curvature_program"}[mode]
        stage_factory = getattr(native, name)

        def preserve_call(*args, **kwargs):
            program = stage_factory(*args, **kwargs)
            return tf.function(program.python_function, input_signature=program.input_signature,
                jit_compile=False, autograph=False, experimental_attributes={"_noinline": True})

        monkeypatch.setattr(native, name, preserve_call)
    scalar, batch = _gaussian_callbacks(np.array([.6]), np.array([[.4]]))
    config = _initializer_config(locator_gradient_tolerance=1e6)
    if mode == "full_xla":
        config = replace(config, locator_config=replace(config.locator_config, jit_compile=True))
    movement = _movement_config(seed=20260826)
    thresholds = _thresholds(1)
    initial, scale = tf.constant([0.], D), tf.constant([1.], D)
    factory = PreparedPosteriorInitializer
    source_edit = None
    if mode == "scalar_dependencies":
        source = inspect.getsource(factory)
        changed = source.replace("tf.control_dependencies(tf.nest.flatten(location))",
            'tf.control_dependencies([location["status"]])').replace(
            "tf.control_dependencies(tf.nest.flatten(curved))", 'tf.control_dependencies([curved["status"]])')
        assert changed != source
        namespace = dict(vars(native))
        exec(compile(changed, "<diagnostic_scalar_dependencies>", "exec"), namespace)  # noqa: S102 - saved diagnostic source transformation
        factory = namespace["PreparedPosteriorInitializer"]
        source_edit = {"before_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(changed.encode()).hexdigest(), "modified_source": changed}
    executor = None
    if mode == "scoped_executor":
        from tensorflow.python.eager import context
        before = context.context().function_call_options.executor_type
        with tf.experimental.function_executor_type("SINGLE_THREADED_EXECUTOR"):
            owner = factory(scalar, 1, config, movement, thresholds, batched_callback=batch)
        after = context.context().function_call_options.executor_type
        assert before == after
        executor = {"before": before, "after": after, "scope": "construction_only"}
    else:
        owner = factory(scalar, 1, config, movement, thresholds, batched_callback=batch)
    prepared = owner.prepare_clouds()
    if mode == "locator":
        def chart(point):
            tanh = tf.math.tanh(point / config.locator_box_radius)
            value, score = scalar(initial + scale * config.locator_box_radius * tanh)
            return value, score * scale * (1. - tanh**2)
        program = make_joint_center_program(chart, 1, config.locator_config, jit_compile=False)
        args = initial, scale
    elif mode in ("movement", "curvature", "fitter", "raw_fitter"):
        evaluator = _EligibilityTrackingEvaluator(scalar, dimension=1,
            max_rows=config.max_exact_evaluations, batched_fn=batch,
            eligibility_fn=None, batched_eligibility_fn=None)
        with ProgramCacheScope().activate():
            if mode == "movement":
                program = make_posterior_movement_program(evaluator, 1, config, movement, jit_compile=False)
            elif mode == "curvature":
                program = native.make_posterior_curvature_program(evaluator, 1, config, thresholds, jit_compile=False)
            else:
                from bayesfilter.inference.dense_validated_fit_tf import (
                    make_dense_validated_fit_program,
                )
                program = make_dense_validated_fit_program(1, config.replicate_count,
                    config.training_rows_per_replicate, config.selection_rows_per_replicate, config.audit_rows,
                    thresholds=thresholds, factor_max=config.factor_max,
                    dense_eigenvalue_floor=config.dense_eigenvalue_floor,
                    max_condition_number=config.max_condition_number, shrinkage_weights=config.shrinkage_weights,
                    structured_target_family=config.structured_target_family, jit_compile=False)
        if mode in ("curvature", "fitter", "raw_fitter"):
            initial = tf.constant([.6], D)
        value, score = scalar(initial)
        args = ((initial, value, score, scale, prepared["directions"], prepared["movement_offsets"], prepared["permutation_keys"])
            if mode == "movement" else (initial, value, score, scale, prepared["curvature_offsets"]))
        if mode in ("fitter", "raw_fitter"):
            offsets = prepared["curvature_offsets"][0]
            points = initial + offsets * scale
            _, scores = batch(tf.reshape(points, [-1, 1]))
            args = initial, score * scale, offsets, tf.reshape(scores * scale, offsets.shape)
            if mode == "raw_fitter":
                closure = inspect.getclosurevars(program.python_function).nonlocals
                program = closure["fit"]
                scores = args[-1]
                r, t, h, a = config.replicate_count, config.training_rows_per_replicate, config.selection_rows_per_replicate, config.audit_rows
                args = (score * scale, offsets[:r, :t], scores[:r, :t], offsets[r:2*r, :h], scores[r:2*r, :h],
                    offsets[-1, :a], scores[-1, :a], *(closure[key] for key in ("cap_values", "enabled", "rank", "weight_values",
                        "eigen_floor", "projection_cap", "require_raw_spd", "audit_cap")))
    else:
        program = owner.compiled
        args = initial, scale, prepared["directions"], prepared["movement_offsets"], prepared["permutation_keys"], prepared["curvature_offsets"]
    raw = program(*args)
    specs = tf.nest.map_structure(lambda x: {"dtype": x.dtype.name, "shape": x.shape.as_list()}, raw)
    wanted_specs = tf.nest.map_structure(lambda x: {"dtype": x.dtype.name, "shape": x.shape.as_list()},
        program.get_concrete_function().structured_outputs)
    save(request, f"posterior-graph-localization-{mode}.json", {"mode": mode,
        "actual_specs": specs, "declared_specs": wanted_specs, "raw": clean(raw), "source_edit": source_edit, "executor": executor, "marked_ops": marked_ops,
        "optimizer_options": tf.config.optimizer.get_experimental_options()})
    _equal_records(specs, wanted_specs)
    if mode not in ("locator", "movement", "curvature", "fitter", "raw_fitter"):
        from bayesfilter.inference.posterior_initializer_reporting import (
            posterior_initializer_result,
        )
        reference, hashes = original_module()
        actual = posterior_initializer_result(raw, config, movement, thresholds).payload(include_arrays=True)
        expected = reference.initialize_posterior_local_location_scale(scalar, initial, scale=scale,
            batched_value_and_score_fn=batch, config=config, movement_config=movement,
            curvature_thresholds=thresholds).payload(include_arrays=True)
        save(request, f"posterior-graph-localization-records-{mode}.json", {
            "reference_sha256": hashes, "actual": clean(actual), "expected": clean(expected)})
        _equal_records(clean(actual), clean(expected))
