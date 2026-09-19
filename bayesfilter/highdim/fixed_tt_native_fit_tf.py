"""Tensor ALS updates and post-run fit records for fixed TT execution.

This preserves the local weighted ridge ALS extension, its sweep schedule,
column scaling and condition veto. It is not the author's adaptive TT-cross.
The rank-aware QR/COD solver implements the same augmented least-squares objective.
Hashes and string-valued reports are assembled only from completed histories.
"""

import math
from collections import OrderedDict
from dataclasses import replace

import tensorflow as tf
from tensorflow.python.eager import record
from tensorflow.python.eager.polymorphic_function.polymorphic_function import (
    OptionalXlaContext,
)

from bayesfilter.highdim import fitting as old
from bayesfilter.highdim.diagnostics import HighDimStatus
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.tt_native_control_tf import (
    fixed_basis_rows,
    fixed_core_matrices,
    pack_tensors,
    row_environments,
)
from bayesfilter.ops.compiled_tensor_program_tf import _capture_pullback_coefficient
from bayesfilter.ops.qr_lstsq_tf import (
    BACKEND,
    condition_number,
    make_complete_orthogonal_lstsq,
)

D = tf.float64
_REASONS = (
    "core_update_accepted",
    "nonfinite_design_matrix",
    "nonfinite_normal_equations",
    "scaled_augmented_lstsq_failure",
    "scaled_augmented_condition_number_veto",
    "nonfinite_solution",
    "static_resource_gate",
)


class NativeFixedTTFit:
    """Prepared schema with a tensor-only callable usable inside an outer XLA."""

    def __init__(
        self, basis, points, weights, config, initial_cores, *, jit_compile=True,
        _report_only=False,
    ):
        self.basis, self.points, self.weights = basis, points, weights
        self.config = replace(config, solver_backend=BACKEND)
        self.jit_compile = jit_compile
        self.shapes = tuple(tuple(core.values.shape) for core in initial_cores)
        self.width = max(math.prod(shape) for shape in self.shapes)
        self.steps = len(config.sweep_order) * config.max_sweeps
        self.schedule = tf.constant(config.sweep_order, tf.int32)
        fitter = old.FixedTTFitter()
        fitter._validate_config_for_basis(basis, config)
        self.gates = tuple(
            fitter._check_design_budget(basis, points, initial_cores, axis, config)
            for axis in range(len(self.shapes))
        )
        self.initial = self.pack(tuple(core.values for core in initial_cores))
        if _report_only:
            return
        self.basis_rows = fixed_basis_rows(basis, points, initial_cores, jit_compile=jit_compile)
        self.branches = self._branches()

    def pack(self, values):
        return tf.stack(
            tuple(
                tf.pad(tf.reshape(value, [-1]), [[0, self.width - math.prod(shape)]])
                for value, shape in zip(values, self.shapes)
            )
        )

    def unpack(self, packed):
        return tuple(
            TTCore(tf.reshape(packed[axis, : math.prod(shape)], shape))
            for axis, shape in enumerate(self.shapes)
        )

    def design_matrix(self, packed, axis, shape):
        cores = self.unpack(packed)
        matrices = fixed_core_matrices(self.basis_rows, cores)
        # The contraction still visits every core in its original order.
        # Padding only permits a tensor coordinate to select its environments;
        # the exact local ranks are restored before forming the design.
        if isinstance(axis, int):
            left = row_environments(matrices)[axis]
            right = row_environments(matrices, reverse=True)[axis]
        else:
            left_rows, _ = pack_tensors(row_environments(matrices))
            right_rows, _ = pack_tensors(row_environments(matrices, reverse=True))
            left = left_rows[axis, :, :shape[0]]
            right = right_rows[axis, :, :shape[2]]
        phi = self.basis_rows[axis, :, :shape[1]]
        return tf.reshape(
            tf.einsum("na,nl,nb->nalb", left, phi, right),
            [self.points.shape[0], math.prod(shape)],
        )

    def _branches(self):
        # Rank-one updates share the same scalar environment operations. Keep
        # higher-rank coordinates specialized: moving their slice across the
        # compiled design graph changes rounding in ill-conditioned histories.
        schemas = tuple((shape, gate["status"] != HighDimStatus.OK.value,
                         None if shape[0] == shape[2] == 1 else axis)
                        for axis, (shape, gate) in enumerate(zip(self.shapes, self.gates, strict=True)))
        unique = tuple(dict.fromkeys(schemas))
        self.branch_indices = tf.constant(tuple(unique.index(schema) for schema in schemas), tf.int32)

        def branch(shape, resource_blocked, fixed_axis):
            if resource_blocked:
                # Static resource rejection at this scheduled axis. Earlier
                # accepted updates are retained, matching the eager oracle.
                def blocked(packed, target, axis):
                    return (
                        packed, tf.zeros([self.width], D), tf.zeros([self.width], D),
                        tf.constant(0.0, D), tf.constant(0.0, D),
                        tf.constant(0.0, D), tf.constant(6),
                    )
                return blocked

            columns = math.prod(shape)
            solver = make_complete_orthogonal_lstsq(self.points.shape[0] + columns, columns)
            schema = (shape, resource_blocked, fixed_axis)
            if fixed_axis is None and schemas.count(schema) == 1:
                fixed_axis = schemas.index(schema)

            def update(packed, target, axis):
                if fixed_axis is not None:
                    axis = fixed_axis
                design = self.design_matrix(packed, axis, shape)
                normal, rhs = old._normal_equations(
                    design, target, self.weights, self.config.ridge
                )
                finite_design = tf.reduce_all(tf.math.is_finite(design))
                finite_normal = tf.reduce_all(
                    tf.math.is_finite(normal)
                ) & tf.reduce_all(tf.math.is_finite(rhs))

                def valid_input():
                    scales, raw, floor = old._weighted_column_scales(
                        design, self.weights, self.config.column_scale_floor
                    )
                    root_weights = tf.sqrt(self.weights)
                    matrix = tf.concat(
                        (
                            design / scales[None, :] * root_weights[:, None],
                            tf.linalg.diag(
                                tf.sqrt(tf.constant(self.config.ridge, D)) / scales
                            ),
                        ),
                        0,
                    )
                    response = tf.concat(
                        ((target * root_weights)[:, None], tf.zeros([columns, 1], D)), 0
                    )
                    finite = tf.reduce_all(tf.math.is_finite(matrix)) & tf.reduce_all(
                        tf.math.is_finite(response)
                    )

                    def solve():
                        condition = condition_number(
                            matrix, jit_compile=self.jit_compile
                        )
                        accepted = tf.math.is_finite(condition) & (
                            condition <= self.config.condition_number_veto
                        )
                        solution = tf.cond(
                            accepted,
                            lambda: solver(matrix, response)[:, 0] / scales,
                            lambda: packed[axis, :columns],
                        )
                        code = tf.where(
                            accepted,
                            tf.where(tf.reduce_all(tf.math.is_finite(solution)), 0, 5),
                            4,
                        )
                        return solution, condition, code

                    solution, cond, code = tf.cond(
                        finite,
                        solve,
                        lambda: (
                            packed[axis, :columns],
                            tf.constant(float("inf"), D),
                            tf.constant(3),
                        ),
                    )
                    unscaled = condition_number(normal, jit_compile=self.jit_compile)
                    return solution, scales, raw, floor, cond, unscaled, code

                def invalid_input():
                    return (
                        packed[axis, :columns],
                        tf.zeros([columns], D),
                        tf.zeros([columns], D),
                        tf.constant(0.0, D),
                        tf.constant(float("inf"), D),
                        tf.constant(float("inf"), D),
                        tf.where(finite_design, 2, 1),
                    )

                solution, scales, raw, floor, cond, unscaled, code = tf.cond(
                    finite_design & finite_normal, valid_input, invalid_input
                )
                pad = lambda value: tf.pad(value, [[0, self.width - columns]])
                candidate = tf.tensor_scatter_nd_update(
                    packed, [[axis]], [pad(solution)]
                )
                return (
                    tf.where(code == 0, candidate, packed),
                    pad(scales),
                    pad(raw),
                    floor,
                    cond,
                    unscaled,
                    code,
                )

            signature = [
                tf.TensorSpec(self.initial.shape, D),
                tf.TensorSpec([self.points.shape[0]], D),
                tf.TensorSpec([], tf.int32),
            ]
            forward = tf.function(update, input_signature=signature, autograph=False)

            @tf.function(
                input_signature=[*signature, tf.TensorSpec(self.initial.shape, D)],
                autograph=False,
            )
            def accepted_backward(packed, target, axis, upstream):
                if fixed_axis is not None:
                    axis = fixed_axis
                with tf.GradientTape() as tape:
                    tape.watch((packed, target))
                    design = self.design_matrix(packed, axis, shape)
                    scales, _, _ = old._weighted_column_scales(
                        design, self.weights, self.config.column_scale_floor
                    )
                    root_weights = tf.sqrt(self.weights)
                    matrix = tf.concat(
                        (
                            design / scales[None, :] * root_weights[:, None],
                            tf.linalg.diag(
                                tf.sqrt(tf.constant(self.config.ridge, D)) / scales
                            ),
                        ),
                        0,
                    )
                    response = tf.concat(
                        ((target * root_weights)[:, None], tf.zeros([columns, 1], D)), 0
                    )
                    solution = solver(matrix, response)[:, 0] / scales
                    padded = tf.pad(solution, [[0, self.width - columns]])
                    result = tf.tensor_scatter_nd_update(packed, [[axis]], [padded])
                return tape.gradient(
                    result, (packed, target), output_gradients=upstream
                )

            forward_concrete = forward.get_concrete_function()
            backward_concrete = None

            def complete_backward():
                nonlocal backward_concrete
                if backward_concrete is None:
                    with forward_concrete.graph.as_default(), OptionalXlaContext(self.jit_compile):
                        backward_concrete = accepted_backward.get_concrete_function()
                return backward_concrete

            @tf.custom_gradient
            def call(packed, target, axis):
                # Bind the fixed design's primal captures without constructing
                # unused accepted-update pullbacks during value-only fitting.
                graph = tf.compat.v1.get_default_graph()
                captures = {value.ref(): graph.capture(value)
                            for value in forward_concrete.captured_inputs}
                with record.stop_recording():
                    result = forward(packed, target, axis)

                def grad(*upstream):
                    backward = complete_backward()

                    def capture(value):
                        if value.ref() in captures:
                            return captures[value.ref()]
                        return _capture_pullback_coefficient(value, tf.compat.v1.get_default_graph())

                    coefficients = [capture(value) for value in backward.captured_inputs]
                    gradients = tf.cond(
                        result[-1] == 0,
                        lambda: backward._call_flat(
                            [packed, target, axis, upstream[0]], captured_inputs=coefficients
                        ),
                        lambda: (upstream[0], tf.zeros_like(target)),
                    )
                    return (*gradients, None)

                # Scales and conditioning reports originally materialized on
                # the host; only updated core values carry score derivatives.
                return (
                    result[0],
                    *tf.nest.map_structure(tf.stop_gradient, result[1:]),
                ), grad

            compiled = tf.function(call, input_signature=signature, autograph=False)
            compiled.get_concrete_function()
            return compiled

        return tuple(branch(*schema) for schema in unique)

    def __call__(self, target, initial):
        branches = self.branches
        precores = tf.TensorArray(
            D, self.steps, element_shape=initial.shape, clear_after_read=False
        )
        scales = tf.TensorArray(D, self.steps, element_shape=[self.width])
        raw = tf.TensorArray(D, self.steps, element_shape=[self.width])
        diagnostics = tf.TensorArray(D, self.steps, element_shape=[3])
        codes = tf.TensorArray(tf.int32, self.steps, element_shape=[])

        def update(index, packed, alive, precores, scales, raw, diagnostics, codes):
            axis = self.schedule[index % len(self.config.sweep_order)]
            precores = precores.write(index, tf.stop_gradient(packed))

            def take_step():
                return tf.switch_case(
                    self.branch_indices[axis],
                    tuple(lambda fn=fn: fn(packed, target, axis) for fn in branches),
                )

            def stopped():
                return (
                    packed,
                    tf.zeros([self.width], D),
                    tf.zeros([self.width], D),
                    tf.constant(0.0, D),
                    tf.constant(0.0, D),
                    tf.constant(0.0, D),
                    tf.constant(-1),
                )

            candidate, scale, norm, floor, cond, unscaled, code = tf.cond(
                alive, take_step, stopped
            )
            return (
                index + 1,
                candidate,
                alive & (code == 0),
                precores,
                scales.write(index, scale),
                raw.write(index, norm),
                diagnostics.write(index, tf.stack((floor, cond, unscaled))),
                codes.write(index, code),
            )

        _, final, alive, precores, scales, raw, diagnostics, codes = tf.while_loop(
            lambda i, *_: i < self.steps,
            update,
            (
                tf.constant(0),
                initial,
                tf.constant(True),
                precores,
                scales,
                raw,
                diagnostics,
                codes,
            ),
            maximum_iterations=self.steps,
            parallel_iterations=1,
        )
        # Fit residuals and snapshots were host diagnostics in the original
        # route. Disconnect their inputs, so a score tape does not build unused
        # loop adjoints merely because reports share the callable's outputs.
        matrices = fixed_core_matrices(
            self.basis_rows, self.unpack(tf.stop_gradient(final))
        )
        left = row_environments(matrices)[-1]
        prediction = tf.einsum("na,nab->nb", left, matrices[-1])[:, 0]
        residual = old._weighted_rms_residual(
            prediction, tf.stop_gradient(target), self.weights
        )
        return {
            "cores": final,
            "precores": precores.stack(),
            "scales": scales.stack(),
            "raw": raw.stack(),
            "diagnostics": diagnostics.stack(),
            "codes": codes.stack(),
            "residual": residual,
            "valid": alive & tf.math.is_finite(residual),
        }

    def report(self, history, target, initial, *, branch_seed, initialization_rule, samples=None):
        """Serialize completed fit histories; nothing here drives numerical ALS."""
        cfg = self.config
        records, hashes = [], []
        codes = history["codes"].numpy().tolist()
        status = HighDimStatus.OK
        reason = stop = "max_sweeps_exhausted"
        for index, code in enumerate(codes):
            if code == -1:
                break
            axis = cfg.sweep_order[index % len(cfg.sweep_order)]
            columns = math.prod(self.shapes[axis])
            pre = self.unpack(history["precores"][index])
            hashes.append(
                old._hash_tensors(
                    "pre_update_cores_hash.v1", [core.values for core in pre]
                )
            )
            scale, raw = (
                history["scales"][index, :columns],
                history["raw"][index, :columns],
            )
            floor, cond, unscaled = history["diagnostics"][index].numpy().tolist()
            record = dict(
                self.gates[axis],
                core_index=axis,
                sweep_index=index // len(cfg.sweep_order),
                termination_reason=_REASONS[code],
                stop_condition_triggered="none",
                status=HighDimStatus.OK.value,
            )
            if code in (0, 3, 4):
                lo, hi = float(tf.reduce_min(scale)), float(tf.reduce_max(scale))
                metric = cfg.ridge / tf.square(scale)
                policy = old._stabilization_policy_payload(cfg)
                record.update(
                    policy,
                    condition_number=old._finite_number_or_text(cond),
                    condition_warning=cond > cfg.condition_number_warning,
                    condition_number_warning=cfg.condition_number_warning,
                    condition_number_veto=cfg.condition_number_veto,
                    condition_number_semantics="scaled_augmented_solve_condition",
                    unscaled_normal_condition_number=old._finite_number_or_text(
                        unscaled
                    ),
                    unscaled_normal_condition_warning=unscaled
                    > cfg.condition_number_warning,
                    unscaled_normal_condition_veto=not math.isfinite(unscaled)
                    or unscaled > cfg.condition_number_veto,
                    stabilization_policy=policy,
                    scale_floor=floor,
                    scale_floor_rule=old._SCALE_FLOOR_RULE,
                    column_scale_min=lo,
                    column_scale_max=hi,
                    column_scale_spread=hi / lo if lo > 0 else "inf",
                    column_scale_hash=old._hash_tensor_or_none(
                        "fixed_tt_fit_column_scales.v1", scale
                    ),
                    raw_column_norm_min=float(tf.reduce_min(raw)),
                    raw_column_norm_max=float(tf.reduce_max(raw)),
                    raw_column_norm_zero_count=int(
                        tf.reduce_sum(tf.cast(raw == 0, tf.int32))
                    ),
                    ridge_metric_summary={
                        "ridge": cfg.ridge,
                        "transformed_ridge_rule": old._TRANSFORMED_RIDGE_RULE,
                        "coordinate_system": "scaled_z_coordinates",
                        "min_diagonal": float(tf.reduce_min(metric)),
                        "max_diagonal": float(tf.reduce_max(metric)),
                    },
                    transformed_system_condition_number=old._finite_number_or_text(
                        cond
                    ),
                    scaled_augmented_condition_number=old._finite_number_or_text(cond),
                    scaled_augmented_rows=int(self.points.shape[0]) + columns,
                    scaled_augmented_cols=columns,
                    nonclaims=(
                        "stable solve is not a Phase 6 diagnostic pass",
                        "stable solve is fixed_hmc_adaptation not source_faithful",
                    ),
                )
            if code:
                status = (
                    HighDimStatus(self.gates[axis]["status"]) if code == 6
                    else HighDimStatus.CONDITION_NUMBER_VETO
                    if code in (3, 4)
                    else HighDimStatus.NONFINITE_VALUE
                )
                reason = self.gates[axis]["gate"] if code == 6 else _REASONS[code]
                stop = status.value
                record.update(status=status.value, stop_condition_triggered=stop, termination_reason=reason)
            records.append(record)
        residual = history["residual"] if status is HighDimStatus.OK else None
        if residual is not None and not bool(tf.math.is_finite(residual)):
            status, reason, stop = (
                HighDimStatus.NONFINITE_VALUE,
                "nonfinite_fit_residual",
                HighDimStatus.NONFINITE_VALUE.value,
            )
        holdout_residual = None
        if status is HighDimStatus.OK and "holdout_residual" in history:
            holdout_residual = history["holdout_residual"]
            if not bool(tf.math.is_finite(holdout_residual)):
                status, reason, stop = (
                    HighDimStatus.NONFINITE_VALUE, "nonfinite_holdout_residual",
                    HighDimStatus.NONFINITE_VALUE.value,
                )
            elif bool(holdout_residual > cfg.holdout_tolerance):
                status, reason, stop = (
                    HighDimStatus.HOLDOUT_RESIDUAL_VETO, "holdout_residual_veto",
                    HighDimStatus.HOLDOUT_RESIDUAL_VETO.value,
                )
        return old.FixedTTFitter()._finalize_result(
            product_basis=self.basis,
            samples=samples or old.FixedTTFitSampleBatch(self.points, target, self.weights),
            config=cfg,
            cores=self.unpack(history["cores"]),
            branch_seed=branch_seed,
            measure_convention=self.basis.convention,
            initial_core_hash=old._hash_tensors(
                "initial_cores_hash.v1", [core.values for core in self.unpack(initial)]
            ),
            initialization_rule=initialization_rule,
            update_records=tuple(records),
            environment_rebuild_hashes=tuple(hashes),
            status=status,
            termination_reason=reason,
            stop_condition=stop,
            fit_residual=residual,
            holdout_residual=holdout_residual,
        )


_FIT_CACHE = OrderedDict()


def native_fixed_tt_fit(
    fitter, product_basis, samples, config, initial_cores, branch_seed,
    measure_convention, initialization_rule, *, jit_compile=True,
):
    """Bounded preparation cache and reporting boundary for the public fit API."""
    from bayesfilter.highdim.bases import ProductBasis
    from bayesfilter.highdim.diagnostics import assert_density_matches_mass

    if not isinstance(product_basis, ProductBasis):
        raise TypeError("product_basis must be a ProductBasis")
    if not isinstance(samples, old.FixedTTFitSampleBatch):
        raise TypeError("samples must be a FixedTTFitSampleBatch")
    if not isinstance(config, old.FixedTTFitConfig):
        raise TypeError("config must be a FixedTTFitConfig")
    if measure_convention != product_basis.convention:
        raise ValueError(f"measure_convention: {HighDimStatus.MEASURE_MISMATCH.value}")
    assert_density_matches_mass(measure_convention)
    initialization_rule = str(initialization_rule)
    if not initialization_rule.strip():
        raise ValueError("initialization_rule must be nonempty")
    fitter._validate_config_for_basis(product_basis, config)
    fitter._validate_batch_dimension(product_basis, samples)
    cores = fitter._validate_initial_cores(product_basis, measure_convention, config, initial_cores)
    shapes = tuple(tuple(core.values.shape) for core in cores)
    key = (id(product_basis), tuple(samples.points.shape), config, shapes,
        None if samples.holdout_points is None else tuple(samples.holdout_points.shape),
        bool(jit_compile))
    if key in _FIT_CACHE:
        _FIT_CACHE.move_to_end(key)
        _, program, _ = _FIT_CACHE[key]
    else:
        core_specs = tuple(tf.TensorSpec(shape, D) for shape in shapes)
        signature = [tf.TensorSpec(samples.points.shape, D), tf.TensorSpec(samples.weights.shape, D),
            tf.TensorSpec(samples.target_values.shape, D), core_specs]
        if samples.holdout_points is not None:
            signature.extend([tf.TensorSpec(samples.holdout_points.shape, D),
                tf.TensorSpec(samples.holdout_values.shape, D), tf.TensorSpec(samples.holdout_weights.shape, D)])

        @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
        def program(points, weights, target, core_values, *holdout):
            typed = tuple(TTCore(value) for value in core_values)
            native = NativeFixedTTFit(product_basis, points, weights, config, typed, jit_compile=jit_compile)
            result = native(target, native.initial)
            if holdout:
                holdout_basis = fixed_basis_rows(product_basis, holdout[0], typed)
                matrices = fixed_core_matrices(holdout_basis, native.unpack(result["cores"]))
                left = row_environments(matrices)[-1]
                prediction = tf.einsum("na,nab->nb", left, matrices[-1])[:, 0]
                result["holdout_residual"] = old._weighted_rms_residual(prediction, holdout[1], holdout[2])
            return result

        # Only the fixed basis/configuration is captured. All sample values,
        # weights, targets and core values cross the stable tensor signature.
        _FIT_CACHE[key] = (product_basis, program, config)
        if len(_FIT_CACHE) > 16:
            _FIT_CACHE.popitem(last=False)
    arguments = (samples.points, samples.weights, samples.target_values, tuple(core.values for core in cores))
    if samples.holdout_points is not None:
        arguments += (samples.holdout_points, samples.holdout_values, samples.holdout_weights)
    history = program(*arguments)
    reporter = NativeFixedTTFit(product_basis, samples.points, samples.weights, config, cores,
        jit_compile=jit_compile, _report_only=True)
    return reporter.report(
        history, samples.target_values, reporter.initial, branch_seed=branch_seed,
        initialization_rule=initialization_rule, samples=samples,
    )
