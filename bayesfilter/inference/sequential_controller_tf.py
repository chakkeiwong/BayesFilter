"""Enclosing native sequential locator, refinement, terminal geometry and records.

This internal controller preserves the public numerical method. Its bounded
owner serializes mutable locator observations and owns callback-dependent
factories as one construction/tracing scope. Progress delivery is outside the numerical call.
"""

from threading import RLock

import tensorflow as tf

from bayesfilter.inference.mass_matrix_tf import precision_program
from bayesfilter.inference.program_cache_scope import ProgramCacheScope
from bayesfilter.inference.sequential_batched_locator_tf import (
    BufferedBatchedLocator,
    batched_locator_program,
)
from bayesfilter.inference.sequential_lifecycle_tf import lifecycle_program
from bayesfilter.inference.sequential_locator_tf import scalar_locator_program
from bayesfilter.inference.sequential_refinement_tf import refinement_program
from bayesfilter.inference.sequential_selection_tf import replay_program
from bayesfilter.inference.sequential_terminal_tf import terminal_program

D = tf.float64
_LOCK = RLock()
_LAST_CONTROLLER = None


def _movement_records(history, origin, scale):
    """Compute completed observation columns without host numerical feedback."""
    refined = history['refine']
    attempts = refined['attempts']
    last = attempts['last']
    evaluated = (refined['action'] >= 2) & attempts['last_evaluated']

    def position(rows):
        return (rows - origin[None, :]) / scale[None, :]

    def norm(rows):
        return tf.linalg.norm(rows * scale[None, :], axis=1)

    return {
        'pre_position_z': position(history['center_before']),
        'search_position_z': position(refined['selected_center']),
        'evaluated_proposal_position_z': position(last['position']),
        'terminal_position_z': position(refined['center']),
        'pre_value': history['value_before'], 'search_value': refined['selected_value'],
        'evaluated_proposal_value': last['value'], 'terminal_value': refined['center_value'],
        'pre_score_norm': norm(history['score_before']),
        'search_score_norm': norm(refined['selected_score']),
        'evaluated_proposal_score_norm': norm(last['score']),
        'terminal_score_norm': norm(refined['center_score']),
        'radius_before': history['radius_before'], 'radius_after': refined['radius_after'],
        'search_recentered': refined['search_recentered'],
        'proposal_accepted': refined['action'] == 2, 'proposal_evaluated': evaluated,
    }


class SequentialController:
    """Own a compiled endpoint and optional private observation resources."""

    def __init__(self, scalar, batched, locator_batch, start_count, dimension,
                 config, search_count, *, progress=False, device=None,
                 jit_compile=True, trace_capacity=None):
        self.dependency_scope = ProgramCacheScope()
        with self.dependency_scope.activate():
            self._initialize(scalar, batched, locator_batch, start_count, dimension,
                config, search_count, progress=progress, device=device,
                jit_compile=jit_compile, trace_capacity=trace_capacity)
            # Trace while the same owner scope is active, including factories
            # resolved lazily from nested graph bodies. This performs no target
            # evaluation and keeps source-compatible unsupported-callback errors.
            self.compiled.get_concrete_function()

    def _initialize(self, scalar, batched, locator_batch, start_count, dimension,
                 config, search_count, *, progress=False, device=None,
                 jit_compile=True, trace_capacity=None):
        cfg = config
        self.recorder = None
        locator_args = (start_count, dimension, cfg.locator_standardized_box_radius,
            cfg.locator_gradient_tolerance, cfg.locator_max_iterations,
            cfg.locator_max_line_search_iterations, cfg.locator_stopping_condition)
        center_first = cfg.locator_policy == 'center_first'
        use_batch = not center_first and locator_batch is not None and start_count > 1
        if center_first:
            locate = replay_program(scalar, start_count, dimension, jit_compile=jit_compile)
        elif use_batch:
            if progress:
                self.recorder = BufferedBatchedLocator(scalar, locator_batch, *locator_args,
                    device=device, jit_compile=jit_compile, capacity=trace_capacity)
                locate = self.recorder.compiled
            else:
                locate = batched_locator_program(scalar, locator_batch, *locator_args,
                    jit_compile=jit_compile)
        else:
            locate = scalar_locator_program(scalar, *locator_args, jit_compile=jit_compile)
        refine = refinement_program(scalar, batched, dimension, cfg, search_count,
            jit_compile=jit_compile)
        terminal = terminal_program(scalar, batched, dimension, cfg, jit_compile=jit_compile)
        lifecycle = lifecycle_program(refine, terminal, dimension, cfg, search_count,
            jit_compile=jit_compile)
        lifecycle_shape = lifecycle.get_concrete_function().structured_outputs
        mass = precision_program(dimension, dense=True, jit_compile=jit_compile)

        @tf.function(input_signature=[tf.TensorSpec([start_count, dimension], D),
            tf.TensorSpec([dimension], D)], jit_compile=jit_compile, autograph=False)
        def execute(starts, scale):
            if center_first:
                selected = locate(starts)
                located = {'selected': selected,
                    'exact_evaluations': tf.constant(start_count, tf.int32),
                    'objective_evaluations': tf.constant(0, tf.int32)}
            else:
                located = locate(starts, scale)
                selected = located['selected']
            overflow = (located['trace_overflow'] if use_batch and progress else tf.constant(False))
            evaluations = (tf.cast(located['exact_evaluations'], tf.int64)
                + tf.cast(located['objective_evaluations'], tf.int64))
            # Priority matches public rejection order. Overflow is a reporting
            # failure and must prevent every subsequent lifecycle target call.
            status = tf.where(overflow, 1, tf.where(selected['finite_count'] == 0, 2,
                tf.where(evaluations > cfg.max_exact_evaluations, 3, 0)))
            empty = tf.nest.map_structure(lambda v: tf.zeros(v.shape, v.dtype), lifecycle_shape)
            result = tf.cond(status == 0,
                lambda: lifecycle(selected['position'], selected['value'], selected['score'], scale, evaluations),
                lambda: empty)
            history = result['history']
            observations = {
                'selected_max_score': tf.reduce_max(tf.abs(scale * selected['score'])),
                'initial_score_norm': tf.linalg.norm(scale * selected['score']),
                'initial_position_z': tf.zeros([dimension], D),
                'max_score_before': tf.reduce_max(tf.abs(scale[None, :] * history['score_before']), axis=1),
                'max_score_after': tf.reduce_max(tf.abs(scale[None, :] * history['refine']['center_score']), axis=1),
                'evaluations_after': history['evaluations_before'] + tf.cast(tf.where(
                    history['terminal_called'], history['terminal']['evaluations'],
                    history['refine']['evaluations']), tf.int64),
            }
            movement = (_movement_records(history, selected['position'], scale)
                if cfg.record_refinement_movement_diagnostics else {})

            def prepare_mass():
                inverse_scale = tf.math.reciprocal(scale)
                precision = (result['terminal']['record']['projected_precision_z']
                    * inverse_scale[:, None] * inverse_scale[None, :])
                return mass(precision, tf.constant(0., D), tf.constant(cfg.eigenvalue_floor, D),
                    tf.constant(cfg.max_condition_number, D))

            prepared = tf.cond((status == 0) & (result['status'] == 0), prepare_mass,
                lambda: (tf.zeros([dimension, dimension], D), tf.zeros([dimension, dimension], D),
                    tf.zeros([8], D), tf.zeros([4], tf.bool), tf.constant(False)))
            # The public preparation API historically materializes its result
            # across a host boundary. Keep that external derivative boundary;
            # analytical target scores and internal optimizer derivatives stay.
            return tf.nest.map_structure(tf.stop_gradient, {
                'status': status, 'locator': located, 'locator_evaluations': evaluations,
                'lifecycle': result, 'observations': observations, 'movement': movement,
                'precision': prepared[0], 'covariance': prepared[1],
                'mass_diagnostics': prepared[2], 'mass_flags': prepared[3],
                'mass_diagonal_valid': prepared[4]})

        self.compiled = execute

    def __call__(self, starts, scale):
        # A completed scalar status synchronizes every output before resource
        # reuse. The caller formats immutable result tensors after release.
        with _LOCK:
            result = self.compiled(starts, scale)
            if int(result['status']) == 1:
                raise RuntimeError(f'Batched locator progress capacity {self.recorder.capacity} exceeded; '
                    f"{int(result['locator']['trace_count'])} objective calls completed. "
                    'No partial progress was delivered.')
            return result


def sequential_controller(scalar, batched, locator_batch, start_count, dimension,
                          config, search_count, *, progress=False, device=None,
                          jit_compile=True):
    """Reuse one callback/signature owner; never cache numerical results."""
    global _LAST_CONTROLLER
    with _LOCK:
        key = (start_count, dimension, config, search_count, progress, device, jit_compile)
        previous = _LAST_CONTROLLER
        if (previous is not None and previous[0] is scalar and previous[1] is batched
                and previous[2] is locator_batch and previous[3] == key):
            return previous[4]
        _LAST_CONTROLLER = None
        del previous
        owner = SequentialController(scalar, batched, locator_batch, start_count, dimension,
            config, search_count, progress=progress, device=device, jit_compile=jit_compile)
        _LAST_CONTROLLER = (scalar, batched, locator_batch, key, owner)
        return owner


def clear_sequential_controller_cache():
    """Release the public owner; does not claim dependency/native eviction."""
    global _LAST_CONTROLLER
    with _LOCK:
        _LAST_CONTROLLER = None
