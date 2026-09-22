"""Native ordered block transactions; complete center remains a runtime operand."""

import math
from threading import RLock

import tensorflow as tf

from bayesfilter.inference.block_conditional_tf import ConditionalSequentialProgram
from bayesfilter.inference.block_coordinate_center import (
    _block_maxima_core,
    _score_summary_core,
    _target_program,
    _terminal_core,
)
from bayesfilter.inference.program_cache_scope import ProgramCacheScope

D = tf.float64
I = tf.int64
_EPS = math.sqrt(math.ulp(1.0))
_LOCK = RLock()
_LAST_CONTROLLER = None


def _put(rows, index, value):
    return tf.tensor_scatter_nd_update(rows, [[index]], value[None])


def _branch(index, program, block, dimension, bounds, cfg, full):
    """Build one static block branch; numerical state is supplied at execution."""
    start, stop = block.start, block.stop

    def execute(state, history, sequential, scale):
        center, value, score = state['center'], state['value'], state['score']
        raw = program.compiled(center, scale)
        sequential = sequential[:index] + (raw,) + sequential[index + 1:]
        life = raw['lifecycle']
        evaluations = tf.where(raw['status'] == 0, life['evaluations'], raw['locator_evaluations'])
        allowed = (raw['status'] == 0) & ((life['status'] == 0) | (life['status'] == 3) | (life['status'] == 4))
        accounting = (evaluations >= 0) & (evaluations <= block.sequential_config.max_exact_evaluations)
        mass_valid = (raw['status'] != 0) | (life['status'] != 0) | tf.reduce_all(raw['mass_flags'][:3])
        error = tf.where(evaluations < 0, 3, tf.where(~mass_valid, 4, 0))
        state = {**state, 'error': error,
            'status': tf.where(accounting & mass_valid, 0, 3),
            'physical_rows': state['physical_rows'] + tf.where(accounting & mass_valid, evaluations, tf.constant(0, I)),
            'sequential_rows': state['sequential_rows'] + tf.where(accounting & mass_valid, evaluations, tf.constant(0, I))}
        history = {**history, 'executed': _put(history['executed'], index, tf.constant(True)),
            'center_before': _put(history['center_before'], index, center),
            'center_after': _put(history['center_after'], index, center),
            'value_before': _put(history['value_before'], index, value),
            'value_after': _put(history['value_after'], index, value),
            'score_before': _put(history['score_before'], index, score),
            'score_after': _put(history['score_after'], index, score),
            'evaluations': _put(history['evaluations'], index, evaluations)}

        def handoff():
            history_ready = {**history, 'has_record': _put(history['has_record'], index, tf.constant(True))}

            def replay():
                candidate = tf.concat([center[:start], life['center'], center[stop:]], axis=0)
                new_value, new_score, finite = full(candidate)
                record = {**history_ready,
                    'center_after': _put(history_ready['center_after'], index, candidate),
                    'value_after': _put(history_ready['value_after'], index, new_value),
                    'score_after': _put(history_ready['score_after'], index, new_score),
                    'displacement': _put(history_ready['displacement'], index, tf.linalg.norm(candidate - center))}
                replayed = {**state, 'physical_rows': state['physical_rows'] + 1,
                    'error': tf.where(finite, 0, 2)}

                def commit():
                    count = state['accepted'] + 1
                    trace = _put(state['trace'], count, candidate / scale)
                    threshold = tf.constant(cfg.repeat_threshold_factor * _EPS * max(1., math.sqrt(dimension)), D)
                    moved = tf.linalg.norm(trace[count] - trace[count - 1]) > threshold
                    older = tf.range(bounds.shape[0] + 1) < count - 1
                    repeat = moved & tf.reduce_any(older & (tf.linalg.norm(trace - trace[count], axis=1) <= threshold))
                    previous = tf.maximum(count - 2, 0)
                    two_step = (count >= 2) & moved & (tf.linalg.norm(trace[count - 1] - trace[previous]) > threshold) & (tf.linalg.norm(trace[count] - trace[previous]) <= threshold)
                    maxima = _block_maxima_core(new_score, scale, bounds)
                    floors = tf.constant(_EPS, D) * tf.maximum(state['post_maxima'], 1.)
                    prior = tf.range(bounds.shape[0]) < index
                    materials = prior & (maxima - state['post_maxima'] > floors) & (maxima > tf.constant(cfg.reversal_ratio, D) * state['post_maxima'])
                    reversal = tf.reduce_any(materials)
                    repeated, returned = state['repeat'] | repeat, state['two_step'] | two_step
                    status = tf.where(reversal & cfg.stop_on_material_reversal, 7,
                        tf.where(repeated | returned, 8, 0))
                    committed = {**replayed, 'center': candidate, 'value': new_value, 'score': new_score,
                        'accepted': count, 'trace': trace, 'repeat': repeated, 'two_step': returned,
                        'material': state['material'] | reversal, 'status': status,
                        'post_maxima': _put(state['post_maxima'], index, maxima[index])}
                    record = {**history_ready,
                        'center_after': _put(history_ready['center_after'], index, candidate),
                        'value_after': _put(history_ready['value_after'], index, new_value),
                        'score_after': _put(history_ready['score_after'], index, new_score),
                        'displacement': _put(history_ready['displacement'], index, tf.linalg.norm(candidate - center)),
                        'committed': _put(history_ready['committed'], index, tf.constant(True)),
                        'prior_maxima': _put(history_ready['prior_maxima'], index, state['post_maxima']),
                        'maxima': _put(history_ready['maxima'], index, maxima),
                        'floors': _put(history_ready['floors'], index, floors),
                        'materials': _put(history_ready['materials'], index, materials)}
                    return committed, record

                def reject():
                    return ({**replayed, 'status': tf.where(finite, 6, 0),
                        'rejections': state['rejections'] + tf.cast(finite, tf.int32)}, record)

                return tf.cond(finite & (new_value >= value), commit, reject)

            return tf.cond(allowed, replay,
                lambda: ({**state, 'status': tf.constant(4)}, history_ready))

        state, history = tf.cond(accounting & mass_valid, handoff, lambda: (state, history))
        return state, history, sequential

    return execute


class BlockController:
    """Own every conditional callback graph for one static ordered block family."""

    def __init__(self, scalar, batched, dimension, blocks, config, *, progress=False, jit_compile=True):
        self.dependency_scope = ProgramCacheScope()
        with self.dependency_scope.activate():
            self._initialize(scalar, batched, dimension, blocks, config,
                progress=progress, jit_compile=jit_compile)
            self.compiled.get_concrete_function()

    def _initialize(self, scalar, batched, dimension, blocks, config, *, progress, jit_compile):
        count, cfg = len(blocks), config
        self.programs = tuple(ConditionalSequentialProgram(scalar, batched, dimension, block,
            progress=progress, jit_compile=jit_compile) for block in blocks)
        shapes = tuple(program.compiled.get_concrete_function().structured_outputs for program in self.programs)
        bounds = tf.constant([(block.start, block.stop) for block in blocks], tf.int32)
        # The callback belongs to this scope; each branch owns its numerical operands.
        full = _target_program(scalar, dimension, None)
        branches = tuple(_branch(index, program, block, dimension, bounds, cfg, full)
            for index, (program, block) in enumerate(zip(self.programs, blocks, strict=True)))

        @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D)],
                     jit_compile=jit_compile, autograph=False)
        def execute(initial, scale):
            value, score, finite = full(initial)
            state = {'center': initial, 'value': value, 'score': score,
                'physical_rows': tf.constant(1, I), 'sequential_rows': tf.constant(0, I),
                'accepted': tf.constant(0), 'rejections': tf.constant(0),
                'error': tf.where(finite, 0, 1), 'status': tf.constant(0),
                'trace': tf.concat([(initial / scale)[None], tf.zeros([count, dimension], D)], axis=0),
                'post_maxima': tf.zeros([count], D), 'repeat': tf.constant(False),
                'two_step': tf.constant(False), 'material': tf.constant(False)}
            history = {'executed': tf.zeros([count], tf.bool), 'has_record': tf.zeros([count], tf.bool),
                'center_before': tf.zeros([count, dimension], D), 'center_after': tf.zeros([count, dimension], D),
                'score_before': tf.zeros([count, dimension], D), 'score_after': tf.zeros([count, dimension], D),
                'value_before': tf.zeros([count], D), 'value_after': tf.zeros([count], D),
                'evaluations': tf.zeros([count], I), 'displacement': tf.zeros([count], D),
                'committed': tf.zeros([count], tf.bool), 'prior_maxima': tf.zeros([count, count], D),
                'maxima': tf.zeros([count, count], D), 'floors': tf.zeros([count, count], D),
                'materials': tf.zeros([count, count], tf.bool)}
            sequential = tf.nest.map_structure(lambda row: tf.zeros(row.shape, row.dtype), shapes)

            def step(index, state, history, sequential):
                # A finite heterogeneous branch table is graph topology only.
                choices = tuple(lambda branch=branch: branch(state, history, sequential, scale) for branch in branches)
                state, history, sequential = tf.switch_case(index, choices)
                return index + 1, state, history, sequential

            index, state, history, sequential = tf.while_loop(
                lambda index, state, *_: (index < count) & (state['error'] == 0) & (state['status'] == 0),
                step, (tf.constant(0), state, history, sequential), maximum_iterations=count, parallel_iterations=1)
            complete = (index == count) & (state['status'] == 0) & (state['error'] == 0)
            no_worse, progress_value = _terminal_core(score, state['score'], scale, bounds, value,
                state['value'], tf.constant(cfg.require_scheduled_block_maxima_no_worse))
            initial_l2, initial_max = _score_summary_core(score, scale)
            final_l2, final_max = _score_summary_core(state['score'], scale)

            def decrease(before, after):
                return before - after > tf.constant(_EPS, D) * tf.maximum(tf.abs(before), 1.)

            summary = {'objective_nondecreasing': state['value'] >= value,
                'objective_progress_resolvable': decrease(-value, -state['value']),
                'score_l2_progress_resolvable': decrease(initial_l2, final_l2),
                'score_max_progress_resolvable': decrease(initial_max, final_max)}
            return tf.nest.map_structure(tf.stop_gradient, {'state': state, 'history': history,
                'sequential': sequential, 'initial_value': value, 'initial_score': score,
                'initial_l2': initial_l2, 'initial_max': initial_max, 'final_l2': final_l2, 'final_max': final_max,
                'completed': complete, 'no_worse': complete & no_worse, 'summary': summary,
                'status': tf.where(complete, tf.where(progress_value, 1, 2), state['status'])})

        self.compiled = execute


    def __call__(self, initial, scale):
        with _LOCK:
            result = self.compiled(initial, scale)
            # Synchronize completion before another caller reuses target resources.
            int(result['state']['error'])
            return result


def block_controller(scalar, batched, dimension, blocks, config, *, progress=False, jit_compile=True):
    global _LAST_CONTROLLER
    with _LOCK:
        key = (dimension, blocks, config, progress, jit_compile)
        previous = _LAST_CONTROLLER
        if previous is not None and previous[0] is scalar and previous[1] is batched and previous[2] == key:
            return previous[3]
        _LAST_CONTROLLER = None
        del previous
        owner = BlockController(scalar, batched, dimension, blocks, config,
            progress=progress, jit_compile=jit_compile)
        _LAST_CONTROLLER = (scalar, batched, key, owner)
        return owner


def clear_block_controller_cache():
    global _LAST_CONTROLLER
    with _LOCK:
        _LAST_CONTROLLER = None
