"""Diagnostic objective histories; instrumentation is never a runtime fitter."""

import hashlib
import json
from pathlib import Path

import pytest
import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_optimization_barrier

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference.sequential_structured_fit_tf import _cached_structured_fit
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_lifecycle_original import _without_new_execution_metadata

D = tf.float64
CAPACITY = 4096
FIELDS = ('center_score_z', 'training_offsets_z', 'training_scores_z',
          'holdout_offsets_z', 'holdout_scores_z', 'training_weights')


def _replay_program(config, count):
    """Current standalone objective at saved states, with explicit data operands."""
    @tf.function(input_signature=[tf.TensorSpec([count, 14], D),
        tf.TensorSpec([5], D), tf.TensorSpec([10, 5], D), tf.TensorSpec([10, 5], D),
        tf.TensorSpec([10], D), tf.TensorSpec([2], tf.int32)],
        jit_compile=True, autograph=False)
    def replay(states, center, offsets, scores, weights, anchors):
        weights = xla_optimization_barrier(input=[weights])[0]
        weights /= tf.reduce_sum(weights)
        response = center[None, :] - scores
        rows = tf.TensorArray(D, count, element_shape=[15])

        def step(index, rows):
            raw = tf.gather(states, index)
            with tf.GradientTape() as tape:
                tape.watch(raw)
                covariance, _, _ = factor._decode_covariance(raw, dimension=5,
                    anchors=(tf.gather(anchors, 0), tf.gather(anchors, 1)), config=config)
                precision = tf.linalg.cholesky_solve(tf.linalg.cholesky(covariance), tf.eye(5, dtype=D))
                prediction = tf.einsum('ij,bj->bi', precision, offsets)
                per_row = tf.reduce_mean(tf.square(prediction - response), axis=1)
                value = tf.reduce_sum(weights * per_row)
            gradient = tape.gradient(value, raw)
            return index + 1, rows.write(index, tf.concat([value[None], gradient], 0))

        _, rows = tf.while_loop(lambda index, _: index < count, step,
            (tf.constant(0), rows), maximum_iterations=count, parallel_iterations=1)
        return rows.stack()

    return replay


@pytest.mark.parametrize('data_name', ['original', 'current'])
def test_crossed_factor_objective_trajectories(data_name, monkeypatch, request):
    directory = Path(request.config.getoption('xmlpath')).parent
    source = directory.parent / 'run-01826/lifecycle-factor-inputs.json'
    archived = json.loads(source.read_text())
    checkpoint = FrozenCheckpoint('3582b4ac', 'factor_trajectory_' + data_name)
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    cfg = current.SequentialMapCovarianceConfig(structured_holdout_score_relative_rmse=.001)
    factor_cfg = factor.FactorCorrelationGeometryConfig(factor_count=2,
        max_condition_number=cfg.max_condition_number,
        holdout_score_relative_rmse=cfg.structured_holdout_score_relative_rmse)
    prepared = dict(archived['prepared'][data_name])
    prepared.update({key: tf.constant(prepared[key], D) for key in FIELDS})
    report = {'role': 'diagnostic_crossed_factor_objective_trajectories',
        'data': data_name, 'baseline': checkpoint.revision,
        'archived_input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'input_sha256': {key: hashlib.sha256(tf.io.serialize_tensor(prepared[key]).numpy()).hexdigest()
                         for key in FIELDS},
        'capacity': CAPACITY, 'arms': {}, 'runtime_modified': False,
        'nonclaims': ['Instrumentation must reproduce archived complete records before attribution.',
            'Standalone objective replay can have different fusion from the enclosing fitter.',
            'Small objective errors do not waive any public-field comparison.']}
    minimize = factor.tfp.optimizer.lbfgs_minimize
    trajectories = {}
    gpu = bool(tf.config.list_logical_devices('GPU'))
    report['gpu_execution'] = gpu
    for name, module in [('original', original), ('current', current)]:
        print(f'FACTOR_TRAJECTORY {data_name} {name} fit_start', flush=True)
        inputs = prepared if name == 'original' else {**prepared,
            '_native_factor_data': {**prepared, 'active_training_rows': tf.constant(10)}}
        archived_record = archived['records'][data_name + '_data_' + name + '_fit']
        # The saved crossed-data fixture was measured on CPU. GPU attribution
        # needs its own uninstrumented comparator on identical prepared arrays.
        uninstrumented = (_without_new_execution_metadata(module._fit_factor_from_data(
            inputs, factor_count=2, config=cfg)) if gpu else archived_record)
        counter = tf.Variable(0, dtype=tf.int64, trainable=False)
        history = tf.Variable(tf.zeros([CAPACITY, 29], D), trainable=False)

        def capture(objective, *, initial_position, counter=counter, history=history, **kwargs):
            def observed(raw):
                value, gradient = objective(raw)
                index = counter.read_value()
                row = tf.concat([raw, value[None], gradient], 0)
                write = history.scatter_nd_update(tf.reshape(tf.minimum(index, CAPACITY - 1), [1, 1]), row[None, :])
                with tf.control_dependencies([write]):
                    advanced = counter.assign_add(1)
                with tf.control_dependencies([advanced]):
                    return tf.identity(value), tf.identity(gradient)
            return minimize(observed, initial_position=initial_position, **kwargs)

        factor._make_factor_program.cache_clear()
        _cached_structured_fit.cache_clear()
        try:
            with monkeypatch.context() as context:
                context.setattr(factor.tfp.optimizer, 'lbfgs_minimize', capture)
                result = module._fit_factor_from_data(inputs, factor_count=2, config=cfg)
        finally:
            factor._make_factor_program.cache_clear()
            _cached_structured_fit.cache_clear()
        count = int(counter)
        record = _without_new_execution_metadata(result)
        trajectory = history[:min(count, CAPACITY)]
        trajectories[name] = trajectory
        report['arms'][name] = {'record': record, 'observed_evaluations': count,
            'overflow': count > CAPACITY, 'trajectory': trajectory.numpy().tolist(),
            'uninstrumented_record': uninstrumented,
            'differences_from_archived_cpu': _record_differences(record, archived_record),
            'differences_from_uninstrumented': _record_differences(record, uninstrumented)}
        print(f'FACTOR_TRAJECTORY {data_name} {name} fit_complete evaluations={count}', flush=True)

    report['frozen_source_sha256'] = checkpoint.hashes()
    valid = all(not arm['overflow'] and not arm['differences_from_uninstrumented']
                for arm in report['arms'].values())
    report['instrumentation_valid'] = valid
    with (directory / f'factor-trajectory-{data_name}-captured.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    if valid:
        for name, trajectory in trajectories.items():
            print(f'FACTOR_TRAJECTORY {data_name} {name} replay_start', flush=True)
            assert bool(tf.reduce_all(tf.math.is_finite(trajectory)))
            count = int(trajectory.shape[0])
            arm = report['arms'][name]
            assert count == arm['record']['diagnostics']['optimizer_objective_evaluations']
            replay = _replay_program(factor_cfg, count)(trajectory[:, :14],
                prepared['center_score_z'], prepared['training_offsets_z'],
                prepared['training_scores_z'], prepared['training_weights'],
                tf.constant(arm['record']['anchor_indices'], tf.int32))
            assert bool(tf.reduce_all(tf.math.is_finite(replay)))
            errors = tf.abs(replay - trajectory[:, 14:])
            arm['current_standalone_replay'] = replay.numpy().tolist()
            arm['replay_value_abs_error'] = errors[:, 0].numpy().tolist()
            arm['replay_gradient_max_abs_error'] = tf.reduce_max(errors[:, 1:], axis=1).numpy().tolist()
            arm['final_gradient_max_abs'] = float(tf.reduce_max(tf.abs(trajectory[-1, 15:])))
        common = min(int(value.shape[0]) for value in trajectories.values())
        original_rows, current_rows = trajectories['original'][:common], trajectories['current'][:common]
        report['paired_evaluation_errors'] = {
            'position_max_abs': tf.reduce_max(tf.abs(original_rows[:, :14] - current_rows[:, :14]), axis=1).numpy().tolist(),
            'value_abs': tf.abs(original_rows[:, 14] - current_rows[:, 14]).numpy().tolist(),
            'gradient_max_abs': tf.reduce_max(tf.abs(original_rows[:, 15:] - current_rows[:, 15:]), axis=1).numpy().tolist(),
        }
    with (directory / f'factor-trajectory-{data_name}.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    assert valid, 'Instrumentation changed a complete record; trajectory attribution is invalid'
