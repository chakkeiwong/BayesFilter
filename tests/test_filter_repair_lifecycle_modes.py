"""Diagnostic attribution of the actual lifecycle's failed graph/XLA fields."""

import hashlib
import json
import math
from functools import partial
from pathlib import Path

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_self_adjoint_eig

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference import sequential_score_fit_tf as score_fit
from bayesfilter.inference.mass_matrix_tf import _eigh
from bayesfilter.inference.sequential_preparation_tf import cloud_program
from bayesfilter.inference.sequential_refinement_tf import refinement_program
from bayesfilter.inference.sequential_selection_tf import search_program
from bayesfilter.inference.sequential_structured_preparation_tf import (
    structured_data_program,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_sequential_terminal import _payload
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64


def _cost_records(directory):
    records, hashes = {}, {}
    for mode, run in [('graph', 1774), ('xla', 1776)]:
        path = directory.parent / f'run-{run:05d}' / 'lifecycle-memory.json'
        records[mode] = json.loads(path.read_text())
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records, hashes


def _config():
    return current.SequentialMapCovarianceConfig(locator_policy='center_first',
        refinement_geometry_policy='factor_correlation', structured_max_factors=2,
        reuse_search_scores=True, structured_holdout_score_relative_rmse=.001,
        max_attempts=2, search_sample_count=32, terminal_sample_count=24,
        max_exact_evaluations=512, initial_radius=.25, terminal_score_max_abs=1e-5,
        record_refinement_movement_diagnostics=True)


def test_terminal_mode_failure_source_and_eigensystem(request):
    directory = Path(request.config.getoption('xmlpath')).parent
    archived, hashes = _cost_records(directory)
    checkpoint = FrozenCheckpoint('cfbc32d2', 'lifecycle_terminal_modes')
    frozen = checkpoint.load('bayesfilter.inference.sequential_score_fit_tf')
    cfg = _config()
    scalar, batched, _ = _inputs(5, 32)
    train, holdout = score_fit.partition_schema(24, cfg.holdout_fraction, pair_disjoint=False)
    programs = {(label, mode): module.score_fit_program(scalar, batched, 24, 5, train, holdout,
        jit_compile=mode == 'xla') for label, module in [('frozen', frozen), ('current', score_fit)]
        for mode in ('graph', 'xla')}
    original = archived['xla']
    endpoints = [original['result'], *(item['result'] for item in original['changed_inputs'])]
    graph_endpoints = [archived['graph']['result'], *(item['result'] for item in archived['graph']['changed_inputs'])]
    observations = []
    first_arguments = None
    for index, endpoint in enumerate(endpoints):
        endpoint = endpoint['result']
        center = tf.constant(endpoint['map_candidate'], D)
        scale = tf.fill([5], tf.constant(1. if index == 0 else 1.1, D))
        radius = tf.constant(endpoint['diagnostics']['history'][-1]['radius_after'], D)
        seed = tf.constant(endpoint['diagnostics']['terminal_seed'], tf.int32)
        arguments = (center, scalar(center)[1], scale, radius, seed,
            tf.constant(cfg.ridge, D), tf.constant(cfg.eigenvalue_floor, D),
            tf.constant(cfg.max_condition_number, D), tf.constant(cfg.score_holdout_relative_rmse, D))
        first_arguments = arguments if first_arguments is None else first_arguments
        records = {}
        for (label, mode), program in programs.items():
            computed = program(*arguments)
            records[label + '_' + mode] = _payload({'record': computed, 'seed': seed,
                'has_best': computed['best_index'] >= 0}, cfg)
        for mode in ('graph', 'xla'):
            _compare(records['current_' + mode], records['frozen_' + mode])
        _compare(records['current_xla'], endpoint['diagnostics']['terminal_fit'])
        observations.append({'input_index': index, 'records': records,
            'archived_graph_fit': graph_endpoints[index]['result']['diagnostics']['terminal_fit'],
            'mode_differences': {label: _record_differences(records[label + '_xla'], records[label + '_graph'])
                for label in ('frozen', 'current')}})

    # Extract the exact frozen solve prefix; expose its pre-eigensystem matrix.
    # This instrumentation is explanatory and never replaces a runtime fit.
    source = checkpoint.sources['bayesfilter/inference/sequential_score_fit_tf.py']
    start = source.index('def fit_numerics(')
    stop = source.index('        eigenvalues, vectors = ', start)
    excerpt = source[start:stop] + '        return precision\n\n    return solve()\n'
    namespace = dict(vars(frozen))
    exec(compile(excerpt, 'frozen_pre_eigensystem_exact_excerpt', 'exec'), namespace)  # noqa: S102 - pinned diagnostic
    center, center_score, scale, radius, seed, *settings = first_arguments
    generator = cloud_program(24, 5, False).python_function

    @tf.function(input_signature=[tf.TensorSpec([5], D), tf.TensorSpec([5], D),
        tf.TensorSpec([], D), tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def data(point, spread, extent, key):
        z = generator(extent, key)
        return z, batched(point[None, :] + z * spread[None, :])[1]

    z, scores = data(center, scale, radius, seed)
    prepared = (z, scores, center_score, scale, *settings)
    precision_program = tf.function(partial(namespace['fit_numerics'], training_indices=train,
        holdout_indices=holdout, jit_compile=True),
        input_signature=[tf.TensorSpec(item.shape, item.dtype) for item in prepared],
        jit_compile=True, autograph=False)
    precision = precision_program(*prepared)
    eigen = {}
    for device, mode in [('/CPU:0', 'graph'), ('/GPU:0', 'graph'), ('/GPU:0', 'raw_xla'), ('/GPU:0', 'refined_xla')]:
        def eig(matrix, mode=mode):
            if mode == 'graph':
                return tf.linalg.eigh(matrix)
            if mode == 'refined_xla':
                return _eigh(matrix)
            return xla_self_adjoint_eig(matrix, lower=True, max_iter=100, epsilon=math.ulp(1.))

        with tf.device(device):
            program = tf.function(eig, input_signature=[tf.TensorSpec([5, 5], D)],
                jit_compile=mode != 'graph', autograph=False)
            values, vectors = program(precision)
            reconstruction = (vectors * values[None, :]) @ tf.transpose(vectors)
            residual = precision @ vectors - vectors * values[None, :]
        eigen[device + '_' + mode] = {'values': values.numpy().tolist(), 'vectors': vectors.numpy().tolist(),
            'reconstruction': reconstruction.numpy().tolist(),
            'max_residual': float(tf.reduce_max(tf.abs(residual))),
            'max_reconstruction_error': float(tf.reduce_max(tf.abs(reconstruction - precision)))}
    report = {'role': 'explanatory_terminal_failure_source_attribution', 'checkpoint': checkpoint.revision,
        'frozen_source_sha256': checkpoint.hashes(), 'cost_artifact_sha256': hashes,
        'observations': observations, 'pre_eigensystem_precision': precision.numpy().tolist(),
        'eigensystems': eigen,
        'nonclaims': ['Same-mode source parity does not waive strict graph/XLA failure.',
            'Refined eigenpairs are a diagnostic comparison only; runtime algorithms remain unchanged.']}
    with (directory / 'lifecycle-terminal-modes.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')


def test_factor_condition_failure_on_identical_optimizer_state(request):
    directory = Path(request.config.getoption('xmlpath')).parent
    archived, hashes = _cost_records(directory)
    checkpoint = FrozenCheckpoint('cfbc32d2', 'lifecycle_factor_modes')
    frozen = checkpoint.load('bayesfilter.inference.factor_correlation_geometry')
    cfg = _config()
    scalar, batched, _ = _inputs(5, 32)
    center = tf.linspace(tf.constant(.002, D), tf.constant(.004, D), 5)
    value, score = scalar(center)
    scale, radius = tf.ones([5], D), tf.constant(.25, D)
    arguments = (center, value, score, scale, radius, tf.constant(0), tf.constant(0))
    records, inputs = {}, {}
    for mode, jit in [('graph', False), ('xla', True)]:
        raw = refinement_program(scalar, batched, 5, cfg, 32, jit_compile=jit)(*arguments)
        fitted = raw['attempts']['second_fit']['computed']['fit']
        assert int(raw['attempts']['attempted']) == 2
        reference = archived[mode]['result']['result']['diagnostics']['history'][0]['fit']['diagnostics']
        _compare(float(fitted['jacobian_condition']), reference['prediction_jacobian_condition_number'])
        selected = search_program(scalar, batched, 32, 5, cfg.orthogonal_antithetic_search, jit_compile=jit)(
            center, value, score, scale, radius, tf.constant([cfg.seed[0], cfg.seed[1] + 1000]))
        data = structured_data_program(scalar, batched, 5, 20, 32, True, jit_compile=jit)(
            selected['position'], selected['score'], scale, radius,
            tf.constant([cfg.seed[0], cfg.seed[1] + 10000]), selected['search_positions'], selected['search_scores'])
        active = int(data['active_training_rows'])
        offsets = data['training_offsets_z'][:active]
        factor_cfg = factor.FactorCorrelationGeometryConfig(factor_count=2,
            max_condition_number=cfg.max_condition_number,
            holdout_score_relative_rmse=cfg.structured_holdout_score_relative_rmse)
        # The public tensor record intentionally keeps only OptimizerSummary.
        # Recover coordinates from the identical full dependency and verify
        # all retained values/counts before using its raw state for diagnosis.
        full = factor._make_factor_program(5, 42, 10, factor_cfg, jit,
            factor._prediction_jacobian_diagnostics, padded_training=True)(
                data['center_score_z'], data['training_offsets_z'], data['training_scores_z'],
                data['holdout_offsets_z'], data['holdout_scores_z'], data['training_weights'],
                data['active_training_rows'])
        summarized = {**full, 'optimizer': type(fitted['optimizer'])(
            *(getattr(full['optimizer'], name) for name in fitted['optimizer']._fields))}
        _compare(current._json_ready(summarized), current._json_ready(fitted))
        position, anchors = full['optimizer'].position, tuple(fitted['anchors'].numpy().tolist())
        inputs[mode] = {'position': position.numpy().tolist(), 'anchors': anchors,
            'offsets': offsets.numpy().tolist(), 'active_rows': active,
            'enclosed_condition': float(fitted['jacobian_condition']),
            'optimizer_iterations': int(fitted['optimizer'].num_iterations),
            'optimizer_objective_evaluations': int(fitted['optimizer'].num_objective_evaluations)}
        for label, module in [('frozen', frozen), ('current', factor)]:
            for evaluation_mode, evaluation_jit in [('graph', False), ('xla', True)]:
                factor_cfg = module.FactorCorrelationGeometryConfig(factor_count=2,
                    max_condition_number=cfg.max_condition_number,
                    holdout_score_relative_rmse=cfg.structured_holdout_score_relative_rmse)
                program = tf.function(partial(module._prediction_jacobian_diagnostics,
                    dimension=5, anchors=anchors, config=factor_cfg, jit_compile=evaluation_jit),
                    input_signature=[tf.TensorSpec(position.shape, D), tf.TensorSpec(offsets.shape, D)],
                    jit_compile=evaluation_jit, autograph=False)
                rank, condition = program(position, offsets)
                records[f'{mode}_state_{label}_{evaluation_mode}'] = {'rank': int(rank), 'condition': float(condition)}
        for evaluation_mode in ('graph', 'xla'):
            _compare(records[f'{mode}_state_current_{evaluation_mode}'], records[f'{mode}_state_frozen_{evaluation_mode}'])
    report = {'role': 'explanatory_factor_condition_source_attribution', 'checkpoint': checkpoint.revision,
        'frozen_source_sha256': checkpoint.hashes(), 'cost_artifact_sha256': hashes,
        'inputs': inputs, 'records': records,
        'nonclaims': ['Crossing fitted states and diagnostic modes localizes rounding; it does not waive failed full records.',
            'No optimizer setting, rank threshold, runtime numerical method or comparison tolerance changed.']}
    with (directory / 'lifecycle-factor-modes.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
