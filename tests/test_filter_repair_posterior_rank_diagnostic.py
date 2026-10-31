"""Read-only diagnostic instrumentation of the shared native COD rank decision."""

import ast
import inspect
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.ops import qr_lstsq_tf as cod
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_posterior_curvature import original
from tests.test_filter_repair_posterior_curvature_extras import pure_fixture


def _eigen_left_update(matrix, v, tau, pivot):
    # Eigen Householder.h:114-120 first forms essential' * bottom, then
    # adds the leading row. Keep that arithmetic boundary as a diagnostic.
    essential = tf.where(tf.range(matrix.shape[0]) > pivot, v, tf.zeros_like(v))
    temporary = tf.linalg.matvec(matrix, essential, transpose_a=True) + tf.gather(matrix, pivot)
    return matrix - tau * v[:, None] * temporary[None, :]


def _instrument(source, namespace, label):
    marker = '    def grad(upstream):'
    assert source.count(marker) == 1
    source = source.replace(marker, '    return solution, rank, pivots, threshold\n\n' + marker, 1)
    scope = dict(namespace)
    exec(compile(source, label, 'exec'), scope)  # noqa: S102 - read-only source instrumentation
    return scope['_complete_orthogonal_lstsq']


@pytest.mark.parametrize('design', ['rank_one', 'ill_conditioned'])
def test_preserve_dense_cod_rank_diagnostic(design, request):
    callback, _, _cfg, args = pure_fixture(3)
    offsets = (tf.ones([33, 3], tf.float64) if design == 'rank_one' else
        tf.reshape(tf.sin(tf.range(99, dtype=tf.float64)), [33, 3]) * tf.constant([1., 1e-8, 1e-10], tf.float64))
    points = args[0][None] + offsets @ tf.transpose(args[1])
    center_z = tf.linalg.matvec(args[1], callback(args[0][None])[1][0], transpose_a=True)
    scores = callback(points)[1] @ args[1]
    rhs = center_z[None] - scores
    checkpoint = FrozenCheckpoint('96e15e9d', 'posterior_cod_rank_before')
    previous = checkpoint.load('bayesfilter.ops.qr_lstsq_tf')
    module_source = checkpoint.sources['bayesfilter/ops/qr_lstsq_tf.py']
    node, = [node for node in ast.parse(module_source).body
             if isinstance(node, ast.FunctionDef) and node.name == '_complete_orthogonal_lstsq']
    source = ast.get_source_segment(module_source, node)
    frozen_diagnostic = _instrument(source, vars(previous), '<diagnostic-pinned-cod-rank>')
    current_source = textwrap.dedent(inspect.getsource(cod._complete_orthogonal_lstsq))
    diagnostic = _instrument(current_source, vars(cod), '<diagnostic-current-cod-rank>')
    _, baseline = original()
    expected = baseline.fit_dense_score_precision_tf(center_z, offsets, scores,
        selection_offsets=offsets, selection_scores=scores)
    outcomes = {}
    trial_source = source
    for name in ('a', 'b'):
        old = f'{name} - tau * v[:, None] * tf.linalg.matvec({name}, v, transpose_a=True)[None, :]'
        assert trial_source.count(old) == 1, 'Pinned arithmetic trial must change exactly one update'
        trial_source = trial_source.replace(old, f'_eigen_left_update({name}, v, tau, k)')
    trial = _instrument(trial_source, {**vars(previous), '_eigen_left_update': _eigen_left_update},
                        '<diagnostic-pinned-cod-tail-update>')
    for variant, operation in (('checkpoint', frozen_diagnostic), ('current', diagnostic), ('tail_dot', trial)):
        for mode in ('eager', 'graph', 'xla'):
            program = operation if mode == 'eager' else tf.function(operation, autograph=False,
                input_signature=[tf.TensorSpec([33, 3], tf.float64), tf.TensorSpec([33, 3], tf.float64)], jit_compile=mode == 'xla')
            solution, rank, pivots, threshold = program(offsets, rhs)
            outcomes[f'{variant}_{mode}'] = {'solution': solution.numpy().tolist(), 'rank': int(rank),
                'pivots': pivots.numpy().tolist(), 'threshold': float(threshold),
                'response_max_residual': float(tf.reduce_max(tf.abs(offsets @ solution - rhs)))}
    numpy_solution, _, numpy_rank, singular = np.linalg.lstsq(offsets.numpy(), rhs.numpy(), rcond=np.finfo(float).eps * 3)
    report = {'design': design, 'offsets': offsets.numpy().tolist(), 'response': rhs.numpy().tolist(),
        'checkpoint_revision': checkpoint.revision, 'checkpoint_source_sha256': checkpoint.hashes(),
        'original_precision': expected['raw_precision'].numpy().tolist(),
        'original_numpy_solution': numpy_solution.tolist(), 'numpy_rank': int(numpy_rank),
        'singular_values': singular.tolist(), 'candidate': outcomes,
        'role': 'explanatory rank localization; no numerical gate waiver'}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'posterior-cod-rank-{design}.json').open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
