"""Diagnostic attribution of the D3 dense-trust graph/XLA discrepancy."""

import json
import math
from pathlib import Path

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_self_adjoint_eig

from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference import sequential_proposal_tf as proposal
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64


def test_dense_trust_graph_xla_source_attribution(request):
    checkpoint = FrozenCheckpoint('93c8e419', 'attempts_modes')
    frozen = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    scalar, batched, preparation = _inputs(3, 4)
    center, score, scale, radius, seed, search, search_scores = preparation
    cfg = frozen.SequentialMapCovarianceConfig(refinement_geometry_policy='factor_correlation',
        structured_max_factors=1, structured_holdout_score_relative_rmse=.001)
    data, _ = frozen._structured_factor_fit_data(scalar, center, score, scale,
        dimension=3, fresh_sample_count=12, radius=float(radius), seed=tuple(seed.numpy().tolist()),
        search_theta=search, search_scores=search_scores, reuse_search_scores=True,
        evaluations=0, batched_value_and_score_fn=batched)
    fit = frozen._fit_factor_from_data(data, factor_count=1, config=cfg)
    assert fit['status'] == 'usable'
    precision = tf.constant(fit['projected_precision_z'], D)
    args = (center, scalar(center)[0], score, scale, precision, radius,
        tf.constant(cfg.score_reduction_factor, D), tf.constant(cfg.acceptance_ratio, D))
    records = {}
    for label, make in [('frozen', frozen.proposal_program), ('current', proposal.proposal_program)]:
        for jit in (False, True):
            with tf.device('/GPU:0'):
                program = make(scalar, 3, cfg.proposal_score_acceptance_policy,
                    cfg.require_proposal_score_reduction, jit_compile=jit)
                records[label + ('_xla' if jit else '_graph_gpu')] = current._json_ready(program(*args))
    # Same installed TensorFlow, explicit CPU graph comparator; no GPU default
    # change and no use of this reference to waive the failed GPU-mode gate.
    with tf.device('/CPU:0'):
        gpu_graph = frozen.proposal_program(scalar, 3, cfg.proposal_score_acceptance_policy,
            cfg.require_proposal_score_reduction, jit_compile=False)
        cpu_graph = tf.function(gpu_graph.python_function, input_signature=gpu_graph.input_signature,
            jit_compile=False, autograph=False)
        records['frozen_graph_cpu'] = current._json_ready(cpu_graph(*args))
    _compare(records['current_xla'], records['frozen_xla'])
    _compare(records['current_graph_gpu'], records['frozen_graph_gpu'])
    def eigen_program(jit):
        @tf.function(input_signature=[tf.TensorSpec([3, 3], D)], jit_compile=jit, autograph=False)
        def eig(matrix):
            values, vectors = (xla_self_adjoint_eig(matrix, lower=True, max_iter=100,
                epsilon=math.ulp(1.)) if jit else tf.linalg.eigh(matrix))
            return values, vectors

        return eig

    eigen = {}
    for device, jit in [('/GPU:0', False), ('/CPU:0', False), ('/GPU:0', True)]:
        with tf.device(device):
            values, vectors = eigen_program(jit)(precision)
            residual = precision @ vectors - vectors * values[None, :]
            reconstruction = (vectors * values[None, :]) @ tf.transpose(vectors) - precision
        eigen[device + ('_xla' if jit else '_graph')] = {
            'eigenvalues': values.numpy().tolist(), 'eigenvectors': vectors.numpy().tolist(),
            'max_residual': float(tf.reduce_max(tf.abs(residual))),
            'max_reconstruction_error': float(tf.reduce_max(tf.abs(reconstruction)))}
    report = {'role': 'explanatory_dense_trust_source_attribution', 'checkpoint': checkpoint.revision,
        'frozen_source_sha256': checkpoint.hashes(), 'precision': precision.numpy().tolist(),
        'records': records, 'eigenpairs': eigen,
        'differences': {key: _record_differences(row, records['frozen_xla']) for key, row in records.items()},
        'nonclaims': ['Failed GPU graph/XLA comparison remains open at unchanged 1e-10.',
            'No optimizer, eigensolver, tolerance or runtime default changed.']}
    path = Path(request.config.getoption('xmlpath')).parent / 'attempts-dense-trust-modes.json'
    with path.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
