"""Independent diagnostic probes of the remaining inference SVD consumers."""

import numpy as np
import tensorflow as tf

from bayesfilter.inference.block_score_geometry_tf import _block_fit
from bayesfilter.inference.quadratic_geometry import _quadratic_fit_kernel
from bayesfilter.inference.score_curvature_tf import fit_dense_score_precision_tf
from bayesfilter.inference.sequential_score_fit_tf import fit_numerics
from bayesfilter.ops.qr_lstsq_tf import condition_number
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64


def _program(jit):
    @tf.function(input_signature=[tf.TensorSpec([7, 3], D)],
        jit_compile=jit, autograph=False)
    def evaluate(offsets):
        precision = tf.linalg.diag(tf.constant([1.3, 1.7, .8], D))
        response = tf.matmul(offsets, precision)
        zero = tf.zeros([3], D)
        dense = fit_dense_score_precision_tf(zero, offsets, -response)
        block, block_report, block_status = _block_fit(offsets, response,
            tf.constant(0., D), tf.constant(1e8, D))
        sequential = fit_numerics(offsets, -response, zero, tf.ones([3], D),
            tf.constant(0., D), tf.constant(1e-8, D), tf.constant(1e8, D),
            tf.constant(.05, D), training_indices=(0, 1, 2, 3, 4),
            holdout_indices=(5, 6), jit_compile=jit)
        quadratic = _quadratic_fit_kernel(offsets,
            -.5 * tf.reduce_sum(offsets * response, axis=1), -response,
            tf.eye(3, dtype=D)[:, :2], zero, tf.constant(1e-8, D),
            tf.constant(1e8, D), use_xla_svd=jit)
        return {'condition': condition_number(offsets, jit_compile=jit),
            'dense': dense, 'block_precision': block, 'block_report': block_report,
            'block_status': block_status, 'sequential': sequential,
            'quadratic': quadratic}

    return evaluate


def test_remaining_svd_consumers_are_scale_invariant(request):
    # Deterministic QR frames are diagnostic fixtures only. All matrices remain
    # well conditioned; this test cannot excuse ill-conditioned discrepancies.
    fixture = np.array([[1., .2, -.3], [.4, 1., .1], [-.1, .3, 1.],
        [.7, -.5, .2], [-.3, .2, .9], [.2, .7, -.6], [-.8, .3, .2]])
    left = np.linalg.qr(fixture)[0]
    right = np.linalg.qr(np.array([[1., .4, -.2], [.3, 1., .5], [-.1, .2, 1.]]))[0]
    expected_precision = np.diag([1.3, 1.7, .8])
    rows, failures = [], []
    programs = {mode: _program(mode == 'xla') for mode in ('graph', 'xla')}
    for spectrum_name, spectrum in (('separated', np.array([3., 1.5, .7])),
            ('near_tied', np.array([1.+1e-8, 1., 1.-1e-8]))):
        for magnitude in (1., 1e-4, 1e-10, 1e4):
            offsets = (left * spectrum) @ right.T * magnitude
            expected_condition = np.linalg.cond(offsets)
            for mode, program in programs.items():
                result = tf.nest.map_structure(lambda tensor: np.asarray(tensor.numpy()),
                    program(tf.constant(offsets, D)))
                checks = {
                    'cod_condition': np.allclose(result['condition'], expected_condition, rtol=1e-10, atol=1e-10),
                    'dense_condition': np.allclose(result['dense']['design_condition'], expected_condition, rtol=1e-10, atol=1e-10),
                    'dense_rank': result['dense']['design_rank'] == 3,
                    'dense_precision': np.allclose(result['dense']['raw_precision'], expected_precision, rtol=1e-10, atol=1e-10),
                    'block_ranks': np.array_equal(result['block_report'][:2], [6, 3]),
                    'block_status': result['block_status'] == 0,
                    'block_precision': np.allclose(result['block_precision'], expected_precision, rtol=1e-10, atol=1e-10),
                    'sequential_rank': result['sequential']['rank'] == 6,
                    'sequential_status': result['sequential']['status'] == 1,
                    'sequential_precision': np.allclose(result['sequential']['projected_precision_z'], expected_precision, rtol=1e-10, atol=1e-10),
                    'quadratic_rank': result['quadratic']['score_design_rank'] == 3,
                    'quadratic_resolved': bool(result['quadratic']['design_resolved']),
                    'quadratic_precision': np.allclose(result['quadratic']['precision'], expected_precision, rtol=1e-10, atol=1e-10),
                }
                checks = {name: bool(passed) for name, passed in checks.items()}
                failed = [name for name, passed in checks.items() if not passed]
                rows.append({'spectrum': spectrum_name, 'magnitude': magnitude,
                    'mode': mode, 'offsets': offsets, 'expected_condition': expected_condition,
                    'expected_precision': expected_precision, 'checks': checks, 'actual': result})
                if failed:
                    failures.append({'spectrum': spectrum_name, 'magnitude': magnitude,
                        'mode': mode, 'failed': failed})
    save(request, 'remaining-svd-scale.json', clean({'rows': rows, 'failures': failures,
        'role': 'remaining_SVD_actual_consumer_scale_diagnostic',
        'nonclaims': ['No public endpoint qualification or ill-conditioned equivalence waiver.',
            'Zero ridge is an explicit exact-quadratic diagnostic, not a runtime default.']}))
    assert not failures, failures
