"""Diagnostic crossed fits on saved CDF callback outputs, never admission.

NumPy only reads the independently preserved NPZ and compares saved evidence.
No CDF target, optimizer start, threshold or runtime default is changed here.
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import (
    xla_optimization_barrier,
    xla_svd,
)

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference.dense_initializer_random_tf import (
    make_dense_initializer_cloud_design,
)
from bayesfilter.inference.dense_validated_fit_tf import (
    make_dense_validated_fit_program,
)
from bayesfilter.inference.mass_matrix_tf import _eigenpairs
from bayesfilter.inference.tensor_npz_archive import write_tensor_npz
from tests.test_filter_repair_dense_validated_fit import normalized, original_fitter
from tests.test_filter_repair_fixed_fitting_localization import _baseline
from tests.test_filter_repair_geometry_control import clean, save

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
D = tf.float64


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(actual, expected, path='', *, atol=1e-10, rtol=1e-10):
    """Preserve all mismatches at the existing full-record tolerances."""
    rows = []
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in sorted(actual.keys() | expected.keys()):
            if key not in actual or key not in expected:
                rows.append({'path': path + '/' + key, 'actual': actual.get(key), 'expected': expected.get(key)})
            else:
                rows.extend(differences(actual[key], expected[key], path + '/' + key, atol=atol, rtol=rtol))
    elif isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            rows.extend(differences(left, right, f'{path}/{index}', atol=atol, rtol=rtol))
    elif (isinstance(actual, (int, float)) and not isinstance(actual, bool)
          and isinstance(expected, (int, float)) and not isinstance(expected, bool)):
        limit = atol + rtol * abs(expected)
        if abs(actual - expected) > limit:
            rows.append({'path': path, 'actual': actual, 'expected': expected,
                         'absolute_error': abs(actual - expected), 'allowed_error': limit})
    elif actual != expected:
        rows.append({'path': path, 'actual': actual, 'expected': expected})
    return rows


def frozen_inputs():
    source = RAW / 'run-04572'
    record = json.loads((source / 'initialize.json').read_text())
    report = json.loads((source / 'dz5-snapshot-import.json').read_text())
    oracle_path = RAW / 'run-04560/dz5-score-oracle.json'
    oracle = json.loads(oracle_path.read_text())
    assert report['passed'] and report['initializer_accepted']
    assert oracle['snapshot_manifest_sha256'] == report['snapshot_manifest_sha256']
    assert len(record['attempts']) == 1
    archive_path = source / 'initializer_evaluations.npz'
    assert sha(archive_path) == next(iter(record['artifacts'].values()))
    recipe_path = RAW / 'dz5-initializer-accepted-inputs-20260928-r1/recipe.json'
    assert sha(recipe_path) == report['recipe_sha256']
    recipe = json.loads(recipe_path.read_text())
    config = recipe['initializer']
    attempt = record['attempts'][0]
    curvature = attempt['curvature']
    center = tf.constant(curvature['center'], D)
    score = tf.constant(curvature['center_score_z'], D)
    scale = tf.constant(oracle['prior_scale'], D)
    dimension = len(curvature['center'])
    # Recreate the same Philox design, then freeze operands for every fit arm.
    # Enclosing-XLA libm rounding is measured, not assumed to be bit-identical.
    generate = make_dense_initializer_cloud_design(dimension, config['replicate_count'],
        config['training_rows_per_replicate'], config['selection_rows_per_replicate'],
        config['audit_rows'], config['max_curvature_attempts'])
    offsets = generate(tf.constant(config['seed'], tf.int32), tf.constant(config['curvature_radius'], D))[0]
    scores, position_errors = [], []
    with np.load(archive_path, allow_pickle=False) as archive:
        for index, rows in enumerate(attempt['partition_rows']):
            prefix = f'attempt_0_partition_{index}_'
            expected = archive[prefix + 'positions']
            observed = center[None, :] + offsets[index, :rows] * scale
            error = float(tf.reduce_max(tf.abs(observed - tf.constant(expected, D))))
            position_errors.append(error)
            np.testing.assert_allclose(observed.numpy(), expected, atol=1e-14, rtol=0.)
            assert archive[prefix + 'valid'].all()
            values = tf.constant(archive[prefix + 'scores'], D) * scale
            scores.append(tf.pad(values, [[0, offsets.shape[1] - rows], [0, 0]]))
    operands = (center, score, offsets, tf.stack(scores))
    provenance = {'source_run': 4572, 'source_sha256': {str(path): sha(path)
        for path in (source / 'run.json', source / 'initialize.json',
                     source / 'dz5-snapshot-import.json', archive_path, recipe_path, oracle_path)},
        'maximum_position_reconstruction_errors': position_errors,
        'snapshot_manifest_sha256': report['snapshot_manifest_sha256'],
        'offset_roundoff_boundary': 'same frozen reconstructed operands for all arms; not bit-exact full-program intermediate capture'}
    return recipe, curvature, operands, provenance


@pytest.mark.parametrize('arm', ['original', 'graph', 'xla'])
def test_saved_cdf_fit_inputs(arm, request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    recipe, saved, operands, provenance = frozen_inputs()
    config = recipe['initializer']
    center, score, offsets, scores = operands
    dimension = center.shape[0]
    count = config['replicate_count']
    train, select, audit = (config[key] for key in
        ('training_rows_per_replicate', 'selection_rows_per_replicate', 'audit_rows'))
    settings = {key: config[key] for key in ('factor_max', 'dense_eigenvalue_floor',
        'max_condition_number', 'shrinkage_weights', 'structured_target_family')}
    lineage = saved['diagnostics']['lineage']
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'frozen-fit-inputs.npz').open('xb') as stream:
        write_tensor_npz(stream,
            dict(zip(('center', 'center_score', 'offsets', 'scores'), operands, strict=True)))
    started = time.monotonic()
    if arm == 'original':
        baseline, hashes = original_fitter()
        thresholds = baseline.FixedCenterCurvatureThresholds(**recipe['curvature_thresholds'])
        result = baseline.fit_fixed_center_curvature(center, score, offsets[:count, :train],
            scores[:count, :train], offsets[count:2 * count, :select], scores[count:2 * count, :select],
            offsets[-1, :audit], scores[-1, :audit], thresholds=thresholds, lineage=lineage,
            **settings).payload()
        execution = {'reference_sources': hashes, 'mode': 'pinned_original_reference'}
    else:
        thresholds = fixed.FixedCenterCurvatureThresholds(**recipe['curvature_thresholds'])
        owner = make_dense_validated_fit_program(dimension, count, train, select, audit,
            thresholds=thresholds, jit_compile=arm == 'xla', **settings)
        raw = owner(*operands)
        tf.nest.map_structure(lambda value: value.numpy(), raw)
        # Preserve failed numerical state before the admission assertion.
        save(request, 'dz5-saved-fit-raw.json', {'arm': arm, 'inputs': provenance,
            'raw': clean(raw), 'trace_count': owner.experimental_get_tracing_count(),
            'input_signature': str(owner.input_signature)})
        assert bool(raw['fit_ran']) and int(raw['fit_error_code']) == 0
        result = fixed._fixed_center_result_from_native(raw['fit'], center, score,
            dimension=dimension, replicates=count, training_rows=train, selection_rows=select,
            audit_rows=audit, thresholds=thresholds, factor_max=config['factor_max'],
            weights=tuple(config['shrinkage_weights']), structured_target_family=config['structured_target_family'],
            lineage=lineage, jit_compile=arm == 'xla').payload()
        execution = {'mode': arm, 'trace_count': owner.experimental_get_tracing_count(),
            'input_signature': str(owner.input_signature), 'jit_compile': arm == 'xla'}
    result = normalized({'result': result})['result']
    saved = normalized({'result': saved})['result']
    report = {'schema': 'filter_dz5_saved_fit_localization.v1', 'arm': arm,
        'role': 'same_saved_callback_input_localization_not_equivalence_admission',
        'elapsed_seconds': time.monotonic() - started, 'inputs': provenance,
        'operand_sha256': {key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for key, value in zip(('center', 'center_score', 'offsets', 'scores'), operands, strict=True)},
        'execution': execution, 'result': clean(result),
        'comparison_tolerance': {'atol': 1e-10, 'rtol': 1e-10},
        'differences_from_saved_full_initializer_fit': differences(result, saved),
        'numerical_equivalence_qualified': False,
        'nonclaims': ['No complete initializer or fresh target execution, HMC, training, timing ranking or tolerance relaxation.']}
    save(request, 'dz5-saved-fit-localization.json', report)
    assert result['selected_family'] is not None


def test_saved_cdf_initialization_and_angle_mechanisms(request):
    """Localize alternative-fit starting charts and padded-SVD accuracy."""
    tf.config.experimental.enable_tensor_float_32_execution(False)
    recipe, saved, operands, provenance = frozen_inputs()
    _, center_score, offsets, scores = operands
    dimension, rows = center_score.shape[0], offsets.shape[1]
    original = _baseline()
    records = []
    for count in (1, 2):
        for replicate in (0, 1):
            arguments = (offsets[replicate], center_score[None] - scores[replicate],
                tf.fill([rows], tf.constant(1. / rows, D)))

            def evaluate(module, z, response, weights, *, mode, count=count):
                native = module is factor
                jit = mode == 'xla'
                if jit:
                    weights = xla_optimization_barrier(input=[weights])[0]
                weights = weights / tf.reduce_sum(weights)
                keywords = {'jit_compile': jit} if native else {}
                precision = module._weighted_dense_precision(z, response, weights,
                    max_condition_number=recipe['initializer']['max_condition_number'], **keywords)
                covariance = tf.linalg.inv(precision)
                deviations, loadings, anchors = module._initial_factor_state(covariance,
                    factor_count=count, loading_margin=1e-6, **keywords)
                config = module.FactorCorrelationGeometryConfig(factor_count=count)
                raw = module._encode_state(deviations, loadings, anchors, config)
                initial_covariance, _, _ = module._decode_covariance(raw,
                    dimension=dimension, anchors=anchors, config=config)
                sd = tf.sqrt(tf.linalg.diag_part(covariance))
                correlation = covariance / (sd[:, None] * sd[None, :])
                eigenvalues, eigenvectors = (_eigenpairs(correlation, jit) if native else tf.linalg.eigh(correlation))
                unscaled = eigenvectors[:, -count:] * tf.sqrt(tf.maximum(eigenvalues[-count:] - 1., 1e-6))
                cap = tf.constant(.8 * math.sqrt(1. - 1e-6), D)
                norms = tf.linalg.norm(unscaled, axis=1, keepdims=True)
                clipped = unscaled * tf.minimum(tf.ones_like(norms), cap / tf.maximum(norms, 1e-15))
                return {'weighted_precision': precision, 'weighted_covariance': covariance,
                    'correlation_eigenvalues': eigenvalues, 'unclipped_row_norms': tf.linalg.norm(unscaled, axis=1),
                    'clipped_row_norms': tf.linalg.norm(clipped, axis=1),
                    'clip_bound': cap, 'anchors': tf.stack(anchors), 'raw': raw,
                    'initial_covariance': initial_covariance, 'loadings': loadings}

            row = {'factor_count': count, 'replicate': replicate}
            row['original'] = clean(evaluate(original, *arguments, mode='original'))
            for mode in ('graph', 'xla'):
                def kernel(z, response, weights, mode=mode):
                    return evaluate(factor, z, response, weights, mode=mode)
                program = tf.function(kernel, input_signature=[tf.TensorSpec([rows, dimension], D),
                    tf.TensorSpec([rows, dimension], D), tf.TensorSpec([rows], D)],
                    jit_compile=mode == 'xla', autograph=False)
                row[mode] = clean(program(*arguments))
            records.append(row)

    dense = [fit for fit in saved['fits'] if fit['family'] == 'dense']
    first, second = (tf.constant(fit['precision_z'], D) for fit in dense)
    rank = recipe['curvature_thresholds']['principal_subspace_rank']

    @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], D)] * 2,
        jit_compile=True, autograph=False)
    def angle_probe(first, second):
        vectors_a = _eigenpairs(first, True)[1]
        vectors_b = _eigenpairs(second, True)[1]
        selected = tf.cast(tf.range(dimension) >= dimension - rank, D)
        overlap = tf.matmul(vectors_a * selected, vectors_b * selected, transpose_a=True)
        default = tf.linalg.svd(overlap, compute_uv=False)
        precise = xla_svd(overlap, max_iter=100, epsilon=sys.float_info.epsilon, precision_config='').s
        return overlap, default, precise

    overlap, default, precise = angle_probe(first, second)
    # Independent LAPACK authority on the exact same compiled overlap operand.
    authority = np.linalg.svd(overlap.numpy(), compute_uv=False)
    def angles(value):
        return np.arccos(np.clip(value[:rank], -1., 1.)) * (180. / math.pi)
    angle_report = {'overlap': clean(overlap), 'reference_singular_values': authority.tolist(),
        'reference_angles': angles(authority).tolist(),
        'default_singular_values': clean(default), 'precise_singular_values': clean(precise),
        'default_angles': angles(default.numpy()).tolist(), 'precise_angles': angles(precise.numpy()).tolist(),
        'default_max_singular_error': float(np.max(np.abs(default.numpy() - authority))),
        'precise_max_singular_error': float(np.max(np.abs(precise.numpy() - authority)))}
    save(request, 'dz5-fit-mechanisms.json', {'inputs': provenance, 'initializations': records,
        'principal_angles': angle_report, 'role': 'diagnostic_mechanisms_not_runtime_repair',
        'nonclaims': ['No anchor-selection policy change, optimizer retuning, full equivalence or admission.']})
    np.testing.assert_allclose(precise.numpy(), authority, atol=1e-14, rtol=1e-14)
