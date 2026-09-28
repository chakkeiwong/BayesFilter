"""Independent matrix/derivative diagnostics for the shared tiny LM solver."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.genut_shape_lm_tf import scaled_lm_coefficients_jvp

ROOT = Path(__file__).resolve().parents[1]
SAVED = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917/run-04677/seeded-lm-tf32-localization.json')


def _independent(jacobian, residual, options):
    jacobian = np.asarray(jacobian, np.float64)
    residual = np.asarray(residual, np.float64)
    # Preserve the configured scalar conversion in the float32 implementation.
    floor = float(np.float32(options['scale_floor']))
    damping = float(np.float32(options['damping']))
    strength = float(np.float32(options['strength']))
    scale = np.sqrt(np.sum(jacobian ** 2, axis=-2) + floor ** 2)
    normalized = jacobian / scale[..., None, :]
    system = np.swapaxes(normalized, -1, -2) @ normalized + damping * np.eye(2)
    rhs = np.einsum('...ki,...k->...i', normalized, residual)
    coefficient = strength * np.linalg.solve(system, rhs[..., None])[..., 0] / scale
    return system, coefficient


@pytest.mark.parametrize('dtype', [tf.float32, tf.float64])
def test_saved_lm_accuracy_and_analytical_directions(dtype, request):
    source = SAVED.read_bytes()
    saved = json.loads(source)
    calls = [call for row in saved['steps'] for call in row['lm_calls']]
    options = calls[0]['options']
    old_source = subprocess.check_output(['git', 'show',
        'aabd2b167:bayesfilter/highdim/genut_shape_lm_tf.py'], cwd=ROOT, text=True)
    prior = ModuleType('_lm_before_precision_diagnostic')
    exec(compile(old_source, '<lm-before-precision-diagnostic>', 'exec'), prior.__dict__)  # noqa: S102 -- pinned diagnostic authority
    signatures = [tf.TensorSpec([2, 2, 2], dtype), tf.TensorSpec([2, 2], dtype),
                  tf.TensorSpec([2, 2, 2, 3], dtype), tf.TensorSpec([2, 2, 3], dtype)]

    def build(implementation, jit):
        return tf.function(lambda j,r,dj,dr: implementation(j,r,dj,dr,**options),
                           input_signature=signatures, jit_compile=jit, autograph=False)

    xla = build(scaled_lm_coefficients_jvp, True)
    graph = build(scaled_lm_coefficients_jvp, False)
    old_xla = build(prior.scaled_lm_coefficients_jvp, True)
    rng = np.random.default_rng(137)
    records = []
    for case, call in enumerate(calls):
        jacobian, residual = (np.asarray(a, dtype.as_numpy_dtype) for a in call['inputs'][:2])
        dj = rng.normal(size=[2, 2, 2, 3]).astype(dtype.as_numpy_dtype) * 0.1
        dr = rng.normal(size=[2, 2, 3]).astype(dtype.as_numpy_dtype) * 0.1
        operands = tuple(tf.constant(a, dtype) for a in (jacobian, residual, dj, dr))
        expected_system, expected_coefficient = _independent(jacobian, residual, options)
        reference_tangents = []
        for step in (2e-4, 1e-4):
            columns = []
            for direction in range(3):
                j64, r64 = jacobian.astype(np.float64), residual.astype(np.float64)
                tj64, tr64 = dj[..., direction].astype(np.float64), dr[..., direction].astype(np.float64)
                def value(offset, j=j64, r=r64, tj=tj64, tr=tr64):
                    return _independent(j + offset * tj, r + offset * tr, options)[1]
                columns.append((value(-2*step) - 8*value(-step) + 8*value(step) - value(2*step)) / (12*step))
            reference_tangents.append(np.stack(columns, axis=-1))
        np.testing.assert_allclose(*reference_tangents, atol=2e-8, rtol=2e-8)
        results = {'eager': scaled_lm_coefficients_jvp(*operands, **options),
                   'graph': graph(*operands), 'xla': xla(*operands)}
        records.append({'case': case, 'inputs': [a.numpy().tolist() for a in operands],
                        'results': {mode: {k: v.numpy().tolist() for k,v in row.items()}
                                    for mode,row in results.items()},
                        'independent_system': expected_system.tolist(),
                        'independent_coefficient': expected_coefficient.tolist(),
                        'independent_tangents': [v.tolist() for v in reference_tangents]})
        directory = Path(request.config.getoption('xmlpath')).parent
        output = directory / f'lm-precision-{dtype.name}.json'
        output.write_text(json.dumps({'records': records}, indent=2) + '\n')
        for mode, row in results.items():
            np.testing.assert_allclose(row['scaled_system'], expected_system,
                atol=3e-7 if dtype == tf.float32 else 1e-12,
                rtol=3e-7 if dtype == tf.float32 else 1e-12, err_msg=mode)
            np.testing.assert_allclose(row['coefficient'], expected_coefficient,
                atol=2e-6 if dtype == tf.float32 else 1e-10,
                rtol=1e-6 if dtype == tf.float32 else 1e-10, err_msg=mode)
            np.testing.assert_allclose(row['coefficient_tangent'], reference_tangents[-1],
                atol=2e-5 if dtype == tf.float32 else 2e-8,
                rtol=3e-5 if dtype == tf.float32 else 2e-8, err_msg=mode)
        previous = old_xla(*operands)
        np.testing.assert_allclose(results['xla']['coefficient'], previous['coefficient'], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(results['xla']['coefficient_tangent'], previous['coefficient_tangent'], atol=1e-6, rtol=1e-6)
        for key, value in xla(*operands).items():
            np.testing.assert_array_equal(value, results['xla'][key])
    assert xla.experimental_get_tracing_count() == graph.experimental_get_tracing_count() == 1
    output.write_text(json.dumps({'records': records, 'source_sha256': hashlib.sha256(source).hexdigest(),
        'old_source_sha256': hashlib.sha256(old_source.encode()).hexdigest(), 'passed': True,
        'trace_count': 1, 'device': results['xla']['coefficient'].device,
        'tf32_enabled': tf.config.experimental.tensor_float_32_execution_enabled()}, indent=2) + '\n')
