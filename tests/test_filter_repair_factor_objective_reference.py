"""Independent 80-digit factor-objective reference and frozen-state diagnostics.

This diagnostic-only module derives the objective and pullback independently of
TensorFlow. It must not be imported by any runtime or selection implementation.
"""

import hashlib
import json
import os
from pathlib import Path

import mpmath as mp
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_optimization_barrier

from bayesfilter.inference import factor_correlation_geometry as factor
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_remaining_factor import BASELINE, RAW, input_arrays, sha


def independent_objective(raw, center, offsets, scores, weights, anchors, config):
    """L=sum(w_i ||P z_i-y_i||^2)/d; dL/dC=-P sym(dL/dP) P.

    C=diag(s)[diag(1-row_norm(L)^2)+LL^T]diag(s). Its diagonal
    is s^2; loading derivatives therefore exclude the diagonal contribution.
    The remaining chain rule is the declared softplus/sigmoid/radial chart.
    All input floats are converted as their exact binary64 values.
    """
    r = list(map(mp.mpf, raw))
    d = len(center)
    count = config['factor_count']
    sigmoid = lambda x: 1/(1+mp.exp(-x))
    std = [mp.mpf(config['standard_deviation_floor'])+mp.log(1+mp.exp(x)) for x in r[:d]]
    bound = mp.sqrt(mp.mpf(1.0-config['loading_margin']))
    loads = mp.matrix(d, count)
    remaining = r[d:]
    if count == 1:
        for i in range(d):
            loads[i, 0] = bound*(sigmoid(remaining[i]) if i == anchors[0] else mp.tanh(remaining[i]))
    else:
        full = remaining.copy()
        full.insert(2*anchors[0]+1, mp.mpf(0))
        for i in range(d):
            radius = mp.sqrt(1+full[2*i]**2+full[2*i+1]**2)
            loads[i, 0], loads[i, 1] = bound*full[2*i]/radius, bound*full[2*i+1]/radius
        a, b = anchors
        loads[a, 0], loads[a, 1] = bound*sigmoid(full[2*a]), mp.mpf(0)
        # The runtime constant is binary64 math.pi, not arbitrary-precision pi.
        angle = mp.mpf(float(np.pi))*sigmoid(full[2*b+1])
        radius = bound*sigmoid(full[2*b])
        loads[b, 0], loads[b, 1] = radius*mp.cos(angle), radius*mp.sin(angle)
    covariance = mp.matrix(d)
    for i in range(d):
        for j in range(d):
            correlation = 1 if i == j else sum(loads[i, k]*loads[j, k] for k in range(count))
            covariance[i, j] = std[i]*std[j]*correlation
    precision = covariance**-1
    z = mp.matrix([[mp.mpf(float(x)) for x in row] for row in offsets])
    response = mp.matrix([[mp.mpf(float(center[j]))-mp.mpf(float(row[j]))
        for j in range(d)] for row in scores])
    weight = list(map(mp.mpf, weights))
    weight_sum = sum(weight)
    weight = [x/weight_sum for x in weight]
    error = z*precision-response
    value = sum(weight[i]*sum(error[i, j]**2 for j in range(d)) for i in range(z.rows))/d
    weighted_error = mp.matrix([[weight[i]*error[i, j] for j in range(d)] for i in range(z.rows)])
    gp = 2/mp.mpf(d)*(weighted_error.T*z)
    gc = -precision*((gp+gp.T)/2)*precision
    gs = [2*sum(gc[i, j]*covariance[i, j] for j in range(d))/std[i] for i in range(d)]
    off_diagonal = mp.matrix([[0 if i == j else gc[i, j]*std[i]*std[j]
        for j in range(d)] for i in range(d)])
    gl = 2*off_diagonal*loads
    gradient = [gs[i]*sigmoid(r[i]) for i in range(d)]
    if count == 1:
        for i in range(d):
            derivative = (sigmoid(remaining[i])*(1-sigmoid(remaining[i]))
                if i == anchors[0] else 1-mp.tanh(remaining[i])**2)
            gradient.append(gl[i, 0]*bound*derivative)
    else:
        full_gradient = []
        for i in range(d):
            x, y = full[2*i:2*i+2]
            if i == anchors[0]:
                gx, gy = gl[i, 0]*bound*sigmoid(x)*(1-sigmoid(x)), mp.mpf(0)
            elif i == anchors[1]:
                radius = bound*sigmoid(x)
                dr = radius*(1-sigmoid(x))
                angle = mp.mpf(float(np.pi))*sigmoid(y)
                da = mp.mpf(float(np.pi))*sigmoid(y)*(1-sigmoid(y))
                gx = dr*(gl[i, 0]*mp.cos(angle)+gl[i, 1]*mp.sin(angle))
                gy = radius*da*(-gl[i, 0]*mp.sin(angle)+gl[i, 1]*mp.cos(angle))
            else:
                norm = mp.sqrt(1+x*x+y*y)
                projection = gl[i, 0]*x+gl[i, 1]*y
                gx = bound*(gl[i, 0]/norm-x*projection/norm**3)
                gy = bound*(gl[i, 1]/norm-y*projection/norm**3)
            full_gradient.extend((gx, gy))
        del full_gradient[2*anchors[0]+1]
        gradient.extend(full_gradient)
    return value, gradient, covariance


def objective_program(config, dimension, rows):
    count = config.factor_count
    parameters = 2*dimension if count == 1 else 3*dimension-1
    domain = tf.Variable(0, dtype=tf.int64, trainable=False)

    @tf.function(input_signature=[tf.TensorSpec([parameters], tf.float64),
        tf.TensorSpec([dimension], tf.float64), tf.TensorSpec([rows, dimension], tf.float64),
        tf.TensorSpec([rows, dimension], tf.float64), tf.TensorSpec([rows], tf.float64),
        tf.TensorSpec([count], tf.int32)], jit_compile=True, autograph=False)
    def compute(raw, center, offsets, scores, weights, anchors):
        domain.assign(0)
        weights = xla_optimization_barrier(input=[weights])[0]
        weights /= tf.reduce_sum(weights)
        response = center[None, :]-scores

        def loss(position):
            covariance, _, _ = factor._decode_covariance(position, dimension=dimension,
                anchors=tf.unstack(anchors), config=config, domain_violations=domain)
            precision = tf.linalg.cholesky_solve(tf.linalg.cholesky(covariance),
                tf.eye(dimension, dtype=tf.float64))
            prediction = tf.einsum('ij,bj->bi', precision, offsets)
            per_row = tf.reduce_mean(tf.square(prediction-response), axis=1)
            return tf.reduce_sum(weights*per_row)

        value, gradient = tfp.math.value_and_gradient(loss, raw)
        return value, gradient, domain.read_value()

    return compute


def observation_pair():
    records = []
    for number in (4925, 4926):
        path = RAW/f'run-{number:05d}/remaining-factor-observation.json'
        manifest = json.loads((path.parent/'run.json').read_text())
        assert manifest['state'] == 'passed'
        record = json.loads(path.read_text())
        assert all(row['instrument_matches_ordinary'] and row['ordinary_matches_saved_endpoint']
            for row in record['cases'])
        records.append(record)
    assert records[0]['device'] == 'CPU' and records[1]['device'] == 'GPU'
    assert records[0]['operand_sha256'] == records[1]['operand_sha256']
    return records


def test_same_operand_objective(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    cpu, gpu = observation_pair()
    arrays, _, hashes = input_arrays()
    records = []
    for c, g in zip(cpu['cases'], gpu['cases'], strict=True):
        assert c['config'] == g['config'] and c['index'] == g['index']
        cfg = factor.FactorCorrelationGeometryConfig(**c['config'])
        owner = objective_program(cfg, 23, 68)
        replicate = c['replicate']
        for device, row in (('CPU', c), ('GPU', g)):
            for point in ('initial', 'terminal'):
                raw = row['observed']['initial_raw'] if point == 'initial' else row['ordinary']['optimizer']['position']
                anchors = row['ordinary']['anchors']
                arguments = (tf.constant(raw, tf.float64), tf.constant(arrays['center_score'], tf.float64),
                    tf.constant(arrays['offsets'][replicate, :68], tf.float64),
                    tf.constant(arrays['scores'][replicate, :68], tf.float64),
                    tf.fill([68], tf.constant(1./68, tf.float64)), tf.constant(anchors, tf.int32))
                value, gradient, invalid = owner(*arguments)
                assert int(invalid) == 0
                gradient_array = gradient.numpy()
                comparisons = None
                if point == 'terminal':
                    saved_gradient = np.asarray(row['ordinary']['optimizer']['objective_gradient'])
                    error = np.abs(gradient_array-saved_gradient)
                    comparisons = {'gradient_max_error': float(error.max()),
                        'gradient_max_bound_ratio': float(np.max(error/(1e-10+1e-10*np.abs(saved_gradient)))),
                        'value_absolute_error': abs(float(value)-row['ordinary']['optimizer']['objective_value'])}
                records.append({'fit': c['index'], 'operand_backend': device, 'point': point,
                    'raw': raw, 'anchors': anchors, 'value': float(value), 'gradient': gradient_array.tolist(),
                    'operand_sha256': hashlib.sha256(np.asarray(raw, dtype=np.float64).tobytes()).hexdigest(),
                    'saved_optimizer_comparison': comparisons})
        assert owner.experimental_get_tracing_count() == 1
    save(request, 'remaining-factor-objective.json', {'baseline': BASELINE,
        'device': 'CPU' if os.environ['CUDA_VISIBLE_DEVICES'] == '-1' else 'GPU',
        'input_sha256': hashes, 'records': records, 'jit_compile': True,
        'nonclaims': ['A replay differing from the saved gradient does not localize the saved optimizer.']})


def test_independent_objective_derivation():
    # Nonuniform weights exercise normalization; both charts have independent
    # 80-digit central differences, not TensorFlow derivatives as their oracle.
    with mp.workdps(80):
        center = [0.1, -0.2, 0.3]
        z = [[0.2, -0.1, 0.4], [0.3, 0.5, -0.2], [-0.4, 0.1, 0.6]]
        scores = [[-0.2, 0.4, 0.1], [0.6, -0.3, 0.2], [0.2, 0.1, -0.1]]
        for count, anchors in ((1, [1]), (2, [0, 2])):
            config = factor.FactorCorrelationGeometryConfig(factor_count=count).__dict__
            raw = [mp.mpf(i)/10-mp.mpf('0.4') for i in range(6 if count == 1 else 8)]
            args = (center, z, scores, [0.2, 0.3, 0.7], anchors, config)
            _, gradient, _ = independent_objective(raw, *args)
            step = mp.mpf('1e-25')
            for index in range(len(raw)):
                left, right = raw.copy(), raw.copy()
                left[index] -= step
                right[index] += step
                finite_difference = (independent_objective(right, *args)[0]-independent_objective(left, *args)[0])/(2*step)
                assert abs(finite_difference-gradient[index]) < mp.mpf('1e-40')


def test_saved_objective_references(request):
    cpu, gpu = observation_pair()
    arrays, _, hashes = input_arrays()
    inputs = []
    for number in (4927, 4928):
        path = RAW/f'run-{number:05d}/remaining-factor-objective.json'
        manifest = json.loads((path.parent/'run.json').read_text())
        assert manifest['state'] == 'passed'
        payload = json.loads(path.read_text())
        assert payload['input_sha256'] == hashes
        inputs.append((path, payload))
    assert inputs[0][1]['device'] == 'CPU' and inputs[1][1]['device'] == 'GPU'
    results = []
    with mp.workdps(80):
        for left, right in zip(inputs[0][1]['records'], inputs[1][1]['records'], strict=True):
            assert left['operand_sha256'] == right['operand_sha256']
            case = next(row for row in cpu['cases'] if row['index'] == left['fit'])
            replicate = case['replicate']
            expected_value, expected_gradient, _ = independent_objective(left['raw'], arrays['center_score'],
                arrays['offsets'][replicate, :68], arrays['scores'][replicate, :68], [1./68]*68,
                left['anchors'], case['config'])
            reference = np.asarray(list(map(float, expected_gradient)))
            errors = {}
            for backend, row in (('CPU', left), ('GPU', right)):
                error = np.abs(np.asarray(row['gradient'])-reference)
                errors[backend] = {'value_error': abs(row['value']-float(expected_value)),
                    'gradient_max_error': float(error.max()),
                    'gradient_max_bound_ratio': float(np.max(error/(1e-10+1e-10*np.abs(reference)))),
                    'saved_optimizer_comparison': row['saved_optimizer_comparison']}
            results.append({'fit': left['fit'], 'point': left['point'], 'operand_backend': left['operand_backend'],
                'reference_value': float(expected_value), 'reference_gradient': reference.tolist(),
                'errors': errors})
    initial = []
    for c, g in zip(cpu['cases'], gpu['cases'], strict=True):
        a, b = np.asarray(c['observed']['initial_raw']), np.asarray(g['observed']['initial_raw'])
        initial.append({'fit': c['index'], 'initial_raw_max_difference': float(np.max(np.abs(a-b))),
            'anchors_equal': c['ordinary']['anchors'] == g['ordinary']['anchors'],
            'CPU_terminal_gradient_norm': c['optimizer_gradient_infinity_norm'],
            'GPU_terminal_gradient_norm': g['optimizer_gradient_infinity_norm']})
    save(request, 'remaining-factor-reference.json', {'baseline': BASELINE, 'precision_digits': 80,
        'reference_source_sha256': sha(Path(__file__)),
        'objective_artifacts': {str(path): sha(path) for path, _ in inputs},
        'initial_state_comparison': initial, 'records': clean(results),
        'nonclaims': ['Independent local derivatives do not prove a unique optimum or optimizer convergence.',
            'Unselected unconverged trajectories can differ despite correct same-operand derivatives.']})


def test_factor_reference_terminal(request):
    """Tie the new scientific conclusion to complete witnesses and saved bytes."""
    observation_pair()
    directory = RAW/'run-04929'
    assert json.loads((directory/'run.json').read_text())['state'] == 'passed'
    path = directory/'remaining-factor-reference.json'
    reference = json.loads(path.read_text())
    # The terminal verifier was added after the reference calculation. Recover
    # the exact preceding source; neither oracle nor measured replay changed.
    source = Path(__file__).read_text()
    reference_source = source[:source.index('\n\ndef test_factor_reference_terminal')]
    assert hashlib.sha256(reference_source.encode()).hexdigest() == reference['reference_source_sha256']
    for name, digest in reference['objective_artifacts'].items():
        assert sha(Path(name)) == digest
    objective_before_constant_fix = reference_source.replace(
        'bound = mp.sqrt(mp.mpf(1.0-config[\'loading_margin\']))',
        'bound = mp.sqrt(1-mp.mpf(config[\'loading_margin\']))')
    relative = str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[1]))
    assert hashlib.sha256(objective_before_constant_fix.encode()).hexdigest() == json.loads(
        (RAW/'run-04927/run.json').read_text())['source_sha256'][relative]
    assert reference['reference_source_sha256'] == json.loads(
        (RAW/'run-04928/run.json').read_text())['source_sha256'][relative]
    for record in reference['records']:
        for backend in ('CPU', 'GPU'):
            errors = record['errors'][backend]
            assert errors['gradient_max_bound_ratio'] <= 1.
            assert errors['value_error'] <= 1e-10+1e-10*abs(record['reference_value'])
            comparison = errors['saved_optimizer_comparison']
            if comparison:
                assert comparison['gradient_max_bound_ratio'] <= 1.
    save(request, 'remaining-factor-terminal.json', {'baseline': BASELINE,
        'reference_sha256': sha(path), 'witnesses_qualified': True,
        'reference_derivation_checked': True, 'same_operand_gradients_within_original_bounds': True,
        'reference_constant_rounding_corrected': True,
        'measured_objective_callable_unchanged': True,
        'remaining': ['No unique converged optimizer solution established.',
            'Full record CPU/GPU differences preserved; no tolerance or selection change.',
            'Isotropic rank-cut reporting and terminal cost/current-consumer gates remain.']})
