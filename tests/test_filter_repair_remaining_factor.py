"""Saved-record and instrumented factor diagnostics; not admission evidence.

NumPy is confined to frozen diagnostic inputs and independent comparisons.
The diagnostic source transformation adds observations to the existing factory;
it does not implement another optimizer or change the ordinary runtime.
"""

import ast
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from tests.test_filter_repair_dz5_initializer_fit_localization import differences
from tests.test_filter_repair_geometry_control import clean, save

BASELINE = 'bebb2591a'
ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
SOURCE = 'bayesfilter/inference/factor_correlation_geometry.py'
DEPENDENCIES = (SOURCE, 'bayesfilter/inference/fixed_center_fitting_tf.py',
    'bayesfilter/inference/fixed_center_selection_tf.py',
    'bayesfilter/inference/fixed_center_stability_tf.py',
    'bayesfilter/inference/fixed_center_curvature.py')
CASES = ((3, 1, 1), (4, 2, 0), (5, 2, 1))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved(number):
    directory = RAW/f'run-{number:05d}'
    manifest = json.loads((directory/'run.json').read_text())
    assert manifest['state'] == 'passed'
    record = json.loads((directory/'dz5-exact-saved-fit.json').read_text())
    return record, manifest


def input_arrays():
    authority = json.loads((RAW/'run-04591/dz5-saved-fit-localization.json').read_text())
    path = RAW/'run-04591/frozen-fit-inputs.npz'
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ('center', 'center_score', 'offsets', 'scores')}
    hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
    assert hashes == authority['operand_sha256']
    recipe_path = RAW/'dz5-initializer-accepted-inputs-20260928-r1/recipe.json'
    assert sha(recipe_path) == authority['inputs']['source_sha256'][str(recipe_path)]
    return arrays, json.loads(recipe_path.read_text()), hashes


def test_saved_factor_consumer_classification(request):
    cpu, cm = saved(4919)
    gpu, gm = saved(4920)
    assert cpu['operand_sha256'] == gpu['operand_sha256']
    current_hashes = {path: sha(ROOT/path) for path in DEPENDENCIES}
    for manifest in (cm, gm):
        assert all(manifest['source_sha256'][path] == value for path, value in current_hashes.items())
    delta = differences(gpu['result'], cpu['result'])
    authority = json.loads((RAW/'run-04922/installed-angle-terminal.json').read_text())
    assert delta == authority['remaining_CPU_GPU_record_differences']
    groups = Counter('/'.join(row['path'].split('/')[:3]) for row in delta)
    scalar_rows = []
    for device, record in (('CPU', cpu), ('GPU', gpu)):
        assert record['result']['selected_family'] == 'consensus_diagonal_consensus'
        for index, count, replicate in CASES:
            row = record['result']['fits'][index]
            diagnostics = row['diagnostics']
            assert row['factor_count'] == count and row['replicate_index'] == replicate
            scalar_rows.append({'device': device, 'fit_index': index,
                **{k: row[k] for k in ('family', 'accepted', 'status', 'selection_holdout_relative_rmse')},
                **{k: diagnostics[k] for k in ('optimizer_converged', 'optimizer_failed',
                    'optimizer_iterations', 'optimizer_objective_evaluations', 'final_loss',
                    'condition_number', 'prediction_jacobian_condition_number', 'anchor_indices')}})
        # Selection needs two accepted, stable replicates. Only factor2 replicate1
        # passes its holdout/status gate; dense consensus is the actual candidate.
        raw_flags = record['raw']['fit']['fits']['flags']
        assert sum(bool(x[0] and x[2]) for x in raw_flags[1]) == 0
        assert sum(bool(x[0] and x[2]) for x in raw_flags[2]) == 1
        assert all(c['family'] == 'consensus_diagonal_consensus'
            for c in record['result']['diagnostics']['selection']['candidates'])
    assert sum(groups.values()) == 4534
    assert {key: groups[key] for key in ('/fits/3', '/fits/4', '/fits/5')} == {
        '/fits/3': 1663, '/fits/4': 1150, '/fits/5': 1664}
    assert not any(row['path'].startswith(('/selected_', '/accepted', '/status', '/audit_relative_rmse'))
        for row in delta)
    save(request, 'remaining-factor-saved.json', {'baseline': BASELINE,
        'dependencies': current_hashes, 'operand_sha256': cpu['operand_sha256'],
        'difference_groups': dict(groups), 'fits': scalar_rows,
        'selected_output_differences': 0, 'whole_record_differences': len(delta),
        'source_roles': {
            'factor_optimizer_gradient': 'TFP autodiff of geometry objective; not a filter analytical score',
            'factor_1': 'no accepted replicate; cannot supply selected geometry',
            'factor_2': 'one accepted replicate; cannot pass two-replicate direct selection',
            'selected_geometry': 'dense consensus, independent of the unselected factor matrices'},
        'nonclaims': ['Nonconvergence and prediction-Jacobian condition do not establish covariance invalidity.',
            'Unselected diagnostics remain exported; absence of selected impact is not full-record equivalence.']})


def instrumented_factory(request):
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{SOURCE}'], cwd=ROOT, text=True)
    assert hashlib.sha256(source.encode()).hexdigest() == sha(ROOT/SOURCE)
    tree = ast.parse(source)
    function = next(node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == '_cached_factor_program')
    function.name = 'instrumented_factor_program'
    function.decorator_list = []
    outputs = [node for node in ast.walk(function) if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == 'optimizer' for key in node.keys)]
    assert len(outputs) == 1
    outputs[0].keys.extend([ast.Constant('initial_raw'), ast.Constant('initial_dense_precision')])
    outputs[0].values.extend([ast.Name('initial_raw', ast.Load()), ast.Name('dense_precision', ast.Load())])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    source = ast.unparse(module)+'\n'
    directory = Path(request.config.getoption('xmlpath')).parent
    path = directory/'remaining-factor-instrumented.py'
    path.write_text(source)
    namespace = dict(factor.__dict__)
    exec(compile(source, str(path), 'exec'), namespace)  # noqa: S102 - frozen local diagnostic only
    return namespace['instrumented_factor_program'], sha(path)


def computed_record(value):
    result = dict(value)
    result['optimizer'] = value['optimizer']._asdict()
    return clean(result)


def saved_projection(value):
    opt = value['optimizer']
    identified = bool(value['jacobian_rank'] == len(opt.position))
    return clean({'precision': value['precision'], 'covariance': value['covariance'],
        'factor_eigenvalues': value['eigenvalues'],
        'factor_metrics': [value['train_rmse'], value['holdout_error'], value['holdout_relative'],
            value['condition_number'], value['jacobian_rank'], value['jacobian_condition'],
            float(opt.converged), float(opt.failed), float(opt.num_iterations),
            float(opt.num_objective_evaluations), opt.objective_value,
            float(identified if len(opt.position) == 68 else True)]})


def test_factor_optimizer_observation(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    original, manifest = saved(4920 if gpu else 4919)
    for path in DEPENDENCIES:
        assert sha(ROOT/path) == manifest['source_sha256'][path]
    arrays, recipe, hashes = input_arrays()
    instrument, source_hash = instrumented_factory(request)
    rows = []
    for index, count, replicate in CASES:
        cfg = factor.FactorCorrelationGeometryConfig(factor_count=count,
            max_condition_number=recipe['initializer']['max_condition_number'],
            holdout_score_relative_rmse=recipe['curvature_thresholds']['selection_holdout_relative_rmse_cap'])
        arguments = (tf.constant(arrays['center_score'], tf.float64),
            tf.constant(arrays['offsets'][replicate, :68], tf.float64),
            tf.constant(arrays['scores'][replicate, :68], tf.float64),
            tf.constant(arrays['offsets'][replicate+2, :68], tf.float64),
            tf.constant(arrays['scores'][replicate+2, :68], tf.float64),
            tf.fill([68], tf.constant(1./68, tf.float64)))
        owner = factor._make_factor_program(23, 68, 68, cfg, True,
            factor._prediction_jacobian_diagnostics)
        ordinary = owner(*arguments)
        witness = instrument(23, 68, 68, cfg, True, factor._prediction_jacobian_diagnostics)
        observed = witness(*arguments)
        observed_plain = {k: v for k, v in computed_record(observed).items()
            if k not in ('initial_raw', 'initial_dense_precision')}
        observation_delta = differences(observed_plain, computed_record(ordinary))
        expected = {k: original['raw']['fit']['fits'][k][count][replicate]
            for k in ('precision', 'covariance', 'factor_eigenvalues', 'factor_metrics')}
        endpoint_delta = differences(saved_projection(ordinary), expected)
        rows.append({'index': index, 'factor_count': count, 'replicate': replicate,
            'ordinary': computed_record(ordinary), 'observed': computed_record(observed),
            'observation_differences': observation_delta, 'saved_endpoint_differences': endpoint_delta,
            'instrument_matches_ordinary': not observation_delta,
            'ordinary_matches_saved_endpoint': not endpoint_delta,
            'ordinary_traces': owner.experimental_get_tracing_count(),
            'instrumented_traces': witness.experimental_get_tracing_count(),
            'optimizer_gradient_infinity_norm': float(tf.reduce_max(tf.abs(ordinary['optimizer'].objective_gradient))),
            'config': cfg.__dict__})
        assert rows[-1]['ordinary_traces'] == rows[-1]['instrumented_traces'] == 1
    save(request, 'remaining-factor-observation.json', {'baseline': BASELINE,
        'device': 'GPU' if gpu else 'CPU', 'instrumented_source_sha256': source_hash,
        'operand_sha256': hashes, 'cases': rows, 'jit_compile': True,
        'nonclaims': ['An observation failing either ordinary-output witness cannot explain the saved fit trajectory.',
            'This is a bounded optimizer-state diagnostic, not a performance or convergence qualification.']})
