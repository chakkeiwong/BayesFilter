"""Diagnostic graph identity and interleaved timing for the SRUKF cost trigger."""

import copy
import hashlib
import json
import re
import statistics
import time

import numpy as np
import pytest
import tensorflow as tf
from google.protobuf.json_format import MessageToDict

from bayesfilter.nonlinear import rectangular_srukf_tf as current
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_srukf_scale import independent_kalman
from tests.test_filter_repair_svd_cost import original_route

D = tf.float64


def normalized_graph(program):
    graph = program.get_concrete_function().graph.as_graph_def()
    return normalized_definition(MessageToDict(graph, preserving_proto_field_name=True))


def normalized_definition(definition):
    names = {function['signature']['name']: re.sub(r'_\d+$', '_ID', function['signature']['name'])
        for function in definition.get('library', {}).get('function', [])}
    assert len(set(names.values())) == len(names), 'Ambiguous function-name normalization'

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()
                if key != 'experimental_debug_info'}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return names.get(value, value) if isinstance(value, str) else value

    result = normalize(definition)

    def node_names(nodes, arguments=()):
        # TensorFlow names eager captures by a global numeric tensor ID and
        # redundant While arguments by the generated condition-function ID.
        # Rename definitions and their edges together, never attribute values.
        mapping = {node['name']: f'CAPTURE_AT_NODE_{index}' for index, node in enumerate(nodes)
            if node['op'] == 'Const' and node['name'].isdigit()}
        mapping.update({arg['name']: re.sub(r'(_cond)_\d+(___redundant_placeholder\d+)$', r'\1_ID\2', arg['name'])
            for arg in arguments})
        original_names = [node['name'] for node in nodes] + [arg['name'] for arg in arguments]
        renamed = [mapping.get(name, name) for name in original_names]
        assert len(set(renamed)) == len(renamed), 'Ambiguous local-name normalization'
        for arg in arguments:
            arg['name'] = mapping.get(arg['name'], arg['name'])
        for node in nodes:
            node['name'] = mapping.get(node['name'], node['name'])
            inputs = []
            for edge in node.get('input', []):
                control, operand = ('^', edge[1:]) if edge.startswith('^') else ('', edge)
                base, separator, port = operand.partition(':')
                inputs.append(control + mapping.get(base, base) + separator + port)
            if 'input' in node:
                node['input'] = inputs

    node_names(result['node'])
    for function in result['library']['function']:
        node_names(function['node_def'], function['signature'].get('input_arg', []))
    result['library']['function'].sort(key=lambda function: function['signature']['name'])
    return result


@pytest.mark.parametrize('change', ['constant', 'operation', 'edge', 'device', 'argument_type', 'capture_shape'])
def test_graph_normalization_preserves_executable_changes(change):
    original = {'node': [{'name': '40', 'op': 'Const', 'attr': {'value': {'s': '40'}, 'shape': [1, 3]}},
        {'name': 'use', 'op': 'Identity', 'input': ['40:0'], 'device': '/device:GPU:0'}],
        'library': {'function': [{'signature': {'name': 'while_cond_43',
            'input_arg': [{'name': 'while_while_cond_43___redundant_placeholder0', 'type': 'DT_DOUBLE'}]},
            'node_def': [{'name': 'use', 'op': 'Identity',
                'input': ['while_while_cond_43___redundant_placeholder0']}]}]}}
    renamed = copy.deepcopy(original)
    renamed['node'][0]['name'] = '413'
    renamed['node'][1]['input'] = ['413:0']
    function = renamed['library']['function'][0]
    function['signature']['name'] = 'while_cond_416'
    function['signature']['input_arg'][0]['name'] = 'while_while_cond_416___redundant_placeholder0'
    function['node_def'][0]['input'] = ['while_while_cond_416___redundant_placeholder0']
    assert normalized_definition(original) == normalized_definition(renamed)
    altered = copy.deepcopy(renamed)
    if change == 'constant':
        altered['node'][0]['attr']['value']['s'] = '413'
    elif change == 'operation':
        altered['node'][1]['op'] = 'Neg'
    elif change == 'edge':
        altered['node'][1]['input'] = ['413:1']
    elif change == 'device':
        altered['node'][1]['device'] = '/device:CPU:0'
    elif change == 'argument_type':
        altered['library']['function'][0]['signature']['input_arg'][0]['type'] = 'DT_FLOAT'
    else:
        altered['node'][0]['attr']['shape'] = [3, 1]
    assert normalized_definition(original) != normalized_definition(altered)


def test_interleaved_original_and_candidate_graph(request):
    original, sources = original_route()
    transition = np.array([[.8, .1, -.05], [.04, .7, .15], [-.02, .06, .9]])
    observation = np.array([[2., 1., -.5], [1., 3., .25], [-.5, .25, 1.]])
    prior, process, noise = np.diag([.5, .3, .4]), np.diag([.2, .1, .15]), np.diag([.15, .1, .2])
    points = np.array([[.1, -.2, .3], [.2, .1, -.1], [-.1, .15, .2]])
    transition_tf, observation_tf = tf.constant(transition, D), tf.constant(observation, D)

    def build(route):
        model = route.TFRectangularSRUKFModel(tf.zeros([1, 3], D), tf.constant(prior[None], D),
            tf.constant(process[None], D), tf.constant(noise[None], D),
            lambda state, noise: tf.linalg.matmul(state, transition_tf, transpose_b=True) + noise,
            lambda state: tf.linalg.matmul(state, observation_tf, transpose_b=True))

        @tf.function(input_signature=[tf.TensorSpec([1, 3, 3], D)],
            autograph=False, jit_compile=False)
        def evaluate(observations):
            result = route.tf_rectangular_srukf_value(observations, model, jit_compile=False)
            return {'likelihood': result.log_likelihood, 'mean': result.filtered_mean,
                'covariance': tf.matmul(result.filtered_factor, result.filtered_factor, transpose_b=True),
                'on_support': result.diagnostics['on_support'],
                'rank': result.diagnostics['minimum_observation_rank'],
                'support_residual': result.diagnostics['maximum_support_residual']}

        return evaluate

    programs = {'prior': build(original), 'candidate': build(current)}
    operands = tf.constant(points[None], D)
    gpu = bool(tf.config.list_logical_devices('GPU'))
    values, samples, replay_failures = {}, [], []
    with GPUProcessMonitor(gpu) as sharing:
        for name, program in programs.items():
            values[name] = tf.nest.map_structure(lambda tensor: tensor.numpy(), program(operands))
        # Balanced ABBA/BAAB blocks observe order and common runtime variation.
        # No sources, environment, thread settings or numerics change between arms.
        for block in range(40):
            order = ('prior', 'candidate', 'candidate', 'prior') if block % 2 == 0 else (
                'candidate', 'prior', 'prior', 'candidate')
            for name in order:
                tick = time.perf_counter()
                actual = tf.nest.map_structure(lambda tensor: tensor.numpy(), programs[name](operands))
                milliseconds = (time.perf_counter() - tick) * 1000
                samples.append({'block': block, 'arm': name, 'milliseconds': milliseconds})
                for key in actual:
                    if not np.array_equal(actual[key], values[name][key]):
                        replay_failures.append({'block': block, 'arm': name,
                            'field': key, 'actual': actual[key], 'expected': values[name][key]})
    definitions = {name: normalized_graph(program) for name, program in programs.items()}
    hashes = {name: hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        for name, definition in definitions.items()}
    likelihood, mean, covariance = independent_kalman(points, transition, observation, prior, process, noise)
    expected = {'likelihood': [likelihood], 'mean': mean[None], 'covariance': covariance[None]}
    numerics = {name: all(np.allclose(value[key], wanted, atol=1e-10, rtol=1e-10)
        for key, wanted in expected.items()) and bool(value['on_support'].all())
        and bool((value['rank'] == 3).all()) for name, value in values.items()}
    medians = {name: statistics.median(row['milliseconds'] for row in samples if row['arm'] == name)
        for name in programs}
    observation_record = sharing.payload()
    unshared = not observation_record['errors'] and (not gpu or (
        len(observation_record['samples']) >= 2 and all(
            process['uuid'] == observation_record['uuid']
            and process['pid'] == observation_record['pid']
            for sample in observation_record['samples'] for process in sample['processes'])))
    report = {'schema': 'filter_srukf_graph_attribution.v1', 'horizon': 3,
        'gpu': gpu, 'original_source_sha256': sources, 'graph_sha256': hashes,
        'graph_equal': definitions['prior'] == definitions['candidate'],
        'normalization': 'Generated FunctionDef suffixes, redundant While formal IDs, numeric Const capture names and matching edges; experimental_debug_info omitted. All operations, edge relations, ports, constants, attributes, shapes and devices retained.',
        'graphs': definitions, 'values': values, 'independent_reference': expected,
        'numerical_passed': numerics, 'samples': samples, 'median_ms': medians,
        'replay_failures': replay_failures, 'in_run_unshared': unshared,
        'candidate_to_prior_ratio': medians['candidate'] / medians['prior'],
        'gpu_process_observation': observation_record,
        'nonclaims': ['Explicit graph reference attribution only; no graph default.',
            'Interleaved calls are dependent descriptive observations, not independent superiority evidence.',
            'This diagnoses the preserved fresh-process trigger and does not overwrite it.']}
    save(request, 'svd-graph-attribution.json', clean(report))
    assert not replay_failures, replay_failures
    assert unshared, observation_record
    assert all(numerics.values()), numerics
    assert report['graph_equal'], hashes
