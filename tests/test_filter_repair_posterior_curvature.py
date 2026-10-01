"""Frozen-original controller and random-stream diagnostics, never a runtime."""

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.posterior_curvature_refinement import (
    PosteriorCurvatureRefinementConfig,
)
from bayesfilter.inference.posterior_curvature_report import posterior_curvature_result
from bayesfilter.inference.posterior_curvature_tf import (
    draw_offsets,
    make_posterior_curvature_controller,
    partition_seed,
    precision_spread,
)
from bayesfilter.ops.stateless_random_tf import philox_normal_float64
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_quadratic_batches import _equal_records


def original():
    checkpoint = FrozenCheckpoint('3582b4ac', 'posterior_curvature')
    module = checkpoint.load('bayesfilter.inference.posterior_curvature_refinement')
    return checkpoint, module


def stable_hlo(text):
    """Ignore only Grappler's synthetic zero-constant node uniquifiers.

    Run02372 has identical operations, constants, operands and shapes; only
    these dummy-source metadata suffixes differ between two IR exports.
    Every other compiler character remains part of the equality check.
    """
    return re.sub(r'(op_name="zeros_\d+/_)\d+(" source_file="dummy_file_name")', r'\1UNIQUE\2', text)


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('design', ['uniform_box', 'uniform_ball'])
def test_original_partition_and_proposal_streams(dimension, design):
    _, baseline = original()

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32)],
                 autograph=False, jit_compile=True)
    def program(seed, index):
        folded = partition_seed(seed, index)
        return folded, draw_offsets(33, dimension, folded, .7, design), philox_normal_float64([33, dimension], folded)

    for seed, index in ((-73, 0), (918, 2), (20260908, 5)):
        folded, offsets, proposal = program(seed, index)
        expected_seed = baseline._seed(seed, index)
        np.testing.assert_array_equal(folded, expected_seed)
        np.testing.assert_allclose(offsets, baseline._draw_offsets(33, dimension, expected_seed, .7, design), atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(proposal, tf.random.stateless_normal([33, dimension], expected_seed, dtype=tf.float64), atol=1e-14, rtol=1e-14)
    assert program.experimental_get_tracing_count() == 1


def fixture(dimension, case='gaussian', design='uniform_box', *, replicates=2, rows=33, batch=8):
    cfg = PosteriorCurvatureRefinementConfig(rows_per_partition=rows, batch_size=batch,
        replicate_count=replicates, seed=-73, max_physical_rows=10000, fit_design=design,
        lineage={'fixture': case, 'frozen': '3582b4ac'})
    center = tf.range(dimension, dtype=tf.float64) * .03
    factor = tf.linalg.band_part(tf.ones([dimension, dimension], tf.float64) * .05, -1, 0) + tf.eye(dimension, dtype=tf.float64) * .9
    if case == 'nonfinite_position':
        center = tf.fill([dimension], tf.constant(1e308, tf.float64))
        factor = tf.eye(dimension, dtype=tf.float64) * 1e308
    if case == 'transformed':
        factor *= 100.
    # These resources measure actual calls, not Python tracing side effects.
    positions = tf.Variable(tf.zeros([100, batch, dimension], tf.float64))
    # TensorFlow pins int32 variables to the host. XLA requires diagnostic
    # resources on the same device as the target and position recorder.
    with tf.device(positions.device):
        calls = (tf.Variable(0, dtype=tf.int64), tf.Variable(0, dtype=tf.int64))
    assert all(counter.device == positions.device for counter in calls)
    batches = (rows + batch - 1) // batch
    failed = {'center': 1, 'training': 3, 'selection': 2 + replicates * batches,
              'audit': 2 + 2 * replicates * batches, 'proposal': 2 + (2 * replicates + 1) * batches}
    bad_call = failed.get(case.split('_')[-1], 0)

    def eligibility(theta):
        count = calls[0].assign_add(1)
        valid = tf.ones([batch], tf.bool)
        if case.startswith('ineligible_'):
            valid &= (count != bad_call) | (tf.range(batch) != 1)
        return valid

    def callback(theta):
        count = calls[1].assign_add(1)
        positions.scatter_nd_update(tf.reshape(count - 1, [1, 1]), theta[None])
        precision = tf.linalg.diag(tf.cast(tf.range(dimension) + 1, tf.float64)) + tf.ones([dimension, dimension], tf.float64) * .05
        delta = theta - .2
        scores = -delta @ precision
        values = -.5 * tf.reduce_sum(delta * (delta @ precision), axis=1)
        if case == 'nonquadratic':
            scores -= .001 * delta**3
            values -= .00025 * tf.reduce_sum(delta**4, axis=1)
        if case == 'nonspd':
            scores = theta
        if case == 'transformed':
            scores = tf.fill([batch, dimension], tf.constant(1e308, tf.float64))
        if case == 'nonfinite_position':
            values, scores = tf.zeros([batch], tf.float64), tf.zeros_like(theta)
        if case.startswith('value_'):
            values = tf.where(count == bad_call, tf.constant(float('nan'), tf.float64), values)
        if case.startswith('score_'):
            scores = tf.where(count == bad_call, tf.constant(float('inf'), tf.float64), scores)
        if case in ('reject_audit', 'reject_proposal'):
            scores *= tf.where(count >= bad_call, tf.constant(2., tf.float64), tf.constant(1., tf.float64))
        return values, scores

    return callback, eligibility, cfg, (center, factor, tf.constant(cfg.seed)), calls, positions


def reference(callback, eligibility, cfg, args):
    checkpoint, baseline = original()
    options = dataclasses.asdict(cfg)
    options['seed'] = int(args[2])
    result = baseline.refine_posterior_local_curvature(callback, args[0], args[1],
        batched_eligibility_fn=eligibility, config=baseline.PosteriorCurvatureRefinementConfig(**options))
    payload = result.payload()
    for fit in payload['diagnostics'].get('replicates', []):
        if fit['design_rank'] < int(args[0].shape[0]):
            fit['design_condition'] = None  # Approved deficient-rank definition.
    return payload, checkpoint.hashes()


CASES = ('gaussian', 'nonquadratic', 'nonspd', 'transformed', 'nonfinite_position',
         'ineligible_center', 'ineligible_training', 'ineligible_selection', 'ineligible_audit', 'ineligible_proposal',
         'value_center', 'score_training', 'score_selection', 'value_audit', 'score_proposal', 'reject_audit', 'reject_proposal')


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('case', CASES)
def test_complete_original_controller_records(dimension, case, request):
    callback, eligibility, cfg, args, counters, positions = fixture(dimension, case)
    expected, hashes = reference(callback, eligibility, cfg, args)
    original_counts = [int(counter) for counter in counters]
    original_positions = positions.numpy()[:original_counts[1]]
    records = {}
    for jit in (False, True):
        for counter in counters:
            counter.assign(0)
        positions.assign(tf.zeros_like(positions))
        program = make_posterior_curvature_controller(callback, eligibility, dimension, cfg, jit_compile=jit)
        raw = program(*args)
        actual = posterior_curvature_result(raw, args[0], args[1], cfg).payload()
        records[str(jit)] = actual
        assert [int(counter) for counter in counters] == original_counts
        np.testing.assert_allclose(positions.numpy()[:original_counts[1]], original_positions, atol=1e-10, rtol=1e-10)
        assert program.experimental_get_tracing_count() == 1
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'posterior-record-{case}-{dimension}.json').open('x') as out:
        json.dump({'baseline': '3582b4ac', 'original_source_sha256': hashes, 'original': expected, 'candidate': records}, out, indent=2, allow_nan=False)
        out.write('\n')
    for actual in records.values():
        _equal_records(actual, expected)


def test_ball_replicates_and_changed_inputs_keep_runtime_operands(request):
    callback, eligibility, cfg, args, counters, _positions = fixture(3, design='uniform_ball', replicates=3, rows=17, batch=7)
    program = make_posterior_curvature_controller(callback, eligibility, 3, cfg)
    changed = (args[0] + .07, args[1] * 1.1, args[2] + 4)
    records = []
    for values in (args, changed):
        for counter in counters:
            counter.assign(0)
        expected, _ = reference(callback, eligibility, cfg, values)
        expected_calls = [int(counter) for counter in counters]
        for counter in counters:
            counter.assign(0)
        raw = program(*values)
        actual = posterior_curvature_result(raw, values[0], values[1], dataclasses.replace(cfg, seed=int(values[2]))).payload()
        records.append(actual)
        _equal_records(actual, expected)
        assert [int(counter) for counter in counters] == expected_calls
    assert records[0]['precision_z'] != records[1]['precision_z']
    assert program.experimental_get_tracing_count() == 1
    concrete = program.get_concrete_function()
    graph = concrete.graph.as_graph_def()
    nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
    assert not {'PyFunc', 'EagerPyFunc', 'PyFuncStateless'} & {node.op for node in nodes}
    assert {'While', 'StatelessWhile'} & {node.op for node in nodes}
    first_hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
    changed_hlo = program.experimental_get_compiler_ir(*changed)(stage='hlo')
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / 'posterior-first.hlo').write_text(first_hlo)
    (directory / 'posterior-changed.hlo').write_text(changed_hlo)
    with (directory / 'posterior-operands.json').open('x') as out:
        json.dump({'records': records, 'traces': program.experimental_get_tracing_count(),
            'expected_operands': 3 + len(concrete.captured_inputs),
            'first_entry': first_hlo[first_hlo.rfind('\nENTRY '):].splitlines()[:14],
            'changed_entry': changed_hlo[changed_hlo.rfind('\nENTRY '):].splitlines()[:14]}, out, indent=2, allow_nan=False)
        out.write('\n')
    assert stable_hlo(first_hlo) == stable_hlo(changed_hlo)
    entry = first_hlo[first_hlo.rfind('\nENTRY '):]
    assert len(re.findall(r'\bparameter\((\d+)\)', entry)) == 3 + len(concrete.captured_inputs)


def test_consensus_checks_nonadjacent_pairs_and_reciprocals():
    _, baseline = original()
    precisions = tf.stack([tf.eye(3, dtype=tf.float64) * scale for scale in (1.4, 1., 1./1.4)])
    program = tf.function(precision_spread, input_signature=[tf.TensorSpec([3, 3, 3], tf.float64)],
        autograph=False, jit_compile=True)
    actual = program(precisions)
    expected = baseline._precision_spread(tuple(tf.unstack(precisions)))
    assert float(actual) == pytest.approx(1.96, abs=1e-12)
    assert float(actual) == pytest.approx(expected, abs=1e-12)
