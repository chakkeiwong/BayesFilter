"""Source-arithmetic localization of a rejected zero-design GPU residual."""

import dataclasses
import hashlib
import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference import posterior_curvature_tf as runtime
from bayesfilter.inference.posterior_curvature_report import posterior_curvature_result
from tests.test_filter_repair_posterior_curvature import original
from tests.test_filter_repair_posterior_curvature_extras import pure_fixture
from tests.test_filter_repair_quadratic_batches import _equal_records


def _reduce_projection(scores, factor):
    return tf.reduce_sum(scores[:, :, None] * factor[None, :, :], axis=1)


def _ordered_projection(scores, factor):
    dimension = factor.shape[0]

    def step(index, total):
        contribution = tf.gather(scores, index, axis=1)[:, None] * tf.gather(factor, index)[None, :]
        return index + 1, total + contribution

    return tf.while_loop(lambda index, _total: index < dimension, step,
        (tf.constant(0), tf.zeros_like(scores)), maximum_iterations=dimension,
        parallel_iterations=1)[1]


def test_zero_design_center_projection_arithmetic(monkeypatch, request):
    callback, eligibility, config, args = pure_fixture(3)
    offsets = tf.zeros([33, 3], tf.float64)
    _, baseline = original()
    monkeypatch.setattr(baseline, '_draw_offsets', lambda *unused: offsets)
    expected = baseline.refine_posterior_local_curvature(callback, args[0], args[1],
        batched_eligibility_fn=eligibility,
        config=baseline.PosteriorCurvatureRefinementConfig(**dataclasses.asdict(config))).payload()
    expected['diagnostics']['replicates'][0]['design_condition'] = None
    monkeypatch.setattr(runtime, 'draw_offsets', lambda *unused: offsets)
    # Freeze the pre-repair factory; other helpers remain explicitly current.
    # This is a local arithmetic attribution, not a historical closure oracle.
    archived = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917/run-02419')
    source = (archived / 'posterior-zero-center-source.py').read_text()
    archived_report = json.loads((archived / 'posterior-zero-center-diagnostic.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest() == archived_report['source_sha256']
    old = 'tf.linalg.matvec(factor, center_score, transpose_a=True)'
    new = 'tf.matmul(center_score[None], factor)[0]'
    assert source.count(old) == 1, 'Diagnostic expects the unrepaired center arithmetic'
    scope = dict(vars(runtime))
    original_scope = dict(vars(runtime))
    exec(compile(source, '<diagnostic-unrepaired-center>', 'exec'), original_scope)  # noqa: S102 - exact archived factory
    trial_source = source.replace(old, new)
    exec(compile(trial_source, '<diagnostic-row-form-center>', 'exec'), scope)  # noqa: S102 - isolated equivalent-arithmetic diagnostic
    records = {}
    stages = {}
    observed_source = source.replace(
        '"scores_z": tf.zeros([fit_partitions, rows, dimension], D),',
        '"scores_z": tf.zeros([fit_partitions, rows, dimension], D),\n'
        '            "scores_theta": tf.zeros([fit_partitions, rows, dimension], D),')
    observed_source = observed_source.replace(
        '"scores_z": tf.tensor_scatter_nd_update(current["scores_z"], slot, transformed[None]),',
        '"scores_z": tf.tensor_scatter_nd_update(current["scores_z"], slot, transformed[None]),\n'
        '                "scores_theta": tf.tensor_scatter_nd_update(current["scores_theta"], slot, scores[None]),')
    observed_source = observed_source.replace(
        '        del state["designs"], state["scores_z"], state["center_score"], state["center_score_z"]', '')
    observed_scope = dict(vars(runtime))
    exec(compile(observed_source, '<diagnostic-observed-center>', 'exec'), observed_scope)  # noqa: S102 - read-only instrumentation
    factories = [('unrepaired_snapshot', original_scope['make_posterior_curvature_controller']),
                 ('current', runtime.make_posterior_curvature_controller),
                 ('row_form', scope['make_posterior_curvature_controller']),
                 ('observed', observed_scope['make_posterior_curvature_controller'])]
    for label, projection in (('shared_reduce', _reduce_projection), ('shared_ordered', _ordered_projection)):
        shared_source = source.replace(old, '_project_rows(center_score[None], factor)[0]')
        assert shared_source.count('tf.matmul(scores, factor)') == 1
        assert shared_source.count('scores @ factor') == 2
        shared_source = shared_source.replace('tf.matmul(scores, factor)', '_project_rows(scores, factor)')
        shared_source = shared_source.replace('scores @ factor', '_project_rows(scores, factor)')
        shared_scope = {**vars(runtime), '_project_rows': projection}
        exec(compile(shared_source, f'<diagnostic-{label}>', 'exec'), shared_scope)  # noqa: S102 - isolated equivalent-arithmetic diagnostic
        factories.append((label, shared_scope['make_posterior_curvature_controller']))
    for name, factory in factories:
        for jit in (False, True):
            program = factory(callback, eligibility, 3, config, jit_compile=jit)
            raw = program(*args)
            records[f'{name}_{jit}'] = posterior_curvature_result(raw, args[0], args[1], config).payload()
            if name == 'observed':
                stages[str(jit)] = {key: raw[key].numpy().tolist() for key in
                    ('center_score', 'center_score_z', 'scores_z', 'scores_theta')}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'posterior-zero-center-source.py').open('x') as output:
        output.write(source)
    with (directory / 'posterior-zero-center-diagnostic.json').open('x') as output:
        json.dump({'original': expected, 'candidate': records, 'observed_stages': stages,
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'replacement': {'before': old, 'trial': new},
            'role': 'isolated center dot arithmetic trial; mandatory complete runtime checks remain separate'},
            output, indent=2, allow_nan=False)
        output.write('\n')
    _equal_records(records['current_False'], expected)
    _equal_records(records['row_form_False'], expected)
    # The failed row-form hypothesis is preserved, not installed. Attribution
    # is valid only if exposing intermediates retains every actual result.
    _equal_records(records['observed_False'], records['unrepaired_snapshot_False'])
    _equal_records(records['observed_True'], records['unrepaired_snapshot_True'])
    for label in ('shared_reduce', 'shared_ordered'):
        for jit in (False, True):
            _equal_records(records[f'{label}_{jit}'], expected)
