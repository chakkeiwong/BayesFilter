"""Diagnostic reference replay of saved first-divergence CDF locator points."""

from tests import test_filter_repair_dz5_initializer_consumer as consumer
from tests import test_filter_repair_dz5_merged as merged
from tests import test_filter_repair_dz5_snapshot as snapshot_test
from tests.test_filter_repair_dz5_initializer_target import SNAPSHOT, audited_child

REPLAY_CHECK = r'''
    from two_currency_double_zlb_credit_target import credit_target_value_and_score
    assert not gpu
    evidence_path = Path(EVIDENCE_PATH)
    evidence = json.loads(evidence_path.read_text())
    assert evidence['source_runs'] == [4584, 4585]
    assert json.loads((evidence_path.parent / 'run.json').read_text())['state'] == 'passed'
    cases = []
    for row in evidence['first_difference_rows']:
        for arm in ('original', 'candidate'):
            cases.append({'label': f"{arm}_{row['callback_index']}", **row[arm]})
    for arm, number in (('original', 4584), ('candidate', 4585)):
        locator = json.loads((evidence_path.parent.parent / f'run-{number:05d}' / 'locator-result.json').read_text())
        cases.append({'label': arm + '_selected_center', 'positions': locator['center'],
            'values': locator['center_value'], 'scores': locator['center_score'], 'valid': locator['accepted']})
    signature = [tf.TensorSpec([1, 23], tf.float64)]
    callback = tf.function(model.training_target.batch_value_score_and_validity,
        input_signature=signature, jit_compile=True, autograph=False)
    @tf.function(input_signature=signature, jit_compile=False, autograph=False)
    def graph_reference(points):
        value, score, diagnostics = credit_target_value_and_score(points, base.fixture,
            jit_compile=False, evaluation_policy='hmc_rejection')
        return value, score, diagnostics['valid_pre_regularized_score'], diagnostics['branch_status_code']
    rows, failures = [], []
    report.update(role='same_position_CDF_trajectory_reference_replay',
        evidence_sha256=sha(evidence_path), target_execution_attempted=True,
        cpu_reference_exception=True, jit_compile=True,
        reference='explicit graph analytical value/score path of the same frozen source',
        comparisons=rows, replay_failures=failures,
        nonclaims=['No bitwise trajectory equivalence, source admission, optimizer repair or scientific promotion.'])
    def error(left, right):
        delta = tf.abs(left - right)
        target_bound = tf.constant(1e-8, tf.float64) + tf.constant(1e-7, tf.float64) * tf.abs(right)
        strict_bound = tf.constant(1e-10, tf.float64) + tf.constant(1e-10, tf.float64) * tf.abs(right)
        finite = bool(tf.reduce_all(tf.math.is_finite(left) & tf.math.is_finite(right)))
        return {'max_absolute_error': float(tf.reduce_max(delta)),
            'max_target_bound_ratio': float(tf.reduce_max(delta / target_bound)),
            'within_target_bounds': finite and bool(tf.reduce_all(delta <= target_bound)),
            'within_strict_record_bounds': finite and bool(tf.reduce_all(delta <= strict_bound))}
    for case in cases:
        points = tf.constant([case['positions']], tf.float64)
        tick = time.monotonic()
        actual = callback(points)
        expected = graph_reference(points)
        comparisons = {}
        for name, index in (('values', 0), ('scores', 1)):
            saved = tf.constant([case[name]], tf.float64)
            comparisons[name] = {
                'standalone_vs_graph': error(actual[index], expected[index]),
                'saved_vs_graph': error(saved, expected[index]),
                'saved_vs_standalone': error(saved, actual[index])}
            for context, result in comparisons[name].items():
                if not result['within_target_bounds']:
                    failures.append({'case': case['label'], 'field': name, 'context': context, **result})
        valid = bool(tf.reduce_all(actual[2] & expected[2] & (expected[3] == 0)))
        if valid != case['valid'] or not valid:
            failures.append({'case': case['label'], 'failure': 'validity_or_branch_status'})
        rows.append({'label': case['label'], 'positions': case['positions'],
            'value': actual[0].numpy().tolist(), 'score': actual[1].numpy().tolist(),
            'reference_value': expected[0].numpy().tolist(), 'reference_score': expected[1].numpy().tolist(),
            'valid': valid, 'comparisons': comparisons, 'elapsed_seconds': time.monotonic() - tick})
    report.update(target_evaluated=True, trace_count=callback.experimental_get_tracing_count(),
        reference_trace_count=graph_reference.experimental_get_tracing_count(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert report['trace_count'] == report['reference_trace_count'] == 1
    definition = callback.get_concrete_function().graph.as_graph_def()
    nodes = [*definition.node, *(node for f in definition.library.function for node in f.node_def)]
    report['host_callbacks'] = sorted({node.op for node in nodes if any(term in node.op.lower()
        for term in ('pyfunc', 'xlahostcompute'))})
    assert not report['host_callbacks']
    assert base.assert_source_pinned()
    assert not failures, failures
'''


def test_saved_locator_points_against_graph_reference(request):
    setup = consumer.INITIALIZER_CHECK.split('    events, owners, refs = [], [], []')[0]
    body = setup + REPLAY_CHECK
    replacements = {'RECIPE_PATH': repr(str(consumer.ACCEPTED_INPUTS / 'recipe.json')),
        'ADMISSION_PATH': repr(str(consumer.ADMISSION)), 'ACCEPTED_CASE': 'True',
        'ACCEPTED_INPUT_PATH': repr(str(consumer.ACCEPTED_INPUTS)),
        'EVIDENCE_PATH': repr(str(consumer.RAW / 'run-04586/dz5-locator-first-divergence.json'))}
    for key, value in replacements.items():
        body = body.replace(key, value)
    audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = audited_child(merged.target_child().replace(merged.TARGET_CHECK, body + audit))
    report = snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=child, scope='saved_locator_points_reference_replay_no_new_admission',
        child_timeout_seconds=840, read_only_paths=(consumer.RAW,))
    assert report['target_evaluated'] and not report['replay_failures']
