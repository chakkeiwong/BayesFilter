"""Independent finite enumeration of the proposed diagnostic, not HMC evidence."""
import itertools

import pytest
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_mcse_randomization import batch_randomization, permutation_statistics


def independent_lag(means, order):
    result = []
    for j in range(len(means[0][0])):
        centered = [[row[j] - sum(x[j] for x in chain)/len(chain) for row in chain] for chain in means]
        numerator = sum(sum(chain[a]*chain[b] for a,b in zip(order[:-1], order[1:])) for chain in centered)
        denominator = sum(sum(x*x for x in chain) for chain in centered)
        result.append(numerator/denominator)
    return result


def test_exact_enumeration_checks_bias_and_joint_p_value():
    means = [[[0., 4.], [1., 2.], [3., 5.], [7., 1.]],
             [[2., 8.], [-1., 1.], [4., 3.], [6., 0.]]]
    orders = list(itertools.permutations(range(4)))
    result = permutation_statistics(tf.constant(means, tf.float64), tf.constant(orders, tf.int32))
    expected = [independent_lag(means, order) for order in orders]
    tf.debugging.assert_near(result['permuted_lag'], tf.constant(expected, tf.float64), atol=1e-14)
    tf.debugging.assert_near(tf.reduce_mean(result['permuted_lag'], axis=0),
                             tf.constant([-.25, -.25], tf.float64), atol=1e-14)
    statistics = [max(abs(x+.25) for x in row) for row in expected]
    observed = max(abs(x+.25) for x in independent_lag(means, tuple(range(4))))
    probability = (1+sum(value >= observed-1e-12*(1+abs(observed)) for value in statistics))/25
    assert bool(result['valid'])
    assert abs(float(result['p_value'])-probability) < 1e-14


def test_affine_rescaling_and_chain_offsets_do_not_hide_constant_functions():
    means = tf.constant([[[1., -1.], [2., 4.], [4., 3.], [8., 2.]],
                         [[3., 5.], [1., 0.], [2., 1.], [7., 4.]]], tf.float64)
    permutations = tf.constant(list(itertools.permutations(range(4))), tf.int32)
    base = permutation_statistics(means, permutations)
    transformed = means*tf.constant([-2., .5], tf.float64) + tf.constant([[[10., -3.]], [[-7., 4.]]], tf.float64)
    changed = permutation_statistics(transformed, permutations)
    tf.debugging.assert_near(base['statistic'], changed['statistic'], atol=1e-14)
    assert float(base['p_value']) == float(changed['p_value'])
    constant = permutation_statistics(tf.ones_like(means), permutations)
    assert not bool(constant['valid'])
    assert float(constant['p_value']) == 1.


def test_invalid_permutations_fail_closed():
    means = tf.reshape(tf.range(16, dtype=tf.float64), [2, 4, 2])
    result = permutation_statistics(means, tf.constant([[0, 0, 2, 3]], tf.int32))
    assert not bool(result['valid'])


def test_prospective_graph_replays_and_detects_batch_order_dependence():
    # Independent noise with a strong monotone batch effect; shuffle test must
    # see the order even though each chain is centered separately.
    values = tf.random.stateless_normal([4, 1000, 2], [971, 8], dtype=tf.float64)
    values += tf.linspace(tf.constant(-4., tf.float64), tf.constant(4., tf.float64), 1000)[None, :, None]
    calculate = batch_randomization(1000, function_count=2)
    first = calculate(values, tf.constant([18, 9], tf.int32))
    second = calculate(values, tf.constant([18, 9], tf.int32))
    assert bool(first['valid'])
    assert float(first['p_value']) <= .01
    for key in first:
        tf.debugging.assert_equal(first[key], second[key])


def test_zero_admitted_samples_cannot_pass_coverage():
    from scripts.calibrate_surrogate_hmc_mcse_randomization import decision, group_summary
    looks = {str(n): dict(candidate_valid=False, original_valid=False, all_functions_covered=True,
                         mean_mcse_oracle_ratio=[1.]*3, naive_mcse_oracle_ratio=[1.]*3)
             for n in (1000, 2000)}
    group = group_summary([dict(looks=looks)]*256, stationary=True)
    names = ('iid', 'ar_positive', 'ar_negative', 'ar_borderline', 'ar_slow', 'chain_offset', 'linear_drift')
    result = decision({name: group for name in names})
    assert result['status'] == 'CANDIDATE_NOT_CALIBRATED'
    assert not result['checks']['iid_coverage']
    assert result['checks']['ar_slow_veto']


def test_undefined_statistics_are_identified_and_saved_without_false_values():
    import json
    from scripts.calibrate_surrogate_hmc_mcse_randomization import nonfinite_paths
    from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
    value = dict(function_rhat=[1., float('nan')], candidate_valid=False)
    assert nonfinite_paths(value) == ['.function_rhat[1]']
    saved = json.loads(json.dumps(finite_report(value), allow_nan=False))
    assert saved == dict(function_rhat=[1., None], candidate_valid=False)


def test_binary_fold_policy_keeps_bulk_check_and_other_degeneracy_vetoes():
    from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_split_rhat
    from bayesfilter.testing.surrogate_hmc_mcse_randomization import binary_function_rhat

    @tf.function(input_signature=[tf.TensorSpec([4, 400, 1], tf.float64)],
                 jit_compile=False, autograph=False)
    def evaluate(values):
        components = rank_normalized_split_rhat(values)
        maximum, flag = binary_function_rhat(values, components)
        return maximum, flag, components

    balanced = tf.tile(tf.constant([0., 1.], tf.float64)[None, :, None], [4, 200, 1])
    maximum, flag, raw = evaluate(balanced)
    assert bool(flag[0]) and float(maximum[0]) == 1.
    # The fold is exactly constant, even when subsequent rank-transform
    # roundoff produces an apparently finite R-hat instead of NaN.
    tf.debugging.assert_equal(tf.abs(balanced-.5), tf.fill(tf.shape(balanced), tf.constant(.5, tf.float64)))
    # Equal pooled frequencies must not hide different event rates by chain.
    different_rates = tf.constant([([1.]*4+[0.])*80]*2 + [([0.]*4+[1.])*80]*2, tf.float64)[:, :, None]
    maximum, flag, _ = evaluate(different_rates)
    assert bool(flag[0]) and float(maximum[0]) > 1.01
    for values in (tf.zeros_like(balanced), tf.ones_like(balanced)):
        maximum, flag, raw = evaluate(values)
        assert not bool(flag[0])
        assert not bool(tf.math.is_finite(maximum[0]))
    maximum, _, _ = evaluate(tf.constant([0., 0., 1., 1.], tf.float64)[:, None, None]*tf.ones_like(balanced))
    assert not bool(tf.math.is_finite(maximum[0])) or float(maximum[0]) > 1.01
    maximum, flag, raw = evaluate(2.*balanced-1.)
    assert not bool(flag[0])
    tf.debugging.assert_equal(tf.math.is_finite(maximum), tf.math.is_finite(raw['maximum']))
    if bool(tf.math.is_finite(maximum[0])):
        tf.debugging.assert_equal(maximum, raw['maximum'])
    unbalanced = tf.constant([([1.]*4+[0.])*80]*4, tf.float64)[:, :, None]
    maximum, flag, raw = evaluate(unbalanced)
    assert not bool(flag[0])
    tf.debugging.assert_equal(maximum, raw['maximum'])


@pytest.mark.parametrize('draws', [1000, 2000])
def test_long_scope_estimator_matches_actual_runtime_at_existing_looks(draws):
    from bayesfilter.inference.surrogate_hmc_equivalence import _batch_statistics
    from bayesfilter.testing.surrogate_hmc_mcse_randomization import diagnostic_batch_statistics
    values = tf.random.stateless_normal([4, draws, 15], [20260923, draws], dtype=tf.float64)
    # Include binary and correlated function shapes in this formula parity check.
    values = tf.concat([values[:, :, :6], tf.cast(values[:, :, 6:] > 0., tf.float64)], axis=2)
    for expected, actual in zip(_batch_statistics(draws, 15)(values), diagnostic_batch_statistics(draws)(values)):
        tf.debugging.assert_near(expected, actual, atol=1e-13, rtol=1e-13)
    with pytest.raises(ValueError, match='unplanned inferential look'):
        _batch_statistics(8000, 15)


def test_binary_exception_cannot_hide_unexpected_nonfinite_diagnostics():
    from scripts.calibrate_surrogate_hmc_mcse_randomization import check_undefined_diagnostics
    row = dict(original_valid=False, candidate_valid=True, function_rhat=[1.],
        function_rhat_original=[float('nan')], function_rhat_folded=[float('nan')],
        folded_binary_uninformative=[True])
    assert len(check_undefined_diagnostics(row)) == 2
    with pytest.raises(ValueError, match='unexpected nonfinite'):
        check_undefined_diagnostics(dict(row, mcse=[float('nan')]))
    with pytest.raises(ValueError, match='unexpected nonfinite'):
        check_undefined_diagnostics(dict(row, folded_binary_uninformative=[False]))
    with pytest.raises(ValueError, match='original rule'):
        check_undefined_diagnostics(dict(row, original_valid=True))


def test_new_protocol_is_prospective_and_long_scope_runs_through_actual_worker(tmp_path, monkeypatch):
    import json
    from scripts import calibrate_surrogate_hmc_mcse_randomization as worker
    old = worker.protocol_definition('randomization-v1')
    new = worker.protocol_definition('binary-fold-v2')
    assert old['data_key'] != new['data_key'] and old['permutation_key'] != new['permutation_key']
    assert old['groups'] == worker.GROUPS
    assert old['looks']['ar_borderline'] == (1000, 2000)
    assert new['looks']['ar_borderline'] == (4000, 8000)
    # One shape/wiring smoke, explicitly incapable of emitting calibration pass.
    new['groups'] = (('ar_borderline', .9, 0, 1, 987654),)
    monkeypatch.setattr(worker, 'decision', lambda groups: dict(status='DEBUG_SMOKE_ONLY'))
    report = {}
    worker.run(tmp_path, report, lambda: 60., lambda: 60., new)
    assert report['status'] == 'DEBUG_SMOKE_ONLY' and report['completed_replications'] == 1
    row = json.loads((tmp_path/'ar_borderline-replications.jsonl').read_text())
    assert set(row['looks']) == {'4000', '8000'}
    assert all(x['all_inputs_finite'] for x in row['looks'].values())
    assert all(len(x['function_rhat']) == 15 for x in row['looks'].values())
