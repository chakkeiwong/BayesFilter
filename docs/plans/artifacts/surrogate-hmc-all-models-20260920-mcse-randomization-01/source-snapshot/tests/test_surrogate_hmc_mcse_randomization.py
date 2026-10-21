"""Independent finite enumeration of the proposed diagnostic, not HMC evidence."""
import itertools

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
