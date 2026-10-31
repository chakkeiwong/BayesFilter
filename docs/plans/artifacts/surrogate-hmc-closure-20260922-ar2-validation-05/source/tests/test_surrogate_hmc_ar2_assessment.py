"""Independent author-formula checks and non-harm of the AR2 MCSE repair."""
import math

import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_ar2_assessment import (
    METHOD, assess, diagnostic_calculator, lugsail_statistics,
)
from bayesfilter.testing.surrogate_hmc_scoped_assessment import critical_value


def test_lugsail_matches_independent_author_formula_with_truncated_short_batches():
    # Standard-library reference explicitly forms each batch; independent of
    # the TensorFlow reshape/reduction implementation. n=1000 truncates b/3.
    rows = [[[math.sin(i*.037+c)+.01*((i+2*c)%17), math.cos(i*.013-c)]
             for i in range(1000)] for c in range(4)]
    result = lugsail_statistics(1000, 2)(tf.constant(rows, tf.float64))
    expected, raw = [], []
    for batches in (40, 20, 10):
        b = 1000//batches
        def asymptotic_variance(chain, function, length):
            a = len(chain)//length
            center = math.fsum(row[function] for row in chain)/len(chain)
            means = [math.fsum(chain[i][function] for i in range(j*length, (j+1)*length))/length
                     for j in range(a)]
            return length*math.fsum((v-center)**2 for v in means)/(a-1)
        long = [math.fsum(asymptotic_variance(chain, f, b) for chain in rows)/16000 for f in range(2)]
        short = [math.fsum(asymptotic_variance(chain, f, b//3) for chain in rows)/16000 for f in range(2)]
        raw.append([2*l-s for l, s in zip(long, short)])
        expected.append([math.sqrt(max(l, 2*l-s)) for l, s in zip(long, short)])
    tf.debugging.assert_near(result["lugsail_raw_variance"], tf.constant(raw, tf.float64), atol=1e-13)
    tf.debugging.assert_near(result["guarded_mcse_estimates"], tf.constant(expected, tf.float64), atol=1e-13)
    tf.debugging.assert_greater_equal(result["mcse"], result["ordinary_mcse"])
    assert bool(tf.reduce_any(result["lugsail_guard_active"]))


def test_retained_assessment_uses_the_same_repaired_diagnostics_and_wider_intervals():
    values = tf.random.stateless_normal([4, 1000, 13], [714, 19], dtype=tf.float64)
    points, seed = values[:, :, :2], [117, 28]
    diagnostics = diagnostic_calculator(1000)(points, values, tf.constant(seed, tf.int32))
    allowance = tf.fill([13], tf.constant(.001, tf.float64))
    row = assess(points, values, tf.zeros([13], tf.float64), allowance,
                 tf.ones([13], tf.float64), seed=seed)
    assert row["method"] == METHOD
    assert row["qualified"] == bool(diagnostics["qualified"])
    tf.debugging.assert_equal(tf.constant(row["mcse"], tf.float64), diagnostics["mcse"])
    tf.debugging.assert_near(tf.constant(row["upper"], tf.float64)-diagnostics["means"],
                             critical_value()*diagnostics["mcse"]+allowance)
    tf.debugging.assert_greater_equal(diagnostics["mcse"], diagnostics["ordinary_mcse"])


def test_conservative_variance_guard_cannot_admit_constant_chains():
    result = diagnostic_calculator(1000)(tf.ones([4, 1000, 2], tf.float64),
        tf.ones([4, 1000, 13], tf.float64), tf.constant([11, 19], tf.int32))
    assert not bool(result["qualified"])
    assert not bool(result["variance_valid"])
