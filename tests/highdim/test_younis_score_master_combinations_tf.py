"""Conditional IWSG centering and held-fixed combinations, CPU/XLA reference."""
import tensorflow as tf

from bayesfilter.score_study.combinations_tf import (
    make_combination_kernels, make_frozen_kdm_controls,
)


def normal_quadrature(n):
    off = tf.sqrt(tf.cast(tf.range(1, n), tf.float64))
    jacobi = tf.linalg.diag(off, k=1) + tf.linalg.diag(off, k=-1)
    nodes, vectors = tf.linalg.eigh(jacobi)
    return nodes, vectors[0]**2


def test_mixture_iwsg_controls_have_zero_integral_for_each_frozen_cloud():
    nodes, quadrature_weights = normal_quadrature(96)
    kernel = make_frozen_kdm_controls(2, 1, 192)
    weights = tf.constant([.35, .65], tf.float64)
    covariance = tf.constant([[[.8]], [[1.1]]], tf.float64)
    innovations = tf.concat([nodes, nodes], axis=0)[:, None]
    uniforms = tf.concat([tf.fill([96], tf.constant(.1, tf.float64)),
                          tf.fill([96], tf.constant(.8, tf.float64))], axis=0)
    integration_weights = tf.reshape(weights[:, None] * quadrature_weights[None, :], [-1])
    for centers in ([[-.4], [.6]], [[.7], [-.2]]):
        controls, valid = kernel(weights, tf.constant(centers, tf.float64), covariance,
                                  uniforms, innovations)
        assert bool(valid)
        integral = tf.reduce_sum(integration_weights[:, None] * controls, axis=0)
        tf.debugging.assert_near(integral, tf.zeros([2], tf.float64), atol=1e-10, rtol=0.)
    assert kernel.experimental_get_tracing_count() == 1


def test_frozen_exact_control_reduces_variance_and_preserves_biased_mean():
    fit, apply, _, _ = make_combination_kernels(1, 2)
    controls_kernel = make_frozen_kdm_controls(1, 1, 64)
    def controls(seed):
        z = tf.random.stateless_normal([64, 1], seed, dtype=tf.float64)
        return controls_kernel(tf.ones([1], tf.float64), tf.zeros([1, 1], tf.float64),
            tf.ones([1, 1, 1], tf.float64), tf.fill([64], tf.constant(.5, tf.float64)), z)[0]
    training = controls([2, 7])
    coefficient = tf.constant([[3.], [.7]], tf.float64)
    # A baseline with expectation 10 and a target of 100 remains biased after
    # centering. The CV identity concerns its expectation, not the true score.
    fitted, rank, valid = fit(10. + training @ coefficient, training)
    assert bool(valid) and int(rank) == 2
    evaluation = controls([8, 11])
    corrected = apply(10. + evaluation @ coefficient, evaluation, fitted)
    tf.debugging.assert_near(corrected, tf.fill([64, 1], tf.constant(10., tf.float64)), atol=1e-11)


def test_oracle_blend_minimizes_error_including_bias_cross_term():
    _, _, fit, apply = make_combination_kernels(2, 1)
    oracle = tf.constant([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], tf.float64)
    baseline = oracle + tf.constant([[3., 1.], [1., 2.], [2., 4.], [4., 1.]], tf.float64)
    auxiliary = oracle + tf.constant([[-1., -.5], [-2., -1.], [-.5, -1.], [-1., -2.]], tf.float64)
    alpha, mse = fit(baseline, auxiliary, oracle)
    residual = apply(baseline, auxiliary, alpha) - oracle
    derivative = tf.reduce_sum(residual * (auxiliary-baseline))
    tf.debugging.assert_near(derivative, tf.constant(0., tf.float64), atol=1e-12)
    for delta in (-.1, .1):
        nearby = apply(baseline, auxiliary, alpha+delta) - oracle
        assert float(tf.reduce_mean(tf.reduce_sum(nearby**2, axis=1))) > float(mse)
    duplicate_alpha, _ = fit(baseline, baseline, oracle)
    assert float(duplicate_alpha) == 0.


def test_validation_values_cannot_change_fitted_coefficients(tmp_path, monkeypatch):
    from bayesfilter.score_study import combinations, runtime
    monkeypatch.setattr(runtime, "configure_runtime", lambda **kw: {})
    # Synthetic independent records isolate the fit/apply boundary. Numerical
    # mixture sampling and actual consumer wiring have separate tests.
    c = [[-2., 1.], [-1., -1.], [1., -.5], [2., .5]]
    def pair(i, shift=0.):
        g = [3*c[i][0] + .7*c[i][1] + shift] * 6
        return {"ledh": {"score": g, "oracle_score": [0.]*6,
                         "diagnostics": {"kdm_zero_control": c[i]}},
                "integrated_kdm": {"score": [1.+shift]*6}}
    records = {"calibration": {(1, i): pair(i) for i in range(4)},
               "validation": {(2, i): pair(i) for i in range(4)}}
    state = {"fingerprint": {}, "study": {"settings": {"dimension": 1}, "evidence_class": "mechanics"}}
    monkeypatch.setattr(combinations, "combination_records", lambda _: (state, records))
    first = combinations.assemble_combinations(tmp_path / "first")
    records["validation"] = {(2, i): pair(i, 1000.) for i in range(4)}
    second = combinations.assemble_combinations(tmp_path / "second")
    assert first["coefficient"] == second["coefficient"]
    assert first["blend_alpha"] == second["blend_alpha"]
    assert first["comparisons"] != second["comparisons"]
