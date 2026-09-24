"""Independent density and genealogy checks for the optional physical score."""
import math
import struct

import pytest
import tensorflow as tf

from bayesfilter.score_study.complete_data_score_tf import initial_score, increment_score
from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel, resampling_score_control
from bayesfilter.score_study.gaussian_tf import parameterized_model


THETA = [.62, -.8, -.6, .9, .25, -.3]


def fixture_constant(value):
    # The existing model uses tf.cast(Python float), which first creates FP32.
    return struct.unpack("f", struct.pack("f", value))[0]


M0, H0 = fixture_constant(.7), 1.+fixture_constant(.15)


def scalar_initial(x):
    residual = x - M0 * THETA[4]
    variance = H0 * math.exp(2 * THETA[5])
    return [0., 0., 0., 0., M0 * residual / variance, residual**2 / variance - 1.]


def scalar_increment(previous, x, y, c=.35, b=.12):
    rq = x - THETA[0] * previous - fixture_constant(c) * math.sin(previous)
    rr = y - H0 * THETA[3] * x - fixture_constant(b) * x*x
    q, r = math.exp(2 * THETA[1]), math.exp(2 * THETA[2])
    return [previous*rq/q, rq*rq/q - 1., rr*rr/r - 1., H0*x*rr/r, 0., 0.]


def test_scalar_partials_hold_states_fixed_and_include_initial_law():
    dtype = tf.float64
    x = tf.constant([[.4], [-1.2]], dtype)
    previous = tf.constant([[-.8], [.7]], dtype)
    model = parameterized_model(tf.constant(THETA, dtype), 1, 1)
    expected_initial = tf.transpose(tf.constant([scalar_initial(.4), scalar_initial(-1.2)], dtype))
    expected_increment = tf.transpose(tf.constant([
        scalar_increment(-.8, .4, .3), scalar_increment(.7, -1.2, .3)], dtype))
    tf.debugging.assert_near(initial_score(x, model), expected_initial, atol=2e-12, rtol=2e-12)
    tf.debugging.assert_near(increment_score(previous, x, tf.constant([.3], dtype), model, .35, .12),
                             expected_increment, atol=2e-12, rtol=2e-12)


def test_multivariate_fixed_state_joint_density_finite_differences():
    dtype = tf.float64
    theta = tf.constant(THETA, dtype)
    previous = tf.constant([[.4, -.3], [-1.2, .9]], dtype)
    x = tf.constant([[-.8, .2], [.7, -1.1]], dtype)
    y = tf.constant([.3, -.4], dtype)

    def log_density(z, mean, covariance):
        residual = z-mean
        return -.5*(tf.linalg.slogdet(covariance)[1] +
                    tf.reduce_sum(residual * tf.transpose(tf.linalg.solve(covariance, tf.transpose(residual))), -1))

    def joint(parameters):
        A, _, H, _, m, _, P, _, Q, _, R, _ = parameterized_model(parameters, 2, 2)
        return (log_density(previous, m, P) + log_density(x, previous @ tf.transpose(A), Q)
                + log_density(tf.broadcast_to(y, x.shape), x @ tf.transpose(H), R))

    model = parameterized_model(theta, 2, 2)
    score = initial_score(previous, model) + increment_score(previous, x, y, model)
    for p, direction in enumerate(tf.unstack(tf.eye(6, dtype=dtype))):
        h = tf.constant(1e-5, dtype)
        fd = (joint(theta+h*direction)-joint(theta-h*direction))/(2*h)
        tf.debugging.assert_near(score[p], fd, rtol=1e-8, atol=1e-8)


def arguments(n=7, horizon=3):
    dtype = tf.float64
    return (tf.constant([[.3], [-.6], [.8]][:horizon], dtype),
        tf.random.stateless_normal([n, 1], [917, 1], dtype=dtype),
        tf.random.stateless_normal([horizon, n, 1], [917, 2], dtype=dtype),
        tf.random.stateless_uniform([horizon+1, n], [917, 3], dtype=dtype),
        tf.constant([[0. if i % 2 else .999999 for i in range(n)]]*horizon, dtype),
        tf.constant([[.2], [-.4], [.9]][:horizon], dtype),
        tf.constant([[[.4]], [[.7]], [[.3]]][:horizon], dtype),
        tf.constant([-1., -2., -1.5][:horizon], dtype))


def scalar_replay(args, constant_twist, uniform_bits=None):
    """Independent Python scalar proposal, weights, and full-path genealogy."""
    obs, initial, noise, uniforms, mixture, centers, covariances, floors = [x.numpy().tolist() for x in args]
    n, horizon = len(initial), len(obs)
    q, r = math.exp(2*THETA[1]), math.exp(2*THETA[2])
    x = [M0*THETA[4]+math.sqrt(H0*math.exp(2*THETA[5]))*z[0] for z in initial]
    paths = [[state] for state in x]

    def normal(z, mean, variance):
        return math.exp(-.5*(z-mean)**2/variance)/math.sqrt(2*math.pi*variance)

    def prediction(state):
        return THETA[0]*state+fixture_constant(.35)*math.sin(state)

    def future(state, t):
        return (1. if constant_twist or t == horizon else
                normal(centers[t][0], prediction(state), q+covariances[t][0][0])+math.exp(floors[t]))

    ancestors = []
    controls = []

    def resample(weights, u, state, history):
        total, cumulative = sum(weights), []
        for w in weights:
            cumulative.append((cumulative[-1] if cumulative else 0.)+w/total)
        ids = [next((i for i, end in enumerate(cumulative) if end > value), n-1) for value in u]
        if uniform_bits is not None and len(ancestors) < horizon:
            full_scores = []
            for path in history:
                score = scalar_initial(path[0])
                for k in range(len(path)-1):
                    score = [a+b for a, b in zip(score, scalar_increment(path[k], path[k+1], obs[k][0]))]
                full_scores.append(score)
            # Enumerate the declared grid independently of the count formula.
            grid_ids = [next((i for i, end in enumerate(cumulative) if end > k/2**uniform_bits), n-1)
                        for k in range(2**uniform_bits)]
            controls.append([math.sqrt(n)*(sum(full_scores[i][p] for i in ids)/n
                -sum(full_scores[i][p] for i in grid_ids)/len(grid_ids)) for p in range(6)])
        ancestors.append(ids)
        return [state[i] for i in ids], [history[i][:] for i in ids]

    x, paths = resample([future(z, 0) for z in x], uniforms[0], x, paths)
    clouds = []
    for t in range(horizon):
        proposed, weights = [], []
        for i, previous in enumerate(x):
            mean = prediction(previous)
            v, center, floor = covariances[t][0][0], centers[t][0], math.exp(floors[t])
            component = normal(center, mean, q+v)
            adapted = not constant_twist and mixture[t][i] < component/(component+floor)
            state = (mean+q/(q+v)*(center-mean)+math.sqrt(q*v/(q+v))*noise[t][i][0]
                     if adapted else mean+math.sqrt(q)*noise[t][i][0])
            proposed.append(state)
            psi = 1. if constant_twist else normal(state, center, v)+floor
            g = normal(obs[t][0], H0*THETA[3]*state+fixture_constant(.12)*state*state, r)
            weights.append(g*future(state, t+1)/psi)
            paths[i].append(state)
        clouds.append([[state] for state in proposed])
        if t+1 == horizon:
            full_scores = []
            for path in paths:
                score = scalar_initial(path[0])
                for k in range(horizon):
                    score = [a+b for a, b in zip(score, scalar_increment(path[k], path[k+1], obs[k][0]))]
                full_scores.append(score)
            fisher = [sum(w*score[p] for w, score in zip(weights, full_scores))/sum(weights) for p in range(6)]
        x, paths = resample(weights, uniforms[t+1], proposed, paths)
    outputs = clouds, fisher, ancestors
    return (*outputs, controls) if uniform_bits is not None else outputs


@pytest.mark.parametrize("constant_twist", [False, True])
def test_score_follows_actual_nonidentity_ancestors(constant_twist):
    args = arguments()
    expected_clouds, expected_score, ancestors = scalar_replay(args, constant_twist)
    assert ancestors[0] != list(range(7))
    assert ancestors[1] != list(range(7))
    kernel = make_fitted_twist_kernel(1, 1, 7, 3, constant_twist=constant_twist,
        transition_curve=.35, observation_curve=.12, include_fisher_score=True)
    _, _, clouds, fisher = kernel(tf.constant(THETA, tf.float64), *args)
    tf.debugging.assert_near(clouds, tf.constant(expected_clouds, tf.float64), rtol=2e-11, atol=2e-11)
    tf.debugging.assert_near(fisher, tf.constant(expected_score, tf.float64), rtol=2e-10, atol=2e-10)
    assert kernel.experimental_get_tracing_count() == 1


def test_existing_outputs_unchanged_and_terminal_resampling_irrelevant():
    theta, args = tf.constant(THETA, tf.float64), arguments()
    kwargs = dict(transition_curve=.35, observation_curve=.12)
    old = make_fitted_twist_kernel(1, 1, 7, 3, **kwargs)(theta, *args)
    kernel = make_fitted_twist_kernel(1, 1, 7, 3, **kwargs, include_fisher_score=True)
    new = kernel(theta, *args)
    for before, after in zip(old, new[:3]):
        tf.debugging.assert_near(before, after, rtol=2e-12, atol=2e-12)
    changed = list(args)
    changed[3] = tf.concat([args[3][:-1], tf.zeros_like(args[3][-1:])], 0)
    other = kernel(theta, *changed)
    tf.debugging.assert_equal(new[3], other[3])


def test_single_particle_single_observation_keeps_initial_score():
    args = arguments(1, 1)
    _, score, _ = scalar_replay(args, False)
    kernel = make_fitted_twist_kernel(1, 1, 1, 1, transition_curve=.35,
        observation_curve=.12, include_fisher_score=True)
    actual = kernel(tf.constant(THETA, tf.float64), *args)[3]
    tf.debugging.assert_near(actual, tf.constant(score, tf.float64), rtol=2e-11, atol=2e-11)
    assert abs(float(actual[4])) > 1e-5


@pytest.mark.parametrize("cdf_values", [[0., .5, .95], [.125, .125, 1.02], [.249, .751, .98]])
def test_resampling_control_center_matches_exhaustive_lattice(cdf_values):
    cdf = tf.constant(cdf_values, tf.float32)
    represented = cdf.numpy().tolist()
    grid_ids = [next((i for i, end in enumerate(represented) if k/8 < end), 2) for k in range(8)]
    scores = [[2., -3., 7.], [-.5, 4., 1.]]
    additive = tf.constant(scores, tf.float32)
    all_draws = resampling_score_control(additive, tf.constant(grid_ids), cdf, 3)
    tf.debugging.assert_near(all_draws, tf.zeros([2], tf.float64), atol=1e-14, rtol=0.)
    selected = [2, 0, 2]
    expected = [math.sqrt(3)*(sum(row[i] for i in selected)/3
                -sum(row[i] for i in grid_ids)/8) for row in scores]
    actual = resampling_score_control(additive, tf.constant(selected), cdf, 3)
    tf.debugging.assert_near(actual, tf.constant(expected, tf.float64), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("constant_twist", [False, True])
def test_controls_follow_actual_genealogy_and_exclude_terminal_draw(constant_twist):
    args = list(arguments())
    args[3] = tf.floor(args[3]*16)/16
    expected_clouds, expected_score, _, expected_controls = scalar_replay(args, constant_twist, 4)
    kernel = make_fitted_twist_kernel(1, 1, 7, 3, constant_twist=constant_twist,
        transition_curve=.35, observation_curve=.12, include_fisher_score=True,
        include_resampling_controls=True, resampling_uniform_bits=4)
    output = kernel(tf.constant(THETA, tf.float64), *args)
    for actual, expected in zip(output[2:], (expected_clouds, expected_score, expected_controls)):
        tf.debugging.assert_near(actual, tf.constant(expected, tf.float64), atol=2e-10, rtol=2e-10)
    args[3] = tf.concat([args[3][:-1], tf.zeros_like(args[3][-1:])], 0)
    changed = kernel(tf.constant(THETA, tf.float64), *args)
    tf.debugging.assert_equal(output[3], changed[3])
    tf.debugging.assert_equal(output[4], changed[4])
    assert kernel.experimental_get_tracing_count() == 1


def test_control_requires_an_explicit_representable_sampling_law():
    with pytest.raises(ValueError, match="Fisher"):
        make_fitted_twist_kernel(1, 1, 7, 2, include_resampling_controls=True, resampling_uniform_bits=23)
    for bits in (None, 0, 24, True):
        with pytest.raises(ValueError, match="lattice"):
            make_fitted_twist_kernel(1, 1, 7, 2, "float32", include_fisher_score=True,
                include_resampling_controls=True, resampling_uniform_bits=bits)


def test_regression_can_discard_input_roundoff_without_changing_resolved_directions():
    from bayesfilter.score_study.combinations_tf import make_combination_kernels
    fit_old, apply_old, _, _ = make_combination_kernels(1, 2)
    fit_safe, apply_safe, _, _ = make_combination_kernels(1, 2, control_input_dtype_name="float32")
    signal = tf.constant([[-1.], [-1.], [1.], [1.]], tf.float64)
    for residual, expected_rank in ((1e-7, 1), (.25, 2)):
        other = signal + tf.constant([[-residual], [residual], [-residual], [residual]], tf.float64)
        controls = tf.concat([signal, other], 1)
        target = signal + tf.constant([[-.5], [.5], [-.5], [.5]], tf.float64)
        old, old_rank, old_valid = fit_old(target, controls)
        safe, rank, valid = fit_safe(target, controls)
        assert bool(valid) and bool(old_valid) and int(old_rank) == 2 and int(rank) == expected_rank
        if expected_rank == 2:
            tf.debugging.assert_equal(old, safe)
            tf.debugging.assert_equal(apply_old(target, controls, old), apply_safe(target, controls, safe))
        else:
            assert float(tf.reduce_max(tf.abs(old))) > 1e6
            assert float(tf.reduce_max(tf.abs(safe))) < 1.
            tf.debugging.assert_all_finite(apply_safe(target, controls, safe), "unsafe rank repair")
