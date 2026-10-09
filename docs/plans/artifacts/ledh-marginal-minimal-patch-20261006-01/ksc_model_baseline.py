"""Frozen diagnostic model for matched pre/post comparison."""
def ksc_sv_canonical_model(theta_fixed: Tensor):
    """KSC mixture SV — CORRECTED 2026-08-23.

    STATE IS 1-DIMENSIONAL: the log-volatility AR(1) h' = gamma*h + eta,
    matching the reference adapter (`dimension=1` in the frozen target).
    `log_beta` is a PARAMETER (theta[1]), entering the observation offset
    as 2*log_beta — the previous onboarding wrongly promoted it to a
    second state with a fabricated near-deterministic transition
    (Q22=1e-8), which made the transition density price the flow's
    legitimate displacement at 1/1e-8 and produced the wrong -19283 value
    (registry note: onboarding error, not an algorithm property).

    Flow input: moment-matched Gaussian of the KSC log-chi-square mixture
    (derived constants); the observed quantity is h + 2*log_beta +
    mixture_noise, so H = [1] and the mixture mean enters the offset.
    The likelihood below retains all seven mixture components. Gaussian
    moment matching is used only for the proposal flow. A transformation
    Jacobian preserves scores within this density family; it does not make
    the mixture exact for native SV or make a Gaussian Kalman filter an
    exact mixture-likelihood oracle.
    """
    import math
    theta_fixed = tf.convert_to_tensor(theta_fixed, DTYPE)
    weights = tf.constant([0.0073, 0.10556, 2e-05, 0.04395, 0.34001, 0.24566, 0.2575], DTYPE)
    means = tf.constant([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819], DTYPE) - tf.constant(1.2704, DTYPE)
    variances = tf.constant([5.79596, 2.61369, 5.1795, 0.16735, 0.64009, 0.34023, 1.26261], DTYPE)
    mixture_mean = tf.reduce_sum(weights * means)
    mixture_var = tf.reduce_sum(weights * (variances + tf.square(means))) - tf.square(mixture_mean)

    def gamma_of(theta):
        return 0.5 * (1.0 + tf.math.erf(theta[0] / tf.sqrt(tf.constant(2.0, DTYPE))))

    def transition_mean_fn(theta, points):
        return gamma_of(theta) * points
    _direction = [tf.zeros([2], DTYPE)]

    def set_score_direction(direction: Tensor) -> None:
        _direction[0] = tf.convert_to_tensor(direction, DTYPE)

    def transition_mean_tangent_fn(theta, points, d_points):
        d_theta = _direction[0]
        gamma = gamma_of(theta)
        normalizer = tf.constant(1.0 / math.sqrt(2.0 * math.pi), DTYPE)
        dgamma = normalizer * tf.exp(-0.5 * tf.square(theta[0])) * d_theta[0]
        return dgamma * points + gamma * d_points
    two_log_beta = 2.0 * theta_fixed[1]

    def observation_log_density_fn(theta, points, observation):
        w = observation[0] - 2.0 * theta[1] - points[:, 0]
        terms = tf.math.log(weights)[None, :] - 0.5 * (tf.square(w[:, None] - means[None, :]) / variances[None, :] + tf.math.log(variances)[None, :] + tf.constant(math.log(2.0 * math.pi), DTYPE))
        return tf.reduce_logsumexp(terms, axis=1)

    def observation_log_density_tangent_fn(theta, points, observation, d_points):
        d_theta = _direction[0]
        w = observation[0] - 2.0 * theta[1] - points[:, 0]
        terms = tf.math.log(weights)[None, :] - 0.5 * (tf.square(w[:, None] - means[None, :]) / variances[None, :] + tf.math.log(variances)[None, :] + tf.constant(math.log(2.0 * math.pi), DTYPE))
        responsibilities = tf.nn.softmax(terms, axis=1)
        location_score = tf.reduce_sum(responsibilities * (w[:, None] - means[None, :]) / variances[None, :], axis=1)
        d_w = -d_points[:, 0] - 2.0 * d_theta[1]
        return -location_score * d_w
    model = NonlinearScoreModel(transition_mean_fn=transition_mean_fn, transition_mean_tangent_fn=transition_mean_tangent_fn, observation_fn=lambda points: points + mixture_mean + two_log_beta, observation_jacobian_fn=lambda points: tf.ones([tf.shape(points)[0], 1, 1], DTYPE), observation_tangent_fn=lambda points, d_points: d_points + 2.0 * _direction[0][1], process_covariance=tf.ones([1, 1], DTYPE), observation_covariance=mixture_var[None, None], observation_log_density_fn=observation_log_density_fn, observation_log_density_tangent_fn=observation_log_density_tangent_fn)
    return (model, set_score_direction)
