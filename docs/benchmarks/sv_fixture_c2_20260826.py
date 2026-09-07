"""SV fixture family for the C2 campaign (Phase C3; plan CF9 scope pin).

Model (ZC24 Example 1, txt:72-85, synthetic values txt:1998-2001,
extended to vector states with the attempt04 coupling device —
a DECLARED extension, audited at D1):

    X_t = A X_{t-1} + sigma * eps_x,   A = gamma*I + 0.1*randn/(n-1)
    Y_{t,i} = eps_{y,i} * beta * exp(X_{t,i}/2)   (diagonal observation)
    X_0 ~ N(0, P0),  P0 solves P = A P A' + sigma^2 I  (stationary)

Pinned parameters: gamma = 0.6, sigma = 1, beta = 0.4 (the paper's
synthetic setting). Engine-facing adapter is TF float64; reference
filters are independent NumPy f64 implementations (backend rule's
diagnostic-reference exception). References:

- n=1: dense-grid HMM filter (near-exact; resolution-validated).
- n=2: 2-D tensor-grid filter (cross-check) + bootstrap particle.
- n>=2: bootstrap particle with per-step ESS degeneracy screen and
  R-replicate SE-of-mean estimator (campaign plan C3 contract).
"""
import math

import numpy as np
import tensorflow as tf

DTYPE = tf.float64
GAMMA, SIGMA, BETA = 0.6, 1.0, 0.4


def sv_model(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if n == 1:
        A = np.array([[GAMMA]])
    else:
        A = GAMMA * np.eye(n) + 0.1 * rng.standard_normal((n, n)) / (n - 1)
    if np.max(np.abs(np.linalg.eigvals(A))) >= 0.999:
        raise ValueError("sv fixture: unstable A draw (fail closed)")
    Q = SIGMA**2 * np.eye(n)
    # stationary covariance: P = A P A' + Q (fixed-point iteration)
    P0 = Q.copy()
    for _ in range(10_000):
        P_next = A @ P0 @ A.T + Q
        if np.max(np.abs(P_next - P0)) < 1e-14:
            P0 = P_next
            break
        P0 = P_next
    return {"A": A, "Q": Q, "P0": P0, "beta": BETA, "n": n}


def sv_simulate(model: dict, horizon: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7)
    n = model["n"]
    x = rng.multivariate_normal(np.zeros(n), model["P0"])
    ys = []
    for t in range(horizon):
        if t > 0:
            x = model["A"] @ x + SIGMA * rng.standard_normal(n)
        ys.append(rng.standard_normal(n) * model["beta"] * np.exp(x / 2.0))
    return np.stack(ys)


def sv_obs_log_density_np(x: np.ndarray, y: np.ndarray, beta: float) -> np.ndarray:
    """log p(y | x) = sum_i N(y_i; 0, beta^2 e^{x_i}); x [..., n], y [n]."""

    return np.sum(
        -0.5 * math.log(2.0 * math.pi)
        - math.log(beta)
        - x / 2.0
        - (y**2) * np.exp(-x) / (2.0 * beta**2),
        axis=-1,
    )


def sv_adapter(model: dict):
    """TF adapter with the DensityKernelAdapter callback contract."""

    from bayesfilter.highdim.squared_tt_engine_v0_tf import DensityKernelAdapter

    n = model["n"]
    A = tf.constant(model["A"], DTYPE)
    beta = float(model["beta"])
    chol_q = tf.constant(np.linalg.cholesky(model["Q"]), DTYPE)
    chol_p0 = tf.constant(np.linalg.cholesky(model["P0"]), DTYPE)
    logdet_q = float(np.sum(np.log(np.diag(np.linalg.cholesky(model["Q"])))))
    logdet_p0 = float(np.sum(np.log(np.diag(np.linalg.cholesky(model["P0"])))))

    def _mvn(x, mean, chol, logdet):
        diff = tf.transpose(
            tf.linalg.triangular_solve(chol, tf.transpose(x - mean), lower=True)
        )
        quad = tf.reduce_sum(tf.square(diff), axis=-1)
        return -0.5 * (n * math.log(2.0 * math.pi) + quad) - logdet

    def transition(xc, xp):
        return _mvn(xc, tf.linalg.matmul(xp, A, transpose_b=True), chol_q, logdet_q)

    def observation(xc, y):
        y = tf.convert_to_tensor(y, DTYPE)
        return tf.reduce_sum(
            -0.5 * math.log(2.0 * math.pi)
            - math.log(beta)
            - xc / 2.0
            - tf.square(y)[None, :] * tf.exp(-xc) / (2.0 * beta**2),
            axis=-1,
        )

    def initial(xc):
        return _mvn(xc, tf.zeros([n], DTYPE), chol_p0, logdet_p0)

    return DensityKernelAdapter(
        state_dim=n,
        transition_log_density=transition,
        observation_log_density=observation,
        initial_log_density=initial,
    )


def sv_grid_reference_1d(model: dict, ys: np.ndarray,
                         width: float = 10.0, points: int = 4001) -> list:
    """Dense-grid HMM filter, n=1: per-step log-likelihood increments."""

    assert model["n"] == 1
    a = float(model["A"][0, 0])
    grid = np.linspace(-width, width, points)
    dx = grid[1] - grid[0]
    trans = np.exp(
        -0.5 * (grid[None, :] - a * grid[:, None]) ** 2 / SIGMA**2
    ) / math.sqrt(2.0 * math.pi * SIGMA**2)
    prior = np.exp(-0.5 * grid**2 / model["P0"][0, 0]) / math.sqrt(
        2.0 * math.pi * model["P0"][0, 0]
    )
    steps = []
    density = prior
    for t, y in enumerate(ys):
        if t > 0:
            density = (density[:, None] * trans).sum(axis=0) * dx
        like = np.exp(sv_obs_log_density_np(grid[:, None], y, model["beta"]))
        joint = density * like
        increment = joint.sum() * dx
        steps.append(math.log(increment))
        density = joint / increment
    return steps


def sv_grid_reference_2d(model: dict, ys: np.ndarray,
                         width: float = 9.0, points: int = 241) -> list:
    """2-D tensor-grid filter (cross-check for the particle machinery)."""

    assert model["n"] == 2
    g = np.linspace(-width, width, points)
    dx = g[1] - g[0]
    X1, X2 = np.meshgrid(g, g, indexing="ij")
    states = np.stack([X1.ravel(), X2.ravel()], axis=1)      # [M, 2]
    A = model["A"]
    means = states @ A.T                                      # [M, 2]
    # transition kernel matrix [M_from, M_to]
    d1 = g[None, :] - means[:, 0:1]
    d2 = g[None, :] - means[:, 1:2]
    k1 = np.exp(-0.5 * d1**2 / SIGMA**2)
    k2 = np.exp(-0.5 * d2**2 / SIGMA**2)
    norm = 1.0 / (2.0 * math.pi * SIGMA**2)
    P0 = model["P0"]
    inv0 = np.linalg.inv(P0)
    quad0 = np.einsum("mi,ij,mj->m", states, inv0, states)
    prior = np.exp(-0.5 * quad0) / (
        2.0 * math.pi * math.sqrt(np.linalg.det(P0))
    )
    density = prior
    steps = []
    M = states.shape[0]
    for t, y in enumerate(ys):
        if t > 0:
            # density[from] -> sum over from of density*k1[from,i]*k2[from,j]
            w = density * dx * dx
            grid_new = np.einsum("m,mi,mj->ij", w, k1, k2) * norm
            density = grid_new.ravel()
        like = np.exp(sv_obs_log_density_np(states, y, model["beta"]))
        joint = density * like
        increment = joint.sum() * dx * dx
        steps.append(math.log(increment))
        density = joint / increment
    return steps


def sv_particle_reference(model: dict, ys: np.ndarray, n_particles: int,
                          replicates: int, seed: int) -> dict:
    """Bootstrap PF log-likelihood with degeneracy screen (plan C3).

    Estimator: per replicate one PF log-likelihood; reference = mean
    over replicates; MC error = SE of that mean. Screen: minimum
    per-step normalized ESS over the run, per replicate."""

    n = model["n"]
    A, beta, P0 = model["A"], model["beta"], model["P0"]
    totals, min_ess = [], []
    per_step = np.zeros((replicates, len(ys)))
    for r in range(replicates):
        rng = np.random.default_rng(seed + 1000 * r)
        x = rng.multivariate_normal(np.zeros(n), P0, size=n_particles)
        loglik, ess_track = 0.0, []
        for t, y in enumerate(ys):
            if t > 0:
                x = x @ A.T + SIGMA * rng.standard_normal((n_particles, n))
            logw = sv_obs_log_density_np(x, y, beta)
            m = logw.max()
            w = np.exp(logw - m)
            increment = math.log(w.mean()) + m
            loglik += increment
            per_step[r, t] = increment
            wn = w / w.sum()
            ess_track.append(1.0 / np.sum(wn**2) / n_particles)
            idx = rng.choice(n_particles, size=n_particles, p=wn)  # systematic-free bootstrap resample
            x = x[idx]
        totals.append(loglik)
        min_ess.append(min(ess_track))
    totals = np.asarray(totals)
    if replicates > 1:
        per_step_covariance = np.cov(per_step, rowvar=False, ddof=1)
        per_step_se = per_step.std(axis=0, ddof=1) / math.sqrt(replicates)
    else:
        per_step_covariance = np.zeros((len(ys), len(ys)), dtype=np.float64)
        per_step_se = np.full(len(ys), np.nan, dtype=np.float64)
    return {
        "mean_total": float(totals.mean()),
        "se_total": float(totals.std(ddof=1) / math.sqrt(replicates)),
        "per_step_mean": per_step.mean(axis=0).tolist(),
        "per_step_se": per_step_se.tolist(),
        "per_step_replicates": per_step.tolist(),
        "per_step_replicate_covariance": per_step_covariance.tolist(),
        "per_step_mean_covariance": (per_step_covariance / replicates).tolist(),
        "total_replicates": totals.tolist(),
        "min_normalized_ess": float(min(min_ess)),
        "replicates": replicates,
        "n_particles": n_particles,
    }


def sv_gh_hint_factory(model: dict, gh_points: int = 9):
    """Deterministic Gauss-Hermite Gaussian-filter hints (Phase C4).

    M2-JOINT contract: predictive_moment_hint(t, y_t) returns the joint
    filtered moments of (x_t, x_{t-1}) | y_{1:t} in (current, previous)
    order with the lag-one cross-covariance; initial_moment_hint(y_0)
    the t=0 posterior moments. Deterministic (tensor GH update, exact
    conditional algebra for the previous block via y_t ⊥ x_{t-1} | x_t);
    frozen step inputs per the M2 contract. NumPy companion in the
    benchmark fixture, mirroring the attempt04 kalman_hint_factory
    pattern; the engine consumes TF tensors."""

    n = model["n"]
    A, Q, P0, beta = model["A"], model["Q"], model["P0"], model["beta"]
    nodes1, weights1 = np.polynomial.hermite_e.hermegauss(gh_points)
    weights1 = weights1 / math.sqrt(2.0 * math.pi)
    mesh = np.meshgrid(*([nodes1] * n), indexing="ij")
    z_nodes = np.stack([m.ravel() for m in mesh], axis=1)          # [K, n]
    wmesh = np.meshgrid(*([weights1] * n), indexing="ij")
    w_nodes = np.prod(np.stack([w.ravel() for w in wmesh], axis=1), axis=1)
    state = {"mean": None, "cov": None, "t": None}

    def _gh_update(mean_p, cov_p, y):
        chol = np.linalg.cholesky(cov_p)
        x_nodes = mean_p[None, :] + z_nodes @ chol.T                # [K, n]
        logw = sv_obs_log_density_np(x_nodes, y, beta)
        w = w_nodes * np.exp(logw - logw.max())
        w = w / w.sum()
        f_mean = w @ x_nodes
        centered = x_nodes - f_mean[None, :]
        f_cov = (centered * w[:, None]).T @ centered
        return f_mean, f_cov

    def initial_moment_hint(y0):
        f_mean, f_cov = _gh_update(np.zeros(n), P0, np.asarray(y0))
        state["mean"], state["cov"], state["t"] = f_mean, f_cov, 0
        return tf.constant(f_mean, DTYPE), tf.constant(f_cov, DTYPE)

    def predictive_moment_hint(t, y_t):
        assert state["t"] is not None, "initial hint must be requested first"
        assert t == state["t"] + 1, "hints must be requested in step order"
        m_prev, P_prev = state["mean"], state["cov"]
        mean_p = A @ m_prev
        cov_p = A @ P_prev @ A.T + Q
        cross_pred = P_prev @ A.T                    # Cov(x_{t-1}, x_t | y_{1:t-1})
        f_mean, f_cov = _gh_update(mean_p, cov_p, np.asarray(y_t))
        gain = cross_pred @ np.linalg.inv(cov_p)     # E[x_{t-1}|x_t] slope
        p_mean = m_prev + gain @ (f_mean - mean_p)
        p_cov = (
            P_prev - gain @ cov_p @ gain.T + gain @ f_cov @ gain.T
        )
        cross_f = gain @ f_cov                        # Cov(x_{t-1}, x_t | y_{1:t})
        joint_mean = np.concatenate([f_mean, p_mean])
        joint_cov = np.block([[f_cov, cross_f.T], [cross_f, p_cov]])
        state["mean"], state["cov"], state["t"] = f_mean, f_cov, t
        return tf.constant(joint_mean, DTYPE), tf.constant(joint_cov, DTYPE)

    return initial_moment_hint, predictive_moment_hint


def sv_grid_moments_1d(model: dict, ys: np.ndarray,
                       width: float = 12.0, points: int = 8001) -> list:
    """Exact per-step filtered mean/variance at n=1 (hint validation)."""

    assert model["n"] == 1
    a = float(model["A"][0, 0])
    grid = np.linspace(-width, width, points)
    dx = grid[1] - grid[0]
    trans = np.exp(-0.5 * (grid[None, :] - a * grid[:, None]) ** 2 / SIGMA**2)
    trans /= math.sqrt(2.0 * math.pi * SIGMA**2)
    density = np.exp(-0.5 * grid**2 / model["P0"][0, 0])
    density /= density.sum() * dx
    moments = []
    for t, y in enumerate(ys):
        if t > 0:
            density = (density[:, None] * trans).sum(axis=0) * dx
        like = np.exp(sv_obs_log_density_np(grid[:, None], y, model["beta"]))
        joint = density * like
        density = joint / (joint.sum() * dx)
        mean = (grid * density).sum() * dx
        var = ((grid - mean) ** 2 * density).sum() * dx
        moments.append((mean, var))
    return moments
