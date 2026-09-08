"""Row-design v2 validation probe: uniform vs proposal rows at one step.

Design note Section 3:
docs/plans/bayesfilter-row-design-v2-proposal-rows-note-2026-08-19.md.
Rebuilds t=0 and t=1 of the branch-axis program OUTSIDE the engine
(same internals: _fixed_als_fit + RetainedQuadraticForm), on the n=4
ladder fixture. Arm A: current uniform Sobol rows. Arm B: F2 conjugate
proposal rows with mu-corrected weights (alpha=0.25 defensive mixture).
Single seed, single step, descriptive; no engine change.
"""
import os, sys, time
LOG = "/tmp/rowdesign_v2_probe.log"
if "--detach" in sys.argv and os.fork() > 0:
    print(f"detached; output -> {LOG}"); sys.exit(0)
if "--detach" in sys.argv:
    os.setsid(); fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1); os.dup2(fd, 2)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np, tensorflow as tf
from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.retained_quadratic_form_tf import (
    RetainedQuadraticForm, prefix_row_vectors, retained_quadratic_form_from_squared_tt,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DensityKernelAdapter, DiscreteIndicatorBasis1D, EngineConfig,
    _design_rows, _fixed_als_fit, _initial_tt_cores, _product_basis,
)
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64
n, seed, N, rank, deg, hw = 4, 46, 8192, 6, 12, 3.0
ALPHA = 0.25

# ---- fixture (ladder _case family, seed 42+n=46) ----
rng = np.random.default_rng(seed)
A = 0.7 * np.eye(n) + 0.1 * rng.standard_normal((n, n)) / max(1, n - 1)
Q = 0.4 * np.eye(n); H = np.eye(n); R = 0.5 * np.eye(n)
m0 = np.zeros(n); P0 = np.eye(n)
x = rng.multivariate_normal(m0, P0)
ys = []
for t in range(2):
    if t > 0:
        x = A @ x + rng.multivariate_normal(np.zeros(n), Q)
    ys.append(H @ x + rng.multivariate_normal(np.zeros(n), R))
ys = np.stack(ys)

def mvn_logpdf_np(x_, mean, cov):
    d = mean.shape[-1] if np.ndim(mean) > 1 else len(cov)
    diff = x_ - mean
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, diff.T)
    return -0.5 * (d * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L)))
                   + np.sum(sol**2, axis=0))

def mvn_log_density_tf(x_, mean, cov):
    d = int(cov.shape[0])
    chol = np.linalg.cholesky(cov)
    solve = tf.linalg.triangular_solve(tf.constant(chol, DTYPE), tf.transpose(x_ - mean), lower=True)
    quad = tf.reduce_sum(tf.square(solve), axis=0)
    return -0.5 * (d * np.log(2.0 * np.pi) + 2.0 * float(np.sum(np.log(np.diag(chol)))) + quad)

adapter = DensityKernelAdapter(
    state_dim=n,
    transition_log_density=lambda xc, xp: mvn_log_density_tf(
        xc, tf.linalg.matvec(tf.constant(A, DTYPE), xp), Q),
    observation_log_density=lambda xc, y: mvn_log_density_tf(xc, tf.convert_to_tensor(y, DTYPE), R),
    initial_log_density=lambda xc: mvn_log_density_tf(xc, tf.constant(m0, DTYPE), P0),
)

# exact Kalman t=0 posterior and t=1 increment
S0 = P0 + R; K0 = P0 @ np.linalg.inv(S0)
m_f = m0 + K0 @ (ys[0] - m0); P_f = P0 - K0 @ S0 @ K0.T
mean1 = A @ m_f; cov1 = A @ P_f @ A.T + Q
S1 = cov1 + R; innov1 = ys[1] - mean1
kalman_incr1 = float(-0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(S1)[1]
                             + innov1 @ np.linalg.solve(S1, innov1)))

config = EngineConfig(basis_degree=deg, rank=rank, row_count=N, sweeps=3,
    ridge=1e-10, tau=1e-6, coordinate_half_width=hw, seed=91046, row_design="sobol")
current_basis = _product_basis(n, deg)
basis_dim = int(current_basis.bases[0].basis_dim)
half = tf.constant(hw, DTYPE)
current_map = AffineCoordinateMap(offset=tf.zeros([n], DTYPE), matrix=tf.eye(n, dtype=DTYPE) * hw)
conversion = tf.cast(n, DTYPE) * (tf.math.log(half) + tf.math.log(tf.constant(2.0, DTYPE)))
tau = tf.constant(config.tau, DTYPE)

# ---- t=0 (shared; uniform rows — localization showed t=0 is fine) ----
rows0 = _design_rows(config, N, n, (config.seed, 17))
w0 = tf.fill([N], tf.constant(1.0 / N, DTYPE))
x0 = rows0 * half
log_f0 = adapter.initial_log_density(x0) + adapter.observation_log_density(x0, ys[0]) + conversion
shift0 = tf.reduce_logsumexp(log_f0) - tf.math.log(tf.cast(N, DTYPE))
target0 = tf.exp(0.5 * (log_f0 - shift0))
cores0, _d = _fixed_als_fit(current_basis, rows0, target0, w0, _initial_tt_cores(n, basis_dim, rank), config)
suffix_core = tf.zeros([int(cores0[-1].right_rank), basis_dim, 1], DTYPE)
suffix_core = tf.tensor_scatter_nd_update(suffix_core, [[0, 0, 0]], [1.0])
ext_basis = _product_basis(n + 1, deg)
base = retained_quadratic_form_from_squared_tt(
    tuple(cores0) + (TTCore(suffix_core),), ext_basis, split_index=n, tau=0.0,
    prefix_basis=current_basis, coordinate_map=current_map)
z_h0 = base.z_complete_ref
retained = RetainedQuadraticForm(
    prefix_cores=base.prefix_cores, suffix_gram=base.suffix_gram,
    tau=tau * z_h0, z_complete_ref=(1.0 + tau) * z_h0,
    prefix_basis=base.prefix_basis, coordinate_map=base.coordinate_map)

# ---- proposal rows for t=1 (F2 conjugate + alpha uniform mixture) ----
prng = np.random.default_rng(777)
C = np.linalg.inv(np.linalg.inv(Q) + np.linalg.inv(R))
Qi, Ri = np.linalg.inv(Q), np.linalg.inv(R)
def draw_mixture(count):
    zs = np.empty((0, 2 * n))
    while len(zs) < count:
        m = count - len(zs)
        pick_u = prng.random(m) < ALPHA
        z = np.empty((m, 2 * n))
        # uniform component
        z[pick_u] = prng.uniform(-1, 1, (pick_u.sum(), 2 * n))
        # proposal component: x_p ~ N(m_f, P_f); x_c | x_p conjugate
        k = (~pick_u).sum()
        xp = prng.multivariate_normal(m_f, P_f, size=k)
        mc = (Qi @ A @ xp.T).T + (Ri @ ys[1])[None, :]
        xc = np.einsum("ij,nj->ni", C, mc) + prng.multivariate_normal(np.zeros(n), C, size=k)
        z[~pick_u, :n] = xc / hw
        z[~pick_u, n:] = xp / hw
        inside = np.all(np.abs(z) <= 1.0, axis=1)
        zs = np.vstack([zs, z[inside]])
    return zs[:count]

def mixture_mu_density(z):
    xc = z[:, :n] * hw; xp = z[:, n:] * hw
    lp_xp = mvn_logpdf_np(xp, m_f[None, :], P_f)
    mc = (Qi @ A @ xp.T).T + (Ri @ ys[1])[None, :]
    mean_c = np.einsum("ij,nj->ni", C, mc)
    lp_xc = mvn_logpdf_np(xc, mean_c, C)
    p_x = np.exp(lp_xp + lp_xc)                      # physical Lebesgue density
    q_prop_mu = p_x * (hw ** (2 * n)) * (2.0 ** (2 * n))  # mu-density on box
    return (1 - ALPHA) * q_prop_mu + ALPHA * 1.0

def run_t1(z_rows_np, weights_np, label, fit_weights_np=None):
    """fit_weights_np: weights for the ALS objective (default = weights_np,
    the mu-quadrature weights). Passing uniform 1/N with proposal rows
    fits in the L2(q*mu) norm (arm B2) while the shift quadrature still
    uses the mu-corrected weights_np."""
    z_rows = tf.constant(z_rows_np, DTYPE)
    z_w = tf.constant(weights_np, DTYPE)
    fit_w = tf.constant(
        fit_weights_np if fit_weights_np is not None else weights_np, DTYPE
    )
    gram = retained.suffix_gram
    floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
    chol = tf.linalg.cholesky(gram + tf.constant(config.branch_gram_floor, DTYPE)
                              * floor_scale * tf.eye(tf.shape(gram)[0], dtype=DTYPE))
    branch_count = retained.boundary_rank + 1
    x_c = z_rows[:, :n] * half; z_p = z_rows[:, n:]; x_p = z_p * half
    log_g = (adapter.transition_log_density(x_c, x_p)
             + adapter.observation_log_density(x_c, ys[1]) + conversion)
    v = tf.einsum("na,ab->nb", prefix_row_vectors(retained.prefix_cores, retained.prefix_basis, z_p), chol)
    tau_abs = tau * (retained.z_complete_ref / (1.0 + tau))
    sum_sq = tf.reduce_sum(tf.square(v), axis=1) + tau_abs
    log_f = tf.math.log(sum_sq) + log_g
    # v0.3 smooth shift generalized to weighted rows: logsumexp of log(w)+log_f
    shift = tf.reduce_logsumexp(tf.math.log(z_w) + log_f)
    sqrt_g = tf.exp(0.5 * (log_g - shift))
    amps = tf.concat([v, tf.ones([int(z_rows.shape[0]), 1], DTYPE) * tf.sqrt(tau_abs)], axis=1)
    targets = amps * sqrt_g[:, None]
    g_codes = tf.tile(tf.range(branch_count, dtype=DTYPE)[None, :], [int(z_rows.shape[0]), 1])
    full_rows = tf.concat([tf.repeat(z_rows[:, :n], branch_count, axis=0),
                           tf.reshape(g_codes, [-1, 1]),
                           tf.repeat(z_rows[:, n:], branch_count, axis=0)], axis=1)
    sqrt_target = tf.reshape(targets, [-1])
    wts = tf.reshape(tf.repeat(fit_w, branch_count, axis=0), [-1])
    mixed = ProductBasis(list(current_basis.bases) + [DiscreteIndicatorBasis1D(branch_count)]
                         + list(current_basis.bases), current_basis.convention)
    mixed_dims = [basis_dim] * n + [branch_count] + [basis_dim] * n
    c0 = tuple(TTCore(0.3 * tf.random.stateless_normal(
        [1 if a == 0 else rank, mixed_dims[a], 1 if a == 2 * n else rank],
        tf.constant((config.seed, 7031 + a), tf.int32), dtype=DTYPE)) for a in range(2 * n + 1))
    t0 = time.time()
    cores, fdiag = _fixed_als_fit(mixed, full_rows, sqrt_target, wts, c0, config)
    b2 = retained_quadratic_form_from_squared_tt(
        tuple(cores), mixed, split_index=n, tau=0.0,
        prefix_basis=current_basis, coordinate_map=current_map)
    zc_new = (1.0 + tau) * b2.z_complete_ref
    incr = float((shift + tf.math.log(zc_new) - tf.math.log(retained.z_complete_ref)).numpy())
    logg_np = log_g.numpy(); wq = weights_np * np.exp(logg_np - logg_np.max())
    ess = wq.sum() ** 2 / (wq ** 2).sum()
    print(f"{label}: incr={incr:.6f} kalman={kalman_incr1:.6f} err={incr - kalman_incr1:+.4f} "
          f"fit_rms={fdiag['weighted_fit_rms']:.2e} target_ESS={ess:.0f} "
          f"norm_check={weights_np.sum():.4f} wall={time.time()-t0:.0f}s", flush=True)

# Arm A: uniform sobol (engine-identical rows/weights)
zA = _design_rows(config, N, 2 * n, (config.seed, 101)).numpy()
run_t1(zA, np.full(N, 1.0 / N), "uniform-sobol")
# Arm B: proposal mixture, IS-weighted least squares (Section 1 as drafted)
zB = draw_mixture(N)
wB = 1.0 / (N * mixture_mu_density(zB))
run_t1(zB, wB, "proposal-F2-ISLS")
# Arm B2: proposal mixture rows, UNIFORM fit weights = L2(q*mu) fit norm;
# shift quadrature keeps the mu-corrected weights.
run_t1(zB, wB, "proposal-F2-qfit", fit_weights_np=np.full(N, 1.0 / N))

# Representation-hypothesis sweep: if the q-fit rms is a RESOLUTION limit,
# it should collapse with degree (rank held); if it is rank-limited or the
# machinery is broken, degree will not move it.
if "--degree-sweep" in sys.argv:
    for deg_s in (16, 20, 24):
        config = EngineConfig(basis_degree=deg_s, rank=rank, row_count=N, sweeps=3,
            ridge=1e-10, tau=1e-6, coordinate_half_width=hw, seed=91046,
            row_design="sobol")
        current_basis = _product_basis(n, deg_s)
        basis_dim = int(current_basis.bases[0].basis_dim)
        run_t1(zB, wB, f"qfit-deg{deg_s}    ", fit_weights_np=np.full(N, 1.0 / N))

# Alpha sweep: mostly-uniform mixtures — box-wide constraint (off-manifold
# rows pin the polynomial where Z_h integrates it) + minority manifold rows
# (pin the peak the uniform design misses). ISLS weighting throughout.
if "--alpha-sweep" in sys.argv:
    for alpha_s in (0.9, 0.75, 0.5):
        ALPHA = alpha_s
        zS = draw_mixture(N)
        wS = 1.0 / (N * mixture_mu_density(zS))
        run_t1(zS, wS, f"ISLS-alpha{alpha_s:.2f}")

# Affine-preconditioning arm (Zhao-Cui Section 5.1-5.2, linear/Gaussian
# bridging): block-diagonal per-step affine maps x_c = m1f + k*L1f z_c,
# x_p = m_f + k*Lf z_p (block-diagonal preserves the retention split;
# cross-block correlation is left to TT rank per the paper's Fig. 2).
# Rows UNIFORM in the new box; exact Kalman moments used as the
# diagnostic bridging estimate (paper 5.2 uses a particle estimate).
if "--affine" in sys.argv:
    from bayesfilter.highdim.retained_quadratic_form_tf import (
        prefix_row_vectors as _prv,
    )
    from bayesfilter.highdim.retained_moments_tf import retained_reference_moments
    m1f_ = mean1 + (cov1 @ np.linalg.inv(S1)) @ innov1
    P1f_ = cov1 - (cov1 @ np.linalg.inv(S1)) @ cov1
    arms = []
    for kappa in (3.0, 4.0):
        arms.append((f"affine-k{kappa:.0f}", kappa,
                     m1f_, np.linalg.cholesky(P1f_), m_f, np.linalg.cholesky(P_f)))
    if "--m1" in sys.argv:
        # M1 moment source: retained-object moments (design note Section 3),
        # z-coordinates -> physical via the identity-scaled t=0 map (x = hw z).
        mz, Cz = retained_reference_moments(retained)
        m_p_m1 = hw * mz.numpy()
        L_p_m1 = hw * np.linalg.cholesky(Cz.numpy())
        # model-free current block: same moments, declared inflation
        for kappa, kc in ((3.0, 1.5), (3.0, 2.0)):
            arms.append((f"m1-k{kappa:.0f}-infl{kc:.1f}", kappa,
                         m_p_m1, kc * L_p_m1, m_p_m1, L_p_m1))
    for label, kappa, mc_, Lc_base, mp_, Lp_base in arms:
        Lc = Lc_base * kappa
        Lp = Lp_base * kappa
        z_rows_np = _design_rows(config, N, 2 * n, (config.seed, 101)).numpy()
        xc_np = mc_[None, :] + z_rows_np[:, :n] @ Lc.T
        xp_np = mp_[None, :] + z_rows_np[:, n:] @ Lp.T
        z_old_p = xp_np / hw
        clip_frac = float((np.abs(z_old_p) > 1.0).any(axis=1).mean())
        # conversion: log|det Mc| + log|det Mp| + n log 2 - n log hw
        conv = (np.linalg.slogdet(Lc)[1] + np.linalg.slogdet(Lp)[1]
                + n * np.log(2.0) - n * np.log(hw))
        z_rows = tf.constant(z_rows_np, DTYPE)
        z_w = tf.constant(np.full(N, 1.0 / N), DTYPE)
        gram = retained.suffix_gram
        floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
        chol = tf.linalg.cholesky(gram + tf.constant(config.branch_gram_floor, DTYPE)
                                  * floor_scale * tf.eye(tf.shape(gram)[0], dtype=DTYPE))
        branch_count = retained.boundary_rank + 1
        x_c = tf.constant(xc_np, DTYPE); x_p = tf.constant(xp_np, DTYPE)
        log_g = (adapter.transition_log_density(x_c, x_p)
                 + adapter.observation_log_density(x_c, ys[1])
                 + tf.constant(conv, DTYPE))
        z_p_old = tf.constant(np.clip(z_old_p, -1.0, 1.0), DTYPE)
        v = tf.einsum("na,ab->nb", _prv(retained.prefix_cores, retained.prefix_basis, z_p_old), chol)
        tau_abs = tau * (retained.z_complete_ref / (1.0 + tau))
        sum_sq = tf.reduce_sum(tf.square(v), axis=1) + tau_abs
        log_f = tf.math.log(sum_sq) + log_g
        shift = tf.reduce_logsumexp(tf.math.log(z_w) + log_f)
        sqrt_g = tf.exp(0.5 * (log_g - shift))
        amps = tf.concat([v, tf.ones([N, 1], DTYPE) * tf.sqrt(tau_abs)], axis=1)
        targets = amps * sqrt_g[:, None]
        g_codes = tf.tile(tf.range(branch_count, dtype=DTYPE)[None, :], [N, 1])
        full_rows = tf.concat([tf.repeat(z_rows[:, :n], branch_count, axis=0),
                               tf.reshape(g_codes, [-1, 1]),
                               tf.repeat(z_rows[:, n:], branch_count, axis=0)], axis=1)
        sqrt_target = tf.reshape(targets, [-1])
        wts = tf.reshape(tf.repeat(z_w, branch_count, axis=0), [-1])
        current_basis = _product_basis(n, deg)
        basis_dim = int(current_basis.bases[0].basis_dim)
        mixed = ProductBasis(list(current_basis.bases) + [DiscreteIndicatorBasis1D(branch_count)]
                             + list(current_basis.bases), current_basis.convention)
        mixed_dims = [basis_dim] * n + [branch_count] + [basis_dim] * n
        config = EngineConfig(basis_degree=deg, rank=rank, row_count=N, sweeps=3,
            ridge=1e-10, tau=1e-6, coordinate_half_width=hw, seed=91046, row_design="sobol")
        c0 = tuple(TTCore(0.3 * tf.random.stateless_normal(
            [1 if a == 0 else rank, mixed_dims[a], 1 if a == 2 * n else rank],
            tf.constant((config.seed, 7031 + a), tf.int32), dtype=DTYPE)) for a in range(2 * n + 1))
        t0 = time.time()
        cores, fdiag = _fixed_als_fit(mixed, full_rows, sqrt_target, wts, c0, config)
        b2 = retained_quadratic_form_from_squared_tt(
            tuple(cores), mixed, split_index=n, tau=0.0,
            prefix_basis=current_basis, coordinate_map=current_map)
        zc_new = (1.0 + tau) * b2.z_complete_ref
        incr = float((shift + tf.math.log(zc_new) - tf.math.log(retained.z_complete_ref)).numpy())
        logg_np = log_g.numpy(); wq = np.full(N, 1.0 / N) * np.exp(logg_np - logg_np.max())
        ess = wq.sum() ** 2 / (wq ** 2).sum()
        print(f"{label}: incr={incr:.6f} kalman={kalman_incr1:.6f} "
              f"err={incr - kalman_incr1:+.4f} fit_rms={fdiag['weighted_fit_rms']:.2e} "
              f"target_ESS={ess:.0f} clip_frac={clip_frac:.3f} wall={time.time()-t0:.0f}s", flush=True)
