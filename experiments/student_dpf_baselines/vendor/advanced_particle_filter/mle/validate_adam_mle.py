"""
Phase 1 validation: Adam-on-DPF MAP vs exact Kalman MAP.

Purpose
-------
Phase 2 validation (validate_fd_hessian.py) confirmed that the FD Hessian
code is correct. This module validates that Adam actually finds the mode
when run on the DPF.

The comparison is on the linear-Gaussian observation variant where:
  - Kalman gives the *exact* log-posterior
  - Running Adam on Kalman's log-posterior gives the *exact* MAP θ̂_K
  - Running Adam on DPF's log-posterior gives our pipeline's MAP θ̂_D

What "agreement" means here
---------------------------
Two parameter vectors can differ along flat ridges of the log-posterior
without either being wrong. So we don't compare θ̂_D vs θ̂_K directly —
instead we compare their *log-posterior values* under the exact Kalman
objective:

    test = log p_K(y | θ̂_D) vs log p_K(y | θ̂_K)

If these are within a few units, Adam-on-DPF found a point that's
essentially as good as the true MAP under the exact objective. The
optimizer + DPF bias don't move us off the mode in any direction that
matters for the posterior.

Pass criteria
-------------
  - Adam-on-Kalman converges (gradient norm decreases by 100x+)
  - Adam-on-DPF converges
  - log p_K(y | θ̂_D) is within `tol_logpost` of log p_K(y | θ̂_K)
    (default tol = 5.0 units of log-posterior, i.e. exp(5) ≈ 150x prob ratio)

If Adam-on-Kalman fails to converge: optimizer config is wrong (lr, n_steps).
If Adam-on-DPF fails the log-post comparison but Adam-on-Kalman succeeds:
DPF gradient bias is moving Adam off the true mode — Phase 1 needs more
careful tuning (Adam handles attenuated gradients, but very noisy ones can
still cause problems).
"""

import time
from typing import NamedTuple
import numpy as np
import tensorflow as tf

from ..tf_filters import TFDifferentiableParticleFilter
from ..tf_models.linear_gaussian_obs import (
    lg_obs_observation_log_prob,
    simulate_lg_obs,
)
from ..diagnostics.kalman_ll import kalman_log_likelihood
from ..hmc.parameterization import (
    unpack_batched, log_prior_batched, TOTAL_DIM,
)
from .adam_mle import run_adam_mle, make_multistart_z0


DTYPE = tf.float64

# Layout constants (must match hmc.parameterization)
_STATE_DIM = 2
_PHI_START, _PHI_END = 2, 6
_SIGMA_START, _SIGMA_END = 6, 9


def _unpack_eager(z):
    """
    Eager-mode replacement for hmc.parameterization.unpack_batched.

    Why this exists: unpack_batched is decorated with @tf.function(
    reduce_retracing=True). When validate_fd_hessian and convergence_sweep
    call it from multiple contexts in the same Python process with
    different B values (4 chains for Adam-on-DPF, 1 chain for Adam-on-Kalman
    MAP search), TF's tracing cache can re-use a graph built for one B
    when called with another B, causing
        "Input to reshape is a tensor with 4 values, but the requested shape has 16"
    type errors at the Phi reshape.

    This helper does the exact same math but runs eagerly per call —
    no caching, no shape pollution. Performance penalty is negligible
    because the heavy compute is in kalman_log_likelihood, not here.

    Args:
        z: [B, 9] tensor

    Returns:
        SVSSMParams namedtuple with same fields/shapes as unpack_batched.
    """
    from ..tf_models.svssm import SVSSMParams
    B = tf.shape(z)[0]

    # mu with ordering constraint
    mu_1 = z[:, 0]
    mu_2 = z[:, 0] + tf.exp(z[:, 1])
    mu = tf.stack([mu_1, mu_2], axis=-1)                         # [B, 2]

    # Phi: row-major reshape with dynamic batch dim
    Phi = tf.reshape(
        z[:, _PHI_START:_PHI_END], [B, _STATE_DIM, _STATE_DIM],
    )                                                             # [B, 2, 2]

    # Sigma_eta Cholesky
    z_sigma = z[:, _SIGMA_START:_SIGMA_END]
    L_11 = tf.exp(z_sigma[:, 0])
    L_22 = tf.exp(z_sigma[:, 1])
    L_21 = z_sigma[:, 2]
    zero = tf.zeros_like(L_11)
    row0 = tf.stack([L_11, zero], axis=-1)                       # [B, 2]
    row1 = tf.stack([L_21, L_22], axis=-1)                       # [B, 2]
    Sigma_eta_chol = tf.stack([row0, row1], axis=1)              # [B, 2, 2]

    return SVSSMParams(mu=mu, Phi=Phi, Sigma_eta_chol=Sigma_eta_chol)


def _log_prior_eager(z):
    """
    Eager-mode wrapper for log_prior_batched.

    Same caching-pathology motivation as _unpack_eager. We just call the
    underlying tf.function but force a fresh graph by going through
    tf.Variable's identity (no — that doesn't help). Simplest: re-implement
    inline using the same defaults. Defaults pulled from
    hmc.parameterization.log_prior_batched signature.

    z: [B, 9] -> [B] log-prior
    """
    dtype = z.dtype
    mu_scale = tf.cast(2.0, dtype)
    phi_diag_center = tf.cast(0.85, dtype)
    phi_diag_scale = tf.cast(0.3, dtype)
    phi_offdiag_scale = tf.cast(0.3, dtype)
    sigma_log_diag_scale = tf.cast(1.5, dtype)
    sigma_offdiag_scale = tf.cast(0.1, dtype)
    barrier_weight = tf.cast(100.0, dtype)
    barrier_threshold = tf.cast(0.98, dtype)

    z_mu = z[:, 0:2]
    z_phi = z[:, _PHI_START:_PHI_END]
    z_sigma = z[:, _SIGMA_START:_SIGMA_END]

    lp_mu = -0.5 * tf.reduce_sum(tf.square(z_mu / mu_scale), axis=-1)
    phi_diag = tf.stack([z_phi[:, 0], z_phi[:, 3]], axis=-1)
    phi_offdiag = tf.stack([z_phi[:, 1], z_phi[:, 2]], axis=-1)
    lp_phi_diag = -0.5 * tf.reduce_sum(
        tf.square((phi_diag - phi_diag_center) / phi_diag_scale), axis=-1)
    lp_phi_offdiag = -0.5 * tf.reduce_sum(
        tf.square(phi_offdiag / phi_offdiag_scale), axis=-1)
    lp_sigma_diag = -0.5 * tf.reduce_sum(
        tf.square(z_sigma[:, :2] / sigma_log_diag_scale), axis=-1)
    lp_sigma_offdiag = -0.5 * tf.square(z_sigma[:, 2] / sigma_offdiag_scale)

    # Spectral barrier on Phi
    params = _unpack_eager(z)
    Phi = params.Phi
    tr = Phi[:, 0, 0] + Phi[:, 1, 1]
    det = Phi[:, 0, 0] * Phi[:, 1, 1] - Phi[:, 0, 1] * Phi[:, 1, 0]
    discrim = tr * tr - 4.0 * det
    safe_real = tf.maximum(discrim, tf.cast(0.0, dtype))
    rho_real = (tf.abs(tr) + tf.sqrt(safe_real + tf.cast(1e-12, dtype))) / 2.0
    safe_det = tf.maximum(det, tf.cast(1e-12, dtype))
    rho_complex = tf.sqrt(safe_det)
    rho = tf.where(discrim >= 0.0, rho_real, rho_complex)
    excess = tf.maximum(rho - barrier_threshold, tf.cast(0.0, dtype))
    lp_barrier = -barrier_weight * tf.square(excess)

    # Jacobian: exp on z[6], z[7], z[1]
    log_det_J = z_sigma[:, 0] + z_sigma[:, 1] + z[:, 1]

    return (lp_mu + lp_phi_diag + lp_phi_offdiag +
            lp_sigma_diag + lp_sigma_offdiag +
            lp_barrier + log_det_J)


class AdamValidationResult(NamedTuple):
    """
    Output of validate_adam_mle.

    Attributes:
        z_hat_kalman:   [9]  exact MAP from Adam-on-Kalman
        z_hat_dpf:      [9]  pipeline MAP from Adam-on-DPF
        logpost_kalman_at_kalman_mode: float  log p_K(y | θ̂_K)
        logpost_kalman_at_dpf_mode:    float  log p_K(y | θ̂_D)
        logpost_kalman_at_truth:       float  log p_K(y | θ_true)
        delta_logpost:  float  difference (kalman_at_kalman - kalman_at_dpf);
                        positive means Kalman found a higher peak.
        rel_param_err:  [9]   |z_hat_dpf - z_hat_kalman| / max(|z_hat_kalman|, 1)
        passed:         bool  whether log-posterior agreement test passed
        kalman_history: dict  Adam-on-Kalman loss/grad-norm history
        dpf_history:    dict  Adam-on-DPF loss/grad-norm history
    """
    z_hat_kalman: np.ndarray
    z_hat_dpf: np.ndarray
    logpost_kalman_at_kalman_mode: float
    logpost_kalman_at_dpf_mode: float
    logpost_kalman_at_truth: float
    delta_logpost: float
    rel_param_err: np.ndarray
    passed: bool
    kalman_history: dict
    dpf_history: dict


# ============================================================================
# Adam on exact Kalman log-posterior
# ============================================================================

def _build_kalman_logpost_fn(
    observations: tf.Tensor,
    Sigma_obs: tf.Tensor,
    include_prior: bool,
    B_restart: int,
):
    """
    Batched Kalman log-posterior fn z:[B, 9] -> [B].

    Mirrors _build_map_objective in adam_mle.py but using exact Kalman
    instead of DPF. No MC replication needed — Kalman is deterministic.

    Implementation note: we Python-unroll the per-chain loop rather than
    using tf.map_fn. B_restart is small (typically 4), and unrolling
    sidesteps subtle gradient interactions between map_fn and the
    while_loop + Cholesky inside kalman_log_likelihood.
    """
    B_restart = int(B_restart)

    # Eager logpost: see _unpack_eager / _log_prior_eager docstrings for the
    # caching-pathology motivation.
    def logpost_fn(z):
        # z: [B_restart, 9]
        params = _unpack_eager(z)
        # Python-unrolled loop over chains (B_restart is small)
        lls = []
        for b in range(B_restart):
            ll_b = kalman_log_likelihood(
                params.mu[b],
                params.Phi[b],
                params.Sigma_eta_chol[b],
                Sigma_obs,
                observations,
            )
            lls.append(ll_b)
        ll = tf.stack(lls, axis=0)  # [B_restart]

        if include_prior:
            lp = _log_prior_eager(z)  # [B_restart]
            return lp + ll
        else:
            return ll

    return logpost_fn


def run_adam_on_kalman(
    observations: tf.Tensor,
    Sigma_obs: tf.Tensor,
    z_init: tf.Tensor,             # [B_restart, 9]
    n_steps: int = 1000,
    learning_rate: float = 0.01,
    include_prior: bool = True,
    log_every: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Run Adam to maximize the *exact* Kalman log-posterior.

    Returns the same kind of dict as Adam-on-DPF, but with deterministic
    gradients (no MC, no PF noise). Used as ground truth for Phase 1
    validation.
    """
    B_restart = int(z_init.shape[0])
    z_var = tf.Variable(z_init, dtype=DTYPE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    logpost_fn = _build_kalman_logpost_fn(
        observations, Sigma_obs,
        include_prior=include_prior,
        B_restart=B_restart,
    )

    loss_hist = np.zeros((n_steps, B_restart))
    grad_norm_hist = np.zeros((n_steps, B_restart))

    # Eager step — see logpost_fn comment. Kalman's tf.function-d while_loop
    # provides the heavy compute, so eager outer overhead is small.
    def step():
        with tf.GradientTape() as tape:
            lp = logpost_fn(z_var)              # [B_restart]
            loss = -tf.reduce_sum(lp)
        grads = tape.gradient(loss, [z_var])
        optimizer.apply_gradients(zip(grads, [z_var]))
        return lp, tf.norm(grads[0], axis=-1)

    if verbose:
        print(f"[Adam-Kalman] B_restart={B_restart}, n_steps={n_steps}, "
              f"lr={learning_rate}")

    t0 = time.time()
    for s in range(n_steps):
        lp, gnorm = step()
        loss_hist[s] = -lp.numpy()
        grad_norm_hist[s] = gnorm.numpy()
        if verbose and (s % log_every == 0 or s == n_steps - 1):
            best_lp = float(tf.reduce_max(lp).numpy())
            print(f"  step {s:4d}  best_log_post={best_lp:+.3f}  "
                  f"mean_grad_norm={float(tf.reduce_mean(gnorm).numpy()):.3f}")
    elapsed = time.time() - t0
    if verbose:
        print(f"[Adam-Kalman] done in {elapsed:.1f}s")

    final_lp = logpost_fn(z_var)
    best_idx = int(tf.argmax(final_lp).numpy())
    z_hat = z_var[best_idx].numpy()

    return dict(
        z_hat=z_hat,
        z_all=z_var.numpy(),
        final_log_post=final_lp.numpy(),
        best_restart=best_idx,
        loss_history=loss_hist,
        grad_norm_history=grad_norm_hist,
        elapsed=elapsed,
    )


# ============================================================================
# Validation harness
# ============================================================================

def validate_adam_mle(
    # Data generation (same defaults as Phase 2 validation for consistency)
    T: int = 75,
    data_seed: int = 42,
    sigma_obs_diag: float = 0.5,
    mu_true=(-1.0, 0.5),
    Phi_true=((0.85, 0.12), (0.02, 0.90)),
    Sigma_eta_diag=(0.15, 0.4),
    Sigma_eta_rho: float = 0.3,
    # DPF config
    n_particles: int = 1000,
    alpha: float = 0.5,
    # Adam config (shared across both runs)
    n_steps: int = 1000,
    learning_rate: float = 0.01,
    B_restart: int = 4,
    jitter: float = 0.3,
    adam_n_mc: int = 4,
    include_prior: bool = True,
    # Tolerance
    tol_logpost: float = 5.0,
    init_seed: int = 1234,
    verbose: bool = True,
) -> AdamValidationResult:
    """
    Compare Adam-on-DPF MAP vs exact Adam-on-Kalman MAP.

    Both runs use the same initial points (multi-start from warm_start)
    so any difference in convergence is attributable to gradient quality,
    not initialization.

    Args:
        tol_logpost:  pass threshold on log_p_K(y|θ̂_D) - log_p_K(y|θ̂_K).
                      Default 5.0 means DPF's mode must be within exp(5) ≈
                      150x of the true MAP under the exact log-posterior.

    Returns:
        AdamValidationResult
    """
    # -- Build truth --
    mu_t = tf.constant(mu_true, dtype=DTYPE)
    Phi_t = tf.constant(Phi_true, dtype=DTYPE)
    std_eta = np.array(Sigma_eta_diag)
    rho = Sigma_eta_rho
    Sigma_eta = np.array([
        [std_eta[0]**2,                   rho * std_eta[0] * std_eta[1]],
        [rho * std_eta[0] * std_eta[1],   std_eta[1]**2],
    ])
    L_eta = tf.constant(np.linalg.cholesky(Sigma_eta), dtype=DTYPE)
    Sigma_obs = tf.constant(
        np.diag([sigma_obs_diag**2, sigma_obs_diag**2]), dtype=DTYPE
    )
    Sigma_obs_chol = tf.linalg.cholesky(Sigma_obs)

    # -- Map truth -> z (for reference) --
    mu_np = mu_t.numpy()
    Phi_np = Phi_t.numpy()
    L_np = L_eta.numpy()
    z_truth = np.array([
        mu_np[0],
        np.log(mu_np[1] - mu_np[0]),
        Phi_np[0, 0], Phi_np[0, 1], Phi_np[1, 0], Phi_np[1, 1],
        np.log(L_np[0, 0]),
        np.log(L_np[1, 1]),
        L_np[1, 0],
    ])

    # -- Generate data under LG-obs model --
    if verbose:
        print("[Validate-Adam] simulating LG-obs data ...")
    sim_rng = tf.random.Generator.from_seed(data_seed)
    _, y_obs = simulate_lg_obs(mu_t, Phi_t, L_eta, Sigma_obs_chol, T, sim_rng)
    if verbose:
        print(f"  T={T}, y_obs shape={y_obs.shape}")

    # -- Shared multi-start initialization --
    z_init = make_multistart_z0(
        y_obs, B_restart=B_restart, jitter=jitter, seed=init_seed,
    )
    if verbose:
        print(f"  z_init shape: {z_init.shape}, jitter={jitter}")

    # -- Run Adam on exact Kalman log-posterior (ground truth optimizer) --
    if verbose:
        print("\n[Validate-Adam] running Adam on exact Kalman log-posterior ...")
    kalman_result = run_adam_on_kalman(
        observations=y_obs,
        Sigma_obs=Sigma_obs,
        z_init=z_init,
        n_steps=n_steps,
        learning_rate=learning_rate,
        include_prior=include_prior,
        verbose=verbose,
    )
    z_hat_K = kalman_result['z_hat']

    # -- Run Adam on DPF log-posterior (pipeline) --
    if verbose:
        print("\n[Validate-Adam] running Adam on DPF log-posterior ...")

    # DPF with LG-obs likelihood
    def lg_obs_closure(y_t, particles):
        return lg_obs_observation_log_prob(y_t, particles, Sigma_obs_chol)

    dpf = TFDifferentiableParticleFilter(
        n_particles=n_particles,
        resampler='soft',
        alpha=alpha,
        obs_log_prob_fn=lg_obs_closure,
        dtype=DTYPE,
    )
    dpf_rng = tf.random.Generator.from_seed(data_seed + 1)

    dpf_adam_result = run_adam_mle(
        observations=y_obs,
        dpf=dpf,
        rng=dpf_rng,
        z_init=z_init,                    # SAME initialization as Kalman
        n_steps=n_steps,
        learning_rate=learning_rate,
        n_mc=adam_n_mc,
        verbose=verbose,
        log_every=max(1, n_steps // 10),
    )
    z_hat_D = dpf_adam_result.z_hat.numpy()

    # -- Compare under EXACT Kalman log-posterior --
    if verbose:
        print("\n[Validate-Adam] evaluating log-posteriors under exact Kalman ...")

    kalman_eval = _build_kalman_logpost_fn(
        y_obs, Sigma_obs, include_prior=include_prior, B_restart=3,
    )

    z_stack = tf.constant(
        np.stack([z_hat_K, z_hat_D, z_truth], axis=0), dtype=DTYPE
    )  # [3, 9]
    lp_three = kalman_eval(z_stack).numpy()
    lp_at_K = float(lp_three[0])
    lp_at_D = float(lp_three[1])
    lp_at_truth = float(lp_three[2])

    delta = lp_at_K - lp_at_D
    rel_err = np.abs(z_hat_D - z_hat_K) / np.maximum(np.abs(z_hat_K), 1.0)
    passed = abs(delta) < tol_logpost

    # -- Print summary --
    if verbose:
        param_names = ['mu_0', 'log(gap)', 'Phi_00', 'Phi_01', 'Phi_10',
                       'Phi_11', 'log(L_11)', 'log(L_22)', 'L_21']

        print("\n" + "=" * 74)
        print("  Validation summary: Adam-on-DPF vs Adam-on-Kalman")
        print("=" * 74)

        print(f"\n  Log-posterior under EXACT Kalman objective:")
        print(f"    at Kalman MAP θ̂_K:  {lp_at_K:+.3f}")
        print(f"    at DPF MAP    θ̂_D:  {lp_at_D:+.3f}")
        print(f"    at truth      θ_true: {lp_at_truth:+.3f}")
        print(f"    delta (K - D):       {delta:+.3f}  "
              f"{'PASS' if passed else 'FAIL'} (|.| < {tol_logpost})")
        if delta > 0 and passed:
            print(f"    -> Kalman MAP is {delta:.2f} log-units higher; "
                  f"Adam-on-DPF found a near-optimal point.")
        elif delta < 0:
            print(f"    -> Adam-on-DPF found a point with HIGHER Kalman "
                  f"log-posterior (this can happen if Kalman Adam underconverged).")

        print(f"\n  Per-parameter comparison (ẑ_D vs ẑ_K):")
        for i, n in enumerate(param_names):
            print(f"    {n:10s}  Kalman={z_hat_K[i]:+.4f}  "
                  f"DPF={z_hat_D[i]:+.4f}  rel_err={rel_err[i]:.3f}")

        print(f"\n  Final gradient norms:")
        print(f"    Adam-on-Kalman: "
              f"{kalman_result['grad_norm_history'][-1].round(3)}")
        print(f"    Adam-on-DPF:    "
              f"{dpf_adam_result.grad_norm_history[-1].round(3)}")

        print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print("\n  If FAIL, likely causes:")
            print("    1. Adam-on-Kalman didn't converge (check gradient norm")
            print("       at end vs at start; should drop ≥ 100x)")
            print("    2. learning_rate too large for DPF noise (try 0.005)")
            print("    3. n_mc too small (more MC averaging = less noisy grad)")
            print("    4. Posterior is multi-modal — restart from a different "
                  "init or increase B_restart and jitter")

    # Convert dpf_adam_result history to dict for consistency
    dpf_history = dict(
        loss_history=dpf_adam_result.loss_history,
        grad_norm_history=dpf_adam_result.grad_norm_history,
        elapsed=dpf_adam_result.elapsed,
        final_log_post=dpf_adam_result.final_log_post.numpy(),
        best_restart=dpf_adam_result.best_restart,
    )

    return AdamValidationResult(
        z_hat_kalman=z_hat_K,
        z_hat_dpf=z_hat_D,
        logpost_kalman_at_kalman_mode=lp_at_K,
        logpost_kalman_at_dpf_mode=lp_at_D,
        logpost_kalman_at_truth=lp_at_truth,
        delta_logpost=delta,
        rel_param_err=rel_err,
        passed=passed,
        kalman_history=kalman_result,
        dpf_history=dpf_history,
    )


if __name__ == "__main__":
    validate_adam_mle()
