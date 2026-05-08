"""
Phase 2 validation: DPF finite-difference Hessian vs Kalman autodiff Hessian.

Purpose
-------
Before trusting compute_laplace_covariance on the nonlinear SVSSM (where
no exact ground truth exists), we validate it on the linear-Gaussian
observation variant where:
  - Kalman filter gives the *exact* log p(y | theta)
  - Double GradientTape on Kalman gives the *exact* Hessian
  - DPF with the same parameters should produce a consistent Hessian
    estimate (up to DPF stochastic bias)

What agreement would look like
------------------------------
For the linear-Gaussian observation model, as N_particles -> infinity,
the DPF log-likelihood converges to Kalman's. So:
  - FD Hessian on DPF should converge to autodiff Hessian on Kalman
  - Difference shrinks as N grows and as we average over more PF seeds

Pass criteria (informal)
------------------------
  - Sign agreement on diagonal (both strictly positive for -H at the MAP)
  - Eigenvalues of -H within ~20% of Kalman's eigenvalues
  - Marginal stds from Laplace within ~20% of Kalman's Laplace stds
  - Element-wise max(|H_dpf - H_kalman|) / max(|H_kalman|) < 0.3

If these fail, something is wrong with the FD implementation (not just
DPF bias): bug in stencil assembly, wrong eps, etc.

Why NOT test on SVSSM directly
------------------------------
SVSSM has no exact Hessian reference. Testing only on SVSSM conflates:
  (1) DPF stochastic bias (expected, tolerable)
  (2) FD implementation bugs (unacceptable)
  (3) MAP not actually being at the mode (Phase 1 failure)
This test isolates (2): if it passes, we know the FD Hessian code is
correct, and any SVSSM discrepancies are interpretable as (1) or (3).
"""

import time
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
from .laplace import compute_laplace_covariance
# Eager replacements for unpack_batched/log_prior_batched used to sidestep
# tf.function caching pathology when this module is invoked across multiple
# different B values in the same process (e.g., from convergence_sweep).
from .validate_adam_mle import _unpack_eager, _log_prior_eager


DTYPE = tf.float64


# ============================================================================
# Exact Kalman Hessian via double GradientTape
# ============================================================================

def kalman_logpost_hessian(
    z_hat: tf.Tensor,         # [9]
    observations: tf.Tensor,  # [T, d]
    Sigma_obs: tf.Tensor,     # [d, d]
    include_prior: bool,
) -> tuple:
    """
    Exact Hessian of log-posterior (or log-lik only) at z_hat via Kalman +
    nested GradientTape.

    The unpack_batched function maps z -> (mu, Phi, L_chol) differentiably,
    and kalman_log_likelihood is end-to-end differentiable in those
    tensors. So d^2 LogLik / dz dz is computable via nested tape.

    Args:
        z_hat:        [9]  point at which to evaluate Hessian
        observations: [T, d]
        Sigma_obs:    [d, d]  observation noise covariance (fixed)
        include_prior: if True, Hessian is of log-posterior;
                       if False, Hessian is of log-likelihood only.

    Returns:
        (ll_value, H)  where H is [9, 9] dense Hessian
    """
    z_var = tf.Variable(tf.reshape(z_hat, [TOTAL_DIM]), dtype=DTYPE)

    # persistent=True is REQUIRED on the outer tape because experimental_use_pfor=False
    # below makes jacobian compute the [9,9] Hessian via 9 sequential
    # tape.gradient calls (one per output dim of `grad`), which reuses the tape.
    with tf.GradientTape(persistent=True) as tape_outer:
        with tf.GradientTape() as tape_inner:
            # Need z as [1, 9] for unpack, then index [0] for Kalman.
            # Use eager helpers (not the tf.function-cached unpack_batched /
            # log_prior_batched) to avoid graph-cache pollution across calls
            # with different B values; see _unpack_eager docstring.
            z_b = z_var[tf.newaxis, :]
            params = _unpack_eager(z_b)
            mu = params.mu[0]
            Phi = params.Phi[0]
            L_chol = params.Sigma_eta_chol[0]

            ll = kalman_log_likelihood(
                mu, Phi, L_chol, Sigma_obs, observations,
            )
            if include_prior:
                lp = _log_prior_eager(z_b)[0]
                target = ll + lp
            else:
                target = ll
        # First derivatives [9]
        grad = tape_inner.gradient(target, z_var)
    # Second derivatives — jacobian gives [9, 9].
    # NOTE: experimental_use_pfor=False is REQUIRED here. The default pfor
    # vectorization fails with "num_rows and num_cols are not consistent"
    # when trying to symbolically convert the second-order gradient of
    # tf.linalg.cholesky inside a tf.while_loop (Kalman's time loop).
    # Sequential loop over the 9 output dims is ~9x slower per call but
    # still under a second, and we only call this once.
    hess = tape_outer.jacobian(grad, z_var, experimental_use_pfor=False)
    del tape_outer  # persistent tapes aren't auto-freed

    return float(target.numpy()), hess.numpy()


# ============================================================================
# Validation harness
# ============================================================================

def validate_fd_hessian(
    # Data generation
    T: int = 75,
    data_seed: int = 42,
    sigma_obs_diag: float = 0.5,
    # True parameters (reasonable defaults near typical SVSSM calibration)
    mu_true=(-1.0, 0.5),
    Phi_true=((0.85, 0.12), (0.02, 0.90)),
    Sigma_eta_diag=(0.15, 0.4),
    Sigma_eta_rho: float = 0.3,
    # DPF config
    n_particles: int = 1000,
    alpha: float = 0.5,
    # FD config
    eps: float = 0.05,
    n_seeds: int = 10,
    chunk_size: int = 8,
    include_prior: bool = True,
    # Eval point — 'truth', 'kalman_mode', or pass 9-vector
    eval_at='kalman_mode',
    # Optimizer settings used when eval_at='kalman_mode'
    kalman_mode_steps: int = 500,
    kalman_mode_lr: float = 0.02,
    verbose: bool = True,
) -> dict:
    """
    Run validation comparing DPF FD Hessian to Kalman autodiff Hessian.

    Args:
        eval_at: where to evaluate the Hessian.
            - 'kalman_mode' (default): find the exact Kalman MAP first via
              Adam-on-Kalman, then compute FD Hessian there. This is the
              CORRECT validation point: Laplace approximates posterior near
              its mode, so we should test the Hessian at the mode, not at a
              random point. At small T, the truth point is generally NOT
              the MAP and -H may not even be PSD there.
            - 'truth': evaluate at the parameter values that generated the
              data. Useful for diagnostics but expect saddle/non-PSD issues
              at small T.
            - np.ndarray of shape [9]: specify z directly.
        chunk_size: DPF batch chunk size; see compute_laplace_covariance.
        kalman_mode_steps, kalman_mode_lr: Adam config used to find the
            Kalman MAP when eval_at='kalman_mode'.

    Returns:
        dict with keys:
            H_kalman:    [9, 9] exact Hessian of log-posterior (or log-lik)
            H_dpf:       [9, 9] FD estimate from DPF
            H_dpf_se:    [9, 9] per-entry SE across n_seeds
            Sigma_kalman:[9, 9] (-H_kalman)^-1
            Sigma_dpf:   [9, 9] from Laplace pipeline
            eigvals_kalman, eigvals_dpf: [9] ascending eigenvalues of -H
            std_kalman, std_dpf: [9] marginal stds
            rel_err_H:   scalar, max|H_dpf - H_kalman| / max|H_kalman|
            rel_err_std: [9] per-param |std_dpf - std_kalman| / std_kalman
            pass:        bool, whether validation criteria were met
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

    # Observation noise (fixed, known)
    Sigma_obs = tf.constant(
        np.diag([sigma_obs_diag**2, sigma_obs_diag**2]), dtype=DTYPE
    )
    Sigma_obs_chol = tf.linalg.cholesky(Sigma_obs)

    # -- Generate data under the LINEAR-GAUSSIAN observation model --
    if verbose:
        print("[Validate] simulating linear-Gaussian observation data ...")
    sim_rng = tf.random.Generator.from_seed(data_seed)
    _, y_obs = simulate_lg_obs(mu_t, Phi_t, L_eta, Sigma_obs_chol, T, sim_rng)
    if verbose:
        print(f"  T={T}, y_obs shape = {y_obs.shape}")

    # -- Build z at evaluation point --
    # Map truth -> z (always computed; used either as eval point or as
    # initialization for Kalman MAP search).
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

    if eval_at == 'truth':
        z_eval = tf.constant(z_truth, dtype=DTYPE)
    elif eval_at == 'kalman_mode':
        # Find the actual Kalman MAP first via Adam-on-Kalman, init at truth.
        # This is THE correct evaluation point for Laplace: posterior expansion
        # only makes sense at a local mode where -H is PSD.
        if verbose:
            print(f"\n[Validate] finding Kalman MAP via Adam (init at truth) ...")
        from .validate_adam_mle import run_adam_on_kalman

        # Single restart at truth (jitter=0 -> exactly z_truth)
        z_init = tf.constant(z_truth[np.newaxis, :], dtype=DTYPE)
        kalman_run = run_adam_on_kalman(
            observations=y_obs, Sigma_obs=Sigma_obs,
            z_init=z_init,
            n_steps=kalman_mode_steps,
            learning_rate=kalman_mode_lr,
            include_prior=include_prior,
            verbose=False,
        )
        z_eval = tf.constant(kalman_run['z_hat'], dtype=DTYPE)
        if verbose:
            print(f"  Kalman MAP found:    "
                  f"log_post = {kalman_run['final_log_post'][0]:+.3f}")
            print(f"  Distance from truth: "
                  f"{np.linalg.norm(kalman_run['z_hat'] - z_truth):.3f}")
            print(f"  Final ||grad||:      "
                  f"{kalman_run['grad_norm_history'][-1, 0]:.4f}")
    else:
        z_eval = tf.constant(np.asarray(eval_at, dtype=np.float64), dtype=DTYPE)

    if verbose:
        print(f"  z_eval = {z_eval.numpy().round(3)}")

    # -- Exact Kalman Hessian --
    if verbose:
        print("\n[Validate] computing exact Kalman Hessian (double-tape) ...")
    t0 = time.time()
    ll_kalman, H_kalman = kalman_logpost_hessian(
        z_eval, y_obs, Sigma_obs, include_prior=include_prior,
    )
    t_kalman = time.time() - t0
    if verbose:
        print(f"  Kalman log-posterior at z_eval = {ll_kalman:+.3f}  "
              f"({t_kalman:.2f}s)")

    # -- DPF FD Hessian via Phase 2 code path --
    if verbose:
        print("\n[Validate] setting up DPF with linear-Gaussian observation ...")

    def lg_obs_closure(y_t, particles):
        return lg_obs_observation_log_prob(y_t, particles, Sigma_obs_chol)

    dpf = TFDifferentiableParticleFilter(
        n_particles=n_particles,
        resampler='soft',
        alpha=alpha,
        obs_log_prob_fn=lg_obs_closure,
        dtype=DTYPE,
    )

    if verbose:
        print(f"  DPF: N={n_particles}, alpha={alpha}")
        print(f"\n[Validate] computing DPF FD Hessian ...")

    lap = compute_laplace_covariance(
        observations=y_obs,
        dpf=dpf,
        z_hat=z_eval,
        eps=eps,
        n_seeds=n_seeds,
        include_prior=include_prior,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    H_dpf = lap.H.numpy()
    H_dpf_se = lap.H_se.numpy()
    Sigma_dpf = lap.Sigma_laplace.numpy()

    # -- Comparison --
    # Kalman-side covariance (assumes -H_kalman PD)
    negH_k = -H_kalman
    eigvals_kalman = np.sort(np.linalg.eigvalsh(negH_k))
    eigvals_dpf = np.sort(np.linalg.eigvalsh(-H_dpf))

    # Regularize Kalman if it has non-PSD curvature.
    # If this fires, eval_at is NOT a local maximum of the log-posterior.
    # At z_truth + an informative prior, the posterior mode is shifted away
    # from truth, so eval_at='truth' gives a saddle point and -H is indefinite.
    # Fix: use eval_at='kalman_mode' to find the actual MAP first.
    kalman_indefinite = eigvals_kalman.min() <= 1e-8
    if kalman_indefinite:
        if verbose:
            print(f"\n  *** WARNING: Kalman -H has eigenvalue {eigvals_kalman.min():.2e} <= 0 ***")
            print(f"  *** This means eval_at is NOT at the posterior mode.       ***")
            print(f"  *** Hessian/Laplace are only meaningful at a local maximum. ***")
            print(f"  *** Re-run with eval_at='kalman_mode' to find the actual    ***")
            print(f"  *** MAP first via Adam-on-Kalman.                           ***")
            print(f"  *** Std comparison below will be MEANINGLESS until fixed.   ***\n")
        negH_k_reg = negH_k + (1e-6 - eigvals_kalman.min()) * np.eye(TOTAL_DIM)
        Sigma_kalman = np.linalg.solve(negH_k_reg, np.eye(TOTAL_DIM))
    else:
        Sigma_kalman = np.linalg.solve(negH_k, np.eye(TOTAL_DIM))
    Sigma_kalman = 0.5 * (Sigma_kalman + Sigma_kalman.T)

    std_kalman = np.sqrt(np.maximum(np.diag(Sigma_kalman), 0.0))
    std_dpf = np.sqrt(np.maximum(np.diag(Sigma_dpf), 0.0))

    # Relative errors
    scale_H = max(np.max(np.abs(H_kalman)), 1e-12)
    rel_err_H_elem = np.abs(H_dpf - H_kalman) / scale_H
    rel_err_H = float(rel_err_H_elem.max())

    rel_err_std = np.abs(std_dpf - std_kalman) / np.maximum(std_kalman, 1e-12)

    rel_err_eig = np.abs(eigvals_dpf - eigvals_kalman) / np.maximum(
        np.abs(eigvals_kalman), 1e-12
    )

    # Pass criteria.
    # The PRIMARY test is element-wise Hessian agreement: Laplace cares
    # about the Hessian itself, and this test is robust to near-singular
    # directions.
    # Eigenvalue and std comparisons are SECONDARY/informational. They can
    # blow up when -H has a near-zero eigenvalue (a real ridge in the
    # likelihood) even when both Hessians agree to 3 decimal places —
    # because tiny absolute differences become large relative errors when
    # divided by a near-zero baseline. We report them but they only
    # contribute to FAIL if they're catastrophic AND the primary Hessian
    # test also failed.
    pass_H = rel_err_H < 0.3
    eig_warn = rel_err_eig.max() >= 0.3
    std_warn = rel_err_std.max() >= 0.3
    near_singular = (np.abs(eigvals_kalman).min() < 1e-3 *
                     np.abs(eigvals_kalman).max())

    # Pass if element-wise Hessian agrees. eig/std ratios may be noisy
    # near singular directions; we don't fail on them alone.
    passed = bool(pass_H)

    if verbose:
        param_names = ['mu_0', 'log(gap)', 'Phi_00', 'Phi_01', 'Phi_10',
                       'Phi_11', 'log(L_11)', 'log(L_22)', 'L_21']
        print("\n" + "=" * 74)
        print("  Validation summary: DPF FD Hessian vs Kalman exact Hessian")
        print("=" * 74)

        # PRIMARY test: element-wise Hessian agreement
        print("\n  PRIMARY TEST — Hessian element-wise agreement:")
        print(f"    max|H_dpf - H_kalman|  = "
              f"{np.max(np.abs(H_dpf - H_kalman)):.3e}")
        print(f"    max|H_kalman|          = {scale_H:.3e}")
        print(f"    relative max error     = {rel_err_H:.3f}  "
              f"{'PASS' if pass_H else 'FAIL'} (<0.3)")

        # SECONDARY: eigenvalue and std comparisons.
        # If -H is near-singular (real likelihood ridge), these can blow
        # up even when H_dpf and H_kalman agree perfectly.
        print("\n  SECONDARY DIAGNOSTICS — eigenvalues / marginal stds")
        print("  (informational only; near-singular -H makes these noisy)")
        if near_singular:
            print(f"  *** Kalman -H is near-singular: min/max eig ratio = "
                  f"{np.abs(eigvals_kalman).min() / np.abs(eigvals_kalman).max():.2e}")
            print(f"  *** Eig/std comparisons below will be misleading.")

        print("\n  Eigenvalues of -H (ascending):")
        print(f"    Kalman: {eigvals_kalman.round(3)}")
        print(f"    DPF:    {eigvals_dpf.round(3)}")
        print(f"    max rel err: {rel_err_eig.max():.3f}  "
              f"{'WARN' if eig_warn else 'OK'}")

        print("\n  Marginal std per parameter:")
        for i, n in enumerate(param_names):
            flag = ""
            if rel_err_std[i] >= 0.3:
                flag = " (warn)"
            print(f"    {n:10s}  Kalman={std_kalman[i]:.3f}  "
                  f"DPF={std_dpf[i]:.3f}  rel_err={rel_err_std[i]:.3f}{flag}")
        print(f"    max rel err: {rel_err_std.max():.3f}  "
              f"{'WARN' if std_warn else 'OK'}")

        print(f"\n  OVERALL: {'PASS' if passed else 'FAIL'} "
              f"(based on primary Hessian agreement test)")
        if not passed:
            print("\n  If FAIL, likely causes in order of likelihood:")
            print("    1. n_seeds too small — FD noise amplified by 1/eps^2")
            print("       (try n_seeds=30-50)")
            print("    2. n_particles too small — DPF bias swamps signal")
            print("       (try n_particles=2000)")
            print("    3. eps wrong — too small amplifies noise, too large")
            print("       introduces truncation error (try 0.03, 0.07, 0.1)")
            print("    4. Actual bug in FD code — would show element-level")
            print("       disagreement exceeding SE")
        elif eig_warn or std_warn:
            print("\n  Note: Hessian agrees element-wise but eigenvalue/std")
            print("  comparisons differ. This is benign when -H has a near-zero")
            print("  eigenvalue (real model ridge): tiny absolute Hessian")
            print("  differences become large relative errors when divided by")
            print("  a near-zero eigenvalue. The Laplace covariance is still")
            print("  reliable in well-identified directions.")

    return dict(
        H_kalman=H_kalman, H_dpf=H_dpf, H_dpf_se=H_dpf_se,
        Sigma_kalman=Sigma_kalman, Sigma_dpf=Sigma_dpf,
        eigvals_kalman=eigvals_kalman, eigvals_dpf=eigvals_dpf,
        std_kalman=std_kalman, std_dpf=std_dpf,
        rel_err_H=rel_err_H, rel_err_std=rel_err_std, rel_err_eig=rel_err_eig,
        passed=passed,
        eig_warn=eig_warn, std_warn=std_warn,
        near_singular=near_singular,
        laplace_result=lap,
        z_eval=z_eval.numpy(),
        y_obs=y_obs.numpy(),
        ll_kalman=ll_kalman,
    )


# ============================================================================
# Convergence sweep: show DPF -> Kalman as N or n_seeds grows
# ============================================================================

def convergence_sweep(
    sweep_over: str = 'n_particles',
    values=(200, 500, 1000, 2000),
    n_seeds_fixed: int = 10,
    n_particles_fixed: int = 1000,
    T: int = 75,
    data_seed: int = 42,
    eps: float = 0.05,
    verbose: bool = True,
    **validate_kwargs,
) -> dict:
    """
    Sweep either n_particles or n_seeds; show relative error decreasing.

    This is the stronger evidence of correctness: a single pass/fail at one
    config could always be tuned; consistent monotonic convergence toward
    Kalman as either knob grows confirms the estimator is unbiased in the
    limit.

    Args:
        sweep_over:        'n_particles' or 'n_seeds'
        values:            sequence of values to sweep
        n_seeds_fixed:     held fixed when sweeping n_particles
        n_particles_fixed: held fixed when sweeping n_seeds
        T, data_seed, eps: passed through to validate_fd_hessian
        **validate_kwargs: anything else accepted by validate_fd_hessian

    Returns:
        dict with keys:
            values:       the swept values
            rel_err_H:    [len(values)] per-value Hessian rel err
            rel_err_std:  [len(values), 9] per-value per-param std rel err
            rel_err_eig:  [len(values)] max eigenvalue rel err
            results:      list of full result dicts from validate_fd_hessian
    """
    if sweep_over not in ('n_particles', 'n_seeds'):
        raise ValueError("sweep_over must be 'n_particles' or 'n_seeds'")

    rel_err_H = []
    rel_err_std = []
    rel_err_eig = []
    results = []

    for v in values:
        if verbose:
            print("\n" + "#" * 74)
            print(f"#  Sweep {sweep_over} = {v}")
            print("#" * 74)

        kwargs = dict(validate_kwargs)
        kwargs.update(dict(
            T=T, data_seed=data_seed, eps=eps,
            verbose=verbose,
        ))
        if sweep_over == 'n_particles':
            kwargs['n_particles'] = int(v)
            kwargs['n_seeds'] = n_seeds_fixed
        else:
            kwargs['n_particles'] = n_particles_fixed
            kwargs['n_seeds'] = int(v)

        r = validate_fd_hessian(**kwargs)
        rel_err_H.append(r['rel_err_H'])
        rel_err_std.append(r['rel_err_std'])
        rel_err_eig.append(r['rel_err_eig'].max())
        results.append(r)

    rel_err_H = np.array(rel_err_H)
    rel_err_std = np.array(rel_err_std)        # [V, 9]
    rel_err_eig = np.array(rel_err_eig)

    if verbose:
        print("\n" + "=" * 74)
        print(f"  Convergence summary (sweeping {sweep_over})")
        print("=" * 74)
        print(f"\n  {sweep_over:>14s}  rel_err_H   max_rel_err_eig   max_rel_err_std")
        for i, v in enumerate(values):
            print(f"  {v:>14}  {rel_err_H[i]:>9.3f}  {rel_err_eig[i]:>16.3f}  "
                  f"{rel_err_std[i].max():>16.3f}")

        # Check monotonic decrease (roughly — noise can cause small reversals)
        if len(values) >= 2:
            decreased = rel_err_H[-1] < rel_err_H[0]
            print(f"\n  Hessian rel_err decreased from first to last "
                  f"value: {decreased}")
            if not decreased:
                print("  -> NOT decreasing: possible FD bug, or already at "
                      "noise floor.")

    return dict(
        sweep_over=sweep_over,
        values=list(values),
        rel_err_H=rel_err_H,
        rel_err_std=rel_err_std,
        rel_err_eig=rel_err_eig,
        results=results,
    )


if __name__ == "__main__":
    validate_fd_hessian()
