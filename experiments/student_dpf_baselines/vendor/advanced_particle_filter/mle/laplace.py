"""
Phase 2: Laplace covariance at the MAP mode.

Theoretical rationale
---------------------
The Laplace approximation sets

    posterior ≈ N(θ_hat, Σ_Laplace),    Σ_Laplace = (-H)^(-1)

where H is the Hessian of the log-posterior at θ_hat. This uses only
log-posterior *values*, not gradients — and DPF log-likelihood values
are accurate even when its gradients are attenuated (A1/A2 diagnostics
in the project confirmed this).

Batching strategy
-----------------
Naive implementation: for each of 45 unique Hessian entries (for d=9),
4 stencil evaluations × n_seeds averaging = lots of serial DPF calls.

We batch along the B dimension: at each Hessian entry, we pack all 4
stencil points and all n_seeds into a single DPF call at B = 4*n_seeds.
This gives 45 DPF calls total for the Hessian, each with B=4*n_seeds.

Independent-seeds averaging: each of the n_seeds slots gets a fresh
rng.split() key, so the PF noise realizations are independent across
seeds. This averages over the DPF stochastic bias at the cost of being
noisier per-stencil than CRN (common random numbers) would be, but
gives an honest SE estimate.

Numerical concern
-----------------
FD second derivatives amplify per-evaluation noise by 1/eps². With
eps=0.05, that's 400x. At N=1000 particles the per-eval log-lik noise
std is a few tenths; squaring-FD then multiplying by 400 can produce
noisy Hessian entries. We report SE per entry and warn if SE/|H_ij|>0.3.
"""

import time
from typing import NamedTuple
import numpy as np
import tensorflow as tf

from ..tf_filters import TFDifferentiableParticleFilter
from ..hmc.parameterization import (
    unpack_batched, log_prior_batched, TOTAL_DIM,
)


DTYPE = tf.float64


class LaplaceResult(NamedTuple):
    """
    Output of Phase 2.

    Attributes:
        Sigma_laplace:      [9, 9]  posterior covariance estimate (after
                                    spectral regularization)
        L_chol:             [9, 9]  Cholesky factor, Sigma = L L^T
        H:                  [9, 9]  raw Hessian of log-posterior at z_hat
                                    (before regularization)
        eigenvalues_negH:   [9]     eigenvalues of raw -H (not regularized)
        eigenvalues_negH_reg: [9]   eigenvalues of regularized -H
        H_se:               [9, 9]  standard error of each Hessian entry
                                    across n_seeds
        logpost_at_mode:    float   log-posterior value at z_hat
        n_eigvals_clipped:  int     how many eigvals were floored by
                                    spectral regularization
        kappa_target:       float   the condition-number ceiling used
        elapsed:            float   seconds
    """
    Sigma_laplace: tf.Tensor
    L_chol: tf.Tensor
    H: tf.Tensor
    eigenvalues_negH: tf.Tensor
    eigenvalues_negH_reg: tf.Tensor
    H_se: tf.Tensor
    logpost_at_mode: float
    n_eigvals_clipped: int
    kappa_target: float
    elapsed: float


def _evaluate_logpost_batched(
    z_batch: tf.Tensor,                 # [B, 9]
    observations: tf.Tensor,
    dpf: TFDifferentiableParticleFilter,
    rng: tf.random.Generator,
    include_prior: bool,
) -> tf.Tensor:
    """
    Evaluate log-posterior (or log-lik only) at a batch of z.

    Note: we don't need gradients here — Phase 2 uses values only.
    We still go through unpack_batched because the DPF expects SVSSMParams.
    """
    params = unpack_batched(z_batch)
    result = dpf.filter(params, observations, rng)
    log_ev = result.log_evidence        # [B]

    if include_prior:
        lp = log_prior_batched(z_batch) # [B]
        return lp + log_ev
    else:
        return log_ev


def _evaluate_logpost_chunked(
    z_batch: tf.Tensor,                 # [B, 9]
    observations: tf.Tensor,
    dpf: TFDifferentiableParticleFilter,
    rng: tf.random.Generator,
    include_prior: bool,
    chunk_size: int,
) -> np.ndarray:
    """
    Evaluate log-posterior at a large batch by splitting it into chunks.

    The soft DPF now generates Gumbel noise per-step (via stateless RNG
    seeded by (root_seed, t)); peak memory per chunk is O(B * N^2), not
    O(T * B * N^2) as it was previously. This makes large T feasible
    without increasing chunk_size bookkeeping.

    Rough memory per step: B * N^2 * 8 bytes. At B=8, N=1000 that's
    64 MB per step — trivially small. Chunking still matters because
    the *activation tape* for backprop is T * B * N * D (not N^2), but
    the dominant soft-resample memory is no longer the bottleneck.

    All chunks within this call share the SAME `rng` state, so the
    per-stencil-point PF noise is consistent for FD purposes (each stencil
    point gets its own slice of the same noise stream as B grows).

    Args:
        z_batch: [B, 9] full stencil
        chunk_size: max B per DPF call. 8 is a safe default; can go larger
                    now that the Gumbel tensor is per-step.

    Returns:
        log_post values as np.ndarray of shape [B]
    """
    B = int(z_batch.shape[0])
    out = np.zeros(B)
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk = z_batch[start:end]
        lp = _evaluate_logpost_batched(
            chunk, observations, dpf, rng, include_prior=include_prior,
        )
        out[start:end] = lp.numpy()
    return out


def _build_stencil(
    z_hat: tf.Tensor,         # [9]
    eps: float,
) -> tuple:
    """
    Build the finite-difference stencil points for second derivatives.

    For d=9 we need:
      - 1 point at z_hat (for diagonal FD: f(θ))
      - 2*d points at z_hat ± eps*e_i (for diagonal FD and for off-diagonal
        2nd FD, we need 4*C(d,2) combo points: z_hat + eps*(±e_i ± e_j)
        for all i<j)

    We assemble a single tensor of all stencil points and return the
    index mapping so the caller can recover f-values by role.

    Returns:
        points: [n_pts, 9]  all FD points
        center_idx:              index of f(z_hat)
        axis_plus:   dict i -> index of f(z_hat + eps*e_i)
        axis_minus:  dict i -> index of f(z_hat - eps*e_i)
        cross: dict (i,j) with i<j -> tuple of 4 indices:
                  (f_pp, f_pm, f_mp, f_mm) for signs of (e_i, e_j)
    """
    d = TOTAL_DIM
    dtype = z_hat.dtype
    points = [z_hat]
    center_idx = 0

    axis_plus = {}
    axis_minus = {}
    e = tf.eye(d, dtype=dtype)

    for i in range(d):
        axis_plus[i] = len(points)
        points.append(z_hat + eps * e[i])
        axis_minus[i] = len(points)
        points.append(z_hat - eps * e[i])

    cross = {}
    for i in range(d):
        for j in range(i + 1, d):
            pp = len(points); points.append(z_hat + eps * (e[i] + e[j]))
            pm = len(points); points.append(z_hat + eps * (e[i] - e[j]))
            mp = len(points); points.append(z_hat + eps * (-e[i] + e[j]))
            mm = len(points); points.append(z_hat + eps * (-e[i] - e[j]))
            cross[(i, j)] = (pp, pm, mp, mm)

    points_tensor = tf.stack(points, axis=0)    # [n_pts, 9]
    return points_tensor, center_idx, axis_plus, axis_minus, cross


def compute_laplace_covariance(
    observations: tf.Tensor,
    dpf: TFDifferentiableParticleFilter,
    z_hat: tf.Tensor,
    eps: float = 0.05,
    n_seeds: int = 10,
    include_prior: bool = True,
    seed_base: int = 5000,
    kappa_target: float = 10.0,
    chunk_size: int = 8,
    verbose: bool = True,
) -> LaplaceResult:
    """
    Estimate Laplace covariance via finite-difference Hessian with
    spectral regularization.

    Strategy: we build a stencil of all FD points ONCE as a [n_pts, 9]
    tensor (n_pts = 1 + 2*9 + 4*36 = 163 for d=9). For each of n_seeds
    independent RNG keys, we evaluate log-posterior at all n_pts stencil
    points using the DPF, sharded into chunks of `chunk_size` to control
    memory. Then we assemble Hessian entries from the f-values.

    After eigendecomposing -H, we regularize by flooring eigenvalues at
    lambda_max / kappa_target. This bounds Σ's condition number at
    kappa_target and prevents the whitening Cholesky from producing
    proposal directions with huge norm (which at an indefinite Hessian
    would otherwise cause HMC accept rates to collapse).

    Args:
        observations:   [T, d]
        dpf:            TFDifferentiableParticleFilter (soft at N=1000)
        z_hat:          [9] MAP mode from Phase 1
        eps:            FD step size. 0.05 is the project default.
        n_seeds:        independent RNG seeds averaged per stencil point.
                        Higher = lower variance on H. Start with 10, go
                        to 30-50 if SE/|H| > 0.3 on important entries.
        include_prior:  if True, Hessian is of log-posterior;
                        if False, Hessian is of log-lik only.
                        Default True since Phase 1 is MAP.
        seed_base:      base seed; each n_seeds key = seed_base + s
        kappa_target:   target condition number of the regularized -H.
                        Eigenvalues below lambda_max / kappa_target are
                        floored to lambda_max / kappa_target. Default 10
                        gives tightly-bounded proposal directions. Use
                        higher (e.g. 100) if you want to preserve more
                        of the small-eigenvalue structure at the cost of
                        noisier HMC.
        chunk_size:     max DPF batch size per call. With the new per-step
                        stateless Gumbel generation, memory per chunk is
                        dominated by activation tape (O(T*B*N*D)) rather
                        than the old O(T*B*N^2) Gumbel buffer. chunk_size=8
                        remains a safe default; can push higher on bigger
                        GPUs to reduce wall time.
        verbose:        print progress

    Returns:
        LaplaceResult
    """
    d = TOTAL_DIM
    z_hat = tf.cast(tf.reshape(z_hat, [d]), DTYPE)

    if verbose:
        print(f"[Laplace] building FD stencil at eps={eps}")
    points, center_idx, axis_plus, axis_minus, cross = _build_stencil(
        z_hat, eps
    )
    n_pts = int(points.shape[0])
    n_chunks = (n_pts + chunk_size - 1) // chunk_size
    if verbose:
        print(f"[Laplace] n_pts={n_pts}, n_seeds={n_seeds}, "
              f"chunk_size={chunk_size} ({n_chunks} chunks per seed)")
        print(f"[Laplace] total DPF evals: {n_pts * n_seeds} "
              f"(across {n_chunks * n_seeds} DPF calls)")

    # --- Per-seed evaluation ---
    # We run n_seeds independent DPF passes over the full stencil. Each
    # pass uses a fresh RNG seeded from seed_base + s, then is sharded
    # into chunks of chunk_size to keep DPF memory bounded.
    f_vals = np.zeros((n_seeds, n_pts))           # [n_seeds, n_pts]

    t0 = time.time()
    for s in range(n_seeds):
        rng_s = tf.random.Generator.from_seed(seed_base + s)
        f_vals[s] = _evaluate_logpost_chunked(
            points, observations, dpf, rng_s,
            include_prior=include_prior,
            chunk_size=chunk_size,
        )
        if verbose and (s + 1) % max(1, n_seeds // 5) == 0:
            print(f"  seed {s + 1}/{n_seeds}: "
                  f"f(z_hat)={f_vals[s, center_idx]:+.3f}")

    # --- Assemble Hessian entries ---
    # We compute H_ij for each seed separately, then average over seeds
    # (gives us an SE per entry), rather than averaging f and then FD-ing.
    # Mathematically equivalent (linearity of FD) but cleaner for SE.

    H_per_seed = np.zeros((n_seeds, d, d))

    for s in range(n_seeds):
        f = f_vals[s]
        f0 = f[center_idx]

        # Diagonal: H_ii = [f(+e_i) - 2 f(0) + f(-e_i)] / eps^2
        for i in range(d):
            f_p = f[axis_plus[i]]
            f_m = f[axis_minus[i]]
            H_per_seed[s, i, i] = (f_p - 2 * f0 + f_m) / (eps ** 2)

        # Off-diagonal: H_ij = [f(++) - f(+-) - f(-+) + f(--)] / (4 eps^2)
        for (i, j), (pp, pm, mp, mm) in cross.items():
            val = (f[pp] - f[pm] - f[mp] + f[mm]) / (4.0 * eps ** 2)
            H_per_seed[s, i, j] = val
            H_per_seed[s, j, i] = val

    H_mean = H_per_seed.mean(axis=0)              # [d, d]
    H_se = H_per_seed.std(axis=0, ddof=1) / np.sqrt(n_seeds)

    # Symmetrize to remove any small asymmetry from numerics
    H_sym = 0.5 * (H_mean + H_mean.T)

    elapsed = time.time() - t0
    if verbose:
        print(f"[Laplace] Hessian computed in {elapsed:.1f}s")

    # --- Eigendecomposition and spectral regularization ---
    # Policy: floor eigenvalues at lambda_max / kappa_target so the final
    # condition number of -H_reg is at most kappa_target. This bounds Σ's
    # largest eigenvalue at kappa_target / lambda_max, which in turn bounds
    # the whitening matrix L_chol's column norms at sqrt(kappa_target/lambda_max).
    # Result: HMC proposals in original coords are bounded regardless of how
    # pathological the raw Hessian is.
    neg_H = -H_sym
    eigvals, eigvecs = np.linalg.eigh(neg_H)      # ascending order
    lambda_max = eigvals.max()

    if verbose:
        print(f"[Laplace] eigenvalues of -H (raw): {eigvals.round(3)}")
        print(f"[Laplace] min eig = {eigvals.min():.3e}, "
              f"max eig = {lambda_max:.3e}, "
              f"cond (raw) = {lambda_max / max(abs(eigvals.min()), 1e-30):.2e}")

    # Regularize: floor at lambda_max / kappa_target so max-to-min ratio <= kappa_target
    floor_value = lambda_max / kappa_target
    n_clipped = int((eigvals < floor_value).sum())
    eigvals_reg = np.maximum(eigvals, floor_value)

    if verbose:
        frac_clipped = n_clipped / d
        # Fraction of trace preserved in the un-clipped (well-identified) subspace
        pos_mask = eigvals > floor_value
        trace_orig = eigvals[pos_mask].sum() if pos_mask.any() else 0.0
        trace_total = eigvals_reg.sum()
        print(f"[Laplace] spectral regularization at kappa_target={kappa_target}:")
        print(f"          floor = lambda_max / kappa_target = {floor_value:.3e}")
        print(f"          eigvals clipped: {n_clipped}/{d} "
              f"({frac_clipped*100:.0f}%)")
        print(f"          trace in well-identified subspace: "
              f"{trace_orig:.2e} / {trace_total:.2e} "
              f"({100*trace_orig/max(trace_total, 1e-30):.1f}%)")
        if frac_clipped > 0.5:
            print(f"[Laplace] NOTE: more than half of directions needed flooring.")
            print(f"          MAP point likely saddle or ridge — posterior")
            print(f"          covariance dominated by prior in clipped directions.")
        print(f"[Laplace] eigenvalues of -H (regularized): {eigvals_reg.round(3)}")

    # Reconstruct regularized -H and Σ directly from the eigendecomposition.
    # Σ = V @ diag(1/λ_reg) @ V.T. No need for a solve — we have eigvecs already.
    neg_H_reg = (eigvecs * eigvals_reg[np.newaxis, :]) @ eigvecs.T
    Sigma = (eigvecs * (1.0 / eigvals_reg)[np.newaxis, :]) @ eigvecs.T
    Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrize against numerical drift
    L_Sigma = np.linalg.cholesky(Sigma)

    # --- Diagnostics: relative SE per entry ---
    if verbose:
        rel_se = np.abs(H_se) / (np.abs(H_sym) + 1e-12)
        max_rel_se = rel_se.max()
        frac_noisy = (rel_se > 0.3).sum() / (d * d)
        print(f"[Laplace] max(SE/|H|) = {max_rel_se:.2f}, "
              f"frac entries w/ SE/|H|>0.3: {frac_noisy:.2f}")
        if max_rel_se > 0.5:
            print("[Laplace] WARNING: large relative SE on Hessian entries; "
                  "consider increasing n_seeds.")

    # --- Report f at the mode for reference ---
    logpost_at_mode = float(f_vals[:, center_idx].mean())

    return LaplaceResult(
        Sigma_laplace=tf.constant(Sigma, dtype=DTYPE),
        L_chol=tf.constant(L_Sigma, dtype=DTYPE),
        H=tf.constant(H_sym, dtype=DTYPE),
        eigenvalues_negH=tf.constant(eigvals, dtype=DTYPE),
        eigenvalues_negH_reg=tf.constant(eigvals_reg, dtype=DTYPE),
        H_se=tf.constant(H_se, dtype=DTYPE),
        logpost_at_mode=logpost_at_mode,
        n_eigvals_clipped=n_clipped,
        kappa_target=float(kappa_target),
        elapsed=elapsed,
    )
