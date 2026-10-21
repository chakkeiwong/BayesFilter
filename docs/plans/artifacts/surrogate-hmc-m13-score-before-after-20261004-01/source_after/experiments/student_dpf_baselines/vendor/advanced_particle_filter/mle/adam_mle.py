"""
Phase 1: MAP estimation via Adam + DPF-soft.

Theoretical rationale
---------------------
The DPF soft resampler produces gradients with correct direction
(cos 0.99 vs Kalman truth) but ~25-30% magnitude attenuation. Adam's
running-second-moment normalization (the v_t denominator in the update
rule) absorbs this attenuation: the per-parameter effective step size
self-scales, so a uniformly attenuated gradient produces the same update
as the unattenuated gradient. This is the same reason Adam tolerates
gradient noise in LLM pre-training.

Consequence: Adam converges to the correct MAP mode even though each
individual gradient is biased. Diagnostic C2 in the project confirmed
soft's gradient zero-crossings align with Kalman's, so the optimum
location is correct — only the path to it is noisy.

Multi-start
-----------
We run B_restart=4 Adam chains in parallel, each starting from a
jittered warm_start, and pick argmax(log-posterior) at the end. This
protects against local optima from non-convexity of the SVSSM posterior
(particularly the Phi-Sigma ridge).

All restarts are batched along the DPF's B dimension: one DPF call per
Adam step at B = B_restart * n_mc. No Python-level parallelism needed.
"""

import time
from typing import NamedTuple, Optional
import numpy as np
import tensorflow as tf

from ..tf_filters import TFDifferentiableParticleFilter
from ..hmc.parameterization import (
    unpack_batched, log_prior_batched, TOTAL_DIM,
)


DTYPE = tf.float64


class AdamMLEResult(NamedTuple):
    """
    Output of Phase 1.

    Attributes:
        z_hat:           [9] MAP point (best restart)
        z_all_restarts:  [B_restart, 9] final z for every restart
        final_log_post:  [B_restart] final log-posterior per restart
        best_restart:    int, index of the winning restart
        loss_history:    [n_steps, B_restart] -log_posterior per step per restart
        grad_norm_history: [n_steps, B_restart]
        elapsed:         float, seconds
    """
    z_hat: tf.Tensor
    z_all_restarts: tf.Tensor
    final_log_post: tf.Tensor
    best_restart: int
    loss_history: np.ndarray
    grad_norm_history: np.ndarray
    elapsed: float


def _build_map_objective(
    observations: tf.Tensor,
    dpf: TFDifferentiableParticleFilter,
    rng: tf.random.Generator,
    n_mc: int,
):
    """
    Build a batched MAP log-posterior function.

    Mirrors build_target_log_prob_fn in hmc/run_hmc_poc.py but lives in
    this module to keep Phase 1 self-contained. Returns [B_restart]
    log-posterior values given z of shape [B_restart, 9].

    The DPF runs once per call with B = B_restart * n_mc, then averages
    over the n_mc axis for variance reduction in the gradient.
    """

    @tf.function(reduce_retracing=True)
    def log_posterior_fn(z):
        # z: [B_restart, 9]
        B_restart = tf.shape(z)[0]

        # Replicate for MC variance reduction: [B_restart*n_mc, 9]
        z_rep = tf.repeat(z, repeats=n_mc, axis=0)
        params = unpack_batched(z_rep)

        result = dpf.filter(params, observations, rng)
        log_ev_flat = result.log_evidence  # [B_restart * n_mc]

        log_ev_2d = tf.reshape(log_ev_flat, [B_restart, n_mc])
        log_ev = tf.reduce_mean(log_ev_2d, axis=-1)  # [B_restart]

        lp = log_prior_batched(z)  # [B_restart]
        return lp + log_ev

    return log_posterior_fn


def make_multistart_z0(
    y_obs: tf.Tensor,
    B_restart: int = 4,
    jitter: float = 0.3,
    seed: int = 1234,
) -> tf.Tensor:
    """
    Multi-start initialization: take the data-driven warm start and
    perturb it B_restart times with larger jitter than used for HMC.

    Rationale: warm_start() in run_hmc_poc.py uses jitter=0.02 because
    HMC chains should start close together. For optimization restarts,
    we want wider spread to escape potential local basins — 0.3 gives
    ~0.3 std dev perturbations, enough to reach different basins in
    non-degenerate directions while staying within the prior's support.

    Args:
        y_obs:     [T, d] observations
        B_restart: number of restarts
        jitter:    perturbation std dev in unconstrained space
        seed:      RNG seed

    Returns:
        z0: [B_restart, 9]
    """
    from ..hmc.run_hmc_poc import warm_start
    # warm_start already jitters; we pass jitter directly.
    return warm_start(y_obs, B_chain=B_restart, jitter=jitter)


def run_adam_mle(
    observations: tf.Tensor,
    dpf: TFDifferentiableParticleFilter,
    rng: tf.random.Generator,
    z_init: Optional[tf.Tensor] = None,
    B_restart: int = 4,
    jitter: float = 0.3,
    n_steps: int = 1000,
    learning_rate: float = 0.01,
    n_mc: int = 4,
    log_every: int = 50,
    rolling_window: int = 100,
    tol_rel: float = 0.0,     # default: DISABLED. With stochastic DPF gradients,
                              # rolling-mean tolerance can fire on random-walk
                              # stationarity rather than true convergence. Run
                              # full n_steps unless user explicitly opts in.
    verbose: bool = True,
    init_seed: int = 1234,
) -> AdamMLEResult:
    """
    Run Adam optimization to find the MAP mode.

    All B_restart restarts run simultaneously — one Adam step per DPF call
    at B = B_restart * n_mc. The best restart is selected at the end by
    final log-posterior.

    Args:
        observations:  [T, d]
        dpf:           TFDifferentiableParticleFilter (soft resampler
                       recommended; N=1000 for best SNR)
        rng:           tf.random.Generator for PF sampling
        z_init:        [B_restart, 9] custom initialization. If None,
                       uses multi-start warm_start with jitter.
        B_restart:     number of parallel restarts (ignored if z_init given)
        jitter:        perturbation size for multi-start
        n_steps:       Adam steps
        learning_rate: Adam lr
        n_mc:          MC replicas per gradient evaluation (variance reduction)
        log_every:     print frequency
        rolling_window: steps over which to compute rolling loss for
                       convergence monitoring
        tol_rel:       relative tolerance on rolling loss change for early stop
        verbose:       print progress
        init_seed:     seed for multi-start jitter

    Returns:
        AdamMLEResult
    """
    # --- Initialization ---
    if z_init is None:
        z_init = make_multistart_z0(
            observations, B_restart=B_restart,
            jitter=jitter, seed=init_seed,
        )
    else:
        z_init = tf.cast(z_init, DTYPE)
        B_restart = int(z_init.shape[0])

    if verbose:
        print(f"[Adam-MAP] B_restart={B_restart}, n_mc={n_mc}, "
              f"effective batch={B_restart * n_mc}")
        print(f"[Adam-MAP] n_steps={n_steps}, lr={learning_rate}")

    # --- Optimizer setup ---
    z_var = tf.Variable(z_init, dtype=DTYPE, name="z_map")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    log_post_fn = _build_map_objective(observations, dpf, rng, n_mc=n_mc)

    # Sanity check initial z: reject-resample any restart whose log-post or
    # gradient is non-finite (typically happens when jitter puts a restart
    # in a region where SVSSM's exp(-h)*y^2 observation likelihood overflows).
    max_resample_tries = 5
    for attempt in range(max_resample_tries):
        with tf.GradientTape() as tape:
            lp_init = log_post_fn(z_var)
            loss_init = -tf.reduce_sum(lp_init)
        g_init = tape.gradient(loss_init, [z_var])[0].numpy()
        lp_np = lp_init.numpy()
        row_ok = np.isfinite(lp_np) & np.isfinite(g_init).all(axis=-1)
        n_bad = int((~row_ok).sum())
        if n_bad == 0:
            if verbose:
                print(f"[Adam-MAP] init OK: all restarts have finite "
                      f"log-post/grad. log_post={lp_np.round(2)}")
            break
        if verbose:
            print(f"[Adam-MAP] init attempt {attempt}: {n_bad}/{B_restart} "
                  f"restart(s) have non-finite log-post/grad; resampling with "
                  f"jitter/2.")
        # Resample only the bad restarts, with halved jitter
        bad_idx = np.where(~row_ok)[0]
        new_jitter = jitter / (2 ** (attempt + 1))
        z_new = make_multistart_z0(
            observations, B_restart=B_restart,
            jitter=new_jitter, seed=init_seed + attempt * 101,
        ).numpy()
        z_arr = z_var.numpy()
        z_arr[bad_idx] = z_new[bad_idx]
        z_var.assign(z_arr)
    else:
        # all tries exhausted
        if n_bad > 0:
            print(f"[Adam-MAP] WARNING: after {max_resample_tries} tries, "
                  f"{n_bad} restart(s) still non-finite. Proceeding; NaN "
                  f"guard in adam_step will freeze them.")

    # History buffers
    loss_hist = np.zeros((n_steps, B_restart))
    grad_norm_hist = np.zeros((n_steps, B_restart))

    # --- Single compiled Adam step ---
    # We compile the full loss+grad+apply in one tf.function for speed.
    # apply_gradients must stay inside the compiled step to avoid
    # per-step retracing via Python-side optimizer state access.
    @tf.function(reduce_retracing=True)
    def adam_step():
        with tf.GradientTape() as tape:
            log_post = log_post_fn(z_var)         # [B_restart]
            # Adam minimizes; we negate to maximize log-posterior.
            # Sum across restarts so the gradient for each restart
            # is its own per-restart grad (independent rows of z_var).
            loss = -tf.reduce_sum(log_post)
        grads = tape.gradient(loss, [z_var])

        # NaN guard: if the SVSSM observation likelihood overflows (happens
        # when a restart drifts into a region where exp(-h)*y^2 is huge),
        # the DPF returns NaN gradients. Without this guard, a single NaN
        # poisons Adam's m/v statistics forever. With it, the restart's
        # update is skipped for that step; the row stays put and can recover
        # once the gradient is clean again.
        grad0 = grads[0]
        grad_is_finite = tf.reduce_all(
            tf.math.is_finite(grad0), axis=-1, keepdims=True
        )                                          # [B_restart, 1]
        grad_clean = tf.where(
            grad_is_finite,
            grad0,
            tf.zeros_like(grad0),
        )
        optimizer.apply_gradients(zip([grad_clean], [z_var]))

        # Per-restart diagnostics (reported on the raw gradient so the user
        # sees NaN if it happened)
        grad_norm = tf.norm(grad0, axis=-1)       # [B_restart]
        return log_post, grad_norm

    # --- Main loop ---
    t0 = time.time()
    prev_rolling = None
    n_nan_warnings = 0
    for step in range(n_steps):
        log_post, grad_norm = adam_step()
        loss_hist[step] = -log_post.numpy()       # store as -log_post (loss)
        grad_norm_hist[step] = grad_norm.numpy()

        # Warn on NaN gradient (early warning — cheaper to catch)
        n_nan = int(np.sum(~np.isfinite(grad_norm_hist[step])))
        if n_nan > 0 and n_nan_warnings < 3:
            print(f"  [step {step}] WARNING: {n_nan}/{B_restart} restart(s) "
                  f"had NaN gradient; update skipped for those restart(s).")
            print(f"  [step {step}] grad_norm per restart: "
                  f"{grad_norm_hist[step]}")
            if n_nan_warnings == 2:
                print(f"  (suppressing further NaN warnings)")
            n_nan_warnings += 1

        if verbose and (step % log_every == 0 or step == n_steps - 1):
            # Report the *best* restart for readability; all restarts tracked
            # in history.
            best_lp = float(tf.reduce_max(log_post).numpy())
            mean_gnorm = float(tf.reduce_mean(grad_norm).numpy())
            print(f"  step {step:4d}  best_log_post={best_lp:+.2f}  "
                  f"mean_grad_norm={mean_gnorm:.3f}")

        # Convergence check: relative change in rolling-mean loss.
        # Skip entirely when tol_rel <= 0 (the default), which runs the full
        # n_steps — robust to noisy gradient magnitudes that can make the
        # rolling-mean criterion fire on random-walk stationarity.
        if tol_rel > 0 and step >= 2 * rolling_window:
            curr = loss_hist[step - rolling_window + 1: step + 1].mean(axis=0)
            prev = loss_hist[
                step - 2 * rolling_window + 1: step - rolling_window + 1
            ].mean(axis=0)
            # Use best restart's progress
            rel_change = np.abs(curr - prev) / (np.abs(prev) + 1e-8)
            if rel_change.min() < tol_rel:
                if verbose:
                    print(f"  [converged at step {step}: min rel_change="
                          f"{rel_change.min():.2e} < tol={tol_rel:.1e}]")
                # Truncate history
                loss_hist = loss_hist[: step + 1]
                grad_norm_hist = grad_norm_hist[: step + 1]
                break

    elapsed = time.time() - t0
    if verbose:
        print(f"[Adam-MAP] done in {elapsed:.1f}s")

    # --- Select best restart ---
    # Evaluate final log-posterior cleanly (not from the training step's stale copy)
    final_lp = log_post_fn(z_var)                  # [B_restart]
    best_idx = int(tf.argmax(final_lp).numpy())
    z_hat = z_var[best_idx]                         # [9]

    if verbose:
        print(f"[Adam-MAP] final log-posteriors per restart: "
              f"{final_lp.numpy().round(2)}")
        print(f"[Adam-MAP] best restart = {best_idx}  "
              f"log_post = {float(final_lp[best_idx].numpy()):+.3f}")

    return AdamMLEResult(
        z_hat=z_hat,
        z_all_restarts=tf.identity(z_var),  # snapshot
        final_log_post=final_lp,
        best_restart=best_idx,
        loss_history=loss_hist,
        grad_norm_history=grad_norm_hist,
        elapsed=elapsed,
    )
