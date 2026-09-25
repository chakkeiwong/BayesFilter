"""Phase 3 Task 3.2: LEDH surrogate-force HMC damping calibration (pilot).

PILOT SCALE: N=252, 2 chains × 200 warmup + 200 samples per arm.
Reduced from plan spec (N=1008, 500+500) due to computational cost.

Each LEDH evaluation takes ~5-10 seconds with N=252.
Total: 4 arms × 2 chains × 400 steps × 2 LEDH calls = 6,400 evaluations
Estimated wall time: 9-18 hours.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# TensorFlow GPU Memory Rule (CLAUDE.md): enable and VERIFY memory growth on
# every visible physical GPU before any logical-device or runtime
# initialization. This run fails closed if the policy cannot be applied.
_VISIBLE_GPUS = tf.config.list_physical_devices("GPU")
for _gpu in _VISIBLE_GPUS:
    tf.config.experimental.set_memory_growth(_gpu, True)
    if not tf.config.experimental.get_memory_growth(_gpu):
        raise RuntimeError(
            f"set_memory_growth failed to take effect on {_gpu.name}; "
            "refusing to run with whole-device preallocation"
        )
GPU_MEMORY_POLICY = (
    "memory_growth_verified" if _VISIBLE_GPUS else "cpu_only_no_gpu_visible"
)

# Graph is the default: the repo TensorFlow Graph policy requires repeated
# numerical kernels to run through tf.function with a stable input_signature,
# and the compiled LEDH target measured 8.65x faster than eager. "eager" is
# retained only as a comparison/diagnostic mode.
EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "graph").lower()
if EXECUTION_MODE not in ("graph", "eager"):
    raise ValueError(
        f"EXECUTION_MODE must be 'graph' or 'eager', got {EXECUTION_MODE!r}"
    )

import tensorflow_probability as tfp

from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    _diagonal_lgssm_fused_model,
    _lgssm_frozen_observations,
)
from bayesfilter.inference.ledh_dual_parameter_target import (
    DualParameterLEDHTarget,
)

tfb = tfp.bijectors
tfd = tfp.distributions


def make_dual_parameter_target(
    model,
    initial_states,
    initial_covariances,
    noises,
    observations,
    damping_ratio: float,
    base_reset_ridge: float,
    base_lm_damping: float,
    **shared_params,
) -> DualParameterLEDHTarget:
    """Create dual-parameter target with specified damping ratio."""
    damped_reset_ridge = base_reset_ridge * damping_ratio
    damped_lm_damping = base_lm_damping * damping_ratio

    exact_params = {
        **shared_params,
        "reset_ridge": base_reset_ridge,
        "correction_lm_damping": base_lm_damping,
    }

    biased_params = {
        **shared_params,
        "reset_ridge": damped_reset_ridge,
        "correction_lm_damping": damped_lm_damping,
    }

    return DualParameterLEDHTarget(
        model=model,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=noises,
        observations=observations,
        exact_params=exact_params,
        biased_params=biased_params,
    )


def run_hmc_chain(
    target_log_prob_fn,
    initial_theta,
    num_burnin_steps: int,
    num_results: int,
    step_size: float,
    num_leapfrog_steps: int,
    seed: int,
):
    """Run one HMC chain with adaptive step size."""
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )

    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=0.65,
    )

    # sample_chain itself is NOT wrapped in tf.function, and that is deliberate:
    # wrapping it unrolls every HMC step into one graph, which exhausted host
    # memory in earlier attempts (see the Task 3.2 status doc, attempts 1-2).
    #
    # The graph boundary belongs around ONE force evaluation instead, where the
    # signature is fixed and the call repeats. The caller supplies that already
    # compiled via DualParameterLEDHTarget.as_graph_callable(P), so the LEDH
    # kernel runs as a traced graph while the chain loop stays in Python.
    #
    # The earlier "NO tf.function - avoid graph issues" comment here drew the
    # wrong conclusion from those OOMs: the fault was graph GRANULARITY, not
    # graph mode. Eager was measured 8.65x slower and did not fix the OOM
    # either (attempts 3-4 OOM'd in eager).
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_theta,
        kernel=adaptive_kernel,
        trace_fn=lambda _, pkr: {
            "is_accepted": pkr.inner_results.is_accepted,
            "step_size": pkr.new_step_size,
        },
        seed=seed,
    )


def score_support(target, theta):
    """Which theta coordinates receive nonzero score.

    A coordinate whose score is identically zero gets no HMC restoring force and
    random-walks. Such coordinates must be excluded from summary ESS and from
    arm-vs-arm distributional comparison, or they contaminate both: ESS would
    measure a random walk, and a pooled W1 max would be dominated by two
    unconstrained coordinates regardless of damping.

    For the diagonal LGSSM fixture used here this is expected to return [0,1,2]:
    `_diagonal_lgssm_fused_model` uses only `theta_rows[:, :3]` and pins the
    noise scales as constants, so theta[3] and theta[4] never enter the model.
    Checked at runtime rather than hardcoded, since another model may identify
    all P coordinates.
    """
    s_exact = target.score_only(theta, exact=True).numpy()
    s_biased = target.score_only(theta, exact=False).numpy()
    live = [i for i in range(s_exact.shape[0])
            if abs(s_exact[i]) > 0.0 or abs(s_biased[i]) > 0.0]
    dead = [i for i in range(s_exact.shape[0]) if i not in live]
    return live, dead


def compute_diagnostics(samples, trace, live_coords=None):
    """Acceptance rate and ESS, reported per coordinate.

    `live_coords` restricts the summary ESS to coordinates that receive score.
    Per-coordinate ESS is always reported in full so nothing is hidden.
    """
    acceptance_rate = float(tf.reduce_mean(tf.cast(trace["is_accepted"], tf.float32)))
    ess = tfp.mcmc.effective_sample_size(samples, filter_beyond_positive_pairs=True)
    ess_np = ess.numpy()

    if live_coords:
        ess_summary = float(np.mean([ess_np[i] for i in live_coords]))
    else:
        ess_summary = float(np.mean(ess_np))

    return {
        "acceptance_rate": acceptance_rate,
        "ess_per_param": ess_summary,   # live coordinates only
        "ess_all_coords": ess_np,       # full detail, nothing hidden
    }


def marginal_w1(draws_a, draws_b):
    """Per-coordinate Wasserstein-1 between two empirical samples.

    Diagnostic numpy (permitted by the backend rule for post-run diagnostics).
    Uses the exact 1-D identity W1 = integral |F_a - F_b|, computed as the mean
    absolute gap between order statistics on a common quantile grid.

    Parameters
    ----------
    draws_a, draws_b : ndarray, shape [S, P]
        Pooled draws from two arms.

    Returns
    -------
    w1 : ndarray, shape [P]
        Per-coordinate W1 distance.
    """
    n_grid = min(draws_a.shape[0], draws_b.shape[0])
    probs = (np.arange(n_grid) + 0.5) / n_grid
    w1 = np.empty(draws_a.shape[1], dtype=np.float64)
    for p in range(draws_a.shape[1]):
        qa = np.quantile(draws_a[:, p], probs)
        qb = np.quantile(draws_b[:, p], probs)
        w1[p] = float(np.mean(np.abs(qa - qb)))
    return w1


def main():
    print("=" * 70)
    print("LEDH Surrogate-Force HMC Damping Calibration (PILOT)")
    print("=" * 70)
    print("\nSCALE: N=252 particles, 200+200 steps (reduced from plan)")
    print("REASON: Each LEDH evaluation takes 5-10s; full scale would take days")
    print("\nCONFIGURATION STATUS (read before any number below):")
    print("  program: PILOT variant, NOT the plan-spec calibration run")
    print("    diff vs plan: N=252 (plan 1008), 200+200 steps (plan 500+500)")
    print("  tuning: UNTUNED for this scope (no per-scope tuning artifact)")
    print("  claim status: descriptive / nomination evidence only")
    print("    Do NOT conclude a damping ranking, W2 agreement, or HMC")
    print("    correctness from this run.")
    print(f"  execution_mode: {EXECUTION_MODE}"
          + ("" if EXECUTION_MODE == "graph" else "  (SLOW: ~8.65x eager tax)"))
    print(f"  gpu_memory_policy: {GPU_MEMORY_POLICY}")
    print(f"  gpu_devices: {[g.name for g in tf.config.list_physical_devices('GPU')]}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
          f"  CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', 'unset')}")
    print()

    # Load LGSSM T=50 fixture
    print("[1/6] Loading LGSSM T=50 fixture...")
    observations = _lgssm_frozen_observations()

    # Create model
    print("[2/6] Creating LGSSM model...")
    model = _diagonal_lgssm_fused_model()

    # PILOT parameters. The SMOKE_* overrides exist only to exercise the full
    # code path (summary, W1, save, manifest) cheaply before a long run; the
    # defaults are the pilot scale and are what a real run uses.
    d = 3
    N = int(os.environ.get("SMOKE_PARTICLES", 252))  # pilot: 252, plan: 1008
    T = observations.shape[0]
    dtype = observations.dtype

    print(f"  State dim: {d}")
    print(f"  Particles: {N} (plan: 1008)")
    print(f"  Horizon: {T}")

    # Initialize
    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=dtype) * 0.1
    initial_covariances = tf.tile(tf.eye(d, dtype=dtype)[None, :, :], [N, 1, 1]) * 0.01
    noises = generator.normal([T, N, d], dtype=dtype) * 0.1

    # Shared parameters
    reset_basis = tf.concat([tf.eye(d, dtype=dtype), -tf.eye(d, dtype=dtype)], axis=0)
    reset_repeats = (N + 2 * d - 1) // (2 * d)
    reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:N]

    shared_params = dict(
        substeps=8,
        reset_policy="contract_e",
        reset_design=reset_design,
        reset_epsilon=2.0,
        reset_sinkhorn_steps=8,
        reset_balance_steps=8,
        correction_steps=4,
        correction_strength=0.2,
        correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5,
        pairwise_steps=4,
        pairwise_strength=0.02,
        pairwise_rms_cap=2.0,
        coordinate_cap=0.0,
        annealed_stages=1,
        annealed_seed=0,
    )

    # Create targets for 4-arm sweep
    print("[3/6] Creating dual-parameter targets for 4 arms...")

    damping_ratios = [1.0, 10.0, 100.0, 1000.0]
    base_reset_ridge = 1e-5
    base_lm_damping = 1e-2

    targets = {}
    for ratio in damping_ratios:
        targets[ratio] = make_dual_parameter_target(
            model=model,
            initial_states=initial_states,
            initial_covariances=initial_covariances,
            noises=noises,
            observations=observations,
            damping_ratio=ratio,
            base_reset_ridge=base_reset_ridge,
            base_lm_damping=base_lm_damping,
            **shared_params,
        )
        print(f"  ✓ {ratio:.0f}× damping target")

    # HMC parameters (PILOT scale). SMOKE_* overrides are for the cheap
    # end-to-end path check only; defaults are the pilot scale.
    num_chains = 2
    num_burnin_steps = int(os.environ.get("SMOKE_BURNIN", 200))  # plan: 500
    num_results = int(os.environ.get("SMOKE_RESULTS", 200))  # plan: 500
    step_size = 0.01
    num_leapfrog_steps = int(os.environ.get("SMOKE_LEAPFROG", 10))
    base_seed = 97701

    true_theta = tf.constant([1.0, 1.0, 1.0, 0.5, 0.3], dtype=dtype)

    print(f"\n[4/6] HMC configuration:")
    print(f"  Chains: {num_chains}")
    print(f"  Burn-in: {num_burnin_steps} (plan: 500)")
    print(f"  Samples: {num_results} (plan: 500)")
    print(f"  Initial step size: {step_size}")
    print(f"  Leapfrog steps: {num_leapfrog_steps}")

    # Run calibration sweep
    print(f"\n[5/6] Running 4-arm calibration sweep...")
    print("NOTE: Each step takes 5-10 seconds. Estimated time: 9-18 hours.\n")

    # Which theta coordinates actually receive score. Coordinates with
    # identically zero score get no HMC restoring force and random-walk, so they
    # must not enter summary ESS or the arm-vs-arm W1 comparison. Determined at
    # runtime, once, from the 1x target.
    LIVE_COORDS, DEAD_COORDS = score_support(targets[1.0], true_theta)
    print(f"\n  score support: live={LIVE_COORDS} dead={DEAD_COORDS}")
    if DEAD_COORDS:
        print(f"  WARNING: theta{DEAD_COORDS} have identically zero score and")
        print("  will random-walk. They are excluded from summary ESS and from")
        print("  the W1 agreement comparison. For the diagonal LGSSM fixture")
        print("  this is expected: the model uses theta[:3] and pins the noise")
        print("  scales as constants, so theta[3:] never enter it. Sampling")
        print("  them is wasted work and they carry no posterior information.")

    results = {}
    for ratio in damping_ratios:
        print(f"  --- {ratio:.0f}× damping ---")
        # Graph-compiled target: traced once per arm, reused for every leapfrog
        # force evaluation. Measured 8.65x faster than the eager target at T=5.
        # Set EXECUTION_MODE=eager to fall back (diagnostics/comparison only).
        if EXECUTION_MODE == "graph":
            target = targets[ratio].as_graph_callable(int(true_theta.shape[0]))
        else:
            target = targets[ratio]

        arm_results = {
            "samples": [],
            "traces": [],
            "wall_times": [],
            "diagnostics": [],
        }

        for chain_idx in range(num_chains):
            chain_seed = base_seed + int(ratio * 10) + chain_idx
            print(f"    Chain {chain_idx + 1}/{num_chains} (seed={chain_seed})...")
            print(f"      Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")

            init_generator = tf.random.Generator.from_seed(chain_seed)
            initial_theta = true_theta + init_generator.normal([5], dtype=dtype) * 0.1

            t0 = time.time()
            samples, trace = run_hmc_chain(
                target_log_prob_fn=target,
                initial_theta=initial_theta,
                num_burnin_steps=num_burnin_steps,
                num_results=num_results,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
                seed=chain_seed,
            )
            wall_time = time.time() - t0

            diagnostics = compute_diagnostics(samples, trace, LIVE_COORDS)

            arm_results["samples"].append(samples)
            arm_results["traces"].append(trace)
            arm_results["wall_times"].append(wall_time)
            arm_results["diagnostics"].append(diagnostics)

            print(f"      Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      accept={diagnostics['acceptance_rate']:.3f}, "
                  f"ESS={diagnostics['ess_per_param']:.0f}, "
                  f"time={wall_time/60:.1f} min\n")

        results[ratio] = arm_results

    # Summary
    print(f"[6/6] Summary:")
    print("\n  Damping  Accept  ESS/param  ESS Ratio  Status")
    print("  -------  ------  ---------  ---------  ------")

    baseline_ess = None
    for ratio in damping_ratios:
        arm = results[ratio]
        mean_accept = np.mean([d["acceptance_rate"] for d in arm["diagnostics"]])
        mean_ess = np.mean([d["ess_per_param"] for d in arm["diagnostics"]])

        if baseline_ess is None:
            baseline_ess = mean_ess

        ess_ratio = mean_ess / baseline_ess if baseline_ess > 0 else 0.0
        status = "✓" if mean_accept >= 0.15 and ess_ratio >= 0.3 else "✗"

        print(f"  {ratio:6.0f}×  {mean_accept:.3f}  {mean_ess:9.0f}     {ess_ratio:.2f}     {status}")

    print("\n  Screen: acceptance ≥ 0.15, ESS ratio ≥ 0.3")
    print("  A passing arm remains VIABLE under this screen; passing is not")
    print("  evidence of superiority, and no ranking is supported by 2 chains")
    print("  x 200 draws from a single seed family.")

    # Arm-vs-arm marginal W1 against the 1x baseline. Corollary 5.2 makes the
    # theta-marginal invariant to force quality by construction, so a large W1
    # here indicates an IMPLEMENTATION defect, not a failure of the corollary.
    print("\n  Marginal W1 vs 1x baseline (per-coordinate, then max):")
    baseline_draws = np.concatenate(
        [s.numpy() for s in results[1.0]["samples"]], axis=0
    )
    baseline_sd = baseline_draws.std(axis=0)
    w1_table = {}
    for ratio in damping_ratios:
        arm_draws = np.concatenate(
            [s.numpy() for s in results[ratio]["samples"]], axis=0
        )
        w1 = marginal_w1(baseline_draws, arm_draws)
        w1_table[ratio] = w1
        # Scale-free: W1 in units of the baseline marginal SD.
        w1_rel = w1 / np.where(baseline_sd > 0, baseline_sd, np.inf)
        # Summary max over LIVE coordinates only. A dead coordinate compares one
        # unconstrained random walk against another, so its W1 is arbitrarily
        # large and unrelated to damping; including it in the max would swamp
        # the comparison.
        live_max_w1 = max(w1[i] for i in LIVE_COORDS)
        live_max_rel = max(w1_rel[i] for i in LIVE_COORDS)
        print(f"    {ratio:6.0f}x  max W1={live_max_w1:.4e}  "
              f"max W1/SD={live_max_rel:.3f}   (live coords {LIVE_COORDS})")
        if DEAD_COORDS:
            dead_max = max(w1[i] for i in DEAD_COORDS)
            print(f"            dead coords {DEAD_COORDS}: max W1={dead_max:.4e}"
                  f"  (random walk vs random walk; not interpretable)")

    print("\n  W1 is descriptive here: with ~400 pooled draws per arm the")
    print("  Monte Carlo floor on W1/SD is O(1/sqrt(ESS)), so small nonzero")
    print("  values are expected even under exact agreement.")

    # Save results
    output_dir = Path(os.environ.get(
        "OUTPUT_DIR",
        "docs/plans/artifacts/ledh-surrogate-hmc-damping-calibration-pilot",
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pilot_results.npz"
    save_arrays = {"damping_ratios": np.asarray(damping_ratios, dtype=np.float64)}
    for ratio in damping_ratios:
        tag = f"{int(ratio)}x"
        arm = results[ratio]
        save_arrays[f"samples_{tag}"] = np.asarray(
            [s.numpy() for s in arm["samples"]]
        )
        save_arrays[f"acceptance_rate_{tag}"] = np.asarray(
            [d["acceptance_rate"] for d in arm["diagnostics"]], dtype=np.float64
        )
        save_arrays[f"ess_per_param_{tag}"] = np.asarray(
            [d["ess_per_param"] for d in arm["diagnostics"]], dtype=np.float64
        )
        save_arrays[f"wall_time_{tag}"] = np.asarray(
            arm["wall_times"], dtype=np.float64
        )
        save_arrays[f"is_accepted_{tag}"] = np.asarray(
            [t["is_accepted"].numpy() for t in arm["traces"]]
        )
        save_arrays[f"step_size_{tag}"] = np.asarray(
            [t["step_size"].numpy() for t in arm["traces"]]
        )
        save_arrays[f"marginal_w1_vs_1x_{tag}"] = np.asarray(
            w1_table[ratio], dtype=np.float64
        )

    # Plain numeric arrays only: loadable without allow_pickle.
    np.savez(output_file, **save_arrays)

    print(f"\n  ✓ Results saved to {output_file}")

    # Run manifest (Scientific Coding Development Policies).
    manifest = {
        "plan_file": "docs/plans/ledh-surrogate-hmc-program-reconciliation-2026-09-12.md",
        "script": "docs/benchmarks/ledh_surrogate_hmc_damping_calibration_pilot.py",
        "command": " ".join(sys.argv),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown",
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
        ),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "gpu_memory_policy": GPU_MEMORY_POLICY,
        "gpu_devices": [g.name for g in tf.config.list_physical_devices("GPU")],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
        "execution_mode": EXECUTION_MODE,
        "execution_mode_detail": (
            "LEDH target compiled via tf.function with input_signature "
            "[TensorSpec([P])], traced once per arm and reused for every "
            "leapfrog force evaluation. sample_chain deliberately NOT wrapped: "
            "wrapping the chain unrolls all HMC steps into one graph and "
            "exhausts host memory."
            if EXECUTION_MODE == "graph" else
            "eager (comparison/diagnostic mode only; measured 8.65x slower "
            "than the compiled target at T=5, substeps=2, N=24)"
        ),
        "scale": {
            "state_dim": d,
            "particles": N,
            "horizon": int(T),
            "dtype": str(dtype),
            "num_chains": num_chains,
            "num_burnin_steps": num_burnin_steps,
            "num_results": num_results,
            "num_leapfrog_steps": num_leapfrog_steps,
            "initial_step_size": step_size,
        },
        "seeds": {
            "fixture_generator": 81100,
            "hmc_base_seed": base_seed,
            "chain_seeds": {
                f"{int(r)}x": [base_seed + int(r * 10) + c for c in range(num_chains)]
                for r in damping_ratios
            },
        },
        "damping": {
            "ratios": damping_ratios,
            "base_reset_ridge": base_reset_ridge,
            "base_correction_lm_damping": base_lm_damping,
        },
        "wall_time_seconds": {
            f"{int(r)}x": results[r]["wall_times"] for r in damping_ratios
        },
        "artifacts": {"results_npz": str(output_file)},
        "scale_deviation_from_plan": (
            "PILOT: N=252 vs plan 1008; 200+200 steps vs plan 500+500. "
            "Descriptive/nomination evidence only; not a certification run."
        ),
    }
    manifest_file = output_dir / "run_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"  ✓ Manifest saved to {manifest_file}")

    print("\n" + "=" * 70)
    print("✓ PILOT CALIBRATION COMPLETE")
    print("=" * 70)
    print("\nNOTE: This pilot used N=252, 200+200 steps.")
    print("Full calibration per plan (N=1008, 500+500) would take ~4× longer.")


if __name__ == "__main__":
    main()
