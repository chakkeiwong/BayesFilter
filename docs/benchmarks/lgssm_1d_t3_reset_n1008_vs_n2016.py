"""Instrument N=1008 vs N=2016 reset to find the pathology.

Context: N=1008 gives bias +0.136, N=2016 gives bias +0.046. Both run the
same reset (contract_e, eps=1.0, sk=8), but N=1008 is 3× worse. This
script captures per-step reset diagnostics to see what differs.

What to capture at each step:
- Input weights (pre-reset)
- Output weights (post-reset, should be uniform)
- Marginal TV error
- Input cloud statistics (mean, variance)
- Output cloud statistics
- Reset displacement magnitude
- Whether dual-cap fired and its magnitude

Hypothesis space:
1. N=1008 hits a bad case in the Sinkhorn iteration (but marginal TV is low)
2. The reset design tensor interacts badly with N=1008
3. The dual-cap fires differently at the two N values
4. The particle geometry at N=1008 causes large transport cost
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

DTYPE = tf.float64
THETA = np.array([0.9, 0.6, 0.8])
HORIZON = 3
SEED = 999000


def exact_kalman(obs):
    phi, sw, sv = THETA
    mean, var, total = 0.0, 1.0, 0.0
    for y in obs:
        mean = phi * mean
        var = phi * var * phi + sw * sw
        s = var + sv * sv
        resid = y - mean
        total += -0.5 * (resid * resid / s + np.log(2.0 * np.pi * s))
        gain = var / s
        mean, var = mean + gain * resid, (1.0 - gain) * var
    return float(total)


def run_with_diagnostics(obs, particles):
    """Run the filter with step-by-step reset diagnostics captured."""
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        CanonicalModelCallbacks,
        canonical_value_and_diagnostics,
    )
    from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import (
        diagonal_lgssm_any_dim,
    )

    theta_t = tf.constant(THETA, DTYPE)
    model, _ = diagonal_lgssm_any_dim(
        theta_t, dim=1, obs_matrix=tf.constant([[1.0]], DTYPE)
    )

    callbacks = CanonicalModelCallbacks(
        model_id=f"reset_diag_N{particles}",
        state_dim=1,
        observation_dim=1,
        transition_mean_fn=lambda p, t: model.transition_mean_fn(theta_t, p),
        transition_log_density_fn=(
            lambda p, m, t: model.transition_log_density_fn(theta_t, p, m)
        ),
        process_noise_covariance=model.process_covariance,
        process_noise_covariance_provenance="model_exact",
        observation_fn=lambda p, t: model.observation_fn(p),
        observation_jacobian_fn=lambda p, t: model.observation_jacobian_fn(p),
        observation_covariance=model.observation_covariance,
        observation_log_density_fn=(
            lambda p, o, t: model.observation_log_density_fn(theta_t, p, o)
        ),
        initial_mean=tf.zeros([1], DTYPE),
        initial_covariance=tf.eye(1, dtype=DTYPE),
        initial_covariance_provenance="model_exact",
    )

    result = canonical_value_and_diagnostics(
        callbacks=callbacks,
        observations=tf.constant(obs[:, None], DTYPE),
        particle_count=particles,
        seed=SEED,
        flow_substeps=12,
        temper_stages=1,
        annealed_resampling=False,
        flow_prior_cap=float("inf"),
        resample_seed=SEED,
        epsilon=1.0,
        sinkhorn_steps=8,
        balance_steps=8,
        ridge=1.0e-5,
        dual_cap_enabled=True,
        trust_region_enabled=True,
        trust_region_lm_damping=1.0e-2,
        trust_region_lm_scale_floor=1.0e-4,
        trust_region_radius=0.5,
    )

    return {
        "value": float(result["value"].numpy()),
        "program_valid": bool(result["program_valid"].numpy()),
        "per_step_ess": result["per_step_ess"].numpy().tolist(),
        "per_step_marginal_tv": result["per_step_marginal_tv_error"]
        .numpy()
        .tolist(),
        "per_step_marginal_valid": result["per_step_marginal_valid"]
        .numpy()
        .tolist(),
        "max_marginal_tv": float(result["max_marginal_tv_error"].numpy()),
        "all_marginals_valid": bool(result["all_marginals_valid"].numpy()),
    }


def compare_diagnostics(d1008, d2016, ref):
    """Compare the two runs step-by-step."""
    print(f"N=1008 value: {d1008['value']:.6f}  bias: {d1008['value']-ref:+.6f}")
    print(f"N=2016 value: {d2016['value']:.6f}  bias: {d2016['value']-ref:+.6f}")
    print()

    print("Per-step ESS:")
    print("  Step  N=1008    N=2016    Δ(2016-1008)")
    for t in range(HORIZON):
        e1 = d1008["per_step_ess"][t]
        e2 = d2016["per_step_ess"][t]
        print(f"    {t}   {e1:7.1f}   {e2:7.1f}   {e2-e1:+8.1f}")
    print()

    print("Per-step marginal TV error:")
    print("  Step  N=1008        N=2016        Ratio(1008/2016)")
    for t in range(HORIZON):
        tv1 = d1008["per_step_marginal_tv"][t]
        tv2 = d2016["per_step_marginal_tv"][t]
        ratio = tv1 / tv2 if tv2 > 0 else float("inf")
        print(f"    {t}   {tv1:.3e}     {tv2:.3e}     {ratio:.2f}")
    print()

    print(f"Max marginal TV: N=1008 {d1008['max_marginal_tv']:.3e}, "
          f"N=2016 {d2016['max_marginal_tv']:.3e}")
    print(f"All marginals valid: N=1008 {d1008['all_marginals_valid']}, "
          f"N=2016 {d2016['all_marginals_valid']}")


def main():
    rng = np.random.default_rng(9000)
    phi, sw, sv = THETA
    x, obs = 0.0, []
    for _ in range(HORIZON):
        x = phi * x + sw * rng.standard_normal()
        obs.append(x + sv * rng.standard_normal())
    obs = np.array(obs)

    ref = exact_kalman(obs)
    print(f"1D LGSSM T={HORIZON}, obs={np.round(obs, 4)}")
    print(f"Exact Kalman: {ref:.6f}\n")

    print("Running N=1008...")
    d1008 = run_with_diagnostics(obs, 1008)

    print("Running N=2016...")
    d2016 = run_with_diagnostics(obs, 2016)

    print("\n" + "=" * 70)
    compare_diagnostics(d1008, d2016, ref)

    output = {
        "study": "lgssm_1d_t3_reset_diagnostic_n1008_vs_n2016",
        "date": "2026-08-28",
        "question": "why does N=1008 have 3× larger bias than N=2016",
        "model": {
            "family": "1D AR(1) linear Gaussian",
            "theta": THETA.tolist(),
            "horizon": HORIZON,
            "observations": obs.tolist(),
        },
        "exact_kalman": ref,
        "n1008": d1008,
        "n2016": d2016,
        "note": (
            "Both runs use the same seed, observations, and controls "
            "(eps=1.0, sk=8, flow=12, dual_cap=True). The only difference "
            "is particle count."
        ),
    }

    Path("docs/benchmarks/lgssm_1d_t3_reset_n1008_vs_n2016.json").write_text(
        json.dumps(output, indent=2)
    )
    print("\nsaved docs/benchmarks/lgssm_1d_t3_reset_n1008_vs_n2016.json")


if __name__ == "__main__":
    main()
