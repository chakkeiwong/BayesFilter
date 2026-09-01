"""Which component carries the T=3 1D LGSSM error? Component ablation.

Context. On a 1D linear Gaussian model at T=3 with N=1008 the value lane
returns -5.7904 against an exact Kalman likelihood of -5.9034 (+1.9%), and
the analytical score matches a finite difference of the filter's own value
to 6 decimals (-0.239417 both) while the exact Kalman score is -0.429926.
So the tangent is the correct derivative OF THE FILTER, and the filter
itself is what disagrees with Kalman. This script asks which piece of the
filter causes that disagreement.

Why this is the right question. The LEDH flow, the entropic Sinkhorn reset,
and the trust-region caps each modify the cloud. For a linear Gaussian
model the bootstrap particle filter's likelihood estimator is unbiased for
the true likelihood, and the UKF recursion IS the Kalman recursion, so a
+1.9% offset at N=1008 in d=1 needs a named cause, not "Monte Carlo".

Ablation ladder, each arm removing one mechanism:
  A. full           flow=12, contract_e reset, dual-cap + pairwise caps
  B. no caps        flow=12, contract_e reset, no trust region
  C. no reset       flow=12, no reset (weights carried), no caps
  D. flow refinement  flow in {12, 24, 48, 96}, reset on, caps off
  E. epsilon ladder   eps in {1.0, 0.5, 0.25, 0.1}, reset on, caps off
  F. particle ladder  N in {504, 1008, 2016, 4032}, full config

Reading the result. A mechanism is implicated if removing it (or refining
it toward its exact limit) moves the value toward the exact Kalman
likelihood. If refining a mechanism's discretization drives the error to
zero, that mechanism's discretization is the cause. If the error is flat
in N but falls with flow refinement, the cause is deterministic
discretization error, not sampling error -- and a deterministic offset in
the value is exactly what would corrupt its derivative.

ESTIMAND WARNING carried from the score lane's own docstring: with
reset_policy="none" and annealed_stages=1 each step assumes uniform
incoming weights but never resamples, so arm C's VALUE is not a
log-likelihood estimator for T>1. Arm C is reported as a mechanism probe
only and is labeled as such; it is not evidence about likelihood accuracy.

Multi-seed. Every arm runs the same seed set so that an arm difference is
not read off one noise draw. Single-seed particle filter values move by
percent-level amounts between draws; that is precisely the confusion this
script exists to avoid.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from bayesfilter.highdim.ledh_canonical_filter_tf import (  # noqa: E402
    CanonicalModelCallbacks,
    canonical_value_and_diagnostics,
)
from bayesfilter.highdim.ledh_diagonal_lgssm_any_dim import (  # noqa: E402
    diagonal_lgssm_any_dim,
)

DTYPE = tf.float64
THETA = np.array([0.9, 0.6, 0.8])  # [phi, sigma_w, sigma_v]
HORIZON = 3
SEEDS = [999000 + i for i in range(8)]


def exact_kalman(obs, theta=THETA):
    phi, sw, sv = theta
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


def bootstrap_reference(obs, seed, particles=200000):
    """Plain bootstrap particle filter: the honest heuristic adversary.

    For a linear Gaussian model this estimator is unbiased for the
    likelihood, with no flow, no transport reset, and no caps. At 200k
    particles in d=1 its Monte Carlo error is small, so it separates
    "particle filtering in this model is hard" (it is not) from "one of
    our mechanisms introduces an offset".
    """
    phi, sw, sv = THETA
    rng = np.random.default_rng(500000 + seed)
    x = rng.standard_normal(particles)
    total = 0.0
    for y in obs:
        x = phi * x + sw * rng.standard_normal(particles)
        logw = -0.5 * (
            ((y - x) / sv) ** 2 + np.log(2.0 * np.pi * sv * sv)
        )
        m = logw.max()
        total += float(m + np.log(np.mean(np.exp(logw - m))))
        w = np.exp(logw - m)
        w /= w.sum()
        idx = rng.choice(particles, size=particles, p=w)
        x = x[idx]
    return total


def make_callbacks(theta_t, model):
    return CanonicalModelCallbacks(
        model_id="lgssm1d_ablation",
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


def value_lane(obs, seed, *, flow_substeps, epsilon, sinkhorn_steps,
               caps, particles):
    theta_t = tf.constant(THETA, DTYPE)
    model, _ = diagonal_lgssm_any_dim(
        theta_t, dim=1, obs_matrix=tf.constant([[1.0]], DTYPE)
    )
    result = canonical_value_and_diagnostics(
        callbacks=make_callbacks(theta_t, model),
        observations=tf.constant(obs[:, None], DTYPE),
        particle_count=particles,
        seed=seed,
        flow_substeps=flow_substeps,
        temper_stages=1,
        annealed_resampling=False,
        flow_prior_cap=float("inf"),
        resample_seed=seed,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=sinkhorn_steps,
        ridge=1.0e-5,
        dual_cap_enabled=caps,
        trust_region_enabled=caps,
        trust_region_lm_damping=1.0e-2,
        trust_region_lm_scale_floor=1.0e-4,
        trust_region_radius=0.5,
    )
    return {
        "value": float(result["value"].numpy()),
        "program_valid": bool(result["program_valid"].numpy()),
        "max_marginal_tv": float(result["max_marginal_tv_error"].numpy()),
        "marginals_valid": bool(result["all_marginals_valid"].numpy()),
        "ess_min": float(np.nanmin(result["per_step_ess"].numpy())),
    }


def summarize(values, ref):
    arr = np.array([v for v in values if np.isfinite(v)])
    if arr.size == 0:
        return {"n": 0, "mean": None, "se": None, "bias": None,
                "bias_se": None, "separated": None}
    err = arr - ref
    se = float(err.std(ddof=1) / np.sqrt(err.size)) if err.size > 1 else None
    return {
        "n": int(arr.size),
        "mean_value": float(arr.mean()),
        "bias": float(err.mean()),
        "bias_se": se,
        "separated_from_zero_2se": (
            bool(abs(err.mean()) > 2.0 * se) if se else None
        ),
    }


def main():
    rng = np.random.default_rng(9000)
    phi, sw, sv = THETA
    x, obs = 0.0, []
    for _ in range(HORIZON):
        x = phi * x + sw * rng.standard_normal()
        obs.append(x + sv * rng.standard_normal())
    obs = np.array(obs)

    ref = exact_kalman(obs)
    print(f"1D LGSSM, T={HORIZON}, obs={np.round(obs, 4)}")
    print(f"exact Kalman log-likelihood: {ref:.6f}\n")

    boot = [bootstrap_reference(obs, s) for s in SEEDS[:4]]
    b = summarize(boot, ref)
    sep = b["separated_from_zero_2se"]
    print(
        f"bootstrap PF (200k, no flow/reset/caps): "
        f"bias {b['bias']:+.4f} +/- {b['bias_se']:.4f}  "
        f"separated={sep}"
    )
    print("  ^ heuristic adversary: if this is ~0 the model is easy and\n"
          "    any large arm bias below belongs to that arm's mechanism.\n")

    arms = []

    def run_arm(name, note, **kw):
        rows = [value_lane(obs, s, **kw) for s in SEEDS]
        vals = [r["value"] for r in rows]
        s = summarize(vals, ref)
        arm = {"arm": name, "note": note, "config": kw, "summary": s,
               "rows": rows}
        arms.append(arm)
        sep = s["separated_from_zero_2se"]
        se_s = f"{s['bias_se']:.4f}" if s["bias_se"] else "n/a"
        tv = max(r["max_marginal_tv"] for r in rows)
        ess = min(r["ess_min"] for r in rows)
        print(
            f"{name:28s} bias {s['bias']:+.4f} +/- {se_s}  "
            f"sep={str(sep):5s}  worstTV {tv:.1e}  minESS {ess:7.1f}"
        )
        return arm

    base = dict(flow_substeps=12, epsilon=1.0, sinkhorn_steps=8,
                caps=True, particles=1008)

    print("--- A/B/C: mechanism removal (flow=12, N=1008) ---")
    run_arm("A full (reset+caps)", "production-shaped", **base)
    run_arm("B no caps", "trust region off",
            **{**base, "caps": False})

    print("\n--- D: flow refinement (reset on, caps off) ---")
    for f in (12, 24, 48, 96):
        run_arm(f"D flow={f}", "flow discretization ladder",
                **{**base, "caps": False, "flow_substeps": f})

    print("\n--- E: epsilon ladder (reset on, caps off, flow=12) ---")
    for e in (1.0, 0.5, 0.25, 0.1):
        run_arm(f"E eps={e}", "entropic regularization ladder",
                **{**base, "caps": False, "epsilon": e,
                   "sinkhorn_steps": 24})

    print("\n--- F: particle ladder (full config, flow=12) ---")
    for n in (504, 1008, 2016, 4032):
        run_arm(f"F N={n}", "sampling error ladder",
                **{**base, "particles": n})

    out = {
        "study": "lgssm_1d_t3_component_ablation",
        "date": "2026-08-28",
        "question": (
            "which filter mechanism causes the +1.9% value offset vs exact "
            "Kalman on a 1D linear Gaussian model at T=3, N=1008"
        ),
        "why_it_matters": (
            "the analytical score was verified to equal a finite difference "
            "of the filter's own value to 6 decimals, so the tangent is "
            "correct and the value offset is what makes the score differ "
            "from the exact Kalman score"
        ),
        "model": {"family": "1D AR(1) linear Gaussian", "dim": 1,
                  "theta": THETA.tolist(), "horizon": HORIZON,
                  "observations": obs.tolist()},
        "exact_kalman": ref,
        "bootstrap_reference": {"particles": 200000, "values": boot,
                                "summary": b},
        "seeds": SEEDS,
        "arms": arms,
        "interpretation_rules": [
            "An arm's bias is 'real' only if separated_from_zero_2se is "
            "true; otherwise it is descriptive at this seed count.",
            "If bias falls monotonically as flow_substeps rises, flow "
            "discretization is a cause.",
            "If bias falls as epsilon falls, entropic regularization of "
            "the reset is a cause.",
            "If bias is flat in N, the offset is deterministic, not "
            "sampling error, and cannot be fixed with more particles.",
        ],
        "non_claims": [
            "Arm C (reset off) VALUE is not a log-likelihood estimator for "
            "T>1 per the score lane's own estimand warning; it is a "
            "mechanism probe only.",
            "This is one model, one horizon, one direction. No result here "
            "transfers to another model or route.",
            "Ranking arms by bias is descriptive; nothing here promotes a "
            "configuration.",
        ],
    }
    Path("docs/benchmarks/lgssm_1d_t3_ablation.json").write_text(
        json.dumps(out, indent=2)
    )
    print("\nsaved docs/benchmarks/lgssm_1d_t3_ablation.json")


if __name__ == "__main__":
    main()
