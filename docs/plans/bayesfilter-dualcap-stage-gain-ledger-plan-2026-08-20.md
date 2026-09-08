# Dual-Cap Stage-Gain Ledger Experiment Plan

Date: 2026-08-20
Status: `BLOCKED_AWAITING_LEDH_COVARIANCE_REPAIR`

Owner: chakwong
Lane: GenUT score accuracy / perturbation damping mechanism

## Context

The 2026-08-01 per-step tangent-growth analysis
(`bayesfilter-genut-score-growth-n4032-comparison-result-2026-08-01.md`)
established that the full particle filter amplifies perturbations at
+0.1–0.27 per step during the contracting phase (steps 8–20) where the
physical SIR dynamics damp at −0.25/step, producing catastrophic 800–1500×
cumulative amplification over T=20 against the physics' 0.36× contraction.
The pairwise higher-moment correction showed the largest marginal gain
(+0.17/step over diagonal-only), but the diagonal-only arm still grew at
+0.10/step, meaning the shared reset/transport/reweighting machinery
contributes substantial gain independently.

The owner identified the core defect: **the algorithm lacks a damping
mechanism**. When the model dynamics become contractive (eigenvalue < 1),
the filter should inherit that contraction and forget perturbations at
nearly the physical rate. Instead, the per-step filter map has perturbation
gain ≥ 1 built into its structure, primarily via the self-anchored
moment-restoration scheme: every step, the dual-cap and Contract-E resets
re-standardize the cloud against its own perturbed empirical moments,
recycling moment-space errors at unit gain rather than damping them against
any external or predicted reference.

## Objective

Decompose the per-step tangent growth across the filter's stages (base
reset → +diagonal → +pairwise capped/uncapped → cap strength ladder →
+trust-region LM proxy) to:

1. Confirm pairwise as the largest marginal amplifier and measure whether
   the dual-cap coordinate/radial caps (introduced 2026-08-07/08 to bound
   pairwise displacement magnitude) reduce tangent *gain* or only bound
   *magnitude*.
2. Quantify how much gain lives in the shared reset stage (which is
   preserved under the keep-self-anchoring policy for mean/covariance) so
   follow-up damping designs (temporally smoothed moments, gain scheduling,
   shrinkage) know their target.
3. Establish a per-stage, per-step gain ledger against the physical −0.25
   reference to guide the score-accuracy campaign's next phase.

## Method

Reuse the 2026-08-01 finite-time tangent-growth probe harness
(`run_genut_score_variance_repair_validation.py`) with seven stage-ablated
arms on the Austria SIR d=18 model, N=1008 (mechanism scale), 8 tangent
probes × 8 seeds, TF32-on XLA. Each arm reports per-step log growth through
T=20.

### Arms

1. **base_reset_only**: `transition_value` + reweight + Contract-E affine
   restoration (Sinkhorn ε=8, balance 16 steps, ridge 1e-5), no higher
   moments, no trust region.
2. **diagonal_uncapped**: +diagonal 4 steps, strength 0.2, no caps.
3. **diagonal_pairwise_uncapped**: +diagonal + pairwise 4 steps, strength
   0.02, no caps (pre-dual-cap baseline).
4. **diagonal_pairwise_capped_current**: +dual-cap coordinate cap (power 8,
   fraction 0.98) + radial cap (rms 2.0) — the frozen 2026-08-18+ default.
5. **diagonal_pairwise_capped_loose**: same, radial cap rms 4.0.
6. **diagonal_pairwise_capped_tight**: same, radial cap rms 1.0.
7. **full_with_trust_region**: +LM trust-region reset (damping 1e-2, radius
   0.5, floor 1e-4) — the complete dual-cap + trust-region route.

Each arm runs the same Austria SIR configuration (β=0.55, γ=0.35,
σ_obs=exp(θ₂), T=20, frozen observations/process-noise/initial-noise from
seed 97701, same state-map fixed location/scale) with frozen RQMC design
(Halton prime [2,3,5]), `hilbert_permutation_one_to_one` ancestry,
`trust_region` reset policy toggle.

### Harness Integration

Port the arms by overriding the harness's `_load_context()` with
stage-specific control dictionaries, calling `_run_arm()` per arm, and
aggregating per-step growth CSVs. The harness's probe computes
finite-horizon directional derivatives via forward-mode AD (JVP through the
full filter map for each tangent, reporting `per_step_log_growth` =
log(‖δₜ‖/‖δₜ₋₁‖) per step and cumulative T=20 amplification).

### Output

`docs/benchmarks/artifacts/dualcap-stage-gain-ledger-20260820/result.json`:
per-arm `{arm_id, controls, per_step_log_growth: [T=20 array],
cumulative_amplification, late_step_mean_growth}` plus CSV
`growth_by_stage.csv` with columns `[time_index, base_reset,
diagonal_uncapped, ..., physical_reference]` where `physical_reference` is
the RK4 transition Jacobian's per-step Lyapunov exponent (measured −0.25 in
steps 8–20 from the 08-01 data).

### Scope Caveat (Critical)

This probe harness propagates particles through the **cubature-GenUT
candidate adapter's raw RK4 transition + reweight**, NOT through the LEDH
invertible particle-flow proposal. The stages it interrogates — Contract-E
reset, dual-cap diagonal/pairwise corrections, trust-region — are literally
the shared code between the cubature and LEDH lanes, so as **mechanism
evidence about those correction/reset stages** the ledger is on-target. But
the LEDH flow step itself (whose per-step gain contribution is unknown and
may damp or amplify) is absent from this measurement, so this is NOT an
end-to-end LEDH-PFPF-OT per-step gain statement.

The 08-01 baseline this extends has the same structure, so the comparison
is internally consistent. A follow-up LEDH-route tangent probe (requiring a
new JVP path through the flow core — not in harness today) is the natural
next validation once the algorithm repair (see Blockers) is complete.

## Execution Record

- **2026-08-20 23:43 UTC**: first attempt failed at GPU memory-growth
  policy initialization (wrapper called policy configuration after
  `_load_context()` built tensors; fail-closed correctly per repo rule).
- **2026-08-20 23:48 UTC**: second attempt failed (policy call at script
  body line 37 still too late; harness module import builds `tf.constant`
  at module load time).
- **2026-08-20 23:52 UTC**: third attempt launched with corrected import
  order (memory-growth policy immediately after `import tensorflow`, before
  any bayesfilter module imports). Task ID `bvc2422x6`, GPU0, expected ~40
  min.
- **2026-08-21 00:12 UTC** (projected): task completed; artifact sits
  unanalyzed at
  `docs/benchmarks/artifacts/dualcap-stage-gain-ledger-20260820/result.json`.

## Blockers (Discovered 2026-08-20, Requires Owner Repair)

During scope verification for this experiment, a **major algorithm-identity
defect** was discovered in the LEDH-PFPF-OT campaign route
(`ledh_pfpf_genut_initial_rqmc_tf.py`), invalidating its interpretation as
"Li(17) particle flow with UKF initialization":

### The Defect

The Austria SIR model callbacks
([ledh_pfpf_genut_model_callbacks_tf.py:448-461](bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py#L448))
wire the LEDH flow's Gaussian linearization inputs as **placeholder
identity matrices** rather than UKF/GenUT sigma-point predicted covariances:

```python
initial_covariance   = lambda theta: tf.eye(18, dtype=theta.dtype)
transition_covariance = lambda theta: tf.eye(18, dtype=theta.dtype)
transition_matrix     = lambda theta: tf.eye(18, dtype=theta.dtype)
```

The transition *mean* is faithful (per-ancestor float64 RK4 via
`sir_score._transition_mean_and_parameter_tangent`). The flow algebra, its
forward log-det Jacobian correction, and all downstream stages (OT reset,
dual-cap, trust-region) are present and correctly wired. But the covariance
structure the flow linearizes against — and the **actual Cholesky
factorization through which pre-flow process noise is injected** ([line
738-741](bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py#L738)) —
is a unit identity in all three slots.

### Consequences

1. **Algorithm-identity drift confirmed.** The N=16128 campaign rows are
   valid finite programs (correct importance weights, no bootstrap-PF
   contamination), but they implement the **identity-covariance variant of
   Li(17) flow**, not the UKF-initialized version. Any result note
   interpretation claiming "LEDH-PFPF-OT with UKF initialization" is false.
2. **Directly implicates the damping problem.** An identity transition
   covariance cannot shrink along the directions where SIR compartment
   dynamics contract (steps 8–20, eigenvalue < 1). A UKF/GenUT-predicted
   covariance would capture that contraction and pull perturbed particles
   back toward the predicted manifold at the physics' rate; `eye(18)` is
   invariant and wildly mis-scaled (raw-count compartment magnitudes,
   ignoring correlation structure). The missing UKF initialization and the
   missing damping mechanism may be **the same defect**.
3. **Blast radius appears Austria-specific.** The same file's LGSSM (line
   249) and SV (288, 327) constructors build covariances from model
   parameters; the identity-stub pattern is isolated to the Austria
   callbacks.

### What Survives

- N=16128 feasibility (memory, streaming transport, while-loop score
  recursion repair, chunk compliance) — these are properties of the
  transport/recursion machinery, not the proposal quality.
- The chaos/perturbation-surface diagnostics (TF32 vs FP32, ordering
  mechanisms, FD validation) remain valid as measurements *of the route
  that ran*, with the open question of how much the degenerate covariance
  contributed to the measured amplification.
- The four-variant table values/gates — allpass their contracts under the
  identity-covariance semantics; reinterpretation after repair is a
  judgment call (bug vs variant framing).

### Owner Repair Scope

Fix the three Austria callback lambdas to return UKF/GenUT sigma-point
predicted covariances (initial: from the model's prior distribution;
transition: propagated per ancestor through the RK4 SIR dynamics or its
cubature approximation) and the RK4 transition Jacobian (or its finite-diff
/ sigma-point equivalent). The `transition_mean` path is already faithful;
the LGSSM/SV lanes provide reference wiring for model-informed covariances.

**Critical note:** because `transition_covariance` feeds the Cholesky that
injects pre-flow process noise, a fix changes the realized particle
distribution, not just the flow's linearization algebra — this is a **new
finite program** requiring its own tuning scope under the LEDH per-scope
rule, and the repaired route's parity/value/score baselines are not
comparable to the identity-covariance campaign's without re-establishing
gates.

## Next Steps After Repair

Once the owner delivers the covariance-repaired LEDH route:

1. **Immediate validation** (before resuming this plan): rerun the N=1008
   and N=4032 smoke rows with the repaired proposal under a fresh tuning
   scope, confirm finite/valid, and establish the repaired route's value
   baseline. The new proposal will have different particle distributions
   and may need ε/balance/ridge retuning.

2. **Analyze the completed stage-gain ledger** (artifact sitting at
   `dualcap-stage-gain-ledger-20260820/result.json`) with the scope caveat
   prominently stated: this measures the correction/reset stages under the
   cubature lane's transition, not end-to-end LEDH. It still answers the
   pairwise-instability question and apportions gain among the shared
   stages.

3. **Build the LEDH-route tangent-growth probe** (new JVP path through
   `batched_ledh_flow_core_tf` required, ~1 day scope) and run a 3-arm
   comparison: (a) repaired UKF-predicted covariance, (b) `eye(18)`
   (identity-covariance baseline for direct A/B against the campaign rows),
   (c) shrinkage blend `λ·predicted + (1−λ)·eye(18)` with λ=0.5. This
   directly measures how much per-step damping the proper initialization
   restores, isolating the covariance input's effect.

4. **Score-accuracy campaign design** — after the above establishes (i) the
   repaired route's baseline and (ii) per-stage gain attribution, design
   the multi-seed variance-scaling ladder (K seeds × N ∈ {1008, 4032,
   16128}, fit SD ∝ N^(−α) per score component, measure whether capped
   pairwise under repaired covariance shows 1/N scaling or degraded
   T-dependence) with the understanding that any damping fix targeting the
   shared reset stage (temporally smoothed self-moments, gain-scheduled
   correction strength, etc.) is a further algorithmic change requiring its
   own scope.

## References

- Perturbation surface diagnostic (TF32 vs FP32 mechanisms):
  `bayesfilter-dualcap-perturbation-surface-localization-diagnostic-2026-08-20.md`
- N=16128 campaign result (identity-covariance variant, pending
  reinterpretation):
  `bayesfilter-genut-sqmc-streaming-n16128-result-2026-08-18.md`
- Score-recursion while-loop repair (enables large-N score evaluation):
  `bayesfilter-score-recursion-while-loop-repair-plan-2026-08-19.md`
- Per-step growth baseline (2026-08-01, N=4032, diagonal vs pairwise):
  `bayesfilter-genut-score-growth-n4032-comparison-result-2026-08-01.md`
- LEDH campaign route (repair target):
  `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py` and
  `ledh_pfpf_genut_model_callbacks_tf.py`

---

**Handoff note for continuation agent:** This plan was authored 2026-08-20
by Fable-5 prior to discovering the covariance-wiring defect during
algorithm-identity verification. The stage-gain ledger artifact completed
and is ready for analysis, but the broader LEDH lane is blocked pending the
owner's repair. When resuming: (1) verify the repair landed (grep for
`eye(18)` in the Austria callbacks — it should be replaced with
model-informed covariances), (2) validate the repaired route's smoke rows
at N=1008/4032 under a fresh tuning scope before interpreting any
prior-campaign results as repaired-route baselines, (3) analyze the
completed ledger with its cubature-lane scope caveat stated, (4) build the
LEDH-route JVP probe for the covariance A/B experiment. The owner's
instinct that led to this finding — "the filter should inherit the model's
contraction when eigenvalue < 1" — is the organizing principle for the
damping mechanism the lane is building toward.
