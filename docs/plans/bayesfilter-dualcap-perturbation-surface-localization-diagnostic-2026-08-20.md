# Perturbation-Surface Localization Diagnostic (Dual-Cap + Trust-Region Score Lane)

Date: 2026-08-20. Owner-directed ("we should do this while waiting").
Device: GPU0 RTX 5080 (campaign occupies GPU1). Env `tftwogpu`.

## Question

Which surface converts a tiny arithmetic perturbation (TF32 on/off) into
trajectory-level divergence on the repaired-permutation trust-region route:
(1) Hilbert-permutation order flips, (2) validity branch flips, or (3) smooth
nonlinear amplification with no discrete flip?

## Method

Run the exact campaign evaluator (N=4032, repaired_permutation, trust_region,
streaming, seed 97701) twice in separate processes on GPU0: arm A TF32 on,
arm B TF32 off. Dump per-step histories the route already returns:
`ancestry_selected_row_identities` [T,N], `likelihood_increment`,
`particle_mean`, `particle_scale`, `hilbert_tie_count`,
`ancestry_permutation_valid`, plus value/score. Compare stepwise:
first step with any identity mismatch vs first step with continuous-state
divergence above FP32 noise.

## Interpretation table (predeclared)

- Continuous divergence strictly BEFORE first permutation flip -> surface 3
  seeds the divergence; permutation flips are a downstream symptom.
- Permutation flip at step k with continuous agreement (<=1e-5 rel) before k
  -> surface 1 is the trigger; stabilized ordering is the right remedy line.
- Any validity-branch flip -> surface 2 contribution; count it.
- One seed, one perturbation pair: mechanism localization only. No accuracy,
  ranking, or route-quality conclusion. Different device class than campaign
  rows (5080 vs 4080S): fine for mechanism, recorded in output.

## Budget

Two N=4032 rows on GPU0 (~6-8 min each) + comparison script. Debug-tier.
Artifacts to scratchpad + summary JSON under docs/plans/ if decisive.

## Result (2026-08-20, executed)

Arms completed on GPU0 / RTX 5080, `tftwogpu`, TF32 verified on/off.
Values `-681.3306` / `-681.4280` (rel `1.4e-4`); scores fully scrambled
(`(319,-142,5.7)` vs `(101,-89,7.6)`), reproducing the campaign chaos on a
second device class.

Stepwise localization:

| Step | Permutation flips (of 4032) | Max continuous rel-diff |
|---:|---:|---:|
| 0 | 0 (identity by construction) | `9.8e-4` <- FIRST DIVERGENCE |
| 1 | 4018 | `1.6e-2` <- first perm difference |
| 2-19 | ~4025-4032 | `2-6e-2` (saturated) |

Zero Hilbert ties in both arms at every step; zero validity-branch
mismatches. Rank-displacement analysis at step 1+: mean displacement
375-1280 ranks, 94-99% of identities moved more than 8 ranks — the two
arms order *genuinely different clouds*, they do not near-tie-swap similar
clouds.

**Verdict (predeclared table): surface 3.** Continuous state diverges at
step 0 — before any resampling decision exists — at `~1e-3` relative,
i.e. at TF32's own epsilon (`5e-4`) amplified once through the step-0
flow/reset linear algebra. The permutation differences from step 1 onward
are a downstream symptom of already-different particle clouds, not a
near-tie ordering instability. Chaotic amplification then saturates
divergence at the few-percent level within two steps.

**Implications.**
1. Stabilizing the Hilbert ordering (robust tie-breaking, soft sorting)
   would NOT have prevented this divergence: the trigger precedes ancestry.
2. The TF32-scale (`5e-4`) perturbation enters through matmuls in the
   flow/reset/trust-region path and is immediately macroscopic. For
   *trajectory reproducibility*, the effective lever is the precision of
   the continuous linear-algebra path, not the discrete ordering.
3. NOT CHECKED: whether FP32-epsilon-scale (`6e-8`) perturbations behave
   the same way. TF32's epsilon is ~8000x larger; at true FP32 scale the
   near-tie flip mechanism could still be the first mover. A follow-up
   with a 1-ulp-scale perturbation (both arms TF32-off, differing by a
   benign reduction-order change) would discriminate. This matters because
   the campaign's frozen policy runs TF32 on.
4. Scope: one seed, one perturbation pair, RTX 5080. Mechanism
   localization only; no accuracy, ranking, or route-quality conclusion.

## Follow-up: FP32-scale arms (2026-08-20, executed)

Arms: dense vs streamed transport, BOTH TF32-off, same seed/route/reset —
a reduction-order-only perturbation at FP32 epsilon scale (primitive parity
`~5e-8`, measured 2026-08-19). GPU0 / RTX 5080.

| Step | Perm flips | Median rank disp | Frac >8 ranks | Max cont rel-diff |
|---:|---:|---:|---:|---:|
| 0 | 0 | — | — | `1.3e-6` (below threshold) |
| 1 | 3498 | **2** | **0.9%** | `7.3e-4` |
| 2+ | ~all | grows | grows | saturates `~1e-2` |

Contrast with the TF32-scale arms: there, step-1 median displacement was
115 ranks (94% moved >8) — genuinely different clouds being ordered. Here,
median displacement is 2 ranks with 99% moving <=8 — the signature of
**near-tie local swaps** while ordering nearly identical clouds (step-0
agreement `1.3e-6`).

**Verdict: perturbation-scale-dependent mechanism.**
- TF32 scale (`5e-4`): surface 3 (smooth chaos) — clouds differ
  macroscopically before any ordering decision.
- FP32 scale (`6e-8`): mixed, with the near-tie swap mechanism (surface 1)
  active and plausibly dominant at onset: entering step 1 the clouds agree
  to `~1e-6`, the Hilbert sort produces thousands of local neighbor swaps,
  and each swap discretely reassigns that particle's RQMC point and reset
  slot — a jump of order the inter-particle spacing, not of order the
  seed perturbation. Observed step-1 divergence (`7.3e-4`) is far above
  what one step of smooth amplification of `1e-6` would suggest,
  consistent with swap-driven injection. Strict causal separation is not
  possible at per-step granularity (both fire within step 1); the rank-
  displacement signature is the discriminating evidence.

**Implications for the score-accuracy lane.**
1. In the FP32/TF32-off regime, ordering stabilization (robust or smoothed
   rank assignment near ties) is a justified investigation: it targets the
   apparent dominant onset amplifier. It cannot remove the `1e-6`-scale
   smooth seed, so it extends trajectory agreement rather than making runs
   bit-stable; the eventual chaotic divergence remains.
2. Under the current frozen TF32-on policy, ordering stabilization is NOT
   the binding constraint (surface 3 dominates at that perturbation scale).
   Reproducibility investment ordering: precision of the continuous path
   first, then ordering stabilization.
3. Scores were realization-scrambled in BOTH regimes (TF32-off did not
   rescue score reproducibility). Expectation-level multi-seed evidence
   remains mandatory for any score-accuracy claim.
4. Scope: one seed, one perturbation pair per scale, RTX 5080, N=4032.
   Mechanism localization only.
