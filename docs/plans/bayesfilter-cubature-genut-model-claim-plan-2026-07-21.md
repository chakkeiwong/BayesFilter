# Cubature/GenUT Model-Claim Gap Closure Plan

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_ENGINEERING_ONLY_SV_SCIENTIFIC_CLAIMS_REVOKED`

> **Correction, 2026-07-22:** This plan generated transformed observations
> directly from an iid Normal convenience distribution rather than from the
> declared SV DGP.  That is a planning error.  All target accuracy, bias,
> tuning, score, and SV model-claim conclusions are wrong relative to an SV
> scientific claim and are revoked.  Only finite-program mechanics, replay,
> resource, placement, residual, and same-scalar derivative checks remain
> eligible engineering evidence.  See
> `bayesfilter-exact-sv-nondgp-fixture-demotion-correction-2026-07-22.md`.

## Objective

Close the remaining model-evidence gaps for the experimental Cubature/GenUT
route without promoting it to the canonical/default Contract E route.

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | Does the loop-native candidate compute the declared exact transformed-SV target value and total recursive score accurately enough for a model-specific diagnostic claim, and can that evidence support a fair future comparator? |
| Candidate | `cubature_genut_nonfused_positive_ot_candidate_v1`, repository-registered exact transformed-SV adapter. |
| Independent reference | Same exact transformed-SV target law evaluated by dense fixed-grid quadrature at increasing orders/radii. This is diagnostic/reference code, not the XLA candidate. |
| Comparator | Contract E scalar-SV is a separate finite proposal program. It is comparator-eligible only if target identity, observation transform, stream, horizon, route identity, and score composition are explicitly paired. Otherwise record `BLOCKED_SAME_TARGET_COMPARATOR`, not a ranking. |
| Primary diagnostic criterion | Candidate finite value, recursive score, and per-time increments are finite; same-scalar central-FD checks pass; dense-reference refinement is stable under a predeclared budget. |
| Promotion veto | Target-law mismatch, score/FD failure, nonfinite reset, stale tuning, claim-seed tuning leakage, unregistered identity, negative/nonrepresentable GenUT mass, or comparator mismatch. |
| Continuation veto | Missing target equations, unavailable dense reference, or exhausted bounded compute budget. |
| Nonclaims | No method superiority, exact nonlinear filtering theorem, HMC readiness, leaderboard admission, default promotion, or NAWM result. |

## Skeptical Plan Audit

1. **Comparator equivalence risk.** Contract E--TP's proposal and continuation
   chart are not the same finite program as Cubature/GenUT. The plan therefore
   forbids direct ranking unless a shared target/stream identity is proven.
2. **Reference bias risk.** A dense finite grid is not an exact oracle unless
   order/radius refinement stabilizes. We report refinement only and use a
   predeclared value/score budget.
3. **Tuning leakage risk.** Controls are selected on disjoint calibration and
   validation seeds. Claim seeds are untouched and used only after controls are
   frozen.
4. **Score risk.** Recursive score is computed by the candidate's tangent path;
   finite differences are validation at representative coordinates only, never
   the runtime score.
5. **Backend risk.** The candidate claim path is float32/TF32 GPU/XLA with no
   NumPy or host conversions in the traced core. Dense reference and reporting
   may use diagnostic host operations only.
6. **Horizon risk.** `T=2` and `T=10` nominate mechanics; `T=50` is the first
   target-horizon diagnostic for this scope. Prefix success cannot be promoted
   to full-horizon readiness.

Audit decision: `PASS_WITH_STRICT_NONCLAIMS`. The exact-SV target-bound phase
may run, but it cannot emit a leaderboard or default claim.

## Phases

### Phase 1: Target And Reference Contract

Freeze exact transformed-SV equations, chart, observations, fixed streams,
target horizon, dense-grid orders/radii, and candidate identity. Verify the
candidate's observation density and stationary initial law against the model
source equations.

### Phase 2: Scope-Specific Tuning

Tune `epsilon`, Sinkhorn steps, ridge, and the declared residual/design controls
on disjoint calibration/validation streams. The tuning loss is a declared
combination of dense-reference value/score discrepancy, recursive-FD parity,
and reset/marginal validity. FD is evaluated only at representative points;
the selected candidate score remains recursive.

### Phase 3: Untouched Replicated Claim Diagnostic

Run `T=2,10,50` with at least 16 untouched seeds at frozen controls. Preserve
per-seed value, score, increments, residuals, finite/placement status, and
manifest metadata. Compute descriptive means, standard deviations, paired
confidence intervals against the dense reference, and score variance. These
are diagnostic unless the predeclared uncertainty gate is met.

### Phase 4: Comparator Audit

Attempt a Contract E scalar-SV pairing only after checking target/observation
measure, parameter chart, horizon, streams, route identity, and score total
derivative. If any field differs, emit a blocked comparator record and do not
rank methods. A same-target Contract E comparison requires a fresh paired
artifact, not reuse of historical Contract E rows.

### Phase 5: Predator-Prey Entry Gate

Inspect the existing predator-prey target and score owners. Implement an
adapter only if its transition, observation, initial law, and total tangent
are explicitly available. Otherwise preserve the existing blocker and do not
force a generic adapter.

### Phase 6: Leaderboard/Default Review

Only rows passing target-horizon, tuning, score, identity, comparator, and
uncertainty gates may be nominated. Conduct a final completeness audit and
retain experimental status unless every required row passes.

## Compute And Artifacts

Use fresh roots under
`docs/benchmarks/artifacts/cubature_genut_model_claim_20260721/`. Each phase
records command, commit, environment, seeds, dtype/TF32/XLA, memory policy,
wall time, tuning scope, and artifact hashes. Failed attempts are preserved.
