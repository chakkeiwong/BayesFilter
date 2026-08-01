# LEDH Per-Model Scope Tuning Master Program

Date: 2026-07-19  
Status: `ACTIVE_CORRECTED_DESIGN`

## Research Intent

Every claim-bearing LEDH run must tune its own execution scope. A scope binds
the model/target, route/reset family, horizon and prepared-data regime,
particle count, dimensions, dtype/backend, chunk policy, and route-specific
control family. No setting is universal. A selected setting from another scope
is a warm-start candidate only.

The primary question for each scope is: does a route-specific offline tuning
procedure select fixed controls that pass that scope's numerical, value, score,
and engineering gates on disjoint validation data and then on untouched claim
data? Runtime adaptation inside HMC remains forbidden.

## Evidence Contract

Each model-scope phase must create:

1. a repository-issued `LEDHTuningScope` identity;
2. a model-specific calibration and validation split;
3. an untouched claim split;
4. a route-specific control grid or analytical selection rule;
5. candidate artifacts and a selected-control artifact bound to the scope;
6. a claim run that rejects any missing or mismatched tuning artifact; and
7. a result note separating engineering, numerical, and scientific evidence.

For streaming OT routes, tune cheaper terminal balancing before annealed
Sinkhorn initialization. For Contract E--TP routes, tune the applicable teacher
quadrature, continuation quadrature, lookahead, feature basis/capacity,
row-scale policy, and ridge/KKT policy. Do not call these Sinkhorn controls.

A claim failure preserves the claim data and triggers a new tuning phase using
fresh tuning data. It does not authorize tuning on the failed claim data,
threshold relaxation, setting transfer, skipping another model, or rejection
of the research direction.

## Scope Ledger

| Phase | Model/scope | Control family | Tuner state | Required action |
| --- | --- | --- | --- | --- |
| 1 | LGSSM T=10, N=1024, TF32/XLA | Sinkhorn + terminal balance | Implemented and tuned | Preserve `(20,3)` only for this exact scope |
| 2 | LGSSM T=50, N=1024, TF32/XLA | Sinkhorn + terminal balance | Scope tuning and untouched claim passed | Preserve `(20,8)` only for scope `451d361a...f2d`; do not transfer it |
| 3 | Latent pre-clipping SIR for each declared horizon/N | Sinkhorn, balance, epsilon/scaling/ridge | Adapter required | Implement TF32/direct-marginal scope tuner, then tune each scope |
| 4 | Actual SV for each declared horizon/data regime | TP feature/chart controls | Tuner required | Implement model-specific TP tuner and claim gate |
| 5 | Generalized SV for each declared horizon/data regime | TP structural feature/chart controls | Tuner required | Implement model-specific TP tuner and claim gate |
| 6 | KSC-SV for each declared horizon/data regime | TP structural feature/chart controls | Tuner required | Implement model-specific TP tuner and claim gate |
| 7 | Predator-prey for each declared horizon/data regime | TP predator-prey feature/chart controls | Tuner required | Implement model-specific TP tuner and claim gate |
| 8 | Paired all-model chart/leaderboard | Consumes exact scope artifacts only | Blocked on Phases 3-7 | Reject missing/stale/cross-scope selections and run paired claims |

## Skeptical Audit

Verdict: `PASS_AFTER_CORRECTING_CROSS_SCOPE_TRANSFER`.

- Wrong baseline: the old program promoted a T=10 setting into a T=50 claim.
  Corrected: T=50 has its own tuning phase.
- Proxy promotion: marginal gates alone cannot establish score correctness.
  Corrected: each model phase must also retain its applicable value/score oracle
  or comparator gates; selection metrics and claim metrics are separately
  declared.
- Hidden assumption: “same model” was treated as “same numerical regime.”
  Corrected: horizon, data regime, N, dtype, and route are scope identity fields.
- Unfair comparison: TP routes were marked incompatible instead of receiving
  their own tuning vocabulary. Corrected: every route gets a model-specific
  tuner.
- Missing continuation rule: one failed claim stopped later model work.
  Corrected: it triggers repair for that scope while independent phases continue
  unless total budget or implementation validity is lost.
- Artifact weakness: callers could reuse a selected pair by convention.
  Corrected: repository-owned scope hashes must match exactly and fail closed.

## Phase Protocol

For every phase:

1. audit the model route and enumerate material tunable controls;
2. record provenance, justification, failure mode, and early diagnostic for
   every candidate grid/default;
3. implement or verify the scope-specific tuner and exact artifact guard;
4. tune on calibration/validation only, using cheaper controls first where a
   cost ordering exists;
5. freeze the selected control vector;
6. execute the untouched claim run for the exact same scope;
7. on candidate/claim failure, preserve evidence and open a fresh repair tuning
   phase rather than stopping the whole program; and
8. write the result and refresh the next model-specific subplan.

## Completed LGSSM T=50 Phase

LGSSM `T=50,N=1024,float32/TF32,XLA,K=1024` was independently tuned with
calibration seeds `81800..81807`, validation seeds `81808..81815`, and untouched
claim seeds `81820..81835`. The cheaper-first ladder rejected `(20,3)` and
`(20,5)`, selected `(20,8)`, and passed the exact-scope claim. This result is
closed in
`bayesfilter-ledh-lgssm-t50-scope-tuning-result-2026-07-19.md`; it says nothing
about another horizon, prepared-data regime, model, or route.

## Immediate Next Phase

Implement and audit the latent pre-clipping SIR scope tuner. Before any claim,
declare the exact SIR horizon/prepared-data regime and tune that scope's own
Sinkhorn, balance, epsilon/scaling, and prepared-ridge controls using disjoint
calibration/validation data. The LGSSM grids and selected pairs are warm-start
hypotheses only. Actual SV, generalized SV, KSC-SV, and predator-prey remain
blocked on their separate Contract E--TP tuner and independent scope tuning.
