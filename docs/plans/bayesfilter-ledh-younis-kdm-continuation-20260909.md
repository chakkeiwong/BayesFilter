# KDM continuation after branch consolidation

Date: 2026-09-09
Branch: `kdm-total-score-continuation-20260909`, based on `284d65fb`

## Scope and current evidence

Continue Steps 5 and 6 of
`bayesfilter-ledh-younis-kdm-phase4b-resampling-reference-plan-2026-09-08.md`.
The user has authorized continued implementation, audit, and bounded local
execution. The scientific question remains whether the declared
`RESKDM-IWSG-FINITE` score reduces error against the exact model score. It is
not a replacement derivative for `ATOM-FINITE`.

The current checkout contains the shared analytical LEDH executor and the
Phase 4A/4B extensions. The old Section 3.6 worktree has a different producer
and must not be copied over this implementation. Its tests do not establish
conformance of this branch. Three unrelated untracked SQMC/capability files
are outside this continuation's edit set.

On this branch, 41 focused CPU reference tests passed in 98.07 seconds with
`CUDA_VISIBLE_DEVICES=-1`. Four Step-5 GPU/XLA cells also passed for
`D=2,N=8,T=2`: float64 and float32 without TF32, each with responsibility-mean
and selected-label covariance marks. Receipts are in
`docs/benchmarks/artifacts/ledh_younis_kdm_phase4b_20260909/attempt01_*`
through `attempt04_*`. These checks establish local derivative and execution
consistency, not score quality.

## Immediate audit: validity under XLA

XLA reports that it ignores graph assertion operations. Inspection located
the assertion at the higher-moment correction; the same boolean is retained
in the trace and included in the Phase 4B endpoint's returned `valid` flag.
Thus the warning alone does not establish a missing guard at this consumer.
Before a campaign, exercise actual invalid inputs through the compiled
endpoint and require host-side rejection through its returned flags or a
domain error.

Extend the existing smoke with negative controls for non-SPD bandwidth,
asymmetric bandwidth tangent, invalid component label, nonfinite proposal,
out-of-domain stratified uniform, and an invalid higher-moment trajectory.
The positive-input checks must still pass. Preserve every negative-control
outcome in the receipt. No numerical formula, cap, ridge, or accepted value
is changed by these observability checks.

Evidence contract: the question is whether this specific compiled consumer
can silently accept these invalid inputs. Its comparator is the declared
input domain and the already-passing valid fixture. All negative controls
must reject, and the valid fixture must retain derivative/replay/recurrence
parity. Any accepted invalid input is a repair trigger and blocks the score
campaign until fixed. Timing is explanatory only. This does not certify every
possible model or every canonical consumer.

Use the same idle physical RTX 4080 SUPER, GPU ordinal 1, TensorFlow
`tftwogpu`, growth enabled before tensor initialization, XLA on, TF32 off.
At most four additional smoke cells and two localized repair retries are
allowed, with a five-GPU-minute total ceiling. Every run uses a fresh
directory under the existing dated root; no prior result is overwritten.

## Step 6 preparation and execution

After the validity controls pass, implement the two-dimensional campaign
already specified by Step 6. A campaign amendment must freeze the exact
runner, settings, calibration/pilot/validation seeds, power calculation,
conditional heuristic comparisons, total budget, and stop conditions before
its first score-quality run. The existing ceiling is 45 GPU minutes over
`(N,T)=(32,5)` and then `(64,20)`, including compile, calibration, pilot,
validation, and localized retries. The second scope is conditional on the
remaining budget and implementation validity; failure to improve is not a
continuation veto.

The baseline calls the public canonical analytical endpoint with Contract-E,
diagonal and pairwise GenUT corrections, and both caps. The exact matrix
Kalman recursion judges score error. The fixed-stream bootstrap and freshly
calibrated Phase 4A route remain heuristic comparators. A nonexistent
control-variate implementation is reported as absent.

The campaign must preserve total derivatives through all declared factors,
raw IWSG weights, covariance marks, and reset feedback. XLA validity flags
must be consumed on the host for every evaluated path. A successful smoke or
an apparently lower sample MSE cannot promote a method.

## Skeptical audit before execution

The bounded validity probe passes audit: its baseline is the input-domain
contract; every control answers the stated guard question; it preserves the
accepted mathematical program; it has an explicit small budget; and it tests
the actual compiled endpoint rather than inferring behavior from an assertion
in source. Input-domain changes are test controls, not tuning parameters.

The campaign audit remains required before launch. In particular, check that
its power calculation matches the *ten-percent improvement* decision, that
both conditional score groups have adequate uncertainty reporting, that no
scope inherits promoted settings from the scalar Phase 4A experiment, and
that a projected over-budget run stops before consuming validation seeds.
The full-rank Gaussian fixture cannot establish degenerate DSGE support.

The current LaTeX note is
`docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex`; it is present
locally but was omitted from the consolidation commit because its directory
is ignored. Restore its TeX/Bib sources to version control when recording the
audit. The document's claim that repaired GPU smokes remain pending must be
updated from the new receipts. Its historical Section 3.6 statement must stay
distinct from the current shared-executor call chain.

## Required handoff

Record engineering checks, numerical validity, and scientific inference
separately. Include the executed commands, receipts, remaining budget,
mathematical/document discrepancies, decision, and next justified action.
Keep compact evidence cited by that result in Git; intermediate outputs
remain ignored. Preserve the other agents' unrelated work.
