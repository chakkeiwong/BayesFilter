# Phase 8 C5 calibration freeze subplan

Date: 2026-08-31  
Status: `CLOSED_PASS_NO_HMC_OR_POSTERIOR_PROMOTION`

Parent program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`

## Purpose and boundary

C5 converts the completed q=20 calibration receipts into a bounded protocol
for any later Phase 9 tuning.  It is metadata-only: it performs no target
training, HMC, posterior sampling, whitening calculation, or consumption of a
reserved Phase 9 random stream.  A frozen representative is a confirmation
configuration, not a claim that the transport approximates the posterior.

The decision is made only from the C2, C3A, C3B, C4A, and C4B receipts.  C4B's
K=4 arm is a valid implementation/resource diagnostic, but it is not retained
when the evidence cannot establish a benefit over the lower-cost K=2 route.

## Evidence contract

| Item | Frozen rule and role |
|---|---|
| Target identity | q=20 SSL-LSTM bridge signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`; all prerequisite manifests must agree |
| Eligibility | Candidate must be hard-valid, have a current reliability receipt, and have no checkpoint or target-status failure |
| K=2 selection | Among compact `(16,16)` rows that pass the C2 paired start-to-final rule on both roots, use the predeclared lower mean paired held-out reverse-KL change as an operational nomination rule; this is not a scientific ranking |
| Capacity | Prefer the compact family over `(32,32)` by predeclared parameter parsimony when no uncertainty-supported difference exists |
| Ladder | Use `L3=(0,.5,1)` when C3B supplies no uncertainty-supported advantage for `L5`; fewer bridge levels win by predeclared parsimony |
| Lineage policy | Pure continuation; neither C3A nor C3B established a consistent branching-diversity advantage |
| Chart selection | Fixed, state-independent uniform `gamma=(1/K,...,1/K)`; state-dependent selection is forbidden |
| K=4 joint arm | Do not retain for Phase 9. C4A/C4B establish feasibility only; C4B endpoint objectives use unpaired banks and have opposite signs across rows, while the arm adds quadratic cross-density work |
| Phase 9 identity | Freeze the protocol and scope description only. Rebuild/tune any confirmation chart under a fresh repository-issued scope artifact; do not silently promote a calibration checkpoint |
| Nonclaims | No whitening, mode discovery, posterior correctness, convergence, HMC readiness, statistical superiority, architecture ranking, or high-dimensional scaling claim |

## Inputs

The freeze script verifies these exact inputs and records their SHA-256 hashes:

- C2 manifest and result;
- C3A and C3B result notes;
- C4A manifest/result;
- C4B manifest/result;
- this subplan and the Phase 8 calibration subplan.

The script fails closed on a missing file, status/signature mismatch, failed
hard screen, or an output directory that already exists.

## Selection calculation

For each eligible compact C2 row, let

\[
  \Delta_r = \bar{\ell}_{r,32}-\bar{\ell}_{r,0}
\]

be its paired held-out reverse-KL change on root `r`.  Both roots must have a
finite two-sided interval whose upper endpoint is below zero.  The operational
nomination value is the arithmetic mean of the two `Delta_r`; it is used only
to choose one representative after the predeclared parsimony screen.  The
freeze artifact must label it as an operational choice, not evidence that one
learning rate is scientifically better.

The C4B K=4 result is checked for implementation/resource validity but is not
eligible for retention because its independent and joint endpoint banks are
different and its two row-level objective contrasts disagree in sign.  This is
an evidence insufficiency and cost decision, not a claim that joint mixture
training is invalid.

## Procedure

1. Run the standard-library freeze evaluator with a fresh output directory.
2. Validate all prerequisite statuses, target/backend identities, manifest
   hashes, and hard-screen fields.
3. Apply the selection calculation above and write an immutable JSON freeze
   manifest containing the selected K=2 protocol, the explicit K=4 non-retention
   decision, all input hashes, Git/environment provenance, and nonclaims.
4. Write a result note with decision and inference-status tables and a
   between-phase repair/red-team section.

## Skeptical pre-execution audit

| Risk | Check and disposition |
|---|---|
| Post-hoc architecture ranking | Selection uses the already declared compact-family/parsimony rule and C2 paired nomination statistic; the artifact calls it operational only. |
| Unpaired C4B objectives treated as paired | The K=4 branch is explicitly non-retained; no objective difference enters selection. |
| Stale or mixed target receipts | Every input target signature, strict backend, and status is checked before writing. |
| Confirmation contamination | The evaluator reads calibration manifests/results only and creates no TensorFlow random state. |
| Hidden gamma adaptation | The frozen protocol records fixed uniform gamma and rejects state-dependent selection. |
| Calibration checkpoint silently promoted | The artifact requires fresh Phase 9 scope/tuning and labels calibration checkpoints non-claim-bearing. |
| Old blocker text mistaken for current state | The parent and execution records are refreshed in the same closeout. |

Audit verdict: `PASS_FOR_METADATA_ONLY_C5_FREEZE`.

## Budget, repair, and stop rules

This subplan has no GPU allocation and a 120-second wall cap for the evaluator.
A missing or inconsistent receipt is a hard stop requiring document repair; it
does not justify guessing a representative. A localized JSON/provenance repair
may be rerun in a fresh directory without changing the selection rule. Any
change to target, ladder, K, gamma, tuning identity, or confirmation streams
requires a new reviewed Phase 9 subplan.

## Exit

The exit artifact is
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c5-freeze/attempt-02/freeze_manifest.json`,
paired with
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c5-freeze-result-2026-08-31.md`.
The only permitted next step is a separately audited Phase 9 tuning and
sequential-validation subplan.  A C5 pass does not open HMC by itself.

## Execution closeout, 2026-08-31

The initial pre-closeout evaluator was preserved in `attempt-01`. A fresh
provenance rerun (`attempt-02`) passed in `0.016574528999626637` seconds and
wrote the terminal manifest with the finalized subplan hash. It selected
`phase8-k2-compact-high-l3-pure` and recorded K=4 as
`NOT_RETAINED_FOR_PHASE9`. The result note is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c5-freeze-result-2026-08-31.md`.
The next active boundary is a fresh Phase 9 tuning/validation subplan; no
confirmation stream has been consumed.
