# Audit: student DPF EDH/PFPF adapter-spike plan

## Date

2026-05-11

## Scope

This audit reviews:

- `docs/plans/bayesfilter-student-dpf-baseline-edh-pfpf-adapter-spike-plan-2026-05-11.md`;
- the MP4 readiness-review result;
- the student-baseline reset memo.

The audit is written as if by a second developer before execution.

## Disposition

Approve with controls.

The plan is the correct follow-up to MP4.  It tests the selected EDH/PFPF-EDH
candidate without expanding into kernel PFF, neural DPF, OT resampling, HMC, or
production BayesFilter work.

## Required Controls

1. Keep the phase bounded.  Use the moderate nonlinear Gaussian range-bearing
   fixture, short horizon, no more than 64 particles, and no more than 10 flow
   steps for the first spike.
2. Keep bridge code adapter-owned.  Reuse existing MP2 model-bridge patterns;
   do not modify vendored student code.
3. Treat failures as evidence.  If either implementation fails due to model
   contract mismatch, return `blocked_missing_assumption` or a structured
   failure record rather than widening scope.
4. Do not compare likelihoods as primary evidence.  Use latent-state RMSE,
   final-position error, finite-output checks, ESS where exposed, resampling
   count where exposed, and runtime.
5. Preserve missing metrics as null/diagnostics.  Do not fabricate ESS or
   resampling semantics if an implementation does not expose them.
6. Do not run kernel PFF, stochastic flow, DPF, dPFPF, neural OT, differentiable
   resampling, HMC, MLE, notebooks, or plotting workflows in this cycle.
7. Preserve lane separation.  Do not edit or stage monograph-lane files,
   production `bayesfilter/` files, `docs/references.bib`, or vendored student
   snapshots.

## Missing Points Added by Audit

The result report should explicitly state:

- the exact command and working directory;
- observed Python/NumPy/TensorFlow/TFP versions;
- source snapshot commits;
- whether each run used a reduced fixture rather than the full MP2 horizon;
- whether the final decision justifies replication, blocks on assumptions, or
  excludes the path due to runtime/numerics.

## Decision

Proceed automatically through S0-S3 if each phase satisfies its exit gate.
Stop for direction only if:

- either selected implementation cannot produce a structured result or blocker;
- the runner needs vendored-code edits or production-code imports;
- final staging cannot be separated from unrelated monograph work.
