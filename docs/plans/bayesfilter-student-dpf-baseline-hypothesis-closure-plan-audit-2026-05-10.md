# Audit: student DPF baseline hypothesis-closure plan

## Date

2026-05-10

## Scope

This audit reviews
`docs/plans/bayesfilter-student-dpf-baseline-hypothesis-closure-plan-2026-05-10.md`
before execution.

## Disposition

Approve with controls.

The plan addresses the correct remaining gaps from the prior student-baseline
cycle and keeps the work out of the DPF monograph writing lane.  It is safe to
execute if each phase records structured results and avoids in-place edits to
vendored student code.

## Required controls

1. Use bounded kernel PFF commands.  The previous `test_kernel_pff.py` run did
   not complete, so H2 must use individual tests and timeouts.
2. Treat advanced bootstrap-PF diagnostics as diagnostics, not as the primary
   Kalman reference result.  The adapters currently report Kalman log
   likelihood as the normalized `log_likelihood`.
3. Do not force MLCOE nonlinear support.  If the wrapper is nontrivial, record
   `blocked_missing_assumption` rather than expanding scope.
4. Avoid new dependency installation.  Missing plotting or optional packages
   should become blockers.
5. Keep generated outputs small.  Store summaries and bounded panel JSON only.
6. Stage only student-baseline files.  The working tree contains unrelated
   monograph reset/planning files.

## Missing point added by this audit

The H1 stress report should explicitly distinguish:

- reference consistency of the Kalman paths;
- particle-filter ESS/runtime diagnostics from `advanced_particle_filter`;
- unsupported MLCOE particle diagnostics if the adapter does not expose a PF
  path in this cycle.

## Decision

Proceed automatically through H0-H4 if phase gates are met.  Stop only if a
phase requires production code edits, vendored-code patches, or broad
environment changes.
