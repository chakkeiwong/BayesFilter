# Codex Substitute Review: LGSSM NeuTra Phase 20 Subplan

Date: 2026-07-08

## Scope

Claude review remains unavailable because the earlier Claude review gate was
rejected by the sandbox approval reviewer as an external-service disclosure
risk. No workaround is attempted here.

This is a same-foreground Codex substitute review of:

`docs/plans/bayesfilter-lgssm-neutra-hmc-phase20-lgssm-reference-validation-subplan-2026-07-08.md`

It is weaker than an independent Claude review and cannot authorize human,
runtime, product, default-policy, or scientific-claim boundaries.

## Findings

No blocking issue found for Phase 20 planning.

The subplan is consistent with the Phase 17-19 gated handoff:

- It treats the exact LGSSM reference posterior as the comparator rather than
  Phase 18 mechanics compile or Phase 19 worker metadata.
- It keeps chain/sample generation CPU-hidden with `CUDA_VISIBLE_DEVICES=-1`.
- It forbids `jit_compile=false`, GPU sample generation, NeuTra training, DSGE
  work, default-policy changes, and scientific/product promotion.
- It requires full-chain diagnostic authority to be scoped to Phase 20 rather
  than silently inferred from target-only XLA mechanics readiness.
- It separates promotion vetoes from explanatory diagnostics and says R-hat/ESS
  are unavailable unless enough chain/sample evidence exists.
- It provides exact local checks, a bounded validation command, post-run JSON
  validation, artifact requirements, and stop conditions.

## Local Checks

- Section coverage check: passed.
- `git diff --check` on the Phase 20/21 subplans and updated program/runbook/
  ledger docs: passed.

## Boundary Verdict

The Phase 20 subplan is safe to keep as the successor validation gate after
Phase 19 passes. It does not authorize Phase 19 to run posterior validation,
and it does not authorize broad HMC, product, default, nonlinear SSM, DSGE/c603,
or scientific claims.

VERDICT: AGREE
