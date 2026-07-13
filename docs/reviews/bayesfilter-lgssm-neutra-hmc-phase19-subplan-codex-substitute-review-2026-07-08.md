# Codex Substitute Review: LGSSM NeuTra Phase 19 Subplan

Date: 2026-07-08

## Scope

Claude review remains unavailable because the earlier Claude review gate was
rejected by the sandbox approval reviewer as an external-service disclosure
risk. No workaround is attempted here.

This is a same-foreground Codex substitute review of:

`docs/plans/bayesfilter-lgssm-neutra-hmc-phase19-cpu-multicore-chain-harness-subplan-2026-07-08.md`

It is weaker than an independent Claude review and cannot authorize human,
runtime, product, default-policy, or scientific-claim boundaries.

## Findings

No blocking issue found for Phase 19 planning.

The subplan is consistent with the Phase 18 handoff:

- It uses the Phase 17 payload and Phase 18 mechanics compile diagnostic as
  required provenance.
- It keeps chain/sample generation CPU-hidden with `CUDA_VISIBLE_DEVICES=-1`.
- It forbids `jit_compile=false`, GPU sample generation, NeuTra training,
  optimizer updates, and Phase 20 reference validation.
- It treats any worker/HMC smoke as harness evidence only, not posterior
  validation, HMC convergence, sampler quality, production readiness, or
  scientific validity.
- It requires worker metadata: seeds, worker indexes, process ids, return
  codes, environment, and artifact paths.
- It provides exact local checks, a bounded smoke command, and post-smoke JSON
  validation commands.

## Local Checks

- Section coverage check: passed.
- `git diff --check` on the Phase 19 subplan and updated program/runbook/result
  docs: passed.

## Boundary Verdict

The Phase 19 subplan is safe to implement narrowly. It does not authorize Phase
20 reference validation or any posterior/HMC-readiness claim.

VERDICT: AGREE
