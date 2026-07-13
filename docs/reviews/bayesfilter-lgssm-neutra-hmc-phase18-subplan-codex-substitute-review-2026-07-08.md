# Codex Substitute Review: LGSSM NeuTra Phase 18 Subplan

Date: 2026-07-08

## Scope

Claude review remains unavailable because the attempted Phase 17 Claude review
gate was rejected by the sandbox approval reviewer as an external-service
disclosure risk. No workaround is attempted here.

This is a same-foreground Codex substitute review of:

`docs/plans/bayesfilter-lgssm-neutra-hmc-phase18-fixed-transport-mechanics-compile-subplan-2026-07-08.md`

It is weaker than an independent Claude review and cannot authorize human,
runtime, product, default-policy, or scientific-claim boundaries.

## Findings

No blocking issue found for implementation planning.

The subplan is consistent with the Phase 17 handoff:

- It uses the Phase 17 frozen payload and current target/adapter signatures as
  the only admissible baseline.
- It names the actual remaining blocker: the fixed-transport HMC mechanics path
  rejects `use_xla=True` because the base generic SSM adapter currently
  advertises `xla_hmc_ready=False`.
- It requires an explicit XLA-HMC readiness opt-in rather than globally
  promoting all generic SSM adapters.
- It keeps Phase 18 to mechanics compile diagnostics only: no HMC chains,
  sampling/tuning, external samples, training, optimizer updates, or
  `jit_compile=false`.
- It requires trusted GPU execution for runtime compile evidence and records
  timing/size proxies.
- It keeps fallback/GradientTape authorities fail-closed.

## Local Checks

- Section coverage check: passed.
- `git diff --check` on the Phase 18 subplan and updated program/runbook/result
  docs: passed.

## Boundary Verdict

The Phase 18 subplan is safe to implement narrowly. Trusted GPU runtime
execution still requires the normal escalated command approval and must not be
treated as authorized by this substitute review alone.

VERDICT: AGREE
