# Canonical LGSSM Balancing And Kalman Repair Reset Memo

Date: 2026-07-17

Status: `CAMPAIGN_CLOSED_WITH_OPEN_SCIENTIFIC_GAPS`

## State To Preserve

- The LGSSM terminal-balancing and both-marginal fail-closed bug is fixed.
- `balance_steps=50` is the smallest tested schedule passing the frozen
  marginal-only design set and untouched audit set.  It is campaign-specific,
  not a universal or cross-model default.
- The fixed balanced program's manual total score matches same-scalar
  HMC-coordinate FD at the tested `T=2,N=128` point; maximum relative error is
  `2.1596e-10`.
- Fresh paired `T=2,N=1024`, 16-seed Contract E and no-reset comparisons against
  differentiated Kalman are both inconclusive.  All paired reset effects are
  inconclusive.
- Do not claim Kalman equivalence, score failure, reset superiority, HMC
  readiness, leaderboard admission, or production XLA-loop readiness.

## Open Gaps

1. `CE-01`: the production Contract E identity factory is empty.  Current
   LGSSM artifacts are diagnostic-only even though their source/preparation
   identities are recorded.
2. `CE-02`: resolve fresh `T=2` uncertainty with a predeclared precision/power
   design, and run `T=10,50` after a loop-core resource repair.
3. Replace the Python horizon loops in candidate primal and manual JVP with a
   fixed-state `tf.while_loop` body before a serious longer-horizon XLA run.
4. Historical Phase 8 artifacts whose preparation recorded zero but callable
   executed one remain historical; do not relabel them after the caller fix.

## Exact Next Step

Create a focused LGSSM loop-core repair plan.  Its first diagnostic must show
that a `tf.while_loop` implementation preserves the current balanced `T=2`
primal, manual score, telemetry, branches, and preparation identity.  Then run
a bounded `T=10,N=1024` resource pilot without changing the 16-seed
certification contract.  Separately design the `T=2` precision/power extension;
do not infer a required seed or particle count from the observed best mean.

## Anchors

- Plan: `docs/plans/bayesfilter-canonical-lgssm-balancing-kalman-repair-plan-2026-07-17.md`
- Result: `docs/plans/bayesfilter-canonical-lgssm-balancing-kalman-repair-result-2026-07-17.md`
- Phase 1: `docs/plans/bayesfilter-canonical-lgssm-balancing-kalman-repair-phase1-result-2026-07-17.md`
- Phase 2: `docs/plans/bayesfilter-canonical-lgssm-balancing-kalman-repair-phase2-result-2026-07-17.md`
- Active ledger: `docs/plans/bayesfilter-contract-e-active-failure-ledger-2026-07-17.md`
- Artifacts: `docs/benchmarks/artifacts/canonical_lgssm_balancing_kalman_repair_20260717/`
- Run manifest: `docs/benchmarks/artifacts/canonical_lgssm_balancing_kalman_repair_20260717/run_manifest.json`
