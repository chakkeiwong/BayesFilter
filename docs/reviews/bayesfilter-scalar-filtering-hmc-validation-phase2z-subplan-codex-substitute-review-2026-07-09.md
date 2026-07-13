# Phase 2Z Subplan Local Substitute Review

Date: 2026-07-09
Reviewer: Codex local substitute reviewer
Review strength: weaker than full Claude material review

## Claude Availability

Claude material review remains unavailable in this session because the local
review gate was blocked by the approval layer for external transfer of private
repository planning and diagnostic context.  No workaround was attempted.

## Scope

Reviewed:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-subplan-2026-07-09.md`
- Phase 2Y result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-result-2026-07-09.md`

## Findings

Initial review found one fixable issue: the first draft named proposal
families but did not fully pin mixture weights, component allocation, local
scales, and seeds.  The subplan was patched to make those choices explicit
before implementation.

No remaining blocking findings for Phase 2Z implementation.

- The subplan correctly treats Phase 2Z as a proposal-strategy pilot, not a
  reference-validity or HMC-agreement phase.
- The candidate proposal families are motivated by Phase 2Y diagnostics and
  preserve the no-HMC-moment-tuning boundary.
- The pilot nomination screen cannot promote a candidate directly to a valid
  independent reference; a Phase 2ZA replication is required.
- The stop conditions cover artifact validity, nonfinite target/proposal/log
  weights, proposal replay mismatch, runtime timeout, and unsupported claims.
- Phase 3 GPU/XLA remains blocked.
- The patched deterministic sample-allocation rule uses intended mixture
  weights for log density, not rounded sample fractions.

## Residual Risks

- Anchor-mixture and ridge-line candidates are allowed only as pilot
  nominations because they use Phase 2Y failed-proposal top-weight anchors.
- Any candidate passing Phase 2Z could be overfit to Phase 2Y diagnostics; the
  required Phase 2ZA independent replication is therefore material, not
  optional.
- The `4096` pilot size is a cost/timeout compromise and not a statistical
  superiority design.

VERDICT: AGREE
