# SSL-LSTM Completion Phase A0 Geometry Contract Review

Date: 2026-07-11

Reviewer type: `CODEX_SUBSTITUTE_REVIEW`

Exact reviewed paths, one at a time:

- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-subplan-2026-07-11.md`
- `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py`

## Trigger

The first runtime geometry check showed that the reviewed raw z-to-theta
equalities fail at roughly the declared `1e-9` mass jitter scale, while factor
and within-coordinate inverse identities pass.

## Source Audit And Repair

The cited source transforms the fitted z precision to raw theta precision,
then `covariance_from_precision` symmetrizes, adds `jitter=1e-9`, applies the
eigenvalue-floor/condition-cap rule, and inverts the regularized precision.
The plan and harness now reproduce that exact path. Raw-to-stored differences
are preserved as explanatory diagnostics, not vetoes.

The first focused plan review requested two precision repairs: store the
effective eigenvalue floor and define exact operands for both explanatory
residuals. Both were patched. Final focused plan review returned
`VERDICT: AGREE`. The focused harness review also returned `VERDICT: AGREE`.

## Nonclaims

This resolves a plan/schema/harness defect only. It does not validate Phase 2S
as posterior covariance, establish HMC readiness, rank a sampler, or change the
SSL-LSTM target.

VERDICT: AGREE
