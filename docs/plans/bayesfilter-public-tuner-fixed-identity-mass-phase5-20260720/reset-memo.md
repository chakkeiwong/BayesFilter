# Phase 5 Reset Memo

Snapshot: `2026-07-20`

Status: `PHASE5_CLOSED_REQUIRES_HOST_EXECUTION_BOUNDARY`

Read first after a reboot:

- `docs/plans/bayesfilter-public-tuner-fixed-identity-mass-phase5-completion-plan-2026-07-20.md`
- `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/phase5-terminal-result.md`
- `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260720/phase4-terminal-result-reconciliation.md`

Phase 4 LGSSM is complete and must not be rerun. Phase 5 produced complete
tuning-veto artifacts for `PP-UKF` and `PP-SGQF`, and incomplete infrastructure
artifacts for `SIR-SGQF` and `STR-UKF`. Preserve all roots exactly.

The new observability repair is active: per-cell `run_state.json` is diagnostic
state, while `result.json` plus `run_manifest.json` are terminal scientific
artifact authority. Campaign child output is streamed to durable per-cell
launch logs. This prevents a stale snapshot from being mistaken for a terminal
scientific result, but it cannot escape the managed process namespace.

No further serious GPU run should be launched from the current Codex-managed
boundary. A future continuation requires an independent host process boundary,
fresh versioned roots, the unchanged target/method/criteria/budget, and a new
focused audit if the tuner repair changes the candidate contract.
