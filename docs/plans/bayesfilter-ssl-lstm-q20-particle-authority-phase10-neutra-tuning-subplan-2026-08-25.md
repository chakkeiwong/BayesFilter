# Phase 10 NeuTra Tuning Ladder Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_HARD_GATES_ROLE_LIMITED_TUNING_INCOMPLETE`  
Budget cap: `7200 s` within the unchanged global `64800 s` cap  
Input: Phase 8 metadata-bound/audited N=300 bank  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase10`

## Objective

Distinguish a short-budget/architecture tuning failure from an input-measure or
representation failure in target-specific NeuTra training. Use one immutable
audited bank, a frozen 180/60/60 split, and a predeclared budget ladder. The
result is a component diagnostic, not a posterior or whitening theorem.

## Skeptical pre-execution audit

The bank has passed the independent finite-measure audit and binds `mode_axis=2`.
The Phase 9 screen passed hard engineering gates but had large latent moment
residuals. A longer run could still pass while overfitting the validation
partition, so audit rows remain untouched and no architecture is promoted from
validation loss alone. The ladder is intentionally small; its values are
hypotheses, not defaults.

## Evidence contract

| Field | Choice |
|---|---|
| Comparator | Phase 9 compact/wide short screen versus three tuning arms on the same bank and split |
| Tuning arms | compact 100 updates at 1e-3; compact 100 updates at 2e-4; wider 100 updates at 5e-4 |
| Primary hard criteria | same GPU/memory/XLA/batch/parity/target-status/audit gates as Phase 9 |
| Promotion criterion | none from this short ladder; an arm can remain viable for a later untouched claim run only if hard gates pass and a reviewed tuning artifact is complete |
| Vetoes | non-finite/status/parity/device/batch/split failure, audit leakage, or HMC launch |
| Explanatory diagnostics | validation loss and latent moments/covariance, clipping, runtime |
| Nonclaims | no IID whitening, posterior correctness, mode discovery, superiority, HMC readiness, or default promotion |
| Artifact | per-arm traces, immutable input hash, tuning profile, manifest, decision/inference tables, repair note |

## Numeric/default provenance

The 100-update budget, hidden widths `(16,16)` and `(64,32)`, and learning
rates `1e-3`, `2e-4`, and `5e-4` are explicitly target-specific hypotheses
chosen to separate optimization speed from capacity. They are not promoted
settings. The audit partition remains read only until all arm selections are
frozen.

## Execution and refresh

Run the runner with `--profile tuning --steps 100` on the Phase 8 bank. Record
whether the latent diagnostics improve materially or remain poor while hard
transport gates pass. If all arms remain poor, refresh toward a representation/
measure diagnosis rather than silently increasing the budget. If a hard gate
fails, repair the harness and rerun the same input. Stop only under the master
real-blocker definition.
