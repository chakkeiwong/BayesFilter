# Phase 44 Repair and Refresh

Date: 2026-08-26  
Source result: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase44-result-2026-08-26.md`  
Branch: `larger_n_descriptively_better_than_bank_a`  
Next version: `v2.7-independent-n512-replication`

Phase 44 passed every engineering and finite target/status gate. The N=512
bank was descriptively better than the isolated N=256 bank A in all four arms,
but its residuals remained material and no uncertainty-supported ranking exists.
The whitening veto therefore remains active. The result does not identify an
objective or architecture defect.

## Repair decision

The next smallest discriminating artifact is a second, independently generated
N=512 bank with a fresh seed, evaluated together with the authority and all
previous N=256 banks after one frozen trainer per arm. The first N=512 bank is
retained as an untouched contextual audit. No bank is pooled, dropped, used for
training, tuning, checkpoint selection, or objective selection.

This replication changes only the independent claim-bank draw. It keeps the
theta measure, root-group training split, proposal semantics, target signature,
four arm configurations, optimizer steps, XLA/GPU policy, and whitening veto
unchanged. The calibration-only size remains 128 so the replication tests bank
draw variability at fixed N and fixed pilot protocol.

## Required gates

1. The new N=512 pilot is finite, `theta_R4`, target-signature exact, and has
   the frozen M0/C0 protocol hashes with particle/calibration counts 512/128.
2. Its pilot and tensor hashes are distinct from authority, A, B, C, and the
   first N=512 bank.
3. One batch-native GPU/XLA trainer per arm consumes only the old 232 training
   rows; every bank is evaluated after the final update.
4. Each terminal state hash equals the Phase 44 state hash for the same arm.
5. Target/status, transport parity, and support tensors are finite.
6. The comparison report includes both N=512 banks, keeps all branches
   descriptive, and preserves the whitening/objective vetoes.

## Interpretation branches

| Result | Role | Next action |
|---|---|---|
| both N=512 banks are materially closer than A but residual-heavy | finite A outlier explanation strengthened | write a scoped support-envelope or proposal repair; no whitening promotion |
| N=512 banks differ materially from each other | support variability persists at larger N | repair proposal/support generation before objective changes |
| both N=512 banks are stable but residual-heavy | finite-count explanation weakened | prepare a separately governed objective/capacity hypothesis plan with uncertainty |
| any hard gate fails | engineering/numerical veto | preserve failed root, repair, and rerun with a new root |

These branches are descriptive. They do not establish IID Gaussian whitening,
posterior correctness, exhaustive mode discovery, HMC readiness, or canonical
LEDH validity.

## Skeptical audit and pre-mortem

The plan passes review on 2026-08-26 because it changes only the independent
post-training bank draw and adds no outcome-dependent selection. The exact
Phase 44 state hashes bind all arms to the same frozen trainer. A copied bank,
protocol drift, fresh-row leakage, or state mismatch is a hard veto. The main
misleading failure is correlated proposal support across both N=512 banks;
pilot/tensor hash separation, root/mode summaries, target/proposal ranges, and
the explicit nonclaims prevent that from being called coverage or whitening.

## Artifact and budget boundary

Artifacts live below
`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/`.
The first N=512 bank and all prior receipts remain immutable. The local phase
cap is 5400 s; the remaining campaign budget is checked from the prior run
manifests before launch. A localized infrastructure failure may retry once in
a fresh root without changing the contract; repeated failure is recorded as a
continuation veto only after the stated threshold is met.

