# Phase 11 NeuTra Long-Trace Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_HARD_GATES_ROLE_LIMITED_RESIDUALS_UNRESOLVED`  
Budget cap: `3600 s` within the unchanged global `64800 s` cap  
Input: the Phase 8 metadata-bound/audited N=300 bank  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase11`

## Objective and audit

Test whether the compact and wider NeuTra arms continue improving at 300
updates under the same frozen data split and target-specific objective. Phase
10 showed a clear budget-sensitive improvement but no IID certificate. A long
trace could overfit the training/validation rows, so the untouched audit rows
remain read-only and no arm is promoted from its loss.

## Evidence contract

| Field | Choice |
|---|---|
| Arms | `compact`, `compact_low_lr`, `wider_mid_lr` from the Phase 10 profile |
| Primary hard criteria | GPU memory policy, XLA/batch-native updates, finite values, parity, target/status audit, input/mode/hash integrity |
| Explanatory criteria | validation loss and latent moment/covariance trajectory; no IID threshold is a promotion gate |
| Vetoes | any hard engineering/status failure, audit leakage, or HMC launch |
| Nonclaims | no posterior correctness, exhaustive mode discovery, IID Gaussian theorem, HMC readiness, superiority, or default change |

The 300-update count is a measured next-budget hypothesis, not a default. If
the compact trace plateaus well above the diagnostic target while hard gates
pass, refresh toward a representation/support diagnosis rather than endlessly
increasing steps.

## Execution

Use the same Phase 10 command and audited bank with `--profile tuning
--steps 300`, a fresh seed, and a fresh output root. Run focused static tests
before and after. Preserve all traces and classify any failure in the companion
repair note.
