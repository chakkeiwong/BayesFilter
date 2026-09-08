# Corrected Parameter-Authority Phase 41 Result

Date: 2026-08-26  
Continuation version: `v2.3-independent-audit-bank`  
Status: `PASS_V2_3_INDEPENDENT_AUDIT_REPORT_REPAIR_TRIGGERED`

## Scope

Phase 41 tested whether the material v2.2 NeuTra residuals were mainly caused
by the finite root-group validation/audit partition. The v2.2 N=256 M0 train
rows and normalized weights were frozen. A new N=256 C0/M0 bank was generated
with the same target, proposal family, schedule, defensive mixture, and
protocol, but a fresh seed. Identity and exact-training-measure affine arms
were trained only on the old 232-row training split and evaluated at their
terminal state on the untouched fresh M0 bank.

The 60-dimensional UKF state remained internal. All particle rows were
`theta in R^4`; no HMC or LEDH route was launched.

## Receipts and repairs

| Artifact | Status | Role |
|---|---|---|
| fresh N=256 C0/M0 pilot | `PASS_THETA_MEASURE_PILOT` | independent theta source; target signature and M0/C0 protocol hashes match the frozen source |
| frozen-training audit attempt 1 | harness veto | C0 arm hash was incorrectly required to equal M0 hash; no training occurred |
| frozen-training audit attempt 2 | harness veto | validation shape relaxation rejected the 256-row audit after the 12-row holdout |
| static-partition regression | `3 passed, 28 deselected` | supports the shape-specific validation repair only |
| frozen-training audit attempt 3 | `PASS_V2_3_INDEPENDENT_AUDIT_BOUNDARY` | GPU/XLA, memory growth, finite target/status, affine oracle, and round-trip gates pass |
| independent-bank report attempt 1 | reporter veto | explicit float64 cast was missing for JSON-decoded float32 covariance |
| independent-bank report attempt 2 | `PASS_V2_3_INDEPENDENT_AUDIT_REPORT` | read-only support and residual comparison |

Primary artifacts:

- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/fresh-n256/`
- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/frozen-training-audit-attempt3/`
- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase41-independent-audit-bank/report-attempt2/`

## Measured comparison

| Arm | old v2.2 validation mean/cov residual | fresh-bank mean/cov residual |
|---|---:|---:|
| identity compact | `0.549102 / 0.882597` | `0.241250 / 0.285645` |
| identity wide, low LR | `0.584842 / 1.003937` | `0.178388 / 0.322732` |
| affine compact | `0.376301 / 0.667096` | `0.400899 / 0.469012` |
| affine wide, low LR | `0.449028 / 0.825264` | `0.420763 / 0.483285` |

The fresh bank had 121 terminal roots, ESS fraction `0.979248`, negative-mode
fraction `0.543599`, and `theta_mean[0] = 0.226798`. The old v2.2 training
partition had 108 roots, ESS fraction `0.950551`, negative-mode fraction
`0.533571`, and `theta_mean[0] = 0.330821`; the old validation and audit
partitions had `theta_mean[0] = 1.181110` and `-1.422781`, respectively.

The identity residual reduction is descriptive. The affine covariance
residual also decreases, but the affine mean and covariance residuals remain
material. The report therefore classifies the finite-holdout mismatch as
plausible, not established.

## Decision ledgers

### Engineering, numerical, and scientific

| Ledger | Status | Evidence | Limit |
|---|---|---|---|
| Engineering correctness | pass | exact target/protocol hashes, `[N,4]` batches, fresh-bank exclusion flags, GPU memory growth, XLA, finite receipts | one frozen state and one fresh bank |
| Numerical validity | pass for finite boundary | affine training oracle and transport round trips pass | no density-identification or target-coverage theorem |
| Scientific interpretation | repair trigger | fresh bank is more representative than old tiny holdouts and residuals remain nonzero | one bank cannot distinguish support variability from objective mismatch |

### Decision

| Decision | Primary criterion | Veto status | Next justified action | Not concluded |
|---|---|---|---|---|
| retain `theta_R4` target | all target/status/measure gates pass | no continuation veto | run two-bank replication under one frozen state | posterior correctness |
| reject whitening promotion | independent fresh residuals remain material | promotion veto | preserve role-limited transport only | IID Gaussian law |
| do not change objective yet | branch evidence is single-bank descriptive | no ranking evidence | quantify bank-to-bank variation first | objective superiority |

### Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | passed after documented harness repairs |
| Statistically supported ranking | none |
| Descriptive-only differences | all moment, loss, ESS, support-range, and residual differences |
| Default readiness | not ready |
| Next evidence needed | two independent fresh banks evaluated by one frozen state, then a predeclared objective/support decision |

## Red team and continuation decision

The strongest alternative is that the new seed still follows the same
mode-biased proposal and that the apparent improvement is finite-sample noise.
The next two-bank replication can overturn the finite-holdout explanation if
both fresh banks have support comparable to the old tiny holdout while residuals
remain high, or if bank-to-bank residual variation is large enough to erase the
observed direction.

Phase 41 did not invalidate the target, harness, or measure contract after the
repairs. No continuation veto fired. Continue to Phase 42; do not promote
NeuTra, HMC, posterior correctness, IID whitening, exhaustive mode discovery,
or canonical LEDH.
