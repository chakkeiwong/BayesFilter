# Corrected Parameter-Authority Phase 29 Result

Date: 2026-08-25  
Status: `PASS_FRESH_THETA_ETPF_ROLE_LIMITED`

## Receipt

`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase29-fresh-theta-etpf/`

The source-faithful second-order ETPF map consumed a hash-bound subset of the
fresh Phase 28 M0 bank. Both input and output were in `theta in R^4`; the
internal UKF state was never passed to the map. The transformed rows were
evaluated by the batch-native q=20 target with finite values and valid status.

| Gate | Value |
|---|---:|
| source shape | `[64,4]` |
| transformed shape | `[32,4]` |
| Riccati convergence | `true`, 54 iterations |
| row/column residual | passed |
| mean residual | `4.44e-16` |
| covariance residual | `8.69e-5` |
| target/status | `32/32` |
| upper source-range excursions | `3` |
| negative correction fraction | `0.5039` |

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain ETPF as a fresh-theta auxiliary mechanism | all finite, marginal, and target/status gates pass | no hard veto | deterministic subset and empirical transform has no density | test GenUT in parameter dimension 4 | no authority replacement, IID law, posterior correctness, mode theorem, LEDH, HMC, or default |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | passed |
| Statistically supported ranking | none; no comparator arm |
| Descriptive-only differences | covariance residual, support excursion, correction signs, mode fraction |
| Default-readiness | not ready |
| Next evidence needed | parameter-space GenUT scope/feasibility and fresh-seed replication if needed |

## Red-team note

The transform can preserve selected moments while creating support excursions
and negative transport entries. Passing the finite target check therefore does
not make the rows posterior samples or make the transform density-corrected.

