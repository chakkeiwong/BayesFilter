# Multi-Model NeuTra Filter-Posterior Terminal Result

Date: 2026-07-17

Program ID: `multimodel-neutra-filter-posterior-20260715`

Decision: `CELL_COMPLETE_WITH_BLOCKERS`

## Outcome

The P0-P7 runbook is complete. All eleven mandatory cells have one evidence-
bound terminal state and an exact re-entry rung. Three cells are
`NEUTRA_CONFIRMED` at narrow same-target physical-posterior-mean scope; eight
retain precise filter, source-route, target-design, implementation, or
comparator blockers.

| Cell | Terminal state | Narrow conclusion | Earliest re-entry |
| --- | --- | --- | --- |
| `SVX-SGQF` | `TARGET_BLOCKED_FILTER_ADMISSION` | no frozen SGQF level passed the numerical filter gate | R1 filter admission |
| `SVX-ZC` | `TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH` | current wrapper is an extension/invention, not the production source route | R0 source-route design |
| `KSC-UKF` | `TARGET_BLOCKED_FILTER_ADMISSION` | declared recurrence failed dense-reference value/score margins | R1 filter admission |
| `PP-SGQF` | `NEUTRA_CONFIRMED` | same-target agreement of six physical posterior means for one T=20 fixture | complete at six-mean scope |
| `PP-UKF` | `NEUTRA_CONFIRMED` | same-target agreement of six physical posterior means for one T=20 fixture | complete at six-mean scope |
| `PP-ZC` | `TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH` | generic retained-grid route is production-ineligible | R0 source-route design |
| `STR-UKF` | `COMPARATOR_BLOCKED_GEOMETRY` | source energy-health and affine mode-score gates blocked a comparator | R2 comparator geometry |
| `STR-ZC` | `TARGET_BLOCKED_EXTENSION_ROUTE_NOT_DESIGNED` | extension/invention target is absent | R0 extension-target design |
| `SIR-SGQF` | `NEUTRA_CONFIRMED` | same-target agreement of three physical posterior means for one T=20 fixture | complete at three-mean scope |
| `SIR-UKF` | `IMPLEMENTATION_BLOCKED_GPU_SCORE_PARITY` | GPU/CPU score gap `5.966e-7` exceeded the prospective `1e-7` limit | R1 GPU score parity |
| `SIR-ZC` | `TARGET_BLOCKED_MISSING_OBSERVED_DATA_SCORE_ROUTE` | retained observed-data parameter-score closure is absent | R0 observed-data score route |

`ALL_CELLS_CONFIRMED` is false and forbidden.

## Cross-Repository Breadth

The broader BayesFilter plus `/home/chakwong/python` evidence inventory counts
`9` distinct model families and `12` materially different posterior-target
configurations that reached learned-NeuTra transformed HMC. Under each
historical result's own contract, `9` configurations have clean or strong
diagnostic passes and `3` are qualified/marginal. Seeds, dimensions, tuning
candidates, transport arms, and training-only canaries are not counted as new
models.

The detailed ledger and prospective one-seed truth-tail policy are in
`docs/plans/bayesfilter-neutra-cross-repository-model-evidence-ledger-2026-07-17.md`.
This breadth is reasonable evidence that NeuTra is a method worth trying. It
does not retroactively turn the current three BayesFilter confirmations into
truth-recovery tests or erase the eight current program blockers.

## Confirmed Evidence

Each confirmed cell binds a repository-issued typed target, fresh target-
specific 5,000-step GPU/XLA transport, admitted same-target plain-HMC
comparator, fresh disjoint tuning verifier, separately retained warm-up, and a
fresh retained confirmation. Modern R-hat is the maximum of rank-normalized
split and folded rank-normalized split R-hat.

| Diagnostic | `PP-UKF` | `PP-SGQF` | `SIR-SGQF` |
| --- | ---: | ---: | ---: |
| tuning-verifier modern R-hat | `1.0054056853` | `1.0013382279` | `1.0005924637` |
| warm-up draws per chain | `2,000` | `2,000` | `2,000` |
| retained draws per chain | `4,000` | `4,000` | `4,000` |
| final maximum modern R-hat | `1.0008110775` | `1.0003275699` | `1.0000688996` |
| final minimum bulk ESS | `27,623.60` | `26,978.49` | `16,358.48` |
| final minimum tail ESS | `13,394.13` | `12,974.65` | `14,568.53` |
| simultaneous mean agreement | all six passed | all six passed | all three passed |
| target/status/energy vetoes | none | none | none |

The P4 active result hashes are
`d9b4f603b28acb06154ab554f41f745c5f544e2516ba4969c6b21d9e5268bacf`
for `PP-UKF` and
`a77d5edf2b8129d6ff95844e9c5d4bb94b7125c9997777b517f36b830fbda9c4`
for `PP-SGQF`. The `SIR-SGQF` result hash is
`e8b6c159648ade9f2919d97674ffc50a8b55d75d591a256291c3abfdcd4dbcce`.

These are one-fixture, mean-level confirmations of each filter-defined
posterior. They are not full-distribution agreement or evidence that the
approximate filter equals the intended exact model.

## Repair Record

P7 attempt 01 correctly found that the old P4 confirmations selected kernels
from short probes without a disjoint tuning verifier. Both states were
provisionally downgraded and R4 was reopened at the earliest invalid rung.

The repair reused hash-verified short probes only for candidate ordering. Step
`0.20` passed fresh 1,000-burn-in plus 1,000-draw disjoint verification for
both cells, followed by fresh warm-up and retained samples under new seeds and
roots. Old samples were not pooled or relabeled. Attempt-01 P7 and old P4
artifacts remain preserved as historical trigger evidence.

## Integrity And Policy Audit

The active structured result is
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p7/attempt-02/result.json`,
SHA-256
`97edb5e182a88591e18af7cd28bc4dae209d8166d790cd3c1638e0027ace9b99`.
The active eleven-cell ledger SHA-256 is
`7e078b56a4fa71ac86d8dd7171825ff4d53d845e746d3276ac577b4640ebaf5d`.

- Registry membership is exactly eleven unique cells with one state each.
- All eleven terminal evidence hashes and all nine confirmation/training/
  comparator recursive ledgers reverified.
- Target signatures are distinct and match their confirmation, training,
  comparator, tuning, and retained archives.
- All three tuning archives are target/seed/grid bound, 1,000 by four chains,
  and excluded from posterior inference.
- Warm-up and retained archives are separate; retained draws are below the
  10,000 cap.
- The active NeuTra scan found no NumPy import, host callback, or Python
  sample-axis loop.
- GPU/XLA/TF32/memory-growth provenance is recorded by the serious runs.
- P7 itself was a standard-library CPU-only audit with
  `CUDA_VISIBLE_DEVICES=-1`.

## Decision And Inference Status

| Field | Status |
| --- | --- |
| Primary criterion | passed: all eleven cells uniquely classified and evidence-bound |
| Hard veto screen | clear for three narrow confirmations; eight precise local blockers remain |
| Statistically supported ranking | none |
| Descriptive only | runtime, acceptance, losses, quantiles, SDs, correlations, and aggregate counts |
| Default readiness | not established for any model/filter cell |
| Next justified action | use the cost-bounded one-seed truth-tail diagnostic on the next runnable model/filter; repair a blocked route only when scientifically useful |

## Post-Run Red Team

The strongest overclaim risk is treating same-target mean agreement as
distributional agreement or filter exactness. The weakest evidence is
generalization beyond one fixture and the declared means. A failure on another
fixture or a distribution-sensitive diagnostic would not contradict the
current narrow mean result, but it would block any broader claim.

Nothing here establishes universal NeuTra success, covariance/tail/mode
agreement, filter exactness or ranking, calibration, cross-fixture robustness,
production readiness, or default readiness. A blocked cell is not evidence
that NeuTra or its research direction is invalid.
