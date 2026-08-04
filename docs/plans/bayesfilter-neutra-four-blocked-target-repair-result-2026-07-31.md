# NeuTra Four-Target Repair Result

Date: 2026-07-31
Plan: `docs/plans/bayesfilter-neutra-four-blocked-target-repair-and-admission-plan-2026-07-31.md`

## Decision

The bounded repair campaign did not authorize NeuTra training or HMC for any
of the four requested targets. `KSC-UKF` now has a CPU/reference-admitted
mass-preserving Gaussian-sum repair, but its required trusted GPU/XLA canary
could not be relaunched because the platform permission review timed out before
process creation. The other three remain blocked by their declared scientific
or source-route gates.

| Target | Primary criterion | Veto/result | Status | Next justified action |
| --- | --- | --- | --- | --- |
| `SVX-SGQF` | dense-prefix/full value and score gates | prefix dense value gap stayed `0.003426` to `0.003440` per observation at levels 10-24, above `0.001`; score FD checks passed | blocked | redesign the single-Gaussian SGQF approximation or explicitly retain it as diagnostic; do not tune NeuTra |
| `KSC-UKF` | dense T20 value gap `<=1e-3/observation`, score gap `<=1e-2`, finite/status/permutation | mass-preserving clustered Gaussian-sum caps 7-256 passed CPU screen; cap 32 max value gap `1.29e-5`, score gap `1.72e-3`, retained mass `1.0`; GPU canary boundary timed out | CPU filter admitted, GPU canary pending | rerun one trusted GPU/XLA cap-32 canary, then issue a scope identity only if parity passes; create a separate tuning/training plan afterward |
| `SVX-ZC` | monograph fixed-branch numerical admission | all tested ranks pass structural/FD checks but fail rank-saturation residual (`0.0564+` vs `1e-8`) | blocked | capacity/fit repair under a fresh bounded plan; do not relax the veto |
| `PP-ZC` | NeuTra target-contract admission | sealed fixed-branch implementation, CPU/GPU tie-out, ESS, and fresh tuning pass; assembled route remains `extension_or_invention`, with no registered batch-native posterior adapter or frozen HMC chart/Jacobian | blocked | build and test the target adapter/chart under a separate plan; no NeuTra training or HMC |

## KSC Repair Evidence

Focused tests: `5 passed` in `tests/test_ksc_gaussian_sum_ukf_neutra_target.py`
and `tests/test_neutra_four_blocked_target_repair.py`.

CPU artifact: `docs/plans/artifacts/bayesfilter-neutra-four-blocked-target-repair-20260731/ksc-ukf/attempt06/result.json`.

The repair retains multiple components and assigns pruned components to
nearest retained centers, moment-merging each cluster. This preserves all
normalized mass, unlike the failed top-weight-only version whose retained mass
was about `0.48-0.57`. The cap-32 CPU result passed the frozen screen with
maximum value gap `1.292e-5` per observation and maximum score gap `1.719e-3`.

Infrastructure attempts are preserved in `attempt05/failure.json` (host OOM
from a quadratic membership tensor) and `attempt07/failure.json` (XLA fixed
tensor-list requirement). Both were repaired in code. `attempt08/failure.json`
records the external permission-review timeout; it is not model evidence.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | `SVX-SGQF` fails dense-prefix value; `SVX-ZC` fails rank-saturation; `PP-ZC` passes sealed implementation but lacks NeuTra target-contract admission; KSC CPU numerical gates pass, GPU parity is untested |
| Statistically supported ranking | none; no stochastic candidate ranking was performed |
| Descriptive-only differences | all approximation gaps, component counts, and route comparator values |
| Default readiness | not assessed for all four targets |
| Next evidence needed | KSC trusted GPU/XLA parity; SVX-ZC capacity/fit repair; PP-ZC batch-native adapter plus HMC chart/Jacobian; a new SGQF approximation design |

## Nonclaims

This result does not claim exact likelihood, posterior correctness, source
faithfulness for either Zhao-Cui extension, NeuTra training quality, HMC
convergence, superiority, leaderboard readiness, or a new default.

## Post-Run Red Team

The strongest alternative explanation for the KSC result is that the dense
reference and repaired mixture share a target convention that differs from a
future author-native KSC target. The artifact therefore binds the transform,
mixture tensors, hashes, horizon, and reference orders explicitly. A failed
GPU parity canary would overturn KSC promotion without invalidating the CPU
diagnostic; a passing canary would still not establish HMC or scientific
readiness.
