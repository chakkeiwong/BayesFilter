# SSL-LSTM q=20 seed-B predictive equivalence result (2026-08-08)

## Decision

`CALIBRATION_INCONCLUSIVE_MATERIAL_COMPARISON_CLOSED`

The formal fixed-parameter predictive-equivalence test was planned and its
mechanics canary passed, but the q=20 calibration could not freeze an MMD
tolerance that both controlled the null and detected the predeclared `+0.20`
output-level alternative across eight independent calibration replications.
The material posterior-mean versus true-control comparison therefore did not
run. This is an underpowered design result, not evidence of predictive
equivalence and not evidence against the SSL-LSTM research direction.

## What Was Tested

The comparison target was correct:

- one posterior-mean physical parameter vector from the authenticated seed-B
  archive;
- one fixed q=20 `PRIOR_CENTER` true-control vector;
- independent ten-step forecast paths with terminal, process, and observation
  noise;
- no propagation of the 4,000 HMC draws as a parameter mixture.

The canary completed in `31.7699 s` with CPU hidden and XLA enabled.

## Calibration Evidence

| Item | Result |
|---|---|
| Calibration artifact | `docs/plans/artifacts/ssl-lstm-q20-seed-b-predictive-equivalence-2026-08-08/r5/calibration.json` |
| Calibration status | `CALIBRATION_INCONCLUSIVE` |
| Replications | 8 independent null and controlled-shift replications |
| Draws per lane | 1,024 draws x 2 forecast replications, four lanes |
| Frozen distance subset | 128 draws per lane for the quadratic bandwidth-scale calculation |
| q=20 median path distance | `3.85406549` |
| Candidate bandwidths | `1.92703275`, `3.85406549`, `7.70813099` |
| Controlled shift | `+0.20` output-level shift |

The candidate-level MMD intervals were:

| Candidate tolerance | Null outcome | Controlled-shift lower bounds |
|---:|---|---|
| `0.005` | 7/8 passed; one was `INCONCLUSIVE_UNDERPOWERED` with null upper `0.00553027` | `0.00487638` to `0.00839043` |
| `0.01` | 8/8 passed | `0.00487638` to `0.00839043`, all below `0.01` |
| `0.02` to `0.16` | 8/8 passed | all below the tolerance |

Thus `0.005` was too close to the null uncertainty, while `0.01` and larger
failed to detect the controlled shift. Selecting an intermediate value such as
`0.007` would be post hoc and is forbidden.

## Decision Table

| Decision | Primary criterion | Veto/status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Formal equivalence comparison | q=20 calibration must freeze bandwidth/tolerance before material data | `CALIBRATION_INCONCLUSIVE` | MMD tolerance resolution and calibration power | Increase q=20 calibration power or predeclare a denser tolerance grid in a new reviewed plan | Posterior correctness, output equivalence, model adequacy |
| Material run | Requires passed calibration receipt | Closed fail-closed; no material paths opened | None for this run | Do not interpret the old descriptive plug-in discrepancy as formal evidence | Mean plug-in pass/fail |
| Harness integrity | Canaries, archive binding, target signatures, finite/XLA checks | Passed | CPU-only diagnostic route | Preserve receipts and review q=20 design repair | GPU/default readiness |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Mechanics and archive screens passed; no hard numerical veto fired |
| Statistically supported ranking | None |
| Descriptive-only differences | The previous 1,024-path plug-in means/variances remain descriptive only |
| Formal predictive equivalence | Not tested because calibration did not freeze |
| Default readiness | Not established |
| Next evidence needed | A new reviewed q=20 calibration design with enough resolution/power to separate null uncertainty near `0.0055` from the controlled signal near `0.007` |

## Artifact and Manifest

| Field | Value |
|---|---|
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-predictive-equivalence-plan-2026-08-08.md` |
| Runner | `docs/benchmarks/run_ssl_lstm_q20_seed_b_predictive_equivalence_2026_08_08.py` |
| Terminal calibration | `docs/plans/artifacts/ssl-lstm-q20-seed-b-predictive-equivalence-2026-08-08/r5/calibration.json` |
| Material closure receipt | `docs/plans/artifacts/ssl-lstm-q20-seed-b-predictive-equivalence-2026-08-08/r5/material-closed.json` |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; float64; XLA |
| Device | CPU-only; `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import |
| Material parameter mixture | `False` |

## Red Team

The strongest alternative explanation is not a model failure but insufficient
MMD resolution: the observed controlled signal is close to the null upper tail
for the available bank size. The conclusion would be overturned by a reviewed
larger calibration that freezes a valid tolerance, or by discovering that the
q=20 calibration path construction was not the intended marginal output law.
The weakest evidence is therefore calibration power, not archive integrity or
forecast finiteness.

