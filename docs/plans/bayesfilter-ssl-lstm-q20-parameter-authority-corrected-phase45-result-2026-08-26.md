# Corrected Parameter-Authority Phase 45 Result

Date: 2026-08-26  
Version: `v2.7-independent-n512-replication`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase45-subplan-2026-08-26.md`  
Status: `PASS_V2_7_INDEPENDENT_N512_REPORT_REPAIR_TRIGGERED`

## Question and scope

Phase 45 asked whether a second independently generated `N=512` theta bank
reproduces the first N=512 bank's residual behavior under one unchanged frozen
NeuTra trainer per arm. The target remained the batch-native q=20 SSL-LSTM
target in `theta in R^4`; the 60D UKF state remained internal. No objective,
architecture, proposal schedule, whitening criterion, HMC route, or canonical
LEDH route changed.

This is finite-support evidence only. The authority cloud supplies the frozen
training measure; banks A, B, C, N512-a, and N512-b are untouched audits.

## Execution and repair record

An initial mechanical script copy was rejected by the compile gate because it
retained v2.6 four-bank assumptions and contained a syntax error. It produced
no experiment artifact. The runner and reporter were replaced with purpose-built
v2.7 files, then passed compilation, `git diff --check`, and a CPU-hidden
reporter import smoke. Two stale read-only inspection snippets used old field
names; they did not modify artifacts and are recorded in the subplan ledger.

The fresh N512-b pilot passed `PASS_THETA_MEASURE_PILOT`. Its pilot hash was
`b065210faf48aa50b214f3aa84f7d4b0dcb201a7c7674b8becf8a556f2e02838`, distinct
from the authority, A, B, C, and N512-a pilot hashes. The trusted audit passed
with status `PASS_V2_7_INDEPENDENT_N512_BOUNDARY` and wall time
`1617.6877457919763 s`.

## Hard-gate evidence

| Gate | Result |
|---|---|
| target and measure | passed; target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, measure `theta_R4` |
| protocol and tensor independence | passed; six pilot receipts and M0/C0 tensor hashes distinct |
| root-group split | passed; 232 training, 12 validation, 12 historical audit rows, complete/disjoint/root-disjoint |
| data use | passed; one trainer per arm used only the old training rows; five audit banks used after the final update |
| deterministic trainer state | passed; all four hashes match Phase 44 |
| numerical/device validity | passed; finite target/score/status, transport round trip, GPU memory growth, XLA, TF32 |

The four frozen state hashes are:

| Arm | Hash |
|---|---|
| affine:compact | `7a7537f0c4d4dbf46a9b60b04d78915bfe59a07122aa50b41ac1c47bbdbd3d96` |
| affine:wide_low_lr | `d5d15f3fe4c25d50f5b4f8bd3dad00d754fd0aaa54e3739de1fea084ebed1eb5` |
| identity:compact | `cfc3d44e02c381c4e3bf34cde78600d7d70d623e6d45391fec92347f3c9e3ada` |
| identity:wide_low_lr | `4893fec8be007b604f4343db36dd1aa08b10efff0290841386e9fd8b10d4bf64` |

## Support receipt

| Source | Rows | Roots | ESS fraction | Negative-mode mass | theta mean[0] |
|---|---:|---:|---:|---:|---:|
| authority | 256 | 122 | 0.952283 | 0.530069 | 0.289568 |
| bank A | 256 | 103 | 0.801812 | 0.756588 | 3.550030 |
| bank B | 256 | 128 | 0.946687 | 0.517590 | 1.013180 |
| bank C | 256 | 125 | 0.975794 | 0.565503 | 0.877022 |
| N512-a | 512 | 248 | 0.927380 | 0.403469 | 1.446191 |
| N512-b | 512 | 233 | 0.968359 | 0.501739 | 0.587732 |

These summaries are descriptive and do not estimate mode probabilities,
exhaustive discovery, common support, or a population limit.

## Frozen-state transport diagnostics

Entries are `weighted latent mean max-abs / off-diagonal covariance max-abs`.

| Arm | A | B | C | N512-a | N512-b |
|---|---:|---:|---:|---:|---:|
| affine:compact | 1.138738 / 0.896074 | 0.199921 / 0.390038 | 0.217778 / 0.432846 | 0.381323 / 0.375972 | 0.368506 / 0.680965 |
| affine:wide_low_lr | 1.099791 / 0.661917 | 0.096337 / 0.208018 | 0.298276 / 0.508351 | 0.389876 / 0.289003 | 0.407516 / 0.538055 |
| identity:compact | 0.914067 / 0.679319 | 0.198587 / 0.344945 | 0.182851 / 0.399164 | 0.324063 / 0.469106 | 0.221470 / 0.214194 |
| identity:wide_low_lr | 1.045722 / 0.774507 | 0.247726 / 0.377084 | 0.235210 / 0.476592 | 0.422946 / 0.411835 | 0.228665 / 0.220628 |

Both N=512 banks are lower than A on both residuals in every arm. N512-b is
not below the historical old comparator in affine:compact covariance, so the
support result is mixed. All residuals remain materially nonzero.

## Decision and inference status

| Decision | Primary criterion | Status | Veto/limitation | Next action | Not concluded |
|---|---|---|---|---|---|
| retain theta target | v2.7 target/status/protocol/hash gates | pass | none | retain parameter authority | posterior correctness |
| promote IID whitening | finite-bank residuals | veto | residuals and no population uncertainty | keep whitening closed | IID Gaussian law |
| change objective | independent support replication | defer | no uncertainty-supported comparison | run support/proposal envelope | objective superiority |
| admit HMC/canonical LEDH | density and downstream gates | veto | role-limited support audit | keep routes closed | HMC/LEDH readiness |

| Inference class | Status |
|---|---|
| hard veto screen | passed |
| statistically supported ranking | none |
| descriptive differences | N512-a/b versus A/B/C and historical comparator |
| default readiness | not ready |
| next evidence | independent N512-c plus fixed-proposal support envelope |

## Red team and nonclaims

The strongest alternative explanation is shared proposal-support bias in the two
N=512 banks, not particle count alone. The weakest evidence is two N=512 banks
under one frozen trainer and no uncertainty interval. A separately generated
support envelope with stable downstream behavior would weaken that explanation;
it would still not prove whitening.

No IID Gaussian law, posterior correctness, normalizer, exhaustive mode
discovery, HMC convergence, canonical LEDH validity, superiority, or default
promotion follows from this result. No bank was pooled, selected, tuned on, or
used for optimizer input.

## Artifacts and manifest

- Pilot: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/fresh-n512-b/`
- GPU audit: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/frozen-five-bank-audit/result.json`
- CPU report: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase45-independent-n512-replication/report/result.json`
- Runner: `docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase45_2026_08_26.py`
- Reporter: `docs/benchmarks/report_ssl_lstm_q20_parameter_authority_corrected_phase45_2026_08_26.py`

The GPU audit manifest records TensorFlow 2.20.0, two RTX 4080 SUPER devices,
memory growth enabled before logical-device initialization, XLA and TF32, the
frozen trainer seed `(20260826,4211)`, source hashes, dirty-tree state, and
wall time. The CPU report is diagnostic-only and intentionally does not claim
GPU or production evidence.
