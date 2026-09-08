# Corrected Parameter-Authority Phase 44 Result

Date: 2026-08-26  
Version: `v2.6-larger-n-support-diagnostic`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase44-subplan-2026-08-26.md`  
Status: `PASS_V2_6_LARGER_N_REPORT_REPAIR_TRIGGERED`

## Question and scope

Phase 44 tested whether increasing one independently generated post-training
theta bank from `N=256` to `N=512` reduces the isolated bank-A support/outlier
behavior under the same frozen NeuTra trainer state. The declared target stayed
the batch-native q=20 SSL-LSTM target in `theta in R^4`; the 60D UKF state
remained internal to the target. No objective, architecture, proposal,
whitening criterion, HMC route, or canonical LEDH route changed.

## Execution and retry record

The first trusted GPU attempt reached target evaluation and was interrupted
before writing an artifact. It is recorded in the subplan retry ledger and is
excluded from scientific interpretation. The second attempt used the unique
root `phase44-larger-n-support/frozen-four-bank-audit-attempt2/` and completed
with `PASS_V2_6_LARGER_N_BOUNDARY`.

| Artifact | Status / location |
|---|---|
| N=512 pilot | `phase44-larger-n-support/fresh-n512/pilot.json`, `PASS_THETA_MEASURE_PILOT` |
| GPU audit | `phase44-larger-n-support/frozen-four-bank-audit-attempt2/result.json`, `PASS_V2_6_LARGER_N_BOUNDARY` |
| CPU report | `phase44-larger-n-support/report/result.json`, `PASS_V2_6_LARGER_N_REPORT` |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| GPU runtime | TensorFlow 2.20.0, XLA and TF32 enabled; two RTX 4080 SUPER devices |
| GPU policy | memory growth verified before logical-device initialization |
| Successful audit wall time | `1155.8831641 s` |
| Focused regression check | `3 passed, 28 deselected` |

The N=512 M0 pilot used seed `(20260826, 4404)` and calibration size 128.
Its ESS fraction was `0.9273800455`, weighted negative-mode mass was
`0.4034691775`, and it retained 248 distinct roots. Pilot, M0/C0 tensor,
target, and protocol hashes were checked and distinct from the authority and
N=256 banks. The calibration-size change was a recorded campaign hypothesis,
not a promoted default.

## Support receipt

| Source | Rows | Roots | ESS fraction | Negative-mode mass | theta mean[0] |
|---|---:|---:|---:|---:|---:|
| authority | 256 | 122 | 0.952283 | 0.530069 | 0.289568 |
| bank A | 256 | 103 | 0.801812 | 0.756588 | 3.550030 |
| bank B | 256 | 128 | 0.946687 | 0.517590 | 1.013180 |
| bank C | 256 | 125 | 0.975794 | 0.565503 | 0.877022 |
| bank N=512 | 512 | 248 | 0.927380 | 0.403469 | 1.446191 |

These are finite support diagnostics. They do not estimate exhaustive mode
probabilities, establish common support, or prove a population limit.

## Frozen-state transport diagnostics

Entries are `weighted latent mean max-abs / off-diagonal covariance max-abs`.
The old validation row is retained only as the historical v2.2 comparator.

| Arm | Old validation | A | B | C | N=512 |
|---|---:|---:|---:|---:|---:|
| affine:compact | 0.424126 / 0.655261 | 1.138738 / 0.896074 | 0.199921 / 0.390038 | 0.217778 / 0.432846 | 0.381323 / 0.375972 |
| affine:wide_low_lr | 0.462624 / 0.637627 | 1.099791 / 0.661917 | 0.096337 / 0.208018 | 0.298276 / 0.508351 | 0.389876 / 0.289003 |
| identity:compact | 0.547410 / 0.893340 | 0.914067 / 0.679319 | 0.198587 / 0.344945 | 0.182851 / 0.399164 | 0.324063 / 0.469106 |
| identity:wide_low_lr | 0.583848 / 1.012230 | 1.045722 / 0.774507 | 0.247726 / 0.377084 | 0.235210 / 0.476592 | 0.422946 / 0.411835 |

The N=512 bank is lower than bank A on both displayed residuals in every arm.
It is still materially nonzero, and it is not uniformly better than the old
comparator on both residuals. The difference is therefore descriptive support
evidence, not a statistical ranking or whitening result.

All terminal state hashes exactly matched the v2.4 frozen references:

| Arm | State hash |
|---|---|
| affine:compact | `7a7537f0c4d4dbf46a9b60b04d78915bfe59a07122aa50b41ac1c47bbdbd3d96` |
| affine:wide_low_lr | `d5d15f3fe4c25d50f5b4f8bd3dad00d754fd0aaa54e3739de1fea084ebed1eb5` |
| identity:compact | `cfc3d44e02c381c4e3bf34cde78600d7d70d623e6d45391fec92347f3c9e3ada` |
| identity:wide_low_lr | `4893fec8be007b604f4343db36dd1aa08b10efff0290841386e9fd8b10d4bf64` |

## Decision table

| Decision | Primary criterion | Status | Veto / limitation | Next justified action | Not concluded |
|---|---|---|---|---|---|
| retain theta target boundary | target/status/protocol and state hashes | pass | none | retain parameter-space authority | posterior correctness |
| promote IID whitening | finite-bank residuals | veto | material residuals remain | keep whitening closed; replicate N=512 | IID Gaussian law |
| change objective or architecture | independent support evidence | defer | one N=512 bank and no uncertainty interval | run independent N=512 replication | objective superiority |
| admit HMC or canonical LEDH | density and downstream gates | veto | role-limited support audit only | keep routes closed | HMC readiness |

## Inference status and ledgers

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed: target/status, hash separation, fresh-use, finite, GPU, XLA, transport parity, and state-hash gates. |
| Statistically supported ranking | None. There is no uncertainty interval or replication-level model. |
| Descriptive-only differences | N=512 is better than the isolated A bank on both residuals in all four arms; residuals remain material. |
| Default readiness | Not ready for whitening, posterior, HMC, or canonical LEDH promotion. |
| Next evidence needed | An independent N=512 replication under the same frozen trainer state. |

Engineering correctness passed. Numerical validity passed only for finite target,
score, transport parity, and status. Scientific interpretation remains
role-limited because all support comparisons are finite and bank-specific.

## Post-run red team

The strongest alternative explanation is shared proposal-support bias, or an
unrepresentative old validation comparator, rather than finite bank size alone.
An independent N=512 bank with materially different residuals would weaken the
finite-count explanation; two stable N=512 banks with persistent residuals would
strengthen the case for a separately scoped support/proposal or objective
repair. The weakest evidence is one N=512 bank and one frozen state.

## Nonclaims and data-use record

No bank was pooled, dropped, tuned on, selected, or used for optimizer input.
This result establishes no IID Gaussian whitening, posterior correctness,
normalizer, exhaustive mode discovery, HMC convergence, canonical LEDH
validity, superiority, or default readiness. The branch
`larger_n_descriptively_better_than_bank_a` is a repair hypothesis, not a
statistical ranking.

