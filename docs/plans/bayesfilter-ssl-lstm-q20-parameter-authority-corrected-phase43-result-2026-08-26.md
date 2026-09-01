# Corrected Parameter-Authority Phase 43 Result

Date: 2026-08-26  
Version: `v2.5-third-bank-support-diagnostic`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase43-subplan-2026-08-26.md`  
Status: `PASS_V2_5_THREE_BANK_REPORT_REPAIR_TRIGGERED`

## Question and scope

Phase 43 tested whether the Phase 42 N=256 bank-A outlier was an isolated
finite-support/mode draw. A fresh N=256 bank C was generated and evaluated
alongside untouched banks A and B after one reconstructed NeuTra trainer per
arm. The target remained the batch-native q=20 SSL-LSTM target in
`theta in R^4`; the 60-dimensional UKF state remained internal to the target.

The old v2.2 root-group training rows, normalized weights, proposal protocol,
four arm configurations, 200 optimizer steps, target signature, and GPU/XLA
policy were frozen. Fresh rows were never used for training, tuning, or
selection. The reconstructed terminal state had to match the v2.4 audit hash
for every arm.

## Artifacts and execution

| Artifact | Result |
|---|---|
| C pilot | `phase43-third-bank-support/fresh-c-n256/pilot.json`, `PASS_THETA_MEASURE_PILOT` |
| GPU audit | `phase43-third-bank-support/frozen-three-bank-audit/result.json`, `PASS_V2_5_THREE_BANK_BOUNDARY` |
| CPU report | `phase43-third-bank-support/report/result.json`, `PASS_V2_5_THREE_BANK_REPORT` |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| GPU runtime | TensorFlow 2.20.0, XLA and TF32 enabled; two RTX 4080 SUPER devices |
| GPU policy | memory growth verified before logical-device initialization; no full-device preallocation |
| Wall time | C pilot 357.672 s; frozen audit 708.644 s; report 0.458 s |

The C pilot used root seed `(20260826, 4104)` and calibration seed
`(20260826, 4114)`. Its M0 ESS fraction was `0.975794`, weighted negative-mode
mass `0.565503`, terminal roots `66/59` (negative/positive), and log-mass
estimate `-34.818845`. C pilot, M0/C0 tensor, target, and protocol receipts
were all hash-checked and distinct from the authority and A/B receipts.

## Support receipt

| Source | Rows | Roots | ESS fraction | Negative-mode mass | theta mean[0] |
|---|---:|---:|---:|---:|---:|
| authority | 256 | 122 | 0.952283 | 0.530069 | 0.289568 |
| bank A | 256 | 103 | 0.801812 | 0.756588 | 3.550030 |
| bank B | 256 | 128 | 0.946687 | 0.517590 | 1.013180 |
| bank C | 256 | 125 | 0.975794 | 0.565503 | 0.877022 |

These are finite support diagnostics. They do not estimate an exhaustive mode
probability or establish common support for the target.

## Frozen-state transport diagnostics

The entries are `weighted latent mean max-abs / off-diagonal covariance max-abs`.
The old validation column is retained as the historical v2.2 comparator.

| Arm | Old validation | A | B | C |
|---|---:|---:|---:|---:|
| affine:compact | 0.424126 / 0.655261 | 1.138738 / 0.896074 | 0.199921 / 0.390038 | 0.217778 / 0.432846 |
| affine:wide_low_lr | 0.462624 / 0.637627 | 1.099791 / 0.661917 | 0.096337 / 0.208018 | 0.298276 / 0.508351 |
| identity:compact | 0.547410 / 0.893340 | 0.914067 / 0.679319 | 0.198587 / 0.344945 | 0.182851 / 0.399164 |
| identity:wide_low_lr | 0.583848 / 1.012230 | 1.045722 / 0.774507 | 0.247726 / 0.377084 | 0.235210 / 0.476592 |

All four reconstructed state hashes matched their v2.4 references exactly:

| Arm | State hash |
|---|---|
| affine:compact | `7a7537f0c4d4dbf46a9b60b04d78915bfe59a07122aa50b41ac1c47bbdbd3d96` |
| affine:wide_low_lr | `d5d15f3fe4c25d50f5b4f8bd3dad00d754fd0aaa54e3739de1fea084ebed1eb5` |
| identity:compact | `cfc3d44e02c381c4e3bf34cde78600d7d70d623e6d45391fec92347f3c9e3ada` |
| identity:wide_low_lr | `4893fec8be007b604f4343db36dd1aa08b10efff0290841386e9fd8b10d4bf64` |

## Decision table

| Decision | Primary criterion | Status | Veto/limitation | Next justified action | Not concluded |
|---|---|---|---|---|---|
| retain theta target boundary | target/status/protocol and state hashes | pass | none | retain parameter-space authority | posterior correctness |
| promote IID whitening | three-bank residuals | veto | residuals remain material and finite | run support/particle-count diagnostic | IID Gaussian law |
| change objective or architecture | independent-bank evidence | defer | no uncertainty-supported comparison | test larger N under frozen state | objective superiority |
| admit HMC or canonical LEDH | density and downstream gates | veto | this phase is role-limited | keep HMC/LEDH closed | HMC readiness |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed: all engineering, finite target/status, independence, GPU, and state-hash gates passed. |
| Statistically supported ranking | None; three finite banks and one frozen state provide no uncertainty interval. |
| Descriptive-only differences | Bank A is a clear support outlier; B and C are descriptively closer to the old comparator. |
| Default readiness | Not ready for whitening, posterior, HMC, or canonical LEDH promotion. |
| Next evidence needed | A larger-N support envelope under the same frozen trainer, followed by a separately scoped proposal/objective repair only if support remains adequate. |

## Ledgers and red-team

Engineering correctness passed: the theta shape, target signature, protocol
receipts, hash separation, fresh-use flags, GPU memory policy, and exact state
reconstruction were checked. Numerical validity passed only for finite target,
score, transport parity, and status; it did not pass a whitening criterion.
Scientific interpretation remains role-limited because the sample banks are
finite and the old validation comparator is not an independent population
reference.

The strongest alternative explanation is that all banks share proposal-support
bias or that the old validation comparator is unrepresentative, rather than A
alone being unusual. A larger independent bank with stable support and
persistent residuals would weaken the isolated-draw explanation. The weakest
part of the evidence is the small number of banks and absence of a sampling
uncertainty model.

## Nonclaims and data-use record

No bank was pooled, dropped, tuned on, selected, or used for optimizer input.
This result establishes no IID Gaussian whitening, posterior correctness,
normalizer estimate, exhaustive mode discovery, HMC convergence, canonical
LEDH validity, superiority, or default promotion. The branch
`bank_a_isolated_outlier_descriptive` is a repair hypothesis, not a statistical
ranking.

The active continuation is refreshed to v2.6 in
`docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`.
