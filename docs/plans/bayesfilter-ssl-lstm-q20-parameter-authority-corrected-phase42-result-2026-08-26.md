# Corrected Parameter-Authority Phase 42 Result

Date: 2026-08-26  
Continuation version: `v2.4-independent-bank-replication`  
Status: `PASS_V2_4_TWO_BANK_REPLICATION_REPORT_REPAIR_TRIGGERED`

## Scope and target

Phase 42 evaluated two independently generated N=256 C0/M0 theta banks with
one frozen-training NeuTra state per arm. The declared target is
`theta in R^4`; the q=20 SSL-LSTM target's 60-dimensional UKF state and
20-dimensional innovation remain internal computations. The old v2.2
root-group-stratified N=256 M0 training rows and normalized weights were the
only optimizer input. Banks A and B were evaluated only after the final update,
with no checkpoint selection, pooling, dropping, or tuning.

## Evidence and artifact ledger

| Artifact | Status | Role |
|---|---|---|
| Fresh bank A, seed `(20260826, 4102)` | `PASS_THETA_MEASURE_PILOT` | independent theta support diagnostic |
| Fresh bank B, seed `(20260826, 4103)` | `PASS_THETA_MEASURE_PILOT` | independent theta support diagnostic |
| Shared GPU/XLA audit | `PASS_V2_4_TWO_BANK_BOUNDARY` | finite/batch/measure/trainer boundary |
| Read-only report | `PASS_V2_4_TWO_BANK_REPLICATION_REPORT` | bank-specific comparison and branch |
| Focused weighted-training regression | passed (`3 passed, 28 deselected`) | validates distinct static validation sizes |

Primary roots:

- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/fresh-a-n256/`
- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/fresh-b-n256/`
- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/frozen-two-bank-audit/`
- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase42-independent-bank-replication/report/`

The GPU audit wall time was `481.103434 s`; memory growth and XLA gates passed.
The CPU report is a read-only, hidden-GPU diagnostic. All pilot, tensor, target,
M0, and C0 hashes are recorded in the manifests and are distinct where the
protocol requires independence.

## Support receipt

| Source | ESS fraction | weighted negative-mode fraction | roots | theta mean[0] |
|---|---:|---:|---:|---:|
| old authority | 0.952283 | 0.530069 | 122 | 0.289568 |
| bank A | 0.801812 | 0.756588 | 103 | 3.550030 |
| bank B | 0.946687 | 0.517590 | 128 | 1.013180 |

Bank A is materially more negative-mode-heavy and has lower ESS than bank B;
bank B is closer to the old authority on these descriptive support summaries.
These quantities diagnose finite-bank variability and do not establish target
coverage or mode discovery.

## Frozen-state transport comparison

Residuals are maximum absolute latent weighted mean and maximum absolute
off-diagonal latent covariance. The old validation row is retained only as a
descriptive historical comparator.

| Arm | old validation mean/cov | bank A mean/cov | bank B mean/cov |
|---|---:|---:|---:|
| identity:compact | 0.547410 / 0.893340 | 0.914067 / 0.679319 | 0.198587 / 0.344945 |
| identity:wide_low_lr | 0.583848 / 1.012230 | 1.045722 / 0.774507 | 0.247726 / 0.377084 |
| affine:compact | 0.424126 / 0.655261 | 1.138738 / 0.896074 | 0.199921 / 0.390038 |
| affine:wide_low_lr | 0.462624 / 0.637627 | 1.099791 / 0.661917 | 0.096337 / 0.208018 |

Bank B is descriptively lower than the old validation comparator for both
metrics in every arm. Bank A is worse in mean residual for every arm and is
worse in covariance for the affine compact arm. The asymmetry prevents a clean
replication claim. The predeclared branch is
`bank_to_bank_variability_repair_triggered`.

## Decision and inference status

| Decision | Primary criterion | Status/veto | Next justified action | Not concluded |
|---|---|---|---|---|
| retain theta target | finite target/status/measure/protocol gates | pass | continue support diagnostic | posterior correctness |
| promote IID whitening | bank-specific residuals | veto: material residuals and variability | add third bank or larger-N diagnostic | IID Gaussian law |
| change objective/architecture | replicated support evidence | defer: two-bank asymmetry | keep target/objective fixed for Phase 43 | objective superiority |

| Inference class | Status |
|---|---|
| hard veto screen | passed; no v2.4 harness repairs |
| statistically supported ranking | none |
| descriptive-only differences | all support, ESS, mode, loss, and residual differences |
| default/HMC/LEDH readiness | not ready; no such route launched |
| next evidence | third independent N=256 bank with exact v2.4 state-hash reconstruction |

## Ledgers and red team

Engineering correctness passed: all four arms had finite traces, `[N,4]`
batch shapes, affine training-measure oracles, transport round trips, valid
q=20 target/status receipts, GPU memory growth, and XLA. Numerical validity is
limited to this finite boundary; it does not identify a density or prove target
coverage. Scientific interpretation is a repair trigger only.

The strongest alternative explanation is that bank A is a proposal/mode
outlier and the residual spread is finite-support noise. A third bank with
A-like behavior would weaken that explanation; a larger-N bank with stable
support would test a different particle-count hypothesis. The weakest evidence
is the two-bank sample with one frozen state and no uncertainty interval.

## Nonclaims and next phase

No posterior correctness, IID Gaussian whitening, exhaustive mode discovery,
normalizer, HMC convergence, canonical LEDH status, statistical superiority,
or default promotion follows from this result. Phase 42 did not invalidate the
target, measure contract, or harness; it only triggered the next support
diagnostic. Phase 43 is authorized under the unchanged target, proposal
protocol, objective, hardware class, and campaign budget.
