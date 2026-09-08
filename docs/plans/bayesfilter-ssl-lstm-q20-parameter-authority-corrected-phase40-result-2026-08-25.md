# Corrected Parameter-Authority Phase 40 Result

Date: 2026-08-25  
Continuation version: `v2.2-root-group-stratified`  
Status: `PASS_V2_2_ROOT_GROUP_BOUNDARY_MEASURE_DIAGNOSTIC_REPAIR_TRIGGERED`

## Scope and question

Phase 40 replaced the v2.1 storage-order split with a deterministic
root-group-stratified split on the fixed N=256 M0 theta bank. Whole SMC root
groups were allocated to train, validation, and audit; no root was allowed to
cross partitions. Identity and exact training-measure affine GPU/XLA traces
were rerun with the same target, proposal density, seed, 200-step budget, and
checkpoint grid.

The question was whether the earlier held-out residuals persisted after
removing ancestry leakage. This phase did not change the target, proposal, or
canonical LEDH boundary. It did not launch HMC.

## Preflight and repairs

The root-group allocator passed deterministic checks: 232 training rows, 12
validation rows, 12 audit rows, six negative and six positive rows in each
holdout, complete row coverage, and zero root overlap. The first identity
launch failed before training because the loader applied a floating-point
finiteness assertion to integer root IDs. The failed root is preserved at
`phase40-root-group-stratified-boundary/identity-trace/`. The loader was
repaired to check finiteness only for floating/complex tensors, and the fresh
`identity-attempt2/` root passed.

The historical v2.1 checkpoint reporter correctly rejected the v2.2 schema;
it was not weakened. A versioned v2.2 reporter was added and passed.

## Receipts

- identity: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase40-root-group-stratified-boundary/identity-attempt2/`
- affine: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase40-root-group-stratified-boundary/affine-trace/`
- checkpoint report: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase40-root-group-stratified-boundary/report-v2/`
- measure separation: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase40-root-group-stratified-boundary/measure-separation/`

Both traces bind target signature
`9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, M0
protocol hash
`a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631`, schema
`bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v3_root_group_stratified_split`,
and plan version `v2.2-root-group-stratified`. GPU memory growth was verified
on both visible RTX 4080 SUPER devices and XLA was enabled.

## Boundary and checkpoint result

| Precondition | Arm | selected step | validation score | selected audit mean | selected audit covariance | terminal audit mean | terminal audit covariance |
|---|---|---:|---:|---:|---:|---:|---:|
| identity | compact | 200 | 9.846842 | 0.520515 | 0.740395 | 0.520515 | 0.740395 |
| identity | wide, low LR | 200 | 10.257230 | 0.576445 | 0.818326 | 0.576445 | 0.818326 |
| affine, exact train measure | compact | 200 | 6.759979 | 0.513618 | 1.133955 | 0.513618 | 1.133955 |
| affine, exact train measure | wide, low LR | 150 | 7.105530 | 0.376715 | 0.631711 | 0.364732 | 0.649199 |

The checkpoint rule changed only the affine wide arm. Its covariance residual
decreased descriptively from `0.649199` to `0.631711`, while its mean residual
increased from `0.364732` to `0.376715`. The residuals remain material. No
whitening promotion is justified.

## Measure-separation result

The exact v2.2 split report gives:

| Partition | rows | roots | ESS fraction | negative-mode fraction | theta mean[0] | affine latent mean max | affine covariance off-diagonal max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 232 | 108 | 0.950551 | 0.533571 | 0.330821 | 0.000000 | 0.000000 |
| validation | 12 | 8 | 0.983826 | 0.491953 | 1.181110 | 0.537741 | 0.733591 |
| audit | 12 | 6 | 0.996452 | 0.491717 | -1.422781 | 0.399638 | 0.579953 |

The affine training oracle remains exact to floating-point precision (mean
maximum `4.51e-17`, covariance residual `1.78e-15`). The validation and audit
partitions are root-disjoint, yet their theta means and affine moments differ
substantially from training. Root disjointness removes ancestry leakage; it
does not make a finite holdout representative of the target measure.

## Decision tables

### Engineering, numerical, scientific ledgers

| Ledger | Status | Evidence | Limit |
|---|---|---|---|
| Engineering correctness | pass | v2.2 schema, target/status, shape, finite, root, XLA, and memory gates pass | no HMC or production route |
| Numerical validity | pass for finite receipts | affine train oracle and transport round trips pass | no density-identification or target-coverage proof |
| Scientific interpretation | repair trigger | split defect fixed; partition mismatch persists | no IID, posterior, mode, or superiority claim |

### Decision

| Decision | Primary criterion | Veto status | Next justified action | Not concluded |
|---|---|---|---|---|
| Retain theta target and parameter-space work | common theta measure and finite target remain valid | no continuation veto | evaluate an independent fresh audit bank | target correctness |
| Close v2.2 as a boundary diagnostic | root-disjoint split and receipts are auditable | whitening promotion veto remains open | do not tune further on this bank | universal split rule |
| Defer identity/affine promotion | audit residuals remain material | candidate promotion veto | compare against independent support under unchanged objective | HMC/LEDH |

### Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for both fresh traces and reports |
| Statistically supported ranking | none; one bank and one seed |
| Descriptive-only differences | losses, moments, ESS, checkpoint, and partition summaries |
| Default readiness | not ready |
| Next evidence needed | independent fresh theta bank used as audit data, with exact target/proposal signatures |

## Red-team and stop classification

The strongest alternative explanation is finite proposal support: the N=256
bank's root groups occupy separated theta regions, so any partition of it can
look unlike the training empirical measure. A second explanation is objective
misspecification in weighted forward-KL training. The independent-bank audit
will distinguish these: if a fresh bank still has large residuals under the
same frozen objective, support is less likely to be only a split artifact; if
the residual falls, the previous comparison was dominated by empirical-bank
partition mismatch.

Phase 40 did not invalidate the target, harness, or measure contract. No true
continuation veto fired. The next phase changes only the evidence boundary by
using a new independent theta bank; it does not relabel internal UKF states or
authorize canonical LEDH.

## Nonclaims

- No IID Gaussian whitening theorem or posterior correctness claim.
- No exhaustive mode-discovery, normalizer, SMC-U, HMC, or canonical LEDH claim.
- No statistical superiority or default promotion claim.
