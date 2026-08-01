# P6 R1B Result: SIR-SGQF Posterior Identity

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `POSTERIOR_IDENTITY_ADMITTED`

## Decision

`SIR-SGQF` has repository-issued typed posterior signature:

`0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc`.

It binds the frozen `y1:y20` observations, three log coordinates, independent
`Normal(0,0.5^2)` prior, zero identity-chart Jacobian convention, unprojected
SIR transition, level-2 37-point SGQF likelihood, manual total score, status
surface, and repository source closure. It does not establish HMC or NeuTra.

## Evidence

CPU identity result:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/r1b-identity/cpu-attempt-01/result.json`

- SHA-256: `dd058001d715c0b54be1c5bbd8f51bed5a3a882d02ca89894945ee9138c0e71a`.
- Target identity file SHA-256:
  `820b94f5158d0db95b9f3ad075d564eef8d0d8a9259b82404093d824d3281c5c`.
- Independent value and score recomposition gaps were exactly zero.
- Maximum analytic/fine FD gap was `1.78e-5`; fine/coarse gap was `5.35e-5`.
- Batch permutation gaps were zero and eager replay was exact.

Trusted GPU replay:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/r1b-identity/gpu-attempt-02/result.json`

- SHA-256: `5cca9efae6147dbdcbd5ad12d0371451b58b6d26cc879ad1c267c0f40d100ea2`.
- Identity and recomposition files are byte-identical to CPU.
- CPU/GPU value and score gaps were `8.03e-15` and `3.75e-13`
  scale-normalized; statuses matched exactly.
- Value, score, and status were GPU-resident; memory growth and TF32 passed.
- Every recursive artifact hash and the single legitimate campaign transition
  event were recomputed successfully.

## Substitution Negatives

Changed observation hash, prior scale, time order, and observation covariance
exponent changed the mathematical signature. Omitting or duplicating the prior
changed posterior values by at least `2.677`. The zero identity-chart Jacobian
convention is signature-bound; it is not falsely claimed to be numerically
detectable when omitted or duplicated.

## Repair History

GPU attempt 01 passed the numerical replay but compared in-memory tuples to
JSON lists and therefore withheld admission. GPU attempt 02 compared
JSON-normalized payloads and passed. No mathematical or numerical setting
changed.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit `SIR-SGQF` identity | recomposition, FD, substitutions, CPU/GPU XLA and exact identity replay passed | clear | nonlinear approximate-posterior geometry | run target-specific geometry and same-target plain HMC | HMC convergence, NeuTra, exactness, superiority, calibration, readiness |

