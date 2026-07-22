# P5 R1B Result: Structural Posterior Identity

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `POSTERIOR_IDENTITY_ADMITTED`

## Decision

`STR-UKF` has a repository-issued typed posterior identity:

`e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665`.

The identity binds the frozen T=100 observations, five-probit chart, physical
Uniform prior, complete Jacobian, scalar-innovation structural model,
principal-square-root UKF likelihood, manual source-coordinate score, status
surface, and repository dependency closure. It does not authorize or imply HMC
convergence or NeuTra quality.

## Evidence

CPU-XLA reference:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p5/STR-UKF/r1b-identity/cpu-attempt-02/`.

- Result SHA-256:
  `73fd7a10fd89999993b2b88b636774df489e984e1c589cb3efff57ce2d3ea83d`.
- Target identity SHA-256:
  `c42bb40cecffc36fa1eda50800d2d3bd472bc43601439258d9cd05c030c9e425`.
- Recomposition SHA-256:
  `2ddb5825756285b9345d7ed1e52518c8d1bffa007de72706b10d15b4803d5833`.
- Independent value and score recomposition gaps: exactly zero.
- Maximum analytic versus fine-FD score gap: `1.44504e-7`.
- Fine/coarse FD step gap: `4.39115e-7`.
- Batch permutation value/score gaps: exactly zero.
- Wall time: `63.33` seconds.

Trusted GPU-XLA replay:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p5/STR-UKF/r1b-identity/gpu-attempt-03/`.

- Result SHA-256:
  `f36f9197c56b2bc88276b234c6aa0e25ea992220511272d88cece86918a910f3`.
- Target identity and recomposition files are byte-identical to CPU.
- Maximum CPU/GPU value gap: `5.68434e-14`.
- Maximum CPU/GPU score gap: `2.09610e-13`.
- Value, score, and status outputs are GPU-resident; memory growth passed.
- Wall time: `173.85` seconds.

Every recursive artifact hash in both final roots was recomputed successfully.

## Substitution Negatives

The following substitutions were rejected:

- changed observation hash;
- explicit two-innovation artificial `eta_k` route;
- changed `R` chart upper bound;
- changed prior parameter order;
- omitted chart Jacobian; and
- duplicated chart Jacobian.

All contract substitutions changed the mathematical signature. Omitting or
duplicating the Jacobian changed posterior values by at least `8.6529` over the
fixed audit set. The artificial-noise route has innovation dimension two and is
ineligible for the scalar-innovation registry binding.

## Repair History

| Attempt | Classification | Effect |
| --- | --- | --- |
| CPU 01 | passed before source repair | superseded because later CPU-pinned dataset replay changed bound source closure |
| GPU 01 | `HARNESS_DEVICE_DEPENDENT_DATASET_REPLAY` | no identity work; frozen dataset generation moved explicitly to CPU |
| CPU 02 | completed | issued the active reference identity |
| GPU 02 | numerically passed but `EVIDENCE_IDENTITY_DEVICE_DRIFT` | audit-point inverse-normal values differed at serialized-bit level by device |
| GPU 03 | completed | CPU-pinned audit points reproduced CPU identity exactly |

No target, data, threshold, score tolerance, or posterior component changed in
these repairs.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit `STR-UKF` posterior identity | recomposition, total score, substitutions, CPU/GPU XLA, and identity replay passed | clear | approximate filter target and possible remote posterior geometry | tune and run a same-target plain-HMC comparator | HMC convergence, NeuTra, filter exactness, global identifiability, calibration, robustness, readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Clear for posterior identity and value/score admission. |
| Statistically supported ranking | None. |
| Descriptive-only differences | CPU/GPU roundoff, runtime, posterior values and scores at audit points. |
| Default-readiness | Not established. |
| Next evidence needed | Separate warm-up and retained same-target plain HMC with modern R-hat/ESS and health gates. |

## Post-Run Red Team

Independent recomposition proves that the implemented components form the
declared approximate-filter posterior; it does not prove that the UKF posterior
is an exact representation of the nonlinear model or globally well behaved.
The strongest alternative explanation for later sampler failure would be
posterior geometry or multimodality rather than target assembly. The current
weakest evidence is global posterior exploration, which the comparator rung is
designed to address.
