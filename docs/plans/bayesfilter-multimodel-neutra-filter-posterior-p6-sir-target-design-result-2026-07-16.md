# P6 SIR Target-Design Result

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `CELL_SPLIT_SIR_SGQF_READY_FOR_R1B`

## Decision

The common three-log-scale SIR extension and its fixed level-2 SGQF filter
passed target design. Only `SIR-SGQF` may continue to R1B posterior identity.

The other cells remain blocked at this rung:

| Cell | Terminal target-design state | Exact reason |
| --- | --- | --- |
| `SIR-SGQF` | `TARGET_DESIGN_READY_FOR_R1B` | CPU and trusted GPU/XLA value, score, status, derivative, curvature, cloud, data, replay, and substitution gates passed |
| `SIR-UKF` | `IMPLEMENTATION_BLOCKED_GPU_SCORE_PARITY` | trusted GPU score gap was `5.97e-7` scale-normalized against the frozen `1e-7` limit |
| `SIR-ZC` | `TARGET_BLOCKED_MISSING_OBSERVED_DATA_SCORE_ROUTE` | retained-marginal and proposal-transport derivative closures remain absent; the three-parameter target is an extension of the fixed-parameter source example |

No UKF or Zhao-Cui blocker is promoted to a program-wide continuation veto.

## Evidence

Active CPU result:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-common/target-design/cpu-attempt-04/result.json`

- SHA-256: `5d0d73f302b160b9f1277cd4ab5ef22ad53200f2c156cf6395d1e6a4ba0f9852`.
- Dataset hashes matched the frozen `x0:x20` and `y1:y20` identities.
- All 4,096 prior-predictive trajectories were finite and below magnitude
  `1e6`; the all-nonnegative susceptible fraction was `0.80298` and is
  explanatory support telemetry for the unprojected Gaussian transition.
- Four 4,096-particle PF likelihood estimates at truth were finite, with mean
  `-681.599` and standard deviation `0.656`; this is explanatory only.
- The 18-dimensional level-2 cloud has 37 points, one negative center weight,
  exact zero mean and identity covariance at the recorded tolerance.

Active trusted GPU canary:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-common/gpu-canary/attempt-02/gpu_canary.json`

- SHA-256: `51d61ea606521fe553555792ff771c1810424344bdcae2c300e42344731716b9`.
- CPU-result binding, manifest, memory growth, TF32, GPU residency, XLA, and
  recursive artifact hashes are present.
- `SIR-SGQF` GPU/CPU value gap was `4.94e-15` scale-normalized and score gap
  was `1.89e-13`; statuses matched exactly.
- `SIR-UKF` values and statuses passed, but its score gap was `5.97e-7`, so it
  remains blocked without a post-result threshold change.

For `SIR-SGQF`, same-mode centered FD passed with maximum analytic gap
`1.78e-5` absolute and `2.34e-7` scale-normalized. All 27 observed-curvature
matrices had rank three and fine/coarse relative Frobenius gap at most about
`1.75e-8`. Batch permutation, replay, CPU-XLA, static-source, negative
substitution, and status gates passed. Its mathematical target signature is:

`43968c975409021dcabe931081f0d1efaaae431b5b9245929a5786fe566e545d`.

## Repair History

| Attempt | Classification | Outcome |
| --- | --- | --- |
| CPU 01 | `INFRASTRUCTURE_HARNESS_MEMORY_TOPOLOGY` | stopped before filter evidence; oversized derivative stencil replaced |
| CPU 02 | `INFRASTRUCTURE_XLA_NESTED_LOOP_COMPILATION_MEMORY` | stopped before filter evidence; bounded six-row XLA scheduling introduced |
| CPU 03 | `PLAN_AND_HARNESS_DIAGNOSTIC_MODE_MISMATCH` | completed but mixed eager scores with XLA FD values and imposed an impossible nonnegative-state veto |
| CPU 04 | completed | active corrected target-design result |
| GPU 01 | reporting incomplete | numerical split preserved; missing manifest/hash evidence repaired |
| GPU 02 | completed | active trusted GPU/XLA canary |

The UKF score itself was not the attempt-03 defect. At the worst audit point it
matched raw TensorFlow autodiff within `1.5e-10` and a same-mode eager FD ladder
within about `1e-6`. XLA value roundoff had been divided by the FD step. The
visible repair separated derivative correctness from compiled parity and did
not change the target or numerical score implementation.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue `SIR-SGQF` | target design and CPU/GPU gates passed | clear for R0/R1 only | approximate filter posterior and global geometry | execute independent R1B recomposition and typed identity | posterior correctness, HMC convergence, NeuTra, filter exactness or superiority |
| Block `SIR-UKF` | eager value/score valid | GPU score parity veto fired | source of device numerical drift | preserve blocker for a separate focused repair | UKF target invalidity or scientific failure |
| Block `SIR-ZC` | source boundary checked | observed-data derivative closure missing | extension route not designed | no HMC or training | Zhao-Cui parameter-posterior capability |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Clear for `SIR-SGQF`; fired for UKF GPU parity and Zhao-Cui derivative closure. |
| Statistically supported ranking | None. |
| Descriptive-only differences | PF estimates, UKF/SGQF value gaps, curvature eigenvalues, runtimes. |
| Default-readiness | Not established. |
| Next evidence needed | `SIR-SGQF` R1B identity, then same-target plain-HMC comparator before any training. |

## Post-Run Red Team

Passing target design establishes that the implementation consistently computes
the declared deterministic SGQF approximate posterior ingredients. It does not
establish that SGQF accurately represents the exact nonlinear SIR posterior.
The strongest alternative explanation for later sampling failure is difficult
posterior geometry or approximation-induced tails, not target assembly. R1B
must still bind the Gaussian prior, zero identity-chart Jacobian, likelihood,
data, filter, and source closure into one repository-issued identity.

