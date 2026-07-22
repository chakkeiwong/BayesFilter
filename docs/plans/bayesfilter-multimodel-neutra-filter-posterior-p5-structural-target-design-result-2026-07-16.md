# P5 Result: Structural Target Design

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Decision: `ADMIT_STRUCTURAL_TARGET_DESIGN_READY_FOR_R1B`

## Outcome

The Chapter 18b five-parameter structural UKF target design passed its
prospective domain, local likelihood-information, structural-identity,
negative-control, derivative, CPU-XLA, and trusted GPU-XLA gates. The final
T=100 synthetic dataset is frozen for the next R1B rung.

This result does **not** issue a typed posterior identity. It admits the design
for independent posterior recomposition, substitution-negative tests, and
typed identity issuance in R1B. No HMC or NeuTra is authorized by this result.

## Frozen Design

The physical parameters are `(rho,sigma,phi,gamma,R)` with prospective
independent Uniform supports
`(0.05,0.98)`, `(0.05,1.25)`, `(0.05,0.98)`, `(0.02,1.00)`, and
`(0.02,1.00)`. The five-probit chart includes the complete Jacobian. The
chapter calibration `(0.8,0.5,0.7,0.4,0.25)` is synthetic truth and the design
center, not a posterior default.

The final dataset uses TensorFlow stateless seed `(20260716,15001)`, horizon
100, and the time order `x0` from the initial law followed by `y0`, then 99
structural transitions and observations.

- State tensor SHA-256:
  `fe77f0e0000db93281116e7e81ddd303e9706b9e402bfaf7141a1aa1005c0ca9`.
- Observation tensor SHA-256:
  `ab7885b135d8098c6e516e06733ef99399ea07f4a39292670b578da4a0efbae3`.
- Maximum generator deterministic residual: roundoff-level and below the
  declared gate.

## Evidence

CPU/XLA design artifact:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p5/STR-UKF/target-design/attempt-03/result.json`.

- Result SHA-256:
  `214c6ba1e79d6589978b233a75015457ea08888e06d26d84203098d2736c4103`.
- Run-manifest SHA-256:
  `25939b75d755052e88397a863eb2b1c9557845e30c3e6194efcc9d4ed01babac`.
- Wall time: `2.60` seconds after XLA cache warm-up in the focused preflight.

Trusted GPU/XLA canary:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p5/STR-UKF/gpu-canary/attempt-01/gpu_canary.json`.

- SHA-256:
  `b64932ed68abe6a7df5bb5548f53820d68916d26e8d930fc399a21d3312f7944`.
- Device: NVIDIA GeForce RTX 4080 SUPER.
- XLA JIT: enabled; value, score, and status outputs on GPU.
- Memory growth: enabled before logical-device initialization.
- Both canary rows: finite value/score, status zero, structurally valid.
- Wall time: `6.45` seconds.

### Likelihood-Information Gate

For three disjoint design trajectories and eleven source points per trajectory,
every T=100 likelihood-only information matrix passed:

| Design seed | Smallest T=100 minimum eigenvalue | Largest T=100 condition number | Fine/coarse derivative step gaps |
| --- | ---: | ---: | ---: |
| `(20260716,15101)` | `0.26344` | `915.94` | mean `3.71e-6`; log variance `4.43e-6` |
| `(20260716,15102)` | `0.35873` | `1248.29` | mean `5.96e-6`; log variance `1.21e-5` |
| `(20260716,15103)` | `0.64240` | `1694.10` | mean `2.73e-5`; log variance `1.70e-5` |

All 33 matrices had numerical rank five, PSD-valid eigenvalues, minimum
eigenvalue above `0.10`, and condition number below `1e6`. Accumulated matrices
were nondecreasing from T=50 to T=100 to T=200. These are local
likelihood-information screens in source coordinates, not proof of global or
practical identifiability.

### Prior-Predictive Domain Gate

All 4,096 physical-prior trajectories were finite and below magnitude `1e6` at
T=50, T=100, and T=200. The observed maximum magnitudes were `202.90`, `314.30`,
and `391.82`, respectively. The 100% pass rate clears the numerical/domain
gate; it is not prior calibration.

### Structural Negative Control

The intended route uses one scalar innovation and every propagated point obeys
`k_t-phi*k_(t-1)-gamma*m_t^2=0`. The diagnostic-only route has an explicit
independent `eta_k ~ N(0,0.04)`, a two-dimensional innovation contract, and
off-manifold residual equal to `eta_k`. It reproduced the chapter change from
innovation variance `0.6121674304` to `0.6521674304` and log likelihood
`-0.7029747609` to `-0.7328186210`. It remains permanently ineligible for
posterior identity, HMC, training, or fallback.

## Attempt History

| Attempt | Classification | Scientific output |
| --- | --- | --- |
| 01 | `HARNESS_EVIDENCE_PERSISTENCE` | none; empty attempt root preserved |
| 02 | `HARNESS_CPU_XLA_TENSORLIST_VARIANT_UNSUPPORTED` | none; failed on first information-graph compile |
| 03 | completed reviewed repair | target design admitted |

The information diagnostic now evaluates all central-FD perturbations in one
batched XLA graph and requires fine/coarse step stability. The production
likelihood score remains the manual analytic principal-square-root path and
passed its separate centered-FD regression.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit target design for R1B | all prospective design gates passed | no structural, information, domain, derivative, XLA, or negative-control veto | local information and one final synthetic fixture | independently recompose posterior, run substitutions, issue typed identity only on pass | posterior correctness, global identifiability, filter exactness, HMC, NeuTra, Zhao-Cui, calibration, readiness |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Result |
| --- | --- |
| Engineering correctness | graph-native simulator, manual score, batch permutation, centered FD, CPU XLA, GPU XLA, memory growth, no active NumPy/callback/Python time loop passed |
| Numerical validity | finite prior-predictive domain, PSD/nondecreasing information, derivative step stability, finite value/score/status, and roundoff structural residuals passed |
| Scientific interpretation | the five-parameter T=100 structural UKF target is sufficiently specified and locally informative to attempt posterior identity admission; no posterior or sampler claim yet |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Clear for target design. |
| Statistically supported ranking | None; no method ranking was run. |
| Descriptive-only differences | Information spectra, prior-predictive magnitudes, runtime, truth location, and negative-control numerical gap beyond its semantic detector role. |
| Default-readiness | Not established. |
| Next evidence needed | R1B target contract, independent recomposition, total source-coordinate score FD, substitution negatives, and typed identity. |

## Post-Run Red Team

The strongest alternative explanation is that local information around the
truth and its fixed neighbors misses remote weakly identified or multimodal
regions inside the broad prior boxes. R1B can establish algebraic target
identity but cannot eliminate that concern; later same-target HMC diagnostics
and posterior exploration are required. Another dataset may also be materially
less informative. The weakest evidence is global identifiability and
cross-fixture robustness, while the strongest evidence is the exact structural
semantics, negative-control separation, and prospective multi-trajectory local
screen.
