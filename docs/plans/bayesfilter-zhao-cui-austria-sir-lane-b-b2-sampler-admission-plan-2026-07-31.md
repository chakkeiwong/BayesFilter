# Zhao-Cui Austria SIR Lane-B B2 Sampler Admission Plan

Date: 2026-07-31

Status: `ACTIVE_EXECUTION`

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | Does the selected T1 artifact define a deterministic, finite, correctly scored retained-state sampler that can supply T2 without changing the admitted T1 scalar? |
| Candidate | A Lane-B-only inversion of exact piecewise-polynomial interval masses on retained axes `0:18`, with proposal density equal to the exact piecewise-linear-CDF inversion law. |
| Expected failure | The grid law is non-monotone, its conditional mass is inaccurate, the reported density differs from the inversion Jacobian, replay changes, correction weights are non-finite, or the T1 frame does not preserve the retained prefix. |
| Promotion criterion | Fresh-process exact replay plus inverse/forward residual `<=2e-12`, proposal/Jacobian log-density residual `<=2e-12`, maximum raw conditional-mass residual `<=5e-10`, finite `log(p_TT/q_grid)`, static workspace below 512 MiB, and CPU process peak below 12 GiB for the sealed 64-sample diagnostic. |
| Promotion veto | Scoring samples with the fitted TT density instead of the actual grid law; changed T1 artifact/hash; non-finite or non-positive interval mass; frame-prefix dependence on marginalized coordinates; tolerance change after seeing the claim; or memory breach. |
| Continuation veto | No finite correctly scored proposal can be constructed from the admitted T1 retained density within the memory bound. |
| Repair trigger | Local table inversion, serialization, device-placement, or harness failure under the unchanged mathematical law. |
| Explanatory diagnostics | `log(p_TT/q_grid)` spread, ESS, raw quadrature mass residual, runtime, and TT-versus-grid density difference. |
| Must not be concluded | No production KR closure, source-faithful conditional-CDF implementation, T2 value, score, T20, HMC, posterior correctness, or superiority. |

## Mathematical Contract

For retained local coordinates `r=(r_1,...,r_18)` and reference coordinates
`u_k=r_k/sqrt(1+r_k^2)`, let `p_TT` be the normalized T1 retained density.
At axis `k`, the selected finite sampler partitions `[-1,1]` into the 64
intervals already bound by the selected artifact's grid size. The selected
piecewise Lagrange basis has degree two on each of two elements, so the squared
TT conditional is degree at most four on each element. Every interval stays
inside one element. Three-point Gauss-Legendre quadrature therefore evaluates
each interval mass exactly up to FP64 contraction error:

\[
  m_j = \tfrac12 \int_{u_j}^{u_{j+1}}h(u)\,du,
  \qquad \Delta F_j=m_j/\sum_l m_l.
\]

It inverts the piecewise-linear CDF. Therefore the exact density of the
implemented reference-coordinate proposal inside interval `j` is

\[
  q_{u,k}=\Delta F_j/(u_{j+1}-u_j),
\]

not the interpolated TT conditional. The physical retained proposal is

\[
  \log q_z(z_1)=\sum_k\log q_{u,k}
    +\sum_k\log|du_k/dr_k|-\log|\det L_{11}|.
\]

The retained correction carried into later use is exactly
`log p_TT(z1)-log q_z(z1)`.

## Source And Classification Ledger

| Operation | Classification | Anchor |
|---|---|---|
| Squared-TT retained marginal | `source_faithful` | Zhao-Cui Proposition 2; author `@TTSIRT/marginalise.m:19-85` |
| Upper conditional dependency order | `source_faithful` operation | Zhao-Cui Eq. (20)-(21), Algorithm 3; author `@TTSIRT/eval_irt_reference.m:43-71` |
| Exact piecewise-quartic interval mass and exact table-law score | `extension_or_invention` | Repository finite numerical definition above; author `LagrangepCDF.m` independently confirms polynomial-CDF treatment; no production-KR claim |
| Frozen reference array and replay | `fixed_hmc_adaptation` | Freezes the paper's uniform draws; does not authorize HMC |

## Evidence Contract

The exact baseline is selected T1 artifact
`e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`.
The runner will preserve a versioned JSON result, the exact serialized
reference tensor, hashes of samples/weights/proposal/target values, device and
memory posture, command, Git state, source closure, and wall time.

The claim is engineering and numerical admission of one finite sampler law.
Raw mass and weight diagnostics can veto but cannot establish production KR or
scientific correctness.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| 65 grid nodes / 64 intervals | Selected T1 artifact; frozen partition | interval crosses a polynomial element or quadrature algebra is wrong | assert the zero breakpoint is a node and gate exact conditional mass |
| 64 reference columns | Bounded diagnostic hypothesis | misses rare bad prefixes | all axes checked plus later T2 validation remains required |
| Seeds 73702 / 73703 | Calibration / untouched claim after invalid attempt 01 | replay drift or tuning on claim | calibrate implementation only on 73702; serialize 73703 and require exact hashes |
| Eager FP64 CPU | Master-plan numerical-reference lane | CPU behavior does not prove final GPU/XLA route | record `CUDA_VISIBLE_DEVICES=-1`; later unchanged-program GPU/XLA gate remains required |
| 512 MiB static / 12 GiB process cap | Existing microbatch/reference contracts | hidden batch/grid growth | static estimate and `ru_maxrss` peak |

## Skeptical Pre-Execution Audit

| Risk | Finding |
|---|---|
| Wrong baseline | The selected T1 identity is required exactly; no P88 or retained-grid substitute is allowed. |
| Proxy promotion | Roundtrip is not enough. The proposal density must also equal the exact inversion Jacobian. |
| Hidden assumption | Cholesky `L` is lower triangular, so physical `z1` depends only on local axes `0:18`; this is asserted at runtime. |
| Unfair comparison | No cross-method comparison occurs in B2. |
| Stale context | B3 passed, but the original B2 sampler requirements were not executed; this plan closes only that gap. |
| Environment mismatch | The claim runner hides CUDA before TensorFlow import and records the explicit FP64 CPU reference exception. Final GPU/XLA evidence remains later. |
| Non-answering artifact | Exact reference/sample/proposal/target/weight hashes and a fresh replay directly answer determinism and scoring-law questions. |

Audit verdict before attempt 01: `PASS_FOR_EXECUTION`. The earlier
proposal-density ambiguity was resolved by scoring the exact table law and
retaining an explicit TT/grid correction.

## Attempt-01/02 Refresh

Attempt 01 used the original trapezoid candidate on trusted GPU and failed with
`NONFINITE_VALUE`. Attempt 02 localized non-finite FP64 GPU marginal
contractions at inconsistent axes and batch sizes. It also showed raw
trapezoid conditional-mass residuals as large as `1.2356`, which rejects the
trapezoid candidate independently of the GPU failure.

Classification:

- the GPU failure is a backend repair trigger, not evidence against the T1 TT;
- the large mass residual is candidate rejection, not a continuation veto;
- B3's independent FP64 CPU contraction remains passed;
- seed 73701 is consumed by the invalid candidate and cannot be reused.

The refreshed exact interval-mass program follows from the checked polynomial
degree and does not select a resolution or tolerance from claim data. It uses
calibration seed 73702 only for an engineering run, then freezes untouched seed
73703 before the claim. Refreshed audit verdict: `PASS_FOR_EXECUTION`.

## Budget And Stop Conditions

- focused CPU-hidden unit tests: at most four launches, twenty minutes total;
- one CPU calibration, one untouched CPU claim, and one fresh replay, twenty
  minutes each;
- output root:
  `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-b2-sampler-20260731/attempt-NN/`;
- stop on a mathematical/identity/measure veto, exhausted repair, or resource
  breach; otherwise refresh B4 and continue.
