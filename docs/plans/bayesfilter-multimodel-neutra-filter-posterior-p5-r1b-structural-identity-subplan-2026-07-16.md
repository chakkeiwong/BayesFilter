# P5 R1B Subplan: Structural Posterior Identity Admission

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `REVIEWED_READY_FOR_EXECUTION`

## Objective And Entry Conditions

Bind the admitted T=100 structural design to a complete repository-issued typed
posterior identity. R1B must implement the adapter/contract and independently
recompose physical prior, structural UKF likelihood, complete five-probit
Jacobian, total unconstrained value, and total score. No HMC or NeuTra is in
scope.

Entry requires target-design result SHA-256
`214c6ba1e79d6589978b233a75015457ea08888e06d26d84203098d2736c4103`,
GPU-canary SHA-256
`b64932ed68abe6a7df5bb5548f53820d68916d26e8d930fc399a21d3312f7944`,
state hash `fe77f0e0000db93281116e7e81ddd303e9706b9e402bfaf7141a1aa1005c0ca9`,
observation hash
`ab7885b135d8098c6e516e06733ef99399ea07f4a39292670b578da4a0efbae3`, and no
post-design change to model, boxes, chart, data, filter, time order, dtype, or
regularization.

## Evidence Contract

| Field | Frozen R1B contract |
| --- | --- |
| Question | Does one repository-owned adapter compute and bind the complete declared structural UKF posterior in source coordinates? |
| Baseline | Independently called physical-prior, likelihood, and chart-Jacobian value/score components |
| Primary pass | exact target/data/source binding, independent recomposition at fixed audit points, total score centered-FD, batch permutation, CPU/GPU XLA, valid status, and all wrong substitutions rejected |
| Hard vetoes | data/chart/prior/filter/time-order/source hash drift; omitted or duplicated Jacobian; artificial-noise route accepted; caller-stamped identity; nonfinite/invalid score; FD/recomposition failure; active NumPy/callback/Python loop |
| Explanatory only | posterior value magnitude, truth score, runtime, local condition diagnostics |
| Not concluded | posterior correctness beyond the declared approximate-filter target, filter exactness, HMC convergence, NeuTra, global identifiability, calibration, robustness, or readiness |

## Required Implementation And Checks

1. Add `StructuralUKFNeuTraAdapter`, an independent likelihood recomposer, and
   an `SSMTargetContract` whose manifests bind model equations, final dataset
   hashes, initial law, parameter boxes/order, five-probit chart, physical
   Uniform prior, principal-square-root structural UKF, scalar innovation,
   structural residual policy, no jitter/floor in the mathematical target, and
   all source dependencies.
2. The adapter value is exactly `likelihood + physical prior + log Jacobian` and
   its score is the same sum. Identity issuance must use repository-owned source
   closure and stable contract signature; callers cannot stamp a signature.
3. Evaluate fixed points: truth source, zero, ten `truth +/- 0.5 e_i` neighbors,
   and two fixed tail points. Require independent recomposition value gap
   `<=1e-9` and score gap `<=1e-8`.
4. Require centered total-score FD at steps `1e-5` and `5e-6`; analytic/FD gap
   `<=3e-5` and cross-step gap `<=1e-5` at truth, zero, and fixed neighbors.
5. Require batch permutation equality, typed target reload, CPU XLA, trusted
   GPU XLA with memory growth, status zero, deterministic residual at roundoff,
   and recursive artifact hashes.
6. Negative substitutions must reject: artificial `eta_k` route/signature,
   changed observation hash, changed horizon/time order, missing Jacobian,
   duplicated Jacobian, changed prior box/order, changed R variance semantics,
   and cross-target transport/adapter identity.
7. Static-source checks must find no active NumPy, host callback, scalar row
   mapping, or Python batch/time loop.

## Result, Handoff, And Stops

On pass, issue the typed `STR-UKF` signature, move only `STR-UKF` to
`POSTERIOR_IDENTITY_ADMITTED`, write the R1B result/manifest/hash/cell ledger,
and draft the same-target plain-HMC comparator subplan. `STR-ZC` remains
`TARGET_BLOCKED_EXTENSION_ROUTE_NOT_DESIGNED`; its later design cannot reuse the
UKF signature.

On any target/recomposition/substitution failure, stop `STR-UKF` at
`TARGET_BLOCKED` or `IMPLEMENTATION_BLOCKED` with the exact class. A localized
serialization or XLA harness failure may be repaired in a fresh root under the
unchanged contract. Stop after three materially identical repairs or 4 CPU-hours
plus 1 trusted GPU-hour in R1B.

## Skeptical Pre-Execution Audit

Decision: `PASS`.

The design result is not being promoted directly to a posterior identity.
R1B's baseline is independent component recomposition, its score check is total
and source-coordinate, and the negative-control route is a substitution veto
rather than a comparator. HMC and training remain forbidden until typed
identity admission. The fixed audit set covers center, truth, local axes, and
tails without tuning thresholds after inspection. Claude review remains
platform-unavailable for the private workspace; this one-path subplan received
a second local audit under the documented proportional-review policy.
