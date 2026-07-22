# P6 R1B Subplan: SIR-SGQF Posterior Identity

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `REVIEWED_READY_FOR_EXECUTION`

## Objective And Entry Conditions

Bind the admitted T=20, three-log-scale, fixed level-2 SGQF filter posterior to
one repository-issued typed `SIR-SGQF` identity. Independently recompose the
observed-data SGQF likelihood, independent `Normal(0,0.5^2)` prior, and zero
identity-chart Jacobian. HMC and NeuTra remain out of scope.

Entry requires:

- CPU result SHA-256
  `5d0d73f302b160b9f1277cd4ab5ef22ad53200f2c156cf6395d1e6a4ba0f9852`;
- trusted GPU canary SHA-256
  `51d61ea606521fe553555792ff771c1810424344bdcae2c300e42344731716b9`;
- observation hash
  `cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07`;
- state hash
  `8cd5a079f5799f0e0b769e5ac21a4bdf460475a72319f07dc27fb037eb5774e0`;
- mathematical target signature
  `43968c975409021dcabe931081f0d1efaaae431b5b9245929a5786fe566e545d`;
- no post-design change to data, time order, three parameters, prior, chart,
  SGQF cloud/filter, transition, observation covariance, dtype, or score route.

## Evidence Contract

| Field | Frozen R1B contract |
| --- | --- |
| Question | Does one repository-owned adapter compute and bind the complete declared `SIR-SGQF` posterior in its three log coordinates? |
| Baseline | Independently called SGQF likelihood, Gaussian prior, and zero identity-chart Jacobian value/score components |
| Primary pass | exact target/data/source binding, independent recomposition, same-mode total-score FD, batch permutation/replay, CPU/GPU XLA, valid status, typed reload, and substitutions rejected |
| Hard vetoes | data/prior/chart/filter/time-order/source drift; omitted or duplicated prior accepted; nonfinite/invalid score; recomposition/FD/parity failure; active NumPy/callback/Python algorithmic loop |
| Explanatory only | posterior value/score magnitude, runtime, PF and cross-filter diagnostics |
| Not concluded | exact posterior correctness, HMC convergence, NeuTra quality, SGQF superiority, calibration, forecasting, robustness, or readiness |

## Required Checks

1. Use the repository-owned `SIRSGQFNeuTraAdapter`,
   `SIRSGQFLikelihoodRecomposer`, and `SSMTargetContract`; callers may not stamp
   a target signature.
2. At truth, all `+/-0.5` and `+/-1` axis points, require independent component
   recomposition gaps `<=1e-9` for value and `<=1e-8` for score.
3. Require same-mode centered total-score FD at steps `5e-5` and `1e-4` with
   analytic/fine and fine/coarse maximum absolute gaps `<=5e-3` and maximum
   scale-normalized gaps `<=5e-4`.
4. Require exact batch permutation/replay and status equality. CPU-XLA and
   trusted GPU-XLA require status equality and scale-normalized value/score
   gaps `<=1e-8` and `<=1e-7`, respectively. Record absolute gaps.
5. Reject changed observation hash/time order, wrong dtype, prior scale/order,
   observation covariance exponent, SGQF cloud/filter identity, and omitted or
   duplicated prior. The identity chart Jacobian is exactly zero: its convention
   must be signature-bound, but missing/duplicated zero cannot honestly be
   detected by numerical posterior values.
6. Require static-source checks, repository-issued identity reload, campaign
   event/ledger, exact run manifests, versioned roots, and recursive hashes.

## Handoff And Stops

On pass, issue only the `SIR-SGQF` typed identity, move it to
`POSTERIOR_IDENTITY_ADMITTED`, write the R1B result, and draft the same-target
plain-HMC comparator subplan. Do not reopen `SIR-UKF` or `SIR-ZC` from SGQF
evidence.

On target, recomposition, substitution, or score failure, block `SIR-SGQF` at
R1B. Local serialization/device-reporting failures may be repaired in fresh
roots under the unchanged contract. Stop after three identical failures or the
R1B budget of four CPU-hours plus one trusted GPU-hour.

## Skeptical Pre-Execution Audit

Decision: `PASS`.

The baseline is the complete posterior decomposition, not target-design values
or PF estimates. FD correctness is kept in one execution mode, while XLA is a
separate parity gate, preventing the attempt-03 diagnostic drift. The plan does
not pretend that a zero chart-Jacobian term can be detected numerically. GPU
memory growth, artifact binding, stop conditions, wrong substitutions, and
nonclaims are explicit. No proxy metric is promoted to posterior or NeuTra
evidence, and neither blocked sibling cell is allowed to inherit this identity.
