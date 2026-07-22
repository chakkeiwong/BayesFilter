# P6 Target-Design Subplan: Parameterized Austria SIR Filter Posteriors

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `REVIEWED_READY_FOR_EXECUTION`

## Objective And Entry Conditions

Freeze and test the scientific target needed before any P6 HMC or NeuTra work:
one three-parameter Austria SIR extension with one common T=20 dataset and
separate observed-data UKF and fixed-SGQF likelihood programs. Independently
audit whether the Zhao-Cui fixed-TTSIRT substrate can issue an observed-data
parameter-posterior identity.

Entry is the verified P5 close package. P0/P1 remain the shared registry/harness
foundation, but their SIR rows are inventory identities only. Existing
parameterized SIR score evidence is explicitly local complete-data or scout
scope and cannot be reused as a posterior identity.

## Research Intent Ledger

| Field | P6 target-design intent |
| --- | --- |
| Main question | Can BayesFilter define finite, graph-native, identifiable observed-data filter posteriors for three SIR log-scale parameters without substituting complete-data or scout quantities? |
| Candidate mechanisms | principal-square-root UKF and fixed-SGQF deterministic filter likelihoods; fixed-TTSIRT source route audited separately |
| Expected failure modes | off-by-one observation order, unstable SIR dynamics over prior support, singular filter covariance, wrong score, SGQF cloud infeasibility in 18 dimensions, or missing retained-marginal/transport derivative for Zhao-Cui |
| Promotion criterion | frozen data/prior/chart/filter contracts; prior-predictive validity; local-information rank; batch/replay/XLA/score/reference gates; negative substitutions detected |
| Promotion veto | local/complete-data or scout target substitution; invalid time order; nonfinite/support failure; rank deficiency at all design points; value/score mismatch; missing Zhao-Cui derivative closure |
| Continuation veto | common dataset/target invalidity or no feasible observed-data UKF and SGQF construction under the phase budget |
| Repair trigger | cell-local filter/status/score failure with unchanged target, or target-design support/identifiability failure |
| Explanatory only | likelihood curvature magnitude, UKF/SGQF gaps, truth distance, runtime, PF summaries |
| Not concluded | HMC convergence, NeuTra quality, filter exactness/ranking, epidemiological calibration, forecasting, robustness, or readiness |

## Frozen Model, Data, Chart, And Prior

The common physical model extends Zhao-Cui Section 6.3 equation (37):

```text
kappa_j(theta) = 0.1 exp(theta_kappa)
nu_j(theta)    = 18  exp(theta_nu)
R(theta)       = 100 exp(2 theta_R) I_9
```

for all nine Austrian compartments. The state, source RK4 variant, Gaussian
process noise, initial law, adjacency, and observation equation otherwise match
the declared Austria contract. The observed-data likelihood uses the
unprojected additive-Gaussian transition stated in the paper and computed by
the local transition density. The local simulator's post-noise susceptible
clipping is not part of that density; the frozen fixture must prove that no
clipping would occur, or the dataset/likelihood target is vetoed. Parameter
inference remains a BayesFilter extension, not a Zhao-Cui SIR reproduction.

- Truth: `theta=(0,0,0)`; truth recovery is explanatory only.
- Dataset generator: CPU-only `SpatialSIRSSM.simulate(final_time=20, seed=81120)`.
- State artifact: `x0:x20`, shape `[21,18]`.
- Observation artifact: freeze `y1:y20 = simulated_observations[1:21]`, shape
  `[20,9]`, matching the paper's `k=1,...,T` observation convention.
- Domain gate: every unprojected frozen state is finite and all susceptible
  coordinates are nonnegative, so source clipping would be inactive on the
  fixture.
- Negative time-order substitution: `simulated_observations[0:20]` must have a
  different hash and fail identity replay.
- Unconstrained parameters: exactly the three log scales above.
- Prior: independent `Normal(0,0.5^2)` on each log scale, with the identity
  chart. There is no separate chart Jacobian term.
- Design region: Cartesian points with each coordinate in
  `{-log(2), 0, log(2)}`, plus deterministic audit tails at `+/-1` along each
  axis. This is a target-design region, not compact posterior support.

The `0.5` prior scale is a target-specific reviewed hypothesis: it puts most
mass within roughly one multiplicative factor `exp(+/-1)` while still allowing
substantial rate/noise variation. Failure mode is unstable dynamics or an
overly prior-dominated posterior. The earliest diagnostics are a 4,096-draw
prior-predictive validity test through T=20 and likelihood-only local
information eigenvalues at all design points. It is not promoted as a global
SIR default.

## Filter Targets

### `SIR-UKF`

Use the graph-native batched principal-square-root UKF engine with state
dimension 18, innovation dimension 18, observation dimension 9, identity
observation on infectious coordinates, exact RK4 state/parameter Jacobians,
analytic `dR/dtheta_R`, zero parameter derivatives for the frozen initial law
and process covariance, strict SPD/status telemetry, and no active NumPy or host
callback.

### `SIR-SGQF`

The mathematical level-2 Smolyak cloud is feasible in 18 dimensions: it has 36
axis points plus the center, for 37 total points. The existing generic builder
is not feasible at this dimension because its tolerance-neighbor merge searches
`3^18` adjacent keys; that is an implementation pathology, not SGQF
dimensionality evidence. Add an exact level-2 axis-cloud constructor whose
weights and moments tie out to the generic builder at dimensions 1-4, then
freeze its 18D points, weights, branch hash, point count `37`, and memory
forecast before serious likelihood evaluation. Level 1 remains a deliberate
one-point negative control. If the exact constructor or filter recursion fails,
record the precise implementation/filter blocker; do not substitute UKF or a
low-dimensional cloud.

### `SIR-ZC`

The Zhao-Cui paper Section 6.3 and author `eg3_sir/mainscript.m` use `d=0` and
fixed `kappa,nu`; the three-parameter target is an extension. Existing local
code also preserves two explicit blockers:
`BLOCK_FIXED_TTSIRT_PREVIOUS_MARGINAL_DERIVATIVE_NOT_IMPLEMENTED` and
`BLOCK_FIXED_TTSIRT_PROPOSAL_TRANSPORT_DERIVATIVE_NOT_IMPLEMENTED`. Unless both
are closed by a paper/source-anchored fixed-HMC adaptation with full observed-
data value/score evidence, this target exits
`TARGET_BLOCKED_MISSING_OBSERVED_DATA_SCORE_ROUTE`. Local complete-data scores
cannot close it.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific question | Are the declared UKF/SGQF quantities full observed-data filter likelihoods for the frozen T=20 SIR extension, and is any Zhao-Cui route eligible to issue the same kind of target? |
| Baseline | exact model simulator and feasible bootstrap-PF/dense lower-rung checks on identical data; local complete-data density is component evidence only |
| Primary pass | frozen target contract, finite/status-valid batch-native likelihood and total posterior, full-rank likelihood-only local information at design points, central-FD score agreement, XLA replay, and negative substitutions |
| Hard vetoes | data/time-order/prior/filter drift, invalid dynamics/support, nonfinite target, SPD/status failure, score mismatch, SGQF infeasibility, or Zhao-Cui derivative gap |
| Explanatory only | PF Monte Carlo estimates, local curvature size, cross-filter gaps, runtime, truth distance |
| Not concluded | posterior correctness, HMC convergence, NeuTra quality, filter superiority/exactness, calibration, forecasting, readiness |

## Required Artifacts And Checks

1. Mathematical target note with the paper/source boundary and claimed versus
   computed quantities.
2. Frozen CPU dataset with generator closure, seed, exact slice, tensor hashes,
   and negative time-order hash.
3. Batched TensorFlow prior-predictive generator: 4,096 stateless draws with
   deterministic seed partition, no Python sample loop, T=20 validity and state
   summaries. A draw is valid only when every latent state and observation is
   finite and the maximum absolute state/observation magnitude is at most
   `1e6`. Require at least 99% valid draws. Record the fraction with negative
   susceptible coordinates as explanatory model-support telemetry, not a veto:
   the frozen likelihood explicitly uses an unprojected additive-Gaussian
   transition, whose support is all of `R^18`. The separate frozen-fixture gate
   still requires clipping to have been inactive for the generated dataset.
4. For each feasible filter: graph-native likelihood-only value/score/status,
   total posterior adapter, filter contract, dependency closure, audit points,
   local information rank/eigenvalues, score FD at `5e-5` and `1e-4`, batch,
   permutation, replay, CPU/GPU XLA, and same-target reference diagnostics.
   At truth, all six axis neighbors `theta +/- 0.5 e_i`, and the six audit tails
   `theta +/- e_i`, require the analytic likelihood score versus the `5e-5`
   centered value FD to have maximum absolute gap `<=5e-3` and maximum
   scale-normalized gap `<=5e-4`, where the scale is
   `max(1,abs(analytic),abs(FD))`; require the fine/coarse FD score gap under
   the same metrics to be `<=5e-3` and `<=5e-4`. These looser-than-unit-test
   thresholds account for the T=20, 18-state recursion but cannot be relaxed
   after observing a result.
5. Define likelihood-only local observed curvature prospectively as the
   negative symmetrized Jacobian of the analytic likelihood score, computed by
   centered parameter batches at steps `5e-5` and `1e-4`. Do not add prior
   curvature. At every 27 Cartesian design point require fine/coarse relative
   Frobenius gap `<=5e-3` and numerical rank three using singular values
   `>1e-8 * largest_singular_value`. Record eigenvalues and signs, but do not
   impose a PSD gate: observed curvature at an arbitrary non-mode point can be
   indefinite. Rank is a local curvature/sensitivity screen, not Fisher
   information or global identifiability.
6. Negative substitutions for `y0:y19`, wrong seed, prior scale, observation
   covariance convention, filter identity, dtype, and local complete-data
   scalar.
7. Low-dimensional parity and moment tests for the exact level-2 SGQF axis
   cloud, plus the frozen 18D point/weight hash and 37-point memory forecast.
8. Low-level replay and batch permutation require exact equality where the
   operation order is unchanged. CPU-XLA and GPU-XLA require status equality
   and maximum scale-normalized gaps `<=1e-8` for value and `<=1e-7` for score
   against eager CPU, with scale `max(1,abs(CPU),abs(XLA))`. Record absolute
   gaps as explanatory diagnostics. Score FD correctness must compare analytic
   scores and centered values evaluated in the same eager mode; separately
   compiled XLA value noise must not be divided by the FD step and mislabeled a
   score defect.
9. Zhao-Cui source-support, claim-support, and blocker ledger with paper and
   author-code anchors. No network metadata is needed for this implementation
   gate; live citation/retraction metadata remains not checked.
10. Target-design result, manifest, recursive hashes, and a reviewed R1B plan
   only for cells passing this rung.

## Handoff And Stops

On a cell pass, draft its independent R1B posterior identity subplan. A failed
filter target remains cell-local. `SIR-ZC` may exit blocked while UKF/SGQF
continue. No HMC or training begins before an R1B identity. Stop target design
for common data/model invalidity, three identical infrastructure failures, or
the four-GPU-hour per-cell admission bucket. Do not change the prior, time
slice, target parameters, filter family, or score threshold after seeing a
serious result without a new material target-design decision.

## Skeptical Pre-Execution Audit

Decision: `PASS_AFTER_REVISION`.

The inherited P6 plan falsely treated data, prior, time order, and three filter
targets as P0-frozen. They were not. The revision freezes those before code and
prevents the local complete-data score from being promoted. It also rejects the
initial concern that level-2 SGQF was dimensionally infeasible was also wrong:
the 37-point mathematical cloud is feasible, while the generic merge algorithm
is not. The revised plan distinguishes those facts and requires an exact cloud
with low-dimensional parity. The dataset uses paper-consistent `y1:y20`, and a
negative hash guards the off-by-one slice. Short likelihood/PF diagnostics can
nominate or veto a target but cannot establish posterior correctness or NeuTra.

Post-execution repair, 2026-07-16: the first complete result exposed two
planning/harness defects without changing the target. Requiring every
prior-predictive susceptible state to remain nonnegative contradicted the
declared unprojected additive-Gaussian transition. The runner also mixed an
eager analytic score with separately XLA-compiled FD values; roughly `1e-5`
value parity noise was divided by `2h` and created a false `0.30` UKF score
gap. At the worst point the manual score matched raw TensorFlow autodiff within
`1.5e-10` and a same-mode eager FD ladder within about `1e-6`. The visible
repair above makes prior support telemetry explanatory, binds score FD to one
execution mode, and uses scale-normalized CPU/GPU-XLA parity. Dataset, prior,
parameters, filters, audit points, FD steps, score tolerances, and scientific
nonclaims are unchanged.
