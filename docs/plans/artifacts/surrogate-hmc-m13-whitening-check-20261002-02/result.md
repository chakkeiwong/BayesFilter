# M13 frozen-whitening diagnostic result, 2026-10-02

The saved affine transform is applied correctly and partly removes linear
scale differences and correlation. It does not make the observed posterior
covariance close to identity in both directions. This is a useful repair
candidate, but the observed mismatch is moderate and does not establish that
whitening causes the later slow exploration. The strongest local warning is
poor acceptance and very small movement in the low process-scale region.

The question is whitening of the M13/data1704/T20 frozen importance-likelihood
posterior used by the endpoint-corrected surrogate-HMC test. These are saved-data
explanatory diagnostics. They change neither the endpoint target nor the sampling
rule and require no independent validation sample set. The historical 9,000-draw
stopping success remains recorded; all later saved draws remain visible.

## What was checked

The exact saved factor is used: theta = c + L z, with

    c = (0.0461679647, -0.0854275546)
    L = ((0.3120756564, 0), (-0.0348474629, 0.2620098131)).

Parameters are log process scale and log observation scale. The claimed and
computed inverse are both z = L^{-1}(theta-c). Factor/covariance agreement has
maximum absolute error 1.39e-17; forward/inverse reconstruction error is
1.11e-16. Executable parity with the actual runtime adapter passed. The history
loader checked archived states, affine reconstruction, checksums, continuation
seeds, and frozen settings. All 12,750 saved draws per chain were included;
9,000 warmup transitions per chain were excluded.

The checked runtime path is
`hmc_tensorflow_verified._chunk_graph` ->
`hmc_tensorflow_tuning._fixed_kernel` -> the issued binding's
`tensor_kernel_factory` -> `build_affine_neural_force_transition_kernel`.
The factory transforms the endpoint through theta=c+Lz and transforms the
position-only proposal force by L^T, with unit momentum covariance in z.
Source anchors: `hmc_tensorflow_verified.py:114`,
`hmc_tensorflow_tuning.py:562,831`, `neural_force_hmc.py:1084,1229`.
The saved state/adapter transformation was checked executably. The force-factory
wiring was inspected in source, not exercised in this offline analysis; this
analysis does not certify force fidelity or every tensor-kernel consumer.

For fixed L, sample covariance obeys
Cov(z) = L^{-1} Cov(theta) L^{-T}. Identity would follow if LL^T equaled the
posterior covariance. These are autocorrelated finite samples, not a known
posterior covariance. A condition number of one means equal principal-axis
variances; identity additionally requires correct absolute scale. There is no
new threshold or posterior-promotion decision based on these numbers.

| Saved draws | Coordinates | Marginal SDs | Correlation | Covariance condition number |
|---|---|---|---|---|
| First 9,000/chain | raw | 0.263, 0.169 | -0.362 | 3.247 |
| First 9,000/chain | whitened | 0.843, 0.615 | -0.199 | 2.121 |
| All 12,750/chain | raw | 0.311, 0.172 | -0.440 | 4.714 |
| All 12,750/chain | whitened | 0.996, 0.609 | -0.256 | 3.083 |

The full-history whitened covariance eigenvalues are 0.334 and 1.030. Scaling
and correlation improve descriptively, but the second transformed coordinate
has SD 0.609 instead of one. This is incomplete whitening, not evidence of a
transpose/inverse implementation defect. Transforming already generated samples
does not improve their mixing; in particular z[0] is just a shifted/scaled
version of the process-scale parameter and has the same autocorrelation.

The additional 3,750 draws alone have whitened covariance condition number
5.237. Chain 2 within that interval has condition number 8.697 and process-scale
lag-one autocorrelation 0.9932. Thus the geometry estimate depends on the part
of the trajectory inspected. The full-history covariance cannot be treated as
a trustworthy exact mass estimate while chain exploration disagrees.

## Where proposals struggle

Region boundaries are descriptive pooled empirical quantiles of transition
starting states, not known posterior quantiles. Acceptance is matched to the
transition leaving that state, and rejected states remain included. The first
saved transition is omitted because its preceding state is outside this array.

For chain 2, the lowest empirical 5% process-scale region (log scale below
-0.59867) contains 2,260 transitions, with acceptance 19.60% and mean squared
whitened movement 0.000835. Its upper 75% region contains 7,999 transitions,
with acceptance 88.82% and mean squared movement 0.132160: about 158
times larger. Chain 3 also visits the lowest region (288 transitions), with
acceptance 30.90%. Chains 1 and 4 have no visits there. Repeated autocorrelated
visits are not independent evidence about tail probability.

This supports a location-dependent proposal-efficiency problem in the saved
run. It does not distinguish endpoint curvature, inaccurate surrogate force,
short integration time, or a combination. No Hessian or force-versus-endpoint
score comparison was computed. Total Hamiltonian energy is not archived, so
E-BFMI is unavailable; delta-H alone cannot supply it.

## Tuning provenance and interpretation

The covariance was learned from successive 128- and 256-transition windows per
chain, pooled over four chains with 10% shrinkage toward prior covariance. The
last window determines the final estimate. These windows use one leapfrog step.
The later 9,000-step warmup checks readiness under the frozen metric; it does not
continue adapting covariance. Final step size is 0.40096758, and the selected
trajectory also uses one leapfrog step. Thus long warmup did not imply long
mass-matrix adaptation, and high average acceptance did not imply efficient
exploration.

A fixed dense metric adjusts global linear scale/correlation; position-dependent
curvature can remain. The Stan Reference Manual describes this limitation and
windowed covariance adaptation in its [Euclidean metric and nonlinearity
sections](https://mc-stan.org/docs/reference-manual/mcmc.html#the-euclidean-metric-aka-mass-matrix).
The matrix facts above follow from the displayed affine identity; the local
cause remains an empirical question.

A future repair should first inspect endpoint curvature and force mismatch at
representative central and low-process-scale states. If the covariance estimate
is the main issue, use longer own-target mass-adaptation windows during discarded
warmup, then retune step size and trajectory length and freeze them before
retained sampling. Test unchanged and repaired kernels against the same endpoint,
including regional acceptance/movement and cumulative convergence/MCSE. Do not
simply fit this pooled sample covariance, transform these same samples, and call
identity proof of success. A changed kernel needs new discarded adaptation and
a separately identified retained segment; it is not the unchanged saved-state
continuation already supported by the master.

## Decision and inference status

| Decision item | Status |
|---|---|
| Decision | Affine implementation parity passes; observed whitening is incomplete; causal attribution remains open. |
| Primary criterion | Explanatory comparison with identity completed; no new promotion criterion. |
| Veto diagnostics | No checksum, state continuity, finite-value or affine parity failure; original numerical-health checks pass. |
| Main uncertainty | Posterior covariance is poorly estimated in the low-process-scale region; curvature and force mismatch were not checked. |
| Next justified action | Bounded regional curvature/force diagnostic, followed by a scoped metric/trajectory repair if supported. |
| Not concluded | Whitening is the sole cause, a new metric will solve mixing, or surrogate HMC is invalid. |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new implementation-validity veto; no sampling or promotion was attempted. |
| Statistically supported ranking | None; there is only one saved kernel and no repaired-kernel comparison. |
| Descriptive-only differences | Raw/whitened covariance summaries, chain/epoch contrasts, regional acceptance and movement. |
| Default readiness | Unchanged; no new metric, tuning default or sampling rule promoted. |
| Next evidence needed | Representative target/force checks and a predeclared same-target kernel comparison with uncertainty and cumulative diagnostics. |

Post-run red team: the strongest alternative to an inadequate global covariance
is local endpoint geometry or force mismatch coupled with one-step trajectories.
The weakest evidence is the empirical tail distribution from poorly mixing
chains. A repaired fixed covariance failing to improve regional movement would
weaken the covariance explanation; representative local target/force checks and
a controlled comparison are needed to separate mechanisms.

## Execution and artifacts

CPU-only TensorFlow/TFP reporting exception: CUDA was deliberately hidden before
import, no XLA compilation, no new HMC transitions and no random draws. Successful
analysis time after imports: 1.981 seconds. TensorFlow's no-device
message reflects deliberate CPU execution, not a GPU-health diagnosis.

Attempt 01 failed before statistics were produced because the offline inverse
solve used an incorrect rank/layout. It was preserved, and attempt 02 flattened
the draw/chain axes and checked against the actual runtime adapter. Review also
corrected pooled lag computation to keep chains separate and aligned regional
acceptance with starting states. This changed only reporting code.

- `analyze.py`: complete reproducible diagnostic with assertions.
- `report.json`: full pooled, chain, epoch and regional results.
- `run-manifest.json`: inputs/checksums, code hash, environment, plan and command.
- `raw-vs-whitened.png` and `.pdf`: generated scatter/trace plots; not visually
  inspected because the sandbox image reader failed. Numeric conclusions use
  saved statistics, not a visual judgment.
- `command.log` and `execution-receipt.json`: exit status and execution record.

The existing sampling allocation expired at 2026-10-01 16:00 UTC. The master is
not running; this user-requested saved-data diagnostic is complete. The existing
master can continue the same frozen chains if time is reallocated. Metric
retuning is a possible new repair branch, not an already running phase. No
sampling or retuning was launched by this analysis.
