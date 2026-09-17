> Stage record preserved. Current result: all three LG1 attempts failed
> calibration; exact-block transport was implemented and checked before the
> N4096 continuation. See the [terminal result](../../ledh-graceful-failure-execution-result-2026-09-18.md)
> and `decision.json` for final scope decisions.

# LG1 execution protocol, attempt 1

This is the active LG1 implementation/calibration stage of the approved
2026-09-17 master program. It uses the isolated execution branch and the campaign
budget ledger. It does not waive later HMC tuning, reference or posterior gates.

The question is whether the fully enabled shared canonical finite program has
the correct analytical derivative for stationary initialization and the declared
prior, and can meet the proposed Kalman approximation budgets within the particle
ladder. The exact comparator is stationary LG1, transition before observation,
Q=1,R=0.5,T=10, phi uniform on (-0.95,0.95). Reference observations are the same
represented observations used by LEDH. The initial cloud and per-particle UKF
design covariance both depend on P0, with their distinct analytical derivatives.

The four calibration, two validation and three claim datasets are generated once
with the master seed and separate fold-ins for partition, attempt, replicate and
purpose. Three one-thread CPU workers generate complete partitions using a batched
TensorFlow/XLA recurrence. Claim files are saved without inspecting their content.
All finite-program random streams are fixed and separately identified.

Primary engineering requirements are correct cumulative status and analytical
directional derivatives. GPU float64 reference checks use the master's seven-step
central-difference ladder and proposed scaled error 1e-6. GPU float32 TF32 uses
1e-2 and is a separate production-target engineering check. Boundary/branch
crossings are reported; no point is dropped after a failure. Central points,
phi near each prior boundary, and positive/negative directions are checked.
Prior/Jacobian derivatives are tested in the actual unconstrained consumer.
Reference/grid agreement is not substituted for finite-program differentiation.

The first calibration particle rung is N=256, then N=1024 if the first fails. The
shared dense reset implements the required single-block geometry for these rungs.
N=4096 requires a genuine 2x2 implementation, currently unavailable in this call
chain. It is unready until repaired and tested; no invalid tiny chunk is allowed.
Calibration uses physical phi=(-.9,-.6,-.3,0,.3,.6,.9), centered at zero. Selection
requires finite value/force status and the master's proposed worst-time whitened
mean/covariance errors <=0.05 and centered log-ratio discrepancy <=0.02 on every
calibration dataset/point. These thresholds are hypotheses under calibration;
the original and any scientifically justified revised thresholds must both be
preserved before validation. No threshold change based on validation/claims is
permitted. The validation design is (-.85,-.45,.15,.55,.85), evaluated once after
the candidate list and selection are frozen. Claims remain untouched until a
matching repository-issued tuning artifact exists.

| Choice | Provenance and justification | Failure mode and earliest diagnostic | Status |
|---|---|---|---|
| LG1 equations/prior/data partitions | Master Sections 7 and 8.3; exact solvable initial law | Wrong timing or stationary covariance; closed-form and Lyapunov tests | Declared fixture |
| GPU float32 TF32 XLA | Current repository default; GPU smoke passed | Rounding/caps invalidate derivative or change target; float64 diagnostic and step ladder | Default backend, scientific admission pending |
| N=256 then 1024 | Master ladder; bounded cost measured at 0.06 seconds/evaluation for the first warm fixture | Monte Carlo error exceeds likelihood/moment budgets; all calibration replicates | Hypotheses, not universal defaults |
| Fixed standard-normal base and residual design | Master frozen-input requirement; centered, whitened residual design is independently validated once | Fixed-design bias or rank loss; design moments and replicated datasets | Explicit design hypothesis |
| Flow steps 10,20,40,80 | Current post-invalidation fixture starts at 10; successive halving tests integration error | Finite-step bias or derivative cancellation; adjacent-rung value/score curves | Warm start and convergence ladder |
| OT epsilon 2 with iteration/balance 8,32,128 | Current fixture warm start, not inherited admission; residual/iteration convergence is required | Inadequate transport convergence or target bias; value/moment convergence plus direct reset residuals | Warm start pending calibration |
| Reset ridge 1e-6,1e-5,1e-4 | Three decades around current fixture; regularization must dominate relative roundoff without material bias | Rank-sensitive factor or moment shift; factor status and ridge-versus-value/moment curves | Class C hypotheses |
| Correction steps 2,4; strength .2 | Current full-feature fixture and a doubled budget | Incomplete shape correction; correction and moment residuals | Hypotheses |
| LM damping 1e-3,1e-2, scale floor 1e-6 | Current fixture and one-decade sensitivity; damping controls an empirical solve | Ill-conditioning or excess correction bias; solve status and response curve | Class C hypotheses |
| Trust radius .1,.5; pairwise 2, strength .02, RMS cap 2, coordinate cap .98, power 8 | Post-invalidation full-feature fixture; retain both caps, inspect trust sensitivity | Cap activation/piecewise score or overly aggressive correction; branch diagnostics and one-sided derivative checks | Hypotheses; no zero/off candidate |

The first diagnostic uses the full warm-start configuration only to expose
implementation errors and measure discrepancies. It does not select controls.
Subsequent bounded candidate combinations will be frozen after inspecting these
calibration curves, before any validation access. Choose the earliest configuration
meeting the frozen validity/approximation rules; do not rank stochastic candidates
by observed runtime or few-seed averages. Turning off canonical features is forbidden.

Derivative/status failure invalidates the implementation for dependent inference
and triggers local repair. Approximation failure rejects this finite candidate and
triggers the next particle/control rung. It does not reject LEDH. Budget exhaustion,
unresolved mathematical mismatch, missing truthful reference or unimplemented chunk
geometry is a continuation veto for the affected claim, not a reason to fabricate
admission. The initial stage has a one-GPU-hour/one-CPU-worker-hour allocation within
the campaign, including retries, with ten-minute feasibility and twenty-minute
initial calibration job limits; forecast 1.5x measured costs before larger launches.

Heuristic comparisons remain reserved for evaluation, never control selection:
stationary unconditional estimate (tests the value of data), prediction-only exact
moments (tests whether updates harm), raw observations with variance R (tests
whether complex filtering adds value over measurements), ordinary Kalman/UKF
(linear oracle), and independent bootstrap PF. Report conditional results at large
innovations, ordinary observations and boundary/central phi. These diagnostics can
veto promotion but cannot certify correctness or tune this candidate.

Pre-mortem: a correct derivative can differentiate a poor approximation; matching
moments can coexist with a wrong likelihood ratio; a finite reference can exclude
numerically difficult posterior mass; and a short chain can appear to mix. Keep
engineering correctness, Kalman approximation and finite-target sampling separate.
No posterior, convergence, scientific superiority or default-readiness claim is
authorized by this calibration stage. Store every command, input/source identity,
curve, failure, resource charge and next decision in the campaign directory.

## Exact-block repair and remaining inherited constants

The first control sweep failed its approximation screens; see
`lg1-calibration-result-01.md`. The next planned scope is N=4096,K=2048. The
shared transport now computes all kernel products and payload contractions in
the prescribed 2x2 blocks. It preserves the same finite Sinkhorn iterations,
row normalization, full covariance carry and analytical tangents. Full returned
transport matrices remain quadratic memory. Before numerical calibration,
require one-block parity, invalid-policy rejection, GPU/XLA two-block row-mass
invariants and finite-difference step ladders using the master derivative budgets.
Forecast the initial calibration from the measured mechanics/runtime, with 1.5x
cost headroom and a 1200-second timeout inside the current allocation.

Cost normalization floor 1e-3, Sinkhorn denominator increment 1e-7 and correction
floor 1e-5 are inherited Class C hypotheses, not reviewed defaults. Their purpose
is to bound divisions or factorizations at collapsing geometry/weights; their
failure mode is scale-dependent perturbation or loss of derivative resolution.
The calibration must record activation/margin sensitivity before promotion.
The two-block repair leaves them unchanged to test the same finite program;
that equivalence alone does not justify them as defaults. The previous single
float32 block FD step failed an ad hoc 0.002 primitive threshold. Preserve that
failure; assess the full master ladder at 1e-2, with the independent sharp
float64 result, and do not construe this as held-out threshold adjustment.

The exact-grid validation passed: float64 derivative ladder minimum scaled error
4.82e-11 and float32/TF32 4.61e-4, with row-mass invariants. Primitive peak GPU
allocation was 538 MB; no claim about full-filter memory follows. The next frozen
calibration list is N4096 with 10, 3, 1 and 40 flow steps and all other warm
controls unchanged. Finer flow did not resolve earlier errors; the coarse arms
test finite-proposal concentration with correction and all canonical features
still enabled, not a feature ablation. Only calibration replicates 0 and 2 and
phi {-0.9,0,0.9} are read. Estimated full run <800 seconds with 1.5x headroom to
the 1200-second timeout; stop on repeated runtime failure or missing diagnostics.
This is continuation of LG1 attempt 1 across its required particle rungs.

## Conditional-mixture localization

Before new candidates, compare the shared N4096 warm flow and its importance logits against independent scalar affine-flow and Gaussian-density formulas at calibration replicates 0/2 and phi -0.9,0,0.9. Relative arithmetic budgets are 1e-4 for TF32, consistent with engineering precision limits; these do not change approximation thresholds. Integrate each actual incoming empirical Gaussian mixture analytically to isolate proposal quadrature error from previous reset/distribution error. This conditional reference cannot certify the global filtering law. Only equality failures veto implementation correctness; mixture discrepancies explain candidate failure. GPU/XLA candidate, float64 diagnostic reference. Previous N4096 eight groups cost 120 seconds; six trace calls forecast <=180 seconds including 1.5x headroom, timeout 300 seconds. No new statistical candidate, no heldout access.
