# LEDH Analytical Score Discrepancy Audit

Date: 2026-08-29  
Status: BLOCKED_FOR_CLAIM (diagnostic result; research direction remains open)

## Scope and provenance

This document answers
docs/requests/ledh_score_discrepancy_audit_request_2026_08_28.md. The audit
was performed in the canonical rebuild worktree
/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild at
39f0b20e512e495fa2d42b4c72ce7bcbd4daf01f.

The request identifies e3292331
(e32923317feddc9003431d3d30872c960e1589f3) as the pilot provenance. That
commit contains the value-lane tuning pilot, but not the later parity JSON.
The parity request and score-lane grid were produced in the current rebuild
worktree. The score grid manifest also records 39f0b20e. The grid and several
parity helper files are currently untracked in the worktree; they are local
diagnostic evidence, not versioned claim artifacts, until committed with
their inputs and source.

The repository policy quarantines pre-2026-08-21 LEDH results from new claims.
Accordingly, this audit does not reuse an old leaderboard or promote the
reported cell. It checks the current implementation and treats the reported
numbers as historical/diagnostic observations.

No new long benchmark, tuning campaign, or GPU claim run was executed for this
audit. A focused CPU-only test run was used to check local derivative
accounting; GPU/CUDA visibility was intentionally disabled for that check.

## Executive verdict

The exact Kalman reference in the request is credible: the two central
differences agree to 4.263256414560601e-09, and the model is linear Gaussian.
The six score errors do not establish an unbiasedness failure or an
unexpected Monte Carlo standard error. They are a six-replication descriptive
sample, and the two lanes did not use common particle clouds. The reported
"paired lane gap" is therefore not a paired comparison.

The strongest immediate explanation is not fixed ancestry. The reported cell
sets annealed_stages=1, and the score implementation only enters the
systematic-resampling branch when annealed_stages > 1
(ledh_canonical_score_tf.py:229-341). The reported cell has no resampling
indices whose tangent could be held fixed.

There are instead several confirmed confounds that prevent the requested
value-versus-score interpretation:

1. The registered score entry point names a callable that does not exist.
   ledh_alg1_contract.py:164-170 registers
   canonical_value_score_and_diagnostics, while
   ledh_canonical_score_tf.py:60 and :614-617 export
   canonical_value_and_analytical_score. The parity script calls the latter
   directly and bypasses the registry.
2. The purportedly matched score cell uses one diagonal correction step, one
   pairwise step, and no coordinate cap
   (r2_lane_parity_dlgssm.py:218-223). The registered production program
   specifies four and four steps and a coordinate cap of 0.98
   (ledh_alg1_contract.py:242-264). These are numerics-altering controls,
   not aliases that can be silently equated.
3. The value and score lanes use different reset implementations and
   different arithmetic dtypes. The value filter calls the canonical
   Contract-E forward core after casting reset inputs to float32
   (ledh_canonical_filter_tf.py:405-420); the score lane calls a vendored
   ledh_canonical_reset_score_tf implementation
   (ledh_canonical_score_tf.py:413-434) and retains its input dtype.
4. The score result returns only a scalar value and a scalar tangent
   (ledh_canonical_score_tf.py:467-469). It does not expose the marginal,
   row/column, factor-condition, cap-activity, or route-identity diagnostics
   that the value lane uses (ledh_canonical_filter_tf.py:447-470).
5. The available tuning artifact is explicitly for the value lane. The score
   grid checks finiteness only and explicitly says that it does not select a
   score configuration (r2_score_lane_grid_dlgssm.json:3820-3838).
   Per-scope tuning therefore has not been satisfied for the score claim.

The hand-coded JVP primitives are locally plausible. The focused tests compare
them with autodiff of the same tiny finite primal program and passed. That
evidence supports local chain-rule accounting; it does not show that the
finite LEDH-PFPF-OT score equals the Kalman score at T=50, or that it is the
canonical Contract-E route.

The correct disposition is to preserve the discrepancy as a diagnostic, repair
the registry and route wiring, match the complete score control family and
dtype, expose diagnostics, tune the score scope on disjoint data, and then
rerun a common-cloud, replicated comparison. Do not attribute the present
error to resampling, particle count, or a derivation error before those
confounds are removed. This repairs the evidence and implementation path; it
does not reject the LEDH research direction.

## Evidence read and interpretation

### Reference and parity numbers

The request reports:

| Quantity | Reported result | What it supports |
|---|---:|---|
| Exact Kalman log likelihood | -230.8826936051034 | A valid scalar reference for this LGSSM |
| Central FD at h=1e-5 | 7.303406597714001 | A directional reference |
| Central FD at h/2 | 7.303406601977257 | Difference truncation is negligible |
| FD Richardson gap | 4.263256414560601e-09 | The reference calculation is resolved |
| Six-seed score error mean | -0.4554092719 | Descriptive sample mean only |
| Six-seed score error SE | 0.3285730915 | Descriptive uncertainty for these six runs |
| Mean divided by SE | 1.39 | Does not cross the two-SE screen |
| Negative score errors | 4/6 | Descriptive sign asymmetry; sign-test p=0.688 |

The relative error -0.4554 / 7.3034 = -6.24% is a useful description of this
six-seed sample. It is not an estimate of persistent bias without a replication
design that defines the randomization, target, and uncertainty calculation.

A standard error from six seeds describes this run design, not every horizon,
direction, observation path, reset route, particle count, or dtype. A score is
the derivative of a long nonlinear finite program, so its variance need not
track value variance; N=1008 and d=3 do not imply a target SE. ESS is an
explanatory diagnostic, not evidence of ancestor-gradient bias.

The value and score values were generated from different streams. The value
lane derives a TensorFlow generator internally, while the score lane creates a
NumPy RNG stream at r2_lane_parity_dlgssm.py:186-193. Subtracting their
per-seed values does not produce a paired estimator, so the reported paired-gap
SE cannot be treated as paired inference. The separate 4/6 sign count for
score errors remains a descriptive count; it does not establish bias.

The six score observations themselves are:

| Seed | Score-lane value error | Score error | Score |
|---:|---:|---:|---:|
| 999000 | -0.7836193243 | +0.3021234878 | 7.6055300855 |
| 999001 | -0.8852773192 | -0.3742575188 | 6.9291490789 |
| 999002 | -0.3599626249 | -1.3573780683 | 5.9460285294 |
| 999003 | -0.3939468069 | -1.5203631515 | 5.7830434462 |
| 999004 | -0.7017834017 | +0.3013644913 | 7.6047710890 |
| 999005 | -0.7517591027 | -0.0839448720 | 7.2194617257 |

The score-error column is the analytical score minus the exact
7.3034066... reference. The value-error column is the score-lane likelihood
minus the exact likelihood for the same observation sequence. Neither column
should be read as a paired lane difference.

### Value pilot and score grid

r2_tuning_pilot_dlgssm.json is a value-lane artifact. It reports that all
epsilon 0.5 configurations were rejected by the value marginal-TV veto, while
the selected value warm start was epsilon 1.0, Sinkhorn/balance 8/8, flow 12,
and N=1008. Its reported worst TV is about 3.4e-5; its minimum ESS fraction is
about 0.052.

Those observations do not transfer automatically to the score lane. The score
grid has 216 cells (72 configurations and three observation replications), all
finite, but its own summary says that score marginal diagnostics are absent
and that the grid only checks finiteness. The three exact likelihood references
in that grid are different observation replications. Raw score values across
those replications cannot be compared as score errors until each is subtracted
from its corresponding reference. The grid does not select a score
configuration. It is evidence of finiteness, not convergence or accuracy.

### Artifact ledger

| Artifact | Provenance | Role in this audit | Limitation |
|---|---|---|---|
| docs/requests/ledh_score_discrepancy_audit_request_2026_08_28.md | Commit 39f0b20e | Audit question, model, controls, hypotheses, and reported numbers | A request is not a validation result |
| docs/benchmarks/r2_lane_parity_dlgssm.py | Commit 39f0b20e | Reproducible parity harness and random-stream implementation | Calls the score function directly and uses non-common clouds |
| docs/benchmarks/r2_lane_parity_dlgssm.json | Manifest commit e32923317... | Six-seed observations and exact reference | Diagnostic/non-claiming; its recorded commit predates the request commit |
| docs/benchmarks/r2_tuning_pilot_dlgssm.json | Commit e32923317... | Value-lane marginal-TV selection and warm-start controls | Value scope only; not score tuning |
| docs/benchmarks/r2_score_lane_grid_dlgssm.json | Manifest commit 39f0b20e; currently untracked | Score-lane finiteness grid | No marginal diagnostics, no score selection, and no claim status |
| bayesfilter/highdim/ledh_canonical_score_tf.py | Current worktree | Analytical score call chain and branch behavior | Registry name does not resolve to its exported callable |
| bayesfilter/highdim/ledh_canonical_reset_score_tf.py | Current worktree | Score reset/JVP implementation | Separate re-derivation, not proven identical to canonical reset |
| bayesfilter/highdim/ledh_contract_e_reset_tf.py | Current worktree | Repository-owned canonical Contract-E implementation | Its existence does not prove score wiring reaches it |
| docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex | Current worktree | Finite-JVP/VJP and claim-boundary derivation | Does not certify Kalman equivalence or the current dual-cap extension |
| docs/chapters/ch09_kalman_score.tex | Current worktree | Exact LGSSM score equations | Reference is exact for the stated model/data, not for the approximate LEDH program |
| tests/highdim/test_ledh_canonical_score_*.py and test_higher_moment_contract_e.py | Current worktree | Local derivative-accounting checks | Tiny fixtures; no registered T=50 Kalman-equivalence check |
| docs/requests/ledh_score_discrepancy_audit_response_2026_08_29.md | This audit | Reader-facing disposition and repair protocol | Does not itself create new scientific evidence |

The ellipsis in e32923317... is intentional shorthand for the full commit
recorded in the manifests; the full hashes are given in the provenance
paragraph above and in the JSON files. No artifact in this ledger should be
silently upgraded from diagnostic to claim-bearing status.

### Evidence classes

| Evidence class | Present evidence | Permitted interpretation |
|---|---|---|
| Hard veto evidence | Unresolved registry target; route/control scope mismatch; missing score diagnostics and tuning | Claim admission is blocked |
| Local correctness evidence | 35 focused derivative tests pass | Tested primitives agree with the same finite primal on tiny fixtures |
| Descriptive stochastic evidence | Six score errors and finite score-grid cells | The observed run was finite and has stated sample statistics |
| Statistical ranking evidence | None | No candidate or lane is statistically ranked |
| Scientific equivalence evidence | None | No Kalman-equivalence or posterior-validity claim is supported |

## Mathematical audit

### The target being differentiated

For the LGSSM, the Kalman reference is the exact log likelihood
\[
\ell(\theta;y_{1:T})=\sum_{t=1}^{T}\ell_t(\theta).
\]
The innovation and covariance derivatives in
docs/chapters/ch09_kalman_score.tex:23-89 give, with S_t w_t=v_t,
\[
\frac{\partial\ell_t}{\partial\theta_i}
=-\frac12\left[
\operatorname{tr}(S_t^{-1}\dot S_t^{(i)})
+2\dot v_t^{(i)\mathsf T}w_t
-w_t^{\mathsf T}\dot S_t^{(i)}w_t
\right].
\]
This is equation eq:bf-kalman-score-contribution, with the solve form in
eq:bf-solve-score, in the cited chapter. The request's central-difference
calculation is a sound scalar reference for the phi1 direction under the
stated fixed data.

The score implementation instead differentiates an executed finite program.
Let
\[
\widehat L_K(\theta;\omega,b)
=\text{finite-}N\text{ LEDH-PFPF-OT value with random stream }\omega
\text{ and branch decisions }b.
\]
The analytical path is intended to return
\[
D_\theta\widehat L_K(\theta;\omega,b)[\dot\theta],
\]
not automatically the gradient of the exact Kalman likelihood. Finite N,
Euler flow steps, finite Sinkhorn and balance iterations, cost scaling, ridge
terms, Contract-E restoration, and dual caps define the finite object.
Agreement of its derivative with its own primal is a necessary engineering
check, not a Kalman-equivalence theorem.

### JVP versus VJP

The chapter's Relation to forward-mode JVP section
(ch32c2_ledh_pfpf_ot_custom_gradient.tex:2347-2368) states that a forward JVP
is mathematically legitimate and returns
D_theta L_K(theta) dot-theta when the same finite program is used. The chapter
emphasizes reverse VJPs because a scalar target and many parameters make one
reverse pass efficient; that preference does not invalidate a one-direction
JVP.

The obligations are that every primal operation has a tangent, total
dependence through particles/weights/covariances is included, declared
branches and random streams are held fixed, and the tested callable is the
claim-bearing registered callable. The current path meets some local
obligations but fails the registry and route obligation.

### Sinkhorn tangent

The score reset forms pairwise costs, a scaled kernel, finite alternating
updates, a coupling, and a row quotient. In simplified notation:
\[
C_{ij}=\lVert x_i-x_j\rVert^2,\quad
s=\max(\operatorname{mean}(C),10^{-3}),\quad
K_{ij}=\exp[-C_{ij}/(s\varepsilon)].
\]
For one update,
\[
\ell_{\rm new}=\frac{u}{Kr+\tau},\qquad
r_{\rm new}=\frac{w}{K^{\mathsf T}\ell_{\rm new}+\tau},
\quad \tau=10^{-7}.
\]
The code differentiates the cost (ledh_canonical_reset_score_tf.py:66-81),
both matrix-vector products and quotient updates (:89-102), the coupling
product (:104-109), and the barycentric row quotient (:110-120). These are
ordinary product and quotient rules. The normalized-softmax weight tangent
\[
\dot w_i=w_i\left(\dot z_i-\sum_jw_j\dot z_j\right)
\]
matches ledh_canonical_score_tf.py:418-421.

No implicit-function theorem is required for the derivative of a fixed,
unrolled sequence of eight Sinkhorn and eight balance updates. The chapter
proves the corresponding finite-JVP chain rule in
ch32c2_ledh_pfpf_ot_custom_gradient.tex:596-653 and :2347-2368.

There are three boundaries. If the target is a converged Sinkhorn fixed point,
this is not an IFT derivative. The max cost-scale floor is nondifferentiable;
the implementation returns zero scale tangent on the floor branch and should
record whether it is active. Finally, tau changes the finite map, so the
tangent is for the regularized map, not the unregularized equations.

The algebra is plausible for the declared finite wrapper, but the score
wrapper is not the canonical reset implementation and does not report the
residuals needed to decide whether eight iterations are adequate.

### Contract-E reset and source terms

The chapter defines Contract-E's row quotient, weighted source moments, ridged
Cholesky factors, and affine restoration
(ch32c2_ledh_pfpf_ot_custom_gradient.tex:121-170). It also states that the
canonical total source pullback contains both direct moment and transport
terms (:2373-2390).

The score wrapper differentiates source weights and source particles in its
target mean and covariance (ledh_canonical_reset_score_tf.py:122-155), then
the Cholesky and triangular solve (:156-205). This is the right local
dependency pattern for a total JVP. The canonical reset module has analogous
product-rule and Cholesky-JVP machinery in
ledh_contract_e_reset_tf.py:229-344.

That pattern does not certify the score route. The score module calls
sinkhorn_contract_e_reset_with_tangent from a separate module whose
documentation calls it a re-derivation/historical lane
(ledh_canonical_reset_score_tf.py:1-20). The value lane calls
_contract_e_chol_cloud_forward_core from the repository-owned canonical module
(genut_guided_proposal_tf.py:900-919). Two similar formulas remain two
implementations until dense/streaming parity and call-chain tests prove
identity. The canonical identity must bind the actual callable, ridge,
residual design, and prepared input; a caller cannot self-stamp it.

### Trust-region and higher-moment tangent

higher_moment_shape_jvp is explicitly a finite bounded correction candidate
(higher_moment_contract_e.py:1-5). Its diagonal and pairwise loops propagate
the tangent through moment residuals, linear solves, smooth RMS caps, and
coordinate caps. Existing tests establish agreement with autodiff of this
same finite map on small fixtures.

This does not establish that the routine differentiates a constrained
optimization solution. It is an iterative map with damping, floors,
trust-radius caps, and smooth clipping. Nor does the cited chapter provide a
theorem identifying the current diagonal-plus-pairwise cap family as its
canonical algorithm; the chapter's boundary section limits claims to the
executed finite scalar and excludes statistical accuracy and default readiness
(ch32c2_ledh_pfpf_ot_custom_gradient.tex:2370-2399).

These are Class-C numerics-altering controls. Their tangent needs a non-harm
calibration and activity/conditioning diagnostics; primary-score tuning alone
does not justify changing them. The present score cell uses one iteration of
each cap, while the registry calls for four. That mismatch is a more immediate
explanation than a subtle sign error in a locally tested JVP.

### Fixed ancestry and resampling

For annealed_stages > 1, the score code computes systematic-resampling indices
through tf.searchsorted and gathers realized rows
(ledh_canonical_score_tf.py:292-334). Holding those indices fixed gives the
derivative of the conditional finite path while indices stay unchanged. At a
cumulative-weight boundary the map is discontinuous or undefined, so this is
not a derivative of ancestor probabilities or of a marginal particle-filter
expectation.

A differentiable resampling relaxation would define a different finite target
unless it is explicitly substituted and calibrated. The chapter claims fixed
branch/randomness composition, not categorical particle-filter equivalence
(ch32c2_ledh_pfpf_ot_custom_gradient.tex:724-740 and :2373-2399).

This convention cannot explain the reported cell: annealed_stages=1 follows
the non-annealed branch (ledh_canonical_score_tf.py:342-393), and the
resampling block is never executed. The value-lane ESS statistic describes
softmax concentration, not a resampling ancestry derivative in this run.

### Other approximations and inactive gaps

The flow is an Euler discretization (ledh_canonical_score_tf.py:472-601).
A value-selected flow_substeps=12 need not resolve a tangent to the same
tolerance. No convergence order should be assumed before measuring it.

The generic Gaussian density fallback
(ledh_canonical_score_stages_tf.py:583-620) differentiates the residual solve
but omits covariance-Cholesky/log-determinant tangents. The DLGSSM parity
model supplies custom transition and observation density callbacks, and the
phi1 direction has constant Q and R, so this inactive fallback is not the
leading explanation for the stated direction. It is still a generality defect
that should be restricted or repaired.

The flow tangent documents a state-independent/linear observation-Jacobian
assumption (ledh_canonical_score_tf.py:504-510). A state-dependent Jacobian
would require its tangent. Finite N, epsilon, ridge, cost scaling, and
Contract-E restoration all change the target relative to exact Kalman
likelihood, and can produce finite-program bias even when local derivative
tests pass.

## Confirmed implementation findings

| ID | Severity | Finding and consequence | Exact anchors | Required repair |
|---|---|---|---|---|
| C1 | P0 | The registry claim-bearing score callable does not exist. The parity runner bypasses the registry, so the tested function is not the registered endpoint. | ledh_alg1_contract.py:164-170; ledh_canonical_score_tf.py:60,614-617; r2_lane_parity_dlgssm.py:170-175 | Resolve the registry name to the intended callable, add an import/wiring test, and make the runner resolve through ENTRY_POINTS. |
| C2 | P0 | The score parity cell is not the registered production program: correction_steps=1, pairwise_steps=1, coordinate_cap=0 versus 4, 4, 0.98. | r2_lane_parity_dlgssm.py:203-225; ledh_alg1_contract.py:242-264; ledh_canonical_score_tf.py:76-85 | Thread every family control from the repository-owned program and rerun with an exact scope manifest. |
| C3 | P0 | Value and score reset routes are separate implementations. This is a call-chain fork; canonical Contract-E identity is not established for score. | ledh_canonical_score_tf.py:413-434; ledh_canonical_reset_score_tf.py:1-20; genut_guided_proposal_tf.py:900-919 | Route score through the canonical reset/JVP or formally replace the owner with a tested shared implementation. Add dense parity and route-identity diagnostics. |
| C4 | P1 | Value reset arithmetic is cast to float32; score arithmetic uses input float64. The pilot labels its scope float64, so lanes are not dtype-matched. | ledh_canonical_filter_tf.py:405-420; ledh_canonical_score_tf.py:119-127; r2_tuning_pilot_dlgssm.json:4-10 | Bind dtype in the scope and use the same dtype, or report separate scopes and never call them matched. |
| C5 | P1 | Score discards marginal and reset diagnostics. Finiteness is the only score-grid veto. | ledh_canonical_score_tf.py:467-469; ledh_canonical_filter_tf.py:447-470; r2_score_lane_grid_dlgssm.json:3820-3838 | Return structured per-step TV/row/column residuals, ESS, factor/cap checks, increments, validity, and route/config IDs. |
| C6 | P1 | No score-lane tuning artifact selects controls. Value selection is only a warm start, not score-scope evidence. | r2_tuning_pilot_dlgssm.json:2-14; r2_score_lane_grid_dlgssm.json:3820-3838; bayesfilter-r2-execution-plan-2026-08-27.md:112-170 | Tune on disjoint score calibration/validation data, freeze, and require exact repository-issued scope match. |
| C7 | P1 | The lane comparison is unpaired: lanes cannot receive a common cloud and use different random generation paths. | r2_lane_parity_dlgssm.py:8-10,186-193,406-408; parity JSON nonclaim :137 | Inject the same cloud/noises and fixed branches, or use a declared common-random-number design and analyze paired differences only then. |
| C8 | P2 | Generic Gaussian density tangent omits covariance dependence; flow tangent omits dH for state-dependent Jacobians. These are inactive for the stated constant-Q/R, linear-H phi1 fixture but violate an unqualified general claim. | ledh_canonical_score_stages_tf.py:504-510,583-620 | Repair terms or fail closed unless the model callback contract guarantees the restricted case. |

## Prioritized hypotheses H1-H7

The likelihood below is for explaining the reported discrepancy after reading
the actual call chain. It is not a ranking of methods. C1-C7 are confounds
that must be repaired before the hypotheses can be cleanly tested.

| Hypothesis | Likelihood for this cell | Why | Discriminating diagnostic | Remediation if confirmed | Anchor |
|---|---|---|---|---|---|
| H1: fixed-ancestry tangent bias | Low for this cell; medium/high only when annealed_stages>1 | No resampling is executed here. In annealed mode, searchsorted indices are piecewise constant and omit ancestor-probability derivatives. | Run stages 2 and 3 with identical clouds; record indices, cumulative-weight boundary margins, and fixed-path FD/JVP, including a near-boundary case. | Retain fixed-path derivative only for that declared estimand. For a marginal score, derive a score-aware filter/smoother or explicitly define a relaxed target. | ledh_canonical_score_tf.py:229-341; chapter :724-740 |
| H2: Sinkhorn marginal non-convergence or tangent instability | Medium/unknown | Value TV does not transfer across duplicate reset, dtype, controls, or random cloud. Score exposes no residuals. | On one common cloud, compare row/column residuals and TV at epsilon 0.5, 1, 2, 4 and iterations 8, 16, 24, 48; compare reset JVP with FD; record cost-floor activity. | Add diagnostics, tune epsilon/iterations in score scope, and apply the declared marginal veto after route unification. | ledh_canonical_reset_score_tf.py:66-120; genut_guided_proposal_tf.py:1090-1100 |
| H3: trust-region correction or tangent | Medium after C2; not first explanation | Live score uses 1/1/0 while production uses 4/4/0.98. Tiny JVP tests make an elementary sign error less likely than a different finite map or active cap. | With common inputs compare no correction, diagonal-only, pairwise-only, and full 4/4/0.98; finite-difference each map; report LM condition and cap activity. | Match the family, repair any failing component JVP, and perform a Class-C non-harm calibration. | ledh_canonical_score_tf.py:435-461; higher_moment_contract_e.py:981-1018,1285-1515 |
| H4: flow discretization too coarse | Medium | flow_substeps=12 was selected on value error, not score convergence; Euler tangent error can accumulate differently. | Freeze cloud and controls; run 12, 24, 48, 96; compare analytic JVP to FD of its finite primal and score error to Kalman, with per-step increments. | Increase substeps only after score tuning and bounded cost/accuracy calibration, or report measured finite-step bias. | ledh_canonical_score_tf.py:472-601; parity :131,210 |
| H5: particle count insufficient | Medium for variance; low as sole cause of current mean | N=1008 does not imply a small gradient SE, but current comparisons confound route, observations, and streams. | Run N=504,1008,2016,4032 with common random numbers, fresh independent seeds, matching observations, exact divisor-cap chunks, and CIs for mean error and variance. | Increase N only if bias/uncertainty curves justify cost. A stable mean with falling SE calls for route repair, not just larger N. | DPF policy; pilot JSON :63-66; request H5 |
| H6: implementation bug in tangent composition | High at registry/config/route boundary; low/medium for local primitives | C1-C4 are concrete mismatches. Local UKF/reset/higher-moment tests do not exercise the registered end-to-end path. | Resolve registry first, then component FD tests for UKF, flow, weights, reset, higher moments, and full T-step scalar on one cloud; check source-moment totals and dtype. | Repair the first failing edge and add claim-endpoint wiring/parity tests; do not tune around a route defect. | ledh_canonical_score_tf.py:395-469; stages :400-571; reset :122-205 |
| H7: mathematical derivation error | Low for finite Sinkhorn product rules; medium for a stronger target or unsupported extension | Finite JVP chain is algebraically consistent and test-backed. A mismatch remains possible if the claim is converged OT, categorical PF likelihood, or a chapter-canonical dual-cap solve. | State the finite primal in one place; compare analytic JVP with autodiff and FD under fixed branches, then compare finite score with Kalman. Use IFT only for a converged Sinkhorn target. | Narrow claim to derivative of executed finite program, or derive/implement the stronger target explicitly with source anchors. | chapter :596-653,724-740,2347-2399 |

The absence of score-scope tuning and the unpaired design are evidence failures,
not numerical mechanisms by themselves. They make the current sampling-noise
versus bias decision underidentified.

## Cheapest discriminating protocol

### 1. Repair and test the call chain

Resolve ENTRY_POINTS to an existing callable. Add a test that imports every
registered canonical endpoint and a runner test that obtains the score through
the registry rather than a direct import. Assert route identity, reset policy,
dtype, flow count, and the complete correction family.

Until this passes, all score numbers remain diagnostic and cannot answer H1-H7.

### 2. Establish same-map component parity

Construct one small float64 cloud, normalized weights, reset design, and
directional tangents. Feed exactly those tensors to the canonical reset
primal/JVP, any temporary score wrapper, and a TensorFlow autodiff or central
FD oracle of the same finite primal. Compare particles, weights, row/column
residuals, Contract-E moments, and tangents. Keep epsilon, iteration counts,
ridge, cost-floor branch, and cap settings fixed.

A mismatch localizes H2, H3, or C3 without a T=50 run.

### 3. Match the complete production program

Use one route, one dtype, one reset design, and one full control record:

~~~text
epsilon = 1.0 (or a newly selected score-scope value)
sinkhorn_steps = balance_steps = 8
flow_substeps = 12
correction_steps = 4
pairwise_steps = 4
correction_strength = 0.2
pairwise_strength = 0.02
pairwise_rms_cap = 2.0
coordinate_cap = 0.98
coordinate_cap_power = 8
lm_damping = 1e-2
lm_scale_floor = 1e-4
trust_radius = 0.5
ridge = 1e-5
~~~

The registry family above is fixed where the registry specifies it; epsilon
and flow remain warm-start candidates until score-scope tuning selects them.
Use the same initial particles, process noises, observations, and fixed branch
decisions for value, analytic score, and the FD oracle. Return per-step score
increments and all reset diagnostics.

If analytic JVP disagrees with FD of this same map, H6/H7 is active. If JVP
agrees with FD but both disagree with Kalman, the discrepancy is a
finite-program approximation, not a local tangent bug. If matching controls
removes it, C2 was causal.

### 4. Separate resampling from the reported cell

Run stages 1 and 2 on the same data first. For stages 2, record the complete
ancestor sequence, cumulative-weight distances to search boundaries, and
fixed-path FD/JVP. A differentiable relaxation, if tested, must be labeled a
new estimand and compared with its own finite primal.

Any difference appearing only at stages greater than one supports H1 as an
annealed-route issue, not as the explanation for the reported stages-one cell.

### 5. Measure flow convergence

With route and cloud frozen, evaluate flow_substeps 12, 24, 48, and 96. Use
both FD versus analytic derivative of each finite primal and score error versus
Kalman. Report per-step increments and wall time. Do not assume second-order
convergence: the implementation uses Euler updates.

### 6. Measure particle-size behavior

Use N=504, 1008, 2016, and 4032 only with
dpf_transport_exact_divisor_cap3000_v1. For each N use the same observations
and common random numbers within a comparison, then independent replicate
seeds for uncertainty. N=504 and 1008 use K=N; larger counts must be selected
by the repository selector, never by a fixture constant.

Report paired confidence intervals for mean score error and a variance or
bootstrap interval. A falling SE with a stable nonzero mean indicates
finite-program bias; a falling mean and SE supports a finite-particle
explanation. Neither conclusion follows from the current six seeds.

### 7. Tune the score scope

Create a score-specific tuning artifact on data disjoint from the claim
partition. Bind model, route, reset family, horizon, prepared-data regime,
particle count, dimensions, dtype/backend, chunk policy, and every tunable
control. Reject a missing, stale, caller-stamped, or cross-lane artifact.

The R2 plan proposes a Fisher gate of 40 tuning replications, three directions,
absolute mean score bias less than 3*SE + 0.05, and SE less than 1.0
(bayesfilter-r2-execution-plan-2026-08-27.md:165-170). That is a future
promotion criterion, not a result of this six-seed diagnostic.

### 8. Run the replicated claim comparison

Only after steps 1-7 pass, run the untouched claim partition with the
predeclared replication count, exact Kalman references for each observation
replication, common random numbers for lane comparisons, and a complete
manifest. Use paired intervals only where pairing is real; otherwise use an
independent or hierarchical analysis. Report hard vetoes first and call
remaining differences descriptive unless uncertainty supports a ranking.

## Proposed evidence contract for the repair run

| Field | Required statement |
|---|---|
| Scientific question | Does the registered canonical score route compute a directionally accurate score for the stated LGSSM, and which finite-program component controls the error? |
| Exact comparator | Kalman log likelihood and score for the same observations and direction; central FD with Richardson agreement is the reference. |
| Primary local criterion | Analytic JVP equals central FD/autodiff of the same matched finite primal at component and full-horizon levels. |
| Primary claim criterion | The reviewed R2 score-scope Fisher gate, applied only after tuning and route wiring. |
| Hard vetoes | Nonfinite values/tangents, unresolved registry endpoint, route/dtype mismatch, missing marginal/condition diagnostics, stale tuning, failed reset validity, or missing manifest. |
| Explanatory diagnostics | ESS, score increments, Sinkhorn residuals and TV, cost-floor activity, Cholesky condition, LM condition, cap activity, and flow convergence. |
| Heuristic adversary set | Exact Kalman filter (the certifying LGSSM oracle: any material loss is a hard veto); plain linear-Gaussian particle/importance sampling without OT correction (tests whether the added reset earns its cost); and finite LEDH flow with reset/correction disabled (tests whether the correction stack harms the basic flow). These are falsification baselines, not tuning targets. |
| Conditional situations | T=50, phi1 direction, low/high ESS steps, stages one versus greater than one, and each N in the ladder. |
| Nonclaims | Passing does not establish categorical PF equivalence, posterior correctness, HMC readiness, default readiness, statistical superiority, nonlinear validity, or unregularized-OT equivalence. |
| Preserving artifact | Committed JSON/Markdown result with plan, source commit, callable/route IDs, controls, seeds, data hash, hardware, dtype, diagnostics, uncertainty analysis, and decision tables. |

The heuristic set is deliberately weak and must not be optimized against. A
complex route losing to the exact Kalman oracle in this salient linear
Gaussian situation is a promotion veto even if it is internally consistent.

The conditional evaluation table for that gate is:

| Situation | Exact Kalman oracle | Plain particle/importance route | Uncorrected finite-flow route |
|---|---|---|---|
| T=50, phi1 direction, reported observations | Reference score and likelihood | Same observations, direction, cloud, and noise | Same observations, direction, cloud, and noise |
| Low-ESS steps recorded in a run | Reference contribution at those times | Conditional error and variance | Conditional error and variance |
| High-ESS steps recorded in a run | Reference contribution at those times | Conditional error and variance | Conditional error and variance |
| annealed stages 1 versus greater than 1 | Reference for the selected finite target | Same branch convention | Same branch convention |
| Each N in the ladder | Reference unchanged | Mean and variance versus N | Mean and variance versus N |

This table is a sanity gate, not a tuning objective. Passing it removes a
cheap counterexample; it does not certify posterior correctness or HMC
readiness.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Treat -0.46 as established bias | Not met: six descriptive seeds, 1.39 SE from zero | Evidence design insufficient; lanes unpaired | True replication variance and route mismatch | Matched common-cloud run after wiring repair | No unbiasedness or bias-magnitude claim |
| Attribute error to fixed ancestry | Not met for stages one; branch inactive | No resampling in reported cell | Annealed stages may differ | Test stages two/three with boundary diagnostics | No claim fixed ancestry is harmless generally |
| Accept value epsilon/flow tuning for score | Not met | Score scope has no selected tuning artifact | Score-sensitive controls may differ | Disjoint score-scope tuning | No score default or promotion |
| Accept analytic tangent locally | Met only on tiny finite maps | End-to-end registered route not wired | Long-horizon composition and finite bias | Full matched FD and registry tests | No Kalman equivalence |
| Compare value and score as same algorithm | Not met | C2-C4 are hard mismatches | Correction, reset, and dtype effects | Match complete family and reset | No lane ranking |
| Promote score cell to canonical/default/HMC | Not met | C1-C6 block admission | All above | Keep diagnostic-only; repair then rerun | No posterior, HMC, or production claim |
| Continue LEDH research direction | Justified as repair path | No continuation veto fired | Whether repaired route passes | Execute bounded diagnostic/tuning sequence | No claim it will pass |

## Inference-status table

| Inference status | Current answer |
|---|---|
| Hard veto screen | Fails claim admission: endpoint unresolved, score route/control scope mismatched, and score diagnostics/tuning absent. Cells were finite; finiteness is insufficient. |
| Statistically supported ranking | None. No candidate, lane, epsilon, flow count, or N is ranked with valid uncertainty evidence. |
| Descriptive-only differences | Six-seed score mean error -0.4554 +/- 0.3286, four negative errors, raw lane differences, and finite score-grid cells. |
| Default readiness | Not ready. The score result cannot support canonical, leaderboard, default, HMC, or scientific admission. |
| Next evidence needed | Wiring and canonical reset; matched controls/dtype; diagnostics; score tuning; common-cloud component/full-horizon FD; flow/N ladders; replicated uncertainty. |

## Verification performed

The focused CPU-only command was:

~~~text
CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/highdim/test_ledh_canonical_score_step.py \
  tests/highdim/test_ledh_canonical_score_recursion.py \
  tests/highdim/test_ledh_canonical_score_full.py \
  tests/highdim/test_higher_moment_contract_e.py
~~~

Result: 35 passed, 2 warnings in 87.40s.

An earlier broader collection that also named
tests/highdim/test_genut_shape_lm_tf.py failed during collection because the
repository currently lacks bayesfilter.highdim.cubature_genut_batch_tf. That
collection failure is a repository completeness gap, not a score-tangent
failure; the relevant score and higher-moment tests were rerun without that
missing module and passed.

These tests are diagnostic/reference checks. They use small fixtures and
autodiff only as an oracle; the claim-bearing score path remains analytical.
No GPU result, T=50 repair run, score tuning campaign, or new scientific
promotion evidence was generated here.

## Post-run red-team note

The strongest alternative explanation for the six-seed discrepancy is the
combination of C1-C4: the parity script did not execute the registered score
endpoint, did not use the registered correction family, and compared a
float64 duplicate reset with a float32 value reset. If a repaired,
common-cloud, full-family run still shows score error while analytic JVP
matches central FD of the same finite scalar, the remaining explanation is a
finite-program approximation (flow, OT regularization, reset, caps, or finite
N), not a missing local derivative term.

The result that would overturn this audit is a committed, score-scope-tuned run
in which the registry resolves, both lanes call the canonical reset with
identical controls/dtype, all validity diagnostics pass, the finite JVP agrees
with common-cloud FD, and replicated score error meets the predeclared
Fisher/Kalman criterion. The weakest current evidence is the six-seed unpaired
comparison and the score grid's finiteness-only check.

## Final disposition

The request is answered as follows:

- The Kalman reference is valid.
- The current score discrepancy is descriptive and underidentified.
- Fixed ancestry is not active in the reported annealed_stages=1 cell.
- The score/value comparison is invalid as a same-control comparison because
  of registry, correction-family, reset-route, dtype, random-stream,
  diagnostics, and tuning-scope defects.
- Local JVP tests support derivatives of selected finite maps, not canonical
  or Kalman score accuracy.
- The score lane is BLOCKED_FOR_CLAIM and must remain diagnostic-only.
- The next action is a bounded implementation-and-evidence repair, not
  rejection of the LEDH research direction.
