# KSC full-seven-mixture comparison correction — 2026-09-29

## Scope and configuration

The owner requests correction of the single-Gaussian Kalman comparison using
all seven KSC observation components. Replace that column in the main report.
Retain the old report as historical evidence for a different-model heuristic.
The existing four particle routes and independent density-grid reference
already use the full mixture. Reuse all 32 saved final datasets and 128 particle
evaluations, actual coordinates and exact-scope tuning evidence. Do not rerun
or retune particle candidates. No native-SV, default, production or HMC claim.

New reference: FP64 TensorFlow GPU/XLA, TF32 off, verified memory growth.
This is a deterministic diagnostic variant, not production FP32/TF32. Particle
columns retain their separately tuned N=1008 configurations. Reference resolution
is checked numerically, not fitted to particle results.

## Target and derivation

Keep h0~N(0,1), h_t=gamma*h_(t-1)+eta_t, eta~N(0,1), gamma=Phi(theta[0]),
z_t=h_t+2*theta[1]+e_t, transition before observation. The seven weights p_j,
means m_j and variances v_j are the existing sqmc_ksc_tf.py constants.
The evaluation point and observations are read from the archived campaign.
This is the KSC mixture approximation, not the exact log-chi-square model.

For predictive component (alpha,a,P), each of the seven observation components
gives the following ordinary scalar Kalman update:

    r_j = z - 2*theta[1] - a - m_j
    S_j = P + v_j
    q_j = alpha*p_j*N(r_j;0,S_j)
    K_j = P/S_j
    mu_j = a + K_j*r_j
    V_j = P*v_j/S_j
    Z = sum(q); log-likelihood increment = log(Z); w_j = q_j/Z

These formulas follow by multiplying the two Gaussian densities and completing
the square. All seven branches must be retained. Exact enumeration grows as
7^T; seven fixed posterior components would not be exact.

For bounded computation, evaluate the posterior Gaussian sum f on fixed
trapezoidal nodes x_r, then normalize alpha_r=f(x_r)*dx_r. Predict from each
atom as N(gamma*x_r,1), and branch into all seven components at the next update.
With M nodes there are 7*M Gaussian branches before projection. The first
update uses N(0,1+gamma^2) exactly, giving seven exact posterior branches.
No observation-component merging or branch pruning is performed. This is a
quadrature-compressed Gaussian-sum Kalman reference; finite-M projection is
an approximation that requires convergence checks, not an exact filter.

Propagate both score coordinates analytically, including normalization:

    dr = -2*dtheta[1] - da; dS = dP
    dlogN = -r*dr/S + .5*(r*r/(S*S)-1/S)*dS
    dq = p*N*dalpha + q*dlogN
    dlogZ = sum(dq)/Z; dw = dq/Z - w*dlogZ
    dK = dP/S - P*dS/(S*S)
    dmu = da + r*dK + K*dr; dV = (v/S)^2*dP

For projection differentiate every mean, variance and weight:
dN=N*((x-mu)/V*dmu + .5*((x-mu)^2/V^2-1/V)*dV).
Differentiate the quadrature mass normalization too. Fixed nodes imply
 da=x_r*phi(theta[0])*dtheta[0] after prediction; initial dP is
2*gamma*phi(theta[0])*dtheta[0]. The score is the total derivative of this
finite numerical likelihood, not a transported-component-only derivative.

## Evidence contract and diagnostics

Primary pass condition: finite likelihood and both scores on all 32 archived
final datasets; last-refinement discrepancies <=1e-8 in log likelihood and
<=1e-7 per score coordinate; independent old density-grid agreement at the
same tolerances. Ladder (M,bound): (401,40),(801,40),(1201,40),(1201,48).
Compare the final result with 801/40 and 1201/40. Record 401/40 as well.
Preserve actual values/scores, posterior moments, projection-mass discrepancy,
positive variance/mass guards, runtime, device placement and allocator peak.
Convergence agreement is empirical evidence, not a rigorous error bound.

Before final data: exact T1/T2 mixture enumeration at several parameter points;
centered finite differences of both score coordinates, including a coarse
projection grid to exercise normalization derivatives; GPU graph/XLA parity
with stable signatures. Audit consumer-to-implementation wiring through the
actual runner, not only a standalone helper. Reference derivatives are
analytical; diagnostic exact-enumeration autodiff is an independent check.

Wrong target/data/timing, missing coordinates, nonfinite results, derivative or
parity failures, unresolved nonconvergence and exhausted budgets are continuation
vetoes. Infrastructure faults are repair triggers with at most two retries/unit.
Poor numerical accuracy is not infrastructure failure. A failed particle
candidate is distinct from invalidating the reference or the research direction.

Recompute particle errors against the new accepted reference; report actual
scores, per-coordinate and L2 errors, likelihood errors, SD, SE, and all 24
paired 95% t intervals. Eight independent data/design pairs estimate joint
variability, not fixed-dataset Monte Carlo error. Pairwise intervals remain
exploratory without multiplicity adjustment. Retain all four particle routes;
no general winner, default readiness or method superiority follows from a smoke.

Constructed heuristic adversaries: zero score (uninformative force), the first
observation score (ignores later data), and IID (standard design), evaluated
separately by horizon and dataset. The old one-Gaussian score is historical
explanatory evidence for a different-model heuristic, not the corrected KSC
reference. The converged Gaussian-sum method is an accuracy reference, not a
stochastic arm to rank by runtime. Heuristic losses veto promotion in their
situations without rejecting a whole research direction.

## Assumptions, pre-mortem and skeptical audit

| Choice | Provenance/justification | Failure diagnostic and status |
| --- | --- | --- |
| Unchanged target/data | Archived campaign isolates comparator change | Hash/data parity; frozen scientific scope |
| Seven branches | Gaussian-product derivation above | Exact enumeration; required mixture target |
| Fixed quadrature projection | Deterministic approximation to state integration | Extent/resolution ladder; numerical approximation |
| FP64, TF32 off | Existing accuracy campaign | Graph/XLA checks; diagnostic variant |
| No pruning/merging | Avoid component-selection bias | 7*M branch count; bounded reference |
| No clipping/ridge/damping | Q and each v_j strictly positive; S,V have analytic positive lower bounds; shifted exponentials control scale | Fail closed on nonpositive mass/variance or nonfinite output; no factorization requiring ridge |
| Tight tolerances | Smaller than particle errors; tighter than prior reference | Independent formulation plus refinement; empirical bounds only |

A successful command could mislead through a hidden one-Gaussian collapse,
shared coding error, wrong derivative, or calling finite quadrature exact.
Explicit branching, short exact enumeration, finite differences and an
independent density-grid formulation address these risks. Grid convergence
refines a deterministic integral; final data never select particle controls.

Skeptical local plan review: the one-Gaussian baseline is a valid crude heuristic
for a different likelihood, but does not answer the requested full-mixture
comparison. This plan replaces it without changing the target or erasing valid
particle evidence. Exponential exact branching is explicitly acknowledged;
projection error is checked before accepting results. Pass for implementation
and bounded checks; numerical acceptance remains conditional on those checks.
No independent agent is requested or launched.

## Budget, preservation and commands

Inherited ledger: docs/plans/artifacts/sqmc-ksc-sv-20260928/budget.json,
SHA256 2a7c3de939d0bd0ae6cb8cca9db6cc57ca9b1afe87a4d6b41d0d6041a2c2ef15.
Prior charge 24410.862528797064s; remaining 18789.137471202936s under the
original 43200s cap, including the unchanged 300s old-hook reserve. Keep the
old ledger immutable and link it from a new correction ledger. Allocate
3600 GPU-owning worker wall seconds to this correction, including checks,
compilation and failed attempts. Deadline: 2026-09-30T16:15:40.010888+00:00.
A supervisor records each attempt before launch and enforces both limits.
CPU-only tests/reporting/commit checks add zero GPU time. Two infrastructure
retries per unit; fresh versioned directories for every launch.

Versioned root: docs/plans/artifacts/sqmc-ksc-full-mixture-20260929/attempt-01/.
Preserve all inputs, snapshots, logs, commands, environments and numerical
resolutions. Write new corrected tables/result note, add a historical notice
to the prior reader-facing note, and replace the active checkpoint.
Interpreter: /home/chakwong/anaconda3/envs/tftwogpu/bin/python.
CPU: CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_TEST_DEVICE_SCOPE=cpu python -m pytest
 tests/highdim/test_sqmc_ksc_gaussian_sum.py -q, full log to the new root.
Escalated GPU supervisor: python docs/benchmarks/run_sqmc_ksc_full_mixture.py
 --output docs/plans/artifacts/sqmc-ksc-full-mixture-20260929/attempt-01
 --mode supervise. Bounded checks precede full saved-data execution.
Finish with arithmetic/engineering/interpretation terminal review and a
CPU-only local Git commit. No push, publication, packages, HMC or promotion.
