# Structural integration in the surrogate-HMC master

This is the next implementation stage of the existing closure roadmap, under
the owner's standing execution request. Finishing the first diagnostic pass
did not finish that roadmap or exhaust its budget. Every subsequent action
must identify whether it belongs to this master; in-scope work continues unless
a stated continuation veto or the remaining campaign budget prevents it.

## Question and scope

Can the actual canonical analytical DCTR consumer preserve an AR(2) transition
with one innovation and an exactly deterministic lag? The goal remains testing
surrogate-force HMC. This prerequisite is not itself posterior validation.
Implement a general additive structural transition
`x = f_theta(ancestor) + B_theta e`, where B has full column rank and
`e ~ N(0, S_theta)`, through the shared canonical executor. This covers AR(2)
and rectangular, rotated innovation maps. Nonlinear dependence on the current
innovation and nonlinear completion manifolds remain unsupported by this
additive contract; do not claim them from this stage.

Chapter 18b, equations and discussion in sections
`bf-linear-kalman-hides-structural-split` and
`bf-nonlinear-filters-need-structural-map`, supplies the integration-space
requirement. The following is a local derivation of its application to the
existing LEDH proposal, not a claim that the chapter specifies this proposal.

For each fixed ancestor, define J = (B'B)^(-1) B'. The pre-flow innovation is
`e0 = chol(S) z`. Run the existing LEDH affine flow in innovation coordinates,
with observation `h(f(ancestor)+B e)`, observation Jacobian `H B`, prior centre
zero, and proposal-design covariance `J P_pred J'`. Reconstruct
`x1 = f(ancestor)+B e1`. The shared UKF still propagates full-state moments
using `Q = B S B'`, which may be singular. No full-state process Cholesky is
required. This proposal-design covariance is a hypothesis for efficiency;
importance weighting does not require it to equal the conditional covariance S.

Both transition factors in the PF-PF ratio use Lebesgue measure on the same
innovation coordinates: `log N(e1;0,S) - log N(e0;0,S) + log|det d e1/d e0|`.
The observation factor is evaluated at x1. If expressed on the affine image,
the common volume factor of B cancels in that ratio. It is wrong to treat either
innovation density as a full-state Lebesgue density. Keep the existing full-state
Contract E reset, dual-cap corrections and per-particle UKF covariance lifecycle
after weighting. Reset approximates the marginal filtering law across ancestors;
its output is the next ancestor cloud, not a conditional transition proposal.

Analytical tangents include dB, dS, df, dP and all observation dependence.
For G=B'B, `dJ=G^(-1)(dB' - dG J)`; for x=f+B e,
`dx=df+dB e+B de`. Differentiate the reduced covariance and Jacobian by the
product rule, and compose with the existing analytical flow/reset derivatives.
Fail closed on rank-deficient B, invalid S, invalid flow or missing declared
tangent partners. Do not ridge S or add artificial lag noise.

## Evidence contract and skeptical audit

Primary engineering criteria: actual batched consumer executes the structural
law, pre/post-flow deterministic residuals are at rounding error, B=I agrees
with the existing full-rank path, and analytical scores agree with central
differences of the same full finite program, including reset and correction.
Use multiple parameter directions (transition, S and B), more than one time
step, and explicit batch/single-cloud parity. Rank loss and nonfinite inputs
are vetoes. A deliberate singular-Q call without structural metadata must still
fail. These checks do not require the DCTR force to equal the endpoint gradient.

Kalman is the exact AR(2) likelihood and filtering comparator. Before HMC,
qualify independent parameter references and freeze scope-specific DCTR and
public HMC tuning on separate partitions. Subsequent posterior criteria remain
the roadmap's B/R, C/D, D/R and C/R comparisons. Bootstrap propagation,
no-flow importance weighting and Kalman provide the cheap diagnostic adversaries;
this structural implementation stage makes no efficiency ranking.

Defaults: retain the existing flow/reset/correction algorithms and numerical
controls solely as mechanics-test fixtures, not tuned AR(2) defaults. Use FP64
first to expose derivative errors, then the GPU/XLA candidate lane; precision
transfer requires parity. J uses a checked Gram solve; poorly conditioned B
can invalidate the route and must be diagnosed, not repaired with silent noise.
The key pre-mortem is a finite value with an off-support proposal or a missing
dB term. Residual, identity-map and score checks distinguish those implementation
failures before sampling. Audit passed for this bounded implementation stage:
the oracle is the actual finite program and exact AR(2) model, no proxy is a
posterior promotion criterion, and the full canonical call chain is tested.

## Execution and resources

Reserve at most 2,400 CPU-worker seconds and 3,600 GPU-wall seconds for this
structural integration pass, including failed checks; three numerical attempts
per stage, with charged local test repairs allowed. This is inside the existing
remaining 25,598.567587 CPU / 200,177.573144 GPU seconds, not expanded compute.
Use the closure master, stage `structural-integration`, and fresh directories
`docs/plans/artifacts/surrogate-hmc-closure-20260922-structural-NN`.
The first pass's unused reservation is not transferred or double-counted.

The master records sources, command, environment, seeds, elapsed resources and
structured findings. GPU execution uses the existing tftwogpu interpreter,
trusted device access and verified memory growth. CPU checks intentionally hide
CUDA. Stop a numerical attempt on invalid mathematics, missing required evidence,
timeout or its resource allowance. A failed implementation check triggers a
localized repair under the same pass budget. A successful prerequisite triggers
the next existing-master task: AR(2) reference, scoped tuning and inference.
