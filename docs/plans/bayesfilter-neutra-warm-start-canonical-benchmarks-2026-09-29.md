# Canonical benchmarks before q20 warm-start training

Status: proposed benchmark selection following the user's instruction to
test simpler models before q20. No training, SMC or HMC run is launched by
this note. The q20 application is deferred until the relevant mechanisms
have been checked. Earlier compute allowances are not renewed by this note.

Complete algorithm flowcharts and training/inference handoffs are in
`bayesfilter-neutra-warm-start-flowchart-2026-09-29.md`.

October 3 generalization extension: add randomized unwarped two-center and
three-center Gaussian-mixture families, with new target specifications held
out from procedure development. The construction, proposed interval choices,
information boundary and exact evaluation identities are in
[the randomized-mixture design](bayesfilter-neutra-random-mixture-generalization-design-2026-10-03.md).
This is a design addition; the current fixed-target implementation and past
campaigns do not constitute execution of these new families.

The [October 3 generic recovery and transfer plan](bayesfilter-neutra-generic-recovery-and-transfer-plan-2026-10-03.md)
integrates the later October 1–2 failure evidence, source-grounded remedies,
mathematical additions and required generalization phases. Its revised
Sections 3.1–3.3 make FAB, Gabrié, annealed AIS, annealed SMC, AFT and CRAFT
mandatory study rows with native-method, common-IAF and downstream assessments.
Generalization covers every qualifying method; one successful teacher does
not close the study. Use that plan for the next proposal; historical run
records retain their original scopes.

## Research question and audit

Can approximate forward training initialize the canonical IAF usefully, and
does subsequent classic RKL retain coverage while supporting useful
frozen-map inference?

Gabrié's procedure supplies concurrent sampling and forward training, with
initial walkers in the relevant basins. Mode discovery is a separate step.
Test training with known-region initialization and test blinded discovery
independently before combining them.

Skeptical audit: a Gaussian cannot test mode coverage; a balanced mixture
cannot expose incorrect basin weights; a curved unimodal target cannot
test lost basins; map-generated samples can conceal missing regions; RKL
can lose coverage achieved during initialization; equal update counts do not
equal total cost; and static densities cannot validate SMC²'s inner filter.
The ladder and checkpoint comparisons below address these risks. No q20
setting or historical trained map becomes a benchmark default.

## Recommended ladder

| Target | Initial reference | Purpose | Provenance |
|---|---|---|---|
| Correlated, anisotropic Gaussian, initially 2D | Exact density, draws, moments, normalizer and affine whitening | Check losses, gradients, weighting, sampler correction and basic training | Analytic control; ill-conditioned Gaussian is also a NeuTra-paper target |
| Unequal two-Gaussian mixture, 2D | Exact draws, responsibilities and region probabilities | Correct wrong initial basin proportions and detect RKL collapse | Gabrié Appendix G.1 / Figure 5 |
| Author's checked-in wiggle, 2D | Actual author-code density and independently refined quadrature | Curved geometry and the three measured local modes | Gabrié Appendix G.2 / Figure 8 motivates the fixture; the inspected broadcasting and September 29 mode search do not support labeling this code density unimodal |
| Invertibly warped unequal mixture, initially 2D | Exact density and independent reference draws through the known transformation | Combine curvature and multiple basins | Explicit local analytic benchmark, not a claimed author experiment |
| NeuTra-paper funnel, initially 10D | Exact generative draws and analytic whitening | Test the neck, tails and changing conditional scale | Hoffman et al. Section 4.1; their experiment uses 100 dimensions |

The 2D Gaussian and 10D funnel screening sizes are economical diagnostic
choices, not source-scale reproductions or validated dimension thresholds.
Increase dimension, separation, conditioning, curvature and rare-region
probability separately after understanding the initial cases. Full paper
replication has its own source settings; this small suite does not replace it.

### Definitions and numerical provenance

For the Gaussian, use `N(mu,Sigma)` with declared positive-definite covariance
and vary conditioning separately from dimension. The exact control
`x=mu+Lz`, `LL^T=Sigma`, checks the transformed target and diagnostics;
it does not prove that training learned this map.

For the initial mixture use

    p(u) = (1/3) N(u;(-5,0),I) + (2/3) N(u;(5,0),I).

Equal unit covariances, ten-standard-deviation separation and 1:2
probabilities come from Gabrié Appendix G.1. Symmetric centering and axis
orientation are declared local coordinate choices; the checked-in Gaussian
example uses other coordinates. Start with equal walker allocation to test
correction of incorrect initial proportions. A one-basin start is an
expected-failure diagnostic. Rare-component and unequal-width variants
follow later, with their numerical choices declared before execution.

Use `r_k(u)=w_k p_k(u)/p(u)` and the exact identity `E_p[r_k]=w_k`, alongside
fixed spatial-region checks. Mixture labels are not disjoint regions:

    P(u_1>0) = (1/3) Phi(-5) + (2/3) Phi(5),

which is not exactly 2/3. Initial allocations and optimizer hit counts are
not posterior mass estimates.

For the warped mixture use the invertible shear

    x_1=u_1,  x_2=u_2+b*(u_1^2-c),  c=E_p[u_1^2]=26.

The inverse subtracts the shear and the Jacobian determinant is one, giving
`p_x(x)=p_u(x_1,x_2-b*(x_1^2-c))`. Exact mixture draws therefore give exact
reference draws after transformation. The constant 26 is derived; curvature
`b` remains an explicit stress axis to set before execution. This preserves
the half-space probabilities and transports the responsibility diagnostics.
The analytic map is withheld from learned training except in an oracle arm.

Use the actual author's wiggle potential. Its radial energy term means that
Gaussian mixture weights do not supply its normalizer. Check the reference
integral by expanding the domain and refining quadrature, with a tail-error
assessment. The radius is nonsmooth at the origin; check the corresponding
score behavior without silently smoothing the target. Until these reference
checks pass, the exact warped mixture supplies the curved-target authority.

For the funnel preserve the NeuTra-paper convention:

    v ~ N(0,1),  z_i ~ N(0,1),  x_i=exp(v)*z_i,
    p(v,x)=N(v;0,1) product_i N(x_i;0,exp(2*v)).

Do not silently substitute the common `v~N(0,9)` / variance `exp(v)` variant.
Inspect neck/tail probabilities, conditional scales and moments against
independent samples. This target alone does not test multimodal coverage.

## Training mechanisms and checkpoints

Use the same canonical IAF family within each target comparison, with a
target-specific capacity/optimizer protocol. Initially compare:

1. Ordinary canonical RKL from the declared initialization: practical baseline.
2. Forward fitting to exact independent target examples, then RKL: a diagnostic
   control for whether architecture and training can use accurate examples.
3. Gabrié concurrent sampling/forward training, then classic RKL.
4. Ordinary annealed SMC, weighted forward fitting of the IAF, then the same
   RKL refinement protocol. Preserve the weighted empirical measure.

Save and assess every map immediately before and after RKL. Warm-start
accuracy is a deliberately varied coarse-to-refined budget, not a demand for
converged posterior samples before training. Known masses and exact samples
are evaluation information, available to training only in its declared
oracle arm; equal known-region initialization supplies locations, not masses.

Then add waste-free SMC with matched mutation, followed by AFT/CRAFT with
their own source-corrected transport and weights. These are distinct methods.
First isolate their effects. A composed mode search -> Gabrié -> SMC ->
forward fit -> RKL route can subsequently be tested as an additional arm.
SMC target/proposal support and bridge identities must be checked on each
fixture; an inherited Gaussian bridge is not automatically well suited to
the funnel or to distant modes.

Charge discovery, transitions, target evaluations, training, compilation and
validation in end-to-end costs. An exact-data or analytic-map oracle is
explanatory, not an affordable q20 competitor. Historical weighted-transport
runners must not replace the current canonical IAF authority.

## Evidence contract and next decision

| Role | Requirement |
|---|---|
| Engineering pass | Source-operation parity, valid density/score/Jacobian/weight identities, forward/RKL gradient checks and canonical IAF caller path |
| Warm-start viability | Independently assessed declared regions and within-region quantities; approximate initialization is allowed |
| Research promotion | Frozen-map posterior estimates meet predeclared accuracy and uncertainty requirements after refinement, with complete cost reporting |
| Promotion veto | Invalid derivatives/weights/target; uncertainty-supported material loss of important regions; failed downstream numerical or posterior checks |
| Repair trigger | Region lost by discovery, sampling, fitting or RKL; inadequate capacity/training; ineffective moves |
| Continuation veto | Invalid harness/reference, corrupted evidence, unavailable required validity check, exhausted approved budget |
| Explanatory only | Losses, ESS/CESS, acceptance, standard 1000-point score probe, runtime and unreplicated tails |
| Not established | Exhaustive q20 discovery, universal settings, perfectly Gaussian transformed density or a requirement to converge the posterior before training |

Assess region probabilities, moments and tail/conditional behavior before
and after RKL. For normalized targets, use independent reference samples
for heldout forward KL. Keep physical-region checks alongside the standard
1000-point base score probe. Exact whitening everywhere is not required for
useful HMC, and Gaussian-looking samples confined to one basin do not prove
coverage.

Predeclare accuracy margins, uncertainty methods and precision before runs.
Use independent populations/training replications for comparisons; resampled
particles are not independent replications. A short one-seed difference is
descriptive. If RKL loses a region present at its starting checkpoint, reject
that schedule for the stated goal; it does not establish failure of SMC or
forward initialization. Mixed refinement is a separate repair hypothesis,
not a silent change to the requested classic-RKL experiment.

If SMC² is included, add a small linear-Gaussian state-space model: Kalman
likelihood/smoothing checks the inner method, and one or two unknown
parameters permit independently checked parameter-posterior quadrature for
the outer method. That parameter posterior is not automatically Gaussian.
A finite-state HMM with exact forward recursion is another reference.
Static-density success cannot certify the nested particle construction.

## Sources and execution boundary

Gabrié paper Sections IV.A–C, Appendix G.1–G.2 and Figures 5/8 are locally
stored in `.localresources/fab-coverage-followup-20260928/gabrie-adaptive-flows.pdf`.
Pinned author code is under `.localresources/flonaco-author-20260929/upstream/`:
`flonaco/croissant_utils.py:46–64`, `tests/test_wiggle.py:22–35`, and the
sampler/trainer anchors in the source-adoption note.

Hoffman et al., *NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using
Neural Transport*, Section 4.1, is saved with its ResearchAssistant extraction
under `.localresources/q20-flow-training-literature-20260923/papers/`.
Author source `code/neutra.py:174–196` in that resource root explicitly builds
the funnel. The extraction marks equations unreliable, so the scaling was
also checked against source. This uses existing resources, with no new
package installation or GPU process.

The next executable plan should begin with source parity and the Gaussian
and unequal-mixture cases, price the batch/training/replication ladder, and
freeze numerical choices, total allowance, output directory and stop rules.
No production readiness or stochastic ranking has been established here.

Bounded MathDevMCP/SymPy checks confirmed the scalar determinant expansion
and responsibility cancellation; results and the unsupported matrix-syntax
attempt are saved in
`artifacts/neutra-warm-start-benchmark-design-2026-09-29/symbolic-checks.json`.
These algebra checks do not validate an executable sampler or trained map.
