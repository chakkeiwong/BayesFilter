# Lorenz full-covariance consumer and fitting diagnosis

This bounded repair belongs to the authorized controlled-SMC/KDM master. It
follows the checked Gaussian-normalizer identity and precedes any Lorenz
empirical matrix. The question is whether a full-covariance TensorFlow consumer
computes the source Lorenz forward and fitting operations, and whether the
source normal-equation fit remains numerically resolved on its own clouds.

## Source and mathematical target

The published controlled-SMC paper section6.2 and equations42--44 specify the
Lorenz model. Pinned author StateSpaceModels/Lorenz/cSMC_VaryF.m:3--89 specifies
d8, H101, N512, RK4 interval.1 with ten substeps, prior.01I, transition.001I,
six observed coordinates and observation variance1e-4. cSMC_Resample.m:48--130
implements Gaussian twisting and forward weights;174--229 implements full
quadratic least squares and additive policy refinement. The full-quadratic
d>1 route does not apply the source diagonal .001 precision projection.
RungeKutta4.m:13--23 fixes the integrator. Source and normalized observation
conventions differ by the already checked constant per observation.

For psi(x)=exp(-x'Ax-b'x-c), base N(m,C), the twisted precision is
P=C^{-1}+2A, covariance K=P^{-1}, mean K(C^{-1}m-b), and log integral
.5(logdetK-logdetC)-c+.5(C^{-1}m-b)'K(C^{-1}m-b)-.5m'C^{-1}m.
Backward fitting minimizes the same unweighted least-squares residual as the
source, including the difference of next-policy Gaussian log integrals.
Include every cross term in the quadratic basis. Derive the source's ordered
interaction coefficients from x2fx and squareform, not from scalar assumptions.

## Frozen checks and diagnostic roles

Use the supplied101x6 data, forcing4.8801, N512, one refinement, and seeds
20260929/20260930. This is a consumer diagnostic, not the40-forcing/100-repeat
Figure6 matrix or the dimension/noise Table1. Execute unchanged source code in
Octave with GPU hidden. Preserve both rounds, particles, policies, weights,
normalizers, ancestry and recovered random inputs. Reference extraction must
bind the unchanged source functions; no modification of the author fitter.

The TF consumer must reproduce analytic multivariate Gaussian integrals,
RK4 steps and fixed-policy source weights, normalizers and replay paths.
Tolerances are1e-7 absolute plus1e-9 relative for weights/paths/log normalizers,
and exact ancestry. Before free replay, source-cloud evaluation separates
algebraic errors from ancestry sensitivity. Shape/dtype, finite values,
positive precision and source normalization identities are validity vetoes.
Report minimum precision margin and least-squares conditioning instead of
discarding them. A source fit failure or unresolved coefficient discrepancy
blocks fitting parity but does not invalidate checked forward algebra.

Compare source normal equations with centered/scaled QR on the same target.
This is an algebraically equivalent least-squares solver diagnostic, not a
ridge, damping or new fitted objective. For z=(x-mu)/s and fitted Az,bz,cz,
recover A=S^{-1}AzS^{-1}, b=S^{-1}bz-2A mu,
c=cz-(S^{-1}bz)'mu+mu'A mu. Check healthy synthetic exact-quadratic recovery,
unweighted residual non-increase to1e-8 relative, fit-value differences and
proposal-precision validity. Do not promote a solver based on downstream
likelihood improvement. If the source fit is numerically unresolved, record
that result; a separate unchanged-objective solver repair must be reviewed
before any empirical comparison relies on it.

The source-relative forward pass is engineering evidence. Fitting residuals,
ESS and conditioning are explanatory or repair triggers; none certify the
empirical Lorenz claim. No state-space dimension, data convention, likelihood
normalization or accepted threshold may be changed to force a pass. Missing
exact Table1 datasets remain a provenance gap separate from executable math.

## Defaults, budgets and skeptical audit

Source settings above are source-recovered diagnostic controls, not universal
defaults. FP64/GPU/XLA and memory growth follow the established parity scope.
Full precision must be positive with a relative64eps margin; invalid matrices
are flagged before a checked Cholesky can supply accepted results. No ridge or
precision projection is introduced. The source solve may be ill-conditioned
because particles concentrate around nonzero trajectories; centering/scaling
is tested against the same objective to expose that failure mode.

c1_lorenz_consumer: <=90 CPU seconds per attempt, <=180 total, at most2 attempts.
c2_lorenz_consumer: <=300 GPU seconds per attempt, <=600 total, at most3 attempts,
requiring the exported source reference. Outputs are new campaign attempt
directories, with source/data hashes, exact commands, seeds, backend/memory,
all checks and failure classifications. These phases fit the existing campaign
balance; total allocation remains CPU300/GPU18000. Extend the attempt ceiling
only to accommodate these five bounded attempts, to30. They run after the active
neuroscience matrix through the same supervisor, without concurrent GPU jobs.

Skeptical audit PASS for this bounded consumer and fitting diagnosis: baseline,
full covariance, source-specific normalization, cross terms, unchanged objective,
validity guards and stop conditions are explicit. Unchanged source execution
comes first. A failed source-normal-equation comparison triggers diagnosis or
equivalent-solver repair, not a conclusion that the published method is wrong.
No full Lorenz or whole-master completion claim is permitted from these controls.

## Implementation review before execution

The new consumer uses the full x2fx order (constant, linear, lexicographic
interactions, squares), verified against the pinned statistics x2fx.m:42--55
and Lorenz cSMC_Resample.m:192--220. Off-diagonal A stores half each cross
coefficient. The scalar covariance shortcut is absent. The actual forward,
source-cloud and backward-fit entry points have stable signatures and native
TensorFlow time loops; no pfor, NumPy runtime, or per-sample Python loop.
Base covariance and twisted precision must be symmetric and positive. Guards
only reject; no accepted covariance is ridged or projected.

The diagnostic tests analytic full Gaussian conditioning, invalid precision
rejection, healthy full-quadratic recovery, author RK4, both author policy
rounds on fixed clouds and recovered inputs, normalized likelihood constants,
and every used signature's trace count. Source coefficient disagreement is
recorded separately from forward validity: an unresolved source normal-equation
fit cannot be described as fitting parity even if the diagnostic completes.
Both solvers are compared on identical backward targets from the source policy;
recursive policy differences therefore cannot masquerade as a residual benefit.
A QR residual floor of1e-12 is used only to define the relative comparison
near zero (absolute slack1e-20); it does not change a fitted value or objective.
Python syntax review passed; executable numerical checks remain pending.

## Source export repair, attempt23

Unchanged author execution completed with finite normalizers and positive
precisions, but nested Octave cell structures survived SciPy simplify_cells
as mat_struct objects and JSON serialization failed. This is an export
failure, not a rejected numerical method. Recursively convert their named
fields to standard dictionaries; verify against the preserved export.mat
before the second source attempt. Source normal-equation singularity
warnings are preserved for the planned conditioning diagnosis. Attempt23
consumed9.684508897 CPU seconds; no GPU consumer launched.
