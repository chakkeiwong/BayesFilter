# Controlled-SMC replication: code, mathematics and root causes

Audit completed 2 October 2026 through attempt129. The current Lorenz failure
is explained substantially better, but no repaired adaptive filter has been
admitted. Both controlled-SMC and KDM empirical replications remain incomplete.
The executable master is `scripts/run_controlled_smc_kdm.py`; its authoritative
scope is [the master plan](controlled-smc-kdm-replication-master-program-20260928.md).
The [repair and investigation plan](controlled-smc-lorenz-root-cause-audit-20261002.md)
defines the next work. No production numerical code was changed by this audit.

## What the actual consumer executes

The failed safety harness calls `LorenzRefinement.run`, which calls the shared
`MultivariateGaussianControlledSMC.fit` and `.forward` endpoints. The former
regresses backward through time, adds the fitted quadratic correction to the
old policy, optionally projects its precision/covariance, and passes the
resulting policy into the Gaussian forward sampler and unchanged Lorenz RK4
map. Attempt128 checks this binding in executable code, verifies the live
controller/kernel against the saved source snapshot, and reconstructs saved
initial particles from their policy and actual recorded standard normals.

Relevant call sites are
[`LorenzRefinement`](../../bayesfilter/replication/lorenz_refinement_tf.py),
[`_fit`, `_forward`, `multivariate_twist` and the projection](../../bayesfilter/replication/multivariate_controlled_smc_tf.py),
and [the safety harness](../benchmarks/validate_lorenz_precision_safety_tf.py).
Exact anchors: controller lines90–120; kernel lines31–110,153–231,305–438.
The loop bodies use native TensorFlow loops. The saved GPU runs report stable
signatures and one trace per endpoint; NumPy appears in the explicitly
independent CPU diagnostics, not in this algorithmic call chain. There is no
evidence that Python iteration, retracing or a NumPy runtime path caused the
current failure. The new forensic runs are deliberately CPU-only graph/reference
exceptions and are not GPU performance evidence.

## The Gaussian mathematics identifies the missing safeguard

Write the policy as

\[
\psi(x)=\exp[-q(x)],\qquad q(x)=x^T A x+b^T x+c.
\]

For a base Gaussian with mean \(s\), covariance \(Q\), and precision
\(B=Q^{-1}\), multiplying by this policy gives

\[
P=B+2A,\qquad C=P^{-1},\qquad m=C(Bs-b).
\]

These are the published Gaussian formulas, not an empirical interpretation.
See the [published paper](../../.localresources/papers/heng2020-controlled-smc-published.txt),
Section 6.2, equations43–44 and the constraints following them, and the
[supplement](../../.localresources/papers/heng-controlled-smc-author-manuscript-and-supplement.txt),
Section 9.2, text lines3591–3640. Positive definiteness of \(P\) makes the
Gaussian integral finite. It puts no bound on \(m\), because \(b\) can be
arbitrarily large. A condition-number limit alone also bounds neither the
absolute covariance scale nor the mean.

For an update \((\Delta A,\Delta b)\) with the same base Gaussian, direct
subtraction gives the exact identity

\[
m_{new}-m_{old}
=-P_{new}^{-1}(\Delta b+2\Delta A\,m_{old}).
\]

The factor in parentheses combines the linear and quadratic updates. A
small eigenvalue of the new precision magnifies its component in that
direction. Attempt128 checks this identity against the saved policy and
particles; attempt129 decomposes the fitted update that supplies it.

## Confirmed causes and their limits

### 1. Unconstrained quadratic fitting can produce an inadmissible Gaussian

The author's Lorenz code forms normal equations at
`StateSpaceModels/Lorenz/cSMC_Resample.m:187–193`. Lines195–212 impose a
positivity constraint only for the scalar or `purequadratic` branch. The full
quadratic branch at213–215 has no corresponding matrix constraint. The
published paper makes Gaussian feasibility conditional on positive
definiteness; footnote 4 suggests projection if necessary but does not specify
our relative floors, activation thresholds, or a bound on the proposal mean.

This is an actual missing robustness provision in the full-quadratic source,
not evidence that the paper's conditional Gaussian formula is wrong. In
attempt95 the original author's backward-fitting body, executed in Octave on
the frozen TensorFlow failure cloud, also loses positive definiteness at
refit 3/time 12: minimum eigenvalue approximately−2077.6, versus−1730.2 in the
TensorFlow QR route. Both methods are valid on the first two frozen refits.
The unchanged source also passes six separate native cases. Consequently,
neither “the author code always fails” nor “copying it guarantees success” is
supported. See [the crossover result](controlled-smc-lorenz-matrix-crossover-result-20261001.md).

### 2. A very accurate local fit extrapolates catastrophically

On the locally safeguarded attempt121 trajectory, refit 4/time 0 produces the
following Gaussian **before projection at that time**. No time 0 projection
activates in 121.

| Quantity | Frozen-input result |
|---|---:|
| Previous largest absolute proposal mean | 1.3231 |
| New largest absolute proposal mean | 635.0103 |
| New maximum coordinate standard deviation | 0.6921 |
| New minimum precision eigenvalue | 1.8157 |
| New precision condition number | 6130.59 |
| Smallest/largest centered-design singular value | 0.124816 |
| In-cloud regression RMS residual | 0.009286 |
| Regression-target standard deviation | 130.879 |
| Mean displacement in previous-covariance Mahalanobis units | 9885.34 |

The design condition number is about 8, so rank loss or a poorly solved least
squares problem does not explain this decisive fit. The largest sampled
coordinate is about 637, close to the proposal mean; a rare noise realization
is not the cause. The seventh coordinate is roughly7660 empirical coordinate
standard deviations from the fitted cloud center. A small residual on that
cloud says nothing about the quadratic's global extrapolation.

The future-value approximation supplies this failure. Let
\(L_{old,t+1}=\log f(\psi_{old,t+1})\), with the same definition for the new
policy. The stored log incremental weight and fitted target satisfy

\[
\ell_t=\log g_t+L_{old,t+1}+q_{old,t},\qquad
y_t=-\ell_t-(L_{new,t+1}-L_{old,t+1}).
\]

Because the old policy is in the full quadratic regression span, the raw
unprojected update obeys

\[
q_{new,t}=q_{old,t}+\operatorname{LS}(y_t)
=\operatorname{LS}[-\log g_t-L_{new,t+1}].
\]

This identity is checked on frozen inputs, including constants, in 129. The
code omits the same known observation-density normalization constant on both
sides; its separate log-normalizer accounting is not changed here. The
observation term is already an exact quadratic with precision contribution
10000 on the first six coordinates. The remaining two coordinates are not
directly observed.

Along the weakest direction at refit 4/time 0, precision decomposes as

\[
1.815712=100+0.480244-98.664531.
\]

The terms are prior precision, observation curvature, and fitted negative
future-log-integral curvature. That direction lies 99.9952% in the two
unobserved coordinates, measured by squared Euclidean mass. Its linear
coefficient is approximately 0.7756 from observations and 1232.3148 from the
future term. The mean-shift identity assigns Euclidean norm677.02 to the
future correction and 0.975 to the current-weight contribution. These are
algebraic contributions on the same saved inputs, not causal interventions.

The earlier invalid-precision event has the same type of mechanism: at
refit 3/time 12 its weakest-direction precision is
\(1000+68.345-2798.509=-1730.163\). The future term can overcome both the
transition and observation curvature. We have not yet established how much
upstream projection, cloud coverage, and backward propagation each contribute
to that future term. The large time 0 mean occurs on our previously safeguarded
trajectory; it must not be attributed unconditionally to an untouched author
trajectory that would already have failed at refit 3.

### 3. The covariance floor controls the wrong quantity for this failure

For an already positive precision spectrum \(\lambda_j\), the implemented
`nearest_covariance` rule computes

\[
\kappa_j=1/\lambda_j,\quad
\kappa'_j=\max(\kappa_j,\rho\max_k\kappa_k),\quad
\lambda'_j=1/\kappa'_j.
\]

This floors small covariance eigenvalues. Equivalently, it caps large
precision eigenvalues. It leaves the smallest positive precision eigenvalue
unchanged. It therefore does not limit the largest variance or the damaging
mean amplification. This conclusion follows directly from the code; it is
not a speculation about optimization.

In attempt126, the trigger activates at time 0 and the0.01 floor reduces
several precision eigenvalues from roughly10626–11131 to 181.57 while leaving
1.8157 unchanged. The condition number improves to 100, but the mean remains
635.0266. Thus a better condition number did not repair the failure. The
implementation does perform its declared covariance projection; using this
quantity as a sufficient test of proposal safety was the mistaken assumption.

### 4. Changing only quadratic curvature does not control the whole update

The projection changes \(A\) while retaining the original global \(b,c\).
Its updates to centered coefficients correctly represent that same global
polynomial; they do not preserve its gradient at the fitted cloud center.
A change \(D\) in \(A\) adds \(x^TDx\), changes that gradient by
\(2D\mu\), and changes the Gaussian mean. Each projected future policy also
changes the target fitted at the preceding time. This is a derived coupling,
not an additional indexing bug. How much it drives the earlier propagation
is still unmeasured. The initial mean 635 in 121 is not caused by a time 0
projection, because there is none.

### 5. Source normal equations and backend behavior confounded early comparisons

On32 captured author regressions, the unscaled design's singular ratio is
about 1.5e−8 to 2.4e−8. Normal equations square its condition number. Very small
Gram/RHS rounding differences can therefore change coefficients substantially.
In94,15 of 32 Octave source solves invoke its singular/minimum-norm fallback;
the other17 use its positive-definite route. TensorFlow raw solves do not
automatically reproduce that selection. Centered QR agrees with an independent
centered-QR reference on all 32 cases, but agreement with a numerically different
source solve is a separate question. MATLAB equivalence remains unchecked.

These facts explain an earlier port discrepancy. They do not explain the
well-conditioned time 0 failure in 128–129, and switching back to the source
normal equations does not rescue the frozen invalid-precision cloud in 95.

### 6. Two local arithmetic defects were real; full-recursion parity is still open

Attempts99/101 found XLA eigenpair residual7.83e−9 despite FP64. Native
Jacobi refinement reduced it to 1.57e−15 in 102. Expanded Gaussian correction
integrals also lost accuracy through cancellation; the completed-square
evaluation subsequently passed selected independent80-digit checks. These
are repaired numerical components, not proof of a repaired filter.

The0.001 covariance candidate still fails three full-recursion coefficient
comparisons at row 0/refit 4 in 114 (maximum A discrepancy0.098895). Its first
meaningful divergence must be traced using identical inputs and downstream
Gaussian quantities. Attempt128's404 successful fixed-time replays use the
saved next-time policies of 121/126 and a diagnostic tolerance. They do not
perform an independent complete backward recursion and cannot clear114.
Rounding, recursive amplification, policy-coordinate cancellation and a
reference defect remain hypotheses until localized.

### 7. RK4 overflow is a downstream consequence; detection comes too late

Attempt123 sends the failed initial cloud through the unchanged author RK4
and a separate equation implementation. Both overflow on all 512 particles
and all eight coordinates; the previous-refit control remains finite.
Silently clipping states or changing the integrator would change the specified
transition map and would not resolve the proposal defect.

There is an independent engineering weakness in `_forward`: it discards
per-step validity returned by the Gaussian helper, computes lookahead and
resampling even after nonfinite quantities appear, and checks full-trajectory
validity at the end. The fit loop similarly continues after its accumulated
validity becomes false. The controller ultimately rejects these runs; no
evidence here shows invalid results being admitted. Early termination and a
preserved first-failure record would save work and make diagnosis much easier.
The controller also reduces rich fit diagnostics over time, losing the
location and components of the first problematic update.

One additional code-review issue deserves a focused check:
`projected_correction_integral` includes `fallback_valid` from every row even
where the stable projected branch supplied the selected answer. An unused
branch could produce a false veto. It is not the explanation of 121/126,
whose backward fits passed validity before forward failure.

## What this says about the replication effort

We combined several different questions too early: matching the author
implementation, replacing unstable linear algebra, inventing robust policy
constraints, and reproducing the empirical efficiency figures. A complete
native multi-refinement baseline and matched-source-input comparison arrived
late. Repeated covariance-floor changes then addressed integrability while
the mean geometry remained unmeasured. Those are correctable methodological
mistakes on our side. The author source's missing full-matrix constraint and
unspecified robustness choices are genuine reproducibility limitations, but
the present evidence does not invalidate the paper or controlled-SMC theory.

There are separate unresolved tasks outside this Lorenz diagnosis. In the
RBF example the recorded audit found a parent/proposal mismatch in the source;
the corrected forward consumer has scoped parity evidence. NNLS conditioning,
stopping and discontinuous top16 retention made full fitting sensitive.
Keeping all positive coefficients and refitting repaired a normalizer screen,
but changes the published M16 procedure and did not establish filtering
performance. See the [RBF result](controlled-smc-rbf-fitting-execution-result-20260929.md),
especially its final assessment of 66–68. The KDM/IWSG primitives and sequential
checks pass; its empirical fitting/training replication remains unfinished.
Those gaps must not be described as another Lorenz/XLA failure. Figure 4 right
has 4000 completed estimates; Figure 6, source-M16/Table2 and KDM learning remain
open. iAPF remains incomplete and deferred under the current master.

## Evidence, decisions and next action

The new [geometry replay128](artifacts/controlled-smc-kdm-replication-20260928/attempt128_c1_lorenz_root_cause_audit/summary.json)
passes 33 explicit checks, including selected high-precision integrals.
All 404 separately recorded fixed-time coefficient replays also pass. The
[target decomposition129](artifacts/controlled-smc-kdm-replication-20260928/attempt129_c1_lorenz_root_cause_attribution/summary.json)
passes 50 checks. Their adjacent manifests contain commands, source/input hashes,
environment, device choice, Git base and timing. The first launcher attempt127
failed before numerical work because a legacy CPU routing branch selected an
absent generic script; an explicit dispatch check now prevents that error.
Total charged process time for attempts 127–129 is 6.850606 CPU seconds;
GPU charge is zero. The [terminal evidence review](artifacts/controlled-smc-kdm-replication-20260928/root-cause-terminal-review-20261002-01.json)
passes 30 checks of saved inputs, source snapshots, numerical-check flags,
syntax, links and the current checkpoint. The live replay script now also
includes every per-time replay flag in its overall PASS decision; saved
attempts and their source snapshots remain unchanged.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept the frozen-input attribution | Reconstruction and algebra checks pass | No failed mandatory check | Upstream interventions not tested | Preserve diagnostics and localize114 | A repaired filter or global causal proof |
| Reject a covariance floor as sufficient mean protection | It leaves the dangerous direction and mean essentially unchanged | Forward nonfiniteness in 121/126 | Other explicitly defined safeguards | Evaluate the complete natural-parameter update | Rejection of every covariance protection for every purpose |
| Keep full numerical admission blocked |114 full recursion still disagrees | Parity and forward validity remain unmet | First divergence and independent-reference accuracy | Same-input localization before new long runs | Default, HMC or Figure 6 completion |

| Inference status | Finding |
|---|---|
| Hard veto screen | Negative precision, nonfinite forward states and unresolved full-recursion mismatch are supported on the specified cases |
| Statistically supported ranking | None; this is deterministic forensic work |
| Descriptive-only differences | Geometry and residual magnitudes on saved failing cases |
| Default-readiness | No candidate admitted |
| Next evidence needed | Numerical localization, calibrated joint-update protection, independent full-recursion validation and untouched conditional comparisons |

Post-run skeptical review: the strongest alternative is that upstream local
projections, rather than an intrinsic limitation of every quadratic controlled
policy, produced the problematic future targets. Our explanation allows that
alternative. A frozen-cloud counterfactual and an independently propagated
recursion can distinguish it. Low residuals, well-conditioned regression and
accurate fixed-input arithmetic already rule out several simpler explanations
at time 0. The weakest remaining evidence is attribution of the upstream
feedback and any claim of robustness across fresh trajectories.
