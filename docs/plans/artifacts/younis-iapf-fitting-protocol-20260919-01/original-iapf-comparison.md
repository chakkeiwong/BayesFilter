# Which difficulties belong to the original iAPF?

2026-09-19. The published method shares the distinction between likelihood
variance and smoothing-expectation variance, and its example approximation
scheme has finite-cloud and Gaussian-family limitations. It does not use our
old automatically passing stopping configuration. The specific dataset-1900
score failure has not been reproduced in verified original-author code.

## Source identity

The primary source is Guarniero, Johansen and Lee, Sections 3--5, especially
Algorithm 4, the paragraph after equation (14), and equations (15)--(16).
The [local paper](../../../../.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf)
was inspected directly. The public executable reference used in our earlier
comparison is [Sempreteamo/iAPF](https://github.com/Sempreteamo/iAPF), commit
`a88114395f6c11075fedc653db9480791b43391a`. Its authorship has not been verified
as that of the original paper's authors. The distinction must remain explicit.

The current local `iapf.R` SHA-256 is
`bf6f283b3eac0c3ef0a390af6be522af04026d47dcd6b1f52e606db6c5719851`,
matching the executed reference-comparison record in `attempt02/results.json`.
This audit inspected its settings (lines 10--12), likelihood CV (95--97),
fitting objective and optimizer (217--235), and outer controller (321 onward).
The prior executable diagnostic used the actual parsed R functions with its
recorded primitive substitutions; it did not reproduce the entire outer
experiment. See the [comparison result](../younis-iapf-kdm-reference-comparison-20260918-01/result.md).

A bounded GitHub repository search for `iapf`, `"iterated auxiliary"`, and
`guarniero` did not establish a new original-author implementation. The latter
two queries returned zero repositories. This is not proof that author code
does not exist. No stronger provenance claim is made.

## Comparison

| Question | Published method / reference evidence | Conclusion |
|---|---|---|
| Is the old CV screen automatically satisfied? | Paper Section 5.1 uses k=5 or 3 and tau=.5 or 1; public R source uses k=5, tau=.5. Local old configuration used k=1, tau=100. | The proved vacuous-threshold defect is local to that configuration, not the paper's reported settings. The published stopping rule remains an empirical likelihood-stability heuristic. |
| Does likelihood-optimal twisting guarantee a precise score? | Section 4 after (14) explicitly distinguishes likelihood optimality from optimality for general smoothing expectations. | No; this limitation applies to the original method as well. |
| Can the fitted twist be inaccurate? | Section 5.1 fits a diagonal Gaussian on a particle cloud and adds a positive floor. The general Algorithm 3 permits other approximations. | Yes in principle; finite-cloud error and family mismatch are shared risks of that example fitting scheme, not a demonstrated failure of every iAPF implementation. |
| Did the public R fitter ever report convergence prematurely? | Saved deterministic Gaussian fixture misses its accuracy criterion; tightening only `optim`'s `factr` resolves that discrepancy. | Yes on that fixture, but this is not verified original-author code or dataset 1900. |
| Does original-author code lose to UKF on dataset 1900? | No matched original-author-code score experiment exists in the inspected evidence. | Not checked. Even our current true-risk ordering against UKF is unresolved. |

## A mathematical weakness in the published fitting objective

Paper (15) minimizes the unnormalized squared discrepancy between Gaussian
values p on a finite cloud and a scaled positive target vector y. Profiling
the scale gives

\[
L(p)=p^\mathsf{T}p-
\frac{(p^\mathsf{T}y)^2}{y^\mathsf{T}y},\qquad
R(p)=\frac{L(p)}{p^\mathsf{T}p}.
\]

For a fixed finite cloud and Gaussian covariance s^2 I, as s tends to
infinity the Gaussian evaluations become proportional to the all-ones vector
with an amplitude tending to zero. Consequently L tends to zero, while

\[
R\longrightarrow
1-\frac{(\sum_i y_i)^2}{N\sum_i y_i^2}>0
\]

for a nonconstant target. Thus an arbitrarily small published fitting objective
does not by itself establish a good relative shape. This is a proof about
unbounded Gaussian parameters in the stated objective, not evidence that the
authors' optimizer took that sequence; implementation bounds or other solver
choices could prevent it. The paper's reported numerical successes remain
separate empirical evidence. Our current relative-shape objective removes
this amplitude mechanism but still lacks a score-variance guarantee.

The public R reference uses a different objective,
F = (y^T y) R/(1-R). It removes candidate-amplitude dependence but retains
target-amplitude dependence. On the saved known-Gaussian fixture, default
`optim` reported convergence with first variance .9959796291, versus the
exact value 1 and required absolute tolerance .002. With only `factr=1`, the
variance was 1.0000005315. This is a checked premature-stopping example; it
does not identify the cause of our nonlinear dataset's proposal mismatch.

## Status and next action

No new fits, filter evaluations or GPU launches were run. This is a source and
saved-result audit, with a self-reviewed mathematical derivation. It supplies
no new superiority, default-readiness or HMC evidence. Further direct claims
about the authors' implementation require authenticated source and a matched
test. The immediate local diagnosis remains the archived 1900 fitting cloud,
optimizer geometry and independent predictive target, as described in the
[mathematical explanation](mathematical-explanation.md).
