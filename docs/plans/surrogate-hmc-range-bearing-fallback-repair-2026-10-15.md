# Range-bearing warmup force-fallback diagnosis and bounded repair

The question is why the frozen range-bearing surrogate force failed its health
check during the first 250-step warmup chunk, and which localized repair
addresses that failure without changing the posterior. This is a repair within
the existing five-model surrogate-HMC campaign. Its remaining models continue.

## Evidence contract and skeptical audit

Baseline: the actual saved literature-prior run, N4032/K2016, FP32, TF32 off,
T20, frozen selected controls and independent fixed-trajectory endpoint. Do not
substitute earlier benchmark-prior results. The warmup archive is diagnostic
data; no retained posterior samples exist for this failed attempt.

First verify archive hashes and count flags by chain and transition. Trace the
consumer from the public sampler through the force adapter to the batched
filter validity checks. Then reproduce an affected parameter input and report
the first failing observation and named validity condition. Replay is a
debugging check, not scope tuning or convergence evidence.

Promotion for a repair requires reproducibility of the original failure,
restored intended invariant at those inputs, and no change at healthy inputs
beyond the declared arithmetic tolerance. Posterior or default promotion still
requires fresh scope-specific tuning where the scope changes and a full
adaptive sampler run. A finite endpoint, high acceptance or no divergences
cannot override a declared invalid force. Counts and location of fallbacks,
acceptance, coordinates and timing are explanatory diagnostics.

Continuation vetoes: corrupted archive, mismatch to the saved source/selection,
unrecoverable target ambiguity or exhausted budget. A failed candidate is a
repair trigger, not a campaign-wide veto. Do not silently relax the no-fallback
screen, clip parameters, change priors, use a partial score, or retune on these
warmup inputs as if they were independent validation.

Assumptions: FP32 and N4032 are owner choices, not guarantees of validity over
the posterior. The 250-transition chunk is inherited batching, not a
convergence assessment. The zero-force extension is an existing kernel policy;
its activation is the observed health failure, not yet its numerical cause.
Exact saved artifacts take precedence over suspected covariance, support or
gradient explanations.

Audit: this plan answers the failure question using the actual failed call
chain and preserves the independently defined endpoint. It does not change
running code or confuse a successful replay with sampler convergence. The
cheap alternative explanations are a genuinely nonfinite score, a finite but
failed filter invariant, prior/coordinate failure, and an aggregation defect.
No ranking of methods is attempted.

## Bounded execution

Output: docs/plans/artifacts/range-bearing-warmup-fallback-20261015-01.
Start with CPU-only TensorFlow archive inspection (GPU hidden before import).
Then at most three focused CPU graph replay checks, at most 10 minutes each,
recording any non-XLA diagnostic exception. No broad sweep or new HMC launch.
GPU replays, if necessary, must use trusted access and the existing cumulative
campaign budget, with the active worker reservation respected. Do not expand
the 86400-second GPU budget. Record actual commands, source, elapsed times and
results. Stop a replay on timeout and preserve its logs.

## Diagnosis and focused extension

The saved-input CPU replay reproduced the failure at observation 14. The
flow component of particle 436 has a whitened information eigenvalue 16.0156;
with eight left-endpoint Euler steps its first multiplier is -0.00097597.
Its covariance margin is 1.146e-7, below the FP32 four-state guard 4.768e-7.
The prior and Gaussian-guide components remain healthy. At the identical
ancestor input, 16 and 32 steps raise the margin to 0.03316 and 0.06447.

This identifies a discretization resonance, not a failed Sinkhorn balance or
covariance reset. FP64 permits the nearly singular map but does not remove
its near-zero multiplier. Two extra CPU-only full-filter replays (16 and 32
steps, at most ten minutes each) are justified to distinguish a local repair
from another failure elsewhere in the recurrence. Total bound is now five
CPU diagnostic runs, with no additional GPU allocation. Changing step counts
is a diagnostic candidate, not scope tuning or promotion. The original
healthy-point likelihood/score is not an accuracy oracle when the integration
rule changes; differences must be reported, not classified automatically as
harm. Promotion requires independent references and fresh scope tuning.

Skeptical extension audit: full recurrences are required because changing the
flow changes earlier particle clouds. The failed warmup points are regression
inputs, not validation data. Doubling a step count cannot prove global
stability: the left-endpoint singularity moves from information eigenvalue
16 to 32. A structural integration repair remains a separate candidate if
the bounded refinement does not cover the needed posterior region.

Completed: all five bounded diagnostic replays finished. The original FP32
failure is reproduced; mixed precision, FP32 16 steps and FP32 32 steps
complete both full recurrences. The detailed result and structural integration
candidate are in `artifacts/range-bearing-warmup-fallback-20261015-01/result.md`.
The next repair phase is fresh scope-specific tuning and GPU/XLA validation of
a fixed integration candidate before a new adaptive HMC attempt. This belongs
to the existing master and must respect its remaining allocation. The active
campaign's frozen runtime has not been changed; two saved inputs are not
enough to promote a repair.
