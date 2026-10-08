# SIR proposal failure diagnosis

Active question: why do both existing LEDH configurations miss the SIR
likelihood and score despite agreement between the TT and bootstrap references?
Checkout verified: `sqmc-development`, clean at `209223fdd` before this work.
This is a bounded diagnostic of the existing computation, not new tuning or a
change to the canonical filter.

## Evidence contract and research intent

Run the actual analytical-score endpoint with its saved SIR T=10 data, N=1008,
FP64, GPU/XLA, TF32 disabled, existing guarded-pairwise controls and marginal
weights. Use the first original design seed, 261006101. At every time step,
compare its likelihood increment with the exact Gaussian predictive integral
conditional on the SAME incoming empirical cloud. SIR has Q=I, linear H selecting
the nine infectious coordinates, and R=100 I, so this integral is available
without the external TT or bootstrap reference. Recover each incoming cloud
from the actual initial states and preceding reset trace.

Construct the cheap diagnostic comparators explicitly: (1) transition samples
with ordinary observation weights, to check whether the flow creates rather
than solves a sampling problem; (2) exact conditional Gaussian proposal per
ancestor, whose weights equal each ancestor's predictive observation density;
(3) exact mixture predictive integration, removing transition sampling error.
Compare conditionally at each of the first ten observations. These are local
diagnostics, not alternative full-filter rankings.

Primary diagnostic: actual log increment minus exact conditional log integral.
A large discrepancy with a healthy bootstrap comparator localizes proposal
integration failure on that same cloud. Record ESS, displacement, UKF covariance,
conditional-posterior/proposal separation, and reset next-step error as
explanation. Numerical invalidity, stale/mismatched observations, failure to
reproduce the saved value, or failure of affine-flow reconstruction are
continuation vetoes requiring localized repair. A bad proposal is a repair
trigger, not a reason to abandon the research direction.

No claim of complete SIR repair, population score accuracy, default readiness,
or statistical superiority follows. The diagnostic is descriptive and
mechanistic. Existing same-scalar derivative parity does not certify accuracy
relative to the model likelihood. Report uncertainty and the limited seed scope.

## Assumption audit and skeptical review

The saved controls and seed are replay baselines, not justified optimal settings.
The incoming empirical cloud is conditioned on explicitly; its exact integral
is not the full model likelihood. This avoids treating a local check as an
oracle. The linear observation and Gaussian transition are verified in the
active SIR adapter; no observation linearization approximation is invoked.
The flow uses accumulated UKF covariance while the sampled transition has Q=I;
this can be a legitimate importance proposal, so covariance mismatch alone
does not prove incorrect weights. Direct integration error and overlap must be
measured. Existing reset diagnostics condition on a possibly already damaged
cloud and cannot exclude earlier loss. Early/late step separation avoids pooled
diagnostics hiding initial collapse. Small finite differences already establish
same-program derivative parity, not statistical score accuracy.

Pre-mortem: a replay could differ because CPU/GPU data or random generation
differs. Load the saved observations verbatim, record their hash and controls,
and compare the directional value with the saved design row. Reconstruct any
proposal using the SAME shared flow helper and check child parity. No filter
equations, moment caps, or default settings are changed. Audit passes for this
limited localization question; no ranking or tuning promotion is attempted.

## Execution and checkpoint

Budget: at most 25 minutes elapsed and three localized attempts, no package or
environment changes. Memory growth must be verified before GPU initialization.
Use the approved conda Python and escalated GPU access. Output roots are unique
`docs/plans/artifacts/ledh-sir-proposal-diagnosis-20261007-01/attempt-NN/`.
The diagnostic script, run manifest, full logs, JSON per-step results, and result
note preserve commands, environment, source hash, seeds, wall time, and findings.

Attempt 01 completed in 35.12 seconds. The saved value and first score coordinate
replayed exactly; affine child reconstruction also matched exactly. Conditional
flow integration errors sum to -279.298 log units, compared with -0.0641 for
bootstrap on the same incoming clouds. At step 4, the actual increment is
-178.233 versus exact -36.989; the UKF infectious variance averages 37.820,
while the conditional transition variance is 1. This localizes a proposal
failure, without proving the cause solely from covariance disparity.

Next action: a diagnostic shadow using the SAME flow and mixture-weight helpers
with covariance Q instead of the UKF covariance, on the SAME incoming clouds,
particles, observations and eight substeps. This is a numerics-altering proposal
hypothesis, not a canonical runtime change. Its principled reference is the
exact Gaussian conditional posterior of each ancestor, not a fitted likelihood
target. Test exact-integral error, finite validity, affine parity and ESS; do
not propagate this shadow cloud or infer T=50 repair. Repeat the saved design
and, if valid, one additional original seed within the original three-attempt
budget. This controlled substitution separates covariance choice from flow-step
count, mixture-density code, reset controls, and target/model changes. No new
scope tuning or default promotion is allowed. Reviewed locally against these
limits before execution.
