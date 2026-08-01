# GenUT SIR Feasibility Diagnostic Plan

Date: 2026-07-22

## Research intent

Question: does the existing positive Gaussian-GenUT/Contract-E candidate core
execute a nonlinear SIR transition with a recursive parameter score in FP32,
TF32, and XLA without non-finite values or a score-path mismatch?

Candidate: `finite_value_score` with a positive replicated Gaussian GenUT
design.  This is a feasibility diagnostic, not a leaderboard or default
promotion run.

Target: a two-compartment reduced continuous SIR model with state `(S,I)` and
parameters `(log_kappa_scale, log_nu_scale, log_obs_noise_scale)`.  The base
values are `kappa=0.1`, `nu=1.0`, process covariance `diag(0.25,0.16)`,
observation variance `0.16`, initial mean `(0.3,0.2)`, and initial covariance
`diag(0.25,0.16)`.  Each candidate evaluation uses an initial cloud followed
by `T` deterministic RK4 transitions of length `0.02` (four internal steps),
then one Gaussian infectious-coordinate observation at each transition.  A
nonnegative susceptible projection is used only when evaluating the physical
state; the filtering state remains preclip.  This is an explicit diagnostic
target and is not the clipped Austria source measure.

## Evidence contract

- Primary pass screen: every requested `T` and seed returns finite value,
  finite recursive score, and finite reset residual diagnostics.
- Score diagnostic: the same finite program's recursive score is compared with
  a central finite-difference evaluation of its scalar value at representative
  points.  The FD check is explanatory only because FP32 cancellation makes a
  fixed small step unreliable; FD is not the runtime score.
- Engineering vetoes: XLA compilation failure, non-FP32 tensors, Python or
  NumPy numerical work in the compiled path, non-finite outputs, or reset
  residuals above `5e-4`.
- Explanatory diagnostics: per-horizon value/score, FD residuals, runtime,
  device, and particle count.
- Nonclaims: no exact observed-data likelihood claim, no exact score claim, no
  comparison to an oracle, no Zhao-Cui source-faithfulness claim, no clipping
  measure equivalence, no HMC/leaderboard/default readiness, and no claim
  about the canonical Austria SIR row.

## Scope and budget

Run `T in {2,5,10}`, eight independent DGP seeds per horizon, and `N=96`
particles.  The Gaussian GenUT design in dimension two has masses
`(1/3,1/6,1/6,1/6,1/6)` and is exactly representable at `N=96`.
Use FP32 tensors, TF32 enabled, GPU/XLA, and memory growth configured before
TensorFlow initialization.  Preserve all outputs under a fresh versioned
artifact directory.

## Skeptical audit

The canonical Austria SIR row is not a valid oracle for this test because the
repository documents a clipped-simulator/Gaussian-density measure mismatch.
The reduced target therefore tests only finite execution and internal score
consistency.  A finite result would establish candidate feasibility, not
scientific accuracy.  The first cheap diagnostic is a `T=2` run; any failure
stops the longer horizons.  Any implementation repair must preserve the target
equations, particle count, dtype, and budget and must be recorded in the result
artifact.

## Artifact

The executable harness and its JSON/Markdown outputs live under
`docs/benchmarks/artifacts/genut_sir_feasibility_20260722/attempt01/`.
