# GenUT Score-Variance Repair Validation Result

Date: 2026-07-31
Status: `DIAGNOSTIC_COMPLETE_CENTRAL_MECHANISM_SUPPORTED_REPAIR_NOT_PROMOTED`

## Verdict

The corrected mathematical analysis is valid for its central finite-program
claim: recursively propagated state/reset Jacobians can amplify score noise,
and this mechanism is active in the Austria SIR route. It is not an asymptotic
Lyapunov theorem and it does not establish a score-variance growth rate in the
horizon.

The pairwise correction descriptively damped the observed Austria mode, but it
did not remove it: mean finite-time directional growth fell from `0.272372` to
`0.239384` and remained positive for every tested probe. Score SD also fell in
all three coordinates. With three particle seeds, these are descriptive
differences only; they do not establish lower bias, statistical superiority,
or promotion readiness.

The fixed Gaussian-target LGSSM arm did not uniformly lower score SD relative
to empirical targets. Therefore noisy empirical higher-moment targets are not
the sole cause of the observed score variance. The contractive LGSSM control
and expansive Austria results support Jacobian amplification as the more
important mechanism in this diagnostic.

## Evidence Contract Outcome

| Item | Outcome |
|---|---|
| Scientific question | Answered for finite-time mechanism discrimination, not asymptotic scaling |
| Comparator | Austria diagonal versus pairwise; LGSSM empirical versus fixed Gaussian targets |
| Primary criterion | Passed: every route finite and exact whitening identities passed |
| Promotion vetoes | None fired for artifact validity; positive Austria growth blocks interpreting the pairwise arm as a complete stability repair |
| Explanatory diagnostics | Finite-time directional growth and three-seed score SD |
| Nonclaims | No `Var(score)=O(T)` theorem, exact nonlinear score oracle, bias reduction, method ranking, HMC readiness, or default promotion |
| Terminal artifact | `docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260731/attempt08/` |

## Detailed Results

The diagnostic used TensorFlow 2.19.1 on an NVIDIA RTX 4080 SUPER, GPU/XLA,
`float32`, TF32 enabled, verified memory growth, `N=1008`, eight probe columns,
and three particle seeds per arm. The terminal wall time was `771.979 s`.

| Arm | Mean growth | Probe range | Score SD |
|---|---:|---:|---:|
| LGSSM empirical targets | `-0.825348` | `[-0.884597, -0.788503]` | `[0.325562, 0.272172, 0.247238, 1.377973, 1.018910]` |
| LGSSM fixed Gaussian targets | `-0.829765` | `[-0.974349, -0.789132]` | `[0.433168, 0.326618, 0.211918, 1.431570, 1.038902]` |
| Austria diagonal-only | `0.272372` | `[0.226388, 0.329196]` | `[74.745858, 27.099027, 41.586037]` |
| Austria pairwise | `0.239384` | `[0.183937, 0.295332]` | `[35.479458, 5.239076, 3.647641]` |

For Austria, the observed score-SD reductions were approximately `52.5%`,
`80.7%`, and `91.2%`. The mean growth reduction was approximately `12.1%`.
These percentages summarize the three seeds; they have no inferential interval
in this campaign.

## Whitening Check

Full whitening gave maximum mean error below `4.59e-17` and maximum covariance
error `8.88e-16`. Across the three whitened Gaussian draws, mean marginal
kurtosis was `2.926`, `3.063`, and `2.994`; mean studentized off-diagonal
co-kurtosis was `1.003`, `0.992`, and `0.995`.

This verifies the exact first/second-moment identities and supports the note's
corrected higher-moment distinction. The raw `m22` statistic has first-order SD
`sqrt(8/N)`, while studentized co-kurtosis has first-order SD `sqrt(4/N)`.
These fluctuation formulas are heuristic distributional diagnostics, not an
exact finite-sample normality claim.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain corrected mechanism analysis | Passed | No artifact-validity veto | D1 is finite-time and directional | Use D1 as an early repair diagnostic | No asymptotic exponent or variance-rate theorem |
| Do not promote pairwise repair | Descriptively favorable only | Austria remains expansive | Three seeds; changed finite scalar; no bias oracle | Retune with non-positive D1 as a veto, then validate value and score against an eligible reference | No superiority, lower bias, or HMC readiness |
| Do not promote fixed Gaussian targets | Mixed score-SD changes | None | LGSSM-only isolation | Keep as a diagnostic comparator | No universal target-noise repair |
| Accept whitening derivation | Exact identities passed | None | Higher moments fluctuate at finite `N` | Preserve raw-versus-studentized distinction | No downstream variance guarantee |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for execution validity; positive Austria growth vetoes a claim that pairwise correction eliminated the unstable mode |
| Statistically supported ranking | None; three seeds are insufficient and no predeclared interval supports a ranking |
| Descriptive-only differences | Pairwise has lower observed Austria growth and score SD; Gaussian targets have mixed LGSSM score-SD changes |
| Default-readiness | Not established |
| Next evidence needed | Scope-specific retuning with D1 veto, untouched multi-seed validation, value/reference agreement, score bias/MCSE evidence, and downstream HMC validation |

## Attempt And Repair History

Attempts 01--04 repaired, respectively, a TensorFlow API typo, whitening solve
orientation, probe tangent packing, and premature output-directory creation.
Attempt 05 completed D1 and whitening. Attempts 06--07 repaired tangent-width
mismatches introduced by the Gaussian-target arm. Attempt 08 completed the full
predeclared scope. Earlier outputs were not overwritten.

## Engineering Verification

- GPU artifact: `hard_valid=true`; exact whitening identity passed.
- CPU-only regression, with `CUDA_VISIBLE_DEVICES=-1`: 37 tests passed in
  `15.70 s`; two TensorFlow Probability `distutils` deprecation warnings only.
- LaTeX: `latexmk -pdf` succeeded, producing an 11-page PDF. Final-log cross
  references resolved; layout has non-fatal overfull/underfull box warnings.
- Static validation: Python compilation and `git diff --check` passed.
- Claude read-only review was unavailable because the environment privacy
  boundary did not permit sending these private workspace documents to the
  external service. No Claude verdict is claimed; local derivation audit,
  mathematical-tool diagnostics, regression tests, and the GPU experiment are
  the available review evidence.

The original `run_manifest.json` is incomplete under the serious-run metadata
policy. `run_manifest_supplement.json` adds environment, device, controls,
seeds, source hashes, history, and paths without rewriting original evidence.
The exact shell command was not captured contemporaneously; its invocation is
explicitly labeled as reconstructed.

## Post-Run Red Team

The strongest alternative explanation is target/route specificity rather than
a universal Jacobian-instability law: Austria and LGSSM differ in model,
dimension, horizon, and correction geometry. The diagnostic isolates a
mechanism but not every causal contribution. Pairwise also changes the finite
scalar, so lower dispersion could accompany worse score bias.

Evidence that would overturn the working conclusion would be an eligible
independent score/reference study showing no relationship between realized
growth and score error, or a probe-basis study showing the positive Austria
growth was a non-reproducible directional artifact. The weakest evidence is
the three-seed comparison and the lack of a nonlinear score oracle with MCSE.

PDS/Fisher smoothing and custom-force Metropolis-corrected HMC remain future
work. This campaign did not validate either one. PDS on OT/reset routes must
not be called an exact oracle without a derivation for the executed target and
branch semantics.
