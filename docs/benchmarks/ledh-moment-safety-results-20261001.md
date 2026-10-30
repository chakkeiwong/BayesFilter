# LEDH moment-fit safety repair — 2026-10-01

The guarded fit passed the declared safety checks on the tested fixtures, but
it lost in combined moment error to the original coordinate cap in 26 of 112
GPU cases. It also lost to the smaller-step comparator in three cases (within
those 26). This vetoes promotion as the more accurate fitter or a new default.
The implemented feature remains optional numerical protection. Pairwise skewness
and co-kurtosis fitting remain present and are covered by multidimensional tests.

Plan and executing-agent skeptical review:
[ledh-moment-safety-repair-20261001.md](../plans/ledh-moment-safety-repair-20261001.md).
Evidence root: `../plans/artifacts/ledh-moment-safety-20261001/`.
Baseline commit: `688878c8f`; branch: `sqmc-development`.

## Problem and repair

The fixed richer residual design prevents the exact scalar binary configuration
from forcing both marginal correction directions to zero. The identity-core
coordinate cap preserves moderate coordinate magnitudes. Neither change alone
controls an unstable fit: displacement caps act before whitening, whitening can
amplify a nearly singular covariance, and a final cap cannot repair an earlier
invalid solve. A finite fourth moment can also be badly wrong.

The shared optional `moment_safety=True` path keeps the LM damping, marginal row
cap, pairwise row cap and final coordinate cap. It tests at most eight step sizes,
checks covariance before Cholesky, and accepts a step only when values/tangents
are finite and the declared scaled moment loss does not increase beyond
roundoff after whitening. It compares the complete capped/restored result to
an identically capped/restored no-fit cloud. A rejected step carries the prior
value and tangent; a rejected complete fit returns that protected baseline and
its total tangent. Diagnostics retain rejections, trial counts, step sizes and
both final losses. An invalid protected baseline fails closed.

The objective includes marginal third/fourth moments and configured pairwise
entries, each divided by `max(1, abs(target))`. This is an explicitly defined
joint moment loss. It is not a guarantee that every kurtosis error decreases,
that weighted-particle targets equal true posterior moments, or that the final
state coordinates satisfy the pre-whitening cap. Full projected-cumulant
correction is explicitly unsupported with this guard.

The analytical derivative differentiates the selected finite branch, including
source particles, source weights, moment targets, solves, caps, relative ridges
and whitening. Discrete acceptance boundaries can be nonsmooth; HMC admission
and globally differentiable scores are not established.

## Execution and provenance

Attempt 01 stopped before numerical work because project imports initialized
TensorFlow before the driver configured memory growth. The standalone trusted
GPU probe passed. The driver now configures and verifies memory growth before
project imports. Attempt 02 stopped during tracing because a diagnostic scalar
was inferred as FP32 against an FP64 target; it was explicitly cast. Both failure
logs are preserved; neither is evidence against the mathematical candidate.

Attempt 03 had a CPU reference phase and a GPU phase. The exact commands,
environment, source snapshots/checksums, input tensors, fixed seeds 197/811,
device/growth policy, XLA/TF32 settings and wall times are in each manifest.
The driver is `run_ledh_moment_safety_20261001.py`. GPU execution used trusted
access on the RTX 5080 with memory growth verified for both visible GPUs.
CPU reference runs explicitly hid GPUs. Every repeated kernel had a stable
input signature and XLA enabled. Distinct dimensions, dtypes and comparator
settings created bounded distinct kernels; framework retracing warnings did
not indicate an unbounded polymorphic graph.

Fresh cases used dimensions 1/2/3/10, `N=48*d`, two fixed fixture seeds, and seven
regimes: healthy, skewed, mixture, source outlier, concentrated weights, nearly
collinear source, and repeated-axis residuals. The five arms were protected
no-fit, original cap, identity-core cap, smaller-step identity-core fit, and
guarded identity-core fit. Controls were fixed comparison hypotheses, not
scope-tuned settings for a filtering accuracy claim. No historical pre-August
21 result was used as a numerical comparator. The original historical explosive
input was not replayed.

| Execution | Cases / arm evaluations | Safety check | Recorded wall time |
|---|---:|---|---:|
| GPU/XLA, FP64 and FP32/TF32 | 112 / 560 | 112 passed | 226.78 s |
| CPU/XLA FP64 reference, 2D | 14 / 70 | 14 passed | 11.81 s |
| Final GPU/XLA mechanics, FP64 and FP32/TF32 | 8 cases | 8 passed | 36.73 s |

The broad comparison preceded one further fail-closed refinement: a skeptical
edge-case audit found that the Loewner roundoff allowance alone could admit a
singular trial from an input close to that allowance. The final helper also
requires the trial's minimum eigenvalue to exceed its own relative roundoff
margin. The constructed regression passes. The broad comparison and final
helper therefore have distinct, preserved source snapshots; the broad campaign
was not silently relabeled as a run of the final source. Focused final GPU checks
covered healthy no-fire behavior, active tail capping, and analytical tangents.

The final current regression selection passed **55 tests** in 37.47 seconds:
`test_higher_moment_contract_e.py`, `test_sqmc_reset_repair.py`, and
`test_moment_safety_tf.py`. It includes actual canonical filtering endpoint
wiring, pairwise trace fields, shared/batched parity, finite-difference total
scores/tangents, collapsing/nonfinite displacements, the singular-trial edge
case, and forced final-fallback value/tangent selection. The forced fallback
fixture isolates that branch mechanically; no real campaign case triggered
final fallback. A broader initial selection could not collect the stale
`test_genut_shape_lm_tf.py`, which imports absent `cubature_genut_batch_tf`.
That unrelated collection problem remains recorded and was not called a pass.

## Numerical findings

The guard rejected at least one whole proposed step in 23 GPU cases. Another
33 cases used a reduced step with no wholly rejected step. All 112 guarded
outputs were valid and finite and no guarded final loss exceeded its protected
baseline beyond the declared allowance. No final complete-fit fallback fired
in that campaign. CPU/GPU FP64 loss differences on matching 2D cases were at
most `4.25e-13` across all five arms.

| Regime | GPU safety passes | Cases losing to a simple comparator |
|---|---:|---:|
| Healthy | 16/16 | 0 |
| Skewed | 16/16 | 0 |
| Mixture | 16/16 | 10 |
| Source outlier | 16/16 | 0 |
| Concentrated weights | 16/16 | 8 |
| Nearly collinear | 16/16 | 0 |
| Repeated-axis residual | 16/16 | 8 |

For example, in FP64 scalar mixture seed 197, the combined losses were 0.78172
for protected no-fit, 0.38085 for the guarded fit, and 0.002750 for the original
cap. The original cap can help a low-kurtosis mixture even though it destroyed
useful kurtosis in the earlier KSC reset. This is a conditional result, not a
reason to replace one universal cap with another.

In final GPU checks, healthy particle values matched exactly and the maximum
healthy tangent difference was `4.34e-19`. Literal bitwise equality of every GPU
tangent was therefore not achieved; agreement is at floating-point roundoff.
The final FP64 finite-difference tangent error was at most `1.18e-10`, with
matching acceptance branches. The scalar `N=1008` and 2D outlier-input checks
activated the identity-core tail cap and remained finite with the declared
complete-map loss bound. Such a bound can still retain a poor no-fit baseline.

The broad campaign's largest relative covariance residual was `1.10e-3` for
FP32/TF32, versus `1.32e-15` for FP64. Exact covariance restoration is an
un-ridged algebraic statement, not an unconditional floating-point claim.
These observed residuals do not by themselves admit the FP32 route for another
scientific scope or establish an acceptable likelihood/score error there.

## Decision and inference

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain optional guard | Tested loss/validity conditions passed | Final focused invariants and derivative checks passed; GPU tangent equality only to roundoff | Unseen pathological inputs and acceptance boundaries | Use explicit option in fresh scope-specific calibration if requested | Universal stability or global smoothness |
| Reject accuracy/default promotion | Joint safety loss is not a likelihood criterion | Heuristic dominance failed in 26/112 cases | Moment trade-offs and teacher error | Keep scope-specific cap/fit choices; do not select on these stress cases | Best fitter, likelihood improvement, HMC readiness |
| Retain pairwise capability | Real 2D/3D call chain and derivatives checked; 10D broad cases executed | No missing wiring or nonfinite guarded result | Full-tensor moments are outside this path | Evaluate mixed-moment accuracy in the intended model scope | Complete tensor matching |

| Inference status | Finding |
|---|---|
| Hard veto screen | Final focused validity/derivative checks passed; literal bitwise GPU tangent equality was not met (maximum difference 4.34e-19); accuracy-promotion veto from simple comparators |
| Statistically supported ranking | None; fixed fixtures and two seeds do not support population ranking |
| Descriptive-only differences | Losses, rejection frequencies, covariance errors and timings |
| Default readiness | Not established; all existing defaults remain unchanged |
| Next evidence needed | Fresh scope-specific calibration/holdout filtering comparisons, original failure replay if available, and boundary-sensitive score assessment before any HMC claim |

The claimed numerical target is a finite guarded moment fit. The quantity
actually computed is that fit on its selected branch; analytic/finite-difference
checks support local equality. It is different from the true filtering law,
exact full-tensor moment matching, and a globally smooth target. Those stronger
claims remain unevaluated.

Strongest alternative explanation: the guard looks successful because the
protected baseline already limits the comparison, while that baseline or its
weighted empirical teacher can be inaccurate. A counterexample violating the
reported finite/conditioning/loss checks would overturn the safety conclusion.
A poor but finite reset does not violate those checks and must be exposed by
scope-specific reference accuracy. The weakest evidence is extrapolation from
these synthetic fixtures to the unreplayed historical instability and long
nonlinear filtering trajectories.

## Documentation and budget

The monograph chapter now derives binary-design degeneracy, the finite normal
quantile design, marginal and pairwise corrections, both coordinate maps,
whitening amplification, the guarded algorithm and its branch-conditioned
total derivative. The mirror is synchronized, including the existing scaled-LM
material needed by the new derivation. The full PDF, build logs and rendered
inspection pages are under `monograph-build/` in the evidence root. The final
full monograph has 602 pages, with zero undefined-reference/citation or duplicate-
label warnings and no rerun warning. PDF pages 209–216 (printed pages 191–198)
were visually inspected after the final path-layout repair; the new section has
no overfull body warning. Pre-existing chapter-header/layout warnings elsewhere
remain. The final PDF/source hashes and inspection record are in
`monograph-build/verification.json`.

Measured successful GPU numerical phases total 263.52 s. Reserving 60 s for
probes and failed startup/tracing yields a conservative charge below 324 of
1800 GPU seconds. Recorded CPU tests/reference work totals about 159 s; a
conservative charge of 200 leaves 1600 of 1800 CPU seconds. Three campaign
attempts were used; the final edge checks are focused implementation
regressions. No HMC, training, environment mutation or likelihood tuning ran.
Review was by the executing agent, not independent. Human readability review
remains pending; compilation and rendered inspection do not replace it.
