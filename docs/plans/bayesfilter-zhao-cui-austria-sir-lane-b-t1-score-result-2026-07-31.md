# Zhao-Cui Austria SIR Lane-B T1 Score Result

Date: 2026-07-31

Status: `BLOCK_T1_TOTAL_SCORE_NOT_IDENTIFIED`

## Verdict

The admitted T1 value remains correct for the declared finite program:
`-31.1290512231882`. The new implementation correctly computes the manual
derivative of any declared external-theta child, and the analytical Austria T1
complete-data score matches diagnostic autodiff to `4.3e-14`. However, the
current frozen-parent tangent construction does **not** identify a correct
Austria observed-data total score.

The terminal untouched score claim failed. No HMC or T2 score work is opened.

## What Passed

| Check | Result |
|---|---|
| Parent origin cores/value | exact; value residual `0` |
| Analytical local parameter score | diagnostic tape residual `4.3e-14` |
| Manual child value/point/prefix derivatives | focused diagnostic tape parity passed |
| Parent immutability and compact storage | passed |
| Trained child save/reload/tamper rejection | passed |
| GPU/XLA training | passed on RTX 4080 SUPER with verified memory growth |
| Frozen child eager/XLA score tie-out | `1.9984e-13` |
| Peak allocation | `628,096,256` bytes at the largest claim, below 6 GiB |
| Focused regression before closeout | 17 tests passed |

## Terminal Untouched Evidence

Selected child identity:
`f3a353f4bdafc2fd33ea38f90a4863e69ecd710bb5fba02201d2c16bfae6a564`.

Untouched artifact:
`docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-score-20260731/attempt-06-untouched-claim/result.json`.

SHA-256:
`0f8e95c586bbabc376a847dfa182843273f132613d3dc3d5882082db8e3ab582`.

| Coordinate | Child score | Untouched Fisher score | MCSE | Absolute difference | Fixed tolerance | Gate |
|---|---:|---:|---:|---:|---:|---|
| log kappa scale | `-5.89273658` | `-5.55497986` | `0.30134211` | `0.33775672` | `0.90403634` | pass |
| log nu scale | `2.03763892` | `2.09450799` | `0.07955694` | `0.05686907` | `0.23868082` | pass |
| log observation-noise scale | `-4.89504405` | `-4.88437960` | `0.00338804` | `0.01066445` | `0.01017411` | **fail** |

The untouched effective sample size was `56,038.43 / 65,536`. The comparator
was informative under every predeclared MCSE ceiling. The failure is therefore
not explained by an underpowered claim or a resource failure.

The preceding 32,768-row calibration passed its fixed interval. Its artifact
used the terminal pass label before the runner distinguished calibration from
claim status. That label is historically overbroad; the artifact is classified
only as `PASS_T1_SCORE_CALIBRATION` and was not admission evidence. It is not
rewritten.

## Why The Candidate Is Not Correct

The claimed target was the total derivative of the declared parameter child
value. The implementation actually computed that derivative exactly for the
chosen tangent cores. The remaining problem is that the admitted fixed parent
specifies only the value slice at `theta=0`; it does not uniquely specify how
that slice changes with theta.

An amplitude-gauge tangent can preserve the origin value and set the
normalizer derivative to an arbitrary three-vector. Training-cloud Fisher
calibration exploited that valid algebraic freedom, but it did not identify the
physical derivative of the finite TT approximation. The pointwise tangent fit
also remained weak on the kappa and nu coordinates: normalized validation RMS
was approximately `0.962` and `0.846`. Therefore another gauge calibration on
a fresh cloud could make the T1 scalar pass while still leaving the carried
prefix derivative wrong. Such a result would be wrong relative to the claimed
T2 total-score target.

## Failure Classification

| Ledger | Verdict |
|---|---|
| Engineering correctness | Passed for local score hooks, manual child algebra, serialization, GPU/XLA, and memory. |
| Numerical validity | Passed for finite contractions and eager/XLA parity. |
| Candidate score admission | Failed untouched observation-scale coordinate. |
| Scientific interpretation | The fixed value slice does not identify its theta derivative; current score is unsupported as the Austria Zhao-Cui total score. |
| Direction rejection | No. This rejects the current tangent/gauge candidate, not the fixed T1/T2 values or all parameter-conditioned Zhao-Cui constructions. |

## Decision Table

| Field | Decision |
|---|---|
| Decision | Preserve T1/T2 value admission; block T1/T2 score admission and HMC. |
| Primary criterion | Failed one of three untouched coordinates. |
| Veto diagnostics | Identity, analytical-hook, XLA, finite, MCSE, and memory vetoes passed. |
| Main uncertainty | How to define and fit a unique parameter-conditioned TT derivative consistent with the sequential target, not merely the origin value slice. |
| Next justified action | New reviewed parameter-conditioned training objective that identifies tangent shape and prefix derivative, followed by fresh calibration/untouched data. |
| Not concluded | No correct total score, T2 score, T5/T10/T20, HMC, posterior, production, or superiority claim. |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Untouched score gate failed; candidate rejected. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Pilot losses, correlations, RMS, and arm comparisons. |
| Default readiness | Not established. |
| Next evidence needed | A derivative-identifying parameter-conditioned target fit with independent prefix and normalizer score validation. |

## Reopen Condition

Reopen score execution only under a refreshed plan that removes the gauge
non-identifiability. A viable route must train or otherwise define the
parameter-conditioned density away from zero, validate the local and retained
prefix score fields independently, and then pass a new untouched normalizer
score gate. It may not tune on attempt 06, merely recalibrate an arbitrary
gauge, relax the tolerance, or run HMC.
