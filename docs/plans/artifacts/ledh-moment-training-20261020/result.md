# Simulated moment training: implemented; heldout comparison invalid

20 October 2026. Branch: `ledh-refactor-with-policy-fix`.
Plan: [simulated moment training](../../ledh-simulated-moment-training-20261020.md).

## Finding

Continuous guide-coefficient training is implemented and executed through the
shared LEDH program. Unlike the earlier finite-bank pilot, both the optimizer
and validation selector use simulated terminal first/second-moment loss, never
score labels. The eight initial runs and two predeclared smaller-step repairs
completed: 548 of 576 proposed updates were accepted, with one bound-hit event,
no invalid probe pairs and no invalid trials. Thirty-five focused tests pass.

The planned independent comparison did not complete. Unguided LEDH became
nonfinite on the seventeenth test dataset. Exact replay reproduced the failure
with unchanged runtime sources and identical regenerated inputs. This is a
genuine baseline-validity veto, not rounding error or a reporting-only failure.
Neither improved heldout moments nor improved scores is established. The
population simulated-label risk identity is not contradicted by this failure.

## What was trained

Scope: continuous SIR, state dimension 18, theta zero, horizon 5, 64 particles,
GPU/XLA FP64 with TF32 disabled as an explicit diagnostic exception. OT stays
at epsilon 1 and five Sinkhorn iterations, plus 32 terminal balancing updates.
No numerical default, canonical-admission policy or model-score method changed.

The two guide families are a quadratic controlled-SMC adapter and a positive
RBF, KDM-inspired adapter, not original neural KDM retraining. Each learns log
strength and susceptible/infected translations, either pooled or time-specific.
Loss is measured on the actual final reset cloud after UKF propagation, LEDH
flow, Contract E and higher-moment correction. At the terminal time its
conditioning information is Y(1:T), avoiding the filtering/smoothing mismatch
that earlier-time future-informed supervision would introduce.

Calibration uses 256 independent terminal states. Fixed centering/scaling
defines 18 first and 171 raw-second standardized features. The minimum state
SD is 5.4072485985 and minimum feature SD is 0.8315826697; no scale floor fires.
Training uses 64 datasets, validation 16, with a separate intended test set of
64 datasets. Particle randomness is independent of each latent label given
the observations. Labels appear only in the loss, never in the guide or filter.

## Training and frozen selection

The following are validation-selection losses, not heldout evidence or an
unbiased estimate of the selected candidate's generalization performance.

| Guide family | Initial validation loss | Frozen validation loss | Selected capacity, seed, update |
| --- | ---: | ---: | --- |
| Quadratic cSMC | 0.744141718 | 0.707827305 | Time-specific, 262025001, 16 |
| Positive RBF | 0.714975980 | 0.689779695 | Pooled, 262025001, 32 |

Both selected coefficient arrays are nonzero and were frozen before opening
test data. The two retries followed the existing training/validation-only
trigger; no test result chose a guide or optimizer setting. Full histories,
candidate scores and frozen coefficients are in campaign-01/training*.json,
validation.jsonl and selection.json. These modest families and optimization
budgets are hypotheses, not evidence that either family has sufficient capacity.

## Exact failure and interpretation

The comparison planned 768 rows: 64 datasets, two particle replicates and six
methods. It recorded 193 rows: 192 valid rows from 16 complete datasets and
one invalid unguided row at dataset 262023017, particle seed 274000034, in the
high-observation regime. The remaining 575 rows were not run. Analysis retains
the invalid row and emits no contrasts or ranking; it does not average only
the surviving datasets. The conditional heuristic and bootstrap tests are
therefore incomplete, not passed.

The exact replay takes 55.213131 seconds. All ten captured runtime source
hashes match, and regenerated observations, latent label, explanatory score
label and fitted quadratic guide match bitwise. All three analytical-score
directions fail validity. Times 1--3 are finite. At time 4, UKF, callback,
transport and reset checks remain valid, but higher-moment validity fails:
all 1,152 entries of the returned 64-by-18 cloud and its directional tangent
are nonfinite. Later-time failures are downstream consequences.

The recorded raw-reset, pre-cap and post-cap clouds at time 4 are finite.
Inspection of higher_moment_contract_e.py around line 1560 places subsequent
affine restoration before the returned corrected cloud. This localizes the
failure to the post-cap restoration/output part of the higher-moment
correction; the exact failing factorization or operation has not been isolated.
The pre-cap diagonal RMS is about 9.57e39 and the finite-particle upper margin
is about -2.92e41. This is not a 1e-13 tolerance dispute. Moment safety is off
in the frozen comparator; that frozen setting is now an exposed robustness
risk, not a justified general default. There was no silent damping, ridge,
cap or safety-policy change after seeing the holdout.

The original invalid-row writer omitted already-computed checks. The exact
diagnostic replay preserves those checks without rewriting the original run.
The runner now retains all directional checks and diagnostics even for invalid
rows, representing nonfinite diagnostic numbers as explicit strings in strict
JSON. This reporting-only repair followed the exact replay and does not alter
the archived experiment or its numerical calculation. The observed
baseline failure invalidates this comparison under the declared contract; it
does not reject the moment-training idea or show that the learned guides fail.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain training implementation | Real updates, label isolation and shared-consumer tests pass | No training validity failures | Capacity and generalization | Keep frozen candidates and complete evidence | Moment or score superiority |
| Withhold comparative conclusion | Full paired heldout criterion not evaluated | Reproduced baseline nonfinite value | Exact restoration failure mechanism | Isolate the failing restoration operation in a bounded diagnostic | Rejection of either guide family |
| Do not promote defaults | Scope-specific diagnostic only | Baseline validity and incomplete heuristic checks | Robustness outside this scope | Evaluate any numerical protection under a separate non-harm contract | Canonical admission or HMC readiness |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Baseline nonfinite cloud reproduced on exact inputs; comparison stops |
| Statistically supported ranking | None; planned paired intervals are not computed on incomplete data |
| Descriptive-only differences | Selected validation loss decreases; no heldout ranking |
| Viable candidates | Both trained families remain research candidates, not admitted methods |
| Default-readiness | Not established; no default changes |
| Next evidence needed | Valid shared baseline, non-harm-tested numerical repair if needed, then a fresh scoped heldout comparison |

## Checks, provenance and compute

Focused tests: 35 passed, two TensorFlow Probability deprecation warnings,
38.44 seconds, deliberate CPU-only execution. GPU smoke-03 separately checks
the actual shared consumer, zero-adaptation parity and changed-guide analytical
directional scores against finite differences. No autodiff score path or pfor
was introduced. The interrupted-analysis regression verifies that incomplete
evaluation cannot yield a ranking. See audit.md and the archived logs.

Main run: 5,686.268834277 process seconds and 11,955 endpoint calls. Replay:
55.213131233 seconds and four calls. Initialization/smokes are conservatively
charged 300 seconds. Total stage charge is 6,041.481965510 seconds out of the
21,600-second stage allowance. Inherited campaign allowance remaining is
27,261.957119700 seconds, about 7.57 hours. The stop is for validity, not budget.
These are process-wall charges, not exclusive GPU usage. The replay's allocator
peak of 36,583,936 bytes is not a measurement of the main campaign peak.

The main and replay manifest.json files record command, Git base, environment,
GPU memory growth, seeds, fixed controls, source hashes and artifact locations.
The working tree contains task and unrelated edits, so source hashes supplement
the Git base. The executed runner is archived with its original hash; a
post-run reporting-only edit is documented in audit.md and source-file-map.json.
The analyzer verifies that the archived bytes match the original manifest.
Commands and failed attempts are retained in execution.md; no
prior result is reused as a current baseline and no completion file is fabricated.

## Post-run skeptical review

The strongest alternative explanation for favorable validation losses is
selection over small validation data, not improved conditional moments. The
strongest alternative explanation for this interruption is a numerical weakness
of the shared higher-moment correction rather than the guide objective. The
exact replay rules out input mismatch and a reporting-only failure, but does
not identify the numerical root operation. A valid, fresh paired comparison
could establish or reject the bounded training benefit. The weakest evidence
is downstream generalization: the intended heldout experiment is incomplete.

Next: retain the failing pre/post-cap cloud and inspect its affine-restoration
covariances, factors and analytical tangents. A protection that changes accepted
numerics requires a prompt, bounded non-harm evaluation against healthy inputs
and this failure, not tuning against heldout moment losses. Any repaired-method
comparison needs fresh heldout seeds and a refreshed scope/plan. Do not rerun
this failed holdout as if it were untouched, or silently skip its invalid row.
