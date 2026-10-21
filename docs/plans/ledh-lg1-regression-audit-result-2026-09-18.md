# LG1 regression investigation — 2026-09-18

The checked LG1 failure is **not a material regression introduced by the
graceful-failure changes**. The starting revision already fails the same Kalman
covariance screen on identical inputs. This conclusion applies to the replayed
case; it does not certify every earlier LGSSM configuration or the changed
multi-block transport path.

The original campaign closeout should have included this comparison before
explaining the apparent loss of earlier LGSSM performance. Engineering failure
containment, approximation of the exact filtering law, and reproduction of an
earlier successful benchmark are different questions. The campaign established
the first in its exercised scope, failed the second in LG1 calibration, and had
not established the third.

## Fixed-input comparison

The [pre-run audit](ledh-lg1-regression-audit-2026-09-18.md) fixes calibration
replicate 2, phi=-0.9, N=1024, T=10, float32/TF32/XLA, scrambled Halton inputs,
three flow steps, coordinate cap 16, and the remaining previously evaluated
controls. Both revisions receive identical materialized inputs, stationary
initial states and covariances, and their analytical derivatives. The prior
revision is `6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef`; the new revision is
`8a4cffa2fd085f43ab5710c24d6e7b48770aea0d`. N1024 uses K=N in both revisions.
This is a debugging comparison, not a new calibration candidate.

| Quantity | Starting recursion | Revised recursion | Revised LG1 consumer |
|---|---:|---:|---:|
| Log likelihood | -43.42171860 | -43.42172241 | -43.42172186 |
| Analytical score | 18.25999832 | 18.26001549 | 18.26001740 |
| Maximum mean error, in exact filtering SDs | 0.04293395 | 0.04293474 | 0.04293395 |
| Maximum relative covariance error | 16.000620% | 16.000531% | 16.000531% |
| Required covariance error at most 5% | Fail | Fail | Fail |

The revised recursion changes the log likelihood by -3.8147e-6 and the score
by 1.7166e-5. The largest absolute differences in filtering means and
covariances are respectively 4.7684e-7 and 3.2783e-7. Full per-time snapshots
of flowed particles, weights, reset particles, and predicted covariances all
pass the predeclared scaled 1e-4 comparison; their maximum scaled discrepancy
is 9.0834e-7. The actual consumer reports valid value and force and agrees
with the direct recursion. The likelihood discrepancy to exact Kalman is
approximately -0.18527 under both revisions, far larger than their difference.

The [comparison](artifacts/ledh-lg1-regression-audit-20260918-01/comparison.json)
and [run manifest](artifacts/ledh-lg1-regression-audit-20260918-01/run-manifest.json)
preserve the evidence. The scalar Kalman diagnostic reproduces the stored
reference means, covariances, and likelihood exactly at the represented
float32 parameter. This checks reference arithmetic for this case.

## What the checked earlier SUCCESS actually established

The September 3
[LGSSM tuning report](../benchmarks/artifacts/ledh_trust_region_lgssm_t50_20260903/result.json)
is real, but it is not the same experiment. Its recorded revision is
`36e1f04260ba350ab796fbd9278d8527d255b8ed`. An
[executable static wiring check](artifacts/ledh-lg1-regression-audit-20260918-01/earlier_success_check.py)
verified that the four relevant source files match that recorded revision.

| Aspect | September 3 tuning report | Current LG1 calibration |
|---|---|---|
| Model and horizon | 3 state dimensions, 5 parameters, T=50 | 1 state dimension, 1 parameter, T=10 |
| Evaluated parameters | Fixed generating vector (0.72, 0.55, 0.35, 0.35, 0.45) | Grid spanning negative and positive persistence within (-0.95, 0.95) |
| Particle/reference design | N=1008; replicated GenUT reset design | Varied N; Gaussian or scrambled Halton path/residual design |
| Executed evaluator | `cubature_genut_filter.finite_value_score` | `canonical_batch_fused_value_score` → canonical analytical recursion |
| Pass criterion | Finite value/score, program validity, affine residuals, coordinate cap | Exact Kalman mean/covariance errors and centered likelihood-ratio error |

The older runner's `_valid` function at
[line 240](../benchmarks/run_ledh_trust_region_phase3_lgssm_t50.py:240)
contains no Kalman comparison. It evaluates the same parameter vector for
four rows spanning two calibration datasets and two seeds. Its `SUCCESS`
means a control setting passed that validity screen.

There is also a substantive implementation distinction. The runner calls the
staged Cubature/GenUT evaluator at
[line 156](../benchmarks/run_ledh_trust_region_phase3_lgssm_t50.py:156).
That evaluator applies the LGSSM adapter's ordinary transition, observation
weighting, Contract E reset, and shape correction; the adapter transition is
`phi*x + q_scale*noise`. This path has no canonical LEDH flow or UKF
per-particle covariance lifecycle. The
[saved wiring result](artifacts/ledh-lg1-regression-audit-20260918-01/earlier-success-call-chain.json)
preserves the checked calls and validity function. Treating that report's
`production_compliant` label as proof of canonical LEDH performance would be
wrong relative to the stated canonical target. This finding concerns that
particular report, not every prior LGSSM result.

The new negative-persistence case is also appreciably harder. The observations
were generated with phi=+0.8; the replay evaluates phi=-0.9. At that parameter,
the second observation lies 5.2161 predictive standard deviations from the
exact Kalman forecast, and the third lies 4.0439 standard deviations away.
These are directly calculated diagnostics. They explain why success around
generating parameters does not settle this test across the prior range.
They do not establish the sole cause of the finite-cloud error or justify
relaxing its threshold.

## Interpretation and remaining work

The shared recursion's output is a finite particle approximation and its
analytical derivative. It is not equal to the exact Kalman filtering law on
this case. Accurate differentiation of that finite program cannot by itself
repair its filtering-distribution error. The prior campaign's independent
scalar-flow localization weakens a simple flow-algebra-error explanation,
but it does not certify every proposal/reset operation or establish a general
consistency claim. No new explanation is promoted solely from this replay.

The inherited coordinate cap 0.98 was separately shown to distort a healthy
Gaussian cloud; that is a confirmed setting problem. This replay uses the
already tested cap16 repair and still fails, so that cap defect alone cannot
explain the remaining discrepancy.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Recent changes are not the cause of this replayed failure | Identical-input revision parity passes | Finite inputs/results and consumer status pass | One parameter/dataset/configuration only | Preserve as a regression witness | Global absence of regressions |
| This LG1 candidate remains unqualified | Exact covariance discrepancy is about 16%, above 5% | Accuracy veto fires in both revisions | Proposal/reset contribution to the approximation error | Localize the failed candidate under a bounded repair plan | Rejection of LEDH as a research direction |
| Earlier SUCCESS is an unsuitable canonical accuracy baseline | Different evaluator and validity criterion verified | Source mismatch veto on using it for that claim | Other successful LGSSM cases have not been inventoried here | Bind a verified post-rebuild success to its exact model, inputs, controls, route, and gate before extending the suite | Earlier LGSSM work never succeeded |

| Inference status | Finding |
|---|---|
| Hard veto screen | The replayed candidate fails exact covariance accuracy before and after the changes |
| Statistically supported ranking | None; this is deterministic revision parity on frozen stochastic inputs |
| Descriptive-only differences | Kalman errors, innovations, and floating-point differences for this case |
| Default readiness | Not established by this audit; the owner-selected production direction is unchanged |
| Next evidence needed | Verified matched reproduction of the intended earlier success, then scope-specific approximation repair and untouched validation |

The strongest remaining alternative is a regression in another configuration
or the N4096 multi-block path. This audit does not exclude that. The baseline's
dense N4096 implementation was not rerun because it violates the current
chunk rule. A matched, policy-compliant counterexample exceeding the stated
tolerance would overturn the absence-of-regression verdict for that case.

## Execution record

Three GPU numerical evaluations completed on the RTX 4080 SUPER with trusted
access and verified memory growth. Their measured in-script elapsed time totals
42.3921 seconds. One earlier consumer launch stopped before filter evaluation
because regenerated residual-design whitening differed between CPU and GPU by
up to 2.3842e-7. Its log and script are preserved. The localized harness repair
feeds the already frozen inputs directly to the consumer; it changes neither
the candidate nor the regression question. Thus four processes were launched
but only the three planned filter evaluations occurred.

Accounting conservatively charges the failed process's full 195-second timeout
and a 6-second startup reserve for the successful GPU processes: 243.3921 GPU
seconds, within 600. CPU preparation and snapshot inspection took 7.7667
in-script seconds; with a 30-second setup/reporting reserve the charge is
37.7667 CPU seconds, within 180. The remaining allowance is closed unused.
CPU operations intentionally hid GPUs. Exact commands, environment, hashes,
seeds, exit codes, and limitations are in the manifest. No new scientific
candidate, environment change, merge, or production-code edit was made.
