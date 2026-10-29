# Marginal importance weights: completed experiment, partial range-bearing repair

The requested LGSSM, KSC stochastic-volatility, and range-bearing experiments
are complete. All 15 LGSSM/KSC conditional comparisons support material
non-inferiority of the new likelihood and score errors. At 4,096 particles,
range-bearing errors decrease substantially relative to the original ancestor
weights, with paired uncertainty evidence. However, simple bootstrap filtering
still has lower observed likelihood error in all four range-bearing comparisons
at that particle count. The change is a partial repair, not a solved
range-bearing likelihood/score method or an HMC-ready default.

The research goal is **testing surrogate HMC**. This experiment tests and
repairs its LEDH likelihood/score component. It does not train a neural
surrogate, run posterior chains, or establish posterior accuracy, acceptance,
R-hat, or convergence. The existing default remains `importance_weight_policy="ancestor"`;
the tested extension is explicitly selected with `"marginal_mixture"`.

## What changed and what was checked

The original ancestor-conditioned importance identity is mathematically valid.
Its accuracy in these nonlinear examples was the problem; the new formula is
an optional change of importance-weight law, not correction of an invalid
ancestor identity.

Conditional on the incoming weighted cloud, let
\(f_j(x)=N(x;m_j,Q)\), and let the actual finite LEDH flow for ancestor \(j\)
be \(F_j(z)=B_jz+c_j\). Its pushed Gaussian density is
\(q_j(x)=N(x;F_j(m_j),B_jQB_j^\top)\). Define
\[
p_N(x)=\sum_j w_jf_j(x),\qquad q_N(x)=\sum_j w_jq_j(x).
\]
With one child \(X_i\) per ancestor, the new unnormalized contribution is
\[
V_i=w_i\,g(y\mid X_i)\frac{p_N(X_i)}{q_N(X_i)}.
\]
The outer \(w_i\) is essential. For random proposal draws and fixed admissible
flow maps, the conditional identity is
\[
\mathbb E\!\left[\sum_i V_i\right]
=\sum_i w_i\int q_i(x)g(y\mid x)\frac{p_N(x)}{q_N(x)}\,dx
=\int g(y\mid x)p_N(x)\,dx.
\]
This requires full-rank Gaussian transitions, nonsingular affine maps, and map
coefficients independent of the current innovation draw. It proves the
conditional importance identity, not unbiasedness of the deterministic
multi-step log likelihood or score, nor universal variance reduction.

The implementation accumulates \(B_j\), its analytical tangent, and the
auxiliary endpoint in the **same flow loop**. It evaluates both Gaussian
mixtures, including all component means, covariances, incoming weights and
their parameter dependence. The score propagates the total derivative of this
finite program analytically. UKF covariance propagation, OT Contract E reset,
and GenUT dual-cap correction continue through the shared implementation.
Unsupported transition/annealing cases fail explicitly.

Checked execution paths and evidence:

- Five focused tests cover mixture density/tangent calculations, nonuniform
  weights, identity flow, invalid covariance, and the actual affine flow map.
- All 14 full-filter FP64 finite-difference cases pass: seven requested
  model/horizon scopes, each with both weight laws and two difference steps.
  These use the batch consumer at batch size one, with every physical parameter
  direction checked. Multiple-row marginal-policy parity was not separately
  tested; no claim about that broader lane is made.
- Three saved old-default regressions have exactly unchanged value and score.
  All 832 recorded pre-change Python files still match their saved checksums.
- All 12 existing batch API regression tests pass.
- An N=4096 XLA check exercises the actual 2048-by-2048 tiles against an
  independently reduced mixture: value error `1.78e-15`, maximum derivative
  error `1.64e-10` at the larger finite-difference step.
- Six final driver tests pass, including missing-baseline rejection before
  framework import and rejection of a changed particle count on resume.

Implementation: [mixture kernel](../../bayesfilter/highdim/ledh_marginal_weights_tf.py),
[shared filter](../../bayesfilter/highdim/ledh_canonical_score_tf.py),
[batch consumer](../../bayesfilter/highdim/ledh_canonical_batch_fused_tf.py),
[master program](../../scripts/run_surrogate_hmc_marginal_weight_campaign.py).

## LGSSM and KSC non-deterioration

Each scope uses N=256 and three physical parameter points, with 32 paired
particle designs per point. Controls were selected from two explicitly
declared hypotheses using separate calibration/validation data and then frozen.
Each model has one untouched claim dataset: replication measures integration
uncertainty conditional on that dataset, not generalization across datasets.

For each design, the errors are absolute total log-likelihood error in nats and
\(0.1\|s-s_{\rm ref}\|_2/\sqrt p\), where \(s\) is the score in the tested
physical parameters. Positive differences mean the new method is worse. The
predeclared materiality margin is 0.05 for each metric. The table shows the
largest simultaneous paired upper confidence bound across each scope's three
points; every bound must be below 0.05. These are error differences, not raw
likelihood differences or likelihood MCSEs.

| Model | State/observation dimensions | Horizon | Likelihood-error upper bound | Scaled-score-error upper bound | Decision |
|---|---:|---:|---:|---:|---|
| LG1 | 1 / 1 | 2 | 0.00878 | 0.00168 | Pass |
| LG3 | 3 / 3 | 2 | 0.02782 | 0.00688 | Pass |
| LG3 | 3 / 3 | 10 | 0.04363 | -0.00074 | Pass |
| KSC SV | 1 / 1 | 10 | 0.01672 | 0.00862 | Pass |
| KSC SV | 1 / 1 | 20 | 0.04609 | 0.00632 | Pass |

Thus all 15 cells pass for both metrics; no statistically supported
deterioration is detected. This is stronger than simply failing to reject a
difference: the upper bounds exclude the specified material deterioration.
The score scaling and margins remain local hypotheses, not universal HMC
tolerances. KSC uses the same seven-component log-chi-square mixture and the
corrected beta derivative, not a Gaussian substitute.

The paired t intervals use tail probability `0.05/180`, allowing for the
15 cells, two metrics, three possible looks and two tails. Bootstrap
sensitivity checks agree, and the declared outlier check does not veto these
results. The t approximation and extreme bootstrap quantiles from 50,000
resamples are finite-sample limitations, not exact coverage guarantees.
No second sampling look was needed.

LGSSM references are independent Kalman calculations. For KSC, doubling the
integration grid from 512 to 1024 cells and extending the boundary from 18 to
24 changes values by at most `3.56e-15` and scaled scores by `6.22e-16`;
reference-score finite-difference error is below `8.8e-9`.

## Range bearing: substantial improvement, remaining failure

M13 has four state variables, two observations, and two inferred log noise
scales. Nominal parameters are `(0,0)`; the difficult lower-Q point is
`(-0.73272085,-0.01465508)`. The N=4096 refinement repeats calibration for that
particle-count scope and uses 32 paired designs per cell. It reproduces the
original particle count, data and difficult regime, not every old control or
random design.

| Horizon / point | Mean absolute log error, old → new | Mean scaled score error, old → new | Bootstrap PF mean absolute log error |
|---|---:|---:|---:|
| T=1, nominal | 1.083 → 0.125 | 0.131 → 0.024 | 0.059 |
| T=1, difficult | 2.428 → 0.497 | 0.171 → 0.074 | 0.045 |
| T=20, nominal | 1.271 → 0.143 | 0.327 → 0.035 | 0.122 |
| T=20, difficult | 21.294 → 0.770 | 6.325 → 0.351 | 0.489 |

All 32 designs are numerically valid for both methods in every N=4096 cell.
Both paired intervals for the change in error are below zero in all four
cells. For difficult T=20 they are `[-26.17,-14.88]` for log error and
`[-7.646,-4.302]` for scaled score error. This supports lower measured error
relative to the ancestor method under the stated conditional comparison.

T=1 uses refined independent quadrature. T=20 uses replicated bootstrap
filtering with Fisher-identity scores, not an exact oracle. At N=65536 and
16 replicates, its likelihood MCSE is 0.0095 nat nominal and 0.0323 nat
difficult; physical score MCSEs are `(0.0778,0.0348)` and `(0.1678,0.0933)`.
Refining from N=32768 yields changes within 1.55 combined MCSEs. Perturbing the
reference through a ±3-MCSE box leaves all N=4096 T=20 paired upper bounds
negative. This addresses reference Monte Carlo noise; it does not bound
particle-reference bias.

The N=256 rung also remains recorded. At difficult T=20, the ancestor method
has one invalid design out of 32, while the marginal method has 32 valid
designs. The latter still has mean absolute log error 10.25 nats. The invalid
old replicate was retained, and no finite paired accuracy conclusion was
computed there. The generic saved `INVALID_CANDIDATE_OUTPUT` label means the
paired comparison failed validity; it must not be read as an invalid marginal
output in this cell.

The simple-comparator check prevents promotion. At N=256 the constructed
comparators are prior-trajectory importance sampling, bootstrap filtering, and
a Gaussian filter (exact Kalman for LGSSM, moment-collapsed KSC filter, EKF for
M13). For example, difficult M13 T=1 mean absolute errors are respectively
`0.189`, `0.229`, and `0.0265`, versus `1.817` for marginal weights. At N=4096,
bootstrap errors are lower in every row of the table above. These are
descriptive losses, not uncertainty-supported rankings against the heuristics;
under the predeclared sanity rule they still veto promotion. Kalman's exact
answer is the expected LGSSM accuracy authority, not a method this general
particle filter is claimed to beat. All conditional comparator rows and the
explicit machine-readable promotion veto are preserved in `aggregate.json`.

N=4096 T=20 median evaluation time is about 1.12–1.13 seconds for marginal
weights versus 0.68–0.69 seconds for ancestor weights, approximately 1.65 times
the cost. These timings are descriptive and are not equal-cost comparisons
with bootstrap filtering.

## Precision and repairs

FP32/TF32 remains the candidate execution mode. Selected FP64 comparisons on
the same rounded inputs give maximum LGSSM/KSC differences of 0.00264 nat and
0.000578 in scaled score. At the difficult N=256 M13 T=20 design, marginal
FP32/FP64 differences are 0.0486 nat and 0.0488 scaled score; ancestor
differences are 0.179 nat and 0.174 scaled score. A repeated marginal FP32
evaluation preserves its saved value but differs by 0.0350 in scaled score.
This is a recorded numerical-sensitivity warning. The exact cause has not
been isolated, and this selected N=256 check is not an N=4096 precision
guarantee. It does not negate the much larger paired error reduction.

Localized repairs consumed the same campaign budget:

1. The initial saved source package needed its compiled support library;
   that infrastructure failure and retry are preserved.
2. Invalid calibration outputs originally failed strict JSON serialization.
   The harness now preserves explicit nonfinite strings and validity flags,
   rejects the invalid setting, and continues to the next planned comparison.
3. N=4096 exposed an XLA dynamic-slice shape problem. Explicit `tf.slice`
   extents repair exact tiling; the numerical formula is unchanged. The old
   kernel, failure, focused regression and independent tile check are saved.
4. Final audit found that a missing baseline directory in a future fresh run
   could fall through to importing current code. The master now requires the
   saved package, verifies its import origin, records its hashes, binds resume
   settings, and clears stale worker PIDs after exit. Existing results used
   the actual saved baseline and are unaffected.

No default, model, promotion margin or claim dataset was changed in response
to holdout results. The selected controls remain diagnostic scope settings,
not a canonical tuning artifact. Two control hypotheses are a narrow search,
not evidence of optimal tuning.

## Decision and uncertainty

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Retain the optional implementation | Analytical/FD and default-parity checks pass | No unresolved tested identity failure | Untested configurations and multiple-row marginal parity | Keep opt-in; preserve evidence | General implementation certification |
| Accept LGSSM/KSC non-deterioration in tested scopes | 15/15 cells pass both upper-bound criteria | No invalid claim designs or reference failures | One dataset per scope; local margins and interval approximation | No additional run needed for this requested comparison | Universal equivalence or HMC correctness |
| Classify M13 as a partial repair | N4096 paired errors decrease in all four cells | Simple-comparator promotion veto; N256 ancestor validity failure | Residual bias, reference bias and precision sensitivity | A follow-up investigation would localize remaining error against PF/EKF | Solved M13, superior general filter, new default |
| Close this experiment program | All planned models and completion checks executed | No missing planned experiment | Downstream surrogate-HMC evidence is outside this experiment | Archive results and report completion | Completion of the broader surrogate-HMC campaign |

| Inference status | Finding |
|---|---|
| Hard veto screen | No N4096 validity failures; one N256 ancestor failure retained; observed heuristic losses block promotion |
| Statistically supported ranking | Marginal errors are lower than ancestor errors in the tested N4096 M13 cells, conditional on the stated references and designs |
| Descriptive-only differences | Runtime, precision differences, and rankings against simple comparators |
| Default-readiness | Not established; the extension remains optional |
| Next evidence needed | A separately specified diagnosis of residual M13 error, broader datasets and scope tuning, then downstream surrogate-HMC evidence before any HMC/default claim |

Post-run skeptical review: the strongest alternative explanation for the
remaining M13 gap is proposal coverage and accumulated filtering/reset error
that a marginal ratio cannot remove. This is a hypothesis, not an established
cause. For T=20, reference bias could also change absolute-error estimates;
the T=1 quadrature failure relative to simple methods remains independently
informative. Broader datasets or a more accurate long-horizon reference could
overturn generalizations beyond the tested cells. The weakest evidence is
cross-dataset applicability and numerical robustness outside the checked
designs. Engineering parity must not be promoted into posterior validity.

## Reproduction and execution record

Plan: [original evidence contract](surrogate-hmc-marginal-weight-execution-2026-10-05.md)
and [completion checks](surrogate-hmc-marginal-weight-follow-through-2026-10-05.md).
Evidence roots are `docs/plans/artifacts/surrogate-hmc-marginal-weight-20261005-`
with suffixes `01`, `02`, `03`, and `supplement`. Root `01` contains the
pre-change source snapshot, default parity, engineering checks and LG/KSC
results. Root `02` contains the repaired N256 M13 runs; root `03` the N4096
runs. Historical failed attempt states are preserved rather than relabeled.

Environment: commit `a55c6be7a16e9401242ee3449248fa347eb58d90` plus recorded
working-tree changes; conda `tftwogpu`; TensorFlow `2.20.0-dev0+selfbuilt`;
RTX 4080 SUPER UUID `GPU-68251639-fe82-8f81-3ccc-2953c32e805b` with escalated
access and verified memory growth. Candidate kernels use FP32/TF32/XLA.
FP64 independent references are explicit exceptions. CPU test processes hide
the GPU. Per-phase manifests retain exact commands, environment, source
hashes, seeds, data and wall time; prepared observations are saved with results.

The main launches used `scripts/run_surrogate_hmc_marginal_weight_campaign.py`:
root `01` with N256 and the pre-existing `source_before` package, root `02`
with `--only m13-t1,m13-t20 --particles 256`, and root `03` with the same
phases at `--particles 4096`. The supplementary script
`scripts/diagnose_marginal_weight_campaign.py` ran `tile`, `precision`, and
`heuristics`. These launches are complete; they are not commands to rerun.

Future fresh full runs must supply the protected baseline, for example:

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_marginal_weight_campaign.py \
  --output docs/plans/artifacts/surrogate-hmc-marginal-weight-FRESH \
  --baseline-source docs/plans/artifacts/surrogate-hmc-marginal-weight-20261005-01/source_before \
  --particles 256 --budget-seconds 21600
```

Use a genuinely fresh output root and the remaining authorized budget, not a
new unaccounted budget. Changing source or scope requires a new attempt root.
This example documents executability; it does not authorize or launch another
campaign. N4096 and completion diagnostics are separate declared phases.
`next-phase.json` is refreshed by the master between model phases.

[Aggregate results](artifacts/surrogate-hmc-marginal-weight-20261005-supplement/aggregate.json)
are reproducible with
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/analyze_marginal_weight_campaign.py`;
this is post-run analysis, not a runtime admission authority.
[Completion record](artifacts/surrogate-hmc-marginal-weight-20261005-supplement/completion.json)
connects all attempt roots, final checks, decisions and provenance.

Accounted experiment time is 2,279.92 seconds (about 38 minutes), including
the failed-tile allowance and a 60-second startup reserve, out of the
21,600-second elapsed GPU-work budget. The principal CPU regressions took
about four minutes within their 1,800-second allowance. All workers have
finished. We stop because the requested experiment is complete, not because
the budget ran out. Further M13 localization is a new follow-up phase to
design from these results, not an unfinished command in this program.

The requested Git checkpoint was additionally checked in an isolated source
checkout containing the selected changes and required comparison dependencies,
without the unrelated staged HMC changes. All 23 mixture, driver and batch API
tests passed in 134.42 seconds. This was an intentional CPU-only engineering
check using the existing generated native support library. Its
[manifest](artifacts/surrogate-hmc-marginal-weight-20261005-supplement/commit-check-manifest.json)
records source hashes and the library identity. The proposed
[further verification](surrogate-hmc-marginal-weight-verification-proposal-2026-10-05.md)
adds independent datasets, stronger references and cost comparisons; it has not
been executed and does not alter the completed experiment's verdict.
