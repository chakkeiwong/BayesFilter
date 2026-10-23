# SIR guidance with independently calibrated proposal probabilities

Completed 19 October 2026 on `ledh-refactor-with-policy-fix`, commit
`ea1018de1da55ed02a29a8d60f08a6827fce8a42` plus the source hashes in the run
manifests. OT remained epsilon 1 and five Sinkhorn iterations throughout.

## Finding

Neither guide satisfies the predeclared improvement criterion. Both guides
reduce squared score error relative to their matched-beta unguided controls
on data 26100832 under the specified paired-bootstrap analysis. Neither
comparison is resolved on data 26100831, and neither guide has an established
gain over independently calibrated unguided proposals on either dataset.
Both also fail the componentwise simple-method screen. Thus there is a
conditional signal worth explaining, but not a successful improved-LEDH result.

All 112 prescribed rows are valid: seven arms, eight paired particle seeds,
two fixed datasets. All 84 finite-difference and 42 Richardson checks pass.
The earlier equal-mixture stage's failed global comparator remains a genuine
rank-loss result; the present run does not erase it. No failed row is dropped
or replaced with a partial score.

## Score comparison

RMSE below means sqrt(mean over particle seeds of squared Euclidean error
in the three score coordinates). It is not error of the mean score. The
coordinates are log kappa scale, log nu scale, and log observation-noise scale.
The numerical references are respectively
`[-157.1103965, 104.5783105, 56.8607014]` and
`[-135.1908139, 56.6962876, 5.8874744]`.

| Method | Data 26100831 | Data 26100832 |
|---|---:|---:|
| Unguided, independently calibrated beta | 94.6240 | 77.6984 |
| cSMC-inspired quadratic, own calibrated beta | 96.3593 | 70.8741 |
| KDM-inspired positive kernel, own calibrated beta | 84.5113 | 65.4509 |
| Unguided at quadratic beta | 82.2319 | 82.4412 |
| Unguided at kernel beta | 83.8621 | 83.6040 |
| Prior only | 82.1886 | 83.8851 |
| Defensive local | 94.6240 | 77.6984 |

These RMSEs are descriptive, not a ranking. Calibrated unguided beta is
effectively (.1,0,.9); its identical scores to defensive local are expected.
The full mean-score vectors, component MSEs, paired differences, and timing
summaries are in `evaluation-01/analysis.json` and `evaluation-01/result.md`.

The difference below is guided minus comparator mean squared Euclidean
error; negative favors guidance. The eight predeclared contrasts use 20,000
paired bootstrap resamples and nominal 99.375% intervals.

| Data | Guide | Comparator | Difference | Interval |
|---|---|---|---:|---|
| 26100831 | Quadratic | Same beta, unguided | 2523.020 | [-5902.929, 15949.535] |
| 26100831 | Kernel | Same beta, unguided | 109.303 | [-4087.144, 4433.124] |
| 26100831 | Quadratic | Independently calibrated unguided | 331.399 | [-10795.919, 15524.884] |
| 26100831 | Kernel | Independently calibrated unguided | -1811.551 | [-6298.608, 3270.145] |
| 26100832 | Quadratic | Same beta, unguided | -1773.424 | [-3845.258, -400.549] |
| 26100832 | Kernel | Same beta, unguided | -2705.807 | [-7192.819, -102.720] |
| 26100832 | Quadratic | Independently calibrated unguided | -1013.901 | [-3192.336, 647.151] |
| 26100832 | Kernel | Independently calibrated unguided | -1753.210 | [-5215.549, 820.084] |

On data 26100831 both guided arms lose to prior-only proposals in all three
component MSEs. The kernel also loses on the observation-noise coordinate
to each simple comparator on data 26100832. These are predeclared promotion
vetoes, not statistically established component rankings. With eight seeds,
the bootstrap's tail coverage is uncertain; nominal multiplicity adjustment
does not make these exact finite-sample confidence bounds.

## What was calibrated and checked

Six fits minimized the same finite pilot objective J(beta), without reference
scores, on independent prior-filter histories. All have finite held-out
validation objectives and satisfy the relative duality-gap tolerance 1e-7.
All assign nearly zero probability to the global map. This is a finding about
proposal densities, not a theorem about score-error variance. Validation J
did not select among candidates or trigger any fallback.

Two optimizer failures consumed 57.481171 and 575.421810 GPU-process seconds.
The first was a projected-gradient line-search failure near zero global mass;
the second was greedy pair-selection cycling. A log-domain, cyclic pairwise
solver preserves the objective, floor and gap certificate and passes exact
failed-pilot replays. Calibration-03 completes all six fits in 168.703374
seconds. The original shared solver and canonical defaults remain unchanged.

The evaluator consumed the frozen calibration and source hashes. A separate
terminal check verifies eight source hashes, both calibration hashes, all
112 unique keys, three complete 50-observation analytical directions per
row, fixed OT, and RMSE arithmetic (maximum discrepancy 1.42e-14). The first
audit command incorrectly expected scalar `steps_completed`; the actual
schema has one value per direction. Correcting that audit assertion to
`[50,50,50]` passes without modifying any experiment or result.
See `terminal-audit-02.json`; the empty `terminal-audit-01.json` is not a pass.

The final focused CPU regression passes 19 tests in 11.02 seconds; two
TensorFlow Probability distutils deprecation warnings are unrelated to the
numerical checks. The updated manuscript builds to 81 pages, with the score
tables and interpretation on pages 74–75 visually checked. Both masters and
the active checkpoint point to this completed result.

The MathDevMCP derivation request remains inconclusive (`not_encodable`).
Its denominator concern is resolved directly by beta0*r0 >= delta*r0 > 0
under the proposition's assumptions. Manual derivation, implementation tests
and finite-program derivative agreement do not constitute formal proof
certification. Details are in `audit.md` and `mathdev-convex-audit.json`.

## Decision and inference status

| Decision field | Status |
|---|---|
| Decision | Do not promote either frozen guide as an improved LEDH method |
| Primary criterion | Failed: no family gains on both datasets and both comparator types |
| Veto diagnostics | No numerical failures; both families fail componentwise heuristic screen |
| Main uncertainty | Eight seeds, one guide per dataset, approximate reference; no reference-uncertainty propagation |
| Next justified action | Separate current-observation proposal/reset error from future-message error on SIR, and verify the canonical consumer call chain before an LEDH claim |
| Not concluded | No rejection of cSMC/KDM generally, no canonical LEDH gain, no default or HMC readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | All numerical rows valid; simple-method screen fails |
| Statistically supported ranking | Two matched-beta contrasts on dataset 26100832 under the predeclared conditional bootstrap analysis only |
| Descriptive-only differences | All reported RMSEs, component MSE comparisons, means and runtimes |
| Default-readiness | Not established; current consumer is explicitly noncanonical |
| Next evidence needed | SIR mechanism attribution, canonical conformance, independent guide fits and fresh datasets with reference uncertainty |

The result rejects promotion of these candidates, not the research direction.
No continuation veto fires: the target, full-horizon arithmetic, frozen
settings, and artifacts remain valid. The completed stage does not justify
an OT sweep, beta selection on held-out scores, or a larger fit merely because
the present guide fails. The next phase needs its own bounded attribution
contract; the unused twelve-hour ceiling is not a reason to run an unspecified
sweep.

## Reproducibility and budget

Plan: `../../ledh-sir-beta-repair-20261019.md`. Exact commands, seeds, dataset
and guide identities, environment, source hashes and device provenance are
in `calibration-03/manifest.json` and `evaluation-01/manifest.json`. Evaluation
uses N256, T50, flow16, no correction, FP64/XLA, TF32 off, and verified memory
growth on the RTX 4080 SUPER under escalated access. This is an explicit
diagnostic exception to the FP32/TF32 production direction.

Evaluation-01 takes 4173.834361 GPU-process seconds, including 162.076230
seconds of derivative smoke checks. The beta stage takes 4975.440715 seconds,
below its 9000-second allocation. Including the first stage and its 50-second
preflight reserve, charged campaign time is 9325.730915 seconds (2.59 hours);
33874.269085 seconds remain under the owner's twelve-hour cap. No GPU
experiment remains running. CPU checks deliberately hide GPUs; their logs
are preserved separately and remain below the 300-second allocation.

The numerical reference is a saved importance/quadratic Zhao-Cui adaptation,
not an exact oracle or an untouched original-author algorithm. The kernel is
a local positive-RBF KDM-inspired adapter, not the replicated neural KDM
learner. This optional three-density consumer is not the canonical
UKF/dual-cap/Contract-E LEDH implementation. Analytical differentiation here
targets the frozen finite program, not an automatically exact model score.

Post-run red team: a single frozen guide and state-distribution mismatch
between calibration and evaluation may explain the remaining losses. The
small seed count and approximate reference are the weakest statistical
evidence. Independently fitted guides in a conforming canonical consumer,
with fresh data and uncertainty-aware reference comparisons, could overturn
the limited findings. Lower pilot J alone cannot do so.
