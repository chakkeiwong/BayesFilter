# Resampling controls reduce Fisher-score error but do not clear the simple filters

2026-09-18. The bounded campaign is complete. The corrected score still loses
to UKF on weak dataset 1500 and to exact Gaussian filtering on the affine
case. Those are promotion vetoes. On curved dataset 1511 its observed MSE is
slightly higher than no resampling, with uncertainty too large to rank them;
the conservative heuristic screen also records that observed loss. The
candidate therefore remains diagnostic and is not a new default.

The subsequently executed [fresh conditioning validation](conditioning-result.md)
also passes. Its final accounting supersedes the intermediate consumption
below: 3 launches, 129.311595 driver seconds, 5 fits and 6884 filter charges.
The optional input-precision safeguard is validated on five fixed scopes;
the numerical comparison reported here remains unchanged.

There is nevertheless conditional evidence for the mechanism: at N4096 the
control reduces error relative to the raw iAPF Fisher statistic on all four
fresh nonlinear datasets. Each predeclared paired 99.75% bootstrap interval
for corrected-minus-raw MSE lies below zero. These four approximate intervals
form the declared Bonferroni 99% primary family. The observed reductions are
29--47%, and all four corrected mean-bias screens pass. This is evidence for
these fixed datasets, fits and short horizons, not a universal ranking.

## What changed mathematically

The raw terminal statistic is S_N = sum_i W_T^i a_T^i, the posterior-weighted
complete-data score along the actual genealogy. It includes the initial-law
derivative. Before each nonterminal ancestor draw, the new kernel records

    C_t = sqrt(N) [mean_j a_t^{J_t^j} - sum_i q_t^i a_t^i].

The conditional expectation of the bracket is zero. The probabilities q
count the grid points actually selected by the same right-sided CDF search,
including ties and the final-index clamp; they are not merely the nominal
softmax weights. Coefficients B are fit on 192 independent calibration
streams per scope and frozen before 128 final streams. The reported candidate
is S_cv = S_N - C B. This correction is outside the self-normalized ratio, so
it preserves the raw statistic's expectation, including finite-N bias. It is
not the gradient of the finite particle likelihood and is not an HMC force.

The optional controls execute inside the existing fitted-twist kernel with
the same particles, ancestors and likelihood as both score outputs. The
default three-output API and optional Fisher-only API remain available.
Control centering/regression/application use FP64 as an explicit diagnostic
exception; particle propagation and analytical scores remain GPU FP32/TF32/XLA.

## Conditional results

The entries below are mean squared Euclidean errors of six-component scores,
conditional on each observed dataset. All N4096 results use 128 final streams.
Full N1024 and affine results, all heuristic comparisons, variance and bias
diagnostics are in [conditional-errors.csv](conditional-errors.csv),
[analysis.json](analysis.json) and [attempt02/results.json](attempt02/results.json).

| Dataset | Regime | Raw Fisher MSE | Corrected MSE | UKF MSE | No-resampling MSE | Corrected-minus-raw 99.75% interval |
|---|---|---:|---:|---:|---:|---|
| 1500 | weak | .00822256 | .00497235 | .00187874 | .0083954 | [-.00470569, -.00193964] |
| 1501 | weak | .00981919 | .00613133 | .00770804 | .0206237 | [-.00536811, -.00219281] |
| 1510 | curved | .0048678 | .0025764 | .02590289 | .0038952 | [-.00337560, -.00134403] |
| 1511 | curved | .0125275 | .0088669 | .34501855 | .0087676 | [-.00656880, -.00135074] |

The total score-variance ratios, corrected/raw, are .604, .624, .537 and .706
respectively. These ratios and percentage reductions are descriptive point
estimates; no uncertainty interval for the ratio itself was specified.
All eight nonlinear size/dataset cells have lower observed MSE than raw
Fisher, but only the four N4096 comparisons belong to the primary family.

At N4096, the corrected-minus-UKF difference on dataset 1500 has exploratory
99% interval [.00229095, .00394444], supporting that conditional loss. On
dataset 1511 the corrected-minus-no-resampling interval is
[-.00230110, .00241562], supporting no ranking. At N1024 the corrected score
also loses to EKF and UKF on both weak datasets. These conditional failures
cannot be hidden by averaging across regimes or particle counts.

The fixed-label derivative still has lower MSE on both weak N4096 datasets;
the corrected Fisher score has lower MSE on both curved N4096 datasets under
exploratory paired intervals. They compute different mathematical quantities,
so this comparison does not establish interchangeability. No candidate may
silently substitute for the finite-likelihood gradient.

## Numerical validity and regression conditioning

All five numerical references pass. Maximum grid/domain/forward-backward
relative discrepancy is 1.60e-15; maximum tail diagnostic is 3.33e-16. Affine
grid/Kalman agreement is 2.11e-15. All five offline fits are valid without
active bounds; dataset 1501 nevertheless needs 4277 optimizer steps. That
does not repair earlier failed fitting holdouts or establish robust defaults.

The 16 focused CPU tests pass, including independent density/genealogy checks,
exhaustive small-lattice centering, terminal-resampling invariance and existing
combination checks. A separate harness check executes the Kalman factory and
checks all 11520 calibration/final seed pairs are distinct. Every scored
kernel specialization and stream generator traces once. All numerical source
hashes match the run manifest. Independent saved-row application agrees with
the GPU result within the FP64 dot-product roundoff bound (maximum absolute
difference 8.62e-11, maximum bound fraction .0376). The largest empirical
control mean/MCSE is 2.64, an explanatory diagnostic across many components.

The affine regression is ill-conditioned: retained condition number 2.54e7
and maximum coefficient 216232. In the affine scalar model, each additive
score is a quadratic in (x0,x1). Conditional centering removes its constant,
leaving at most the five directions x0, x1, x0^2, x0*x1 and x1^2. The initial
draw contributes two more directions. Exact combined rank is therefore at
most seven. FP32 score rounding creates an apparent eighth singular value
8.64e-7; FP64 regression's inherited threshold incorrectly treats that
roundoff direction as resolved. This affects the affine diagnostic's numerical
robustness; nonlinear retained condition numbers are only 27--72.

The dedicated [conditioning safety check](conditioning-probe.json) uses saved
calibration rows only. A threshold based on FP32 input precision removes the
affine eighth direction, reduces its maximum coefficient to .01945 and leaves
every nonlinear coefficient and calibration correction exactly unchanged.
All nine non-harm checks pass. The threshold is a safety candidate requiring
fresh downstream validation; it has not changed the completed results or
runtime default. No final MSE was used to choose or tune this rule.

## Execution, accounting and evidence limits

Plan: [reviewed campaign](../../younis-iapf-resampling-control-2026-09-18.md).
Numerical code: fitted_twist_tf.py, complete_data_score_tf.py and existing
combinations_tf.py. Driver: diagnose_younis_iapf_resampling_control.py.
Exact commands, seeds, commit, source hashes, dtype, runtime and output paths
are in [attempt02/manifest.json](attempt02/manifest.json). Hardware is the
RTX5080, trusted escalated execution, memory growth verified before logical
initialization. Peak TensorFlow allocator memory is 33,785,856 bytes. CPU
references intentionally use FP64; CPU tests/probes intentionally hide GPU.
No runtime comparison is made on this shared device.

Attempt01 stopped at a nonexistent Kalman-factory import before any fit/filter
work. It consumed 9.419787 seconds. The corrected existing factory passed a
focused CPU execution check; attempt02 resumed the cumulative budget and
completed in 92.096776 seconds. Its full console log is attempt02.log.
Attempt01's manifest and terminal error are preserved; startup console lines
were not saved verbatim. No scientific contract or numerical kernel changed
during this localized harness repair.

Consumption: 2/4 attempts, 101.516563/1800 driver seconds, 5/8 adaptive fits,
5284/8000 filter charges, conservative 120/600 CPU-test/probe seconds including
the conditioning safety check. Unused capacity is 2 attempts, 1698.483437
driver seconds, 3 fits and 2716 filter charges; it does not authorize tuning
against these final streams. Current comparisons are complete; no process
remains running. Any follow-on run must record its new evidence contract and
consume this remaining campaign capacity or explicitly establish a new budget.

## Decisions and terminal skeptical review

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Mechanism supported on four nonlinear scopes | 4/4 paired primary intervals negative | Valid references and finite computations | Only two datasets per nonlinear regime; T=2 | Retain as a diagnostic candidate | Universal improvement or zero bias |
| No default/integration promotion | Mechanism criterion passes | UKF/affine losses; conservative no-resampling veto; affine conditioning | Broader regimes, tuning robustness, downstream usefulness | Validate conditioning safeguard on fresh streams, then address residual variance | HMC or LEDH readiness |
| Safety candidate passes calibration non-harm | Eight healthy cells identical; affine rank repaired | No nonfinite output | Fresh downstream behavior and non-affine near-dependencies | Freeze rule before fresh evaluation | Runtime/default adoption |
| Continue master research | Harness and centering identity remain valid | No scientific continuation veto | Residual innovation noise versus missing useful controls | Derive next target-preserving control/integration mechanism after conditioning validation | Rejection of iAPF, KDM or Fisher's identity |

| Inference status | Finding |
|---|---|
| Hard veto screen | Numerical/reference screens pass; explicit heuristic promotion vetoes remain. Earlier fitting-bound failures stay open. |
| Statistically supported ranking | Corrected versus raw Fisher on the four fixed nonlinear N4096 scopes, under the declared approximate primary intervals. |
| Descriptive-only differences | Variance ratios, percent gains, conditioning, optimizer effort and cross-dataset generalization. Most other intervals are exploratory. |
| Default-readiness | No; FP64 diagnostic controls, heuristic losses, affine rank repair and broad validation remain. |
| Next evidence needed | Fresh validation of the conditioning safeguard and downstream tests of a further variance mechanism, with stronger conditional baselines and untouched streams. |

Post-run red team: the strongest alternative to broad usefulness is that this
is only a favorable T=2 control for these fixed fits, with UKF still adequate
in weak regimes. A fresh downstream reversal or reference failure would
overturn nomination. The weakest evidence is affine numerical conditioning
and generalization from two datasets per nonlinear regime. Engineering
correctness, numerical-reference agreement and conditional statistical
usefulness are recorded separately; none grants HMC/default status. The
observed losses reject promotion of this candidate, not the research direction.
