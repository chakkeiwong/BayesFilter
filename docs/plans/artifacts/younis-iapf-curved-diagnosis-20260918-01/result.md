# Curved iAPF diagnosis: the remaining score error is not just an optimizer cap

The [reviewed plan](../../younis-iapf-curved-diagnosis-2026-09-18.md) is executed.
The nominated iAPF loses to a simple no-resampling importance sampler on the
curved calibration dataset at N=4096: score MSE 0.009514 versus 0.005106. The
paired 99% interval for the MSE difference is [0.001121, 0.007394], conditional
on this dataset and frozen fit. It also has an observed, statistically
unresolved loss on curved validation. Both confirmation datasets encounter
an active bound during offline fitting. These are promotion vetoes.

The evidence points to two distinct remaining problems: fitting choices still
matter, and the fixed-label derivative does not account for how categorical
sampling probabilities change with parameters. This campaign does not reject
iAPF as a likelihood estimator, KDM, or the LEDH research direction.

## What was checked

Six fresh scalar T=2 datasets cover weak and curved nonlinear regimes:
calibration 1300/1301, validation 1310/1311, confirmation 1320/1321.
Fitting controls were selected using calibration shape and validity only,
then frozen before validation and confirmation. Fitted coefficients were
held fixed across the particle ladder. Each final particle comparison uses
32 independent replicates, with common random streams across methods.

The comparator set contains the original density fit, relative-shape fit,
selected floor candidate, EKF, UKF, bootstrap, local-linear particle filter,
and bootstrap without resampling. The last method isolates resampling at this
short horizon; it is not proposed as a general long-horizon replacement.
The score MSE is the replicate mean of squared Euclidean error across all
six parameters. Value MSE refers to log likelihood.

The refined CPU FP64 oracle passes all six datasets. Maximum relative
mesh/domain discrepancies are 4.85e-16 and 1.56e-15; backward/forward likelihood
disagreement is at most 1.62e-16, and recorded tail mass at most 2.23e-16.
These are agreement diagnostics, not rigorous universal quadrature bounds.
Data are rounded to the observations actually consumed by the FP32 kernels.

## Fitting and approximation

On identical calibration clouds, the relative-shape fits stop at 16/47 steps
(weak) and 47/13 steps (curved), with identical results under the 2000, 5000,
and 10000 caps. The complete adaptive calibration fits also converge early.
Raising the cap therefore cannot explain their downstream discrepancy.
However, fresh validation needs 3083 steps, and weak confirmation needs 3823;
the old 2000-step cap remains insufficient for some other datasets.

An independent parameter-box search closely reproduces the projected fit to
the true backward functions. Original-bound shape residuals are
8.73e-5/1.05e-4 for weak and 0.002969/0.001618 for curved. Wider bounds do not
remove these residuals. This supports a real approximation component but
does not prove global optimality or a practically irreducible Gaussian error.
The small particle fitting clouds remain another approximation source.

The floor 0.001 candidate was nominated because its worst calibration
predictive-weighted shape residual was 0.010676, compared with 0.011292 for
floor 0.01 and 0.020500 for widened bounds. It is a diagnostic nomination,
not a scope-specific admission artifact or a new default. Original density
fits hit a bound on every dataset. The nominated candidate encounters a bound
during offline fitting on both confirmation datasets; its final fitting
iteration remains bound-active on 1320, but is interior on 1321. The declared
veto concerns any offline bound activity, not just the final iteration.

## Likelihood and score separate as N grows

![Conditional particle ladders](score-ladder.png)

The figure is descriptive, conditional on two calibration datasets and one
frozen fit per method. [PDF](score-ladder.pdf), [all N=4096 methods](n4096-summary.csv),
and [replicate-level results](attempt03/results.json) are preserved.

| Curved calibration, nominated iAPF | N=256 | N=1024 | N=4096 |
|---|---:|---:|---:|
| Log-likelihood MSE | 0.005386 | 0.000963 | 0.000239 |
| Score MSE | 0.05565 | 0.01343 | 0.009514 |
| Mean error in log process-noise SD score | 0.04300 | 0.03815 | 0.04327 |
| Monte Carlo SE of that mean | 0.00853 | 0.00436 | 0.00206 |

That score component stays away from zero as sampling uncertainty shrinks.
At N=4096 it is 21.05 Monte Carlo SEs from the reference. The no-resampling
control's largest component discrepancy is 2.23 SEs on this dataset. The
bootstrap fixed-label control has a still larger persistent discrepancy.
The optional N=16384 extension was unnecessary for identifying the next
repair. A finite ladder does not establish a universal asymptotic iAPF bias.

For the same frozen coefficients and uniforms, analytical iAPF derivatives
match central finite differences at h=1e-5 to 7.86e-11 (weak) and 1.68e-10
(curved). Thus the checked local calculus is correct for its stated finite
program. It is wrong to identify that quantity with the exact physical-model
score without further justification.

The code samples ancestors using `tf.searchsorted` and then gathers their
state tangents in `fitted_twist_tf.py::make_fitted_twist_kernel` (lines 66--70).
The mixture choice in `twisted_transition` is also held fixed. For a categorical
choice with probabilities q_k(theta),

\[
\nabla_\theta\sum_k q_k F_k
=\sum_k q_k\nabla_\theta F_k+\sum_k F_k\nabla_\theta q_k.
\]

Differentiating the realized branch covers the first contribution; the second
requires accounting for the changing sampling law. The [mathematical note](math.md)
derives the explicit missing term for the two-step bootstrap control. The iAPF
has additional initial and mixture choices, so that bootstrap formula is not
being claimed as its complete bias formula. The observed iAPF discrepancies
and this mechanism justify score-construction work; they do not isolate the
numerical contribution of each categorical choice.
Moreover, even a complete derivative of E[log Zhat] generally differs from
the physical score, which is (gradient E[Zhat])/E[Zhat] for an unbiased
likelihood estimator. A sampling-law correction must therefore be derived
together with its normalization target; adding a term is not by itself a
finite-N unbiased-score proof.

## Fresh data and uncertainty

| Dataset | Regime | Original density MSE | Relative shape, floor .01 | Nominated floor .001 | Lowest observed heuristic MSE |
|---|---|---:|---:|---:|---|
| 1310 validation | weak | 0.06815 | 0.01093 | 0.003580 | UKF 0.004929 |
| 1311 validation | curved | 0.006323 | 0.003979 | 0.003600 | No resampling 0.003046 |
| 1320 confirmation | weak | 0.11358 | 0.10803 | 0.009356 | UKF 0.01750 |
| 1321 confirmation | curved | 1.30423 | 0.11521 | 0.022008 | No resampling 0.06448 |

These values describe individual datasets; the last column is not an assertion
of a globally best heuristic. Full comparator rows and conditional intervals
remain in the JSON. The floor candidate's observed loss to no-resampling on
1311 has paired 99% difference interval [-0.001418, 0.002140], so no ranking is
supported there. Its comparisons with all five heuristics have negative
conditional intervals on 1310, 1320, and 1321. Those observations do not clear
the separate fitting veto or establish population-wide superiority. The weak
calibration loss to UKF is also unresolved: [-0.000196, 0.000832]. No familywise
claim across datasets and comparators is made.

## Decision

| Field | Finding |
|---|---|
| Decision | Diagnosis completed; no iAPF setting promoted |
| Primary diagnostic criterion | Same-cloud cap effect ruled out on calibration; floor sensitivity and persistent mean-score discrepancies identified |
| Veto diagnostics | References and finite outputs pass; heuristic losses and offline fit bounds block promotion |
| Main uncertainty | Six datasets, one fit per arm/dataset, conditional 32-replicate uncertainty; no exact iAPF asymptotic bias calculation |
| Next justified action | Derive and verify an analytical model-score estimator accounting for sampling-law effects, with the existing finite-program derivative retained as a separate diagnostic; repair fitting bounds on fresh data |
| What is not concluded | No rejection of iAPF likelihood estimation, KDM, or LEDH; no unbiased-score, default, HMC, or canonical LEDH claim |

| Inference status | Conclusion |
|---|---|
| Hard veto screen | Fit boundaries and declared observed heuristic losses are supported; no reference or numerical-validity failure in attempt03 |
| Statistically supported ranking | Specified paired 99% comparisons support conditional differences on individual datasets only; no broad ranking |
| Descriptive-only differences | Cross-dataset means, shape residuals, particle-ladder trends, and comparisons not covered by the declared intervals |
| Default readiness | None; floor .001 and wider/family alternatives remain research hypotheses |
| Next evidence needed | Sampling-law derivation and analytical implementation checks, followed by fresh fit calibration and replicated model-score comparisons |

The current candidate failed promotion. The harness, physical target, and
research direction were not invalidated. A natural next comparator is the
master program's complete-data/Fisher score for this regular transition model.
It must be derived and tested as a model-score estimator; it must not silently
replace the derivative of the finite likelihood program or inherit HMC claims.

Post-run skeptical review: the strongest alternative explanations are small
fitting clouds, a nonzero floor, and finite-N bias. Removing resampling changes
the sampling distribution and variance, so it is not an exact isolation of
every iAPF branch contribution. An independently derived iAPF limiting score
equal to the model score, or a larger-N study showing the discrepancy decaying,
would weaken the attribution to categorical sampling. The weakest evidence is
generality across datasets and longer horizons. The Gaussian-family result
does not justify declaring that family unusable.

## Repairs, verification, and provenance

Attempt01 stopped because the new diagnostic omitted the consumer's model
identifier. The identifier and an actual-consumer regression were added.
Attempt02 completed calibration, then exposed API drift in the existing UKF
baseline: shared quadrature now returns validity flags that the consumer did
not unpack. `nonlinear_tf.py::make_moment_filter` now unpacks and propagates
both predict/update validity flags and rejects invalid results. This is the
only runtime source change; iAPF/KDM formulas and defaults are unchanged.
Attempt03 completed all planned comparisons. Previous attempts and logs remain.

Twelve focused tests pass: independent reference/fit checks, the actual iAPF
consumer, affine and nonlinear derivative identities, rejection of false UKF
validity flags, and execution of all five baselines. CPU tests intentionally
hide GPU. GPU final kernels execute FP32/TF32/XLA on RTX 5080 UUID
`GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`; memory growth is verified. Grid
references are CPU FP64; fitting and finite-difference diagnostics use FP64.
Concurrent GPU use precludes runtime ranking.

Source base: `6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef`, branch `surrogate-hmc`,
with source SHA-256 records and actual command in each attempt's manifest.
The successful [manifest](attempt03/manifest.json) records environment, seeds,
data identity, settings, source closure, placement, output paths, and timings.
The command uses `/home/chakwong/anaconda3/envs/tftwogpu/bin/python
docs/benchmarks/diagnose_younis_iapf_curved.py --output
docs/plans/artifacts/younis-iapf-curved-diagnosis-20260918-01/attempt03
--wall-seconds 1800`, with the above CUDA UUID and
`TF_FORCE_GPU_ALLOW_GROWTH=true` exported before import.

Total campaign use: 3/4 driver attempts, 252.093/2400 wall seconds (including
startup), 37/40 adaptive fit charges, 90/320 fixed-cloud fit charges, and
3141/3500 filter-call charges. These are conservative charges, including
failed attempts, not claims about FLOPs. Focused tests took 35.173 seconds;
adding the conservative 10-second probe remains below the 600-second allowance.
No further launch is needed. [Machine-readable decision](decision.json) and
[reporting script](report.py) preserve the rollup.

Final verification: all 36 saved source fingerprints still match the successful
run, all six dataset outputs are present, and 2688 final stochastic filter rows
are preserved. Driver/report compilation and `git diff --check` pass. The
rendered PNG was inspected for legible labels and layout.
