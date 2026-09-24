# Log-quadratic R iAPF repair: bounded d20 result

2026-09-20. The explicit log-quadratic fitter completed four pilot repetitions
and 16 repetitions on a second, untouched dataset without a fitting failure.
On the second dataset, the mean likelihood ratio to exact Kalman was 0.989741,
with a 95% bootstrap interval [0.951537, 1.029334]. It passes the predeclared
conditional accuracy screen. This repairs the observed d20 execution failure
for this alternative R method; it does not reproduce equation (15) or the
paper's complete study.

The [plan](../../iapf-r-log-fit-repair-2026-09-20.md) and its continuation were
skeptically reviewed before each launch. The preceding
[paper/code audit](../iapf-r-paper-code-audit-20260920-01/result.md) explains why
merely increasing the relative-loss optimizer budget was insufficient.

## Method and evidence

Regress log backward targets on an intercept, centered/scaled coordinates and
their coordinatewise squares. A full-rank QR solution with strictly negative
quadratic coefficients defines a proper diagonal Gaussian. Nonconcave and
rank-deficient fits are rejected; there is no clipping or weighted-moment
fallback in this mode. The callable observation interface, backward recursion,
positive floor, sampling and importance correction are shared with the other
R modes. No linear-model oracle is substituted into the fitter. The literal
`paper_eq15` mode remains the default. A final captured-source regression also
calls the new mode on both original failed fitting inputs. Both pass, and their
returned parameters exactly equal the recorded log-quadratic initializers.
Their continuous errors are 0.175331 and 0.250284; this improves on the failed
optimized fits on those fixed functions, but remains above the diagnostic
target-moment approximations. Evidence is in `mechanics01-frozen-inputs/`.

Both datasets use the published Section 5.2 linear-Gaussian model, dimension 20
and horizon 100. Data seeds are 68000020 (pilot) and 69000020 (follow-on).
Method seeds are `53000000 + 20*100000 + replication*10 + method_id`, with
replication IDs 1--4 and 101--116, and method IDs 1--4 for iAPF, BPF, fully
adapted APF and SIS. The floor exponent 2, first-full-window controller and
all particle counts were frozen before the pilot. No setting was tuned on
either dataset. The driver executed captured sources, and the reporting step
verified source/data hashes, complete repetition sets and identical fitter and
runner sources between launches.

All 12,000 backward fits passed rank, concavity and variance guards. The largest
scaled-design condition number was 18.249 and the largest normal-equation
gradient magnitude was 4.84e-16. Original relative-density residuals reached
0.9243: the repair did not minimize that objective. The evidence for advancement
comes from the actual likelihood comparison, not small log-regression residuals.

| Method, second dataset | Repetitions | Final particles | Mean likelihood ratio | Observed SD |
| --- | ---: | ---: | ---: | ---: |
| Log-quadratic iAPF | 16 | 2000 | 0.989741 | 0.081718 |
| Fully adapted APF | 16 | 5000 | 1.089371 | 0.232597 |
| BPF | 16 | 10000 | 9.79e-18 | 3.92e-17 |
| SIS | 16 | 10000 | Underflow | Not interpretable in ratio units |

The BPF and SIS runs show severe particle degeneracy at these budgets. SIS log
ratios remain finite, between -2146.95 and -1928.58. A printed ratio and sample
SD of zero are floating-point underflow, not exactness or low variance. The
small sample cannot resolve rare large likelihood estimates for these methods.
The displayed SD and runtime differences are descriptive. No efficiency or
cross-method statistical ranking is claimed, and computing budgets are unequal.

The heuristic comparison uses mean squared log-prefix error against Kalman,
computed per repetition and then averaged, separately for ordinary and large
innovations. Large innovations exceed the unchanged chi-square 90% contour.

| Situation, second dataset | iAPF | Fully adapted APF | BPF | SIS |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | .007147 | .034241 | 1519.72 | 1,241,816.52 |
| Large innovation | .006042 | .030993 | 992.19 | 704,820.81 |

Neither situation raises the prespecified heuristic veto. These are descriptive
sanity checks, not targets used to choose the fitter. The four-repeat pilot also
raised no veto; its likelihood ratios ranged from 0.981063 to 1.066358. Pilot and
follow-on estimates are reported separately rather than pooled across datasets.

## Decision, limitations and next step

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain log fit as a viable explicit reference method | Complete 16-repeat batch; mean-ratio interval inside [.9,1.1] | No fit, numerical, provenance or conditional heuristic veto | Two datasets, only 16 confirmation repetitions; approximate bootstrap | Larger untouched d20 validation, then bounded d40/d80 pilots | Full paper replication, superiority or production readiness |
| Preserve strict equation15 separately | Its source identity is explicit | Known fitting pathology remains | Author solver, floor, initial conditions and controller behavior unknown | Resolve source choices before claiming published-result reproduction | That authors encountered the same numerical failure |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No veto in either completed log-fit batch |
| Statistically supported ranking | None; the interval supports only the declared conditional mean-accuracy screen |
| Descriptive-only differences | Method SD, runtime, prefix MSE and pilot range |
| Default-readiness | Not established; no default changed |
| Next evidence needed | Larger independent repetitions/datasets, d40/d80 and source-faithful author settings |

The 16-repeat screen is smaller than the earlier 32-repeat criterion and the
published 1000 repetitions per dimension. It does not close either requirement.
The R reference remains independent and noncanonical. TensorFlow parity,
nonlinear-model behavior, analytical scores, HMC, KDM and canonical LEDH are
outside this result.

Terminal red-team review: log fitting can emphasize tails, and a nonconcave
approximation can still fail on other data or nonlinear models. The current
linear-model datasets may be unusually favorable. A failed fresh conditional
accuracy check or concavity/rank rejection would overturn advancement. Positive
twists preserve the importance correction, but do not guarantee low variance.
The original relative-loss failures therefore remain evidence about those
fits, not evidence against every iAPF implementation.

## Reproducibility and budget

`summary.json` contains the machine-readable decision and conditional tables;
`manifest.json` records reporting inputs, seeds and source hashes. Full commands,
CPU/R environment and logs are in the attempt11 and attempt12 manifests under
`../iapf-r-reference-gap-repair-20260920-01/`. Both ran CPU-only base R 4.1.2 with
CUDA hidden and one BLAS/OpenMP thread.

The workers used 31.882971 and 161.507472 seconds, totaling 193.390443 of the
200-second follow-on allowance. Total repair use is 1516.719870/1550 seconds
across all 12 launches. Together with the earlier 248.382526 seconds, total
worker use is 1765.102396/1800 seconds. There are 33.280130 repair seconds but
no experiment launches left; a larger campaign is the next allocation, not
another retry within this one. No process is running.

Final verification: seven pytest cases pass (1.78 seconds), including 88 R
reference checks, 33 alternative-fit checks, three executed-source mutations
and captured-source regression tests. Test logs and CSVs are retained here.
Mechanics are conservatively charged 55/120 seconds for the full repair/audit
sequence. The current master checkpoint points here and preserves earlier
results as completed stages of this still-active research program.
