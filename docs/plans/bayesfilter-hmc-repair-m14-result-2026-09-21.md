# M14 preparation comparison and schedule repair

M14 completed its bounded comparisons, repaired two localized defects, and
finished the five-model public matrix. Posterior caps remain explicit failures.
The next phase is [M15 stopping and whole-fit calibration](bayesfilter-hmc-repair-m15-design-2026-09-21.md).

All evidence is under `artifacts/hmc-repair-master-2026-09-16/m14-r1/`.
`reconciliation-terminal.json` records source snapshots, commands, tests, receipts,
tensor hashes and worker costs. The experiments used TensorFlow/TFP GPU/XLA with
verified incremental allocation on shared GPUs0/2. CPU tests hid GPUs. Shared
device timing supports accounting, not performance comparisons.

## Repairs and checked results

The optional finite-window preparation now groups its existing slow transitions
into count-capable windows when affordable. It uses the existing dense or diagonal
count floor; total transitions and initial/final buffers stay fixed. The default
temporal-information schedule and explicitly supplied schedules retain their
declared windows. Numerical health, covariance screening, coordinate parity and
fresh step qualification still apply. This repairs a structural count failure,
not a posterior-convergence guarantee or a new default.

| Paired standard/finite-window preparation | Previous slow windows and result | Repaired windows and result |
| --- | --- | --- |
| Gaussian scale0.02 | 26/52/26, diagonal update only | 104, dense update |
| Rotated Gaussian condition100 | 26/52/26, diagonal update only | 104, dense update |
| Matched regression | 30/60/30, no update | 120, dense update |

These six GPU runs used identical within-model starts, data and random seeds on
frozen old/new implementations. They establish the expected schedule mechanism;
one paired seed per model cannot establish statistical superiority. A fresh public
rotated-Gaussian fit on repaired source retained all14 verified candidates. Its
selected posterior reached1024 retained transitions without passing all requested
checks, so its posterior screen failed under that small allocation.

The recovery artifact reader previously accepted a positive probe-work count
that disagreed with its saved probes. Two new tests first reproduced acceptance
of both an undercount and overcount. The reader now reconstructs discarded,
restart and successful probe work and requires equality. The fix changes
validation, preserving the numerical experiment settings and original artifacts.

The22 preparation comparisons cover scaled/rotated Gaussian, beta-binomial,
LGSSM location, regression and eight-schools, across standard versus serious
budgets and temporal-information versus finite-window policies. All18 main
comparisons completed. Standard/temporal-information made no regression metric
update; both serious policies made three. Both supplied-geometry arms completed.
The startup-disabled regression control failed bootstrap; the matched arm with
startup backoff completed. The startup-disabled scaled-Gaussian arm completed.
Startup support is therefore target-dependent, not universally necessary.

The four M13 resource continuations kept the same source, design, seed and
checkpoint; they are continuations of the original fits, not fresh replications.

| Public ordinary model | Verified members retained | Selected posterior outcome |
| --- | ---: | --- |
| Gaussian | 19 | Declared small-allocation checks passed |
| Beta-binomial | 26 | Declared small-allocation checks passed |
| LGSSM location | 19 | Declared small-allocation checks passed |
| Banana | 4 | Declared small-allocation checks passed in M13 |
| Noncentered funnel | 27 | Warmup cap1024, no retained draws |

All searches completed. The funnel outcome shows why successful tuning is
separate from posterior assessment. Its small cap motivates the later-count
experiments; it does not invalidate tuning membership or the entire model family.

## Terminal audit and next phase

Latest distinct tests: **160 passed, zero failed**. The first wider artifact-test
invocation hit its240-second resource limit; its preserved retry passed with a
660-second ceiling, costing251.43 seconds. The schedule tests cover multiple
dimensions, dense/diagonal/insufficient budgets, unchanged total work and default
behavior. The new offline phase-refresh utility has tests proving that candidate
failures permit continuation while corrupt evidence, unfinished workers, missing
designs or insufficient budgets prevent an invalid phase transition.

The official book was rebuilt to567 pages; the updated schedule page was rendered,
visually inspected and installed as `docs/main.pdf`. The agent reference and book
agree on the optional schedule's scope. The same three historical citation keys
remain assigned to M18 source checking.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain schedule repair on optional path | Count-capable windows and three paired metric updates observed | No new numerical failure | Geometry usefulness across targets | Replicated whole fits | Superior or default-ready policy |
| Accept accounting-reader repair | Both tampered-count regressions now rejected | No outstanding reader failure | Broader historical inputs remain separately versioned | Preserve source binding and run full pipeline | Sampler calibration from schema validation |
| Advance M15 | All M14 jobs terminal; integrity and budgets pass | Startup control and posterior caps remain recorded failures | Stopping and output-law accuracy | Fresh data, whole-fit SBC, fixed/stopped comparisons | Program or scientific-gap completion |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Startup-disabled regression bootstrap fails; no invalid artifact |
| Statistically supported ranking | None |
| Descriptive-only differences | Metric counts, candidate counts and wall times |
| Default readiness | Not established; existing default is preserved |
| Next evidence needed | Later stopping/caps, independent datasets and defect sensitivity |

M14 charged **1448.630407237215 CPU** and **4056.9207649671007 GPU seconds**,
including the failed test attempt and all new continuations. Original M13 attempts
remain charged once to M13. Remaining campaign funds are **146468.9776886585 CPU**
and **48626.235068081936 GPU seconds**. No worker remains active in M14.

Post-run red team: count sufficiency is a necessary engineering precondition,
not proof that a short covariance estimate is useful. The unchanged dense floor
is itself an inherited heuristic. Later posterior failure would not contradict
the schedule repair; it would identify another needed investigation. Statistical
calibration remains the weakest evidence, which is why the program continues.
