# M16 boundary behavior and diagnostic sensitivity

Bounded execution is complete, with no invalid numerical evidence. The source
and budget audit is `artifacts/hmc-repair-master-2026-09-16/m16-r1/reconciliation-terminal.json`;
the full frequencies, exact binomial intervals and look counts are in
`terminal-summary.json` in that directory. These results support the stated
mechanisms and expose statistical limits; they do not establish universal
calibration or change a tuning default.

## Public acceptance screen

All 384 CPU-reference and 192 trusted GPU/XLA fresh searches completed. All
six independent stationary Gaussian references on each device passed the
predeclared 99% familywise approximate Monte Carlo radius and independent
endpoint-energy reconstruction. The twelve pilot searches are excluded.

| Stationary acceptance | Qualified CPU /64 | Qualified GPU /32 |
| ---: | ---: | ---: |
| .64 | 44 | 23 |
| .65 | 55 | 28 |
| .66 | 62 | 29 |
| .74 | 61 | 29 |
| .75 | 57 | 26 |
| .76 | 38 | 22 |

This is an uncertainty-aware compatibility screen, not a strict classifier
of stationary means inside [.65,.75]. Its short fixed-start window is also a
different quantity from stationary acceptance. Rates near the boundaries,
including outside them, must not be called nominal false-admission rates.
The 3072 synthetic searches separately show that dependence changes these
operating characteristics. They test controller arithmetic, not HMC dynamics.
Every pair retains its own measurement and fresh verification, and R-hat/ESS
never contribute to membership.

## Independent-look sequential validation

The implementation follows inspected Gandy--Scott Algorithm 3/Theorem 3.1 and
the official mcunit code. Each look uses a fresh complete experiment; the
fixed multiplicity is applied once and sample count increases once after
look one. Focused tests cover the source equations, exact independent discrete
null enumeration, fresh streams, actual TF/TFP composition and invalid inputs.
The theorem remains conditional on independent look vectors and superuniform
component p-values; it does not confer coverage on tuning or posterior stopping.

| Device / arm | Rejections /128 | Exact 95% interval |
| --- | ---: | --- |
| CPU baseline | 8 | [.0274,.1194] |
| CPU no-op | 8 | [.0274,.1194] |
| CPU reversed Metropolis ratio | 128 | [.9716,1] |
| GPU baseline | 5 | [.0128,.0888] |
| GPU no-op | 8 | [.0274,.1194] |
| GPU reversed Metropolis ratio | 128 | [.9716,1] |

All arms have 128 valid experiments. CPU baseline/no-op reached later looks
28/39 times; GPU baseline/no-op did so 25/34 times. Every defect rejected at
look one. The lower power bound exceeds the declared .8 threshold on each
device for this particular defect. Three null arms fail the development
precision screen requiring an upper bound at most .10; they do not demonstrate
that nominal .05 size is wrong. Do not continue drawing trials merely until
that screen passes. Any larger null study needs a fresh fixed design.

## SBC sensitivity

Eight analytic normal-posterior designs each repeated 256 complete rank
experiments. Correct-null rejection counts are 9--14; their upper intervals
are below .10. Ignoring all data is detected in every trial. The M15 normal
design, 32 datasets with three rank draws, detects a .25-posterior-SD location
shift in 27/256 trials and a .5-SD shift in 64/256. Thus its earlier
nonrejection provides weak evidence against those errors. With 128 datasets
and three rank draws, the .5-SD detection count is 237/256, interval
[.887,.955]; with 15 draws it is 222/256, interval [.819,.906]. Both pass the
predeclared lower-.8 sensitivity screen for that larger shift. Neither design
has sufficient sensitivity to the quarter-SD shift.

These are independently generated analytic-output controls for the statistic.
They are not controlled-defect power measurements of the public whole-HMC fit,
and cannot replace that costly missing experiment. Counts across parameter,
radius and likelihood observables are multiplicity-adjusted within each
experiment; candidate siblings never create replications.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep compatibility rule unchanged | Complete public boundary measurements and independent energy/reference checks | None | Meaning of short-window screen versus stationary mean | Document empirical behavior | Strict band-membership confidence |
| Retain optional sequential experiment | Source equations, fresh-look tests, 128/128 defect detections per device | No invalid trials | Null-size precision and other alternatives | Broader predeclared studies when needed | Universal defect sensitivity |
| Downgrade small SBC nonrejection | Measured low sensitivity at M15 counts | No numerical veto | Whole-fit defect power remains unmeasured | Preserve limitation; proceed to geometry/reference matrix | Posterior calibration from a small rank test |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No corrupt numerical records or invalid complete trials |
| Statistically supported ranking | None; one declared defect meets the sensitivity criterion |
| Descriptive-only differences | Device frequencies, boundary qualification and runtime |
| Default readiness | No new tuning or posterior default promoted |
| Next evidence needed | More precise null controls, subtle whole-fit defects, difficult geometry and exact consumer references |

## Integrity, cost and terminal review

Frozen source identity is
`b5d2d662171ed61dab32a681693400abc1285ec5c13981a6118d1ce33624b54a`.
The audit checks all 488 Python files, 1439 numerical receipts and 18 saved
reference tensors, plus every candidate inventory. Trusted GPU runs used
verified memory growth and XLA; CPU references deliberately hid GPUs. The
suite designs and individual manifests preserve exact commands, seeds,+environment, target law, wall time and source. A partial pilot-review JSON
failed only at diagnostic boolean serialization; its corrected review is
recorded, with no numerical evidence replaced.

M16 charges 5316.570117719937 CPU seconds (including 900 inspection/accounting)
and 6994.797311380971 GPU seconds. Remaining campaign allowance is
86417.5601492732 CPU and 23255.818866750866 GPU seconds. The refreshed M17
12000/10000 and M18 6000/2000 ceilings fit without new authorization.

The strongest alternative explanation for favorable small-SBC results is low
power, now directly supported. The strongest limit of the boundary experiment
is its supplied identity geometry and fixed starts; it does not test automatic
preparation. Neither weak power nor a failed candidate invalidates the next
geometry/route/reference phase. M17 keeps posterior failures and global-mode
errors visible, preselects one member by identity, and gives no ranking or
default claim to its single-fit cells. Proceed after the recorded refresh.
