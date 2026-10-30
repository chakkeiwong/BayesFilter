# Zhao–Cui replication and numerical scores: actual evidence

The publication comparison is complete. The author algorithms produce finite
full-horizon smoothing proposals in all eight completed paper/source runs, but
the local experiments do not reproduce every published ESS level or the
nonlinear predator–prey advantage under the reconstructed paper settings.
Separate released-driver runs recover that qualitative PP advantage.

On our fixed T20 targets, likelihood estimates are close to independent
bootstrap references. Numerical scores require a much smaller neighborhood
than the initial regression design: wide SIR perturbations concentrate almost
all importance weight on one path. Shrinking the neighborhood repairs this
local overlap problem, while sampling uncertainty remains. No score is certified as an oracle.

Plan: [replication and score plan](../plans/zhao-cui-publication-replication-and-score-20261004.md).
Evidence root: `docs/plans/artifacts/zhao-cui-publication-replication-20261004/`.
Branch `sqmc-development`, initial HEAD `0b91a64f6d38009b5f081d7981f51fab958425c0`.
Final budget snapshot 2026-10-05T11:47:00.485970+00:00: 47.004 of48 additional aggregate job-hours
used; 0.996 remains. The queue is finished and no further full run is authorized
under this campaign. Prior4.0023h are excluded. Max2 full
CPU jobs, two threads, GPU intentionally hidden. These references do not
measure production GPU performance. Per-run manifests preserve commands,
source and observation hashes, seeds, environment, time and result paths.

## Published and local path ESS

Each draw contains10,000 paths. Forty draws below condition on one fitted TT
and one dataset; they are not40 independent fitting/data experiments. Published
values are digitized vector-figure medians. The paper's PP Figure17 stops at
T10; it must not be rescaled to20.

| Model/settings | T | Local ESS (%) | Local repetition scope | Published median (%) |
|---|---:|---:|---|---:|
| PP paper linear | 10 | 88.388 | one draw | 31.243 |
| PP paper nonlinear | 10 | 81.915 (IQR80.748–82.554) | 40draws | 80.706 |
| PP paper linear | 20 | 84.396 (80.189–86.448) | 40draws | no plottedT20 value |
| PP paper nonlinear | 20 | 75.184 (72.006–77.119) | 40draws | prose approximately40 |
| SIR paper rank10 | 20 | 11.302 (8.968–14.138) | 40draws | 15.402 |
| SIR paper rank20 | 20 | 36.546 (32.708–39.194) | 40draws | 57.845 |
| SIR paper rank40 | 20 | 50.081 | one draw | 69.335 |
| PP released linear | 20 | 33.843 | one draw | different settings/data |
| PP released nonlinear | 20 | 60.291 | one draw | different settings/data |
| SIR released rank40 | 20 | 25.491 | one draw | different settings/data |

Full curves, weighted/unweighted path summaries and source hashes:
`report-publication-02`, `report-released-driver-02`, `published-figure-values-01`.
Both refreshed ESS figures were visually inspected. Source/paper differences
include the RK4 fourth stage, initial-state generation, PP parameters/priors,
subsequent ALS sweeps and the nonlinear reference domain. The data realization
is also different from the unavailable MATLAB original. Joint changes prevent
causal attribution to any one difference. The SIR rank ordering is descriptive,
not a statistically established ranking.

## Same-target likelihoods

Conditional rank20 fits use seeds2/17/29 and fixed current observations. Their
10,000-path log likelihoods, alongside the independent524,288-particle, four-
replication bootstrap estimate, are:

| Model | TT seed2 | TT seed17 | TT seed29 | Bootstrap log likelihood | Bootstrap MCSE |
|---|---:|---:|---:|---:|---:|
| PP | -97.344056298 | -97.344035221 | -97.344030690 | -97.342973851 | .006574815 |
| SIR d18 | -678.067711876 | -678.045207144 | -678.052340241 | -678.077466314 | .014864040 |

These are the same observations and model laws. The saved PP bootstrap point
rounds r and s to float32; a paired full-path check measures log-likelihood
shift-7.3717e-7 with conditionalSE3.20e-9, much smaller than its displayed
MCSE. This measured likelihood shift is not a global bound or a score-shift
bound. The bootstrap reference has unmeasured finite-particle bias. Close
likelihood values alone do not establish close derivatives.

The source's returned `lml` diagnostic is a mean of finite log importance
weights, not the log of their mean. It is wrong relative to a claim that this
returned value is the marginal log likelihood. The paper's ESS experiments do
not depend on that diagnostic. Our reference uses logmeanexp over allN paths
and the cumulative proposal log density from `models/full_sol.m:171–172`.
Nonlinear `pre_sol` density offsets permit normalized-weight diagnostics but
not absolute likelihoods without their missing constants; only the validated
linear conditional route supplies the likelihood/score calculation here.

## Numerical scores and uncertainty

The score uses a full quadratic fitted to canonical TF/XLA log-joint values
on common author-TT paths: no analytic score or autodiff is supplied. There
are 1,024 antithetic training points, 256 separate held-out points and
coordinate central differences at every full radius. Twenty equal deleted path
groups measure conditional sampling error. Independent fits estimate
between-fit variation. Regression residuals do not estimate either uncertainty.

The final report is `report-score-final-01`. It combines the completed PP
10,000-path small-radius run, the completed PP 100,000-path run, the completed
SIR rank-20 wide-radius run, and the completed SIR rank-40 small-radius run.
The arithmetic mean is taken over separately fitted finite-sample derivatives;
it is not a derivative of a likelihood pooled across proposals. The smallest
radius in each completed series is shown below, while every radius and fit is
retained in the report CSV files.

### Predator--prey

| Paths | Rank | h | Score `(r,K,a,s,u,v)` | Between-fit 95% t halfwidth | Conditional mean SE | Minimum ESS | Held-out RMS |
|---:|---:|---:|---|---|---|---:|---:|
| 10,000 | 20 | 0.00125 | `(-33.304023,-0.55864759,0.020413745,4.5071742,-8.7160499,10.847440)` | `(0.398694,0.0358593,0.000206453,0.0768389,0.0141339,0.0118746)` | `(0.114709,0.00589052,0.000117006,0.0207010,0.0237740,0.0296156)` | 9,824 | `2.945e-6` |
| 100,000 | 20 | 0.0003125 | `(-33.051124,-0.55078735,0.020627942,4.4686647,-8.7462792,10.883935)` | `(0.187941,0.0147903,0.000094596,0.0283026,0.0265643,0.0280803)` | `(0.0330587,0.00174556,0.0000323612,0.00586542,0.00661567,0.00828419)` | 99,892 | `4.60e-8` |

The increase from 10,000 to 100,000 paths reduces the conditional sampling
errors and the held-out residual. The score coordinates also move toward the
independent 524,288-particle bootstrap values
`(-32.905236,-0.543943,0.02052955,4.487344,-8.732397,10.865452)`, but the
remaining differences include finite-particle bias, shared proposal support
and radius error. These measurements are a substantially better numerical
reference than the first 10,000-path wide-radius run; they still do not certify
an oracle.

### SIR d=18

The completed rank-40 small-radius run gives, at `h=0.00015625`,
`(112.685217,-65.165591,5.576446)`, with conditional path-jackknife SE
`(12.4285,5.4720,0.08618)`, minimum ESS `4,549/10,000` and held-out RMS
`7.24e-7`. It is a single independently fitted proposal, so there is no
between-fit interval. The values are close to the independent bootstrap
`(106.313211,-65.960310,5.586350)` at the scale of the reported sampling
uncertainties, but this remains one fit and not an oracle certification.

The original rank-20 wide-radius design remains a failure diagnostic: at
`h=.01` its mean `(197.387,-113.825,5.4501)` has between-fit halfwidth
`(238.907,109.711,2.9922)`, minimum ESS `1.17--1.34/10,000`, and held-out RMS
`.766--.846`. The predeclared rank-20 small-radius queue reached its 3,600 s
limit after writing eight of nine fit/radius estimates; it is incomplete and
has no aggregate promotion status. This timeout consumes budget and is retained
as evidence of the cost of the three-fit SIR calculation. It does not invalidate
the value or quadratic-fit harness.

The SIR rank-40 result therefore repairs local overlap relative to the wide
rank-20 regression, while the remaining uncertainty is sampling and
independent-fit replication. No radius, rank or path count was selected by
agreement with the bootstrap or with an LEDH score.

## Engineering and manuscript

One PP transition overflowed at time5 while sampled paths/proposal densities
were finite. Exact-prefix replay reproduced24 arrays and observations. The
optional100/200-digit replay evaluates the same RK4 equation and confirms an
extremely small density whose weight rounds to zero. All49,999 healthy
transitions remain bit-identical, allN paths are retained, and each fallback
is recorded. Six tail tests pass. This is an explicit local arithmetic
extension, not an original-author algorithm or production change.

A separate extreme-tail defect in the pinned Octave normal-quantile shim is
unsubstituted and was not the transition-overflow cause. Exact MATLAB execution
is not claimed. Original author files remain immutable. Incremental reporting
restores fitting RNG; PP/SIR smokes preserve16 fitted arrays exactly.

All22 focused quadratic, path-value and exceptional-tail tests pass. Adding
jackknife/prefix diagnostics changes the T2 smoke scores by exactly zero.
TF/Octave path-joint parity errors there are1.78e-14(PP)/3.98e-13(SIR);
full-T20 PP baseline parity is1.42e-13. These checks establish mechanics and
target agreement, not statistical accuracy.

The monograph source `ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`
renders as Chapter39. It derives the fixed-proposal importance identity, full
quadratic, parameter units, symmetric-design error, conditional jackknife and
source mean-log-weight distinction, and now reports the publication comparison.
The final build (pass 19, using pdflatex/bibtex because latexmk was unavailable)
compiles without undefined references, citations or rerun warnings. Physical
pages 401–402 of the 612-page PDF and the final PP/SIR score figures were
inspected. The build metadata records the actual page count; the old 581-page
statement was wrong. Human readability review is pending.

## Decisions and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain paper/source comparison | All8 full-horizon runs complete | Finite outputs, unchanged source | Missing original data; one fit per arm | Report actual curves/settings | Exact-number replication or paper refutation |
| Retain optional PP arithmetic repair | Healthy outputs identical; precision replay agrees | Invalid targets still reject | Other exceptional tails | Keep diagnostics | New production default |
| PP numerical reference remains under validation | Radius convergence and N100000 precision check | No finite/parity veto; shared support and finite-particle bias remain | Independent higher-N/rank replication | Preserve PP100000 and report all radii | Oracle or LEDH improvement |
| Reject wide SIR regression as reference; retain rank-40 local repair | Off-center overlap fails; rank-20 small-radius run timed out | Wide route vetoed; timeout is incomplete evidence | Single rank-40 fit and path sampling | Independent rank-40 fits or higher-N paths | Oracle or research-direction rejection |

| Inference category | Finding |
|---|---|
| Hard veto evidence | Original PP arithmetic failure localized/repaired; current wide SIR neighborhoods lose overlap |
| Statistically supported ranking | None; no filter ranking tested |
| Descriptive-only differences | Publication ESS, runtimes, radius/prefix changes and few-fit comparisons |
| Default readiness | Not evaluated; independent CPU reference only |
| Next evidence needed | Independent rank-40 SIR replication or higher-N paths, exact source-data tie-out, and score-shift checks |

The strongest alternative explanation for publication differences is the joint
change in data/settings. For score differences, finite sampling, proposal
support and regression truncation can all matter; a tiny residual addresses
only truncation of the computed finite-sample function. Agreement across
larger samples/ranks/independent fits would weaken those explanations, while
persistent discrepancies would require further reference work. No categorical
claim that Zhao–Cui is wrong or that LEDH has been improved follows.
