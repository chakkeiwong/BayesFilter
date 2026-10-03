# Plausible iAPF reconstructions: useful partial reference, incomplete reproduction

Four disclosed numerical reconstructions were tried. Delayed doubling with the existing diagonal log-quadratic fit and positive floor passes the practical reference screens at d5 and d10. It does not pass across the study: d80 needs too many particles, and d20 has observed conditional underperformance against the fully adapted filter. The d20 difference is uncertain, not statistically established inferiority. None reproduces the paper's full variability/resampling pattern. The constrained Eq15 fit depends on an imposed bound.

The numerical R core is unchanged. These are independent CPU R results. The user expressly authorized guesses; missing author settings limited identity claims but did not block execution. The plan was skeptically reviewed before implementation and all stages ran without phase approvals.

## Choices actually tested

| Choice | d10 calibration SD | Mean final particles | Disposition |
|---|---:|---:|---|
| Current QR, floor8, first-full-window doubling | 0.0389 | 2000 | Calibration only |
| QR, floor8, delayed doubling | 0.0634 | 1000 | Frozen for fresh validation |
| QR, floor4, delayed doubling | 0.0584 | 1000 | Calibration only |
| Bounded local Eq15, floor8, delayed doubling | — | — | Reached an imposed bound at backward time 99 |

Each completed d10 calibration cell has eight repeats on the same new data. The two surviving alternatives also completed two d80 calibration repeats each. Their observed average particle counts were 2000 (floor8) and 1500 (floor4); two repeats do not establish a difference. The predeclared d10 discrepancy rule nominated floor8/delayed; its score 1.457 versus 1.539 does not establish superiority. Neither calibration likelihood mean is a validation result.

## Fresh-data validation

| Dimension | Repeats | Mean Zhat/Z (95% bootstrap CI) | SD: reference / paper | Mean N: reference / paper | Mean resamplings: reference / paper | Practical screen |
|---|---:|---|---|---|---|---|
| 5 | 8 | 0.9934 [0.9577, 1.0247] | 0.0523 / 0.09 | 1000.0 / 1000 | 1.00 / 6.93 | Pass |
| 10 | 32 | 0.9995 [0.9751, 1.0255] | 0.0749 / 0.14 | 1000.0 / 1000 | 4.00 / 15.11 | Pass |
| 20 | 8 | 1.0738 [1.0037, 1.1492] | 0.1146 / 0.19 | 1000.0 / 1000 | 9.50 / 27.61 | Conditional heuristic veto |
| 80 | 16 | 0.9588 [0.8706, 1.0549] | 0.1983 / 0.35 | 1937.5 / 1142 | 37.56 / 71.88 | Particle count too large |

All intervals condition on one fresh simulated observation sequence per dimension. The paper used 1000 repeats per dimension and different, unavailable observations. These small samples cannot establish rare-tail behavior or a full table reproduction. The practical screen was explicitly broad (mean CI inside [.8,1.2], SD upper CI at most twice published, mean N at most 1.5 times published). The prior stricter [.9,1.1] mean-CI screen passes at d5/d10 only. Literal variability/resampling agreement fails in every checked dimension. New-setting d40 validation was not run: the eight-repeat cell and reporting reserve exceeded the remaining allowance. Earlier d40 evidence uses the old doubling rule.

## Heuristic and numerical checks

The constructed adversaries were BPF10000, fully adapted APF5000, and SIS10000; Kalman supplied exact likelihoods. Original-prefix ratio errors were evaluated separately on ordinary and top-decile innovation observations. All methods were run on identical data with recorded seeds; comparisons retain the replicate, not individual times, as the bootstrap unit.

At d20, ordinary-observation MSE was 0.026963 for the selected iAPF reconstruction versus 0.020977 for FA-APF. The paired difference interval is [-0.019608, 0.035869]. This observed loss triggers the predeclared conservative promotion veto; it does not show a statistically reliable ranking, invalidate the likelihood identity, or reject the iAPF idea. The other dimension/situation comparisons show no observed heuristic veto. Full conditional tables and paired intervals are in summary.json.

The d10 terminal variance difference against FA-APF has bootstrap interval [-0.04478,-0.01335], with both mean-accuracy checks passing. This is evidence conditional on that dataset and unequal algorithmic work, not overall superiority or published-runtime replication. High-dimensional BPF/SIS and some FA-APF cells severely underestimate likelihood in these finite samples; their near-zero empirical variance must not be interpreted as accuracy.

All 40,000 heldout QR fits and 6,400 heldout Gaussian-limit tail checks pass. The minimum dimension-specific relative precision margin is above 0.306. This is a checked tail diagnostic, not a universal variance guarantee for every adaptive run. Eleven existing tests pass, along with new exact-Gaussian no-fire, active-bound, and actual-consumer wiring checks. A terminal mutation check confirmed missing prefix evidence and a failed tail cannot be accepted by the report inspector.

## Why particle doubling persists

The saved d80 controller histories give a six-estimate CV between 0.5172 and 0.7439 at l=6; all exceed tau=.5, and fifteen also satisfy the nonmonotonicity condition and double. The last five estimates alone give CV below .5 in fifteen of sixteen histories. This is explanatory replay only: the early estimate influences the controller, while the CV mixes estimates from different learned guides and does not directly measure final-guide variance. A shorter-window trial would change the paper's printed rule and must be labeled an extension. No such change was fitted, selected or validated here.

The constrained Eq15 optimizer actually converged (code zero) but used an arbitrary active constraint. Its absolute loss and relative residual both decreased. Calling this numerical nonconvergence or a proof of bad filtering would be wrong; it specifically failed our predeclared boundary-independence screen. The full failing cloud and parameters are saved. See mathematical-interpretation.md for the derivation and source limits.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain delayed QR as a partial debugging reference | Practical screen passes d5/d10 | No numerical veto there | Few datasets/repeats; changed objective | Use checked components as independent diagnostics | Original Eq15 or five-dimension replication |
| Withhold study-wide reproduction | d80 particle budget fails | d20 observed conditional heuristic veto | d20 interval spans zero; d40 new setting unchecked | Fresh d20 confirmation, d40 completion, then controlled guide/controller trials | Failure of the iAPF theory or author code |
| Do not nominate the local Eq15 guess | Interior-solution criterion fails | Active arbitrary constraint | Other bounds/solvers untested | Preserve case; test a principled fitting prescription separately | All constrained Eq15 approaches fail |

## Inference status

| Item | Status |
|---|---|
| Hard validity screen | QR validation is finite, complete, and passes recorded fit/tail checks; local Eq15 fails its boundary screen |
| Statistically supported ranking | Conditional d10 terminal variance evidence against FA-APF only; no overall method ranking; d20 inferiority not established |
| Descriptive differences | Particle counts, resampling counts, small calibration scores, observed d20 loss, timing |
| Default readiness | Not evaluated; no algorithmic or backend default changed |
| Next evidence | More independent data/repeats for d20; d40 at the new setting; larger conditional variance study and an explicitly specified Eq15 numerical fit |

## Terminal review and provenance

Strongest alternative explanation: observations differ from the paper and small samples miss rare likelihood weights. Calibration ranking can reflect Monte Carlo noise. The known objective difference is also substantive. Fresh multi-dataset comparisons could overturn the d20 veto; a specified source-consistent fit could change both variance and resampling. The weakest empirical evidence is the eight-repeat d5/d20 comparison and absent new-setting d40. The final-count interpretation of Table 2 is disclosed in the mathematical note.

Source: cached Guarniero--Johansen--Lee paper, Section 5.1 equations (15)--(16), Algorithm 4 and Section 5.2 Tables 1--2; recovered TeX lines 517--546, 655--698, 760--807. The artifact manifest records paper SHA, Git commit, eight captured sources, CPU-only R environment, actual commands, seeds, output hashes and per-attempt wall time. There were ten filter launches: nine completed, one predeclared candidate rejection, no infrastructure failure or retry. No worker remains running.

Accounted budget: 795.381162 of 868.022749 summed worker/report seconds; 72.641587 remain. This includes the two saved-history inspections and report mutation check. The remaining balance cannot cover the next planned d40 cell plus its reporting reserve. No earlier allowance was reused.
