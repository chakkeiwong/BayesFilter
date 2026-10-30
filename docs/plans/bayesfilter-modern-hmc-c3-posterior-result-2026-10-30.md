# C3 seven-candidate posterior assessment: terminal result

30 October 2026. The seven-member assessment completed at **05:03:49
Asia/Shanghai**, with supervisor exit code 0, after 18,347.479901695042 seconds
(5h 05m 47s). Four members passed both the sequential posterior checks and
the independent-reference agreement screen. The master program is not complete.

## Question, target and evidence

The [reviewed C3 plan](bayesfilter-modern-hmc-c3-reference-continuation-2026-10-29.md)
asks whether every retained candidate can satisfy declared posterior checks on
the same frozen 48-observation, nine-coordinate MacroFinance Phase9 OU target.
This run assesses all seven verified randomized-HMC candidates; it does not
rank methods or select a new tuning representative. The comparator is the
independently evaluated covariance-Kalman posterior reference described in the
[reference result](bayesfilter-modern-hmc-c3-reference-result-2026-10-29.md),
including the same chart prior and prediction-first timing.

Each member used four chains, 2,000 discarded warmup transitions per chain,
and cumulative retained sampling in 1,000-transition chunks. Checks cover 31
quantities: nine coordinates, their standardized squares, nine smooth-tail
functions, two state standard deviations and two lag-one autocorrelations.
The predeclared screens require modern R-hat <=1.01, bulk/tail ESS >=400,
lugsail MCSE/SD <=.02, and reference agreement
`(absolute error + 3*MCSE + reference uncertainty)/reference SD <=.1`.
These are operational criteria inherited from the plan, not calibrated
simultaneous or anytime probability guarantees.

The exact evidence root is
`artifacts/modern-hmc-implementation-2026-10-07/N5/c3-posterior-20261029-01`.
The terminal records are `posterior-01/manifest.json`,
`posterior-01/evidence/result.json`, `posterior-01/evidence/posterior/result.json`
and `continuation-01/status.json`. Individual member directories preserve
chunk evidence and final diagnostics.

## Member results

| Maximum L | Epsilon | Retained draws/chain | Maximum R-hat | Maximum lugsail MCSE/SD | Maximum reference screen | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 3 | .4 | 10,000 | 1.002311 | .035256 | .173487 | Precision and reference agreement unmet |
| 5 | .4 | 10,000 | 1.001719 | .022209 | .088999 | Precision unmet; reference screen passed |
| 9 | .4 | 4,000 | 1.001735 | .019553 | .082581 | All declared checks passed |
| 13 | .4 | 3,000 | 1.001566 | .019640 | .079624 | All declared checks passed |
| 18 | .4 | 4,000 | 1.001862 | .016207 | .070016 | All declared checks passed |
| 25 | .2 | 2,000 | 1.001045 | .018791 | .101648 | Sequential checks passed; reference agreement unmet |
| 25 | .4 | 3,000 | 1.001769 | .019229 | .090416 | All declared checks passed |

All seven passed warmup, final R-hat and final bulk/tail ESS checks. None
recorded a numerical hard veto. L=3/.4 reached the retained cap with seven
precision failures, including omega, its square and smooth-tail function,
the lower process-noise Cholesky coordinate and its smooth-tail function,
and both autocorrelations. L=5/.4 reached the cap with precision failures
for omega and its two derived functions.

L=25/.2 passed the sequential controller after 2,000 retained draws per chain
but failed the subsequent reference screen for `prior_square_S_lower_Pi_G`: .101648
against .1. The controller and reference verdicts are separate; the cohort
correctly records `delivered=false`. This is insufficient reference agreement
under the stated screen, not proof of sampler bias. Do not round the result
into a pass, relax the threshold, or rerun until favorable. Any further
assessment needs a prospective stopping/replication design that preserves this
failure and separates reference validation from internal stopping.

Tuning membership remains unchanged for all seven candidates. R-hat and
precision here are posterior-assessment criteria, not tuning requirements.

## Terminal audit and provenance

The terminal audit checked all seven result files and recomputed all **643**
recorded source/input SHA-256 hashes: none was missing or changed. Final
driver, cohort and observer counts agree on four passing members. Candidate
order and membership match the seven exported identifiers. All required
31-quantity reference screens are finite, and each delivery decision equals
the conjunction of sequential checks and reference agreement. Process inspection
at 15:36 Asia/Shanghai confirms the recorded observer, supervisor and worker
have exited. No restart is needed.

The numerical commit is `f9291ee4437aa8c00dabbe331bffc6871b574c5c`, with frozen
consumer and orchestration sources identified by the worker manifest. The
exact command, target/data dependencies and seeds are preserved in
`posterior-01/evidence/manifest.json`; the supervisor command is in
`posterior-01/manifest.json`. Environment: `tfgpu`, TensorFlow 2.20.0, TFP
0.25.0, float64, XLA enabled and TF32 disabled. Trusted GPU execution used
UUID `GPU-a1ea1946-07c0-8ed5-2ba1-d96f82c89cd3`, with memory growth configured
and verified before initialization. NUTS was excluded. The existing preparation
records preserve 66 focused tests and seven exact-source checks; this closeout
does not claim another test run.

The supervisor's inclusive charge is counted once, without adding child wall
time. Of the 98,215-second reservation, 79,867.52009830496 seconds were unused.
The remaining upper balance is **160,729.47980995933 seconds (44.6471h)**, with
preparation overhead still withheld. Separate SIR funds remain
7,363.064379416 seconds. Released reservations are not additional credit.

## Decision and next work

| Decision | Primary criterion | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Close this C3 assessment | All seven assessed; four passed every screen | No recorded numerical hard veto; three members fail at least one posterior screen | Finite-run precision, reference uncertainty and repeated-look calibration | Preserve results; continue C4 reliability analysis | Complete master, release readiness or universal posterior validity |
| Retain original tuning set | All seven remain verified | Posterior failures do not reject tuning membership | Scope-specific posterior adequacy | Keep posterior outcomes separate from tuning | R-hat as a tuning requirement |

| Inference status | Finding |
|---|---|
| Hard veto screen | No recorded numerical hard veto in this cohort; all 643 source/input hashes match |
| Statistically supported ranking | None; no comparative superiority claim is supported |
| Descriptive-only differences | Draw counts, runtimes, R-hat, ESS, MCSE and reference discrepancies |
| Default-readiness | Not established; candidates remain experimental |
| Next evidence needed | C4 precision/tail reliability and prospective replication; remaining integration and documentation work |

The result rejects posterior adequacy of three candidates under this finite
assessment; it does not invalidate the target, harness or research direction.
The next discriminating work is C4 estimator/tail reliability, including the
observed difference between internal stopping and reference validation. C2
wider consumer migration/P19, C5 independent-fit SIR and C6/N8 integration,
examples, official-book and release reconciliation also remain open.

Post-run skeptical review: the strongest alternative explanations are ordinary
Monte Carlo fluctuation, imperfect MCSE calibration and reference-proposal
coverage limits. The independent reference's refinement/randomization envelope
is empirical, not proof of global coverage. Fresh predeclared replications and
independent estimator checks could overturn an individual pass or failure.
Successful process exit, small R-hat and four passing members do not establish
a ranking or close these uncertainties.
