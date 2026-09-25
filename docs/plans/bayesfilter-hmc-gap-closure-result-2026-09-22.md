# HMC gap-closure execution and remaining limits

The reviewed M25 repair is implemented and its numerical experiments are
complete. It fixes an unconditional posterior-count ceiling, the validation
harness's count/estimator option wiring, and misleading unavailable-member
accounting. The enlarged window/count allocation passes the predeclared screen
on the difficult exact AR(1) case. Six supplied-map GPU fits preserve every
verified member, and the complete 272-fit public-HMC inventory has been audited.
These results support specific repairs; general HMC stopping calibration,
partial-map posterior usability and subtle full-fit defect power remain open.

Plan: [reviewed continuation](bayesfilter-hmc-gap-closure-continuation-2026-09-22.md).
Evidence root: `artifacts/hmc-repair-master-2026-09-16/m25-r1/`.
The [terminal audit](artifacts/hmc-repair-master-2026-09-16/m25-r1/terminal-audit-r1/result.json)
checks frozen sources, streams, exact-pair receipts, all sibling inventories,
stored tensor checksums and estimator agreement with the actual controller.
Later harness checks have separate source receipts; they do not relabel these
frozen experiments.

## Implemented and checked

The shared public posterior configuration and exact-transition configuration
now allow an explicit finite `max_results_per_chain`, requiring a nonempty
`count_budget_reason` above 10000. Defaults and default serialized payloads
remain unchanged. All maxima must fit the declared limit. The larger allocation
is part of the posterior/checkpoint identity and does not change thresholds,
seeds, numerical computations or tuning membership.

Validation designs forward `options.posterior_count_budget` and validate both
posterior and fixed-comparator counts against it. They also forward the existing
`options.posterior_precision_method`, retaining lugsail if omitted. The interval
diagnostic records its declared estimator and uses the same method as the
controller being calibrated. Unknown methods and malformed budgets fail early.

Reporting now separates requested-but-unavailable members from siblings that
were deliberately unassessed. A separate total records every member without
posterior output. Missing selected fits still count against the full replication
denominator. Frozen historical reports retain their original fields; the audit
explains their older semantics instead of rewriting them.

Tests cover actual public tuning, member export/reload, 12000 warmup and retained
draws, fixed comparison, durable restart, invalid budgets, estimator choices,
multiple model targets and documentation contracts. An inactive-option replay
of two original smoke trials preserves all 55 serialized tensor files exactly
and all numerical reports after excluding elapsed time and output directories.
That replay is engineering evidence, excluded from calibration denominators.
The final inventory contains **203 distinct passing tests** across 411 passing
test executions, including repeated checks after edits. The
[final verification record](artifacts/hmc-repair-master-2026-09-16/m25-r1/verification-final.json)
preserves per-suite counts and source hashes; totals are not summed as though
repeated tests were independent coverage.

The official book's `ch25_diagnostics.tex` and agent reference explain readiness
window information, target-specific count planning and the explicit count
exception. `docs/main.tex` compiled successfully; the changed rendered pages
were inspected. The book remains the scientific guide and the Markdown file
the API reference. Neither becomes a second tuning procedure.

## Actual-controller calibration

The first two cells use the existing autocorrelation estimator with the original
1000-transition recent window and 10000 warmup/retained ceilings. The repair
uses the same transition, dispersed starts, estimator, MCSE target 0.05 and
health/R-hat thresholds, but a 20000-transition window, 20000 minimum warmup,
40000 warmup cap, 40000 minimum retained and 80000 retained cap. These are
derived/development allocations for this exact law, not universal defaults.

| Cell, 400 fresh trials each | Posterior checks passed | Warmup caps | Retained caps | Stopped intervals covering the mean / all 400 | Independent fixed intervals covered / 400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stationary AR(1), rho .98, original counts | 137 | 46 | 217 | 332 | 382 |
| Dispersed AR(1), rho .995, original counts | 0 | 396 | 4 | 4 | 373 |
| Dispersed AR(1), rho .995, repaired counts/window | 394 | 0 | 6 | 389 | 382 |

All 1200 trials completed without transition health vetoes or implementation
failures. Original-count delivery and unconditional stopped-interval screens
fail. The repaired cell's exact pointwise 95% interval is [0.96764, 0.99448]
for delivery and [0.95133, 0.98619] for coverage, above the declared .90 lower
floor. Every interval was available. Its six retained-cap outcomes failed the
posterior R-hat check; none failed the precision check. They remain failures
of posterior delivery. All three exact Gaussian oracle screens pass.

For four stationary unit-variance AR(1) chains, the long-chain pooled mean
variance is approximately `(1+rho)/((1-rho)*4*n)`. At rho .995 this is
`399/(4*n)`, implying about 39900 draws per chain for MCSE .05. A 10000 cap was
an inadequate typical precision allocation. Separately, a short recent window
contains little effective information for readiness. The repair addresses both
mechanisms. It does not establish which contribution dominates or constitute
a same-cost method comparison. The .90 adequacy floor is weaker than a nominal
95% coverage theorem; repeated stopping is not made anytime-valid by this test.

## Supplied-map geometry and GPU execution

For the scale-3 funnel with two children, let `v=3*z0`,
`x_i=exp(s(v))*z_i`, and `r(v)=2*s(v)-v`. Including the map Jacobian gives
`U(z)=z0^2/2 + exp(r)*sum(z_i^2)/2 - r`, up to a constant. Exact whitening
sets `s(v)=v/2`, so the Hessian is identity. The tested partial maps have bounded
`s(v)`, making their child curvature `exp(r)` unbounded as `v` tends to negative
infinity. Analytic score/Hessian, density/Jacobian and independent derivative
tests agree. This demonstrates residual geometry; accepted states cannot
identify an unrecorded rejected trajectory's first failing leapfrog step.

All six GPU/XLA fits used trusted host GPU 1 with memory growth verified before
logical initialization, frozen source r2 and inverse-mapped common model starts.
The audit checked 680 measurement/verification receipts and all 471 candidate
records. All 19 verified members remain retained; six were assessed under the
predeclared first-member rule and 13 remain unassessed.

| Supplied map | Seed suffix | Verified members | Selected posterior outcome |
| --- | ---: | ---: | --- |
| Exact | 2251 | 3 | 2000 warmup / 10000 retained; precision cap, no health veto |
| Exact | 2252 | 6 | 2000 warmup / 10000 retained; precision cap, no health veto |
| Partial | 2251 | 3 | Nonfinite proposal diagnostics during warmup |
| Partial | 2252 | 1 | Nonfinite proposal diagnostics during retained sampling |
| Partial-half | 2251 | 4 | Nonfinite proposal diagnostics during retained sampling |
| Partial-half | 2252 | 2 | Nonfinite proposal diagnostics during warmup |

The supplied-coordinate tuning and retention checks pass. The four selected
partial-map posterior outcomes are rejected on numerical health. The two exact
controls meet numerical health but miss their model-quantity precision request.
No sibling is removed because another member's posterior fails. These six
development cells establish neither a reliability rate nor a ranking. The
appropriate geometry follow-up is a better supplied map under a new scope, or
a predeclared assessment of preserved siblings; a third blind epsilon sweep
has no demonstrated justification. Learned-map quality remains upstream work.

## Complete public-HMC confirmation

All 272 frozen M21 fits preserve source `f9c86f41a`; they cannot certify later
edits. The audit checked 42101 numerical evidence records and linked receipts.
All 5397 verified members remain recorded, including 5127 unassessed siblings.

| Model | Planned/completed fits | Verified members | Selected posterior checks passed | Selected caps or absence | Health vetoes |
| --- | ---: | ---: | ---: | --- | ---: |
| Gaussian | 128/128 | 2878 | 119 | 7 warmup caps, 2 retained caps | 0 |
| Beta-binomial | 128/128 | 2236 | 126 | 2 fits have no verified L=3 member | 0 |
| Rotated Gaussian | 8/8 | 135 | 0 | 8 retained caps | 0 |
| LGSSM location | 8/8 | 148 | 8 | None | 0 |

Selection was the declared first candidate ID at L=3, without posterior or truth
selection. Two beta-binomial fits retain other verified L values but supply no
selected posterior under that rule. Those missing fits remain in the denominator.
Gaussian stopped mean coverage is 116/128 and 117/128; median coverage is
118/128 and 116/128. All fail the .90 lower-endpoint screen, although their
independent fixed arms pass it. Beta-binomial stopped mean coverage is 122/128
(lower endpoint .90076), but median coverage is 120/128 (lower endpoint .88056);
the complete quantity screen therefore fails. Eight-fit groups are descriptive
stress evidence only. Posterior allocation and member usability still need
target-specific validation; these outcomes do not invalidate tuning membership.

The beta-binomial worker wrote its complete results before an extended teardown
with observed resident memory near 90 GiB. After independent audit and a final
60-second grace, it was terminated. Worker exit -15 and coordinator return 1
remain preserved; the final files are unchanged and all worker time is charged.
This is completed numerical evidence with abnormal process shutdown, not normal
execution success. No confirmation rerun is needed. The suspected framework
lifetime problem requires diagnosis before another large repeated-fit batch.

## Cost, boundaries and next work

Complete baseline/no-op/shift cost profiles consumed 238.03, 230.46 and 227.55
CPU worker-seconds, including startup and profiling. Tuning dominates, with
about 54 seconds of checkpoint-writing cumulative time in the first two fits.
The shift arm retained 16 members but none at L=3, so it supplied no selected
posterior. This is cost/activation evidence, not a successful shifted-posterior
fit or a power estimate. Even eliminating that observed writing cost cannot
make repeated 384-fit SBC experiments affordable within the remaining phase
allocation. Broad refactoring without a measured correctness/cost contract is
not justified; bounded process lifetime and checkpoint cost are concrete targets.

The exact nine-parameter, 48-observation MacroFinance bootstrap bundle is absent
at both supplied roots. The consumer explicitly says the old run is consumed
and must not be rerun. Fresh target/source qualification, prepared data, priors
and coordinates are required. The older full-joint MIDAS reference is a separate
missing-input gap; block-factorized fits cannot replace its matching reference.

The failed larger-count pilot exposed the old hard cap before any transition
and remains charged. The first final-audit attempt incorrectly counted passed
measurement receipts as fresh verifications; the audit predicate was corrected
to require the verification stage. Both failed attempts are preserved. Neither
is evidence of a sampler failure, and no numerical result was replaced.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep explicit count/estimator options and accounting repair | Real-route tests and default parity pass | No engineering veto | Scope of stochastic calibration | Use them in declared posterior designs | New default or universal count |
| Close the exact AR(1) allocation cell | Delivery and coverage screens pass | Six retained caps preserved; oracles pass | Transfer to HMC and other laws | Fresh public-HMC calibration | Anytime-valid coverage |
| Close supplied-map GPU engineering cell | Starts, verification, retention and GPU checks pass | Four selected partial posteriors fail health | Unassessed siblings and residual tails | Better map or fixed sibling assessment | Every partial map is usable |
| Keep general posterior calibration open | Gaussian and beta all-quantity screens fail | Caps/absence remain in denominator | Model quantities and member dependence | Target-specific count/window/estimator pilot | All tuning candidates failed |
| Defer repeated full-fit power confirmation | Cost measured; adequate repeated inventory unaffordable | Missing selected shift output | Power and process lifetime | Profile/repair cost, then size a fresh inventory | Small-study nonrejection proves sensitivity |
| Leave exact consumer integration input-dependent | Matching bundle absent | Scope/input continuation veto | Consumer qualification/reference | Execute newly qualified bundle when supplied | Substitute-target validity |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Partial-map selected posteriors have nonfinite proposal diagnostics; missing outputs and caps remain explicit |
| Statistically supported ranking | None; only the predeclared pointwise adequacy screens are assessed |
| Descriptive-only differences | Runtime, method differences, local curvature, eight-fit controls and six GPU development fits |
| Default readiness | No count, estimator, tuning or transport default promoted |
| Next evidence needed | Public-HMC stopping transfer, supplied-map/sibling precision, affordable full-fit power, matching consumer inputs |

The strongest alternative explanation for the successful count repair is its
special exact Gaussian transition and favorable finite set of replications.
A failure under unchanged criteria on fresh HMC or non-Gaussian targets would
overturn broader adequacy, which is currently unproved. The weakest evidence is
the small number of GPU/map cases and the unobserved rejected proposal paths.
The failed posterior candidates motivate the next bounded experiment; they do
not reject the supplied-whitening approach or the research direction.

M25 consumed **5106.859953 CPU / 1596.479894 GPU worker-seconds**, including
failed attempts and final option/reporting regressions, within its 12000/4200
ceilings. The combined September 22 campaign has **84512.572082 CPU /
84803.520106 GPU seconds remaining**. No M21/M25 worker or reservation remains.
The [final ledger](artifacts/hmc-repair-master-2026-09-16/m25-r1/reconciliation-terminal-r2.json)
counts each completed worker once; GPU time includes its host work. Commands,
environment, seeds and source hashes are in each attempt's manifest. The earlier
terminal-named ledger is preserved as an interim snapshot before the final
estimator bridge tests.

The [next-phase design](bayesfilter-hmc-post-m25-next-phase-2026-09-22.md)
records ordered repairs, evidence requirements and the genuine limits. The
[master program](bayesfilter-hmc-repair-master-program-2026-09-16.md) and progress
record now point to it. Numerical failures rejected individual posterior
candidates; the bounded plan itself completed. Remaining scientific claims
need their own evidence and the exact consumer claims need missing inputs.
