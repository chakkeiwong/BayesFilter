# Acceptance tuner robustness: corner cases, models and decision oracles

This is the concrete validation matrix for the
[acceptance repair program](bayesfilter-hmc-acceptance-decision-repair-plan-2026-10-02.md).
It adds specificity to the owner's execute instruction and subsequent question
about corner and ambiguous cases. A passing suite cannot guarantee arbitrary
target behavior. It must demonstrate correct decisions on controlled examples,
honest uncertainty, meaningful delivery on relevant models, and recoverable
execution without weakening numerical or statistical checks.

## Three independent questions

1. Does the statistical procedure control wrong decisions for its declared
   finite-trial quantity? Use controlled laws with known means; model names
   alone cannot answer this question.
2. Does the actual public tuner collect, classify and preserve evidence
   correctly? Use real models, both public coordinate routes, multiple
   candidates and the complete checkpoint/export lifecycle.
3. Does a qualified kernel support good posterior inference? Use separate
   posterior reference and convergence checks. Acceptance qualification is not
   a posterior-correctness certificate; high R-hat cannot become a tuning veto.

Every case declares which question it answers and its required outcome before
sampling. Inconclusive, rejected, preparation review, and resource deferred
are distinct outcomes. Tests that merely accept any outcome or merely observe
finite values cannot establish robust tuning.

## Controlled acceptance laws

These fixtures supplement actual HMC runs. They deliberately isolate the
statistical mechanism so that incorrect decisions have a known definition.

| Case | Construction | Required behavior |
| --- | --- | --- |
| Clear qualification | Bounded scores centered at .70; starts individually inside the declared [.55,.85] qualification band | High qualification delivery within a declared budget, with independent verification |
| Clear direction | Means well below .65 or above .75, with healthy trajectories in the execution layer | Correct epsilon proposal; new child, not retroactive editing of parent |
| Qualification boundary | Mean at .55/.85 and small offsets on either side | No confident false membership; exact boundaries may remain unresolved |
| Preferred boundary | Mean at .65/.75 but well within qualification bounds | May qualify without proving narrow-band equivalence; no obligatory endless extension |
| Heterogeneous valid starts | Means [.60,.68,.72,.80] | Preserve each mean; inequality alone does not imply invalidity or nonstationarity |
| Misleading pooled mean | Means [.40,.80,.80,.80] average .70 | No all-start qualification; report which start violates the declared condition |
| Opposing changes | Starts change in opposite directions with unchanged pooled mean | Per-start temporal diagnostics detect the changes; their role stays separate from acceptance admission |
| Single-start transient | Only one start's expected acceptance changes, eventually settling | Known finite-horizon expectation, not a stationary target substituted for it |
| Strong persistence | Bounded two-state chains at rho=0,.8,.95,.995 and -.8 | Trial-level error control without counting transitions as independent repetitions |
| Identical observed scores | A rare-event distribution happens to emit only .70 | No zero-width uncertainty or assumption of determinism |
| True deterministic score | Independently specified deterministic law | Arithmetic tie-out; do not infer this law merely from constant samples |
| Rare extremes | Mostly moderate acceptance with occasional zero/one scores | Bounds stay valid and reveal insufficient data; no Gaussian small-sample presumption |
| Shared start randomness | Correlated start-vector components within a repetition | Preserve covariance; independent unit is the whole vector |
| Duplicate repetition seed | Reuse a stream or a completed trial | Reject duplicated information; no artificial interval tightening |
| Sequential selection | Adaptive epsilons, many candidates, repeated looks, adverse fresh verification | Bound the probability of any incorrect verified member, with all attempts counted |

The numerical bands above are the repair plan's inherited test settings, not
universally optimal HMC bands. Drift sizes, rare-event probabilities, offsets,
repetition limits and seeds must be fixed in the executable cell configuration
before held-out calibration; exploratory results cannot choose favorable
holdout settings. Report decision errors, qualification delivery, correct
direction delivery, preparation alerts, inconclusive outcomes, work and wall
time, with uncertainty and original planned denominators.

## Real model matrix

Prioritize state-space targets. Gaussian and support-constrained examples are
small controls with useful independent references, not substitutes for those
targets. The repository fixtures and executable configuration inventory exist. The
checked-status section below distinguishes confirmations, negative regressions
and development pricing; none supplies a universal model guarantee.

| Model / existing fixture | Corner being exercised | Required evidence and expected scope |
| --- | --- | --- |
| Standard Gaussian and `ssm_campaign_location` (K0) | Positive control; small/large epsilon; resonant L; several surviving pairs | Exact density/score and posterior references; successful bounded tuning on nominated interior controls; retain all verified pairs |
| Correlated/scaled Gaussian | Ill-conditioning, bad initial scales, invariance under an equivalent affine chart | Compare matched physical starts and metric transformations; no claim that unprepared geometry must tune successfully |
| `ssm_lgssm_qr`, `ssm_campaign_interior` (K1) | Actual Kalman likelihood/score path and ordinary state-space parameter coupling | Public tuning, fresh verification, resume/export; independent filter/reference checks remain separate |
| `ssm_campaign_near_unit` (K2), `ssm_lgssm_near_unit` | Strong persistence, long memory, dispersed starts, long transients | Verify realized persistence in data/starts; the fixture name or persistence cap alone does not prove the case is difficult |
| `ssm_campaign_small_noise` (K3), `ssm_lgssm_small_noise` | Sharp likelihood curvature and large energy error from oversized epsilon | Supported lower-step child or explicit numerical failure; no direction from an invalid acceptance trace |
| `ssm_campaign_two_noises` (K4) | Process/measurement-noise tradeoff and weak identification | Do not confuse start heterogeneity with estimated sampling variance; agreement with declared numerical reference where available |
| `ssm_campaign_two_noises_long` (K5), `ssm_lgssm_long_horizon` | Cost growth, chunking, deadline and evidence-budget feasibility | Positive short matched fixture first; longer stress must defer or resume honestly without shortening the evidence requirement |
| `ssm_campaign_multivariate` (K6) | Correlated multi-parameter geometry, batch shape and costly compilation | Bounded mechanics/resource stress; no posterior-accuracy claim because the current fixture has no independent joint posterior oracle |
| `ssm_nonlinear`, `ssm_campaign_nonlinear` (K7) | Nonlinear filtering, sigma-point likelihood curvature and target-status checks | Public tuner with the declared approximate likelihood; do not call approximation error an HMC failure or the approximation an exact posterior |
| Exactly whitened funnel | Known coordinate map and Jacobian | Fixed-transport route matches analytic transformed target; positive tuning control |
| Residual-whitened funnel | Imperfect geometry with controlled remaining curvature | Independent per-L/epsilon repair, qualified or explicitly unresolved outcome under measured budget; supplied map is frozen |
| Centered funnel | Geometry outside the intended prepared positive-test scope | Bounded negative control; no requirement that epsilon/L search fix missing whitening |
| Separated mixture | Healthy local acceptance despite failure to move between modes | Tuning may qualify locally; independent posterior assessment must expose inadequate exploration |
| Student-t/Cauchy and transformed positive/simplex controls | Heavy tails, undefined posterior moments, constrained support, Jacobians | Acceptance means remain bounded; posterior mean/variance claims require their own moment assumptions; legitimate support rejection differs from numerical invalidity |

Use existing tiny state-space fixtures for frequent regression. Do not run a
full K0--K7 posterior campaign on every code change. K5/K6, long horizons and
full posterior-reference work belong to scheduled bounded validation. BGS
saved traces are immutable replay fixtures; fresh BGS trials are a later
target-specific integration, not a source for calibrating thresholds until
that one candidate passes.

Additional state-space variants should exercise very short observation series,
diffuse or poorly chosen initial state distributions, nearly singular innovation
covariances, and parameters close to their declared support boundary. Where the
filter API supports missing observations, include a missing-data block and the
all-missing limit; verify the likelihood against its declared observation model.
Unsupported missing-data or degenerate-covariance inputs should fail clearly at
the target boundary. They cannot become a request for the tuner to manufacture
a valid likelihood. These are proposed variants, not claims that every named
fixture or API already supports them. Use a Kalman reference, a target-status
oracle or an explicit unsupported-input expectation for each variant.

## Execution, identity and failure injection

| Injection / transformation | Required invariant |
| --- | --- |
| Reorder L candidates | All funded candidates processed; no early first-pass termination; compare assigned streams explicitly rather than demanding bitwise equality under changed seed identity |
| Reorder starts with labels/weights | Same scientific summaries up to the same permutation |
| Equivalent affine transformation | Same physical target and transformed kernel probabilities within declared numerical tolerance |
| NaN score or accepted-state inconsistency | Correct candidate/scope failure classification; favorable acceptance cannot override it |
| Health failure in discarded prefix | Veto preserved even though prefix is excluded from acceptance statistics |
| Low acceptance plus immobility | Parent remains unqualified; valid directional evidence can propose a smaller child |
| High acceptance plus recurrence | No automatic larger-step remedy for an unresolved trajectory pathology |
| Changing map, metric, start bank, horizon or source | No reuse of qualification under a different target/protocol identity |
| Odd T and uneven chunk boundaries | Every measured draw counted exactly once; reset boundaries never become movement/recurrence transitions |
| Interrupted chunk, trial, or post-trial receipt write | Resume same stream/state; one committed score; no duplicate cost or information |
| Invalid trial among good survivors | Preserve invalid outcome and planned denominator; no survivor-only qualification |
| GPU contention or worker resource failure | Recover under same numerical policy and total budget; no renewed statistical error allowance |
| Exhausted budget | Explicit unresolved/deferred state, never implicit rejection or pass |
| Checkpoint tampering or old policy receipt under new version | Fail identity/recomputation checks |
| High, missing or exception-producing R-hat | No change to tuning decisions; retain reporting-only provenance |
| Epsilon too small to change a float32 state, or at the stability boundary | Detect actual loss of movement or numerical invalidity; do not infer usefulness from acceptance near one |
| All initial proposals fail, or the useful pair lies beyond the search bounds | Bounded repair/refinement or explicit unresolved outcome; no claim that the target has no usable kernel |

Run interruption tests at deterministic boundaries, including the boundary
after persisting numerical evidence but before updating the controller. Test
replay in a fresh process as well as in-memory resume. Serial/threaded/batched
implementations need equivalence or distributional tests appropriate to their
declared stream layouts; different random streams need not be bitwise equal.

## Test the tests with deliberate defects

A robust regression suite must fail when a relevant mechanism is broken.
Use temporary test doubles or isolated mutations; do not alter campaign source
while a numerical checkpoint test is running, because source binding correctly
invalidates that experiment.

| Deliberate defect | Test that must detect it |
| --- | --- |
| Use the last tuned epsilon for every L | Two L values with opposite required step repairs, checked against each candidate's numerical metadata |
| Stop at the first passing pair | Controller fixture with several passing members, followed by numerical export of every verified member |
| Treat four different starts as four repetitions | Heterogeneous start means with the same pooled mean, and within-trial correlated start vectors |
| Treat identical observed scores as zero uncertainty | Rare-event law and short constant-score sequences; the interval must retain positive width |
| Multiply trajectory length at a new repetition rung | Fixed T/W assertions and independent reset/seed checks across cumulative trial extensions |
| Drop the final observations when T is odd | Independent arithmetic using a deliberately different last observation |
| Omit health checks on discarded draws | Divergence or invalid target/score injected only in the discarded prefix |
| Turn high or missing R-hat into a tuning veto | Same stored acceptance and health evidence with changed/failed R-hat reporting; membership is unchanged |
| Re-run a completed trial after a receipt-write crash | Identical numerical evidence identity, one admitted score, and unchanged numerical work count after restart |
| Reuse a completed or failed stream as fresh verification | Trial/stream provenance validation and retained-runner seed exclusion, including lost native calls |
| Trust caller-edited means or endpoints after recomputing an outer checksum | Reconstruction from every raw trial and comparison with the declared terminal verification trial |
| Spend the fresh-verification allocation on optional measurement extensions | A cap that funds precisely the first measurement and verification; the extra measurement must defer |

Wrong gradient signs, missing transform Jacobians and incorrect Metropolis
state selection also need numerical mutation tests against independent density,
score, reversibility and state-update references. These are sampler-validity
tests. Good acceptance alone cannot detect all of these defects.

Posterior assessment adds separate known-answer and simulation-based calibration
checks. Linear Gaussian fixtures support exact filter/likelihood and analytic
posterior checks where the parameters permit them; nonlinear fixtures require
an explicitly identified reference for the declared approximate likelihood.
Use repeated simulated data and posterior rank/coverage diagnostics where a
matched prior predictive simulator exists. Account for MCMC dependence in rank
checks and MCSE in moment comparisons. A target lacking a matched simulator or
independent reference remains a mechanics/resource test and cannot acquire a
posterior-correctness claim merely by being in the model catalog.

## Test tiers and pass criteria

* Fast regression: deterministic protocol validation, exact/reference arithmetic,
  decision-boundary tables, seed/cost invariants, malformed evidence and fault
  injection. These run on CPU with GPU intentionally hidden.
* Public-route integration: tiny real Gaussian, LGSSM, nonlinear SSM and supplied
  map fixtures, through preparation, search, repair, verification, retained export
  and restart. Include an informative positive case and a deliberate negative
  case, not just finite-output or all-inconclusive assertions.
* Statistical calibration: independent replicated searches on known laws;
  candidate-family error rates, useful delivery and cost with simultaneous
  rate uncertainty. Seeds/policy are frozen before holdout execution.
* Scheduled GPU validation: trusted GPU/XLA and memory-growth verification,
  selected harder SSMs, limited long-horizon/resource tests, and separate
  posterior/reference checks within the repair program's reconciled budget.

Success requires both error control and useful delivery. Valid abstention is
the correct result for an underinformed boundary case, but abstention on every
easy model is a failed repair. A disagreement with a valid reference invalidates
the affected claim; it does not automatically establish whether the target,
preparation, integrator, statistical decision or recovery mechanism is at fault.
The test report must identify that layer and the next discriminating check.

Review: the main risks are wrong finite-versus-stationary truth, mislabeled
near-unit fixtures, expensive but uninformative test grids, arbitrary narrow
acceptance gates, and confusing posterior failure with tuning failure. The
matrix addresses these by giving every case an oracle and a role, using cheap
positive controls before stress cases, and separating statistical, numerical
and posterior criteria. No new campaign is launched by this matrix alone.

## What is checked now, and what remains

The latest continuation adds maintained native-accounting, real-filter recovery,
GPU parity and model integration test files to the executable inventory. Raw
expanded calibration audits are in
`artifacts/hmc-acceptance-decision-repair-2026-10-02/expanded/continuation-01/calibration-audit-01/result.json`:
16,384 expanded searches and 1,024 actual 100-candidate searches reconcile.
The latter visits every candidate; all 100 interior candidates receive fresh
verification in each applicable search. Four QR/nonlinear process-recovery
tests and four actual CPU/GPU/XLA parity tests passed. These close those precise
engineering checks, not generic posterior or GPU-delivery claims.

K0, QR LGSSM, nonlinear SSM, exact/residual-whitened funnels and the local
mixture have positive CPU confirmations. QR, nonlinear and residual-map public
tuning also qualified/exported/reloaded on trusted GPU/XLA with every native
chunk's placement checked. Maintained tests cover these configurations and
explicit underinformed abstention on K2/K3, centered funnel, Cauchy and simplex.
Other families retain development pricing with null test outcomes, not delivery
passes. The K0 posterior mean/variance reference check passed. The mixture
negative control had left-mode occupancy 1.0 versus exact .300000115; its
posterior assessment correctly failed without changing tuning membership.

Input support is now explicit. Generic fixed-horizon SSM wrappers reject
unknown covariance/initialization parameters, empty/short panels and missing
observations. This prevents unchanged default models from masquerading as
diffuse/degenerate-input tests. It does not test a filter's mathematical behavior
under a supported diffuse or degenerate law. Direct hard-support proposal
rejections are currently tested by the separate uncertainty diagnostic; v7's
native numerical route conservatively rejects nonfinite proposal evidence.
The positive/simplex v7 controls use smooth unconstrained charts and Jacobians.
Do not report those as native hard-support rejection coverage.

The [execution note](bayesfilter-hmc-acceptance-repair-execution-2026-10-02.md)
records exact artifacts, commands, failures and remaining limits. Older v5/v6
model results do not substitute for executing v7. The latest coverage is:

| Layer | Completed evidence | Remaining discriminating coverage |
| --- | --- | --- |
| Protocol, statistics, execution and legacy compatibility | 428 combined checks plus focused additions; odd horizons, health preservation, duplicate/lost streams, source identity, raw receipts and attempted costs | No known failing tested invariant; new target/backend cases still need their own checks |
| Public numerical tuners | Two Gaussian members retained/exported by each route; K0, QR/nonlinear, supplied exact/residual maps and local mixture deliver; serial/threaded actual trial streams/replay pass | Replicated delivery probability, candidate-specific numerical repairs on harder prepared SSMs and broad-cohort real-model affordability |
| Known-law searches | 11,264 initial + 16,384 expanded + 1,024 actual M100 searches; raw denominators audited; all applicable errors zero observed; all 100 interior members verified | Bounds remain conditional on the frozen laws; naive/legacy comparisons keep their different meanings, not a same-target ranking |
| Recovery and devices | Four real-filter fresh-process recoveries, four explicit CPU/GPU parity checks and three native GPU public-tuner deliveries | Larger K5/K6 resource recovery and target-specific latency/compilation behavior; short results do not establish production performance |
| Posterior assessment | Independent K0 moments with MCSE-aware tolerance; mixture missed-mode negative control despite coordinate R-hat near one | Replicated known-answer coverage/SBC using matched prior-predictive simulators; K6 currently lacks a joint reference |
| Official procedure | Book/reference/registry agree; documentation contracts pass; full book built and changed pages inspected | Experimental v7 remains explicit; default release awaits scope-specific evidence; unrelated book citations remain unresolved |

The Hoeffding baseline's 0/512 heterogeneous-start delivery and weak temporal
power are deliberately retained failures. Boundary abstention is allowed;
easy-model all-abstention is not a successful repair. R-hat, posterior
references and precision estimates keep their separate assessment roles.

## Remaining work stays in P4/P5

The next bounded slice should freeze full configurations, including starts and
geometry, for replicated QR/nonlinear/residual-map delivery and matched
K0/Kalman posterior-reference checks. Price K5/K6 before scaling. Its question
is target-specific reliability/cost, not another recalibration of acceptance
bands. A matched SBC plan must define the simulator, independent reference,
prior/data seeds, retained-draw dependence correction and original fit/rank
denominators before running; there is no generic nonlinear exact-posterior
oracle to assume. Reference-absent cases remain mechanics/resource tests.

Native support rejection requires repository-bound support telemetry before
it can become valid v7 evidence. Unsupported diffuse, missing-data and singular
input variants currently test boundary rejection only. If actual supported
filter variants are added, give them a separate independent likelihood/score
oracle; never change a test's label while still running an unchanged default.

For every next allocation, refresh the same progress file and master with
measured costs, a reconciled budget, expected outcome and a distinct output
root. A failed candidate may trigger preparation or evidence repair; corrupted
truth/streams/health invalidate that run. R-hat cannot silently become a tuning
veto during either kind of repair. No new tuner entry point is needed.
