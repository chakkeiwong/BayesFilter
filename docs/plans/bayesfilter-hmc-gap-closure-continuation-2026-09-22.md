# HMC remaining-gap closure and bounded continuation

Status: reviewed and executed on September 22; the bounded tranche is complete. Results are recorded separately
in `bayesfilter-hmc-gap-closure-result-2026-09-22.md`. This is a continuation of
the repair master, not a replacement tuning procedure. Baseline: `203fde46f`.
Preserve unrelated Q20/training edits and the M21 frozen source
`f9c86f41a`. R-hat, ESS and MCSE remain posterior diagnostics only; they cannot
qualify, rank, repair or remove a tuning candidate.

## What we know how to repair

We know the next discriminating action for every remaining gap. We do not yet
know that every requested posterior precision or geometry is affordable, that
a supplied partial map removes its tail stiffness, or that a learned map will
succeed. There is no finite universal burn-in certificate. The completion
criterion is a reproducible, bounded procedure with tested failure reporting
and scope-specific evidence, not guaranteed success on every target.

| Gap | Concrete closure evidence and repair | Dependency or limit |
| --- | --- | --- |
| Supplied-map residual geometry | Audit all four saved partial-map failures, inverse coordinates, exact residual curvature, accepted-state versus proposal health, search caps and verified siblings. Repair a demonstrated implementation error; otherwise retain the exact-map positive control and make a better supplied map a new scope, with target/score/Jacobian checks and fresh tuning. | No third epsilon search without a specific missed mechanism. An imperfect supplied map need not yield a usable posterior. |
| Member precision and global exploration | Assess each declared sibling with its own posterior streams; plan counts in model-quantity units and preserve the requested tolerance. For mixtures require known mode occupancy and crossings, not local MCSE alone. | Audit M21 first. A failed member does not reject its unassessed siblings. New maps require new scopes. |
| Warmup, MCSE and stopping | Run the actual sequential controller with the nominated autocorrelation estimator on fresh exact AR(1) streams, alongside independent fixed-count intervals and exact finite-count laws. Distinguish interval coverage, initialization bias, readiness failures and insufficient precision budget. | Fixed-count estimator eligibility is not optional-stopping coverage. No estimator or controller default changes in this tranche. |
| Subtle full-fit defect power | Profile complete ordinary preparation, tuning and posterior fits for baseline, no-op and the existing conjugate-normal location shift. Use the measured cost and an explicit power calculation to freeze a feasible repeated inventory, or retain an under-budgeted result. | The earlier 18-fit pilot already showed high cost; do not repeat a large SBC inventory without an affordability improvement. Small nonrejection is not adequate power. |
| Exact MacroFinance integration | Read-only search for the matching full-joint target, prepared data, priors, coordinates and uncertainty-bearing reference. Execute only a matched bundle; issue an explicit missing-input list if absent. | Newer block-factorized fits do not substitute for the original joint target. |
| Current-source coverage and maintenance | Run named public ordinary, prepared, fixed-map, Gaussian, beta-binomial, rotated-Gaussian, LGSSM, supplied-map, mutation and diagnostic tests. Profile a complete fit before changing structure. | Current tests cannot relabel frozen M21 evidence. GPU/XLA repetition needs permitted idle capacity. |
| Learned-map quality | Target-specific batched GPU training, held-out map checks, freezing, fresh public tuning and downstream posterior assessment. | Upstream work, separate from supplied-map tuning. No CPU training substitution or borrowed default. |

## Research intent and evidence contract

Question: which remaining failures are implementation errors, residual
geometry, estimator bias, readiness limitations or insufficient sampling, and
which bounded repairs are supported by the evidence?

The comparators are the preserved M20 exact/partial supplied maps, the M21
declared lugsail controller baseline, exact AR(1) fixed-count Gaussian laws,
and unchanged baseline/no-op full-fit mutations. Promotion of engineering
changes requires algebra/inventory regressions and real public-route tests.
The new random-stop screen requires complete planned records, no invalid
transition/oracle artifacts, and pointwise exact 95% binomial lower bounds
of at least 0.90 for both posterior delivery and reported interval coverage.
This inherited adequacy floor is not a nominal-95%-coverage theorem.

Numerical nonfiniteness is a posterior promotion veto. Missing essential
evidence, wrong target/score/Jacobian, source mismatch, budget exhaustion or a
broken oracle is a continuation veto for the affected experiment. Caps,
missed modes, failed precision and a rejected map are repair triggers, not
rejection of the research direction. Acceptance, elapsed time, curvature and
MCSE ratios are explanatory; they do not rank viable tuning members.

The saved-map audit can establish mathematical residual curvature and recorded
failure types. Accepted states alone cannot identify an unarchived rejected
proposal's first failing leapfrog step. A diagnostic replay, if necessary,
must be labelled as replay rather than new independent confirmation.

No universal tuning guarantee, universal sufficient burn-in, anytime-valid
interval, superiority ranking, GPU readiness, learned-map quality, or exact
MacroFinance validity follows from this tranche. Artifacts live in the new
versioned root `docs/plans/artifacts/hmc-repair-master-2026-09-16/m25-r1/`.
Retain all attempts, failures, unavailable intervals and capped outcomes.

## Ordered execution

1. Reconcile source, existing M21 service and outstanding reservations. Its
   272-fit inventory and old source remain frozen. Audit terminal groups when
   available, without stopping, retuning or selectively replacing failed fits.
2. Implement a read-only saved-map diagnostic with the residual-law derivation
   and tests, then run it on all four partial outcomes and both exact controls.
   Record absent proposal detail rather than guessing a failure location.
3. Make the existing diagnostic interval helper accept an explicit estimator;
   keep its historical lugsail default. Add tests that the stopped and fixed
   intervals use the declared estimator and that caps/missing outcomes remain
   in the denominator. Run focused checks before any calibration.
4. Run a two-replication-per-case affordability smoke, excluded from
   confirmation. Then execute a fixed 400 fresh replications each at stationary
   rho=0.98 and deterministic starts (-8,-4,4,8), rho=0.995, using the existing
   autocorrelation precision option. Root seeds 2026092261 (pilot) and
   2026092262 (confirmation) are disjoint convenience choices. Use four chains,
   500-transition chunks, 2000 minimum warmup, latest 1000-window readiness,
   warmup/retained caps of 10000, retained minimum 1000 and mean MCSE 0.05.
   Keep independent fixed arms at 10000 warmup and 10000 retained transitions.
   These inherited settings intentionally test the current controller; the
   slow cell is a stress test, not a promised positive case. Use exact Gaussian
   oracle intervals around the finite-count transient expectation, with
   Bonferroni alpha=0.05/2. Do not treat those intervals as proof of stationarity.
   At 400 trials, a rate near 0.95 has binomial standard error about 0.0109;
   this resolves the declared 0.90 floor without claiming exact nominal coverage.
5. Run current-source integration checks and one profiled baseline/no-op/shift
   cost pilot only if the saved timings leave an identifiable performance
   question. Use the profile to decide whether a focused repair can make power
   confirmation affordable. Freeze a larger inventory only after that review.
6. Search the exact consumer inputs read-only. Recheck GPU capacity only in a
   trusted context; no workload launches without permitted capacity, memory
   growth and recorded XLA settings. Defer dependent cells explicitly.
7. Audit results, reconcile all worker charges, update the master/progress,
   and write the next phase decision. Expected candidate failures continue the
   next funded repair. A changed map or longer precision experiment gets a
   fresh design and output identity, not a relabelled successful prior run.

## Budget, environment and assumptions

The owner's September 22 allowance is 172800 CPU worker-seconds and 86400 GPU
worker-seconds, replacing the older unused balance. Last closed receipts charge
20464.122420 CPU seconds; live M21 reservations are 69000 seconds. Thus at least
83335.877580 seconds are uncharged and unreserved at this snapshot.
Initially reserve 7200 seconds from the master's 18000-second CPU repair reserve for this tranche:
600 saved-evidence audit; 3600 controller pilot/confirmation; 1800 integration
and profiling; 1200 localized repairs and result audit. These are engineering
ceilings, not numerical defaults. All failed attempts count. The campaign
reconciliation must combine these receipts with, not overwrite, the live ledger.

Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, `CUDA_VISIBLE_DEVICES=-1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, one intra/inter-op, OMP and OpenBLAS thread,
and `BAYESFILTER_PRELOAD_CUSTOM_OP=0`. Numerical work here is an explicit CPU
diagnostic/reference exception with stable TensorFlow graphs and XLA off.
Allow one temporary additional numerical worker, serial across this tranche,
beside the two existing frozen workers. This is a scheduling revision, within
the same CPU allowance; the host presently has about 29 GiB available. Each
child is time-bounded and releases its TensorFlow process at case completion.
End this exception after the tranche. No new GPU allocation is made.
The count-budget repair below increases this tranche's reservation to 12000
seconds by transferring another 4800 from that same reserve; 6000 remain in
the general repair reserve. Total campaign funding is unchanged.

| Material choice | Provenance/status | Failure mode and earliest check |
| --- | --- | --- |
| Supplied exact and tanh partial maps | Owner premise and M20 analytic fixtures; baseline | Partial map can remain stiff in tails; derive transformed law and inspect saved ranges before new runs. |
| Autocorrelation precision estimator | Existing option; passed fresh fixed-count screens in two AR(1) cells | Repeated stopping or readiness can still fail; use actual controller and independent fixed arm. |
| 400 trials, 0.90 floor, oracle alpha | Count from binomial precision; floor inherited from M21; alpha derived for two cells | A floor is weaker than 95% coverage; preserve all denominators and label adequacy only. |
| 1000 warmup window and 10000 caps | Inherited operational baseline, not validated universal choices | Short window can have low effective information; report warmup caps separately from transient bias. |
| Absolute mean MCSE 0.05 | Existing scientific request | For rho=0.995, asymptotic variance factor (1+rho)/(1-rho)=399 implies about 39900 draws per chain for four chains. This planning comparator predicts budget pressure; it is not a lower bound on a random estimate. |
| Cost pilot and one extra worker | Measured previous fit cost and local scheduling convenience | Cache/compilation may dominate or host memory may constrain; measure complete fits and preserve failures. |
| Current checkout | Commit plus touched-file hashes; development baseline | Concurrent Q20 edits must not enter a frozen HMC source identity; archive the actual dependency hashes. |

## Skeptical review before execution

### Bounded repair after the original 800 trials

The first count-repair pilot exposed a configuration restriction before any
transition: both shared and exact-transition configurations enforce 10000 as
an unconditional maximum. The prior count-only amendment incorrectly assumed
these bounds were configurable; that pilot is infrastructure evidence only
and remains charged. Correct the implementation with an explicit
`max_results_per_chain` budget and a required `count_budget_reason` when the
budget exceeds 10000. Preserve the default ceiling, default payload, thresholds,
seeds and numerical computations. Serialize every nondefault budget and reason
into the existing policy/checkpoint identity. This is an optional posterior
budget exception under the owner's repair request, not promotion of a different
canonical count default. The older archived wrapper is outside this change.

Before another pilot, test invalid/absent exceptions, exceeded bounds, preserved
default payload and execution parity, changed checkpoint identity, and actual
shared/exact controller execution above 10000. Freeze a new source for this
repair. Baseline M25 and M21 evidence retain their old source identity.

The validation suite must also be able to request the same option: add an
explicit `options.posterior_count_budget` mapping, validate it before launching,
and pass it to the existing public posterior controller. Test an actual
prepared-tuner/reload/long-posterior/fixed-comparator path above 10000 and
reject missing reasons and counts above the declared limit. This harness
wiring is separately tested on current source; it does not relabel frozen
M25 exact-transition or GPU experiments, whose existing calls are unchanged.
Skeptical review: increasing an explicit finite budget supplies additional
evidence but cannot guarantee stationarity or grant missing precision. An
unbounded override, silent default change, or relabelling the failed pilot
would be wrong. The finite opt-in with unchanged checks avoids those errors.

The corrected pilot completes both trials, with 20000 warmup and 45000/40000
retained draws, costing 17.41 worker-seconds including startup. These are
execution/cost observations, not reliability evidence. The maximum trial cost
is 8.442624 seconds including first compilation; twice that times 400 is
6754.10 seconds. This does not fit the original tranche remainder. Transfer
4800 seconds from the unspent common repair reserve, increasing M25 to 12000
seconds within the unchanged 48-hour campaign. Reserve 6800 hard / 6700 soft
seconds for all 400 fresh trials, after the profiles, and at least 600 for
terminal checks. This is cost-based allocation, not outcome-based stopping.
Freeze corrected source
`d2cc54cf4f6163ba629f08f4ef989cd1355ea51bf6fbb71bcef9cc665f9bc83e`.

The original counts are now terminal and preserved. They reject sufficiency
of changing the estimator alone: stationary rho=.98 delivers 137/400 passing
posteriors; dispersed rho=.995 delivers 0/400, with 396 warmup caps. This is a
repair trigger. The independent fixed-count interval screens and both exact
oracle screens pass. Execute one count-only development/confirmation repair
for rho=.995, with unchanged transition, dispersed starts, autocorrelation
estimator, MCSE .05 and all health/R-hat thresholds. This tests a targeted
posterior allocation, not a new tuning procedure or a new default.

The variance inflation is 399. Use a 20000-draw recent warmup window, minimum
warmup 20000, maximum 40000, and 5000-draw warmup chunks. Splitting the window
gives about 10000/399=25 effective observations per half-chain; the leading
stationary raw split variance-ratio approximation, before ranks and folding,
is about 2*399/20000=.0399 above one. It is an explanatory planning heuristic
and does not predict the actual maximum rank/folded R-hat pass probability.
That motivates this window as a testable hypothesis, not a calibrated pass
probability. At minimum warmup, |8*rho^20000| is below 3e-43, so residual
start bias is negligible in this exact model. Keep the R-hat screen unchanged.
Use retained chunks 5000, minimum 40000 (the rounded variance-derived 39900
planning count), maximum 80000. The maximum doubles the planning count to
leave room for diagnostic uncertainty; it is a convenience ceiling, not a
required count or relaxed precision target. Fixed comparator counts are
20000 discarded and 80000 retained, with their exact finite-count variance.

First run two pilot trials with seed 2026092264. If twice the maximum pilot
cost times 400 exceeds the unspent M25 allowance after reserving remaining
profiles/integration/audit, do not launch confirmation: record that this
particular repair is under-budgeted. Otherwise freeze these counts and run
400 fresh trials with seed 2026092265, no replacement. The measured corrected
pilot justified the 12000-second phase ceiling recorded above. The new arm gets
a distinct output and driver identity and uses frozen M25 source r2.

The same .90 exact-binomial lower-endpoint screens apply to delivery and
reported interval coverage; the oracle must contain .95 at alpha=.025.
Comparisons with the original arm are descriptive because work budgets differ.
A passing arm closes only this exact-model allocation cell. It cannot establish
universal burn-in, unknown-mode discovery, nominal anytime coverage, general
HMC delivery, or a default. Failure preserves the records and informs a later
design; no thresholds or counts are searched until a favorable seed appears.

Skeptical review: the original warmup window, not just its cap, was a wrong
baseline for claiming information sufficiency under rho=.995. The repair now
changes both observation window and precision allocation for explicit derived
reasons. Longer runs cannot by themselves prove general convergence, and their
runtime is not a fair speed comparison. The exact oracle and predeclared fresh
denominator make this a valid bounded mechanism test. No mathematical,
source, environment or missing-input blocker prevents this diagnostic arm.

Reporting amendment after reading the completed Gaussian inventory: the old
`posterior_unavailable_members` count includes siblings explicitly unassessed
by design. Separate requested-but-unavailable members from those siblings and
retain an explicitly named total-without-output count. This is a diagnostic
accounting correction, with regression cases for an assessed subset and a
missing requested member. It changes no run, tuning member, numerical result
or coverage denominator. Preserve the frozen M21 reports and explain their old
field semantics in the terminal audit. The amendment passes skeptical review:
the source rows already distinguish these categories and require no inference
about unrun candidates.

Reviewed the plan against wrong baselines, proxy promotion, hidden defaults,
unfair comparisons, stale source, environment differences and invalid artifacts.
Two flaws were corrected before execution: estimator eligibility alone cannot
close readiness/precision delivery, and the rho=0.995 fixed cap cannot normally
meet the requested MCSE. The experiment now reports those as distinct outcomes
and predeclares the slow cell as stress evidence. Saved accepted states cannot
prove where a rejected proposal first overflowed; the audit states that limit.
Frozen M21 cannot certify later source, so new tests and calibration have their
own identity. Repeating expensive SBC without a cost repair was rejected.

The bounded tranche is executable. MacroFinance and GPU/learned-map cells are
input/capacity dependent. No unresolved mathematical assumption is silently
promoted to a default, and no candidate failure stops independent funded work.

Execution refinement before confirmation: the two-case smoke completed all
four planned trials, costing 21.58 worker-seconds including process startup.
Slow-case steady replication cost was about 1.27 seconds for a warmup-cap
outcome; the longest first compile plus retained run cost 8.89 seconds. Frozen
source `b3cec2ad20ba0ee70f55cae4aa35711d26e7563205fcd2acd72bfccbe864052c`
contains committed `203fde46f` plus only the two declared diagnostic-module
edits; unrelated dirty files are excluded. Reserve 1600 seconds per 400-trial
case (1500 soft limit), within 3600 including pilot and focused checks.
Twenty finite retained lengths bound diagnostic shape specialization. The
transition has XLA disabled; existing diagnostic primitives may compile on the
CPU internally, which is not GPU/XLA route evidence.

Saved complete-fit timings attribute roughly 154--167 of 169--184 seconds to
ordinary tuning, with about 5--7 seconds of preparation and 6--8 seconds of
posterior work. This identifies a useful profiling question. The cost pilot
uses the first preserved M22 dataset and otherwise its unchanged native broad
search, once each for baseline/no-op/location shift, fresh convenience root
seed 2026092263. It is reused development data for cost/activation only. Each
attempt has a 400-second soft and 420-second hard ceiling; integration gets
500 seconds. The combined ceilings fit the 1800-second allocation. Profiling
overhead is reported and cannot establish a performance ranking.

## GPU supplied-map continuation after capacity recovery

The trusted 11:54 UTC probe now selects idle host GPU 1. Execute the previously
declared supplied-map GPU matrix on frozen M25 source r2: exact/native,
partial/intermediate-grid and partial-half/intermediate-grid, each with the
same two predeclared development seeds 2026092251 and 2026092252. These compare
the same final map/search definitions used by the preserved CPU fits; source
and backend differences prohibit a numerical-parity or speed claim. Use the
existing supplied-funnel driver, copied into the frozen source tree with its
own checksum. The package snapshot remains unchanged. Initial starts are
inverse-mapped and checked in model coordinates; target, Jacobian, broad L
grid, acceptance/health rules and posterior .05 precision targets stay fixed.

Primary engineering criteria are correct map/starts, own fresh verification,
complete sibling retention and checked model-coordinate replay. Both exact
positive controls must retain a member for a positive usability result.
Partial numerical failures reject their individual posterior outcomes and
remain residual-geometry evidence; they do not stop the other planned cells
or justify a third epsilon search. No learned-map, calibrated-coverage,
superiority or universal posterior success conclusion is available.

Transfer 4200 GPU worker-seconds from the master's unused 10800-second GPU
repair reserve, leaving 6600 there. Six fits have 590-second soft and
600-second hard ceilings, leaving 600 seconds for a localized infrastructure
repair. These cost limits allow about twice the slowest CPU grid fit plus
compilation margin, a planning hypothesis rather than a speed prediction.
Inspect the first exact fit before admitting the other five. A shared invalid
target/map/source, invalid memory policy or unavailable device stops affected
work. A failed candidate does not. Use a fresh directory for every attempt.
GPU worker charges include host work and are not also charged as CPU workers.

Every launch rechecks the pinned device with the trusted probe, sets
TF_FORCE_GPU_ALLOW_GROWTH=true before import, verifies memory growth before
logical initialization, and uses XLA with one host intra/inter-op thread.
The existing CPU calibration may continue alongside one GPU worker. No
training, package changes, contested device or altered numerical gate is
included. The M25 GPU zero-allocation statement above describes the earlier
capacity-unavailable phase and is superseded by this bounded amendment.

Skeptical review: an idle-device probe is capacity evidence only. Actual memory
policy and successful XLA execution must be in each result. A partial map is
not presumed globally well conditioned; a selected member's posterior failure
cannot invalidate its tuning siblings. This is a six-cell engineering matrix,
not an adequate stochastic reliability denominator. These limits keep the
experiment informative even when partial-map posteriors fail.

## Terminal integration scheduling and review

The completed M21 rotated-Gaussian/LGSSM sequence releases its CPU worker slot.
Use that slot for the bounded current-source count-budget integration tests
while M25 calibration finishes; the total remains at most three CPU numerical
workers and one separately charged GPU worker. Reserve 400 seconds within the
12000-second M25 cap for campaign/design/engine, count-budget and guide-contract
regressions. This checks actual public tuning, export/reload, 12000-draw posterior
sampling, a fixed comparator and durable restart, plus invalid design rejection.
It provides engineering evidence only. No frozen experiment source is changed.
The scheduling amendment changes neither the scientific inventory nor funding.

After those regressions, use at most 120 more CPU seconds for an inactive-option
parity replay of the original two slow-stationary pilot trials, seed 2026092261,
on source r2 with the original counts. Require identical serialized tensor
hashes and numerical controller/interval results after removing only execution
time. This reused smoke is excluded from every confirmation denominator. It
checks that introducing the optional count budget did not change the default
transition/diagnostic computation.

### Completed-output worker teardown

The beta-binomial worker wrote all 128 fits, assessment and complete final result
at 12:14 UTC but remained CPU-active during framework teardown for over ten
minutes, with observed resident memory near 90 GiB. This is an observed process
lifetime/resource issue; graph accumulation is an unconfirmed cause. First audit
all final-result, source, candidate, posterior and tensor evidence independently.
If that passes and the complete files remain unchanged, allow a bounded
60-second final grace and terminate only this campaign's completed numerical
worker if it still has not exited. Preserve its nonzero process exit and add an
explicit post-output termination receipt; never relabel it as normal exit.
Charge all worker lifetime through termination. The minute is an operational
cleanup allowance after more than ten minutes of observed teardown, not a
scientific timeout. Missing/incomplete/changed final evidence forbids this
completion treatment. Existing total hard limits remain in force. This localized
resource cleanup changes no sample, seed, result or planned denominator.

### Final harness consistency repair

The terminal review found that the campaign harness still hardcodes lugsail,
although the public posterior policy and exact-controller experiment accept
three named estimators. Expose `options.posterior_precision_method` with exactly
`lugsail`, `autocorrelation`, or `batch_means`; retain lugsail when omitted.
Bind the choice through the existing design identity and posterior policy.
This is a missing option bridge, not a new estimator or default. Validate wrong
types/names before launching and run the real long-count public-pipeline
integration with the explicit autocorrelation option. Existing default-route
integration remains baseline coverage. Reserve 200 seconds within the unchanged
M25 CPU ceiling; frozen experiments and their outcomes are unaffected.
Skeptical review: exposing a tested estimator cannot grant calibration. The
option enables the next actual-HMC experiment but does not transfer the AR(1)
result into HMC or let precision alter tuning membership.

Follow-through review found a second consumer of the old hardcoded estimator:
the independent stopped/fixed reporting helper. Forward the same declared
method there, including event-probability means, and record its identifier.
The final integration now runs the validation engine through independent
assessment as well as tuning/sampling, and checks that its stopped MCSE equals
the actual controller's final MCSE. Preserve default lugsail arithmetic.
This extends the option-bridge repair to every affected diagnostic consumer;
no prior experiment used the new option, so no frozen evidence changes.

Completed outcome and accounting are in [the M25 result](bayesfilter-hmc-gap-closure-result-2026-09-22.md); [the refreshed next design](bayesfilter-hmc-post-m25-next-phase-2026-09-22.md) identifies the remaining evidence questions.
