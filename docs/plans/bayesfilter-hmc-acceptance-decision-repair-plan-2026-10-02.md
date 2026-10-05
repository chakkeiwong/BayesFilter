# HMC acceptance decision repair: code audit and implementation plan

Status: execution authorized and in progress. P0/P1/P2 are implemented;
428 combined regressions passed, followed by focused model/layout additions.
P3 completed and audited the first 11,264 searches, 16,384 expanded searches
and 1,024 full 100-candidate searches. P4 has positive CPU confirmations on
K0, QR LGSSM, nonlinear SSM, exact/residual funnels and local mixture acceptance;
real-filter fresh-process recovery and trusted CPU/GPU numerical parity passed.
Three native GPU delivery checks and the known-answer K0 retained posterior
check also pass; a missed-mode mixture correctly fails posterior assessment.
The official chapter, agent reference and registry are updated; the full book
builds and the changed pages have been visually inspected. No default policy
has been promoted. Exact terminal evidence is recorded in the
[execution note](bayesfilter-hmc-acceptance-repair-execution-2026-10-02.md).

The [robustness matrix](bayesfilter-hmc-acceptance-robustness-matrix-2026-10-02.md)
assigns corner cases, model references, expected outcomes and deliberate defects
to executable tests. Development-only pricing is not a successful integration.
Multi-seed delivery and posterior rank/coverage calibration remain release
obligations. The separate October 2 SSM service keeps its frozen source,
evidence and reservation; its older policy cannot validate v7.

The recommendation is to retain the common candidate-set controller, introduce
one versioned finite-trial evidence protocol, and give its qualification band
a defensible statistical meaning. Repair admission separately from temporal
preparation diagnostics. The first engineering milestone is an exact,
resumable repeated-trial reference; the release milestone additionally requires
useful decisions at an affordable cost on state-space models.

## Audit intent and evidence contract

Question: can the public candidate-set tuner qualify and repair fixed kernels
using an explicitly defined acceptance quantity, defensible uncertainty and
bounded work, without making acceptance a posterior-convergence test?

The inspected baseline is the shared working tree at commit
`70a6d7e9611a9f859a2519f37f236b61cac35c1e`, including existing local changes.
Source checksums in `artifacts/hmc-acceptance-decision-repair-2026-10-02/audit-r1/`
identify the actual files inspected. Historical v5, optional v6 and the failed
experimental batch-means/lugsail rule are separate comparators.

Current phase: P4/P5 bounded model and release audit. Remaining work must use
the current machine-readable record,
`artifacts/hmc-acceptance-decision-repair-2026-10-02/progress.json`, and the
execution note rather than historical status paragraphs. The three calibration
slices retain separate simultaneous-rate families; their totals cannot be
pooled to invent a tighter error guarantee. P4 measures actual work and keeps
posterior validity separate from acceptance qualification. Default promotion
waits for a declared scope's replicated delivery, affordability and independent
inference evidence.

## Execution amendment: required root-cause coverage

The owner's next instruction incorporates every case in the robustness matrix
into this plan and the master and authorizes execution. A case closes only with
an executable assertion or calibrated result for its declared expected outcome.
Naming a model, observing finite output, or counting tests is insufficient.

The skeptical whole-program review found four material gaps. First, the first
calibration declares M=8 but starts two pairs; changing the declared cap to 100
would not test a hundred-candidate cohort. Second, the model catalog lacks an
executable per-model outcome/cost record. Third, temporal detection and false
preparation reports are not calibrated, and adverse verification needs a
deterministic fault test, not a spurious stationary law with different means
between stages. Fourth, mutation checks need maintained executable assertions.
The work below repairs these gaps while retaining the fixed W/T estimand,
independent whole-start-bank repetitions, per-start qualification, separate
search/verification error allocations, health vetoes and all-member retention.

| Work | Root causes and required cases | Exit evidence |
| --- | --- | --- |
| P2-A: controller/evidence | Starts, chunks and repetitions confused; invalid trials discarded; shared epsilon; first-winner stopping; changed scope; adverse verification; R-hat gating | Protocol/statistics/trial assertions preserve fixed W/T, all survivors, candidate-specific repairs, odd T, immutable scope, seed/receipt recovery, all attempted costs and fresh-verification reserve. High or failed R-hat cannot alter admission. |
| P2-B: numerical mutations | Wrong gradient, missing Jacobian, incorrect Metropolis update, resonance, finite-precision immobility | Existing independent density/score/energy references must detect activated defects; matched affine coordinates and labeled start permutations agree. Valid support rejection remains distinct from numerical invalidity. |
| P3-A: statistical corners | Boundaries, dependence, heterogeneity, observed constant scores, rare extremes, correlated starts | Both bounded methods; rho=0/.8/.95/.995/-.8; both qualification boundaries and nearby offsets; both preferred boundaries; valid heterogeneous and misleading pooled means. Report any false membership, direction/preparation errors, delivery, all denominators and work. |
| P3-B: multiplicity and temporal power | Many adaptive pairs; opposing drift hidden by pooling | An actual M=100 cohort with every pair visited and independent verification; stationary/opposite/single-start/nonlinear window laws with exact contrast means. Measure false temporal alerts and positive detection without changing admission. |
| P4-A: small public integrations | Synthetic delivery need not imply real numerical usefulness | Standard/correlated Gaussian, analytic K0, actual QR LGSSM, nonlinear sigma-point SSM, supplied exact/residual-whitened funnel; public preparation/binding, repair, measurement, verification, export/reload. Gaussian/K0 are positive controls; SSM cost/delivery are measured development hypotheses before confirmation. |
| P4-B: difficult and negative models | Poor geometry or local acceptance confused with posterior validity | Near-unit, small-noise and two-noise LGSSMs; long/multivariate SSM; centered funnel; separated mixture; Student-t/Cauchy; positive/simplex transforms. Assert realized regimes. Missing-data, short-series, diffuse-initialization and degenerate-covariance variants get supported reference tests or explicit unsupported-input rejection. No fabricated likelihood support. |
| P4-C: recovery and parity | Resource failure confused with candidate failure | Linear/nonlinear checkpoint recovery, serial/threaded/batched stream rules, CPU/GPU numeric tolerances, FP32 movement edge, XLA, growth policy, contention and cumulative budgets. Different declared streams need not produce identical draws. |
| P4-D: independent inference | Acceptance confused with adequate burn-in or posterior correctness | Existing analytic/Kalman references and matched prior-predictive SBC where supported; mixture missed-mode detection and correct moment assumptions. R-hat/ESS/MCSE stay out of tuning admission. K6 has no joint posterior oracle. |
| P5: one procedure | Competing guides and conflated evidence | Official chapter/reference/registry updated together, documentation contracts and rendered book checks; experimental scope, unsupported cases and measured cost explicit. Promotion requires useful affordable evidence, not checklist completion. |

The executable case inventory distinguishes regression, calibration, development,
confirmation and scheduled stress. Existing legacy tests count only for the
semantics they exercise. Every matrix case remains assigned to the rows above;
unfinished cases cannot disappear when a short run completes. Each phase
refreshes the same progress record and master with evidence, failures, remaining
budget, repair triggers and the next executable configuration.

The baseline is Hoeffding for the same finite-trial quantity, with preserved
legacy v5/v6/lugsail findings under their actual meanings. Statistical criteria
remain declared family-error events and simultaneous delivery lower bound >=.80
on easy laws. Numerical integration uses observable public behavior and an
independent per-case oracle; one-seed delivery is descriptive. Wrong targets,
scores, references, missing health evidence, broken independence or corrupt
replay invalidate the affected run. Legitimate inconclusive candidates or poor
delivery trigger diagnosis and repair, not threshold relaxation or abandonment.
Posterior diagnostics and runtime cannot silently become tuning gates or ranks.

New laws, settings and holdout namespaces freeze before launch. Keep 512 searches
per ordinary new calibration cell from the existing precision allocation. Price
the actual full-cap cohort before freezing its fundable replication count;
report any shortfall. Bands, bet grid, temporal tolerance .10, W=3/T=65 and the
R ladder are inherited test hypotheses with exact-law checks, not defaults.
Real-model development may select candidate hypotheses, but confirmation uses
fresh streams and never retroactively relabels a failed positive control.

Reserve **12 additional CPU core-hours** (43,200 seconds) against the shared
142,536.184-second remaining CPU balance. This conservative convenience ceiling
allows three aggregate four-CPU worker-hours with affinity 0--3, bounded threads
and failures included. Small model pricing initially has a 300-second per-case
ceiling; record costs before scaling. Use fresh output directories below the
existing acceptance-repair artifact root. The planned two-GPU-hour P4 allowance
requires reconciliation of uncommitted GPU balance; the frozen running SSM
campaign keeps its reservation. CPU runs remain diagnostic/reference exceptions.

Premortem: a suite can pass by testing easy laws, a nominal but unused candidate
cap, or a transformed density that changed the model. Check actual cohort counts,
realized geometry, independent value/score/map references and per-model results.
Synthetic inference may be sound yet expensive; record transitions/gradient
calls and wall time. No cheaper estimand is silently substituted. This revised
plan passes skeptical review for bounded execution; promotion remains conditional.

This audit uses source tracing and focused existing regression tests. The tests
check public wiring, existing decision semantics and serialization; passing them
does not establish statistical calibration. GPUs are deliberately hidden before
framework import. Each diagnostic command has a 240-second convenience ceiling,
with at most two TensorFlow intra-operation threads and one inter-operation
thread. No new stochastic calibration, BGS HMC, training or GPU run is part of
the audit. Logs and test results are preserved below the versioned audit root.

Skeptical pre-execution audit: the relevant baseline is actual public dispatch,
not the historical tuner or standalone experimental diagnostic. Static reads
alone cannot establish executable wiring, so exercise the ordinary,
fixed-transport and position-field public routes with existing CPU mechanics
tests. Their diagnostic execution cannot close the statistical gap. Preserve
unrelated dirty files and active campaign evidence. Stop a diagnostic on a
source/environment mismatch or its runtime ceiling; record the failure rather
than interpreting it as numerical evidence. This bounded audit passes those
checks and may proceed.

### Continuation audit: executable outcomes and model eligibility

The next source audit found three remaining implementation gaps. The QR LGSSM
and three-parameter nonlinear targets are named above but absent from the
executable inventory. The latter's first pricing attempt correctly stopped at
the missing full-chain XLA capability; it did not execute candidate evidence.
Also, `bounded_diagnostic` currently has no asserted outcome. It cannot count as
a passing robustness test. Finally, Gaussian recovery alone does not establish
that real filtering targets reconstruct their trial streams in a new process.

This continuation adds both targets, specific terminal-outcome assertions,
trial/cost/export accounting, and independent-process recovery on actual linear
and nonlinear filters. Full-chain graph/XLA tests with explicit momenta must
pass before a validation wrapper can advertise XLA eligibility; a label change
alone is forbidden. Unsupported filter inputs remain boundary failures, not
acceptance failures. Existing model-level value/score and Metropolis mutation
tests are reused for the precise mechanisms they check.

The model ladder is development pricing (four independent trials, no delivery
claim), then fresh-stream positive confirmation at the frozen candidate and
geometry when pricing supports it. Negative fixtures assert a particular
inconclusive, health-rejection or resource outcome. A model must not acquire a
positive-test pass by relaxing its assertion after a failure. The original
pricing artifacts and seeds remain preserved. K0 has already qualified and
exported one member in 85.958 seconds; that is one-seed delivery/cost evidence.
The expanded 32 method/cells and the actual 100-candidate calibration have now
completed; their raw denominators, streams, costs and event counts are audited.
Neither completion nor a model name is itself a criterion.

Evidence contract: compare resumed trials against uninterrupted trials with the
same scope, starts, backend and streams; compare target values/scores and
explicit-momentum proposals with independent references. Require identical
decisions and stream/count/cost invariants and the existing numerical
tolerances. Wrong targets, missing telemetry, lost attempts and duplicate
information invalidate the affected test. R-hat and temporal contrasts remain
explanatory in admission. Model delivery and elapsed time are descriptive;
posterior validity and default readiness require their separate tests.

Reserve a further **28,800 conservative CPU seconds** (eight core-hours, at
most 7,200 worker seconds with four-CPU affinity) from the reconciled
99,336.184-worker-second balance. This pays for focused regressions, model
development/confirmation, failed attempts and book validation, separately from
the already reserved expanded calibration. Development cases have a 120-second
inner/150-second process ceiling; confirmations have 270/300 seconds. These
are convenience ceilings informed by the 6.7--14.1 second completed pricing
cases and 86-second K0 confirmation, not scientific thresholds. Price costly
K5/K6 before any larger allocation. All new runs use distinct directories under
`expanded/continuation-01`; freeze each config before its first target call.
The GPU reservation is unchanged. No worker/environment installation is needed.

Skeptical review: this uses the actual public tuner rather than a model-name
proxy; preserves predeclared outcomes and source-bound replay; distinguishes
approximate nonlinear likelihood from an exact model posterior; and stops each
command within the allocation. Weak temporal detection at admission stopping
time cannot become an acceptance veto or a claim that drift is absent. The
program needs a separately priced diagnostic sample if it requires temporal
power beyond what admission happens to collect. Implementation and bounded CPU
execution may proceed; broad policy promotion remains open.

## What the trace establishes

### Continuation audit: native cost and recovery semantics

The October 2 continuation checks individual attempted chunk extents against
the frozen trial horizon and chunk layout, as well as aggregate costs. A
checksum-consistent receipt with shortened transitions and proportionally
shortened gradient work must fail reconstruction. A committed chunk must have
a matching charged attempt. Repeated charges for an interrupted native call
are legitimate conservative costs; repeated completed evidence is not extra
information. The tests must preserve this distinction instead of rejecting
every repeated charge or silently deduplicating every attempted stream.

Add fresh-process QR LGSSM and nonlinear-filter recovery against uninterrupted
runs at both a partial-trial boundary and the evidence-before-receipt boundary.
Compare states, raw numerical draws, seeds, trial counts and attempted cost;
elapsed times and infrastructure receipts may differ. This is an engineering
test with a small fixed replication cap, not a positive-delivery experiment.
The changes use the existing continuation CPU allocation. The first combined
filter-recovery run hit its 300-second ceiling after three cases, so remaining
recovery checks run as separate bounded test processes with four-CPU affinity
4--7, away from the ongoing calibration/model workers on 0--3. This changes
resource scheduling only, not numerical settings or expected outcomes. Source edits finish
before each reconstruction test begins. The frozen calibration and separate
GPU campaign remain outside the edited source scope. Skeptical audit passes:
the controls can expose the lost-attempt and double-information defects, and
their oracles do not depend on posterior convergence or on a passing candidate.

The input-boundary audit also found that generic SSM fixtures silently retained
unknown parameters without using them in their filters. A requested diffuse or
zero initial covariance therefore did not create that test case. Reject unknown
parameters before target execution and test empty/short panels and missing-data
blocks explicitly. The current fixed-horizon fixture supports neither arbitrary
initial laws nor missing data; rejection is its correct boundary outcome. This
does not declare those features unsupported by every BayesFilter filter.

Execution follow-up: exact-map and nonlinear positive confirmations passed.
QR pricing at epsilon 1.3 produced .415, so its predeclared nomination window
correctly withheld confirmation. The further .95 hypothesis, with the same
geometry and fresh streams, qualified. Residual-map and local-mixture
confirmations also passed at the unchanged [.55,.85] qualification band.
The mixture's separate retained check found left-mode occupancy 1.0 against
exact mass .300000115; posterior assessment failed while tuning stayed verified.
Maintained regressions now cover these outcomes. A later QR test accidentally
copied epsilon without its geometry/starts; it correctly failed and was repaired
to use the full frozen configuration, with the same seed and criteria. All work
stays within the continuation CPU allocation.

The repair needs to change the **evidence protocol and decision contract**, not
just replace a variance estimator. The existing controller is worth retaining:
it already keeps all verified candidates, repairs each exact pair independently,
checks fresh verification, and preserves interrupted work. Its observations,
however, have a heuristic statistical interpretation.

| Finding | Source anchor in the audited working tree | Required repair |
| --- | --- | --- |
| Both public exact-score tuners reach the same controller | `hmc_tuning_dispatch.py:29`, `fixed_transport_hmc_tuning_tf.py:823`, `hmc_candidate_set_public.py:135,230`, `hmc_candidate_set_adapters.py:191` | Keep these entry points and controller; do not add another tuner |
| The policy fixes four chains, four blocks, a 90% t interval and df=3 | `hmc_verification.py:514,1634` | Distinguish start identities from independent repetitions; replace the working interval only in a new policy version |
| Equal observed chain means give zero interval width | `hmc_verification.py:1634`; diagnostic and existing regression below | Equal observations cannot establish zero sampling uncertainty |
| Passing means interval overlap with the practical band, containment in the wider repair band and raw per-chain guards | `hmc_verification.py:2254` | Define what qualification asserts; overlap cannot prove the expectation is in the practical band |
| Raw chain crossing and v5 temporal crossing use no calibrated uncertainty; v6 pools signed temporal changes | `hmc_verification.py:2221,2254` | Remove these screens from the new acceptance-admission rule; preserve per-start temporal information as a separate diagnostic |
| Measurement and verification may have different lengths | `hmc_candidate_set_public.py:67,213,272`; `hmc_candidate_set_execution.py:117` | Bind the exact measured horizon to the claim; no silent transfer between finite-start estimands |
| Each evidence rung starts a fresh, longer run from the original start bank | `hmc_candidate_set_execution.py:545,614`; `hmc_candidate_set_tuning.py:1318,1557` | A longer horizon is a changed finite-start target, not simply greater precision for the old target |
| Legacy equal-block summaries exclude T modulo four terminal observations from the acceptance mean | `hmc_verification.py:1584,1454` | New finite-T means must use all declared measured draws; window diagnostics cannot silently shorten the estimand |
| Chunks continue the current path; stage/work seeds and chain seeds are distinct | `hmc_candidate_runtime.py:9`; `hmc_candidate_set_execution.py:434,632`; `hmc.py:3218` | Add an explicit repetition identity and reset between trials; never treat consecutive chunks as repetitions |
| Position-field execution duplicates evidence-length, seed, chunk and analysis orchestration | `hmc_candidate_set_position_field.py:73,90,143` | Share the new trial protocol across execution adapters; retain its mechanics-only authority |
| Checkpoint loading and retained export recompute analysis and assume one draw range | `hmc_candidate_set_checkpoint.py:88`; `hmc_candidate_set_retained.py:59` | Version and replay the repetition ledger, decision state and terminal endpoint; changing only the evaluator would break or misrepresent replay |
| Experimental batching fails before statistical interpretation at the BGS allocation | `hmc_acceptance_uncertainty.py:99,214` | Preflight window/batch counts without importing TensorFlow; this arithmetic check does not certify mixing |
| Experimental integration tests require all seven models to abstain | `tests/test_hmc_acceptance_uncertainty_models.py:16` | Add informative positive controls and negative controls through actual public tuners |
| The synthetic search stress uses nested prefixes and selects the first passing candidate | `acceptance_uncertainty_validation.py:180` | Exercise the real all-candidate controller, immutable repair children and independent verification; retain this old test only under its actual scope |

The important call chains are:

```text
tune_hmc_kernel / tune_fixed_transport_hmc_kernel
  -> preparation translation or already-issued numerical binding
  -> run_typed_hmc_candidate_set
  -> HMCTuningCandidateSetController.run
  -> HMCCandidateExecutionBinding.observe
  -> reusable TF/TFP runner, chunk seeds, independent chain seeds
  -> health_failures + analyze -> evaluate_hmc_acceptance_evidence
  -> HMCCandidateDecision -> receipt -> _apply_observation
  -> next evidence rung / exact-pair child / fresh verification / retained set
  -> numerical checkpoint recomputation and retained-member validation

position-field public branch
  -> run_shared_position_field_tuning -> _issued_adapter.observe
  -> same evaluator and controller, separate numerical execution wrapper
  -> mechanics-only result, no exact-score retained-member authority
```

The existing experimental `evaluate_acceptance_uncertainty` is outside these
authority chains. It preserves the historical decision and cannot repair public
admission merely by being implemented or imported.

### Executable audit findings

Thirteen focused existing checks passed in 17.48 seconds. They exercised the
ordinary and fixed-transport public routes, the position-field route and its
resume path, existing interval/temporal semantics, policy roundtrips and
inconclusive-rung continuation. This is executable wiring evidence, not a
statistical calibration result. Commands and environment are in
`audit-r1/manifest.json`; output is in `pytest.log` and `pytest.xml`.

Two additional evaluator-only fixtures reproduce:

* constant observed acceptance .68 produces the point interval [.68,.68] and
  `passed`;
* chain means [.66,.70,.74,.78] produce [.6592364,.7807636] and `passed`, although
  that interval is not contained in [.65,.75].

These inputs test decision arithmetic, not actual HMC trajectories. They show
why calling the existing result a calibrated practical-band qualification
would be wrong. The current documentation more cautiously calls it a working
compatibility rule. See `audit-r1/decision-probes.json` and its diagnostic script.

Previously saved calibration still supplies the substantive falsification:
37/256 false conflicts under stationary rho=.995, only 20/256 joint coverage
deliveries, and 0/256 v6 detections of opposing drifts. The BGS replays remain
11/11 unchanged; L=3 is unresolved. Those experiments were not rerun here.

## Proposed statistical target and decisions

### Freeze a finite trial, then estimate across repetitions

For candidate c=(L,epsilon), let G contain the target/data identity, frozen map
and metric, fixed start bank, backend, precision, transition/telemetry semantics
and source identity. A trial resets to each declared start, runs W discarded
fixed-kernel transitions, then T measured transitions. Preparation adaptation
has already ended. W is a discarded prefix, not a claim of adequate burn-in.
Both W and T are fixed before qualification; neither changes when more evidence
is requested. The first implementation keeps the existing four-start bank.

For repetition r and start s, define

\[
A_{rs}=\frac1T\sum_{t=1}^{T}\min(1,\exp\ell_{rs,W+t}),\qquad
\mu_s(c;G,W,T)=E[A_{rs}\mid c,G,W,T],
\]
\[
Z_r=\sum_s w_s A_{rs},\qquad \mu=\sum_s w_s\mu_s,\qquad
w_s\ge0,\quad\sum_s w_s=1.
\]

Use equal weights in the initial development contract to match current pooled
averaging, but serialize them explicitly. The whole start vector is one
independent repetition. This permits dependence between its start columns;
the calculation must not obtain extra precision by assuming those columns
are independent. Across repetitions, independent random streams and identical
reset/protocol semantics are required. Different starts are not replicates.

For R independent repetitions the covariance of their vector average is
`Cov(A_r)/R`. The sample covariance divided by R is a useful MCSE estimate;
it is not sufficient justification for a small-R t interval. No stationary
distribution assumption enters this finite-trial target. It does not estimate
initialization bias, posterior convergence or uncertainty from fitting G.

Measure and verify the same (G,W,T,weights) by default. A short pilot may use
another horizon to suggest hypotheses, but its evidence cannot enter final
qualification. If a different verification horizon is deliberately chosen,
only that horizon receives the final qualification, with its own evidence and
identity. Do not pool the two estimands. Changing G, W, T or the weights requires
fresh qualification; extending R does not.

### State an honest meaning for qualification

Recommended new contract: **each named start's finite-trial acceptance
expectation lies in the declared qualification band, subject to the declared
family error bound and observed health checks**. A weighted average of those
expectations then lies in that band too. Keep a narrower preferred band to
guide epsilon exploration. Do not silently make statistical equivalence to
the narrow target a new admission requirement: acceptance targeting is an
efficiency preference, not mathematical validity of HMC.

Initially use the existing wider repair band [.55,.85] as the explicit
qualification-band comparison setting and [.65,.75] as the preferred band.
This preserves the intended broad compatibility role while repairing its
uncertainty. Final qualification requires simultaneous containment in the
qualification band; overlap with the preferred band supplies no stronger
claim about the mean. If a consumer explicitly requires the tighter band,
it must declare that qualification band and pay its additional information
cost. There is no automatic widening of a consumer's stated requirement.

The all-start requirement formalizes the existing raw per-chain repair-band
guard. It is a design choice, not a theorem that every valid HMC kernel needs
it. Report its incremental cost and rejection rate separately. Neither band
is universally optimal, and rejection by these preferences does not imply
a mathematically invalid kernel. The new policy makes this two-band meaning
explicit rather than calling interval overlap proof of practical-band
equivalence. No extra all-start preferred-band condition is introduced.

| Evidence, evaluated in this order | Controller action under the new policy |
| --- | --- |
| Shared numerical inconsistency or corrupted provenance | Stop the scope and disable its replayable members |
| Candidate numerical invalidity or missing required telemetry | Reject that candidate; do not estimate acceptance from a favorable surviving subset |
| Supported low weighted expectation, with valid acceptance evidence | Propose a smaller same-L child; any parent movement/divergence veto remains |
| Movement/recurrence/divergence promotion veto | Do not qualify; preserve the existing permitted repair semantics and measured evidence |
| Supported start-specific violations of the declared qualification band, without a usable weighted directional proposal | Record `preparation_review_required`; continue other candidates, do not spend identical evidence extensions to cure a resolved discrepancy |
| Supported high weighted expectation and no higher-priority trajectory issue | Propose a larger same-L child; do not assume globally monotone acceptance |
| Intervals for all mu_s contained in the qualification band, and the weighted interval overlaps the preferred band | Measurement may nominate; an independently sampled verification meeting the same requirements may qualify; only qualification-band containment is asserted |
| Intervals overlap boundaries without establishing a decision | Add repetitions at the next funded rung, then explicitly stop as inconclusive at the cap |
| Resource interruption | Resume the same trial or defer work; it supplies no statistical decision |

Support for an admission-relevant start discrepancy means an interval
demonstrates a violation of the explicit qualification band. Means on opposite
sides of the preferred band can still qualify if they satisfy the declared
qualification condition; record their opposing preferences as diagnostics.
Mere inequality between start means is not sufficient. A proposed epsilon direction
is a measured hypothesis for the next pair, not proof that changing epsilon
will repair geometry. Every child is independently measured and verified.

Keep per-start temporal block contrasts, estimated across repetitions. For
blocks a,b use `D_rs=A_rs,b-A_rs,a`; never average their signs across starts
before asking whether one start changes. Such a contrast describes expected
change during the specified finite trial. It is an explanatory/preparation
investigation diagnostic, **not a required stationarity or equivalence test**.
No admission work is delayed because all temporal contrast intervals have not
become narrow. If a future consumer requires temporal robustness, that is a
separately declared, validated policy. R-hat remains reporting-only in tuning;
posterior assessment and sequential warm-up retain their separate contracts.

### Replace unproved t coverage with a checked bounded-data procedure

Implement a simple finite-sample concentration reference first. For independent
trial scores Y_r in [0,1] with common conditional mean m, convexity of the
exponential gives

\[
E[e^{\lambda(Y-m)}]\le e^{-\lambda m}(1-m+me^\lambda)
\le e^{\lambda^2/8}.
\]

For the second inequality, the log of the middle expression has value and
derivative zero at lambda=0 and second derivative at most 1/4. Integrating
that bound twice gives lambda squared over eight. Independence, exponential
Markov inequality and minimization at lambda=4h then give

\[
P(|\bar Y_R-m|\ge h)\le2e^{-2Rh^2}.
\]

This Hoeffding reference has no stationarity or normality approximation. A
finite look schedule must allocate its error across looks. It is a correctness
and cost baseline, not automatically the production estimator.

Prototype a more adaptive bounded-score confidence sequence alongside it,
with a short derivation that can be checked independently. For fixed nonnegative
bets lambda_j <=1 and weights pi_j summing to one, define

\[
K_n^+(m)=\sum_j\pi_j\prod_{r=1}^n[1+\lambda_j(Y_r-m)],\quad
K_n^-(m)=\sum_j\pi_j\prod_{r=1}^n[1-\lambda_j(Y_r-m)].
\]

Under `E[Y_r|past] <= m`, every plus factor is nonnegative and has conditional
expectation at most one; its product and their fixed mixture are nonnegative
supermartingales starting at one. The minus construction has the corresponding
property under `E[Y_r|past] >= m`. Stop at the first crossing of 1/eta, or a
fixed terminal look. Its expected capital is at most one, while a crossing
contributes at least `P(crossing)/eta`; therefore the chance of any crossing
is at most eta. Increasing the terminal look preserves that bound. This is
the reason optional repeated looks are allowed; a t interval plus a new label
does not have this property.

Invert the two tests for simultaneous lower/upper confidence bounds, keeping
their intersection over looks. Fixed bets make the capital monotone in the
tested mean, so conservative bisection is sufficient. Evaluate products in
log space; zero factors represent zero capital, not invalid data. An empty
confidence set is a flagged contradictory-evidence outcome, never a forced
qualification. Interval endpoints must round outwards, and accumulated
floating-point error near a rejection threshold must have a conservative
bound or leave the decision unresolved; rounding a final root alone does not
bound earlier log/product error. The initial bet grid
`{1,1/2,1/4,1/8}` with equal weights is a convenience **power hypothesis**;
validity follows for any such frozen grid, but useful delivery and its cost
must be measured. Do not optimize this grid on qualification holdouts.

This is the implemented experimental estimator, not a claim of a novel method
or default readiness. Its independent rational-arithmetic checks and empirical
calibration are in the execution note. Check the algebra by exact small-support
enumeration and an independent test implementation before larger simulations.
If a literature estimator replaces it, inspect its technical assumptions and
official code when available, keep a local source copy, and update the
derivation/contract before implementation. Do not improvise an empirical-
Bernstein constant or describe a normal approximation as finite-sample control.

Use one process for Z and one for each start mean, so Q=5 in the initial
four-start design. For one declared tuning search, with predeclared maximum
M candidate qualifications and total verification error alpha, allocate at
most `alpha/(2*M*Q)` to each
one-sided process. Union bounding the processes controls erroneous verified
membership across all candidates and all looks, conditional on their frozen
preparations and the random-stream assumptions. Correlation between these
Q summaries does not invalidate the union bound. Adaptive candidates are
permitted because each verification starts with streams fresh conditional
on everything that proposed that candidate.

The guarantee concerns erroneous acceptance assertions over all planned
searches under the stated complete-trial law. It is not conditional coverage
among only delivered or health-passing searches, and it does not bound the
probability of an unseen future numerical failure. Failed, censored or invalid
trials cannot be deleted and replaced with healthy survivors. An execution
whose acceptance law is undefined because of numerical invalidity cannot
obtain that statistical guarantee by conditioning on successful executions.

Exploratory directional assertions use a separately declared search error
budget; report it separately. A heuristic pilot is explicitly a proposal and
does not claim that budget's error control. Diagnostic contrast intervals
have their own declared multiplicity and reporting level, with no admission
budget spent on proving temporal equivalence. Transform contrasts in [-1,1]
to [0,1] before applying the same bounded calculation.

Allocate a candidate's budget once, before its verification data. Resume,
additional looks, repaired output directories and worker retries cannot renew
it. A new epsilon consumes another candidate slot. Extending a campaign's
maximum cannot retroactively increase earlier error allocations. Use a fixed
cap initially, matching the controller's existing `max_candidates`; do not
add a complicated online allocation system unless needed. Fresh verification
protects against exploratory selection; it does not make repeated unadjusted
verification attempts safe.

## Concrete implementation phases

### P0 — characterize and specify (implemented; before-target checks in P2)

Create a framework-free `hmc_acceptance_protocol.py` containing the versioned
trial specification, statistic/diagnostic roles, evidence units, error-budget
allocation and feasibility report. The proposed policy schema is v7; keep
v5/v6 decoding and arithmetic unchanged. It must define W, T, start weights,
base repetitions, replication caps/rungs, candidate cap and search/verification
error allocations explicitly. No inherited draw count may silently become a
replication count. Require the settings to agree with `HMCControllerConfig`
and `HMCCandidateExecutionConfig` before preparation or compilation.

The framework-free preflight must report:

* structural feasibility, including `T >= 8*b*m` for the old four-window,
  two-scale batch diagnostic when b is fixed;
* required fresh verification reservations, total transition/gradient work,
  minimum one-rung cost and the additional cost of per-start robustness;
* the supported estimand and any difference between pilot, measurement and
  verification horizons;
* whether budget can buy even the requested first decision allocation.

A precision/cost projection is a bound or planning hypothesis, never a claim
that the actual evidence will be informative. BGS's 704-draw structural floor
does not become a new default. Exit: hand-checkable specification, policy
roundtrip tests and preflight rejection before any target invocation.

### P1 — shared health and bounded statistical primitives (focused regressions pass)

Extract only the health/validity operations needed from `hmc_verification.py`
into a shared trial-health evaluator. Keep the old evaluator as an explicit
legacy composition. Golden v5/v6 decision tests must continue to pass. The new
decision path must not accidentally execute the raw temporal/chain screen or
the four-chain t interval through a reused helper.

Implement TensorFlow summaries and the bounded uncertainty algorithms in
`hmc_acceptance_statistics.py`, with stable input signatures and XLA numerical
kernels. Keep policy parsing, JSON, provenance and decisions outside those
kernels. Inputs preserve `[repetition, draw, start, parameter]` and the
corresponding acceptance axes; online state preserves trial mean vectors,
cross-products and confidence-process state. Avoid unbounded graph growth
or stacking a complete repetition campaign on the GPU.

Compute the mean over all T measured transitions, including terminal remainders.
Temporal windows use predeclared index ranges and can have unequal lengths;
their boundaries neither discard acceptance data nor turn chunks into trials.

Health checks inspect the discarded prefix as well as measured transitions.
Apply movement and recurrence within each complete trajectory, never across
reset boundaries; preserve every individual veto and measure the effect of
more trials on the probability of encountering a heuristic movement veto.
Do not let new interval logic erase divergence, state/target/score invalidity,
Metropolis consistency or target-status failures. Valid support rejection is
alpha=0 only with repository-bound support telemetry and rejection consistency;
arbitrary NaN/-infinity is not a zero-acceptance observation. Preserve raw
energy/target telemetry rather than using a finite sentinel as Hamiltonian
evidence. The experimental reader's sentinel behavior is not a new runtime
health definition.

Exit: independent arithmetic, finite-support probability enumeration, endpoint,
nonfinite, axis, state-serialization, XLA/reference parity and legacy-health
mutation tests. A theorem about ideal arithmetic does not close these tests.

### P2 — repeated trial execution, controller and durable evidence

Implement the common host protocol in `hmc_acceptance_trials.py`. Keep exact-
score and position-field transition/health adapters behind it, so their
authority remains distinct while replication, seeding and continuation agree.
These are internal adapters, not new public observation callbacks.

Update `hmc_candidate_set_execution.py`, `hmc_candidate_runtime.py`,
`hmc_candidate_set_position_field.py` and the shared controller as follows:

1. v7 evidence rungs increase the cumulative number of **complete independent
   trials**, not T. The versioned work item states its evidence unit, trial
   range and predecessor; v5/v6 retain their historical fresh-longer-run rule.
2. Each trial starts from the immutable bank. Seed identity includes scope,
   exact candidate, stage, repetition, start and chunk; serialized provenance
   identifies each actual derived stream. Schedule/order/resume cannot change
   a trial's stream. Measurement and verification have disjoint namespaces.
3. A partial trial resumes from its last committed chunk. Its acceptance score
   is not admitted until the prescribed W+T transitions complete. All attempted
   work is charged. An invalid trial remains a veto; infrastructure recovery
   cannot discard an inconvenient completed result or draw another seed until
   it passes. Invalid/aborted outcomes stay in the original planned denominator.
4. Extended evidence aggregates all previous complete trials at the same
   candidate/stage/design exactly once. More repetitions do not repeat W for
   previously completed trials or double-charge old transitions. New trials
   pay their full W+T cost. Reserve fresh verification before optional repairs.
5. Add a typed next-action/reason distinguishing more information from
   `preparation_review_required`, invalid data and budget deferral. Supported
   preparation discrepancy is terminal for that evidence allocation, not a
   scope-wide continuation veto. Other L/epsilon families continue. Retain every
   verified member and every unresolved candidate's explicit status; no nominee
   or descriptive ranking is introduced.

Version numerical evidence, work/receipt fields and checkpoint reconstruction
in `hmc_candidate_set_checkpoint.py`, `hmc_candidate_set_artifacts.py`,
`hmc_candidate_decisions.py` and `hmc_candidate_set_retained.py`. Bind the trial
design, error allocation, all trial IDs, partials, raw-health references,
sufficient statistics and recomputed decision. Extend the existing source
closure to the new modules. One `(stream_id, draw_range)` cannot stand in for
several independent trials.

Choose the terminal state for a retained runner by a predeclared trial ordinal
(the last completed verification trial at the terminal declared look), not
by favorable acceptance or movement. Retained posterior sampling still uses
fresh streams and its separate warm-up/assessment. Preserve all tuning seeds,
including failed attempts, in the retained runner's exclusion inventory.

Historical artifacts retain their original policy. They can be inspected, and
their original frozen source can resume them; editing a shared module must not
silently bypass existing source checks. A v7 result requires fresh v7 evidence.
No migration can merely relabel a v5/v6 receipt or fabricate repetitions from
old blocks. Exit: public-route wiring, incremental-cost, interruption/resume,
tamper, decision recomputation and retained-export tests.

### P3 — controlled calibration through the real controller

Extend the validation infrastructure rather than another one-off first-winner
simulator. Supply known-law trial observations to the actual controller for
statistical stress, and use actual numerical bindings for model integration.
The former is a calibration adapter, never production artifact authority.

Use exact means from bounded IID and two-state Markov constructions, including
nonstationary initial laws with analytically propagated expectations. Preserve
finite-horizon truth when changing T; a stationary reference is not a substitute.
Include rho=.995 as a fixed regression, not a value used to choose a favorable
estimator. Use distinct development and held-out seed namespaces, with the
policy, bet grid, horizons and allocations frozen before holdout execution.

Compare the following actual baselines: v5 raw compatibility, v6 pooled contrast,
the failed batch/lugsail diagnostic, naive IID-transition uncertainty as a
negative control, and the bounded-trial Hoeffding reference. They have different
decision meanings; report each against its stated target and do not call
heuristic delivery statistically correct because it is faster. Compare useful
delivery, invalid admission, direction error and cost within each case, not
only an aggregate average. If the proposed complexity is less useful at the
same validity standard than the simple reference, keep the reference and
investigate the cause before promotion.

The controller tests must include all retained members, increasing and
decreasing epsilon children, duplicate pair avoidance, opposing directions,
inconclusive measurement and verification, pilots, finite rungs, dynamic work
allocation, every attempted candidate and actual fresh verification. Measure
the probability that **any** incorrect candidate is verified in a search,
not merely the first candidate's pointwise error.

Initial validation settings are alpha=.05 for the verification family and a
separately reported .05 for exploratory supported directions. The .05 choice
continues the earlier error-screen convention but concerns newly defined
events; it is not a scientific universal or a public default. Joint error
across these two kinds of assertion is bounded by their sum, not .05. Diagnostic
levels/tolerances are separately explicit. Start with 512 independent synthetic
searches per declared cell; its worst-case rate standard error is
`sqrt(.25/512)=.0221`. This is a reproducibility/precision allocation, not proof
of a rare-error guarantee. Report uncertainty for every rate using simultaneous
binomial bounds over the predeclared evaluated cells.

The mathematical error bound plus its assumption and implementation tests are
the foundation. Empirical lower bounds above the nominal error level falsify
the implementation/assumptions. Absence of that failure alone does not prove
calibration; report the upper bounds too. Do not require an empirical upper
bound below the exact nominal limit for an algorithm that can attain that
limit without allowing a calibration margin. Useful delivery must separately
have a simultaneous lower bound >=.80 on predeclared easy cases: interior
in-band means and clearly directional means, separated from the boundaries by
at least half the practical-band width. This .80 target is inherited from the
prior validation plan as an engineering requirement; exact-boundary cases have
no such delivery requirement. Fixed temporal positive controls must demonstrate
per-start detection without changing candidate admission. Freeze their
tolerances and case count in the executable validation config before running.

Define truth labels from the explicit contract: a false verified member has
at least one start expectation outside the qualification band; a wrong
directional assertion puts the weighted expectation on the wrong side of its
preferred-band boundary. A candidate whose true mean is .80 inside a declared
[.55,.85] qualification band is not a false qualification merely because it
is outside [.65,.75]. Count false or unresolved preparation reports separately.
Every table includes all planned searches and an additional conditional-on-
delivery table; the latter cannot replace the former.

Start controller development with M=8, matching the existing stress fixture's
candidate count, then include a separate M=100 cap stress matching the public
default ceiling. A development replication ladder `(4,16,64,256,1024)` is a
bounded power/cost hypothesis, never the new runtime default. Stop early when
a valid decision is reached. Use cheap direct bounded trial-vector laws for
the full controller/multiplicity ladder, then actual dependent traces for a
small predeclared subset; label these evidence classes. Do not allocate a
`search x candidate x repetition x T x start` tensor or launch the full Cartesian
product of models and lengths. The executable preflight prices each cell and
streams simulations in bounded batches before the held-out allocation is frozen.

Exit: exact math checks, false-decision evidence, informative delivery, complete
denominators and measured cost together. A method that safely abstains on
everything has not completed the repair.

### P4 — real models, cost, and limited adoption

Run the same public protocol on a small discriminating model set:

* Gaussian controls: analytic target/score, unequal starts and resonant L;
* LGSSM with the actual Kalman route, including the near-unit-root case;
* the existing nonlinear state-space target with its declared approximate
  likelihood, without claiming that approximation is the exact posterior;
* supplied exactly whitened and residual-whitened funnel maps; no requirement
  for epsilon/L tuning to repair a centered funnel;
* a mixture as a negative inference control: healthy local acceptance can
  coexist with poor mode exploration, and must not claim posterior success.

Exercise ordinary and fixed-transport public tuners, serial/threaded/batched
stream layouts where supported, checkpoint/resume and retained export. Add a
position-field mechanics integration test so BGS-related consumers cannot be
left on the old statistical path unnoticed. That path's evidence does not
acquire exact-score authority from statistical repair. CPU graph/reference
checks precede a small trusted GPU/XLA parity and cost run with memory growth.
No learned transport training is needed for these fixtures.

Use BGS's eleven saved traces only as immutable regression and cost/context
evidence; they contain no independent trial repetitions. A later fresh BGS
integration must pin its own target, starts, protocol and policy. Its L=3
candidate is neither accepted nor rejected by this plan. BGS investigates
whether different starts/geometry or an affordable horizon are appropriate;
BayesFilter supplies the general decision mechanism and feasibility report.

Cost is a real exit criterion. With M=100, Q=5, family alpha=.05 and one
fixed look, Hoeffding requires

\[
R\ge\left\lceil\frac{\log(2MQ/\alpha)}{2h^2}\right\rceil.
\]

For h=.05 this is 1,981 repetitions: at T=512 and four starts, **4,057,088
transitions for one verification candidate before its discarded prefixes**.
This is a sufficient conservative precision allocation, not an information-
theoretic lower bound or a cost forecast for the betting method. A centered
true mean still needs sampling error margin for high-probability delivery.
It establishes why simply prescribing thousands of independent trials is not
an acceptable unpriced BGS repair.

For half-width .15 the same arithmetic gives 221 repetitions and 452,608
transitions. That is still substantial, and half-width .15 alone does not
guarantee containment in a band of width .30 when the sample mean fluctuates.
These two calculations expose the cost of the contract; they do not recommend
either fixed count. A confidence-sequence method must be measured against this
reference, not assumed to be cheaper because its interval is sequential.

If a trustworthy finite-trial procedure is unaffordable, record that outcome
and keep it as a reference. The next bounded investigation may compare fixed-
state fresh-momentum probes against it: those estimate acceptance conditional
on an explicitly frozen position bank and can be cheaper. They do **not**
estimate the same finite-trajectory expectation and cannot replace trajectory
health checks. Similarly, a martingale interval along visited states would
target path-averaged conditional acceptance, not the same finite-start mean.
Neither change is smuggled in as an estimator optimization. A revised scientific
contract and target-specific downstream evidence would be required before
adopting either. No such alternative is implemented by this plan.

### P5 — documentation and release decision

Update the official `docs/chapters/ch21b_hmc_tuning_interfaces.tex` under
`docs/main.tex`, the agent reference, generated registry, examples and
`docs/validation/hmc-regression-tests.txt`. Explain the estimand, replication
unit, exact qualification meaning, budget/inconclusive outcomes and migration
in one procedure. Keep the Markdown reference aligned with the official book;
do not create a second competing tuning guide. Compile and inspect the changed
book pages, and run documentation-contract tests.

First expose the new version explicitly while preserving old policy semantics.
Promote a public default only for a declared scope that passes correctness,
useful-delivery and affordability checks. This is the final release decision
in the repair, not an indefinite substitute for execution. If the evidence
fails, name the remaining failure and next bounded repair; do not call the
program complete. The expected deliverable is one common tuning procedure
with a versioned statistical policy, not a growing menu of public tuners.

## Required regressions and integration evidence

| Difficult case | Expected behavior and discriminating test |
| --- | --- |
| Four different starts, different finite-trial expectations | Estimate each start across repeats; no df=3 pseudo-replication; known-law coverage |
| Equal observed acceptance in a few trials | Nonzero uncertainty unless determinism is independently established; include rare-event laws producing identical observed values |
| Persistent stationary or transient Markov traces | Correct finite-T expectation, valid bounded-score coverage; no independence assumption across transitions |
| Opposing, one-start or nonlinear temporal change | Retain per-start contrasts and detect large known changes; diagnostic alone does not veto acceptance |
| Mixed good/bad starts with pooled mean .70 | Wider per-start robustness contract can prevent qualification; cost and incremental rejections reported separately |
| Mean exactly or near a qualification-band boundary | May remain inconclusive; no forced pass, oscillating direction or infinite allocation |
| Mean on a preferred-band boundary but well inside qualification band | Can qualify without proving strict preferred-band membership; test two-band semantics explicitly |
| Raw values at 0/1, rare support rejections, malformed masks | Bounded arithmetic remains valid; no zero-variance certainty or conversion of numerical failure into support rejection |
| Resonance, immobility and divergence with favorable mean | Health veto remains; no recurrence calculation spanning resets; supported lower-epsilon repair tested separately |
| Adaptation or changing map/metric/start/horizon | Reject pooling or scope reuse; frozen protocol required |
| Odd T or an incomplete temporal window | Mean uses all T transitions; diagnostic windows preserve explicit boundaries and counts |
| Duplicate seeds, reordered starts, duplicated finished trials | Reject identity corruption; permutation with labels/weights gives corresponding result; no double-counted information |
| Crash halfway through trial or after completed trial before receipt | Resume identical stream/state, commit one score, charge attempts, reproduce uninterrupted result |
| Invalid one trial among favorable others | No survivor-only qualification; retain original denominator and veto |
| Many adaptive candidates and repeated verification looks | Search-wide any-false-member rate, persistent error allocation and actual controller state tested |
| Noisy measurement favorable, independent verification adverse | No membership without successful independent verification; repair is a new measured candidate |
| New policy payload with old receipt/checkpoint | Fail on version/identity mismatch, preserve historical reader semantics |
| Several truly qualified L/epsilon pairs | All retained and exportable; no first-winner or descriptive ranking |
| 512 draws with the old [11,22] batches and eight-batch minimum | Framework-free preflight says structurally infeasible; not candidate failure |
| LGSSM/nonlinear/funnel tests all inconclusive | Wiring may pass but useful-delivery phase fails; do not report completed statistical repair |
| GPU contention, deadline or allocation exhaustion | Preserve partials, planned slots and cost; no changed thresholds, favorable restart or fresh error budget |
| High/missing/error R-hat and poor mixture mode exploration | R-hat cannot change tuning decisions; separate posterior assessment remains responsible |

Use deterministic algebra/serialization tests in the fast regression tier,
bounded seeded public-route tests in integration, and multi-search error/power
experiments in the calibration tier. Seeded integration success is not an
error-rate certificate. New tests must exercise observable behavior rather
than only reproducing the new implementation's formula.

## Budget, stop conditions and phase refresh

The subsequent execute instruction authorizes this program within its stated
allocations; the follow-up test question sharpens that execution. The active
state-space service has reserved almost the entire
new 48-GPU-hour allowance, so this plan must not double-book that reservation.

The initial audit charged 480 conservative CPU worker-seconds (pytest itself
17.471 seconds). Subsequent allocated ceilings were 28,800, 43,200 and 28,800
CPU seconds, including failures. The current shared CPU ledger records
70,536.184 worker-seconds remaining. These conservative core-second debits
exceed actual worker consumption and must not be charged twice. See the
execution note for per-attempt evidence and the remaining continuation cap.

The initial P0--P3 allocation was capped at **eight CPU core-hours**
(at most two hours elapsed with CPU affinity restricted to four allowed CPUs,
including unsuccessful
attempts), drawn from the existing CPU grant after recording this allocation.
This is a convenience ceiling, not a statistical sample-size argument. Use
the cheap structural checks before spending it. P4's first GPU allocation is
at most **two GPU-hours**, only from reconciled uncommitted balance after
subtracting the active SSM reservation and protected allowances. The active
reservation need not be released: earlier additive grants leave separate
uncommitted capacity. Parity reserves 600 seconds and the three public-delivery
checks reserve 1,800 seconds in additional charge files; both remain below this
ceiling and leave 67,612.317 GPU seconds uncommitted. Shared execution is allowed
under the owner's contention/recovery instruction; capacity, actual placement
and verified memory growth must be recorded. No automatic
grant increase and no BGS-sized campaign is included. Unit work and plan/code
review do not need a new approval ceremony.

Use a fresh `phase-N/attempt-N` directory under
`artifacts/hmc-acceptance-decision-repair-2026-10-02/` for every run. One concise
manifest records Git plus actual source hashes, command, environment, CPU/GPU
and memory-growth/XLA status, model/data, all seed identities, planned trials,
completed/invalid/aborted outcomes, wall/core/GPU accounting, plan and result.
Do not overwrite earlier failed attempts. There is no approval-token system.

After each phase, update a single progress record and the master with the
completed checks, failed criteria, remaining budget and the next justified
phase. Record engineering correctness, statistical validity and scientific
interpretation separately. Refresh exact commands/config before the next
phase, without moving failed criteria or promoting exploratory seeds to
holdouts. A localized harness failure triggers a focused regression and retry
within the same cap; a candidate failure does not stop the research direction.

True continuation vetoes are invalid/corrupted evidence, broken independence
or target assumptions, missing required health telemetry, a mathematical
counterexample to the claimed guarantee, or exhausted authorized resources.
Poor delivery/cost blocks promotion and triggers the declared next bounded
investigation. It does not justify weakening an admission threshold or
claiming that the BGS model, HMC or NeuTra is invalid.

### Maintained executable interfaces

The following interfaces now exist. The tables describe their evidence scope;
the execution note, not the existence of a command, establishes completion.
Keep statistical model counts, horizons and seeds in versioned JSON.

| Phase | Concrete command interface to deliver | Required output |
| --- | --- | --- |
| P0 | `python -m bayesfilter.testing.acceptance_decision_validation preflight --config <config.json> --output <new-dir>` | Framework-free identity, batch feasibility, replication/work/error allocation and cost report |
| P1/P2 | `python -m pytest -q tests/test_hmc_acceptance_protocol.py tests/test_hmc_acceptance_statistics.py tests/test_hmc_acceptance_trials.py tests/test_hmc_candidate_set_execution.py tests/test_hmc_whole_procedure_repair.py` | Independent math, protocol, legacy and public-route regression results |
| P3 | `python -m bayesfilter.testing.acceptance_decision_validation calibrate --config <frozen-config.json> --output <new-dir>` | All-cell/error/delivery/cost tables, raw counts and denominator ledger |
| P4 | `python -m bayesfilter.testing.acceptance_decision_validation models --config <model-config.json> --output <new-dir>` | CPU public-route model evidence and measured affordability; GPU checked separately |
| P5 | `python scripts/render_hmc_tuning_interface_docs.py`, documentation-contract tests and `latexmk -pdf main.tex` from `docs/` | Consistent registry/reference/book, build log and inspected pages |

Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. CPU validation sets
`CUDA_VISIBLE_DEVICES=-1` before import, four permitted CPUs by affinity, bounded
TensorFlow/BLAS threads and an enclosing 7,200-second timeout for the combined
P0--P3 allocation. The `models` CLI requires CPU hiding and does not currently launch GPU work.
The reviewed `expanded/continuation-01/gpu-delivery-01/run.py` calls the same
model executor on frozen source and checks all native GPU evidence. It requires trusted execution,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, verified repository memory-growth setup, a
specific available GPU, and its reconciled allocation. These launch constraints
belong in the runner/manifest, not in mutable numerical-policy defaults.

## Assumption audit and skeptical review

| Material choice | Provenance and purpose | Failure mode / early check | Status |
| --- | --- | --- | --- |
| Conditional finite-trial mean | Explicit repair of the previously ambiguous target | Horizon/start changes alter the target; identity and known-law tests | Proposed contract |
| Four starts and equal weights | Existing public execution shape and pooled averaging | Poor start coverage; do not claim stationary or global geometry coverage | Compatibility baseline |
| Fixed W and T, increasing R | Derived requirement to estimate one finite-trial target | Repeating long prefixes is unaffordable; preflight and cost ladder | Proposed protocol |
| Practical/robustness bands | Existing [.65,.75]/[.55,.85] policy | Exclude healthy useful kernels; report preference rejection separately | Inherited comparison settings |
| Simultaneous per-start robustness | Formalizes existing raw guard | More conservative/costly than pooled qualification; report both delivery and incremental cost | Explicit policy hypothesis |
| Independent repeated random streams | Current stateless-seed infrastructure plus new reset identity | Shared/duplicated streams, adaptive target or incomplete-trial selection | Required assumption, verified operationally, not proved by distinct integers alone |
| Bounded-score concentration | Derivation above | Numerical inversion or inconsistent score bounds; enumeration and reference parity | Proposed checked method |
| Bet grid and base replication count | Grid is a stated power hypothesis; R comes from cost/precision design | Slow decisions or arbitrary inherited budget | Not a default |
| Search/verification alpha=.05 each | Earlier .05 screen as explicit development convention | Confusing two separate error budgets with a joint .05 guarantee | Calibration setting |
| Temporal contrasts reporting-only | Finite-start target does not require stationarity | Conceal a real preparation problem; retain diagnostics and follow-up reasons | Proposed role correction |
| Eight CPU/two GPU hours, 512 searches | Bounded initial allocation and stated binomial precision | Underpowered/unfinished validation | Convenience caps, incomplete results remain incomplete |

The skeptical review found material flaws in the earlier four-step sketch:
it did not fix the meaning of `passed`, overlooked changing horizons, omitted
the position-field executor and retained replay assumptions, did not price
repeated trials, and could count all-abstention model tests as progress toward
admission. This plan repairs those omissions explicitly. The review also
rejected a tempting overcorrection: requiring practical-band equivalence as
a new universal gate. The final proposal retains separate qualification and
preferred bands and prices the stronger condition only when explicitly asked.

The strongest remaining risk is **an accurate but unaffordable contract**.
Another is that per-start broad-band robustness is an overly costly expression
of the user's practical requirement. These are why P3/P4 measure delivery and
cost, including the separate contribution of that guard, before default
promotion. Small-model confirmations do not settle expensive-target risk. The bet mixture's
mathematical construction has a stated argument, but its implementation,
statistical efficiency and GPU execution now have scoped checks; their measured
results do not settle arbitrary-target affordability or default readiness.

| Decision | Primary status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep the shared controller and public entry points | Source trace and 13 executable checks agree | No route fork required | Broader model delivery/affordability remains | Complete scoped P4 release evidence | Existing statistics are calibrated |
| Give broad compatibility a versioned finite-trial qualification claim | Target and two-band decision conditions explicit | Must not reinterpret historical receipts | Added cost and robustness choice | Replicate prepared-model outcomes with fresh seeds | Narrow preferred-band equivalence or a justified default change |
| Develop bounded confidence sequences for independent trial vectors | Conditional expectation argument supplied | Counterexample or numerical failure stops that method | Delivery and runtime | Preserve audited calibration; measure wider model delivery | Small-R t validity or universal affordable tuning |
| Keep temporal diagnostics separate from admission | Follows the proposed finite-trial target | Health and posterior checks remain | Appropriate target-specific preparation | Report discrepancies with explicit roles | Burn-in sufficiency or stationary acceptance |

| Inference status | Audit outcome |
| --- | --- |
| Hard veto screen | Saved calibration rejects promotion of the tested batch/lugsail rule; current health invariants remain required |
| Statistically supported ranking | None established among competing admission procedures or HMC candidates |
| Descriptive-only differences | BGS block changes and prospective repetition-cost projections beyond the derived reference bound |
| Default readiness | Open; this is a concrete repair plan, not completed statistical repair |
| Next evidence needed | Multi-seed prepared-model delivery/cost and matched posterior calibration before promotion |
