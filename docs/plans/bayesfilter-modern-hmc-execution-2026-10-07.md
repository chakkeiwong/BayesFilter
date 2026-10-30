# Modern HMC implementation: active execution record

**Current scope, October 10:** NUTS is excluded, including diagnostic
comparators. Earlier NUTS work below is historical and its cost remains charged.
The active comparison has twenty core cells and six full-model SNAPER/MAMS
prices. See [the exclusion repair](bayesfilter-nuts-exclusion-repair-2026-10-10.md)
and `N5/nuts-exclusion-01/nonlinear-comparison-matrix.json`. Earlier references
to NUTS as unsupported or an implementation gap are superseded.

The user authorized review and execution of the modern-method implementation
program on 7 October 2026. The controlling design is
[the implementation plan](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md),
including its required nonlinear comparison matrix. This record tracks actual
implementation and evidence; unimplemented phases remain open.

**Latest continuation, 9 October:** the same 72-hour aggregate CPU/GPU grant
governs new work through the
[funded N5 campaign](bayesfilter-modern-hmc-n5-campaign-2026-10-08.md).
D0/D1/E0/E1 are terminal: 32/32 short-model fits and 64/64 members pass, both
full PP-SGQF L=9 members pass, both L=3 members are precision-limited, and the
structural reference remains limited on one second moment. The settled balance
was 45.01 hours after reserve. E2 now replays structural weight contributions;
D2a prices the extended full UKF/SIR harness on GPU 0. Earlier active/terminal
statements are timestamped history. Posterior/release qualification remains open.

## Pre-execution skeptical audit

The plan's scientific separation is sound: invariant kernels, useful tuning,
posterior precision and performance comparisons require different evidence.
The nonlinear matrix now covers existing application adapters rather than
substituting a Gaussian or one reduced SSM. Checked source baseline is the
shared checkout at 70a6d7e96 with subsequent working changes. Scoped pre-edit
bytes and hashes are archived under
`artifacts/modern-hmc-implementation-2026-10-07/N0/attempt-01/`.

Implementation refinements required before execution:

* First support one explicit sampler family per tuning scope. Its immutable
  typed specification must survive measurement, independent verification,
  export/reload and retained continuation. A private numerical kernel alone
  does not establish public support.
* Preserve the legacy fixed-HMC artifact and policy meanings. New policy
  versions must reject unsupported/stale payloads, and optional new-method
  execution must reject unsupported serial, trial-batch or augmented-state
  combinations rather than silently use fixed HMC.
* For initial randomized HMC, a grid of integer maximum counts can parameterize
  `tau_max = epsilon * max_steps`. This is an explicit restricted duration
  family. It is not a fixed realized L and must not be described as the general
  arbitrary-duration API. Every repaired epsilon creates a new full policy.
* Retain current exact-score capability enforcement. The BGS position-field
  route cannot acquire new-method support from configuration alone.
* Numerical kernels require independent formula/correction oracles and activated
  defects. New validation metadata alone cannot close a sampler test requirement.
* The nonlinear model registry is an inventory, not current qualification.
  Historical map artifacts and old filter-readiness labels require rechecking.

Audit verdict: proceed with implementation and focused engineering checks.
Do not launch an unspecified full research campaign. Complete-fit pricing,
statistical power and model-specific accuracy criteria remain required before
confirmation. An observed failed candidate is a repair trigger, not an automatic
reason to stop later phases designed to address that failure.

## Initial engineering evidence contract

Question: can the existing public candidate-set lifecycle bind and replay a new
corrected transition without changing legacy semantics or losing evidence/state?
Comparator: the archived current fixed-HMC behavior and independent analytic
Gaussian/integrator references. Pass: deterministic contract/replay regressions,
correct correction and stream accounting, and focused public-path integration.
Veto: wrong density/score/correction, state loss, corrupted records, unsupported
fallback or unaccounted work. Runtime and short-chain diagnostics are explanatory.
No posterior convergence, speed ranking or default-readiness claim follows.

Routine checks use the installed tfgpu Python, CUDA hidden for CPU reference
checks; native GPU/XLA smokes use trusted execution and verified memory growth.
The initial engineering session has a convenience safety cap of two hours of
local test wall time and fifteen GPU-device minutes of short native smokes,
not recycled research-campaign authority. Tests stop on a mathematical failure
or resource cap, preserve logs and trigger focused repair. This cap does not
price the later full nonlinear campaign or restrict ordinary implementation time.

## Progress

* **N0 source/plan audit and target inventory: complete for the current scope.**
  The core nonlinear adapters are available for five required targets. NL-GROWTH
  still lacks a complete posterior adapter/reference and NL-SV still requires a
  current source-route/score eligibility review.
* **N1 shared specification, policy and replay infrastructure: engineering
  complete.** `HMCTransitionProfile`, `FrozenTransitionSpec`, checked payload
  identity, full-refresh stream offsets, retained export/reload, and modern
  work accounting are exercised by the focused suites. A preflight defect that
  priced modern candidates as legacy `L+1` work was repaired; modern preflight
  now includes the separate value/score and status-call reserve.
* **N2 randomized HMC: mechanics and public dispatch complete; qualification
  open.** The uniform-duration profile is implemented through both public
  tuners and the retained bridge. It uses an independent duration per chain and
  transition and records maximum versus realized counts. It remains
  experimental: short GPU mechanics have passed, but scoped posterior and statistical qualification remain open.
* **N3 ChEES/SNAPER: proposer mechanics complete; evidence open.** The inspected
  TFP 0.25 primitives are adapted with fixed identity variance and explicit
  source hashes. Pilot draws are discarded and proposed pairs are routed back
  through fresh shared measurement and verification. No adaptation-quality or
  efficiency claim is made.
* **N4 deterministic MAMS: mechanics and public dispatch complete; qualification
  open.** The full-refresh isokinetic velocity-Verlet path, sphere correction,
  poles/zero-force limits, random-length law, and dimension guard are tested.
  Persistent stochastic MAMS, LAPS and MALT remain unimplemented.
* **N5 public mechanics bridge: broad coverage executed; scientific comparison
  in progress.** CPU runs verified both declared randomized-HMC
  candidates on LGSSM, accumulation, predator–prey UKF, predator–prey sparse
  quadrature and structural UKF. The latter two also passed same-source
  all-member export/reload. The GPU/XLA LGSSM public-tuner run completed all
  64 trials with both members exported/reloaded and exact attempted-work
  reconciliation. Short CPU/XLA--GPU/XLA comparisons passed for randomized HMC
  and MAMS on LGSSM and accumulation, with exact same-device replay. SIR's
  missing native conditioning diagnostic was repaired without changing its
  value/score. A too-large epsilon then caused an invalid filter covariance;
  local-curvature nomination produced finite trials at a smaller epsilon.
  Its early CPU and GPU attempts reached their hard wall caps before fresh
  verification; funded attempt 17 subsequently completed both SIR members and
  their export/reload. D0/D1 provide the scoped posterior evidence detailed
  below. The original 25-cell inventory is preserved in
  `artifacts/modern-hmc-implementation-2026-10-07/N5/terminal-audit/nonlinear-comparison-matrix.json`;
  the full model/method matrix and uncertainty-supported efficiency comparisons
  remain unfinished.
* **N6 LAPS: not activated.** It remains contingent on a qualified MAMS path,
  ensemble-aware evidence, and measured batch scaling.
* **N7 conditional alternatives: not activated.** MALT/MEADS/DRGHMC require a
  target-specific residual failure and separate correction/state tests.
* **N8 consumer/documentation closure: partially complete.** The existing agent
  reference and official tuning chapter now document the typed experimental
  profiles, maximum-step semantics, v8 lower-bound policy, status/work
  accounting, and release limits. The modern-method chapter cross-references
  the implemented subset. Full MacroFinance/BGS consumer qualification remains
  open.

## Current evidence and repairs

The final focused CPU/XLA suite is recorded in
`artifacts/modern-hmc-implementation-2026-10-07/N0/attempt-01/final-modern-test-r9.log`:
49 tests passed. After the latest fixture repair the same focused suite passed
54 tests. The preceding dispatch and validation repair suite recorded
21 passes in `dispatch-test-r8.log`; the broader pre-repair regression had two
expected failures from the validation control's float32/float64 literal and
was repaired before r8. Earlier unchanged regressions recorded 127 passes in
`regression-test.log`. Document builds produced the chapter and complete-book
PDFs under the same attempt's `chapter-build/` and `book-build/` directories.

The earlier no-idle-device probe is historical. Trusted October 8 probes
selected idle GPU 0 with memory growth verified before target initialization.
Successful GPU artifacts are in N5 attempts 09 and 12. SIR attempt 14 is a
partial GPU result, not tuning qualification. GPU use is described below.

The nonlinear wrapper was repaired after the first diagnostic exposed that the
generic accumulation adapter lacked `target_status_telemetry`; the diagnostic
now records `target_status_policy="none"` for that target rather than inventing
filter status. A second repair preserved full status fields for predator–prey,
SIR and structural adapters instead of using the older reduced telemetry view.
The public-dispatch audit then found and repaired a misplaced modern-runner
branch that had silently executed legacy TFP HMC. Tests now assert the actual
runner type, transition identity, random-duration metadata and direct-kernel
replay.

## October 8 continuation: terminal engineering review

The active design is [the consolidated N5 plan](bayesfilter-modern-hmc-n5-bridge-2026-10-08.md).
All attempts are preserved under
`artifacts/modern-hmc-implementation-2026-10-07/N5/`. No run remains active.

| Attempt | Outcome | Wall time | Interpretation |
| --- | --- | ---: | --- |
| 07: CPU remaining models | PP-SGQF and structural each verified two members; SIR preparation rejected incomplete status | 1,985.17 s total | Public mechanics passed on two targets; real SIR status defect |
| 07 post-run audit | Both complete candidate sets exported and reloaded | 227.54 s | Same-source CPU replay; historical explicit target closure was incomplete |
| 08: GPU parity | Finite LGSSM and repeatable GPU path; CPU/non-XLA RNG mismatch; wrong NL-ACC factory flag | 38.14 s | Invalid parity comparator and diagnostic configuration; no sampler verdict |
| 09: corrected parity | All four LGSSM/ACC × randomized-HMC/MAMS checks passed | 73.50 s | CPU/XLA--GPU/XLA errors below 2e-15; exact GPU replay, finite health |
| 10: GPU public LGSSM | Rejected target-scope mismatch | 9.50 s | Launcher replaced scoped XLA capability; guard behaved correctly |
| 11: CPU SIR | Status repaired; epsilon 0.1 caused eigensolver failure | 61.48 s | Candidate/geometry failure, not an infrastructure timeout |
| 12: GPU public scope repair | Two members verified, exported and reloaded | 40.56 s | 64 returned trials, 114,688 charged units, 55,064 actual value/score rows |
| 13: SIR local epsilon, CPU | Four finite returned trials, fifth charged/interrupted | 780.09 s | Explicit partial evidence; no candidate verified |
| 14: SIR local epsilon, GPU/XLA | 19 finite returned trials, twentieth charged/interrupted; first candidate measurement passed | 600.29 s | Verification unfinished; no candidate verified |

The repaired launcher preserves actual data and transitive source hashes,
retains partial/failure manifests, enforces a parent-process wall cap, uses the
base adapter's target scope, checks raw trial streams and actual native calls,
and exports/reloads all verified members. The final source closure for the
GPU public cell contains 147 files. Its native trace confirms GPU placement
and XLA; memory growth was verified before initialization. Peak TensorFlow
allocator usage was 551,936 bytes for this tiny LGSSM cell, which is not a
capacity statement for other models.

The SIR status repair computes `max_t lambda_max(S_t)/lambda_min(S_t)` from
its existing innovation eigendecomposition. No floor, value, score or filter
recursion changed. The full-wrapper health test and existing finite-difference
and independent value-only parity checks passed. Two central-difference steps
at the four starts agreed on a maximum local curvature about 22,083.4;
`0.25/sqrt(curvature)` nominated epsilon 0.001682313009806192. This local
quadratic argument does not guarantee stability elsewhere or useful mixing.
Returned SIR traces were finite, had zero status/floor failures, and remained
unqualified because the full independent verification design was unfinished.

The [terminal raw audit](artifacts/modern-hmc-implementation-2026-10-07/N5/terminal-audit/result.json)
checked all returned trial indices, stage streams, frozen starts, transition
specifications, devices, realized step counts and work charges. It reconciled
114,688 units in each complete cell, 7,680 in the partial CPU SIR cell and
32,768 in the partial GPU SIR cell. Interrupted calls remain charged without
invented samples. This is an offline artifact audit, not current-source/device
replay of historical attempts. The separate same-source export/reload run is
identified explicitly above.

Final focused launcher/audit suite: **22 passed** in 31.07 seconds, logged in
`N5/launcher-repair-tests/pytest-final.log`. It includes negative tests for
wrong seeds, missing trials, incorrect profiles/call counts, uncharged work,
incomplete candidate retention, dependency closure and timeout preservation.
The scope-only regression passed two tests. The SIR suite's nine selected
regressions passed, including gradient and value-only checks; its initial
combined run also exposed a test fixture trying an unsupported shortened
SIR observation horizon. The corrected wrapper test uses the actual frozen
20-observation contract. The earlier 54-test modern suite and 139 legacy
regressions remain valid for unchanged kernel/lifecycle code; their later
counts came from tool outputs, not the earlier 49-/127-test log files.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep experimental randomized HMC/MAMS paths | Public mechanics and scoped GPU checks pass | No native failure in completed cells | Posterior correctness/precision across models | Price and execute the declared posterior design | Release or default readiness |
| Retain complete bridge candidate sets | Both members independently verified, no nominee | Raw accounting/reload checks pass | Low acceptance floor and short trajectories do not establish mixing | Use prepared geometry and longer posterior assessment | Sampler superiority |
| Preserve SIR partial results | First GPU measurement passes; complete verification missing | Original 0.1 candidate failed; smaller-step returned paths finite | Full cohort price, geometry and posterior precision | Resume unchanged saved GPU checkpoint in a funded phase or restart under a new frozen geometry scope | Qualified SIR tuning |
| Close this bounded engineering continuation | All processes terminal, tests/artifacts audited | Compute caps enforced | Full campaign still unpriced | Refresh N5 complete-fit design | All program phases complete |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Real scope/status defects repaired; original SIR epsilon failed; returned smaller-step traces passed finite/status checks |
| Statistically supported ranking | None |
| Descriptive-only differences | Trial times, acceptance and CPU/GPU observations |
| Default-readiness | Not established; modern claim-bearing builder remains closed |
| Next evidence needed | Complete independent verification, prepared geometry, posterior references/precision, replicated fair comparisons |

The strongest alternative explanation for short healthy paths is tiny steps
and a narrow start bank exploring too little of the posterior. Independent
posterior/reference disagreement would overturn any usefulness interpretation.
The weakest evidence remains target-specific geometry and posterior precision;
no amount of acceptance or short-run parity closes that gap.

## Budget and next phase refresh

The single agent reference and official tuning chapter now explain the tested
RNG compilation-mode boundary. The main monograph rebuilt successfully to
`N5/book-build/main.pdf` (692 pages). A stale local bibliography was bypassed
by generating the bibliography in the isolated build directory and putting
that directory first in `TEXINPUTS`; the final build has no unresolved citation
or cross-reference warnings. The added paragraph was checked in rendered PDF
text on page 422. Existing unrelated layout warnings are outside this change.

The original 900 GPU-second engineering allocation has about 125 seconds
remaining after conservatively charging 775 seconds for all launches including
RNG localization/startup. It cannot fund another complete SIR cohort. The
original two-hour local-test wall allowance is not reset. Recorded prior and
current test/diagnostic times remain inside it, with an explicit reserve for
unlogged early short checks. Implementation time is separate from test wall
and neither is a new scientific campaign allocation.

Measured GPU SIR trials take roughly 24 seconds at maximum length 2 and
30–31 seconds at maximum length 3; 45 fresh trials plus checking/export remain
in its checkpoint. A rough remaining price is 25–40 GPU minutes under similar
load, not a guarantee. Resume must preserve the original source/runtime scope,
starts, epsilon and stream accounting. Source changes or new geometry require
a fresh run. Do not extend the just-completed attempt's frozen wall policy
silently or recycle earlier large campaign allowances.

The next scientific phase remains **N5 prepared-geometry, complete-fit pricing**.
First declare model-specific geometry, independent posterior estimands/reference,
warm-up and MCSE stopping rules and replication, then price tuning plus retained
sampling on LGSSM and accumulation with the fixed-HMC/randomized-HMC baseline.
SNAPER needs proposer-to-verification comparison on the same geometry; MAMS
needs separate posterior qualification. Preserve the five-by-five nonlinear
matrix. NL-GROWTH adapter/reference and NL-SV source eligibility remain open;
LAPS and conditional alternatives retain their explicit dependencies. No
unpriced full campaign has been launched.

## Funded continuation: first complete-fit pricing and recovery

The 72-hour grant is 259,200 aggregate worker-wall seconds. Latest allocations
are A=3.5, B=7, C=9, D=40, E=8.5, F=4 hours; reallocation within that total is
documented in the active N5 campaign. Overlapping worker time is additive;
passive monitor/supervisor elapsed time is not a third numerical worker charge.

The first four-cell GPU/XLA ladder finished at 07:37:25 UTC in 1,749.17 seconds,
including independent CPU assessments. Artifacts are in
`N5/complete-fit-pricing-01/`, references in `N5/reference-01/`. All four fits
measured and independently verified both candidates, exported/reloaded both,
and ran the common posterior controller. Each member used 2,000 warm-up draws
per chain. All returned posterior checks passed without a hard veto.

| Target/method | Fit seconds | Retained draws per chain, members 1/2 | Independent agreement |
| --- | ---: | --- | --- |
| LGSSM fixed | 543.11 | 1,000 / 5,000 | Both passed |
| LGSSM randomized | 442.76 | 2,000 / 3,000 | First passed; second not established |
| Accumulation fixed | 384.72 | 1,000 / 2,000 | Both passed |
| Accumulation randomized | 365.98 | 3,000 / 2,000 | Both passed |

The second randomized LGSSM member's standardized-coordinate second moment
was 0.84507 versus reference 0.77836, MCSE 0.02119. Its declared agreement
quantity `(absolute error + 3 MCSE + reference sensitivity)/reference SD` was
0.10595, above 0.10. Absolute error alone was 0.05386 reference SD. This is a
failed operational agreement screen, not proof of bias or a tuning failure.
The saved fit is unchanged. Fresh complete-run replication will test this
stopping/precision outcome; inspecting a failed reference does not permit
retuning the same claim sample or dropping that member from the denominator.

SIR attempt 16 stopped only at its 3,599-second allowance, with 36 committed
trials and both candidates measurement-qualified. Attempt 17 resumed at
07:41:25 UTC under the planned remaining Phase A cap, preserving all numerical
settings, hashes, streams, prior charges and interrupted-call history. Its
posterior remains untested. Shared-device trials are substantially slower than
the earlier idle-device price; checkpoints continue to progress.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue short-model validation | Public tuning completed; seven of eight member reference screens pass | One reference screen fails; no native numerical veto | Finite-chain variation, optional stopping and MCSE calibration | Fresh paired replications with full denominator | Bias, superiority, or posterior qualification from one seed |
| Price SNAPER/MAMS/NUTS | Focused actual-target mechanics being checked | Unbounded NUTS pilot overshoot repaired with declared-domain proposal projection | Adaptation quality, complete fit cost and source variants | Separate six-cell GPU ladder after integration checks | Fully tuned baseline, new default or release |
| Continue SIR saved verification | Both measurement stages complete | Wall cap only; no numerical failure | Shared-device cost and unfinished verification | Bounded unchanged-checkpoint continuation | Useful SIR mixing |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No native veto in the four complete fits; one independent agreement screen failed |
| Statistically supported ranking | None; one paired seed is pricing evidence |
| Descriptive-only differences | Runtime, retained counts, point errors and observed diagnostics |
| Default-readiness | Not established; modern claim-bearing builder remains closed |
| Next evidence needed | Fresh paired replication, stopping/reference calibration, additional methods and harder nonlinear targets |

Red-team assessment: the short horizons and prior geometry may hide the
application-scale difficulty. A shared reference error could make all fits
look accurate; independently implemented values, two-spacing scores and
quadrature refinement reduce but do not eliminate that risk. The failed second
moment could be ordinary Monte Carlo variation; fresh-replication persistence
or disappearance is the next discriminating evidence. No runtime ranking is
supported by these four descriptive prices.

## October 8 comparator and full-model reference update

The six additional complete-fit comparator cells finished at 08:14:37 UTC,
using 1,685.07 occupied worker-seconds including their reference assessments.
Both members of every MAMS/SNAPER fit and the single NUTS comparator member
passed the common posterior checks and independent reference screen.

| Target/method | Complete-fit seconds | Members passing posterior and reference screens |
| --- | ---: | ---: |
| LGSSM MAMS | 127.51 | 2/2 |
| LGSSM SNAPER | 274.70 | 2/2 |
| LGSSM NUTS | 153.41 | 1/1 |
| Accumulation MAMS | 442.33 | 2/2 |
| Accumulation SNAPER | 564.66 | 2/2 |
| Accumulation NUTS | 102.36 | 1/1 |

These are prices at seed 1000 with narrow candidate grids. They do not establish
a performance ranking. The two NUTS fits had no frozen depth saturation or
boundary epsilon. Their discarded dual-averaging proposal projection is fully
recorded; no pilot draws enter the posterior. The independent outcome audit
found no missing or mismatched member evidence. Results and audit are in
`N5/complete-fit-comparators-01/` and
`N5/campaign-72h-audit/comparator-pricing-report.json`.

Full-model CPU geometry checks produced positive local factors for structural
UKF and predator–prey UKF/SGQF. The first SIR optimizer trial failed an
eigendecomposition; that failure is preserved. Scaling the optimizer chart
using the measured initial prior-coordinate curvature produced stationarity
in eight iterations, with prior-scaled score norm 2.95e-7 and positive local
curvature eigenvalues 84.43, 1399.60 and 4334.58. The numerical target did not
change. None of these factors has global coverage or GPU admission evidence.
Artifacts: `N5/full-model-geometry-01/` and `N5/full-model-geometry-02-sir/`.

The independent structural reference agrees with the native full 100-observation
likelihood to 2.26e-11; two-spacing scaled score discrepancies are 9.29e-7 and
7.54e-9. Its first integration proposal nevertheless failed the declared
refinement screen: largest last-order mean change .0752 posterior SD and
largest pointwise uncertainty half-width .1068 versus .002 required. It cannot
judge posterior accuracy. `N5/structural-reference-01/` preserves this failure.
The approved-in-scope repair fits an auxiliary eight-Gaussian proposal using
discarded independent integration samples, then freezes it for new scrambles;
the target, quantity list and precision requirement stay fixed. This second
reference is active in `N5/structural-reference-02/`. Its first final rung has
minimum importance ESS 7,652 among 16,384-point scrambles, but that concentration
diagnostic alone cannot qualify the reference.

At 08:37 UTC, the fresh 32-fit replication is on its third fit; both completed
LGSSM fits passed all-member agreement. SIR recovery has continued past 53/64
returned trials. The recorded numerical expenditure including active elapsed
time is approximately 3.50 worker-hours, with 19.49 hours reserved for the
remaining active allowances and ten minutes reserved for unmeasured launch,
probe and test-startup overhead. Approximately 48.84 hours remain unallocated.
This is a snapshot, not a runtime forecast; see the exact source manifests and
`N5/campaign-72h-audit/budget-ledger.json`. Passive waiting is not charged as an
extra numerical worker, and failed attempts remain charged.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep short-model comparators viable | 10/10 additional member agreement screens pass | No frozen numerical, depth or boundary veto | One seed, narrow grids, shared-device costs | Complete fresh replication and broader scoped baselines | Method ranking or general reliability |
| Preserve structural reference as limited | Value/score parity passes; integration precision fails | Reference cannot adjudicate sampler accuracy | Tail coverage and importance concentration | Frozen flexible-proposal integration on new scrambles | Sampler bias or posterior qualification |
| Retain local geometry proposals | Stationary positive factors obtained in all four full-model scopes | First SIR localization failed and was repaired | Local curvature need not represent posterior geometry | GPU/XLA checks and separately qualified complete fits | Global whitening or learned-NeuTra quality |

| Inference status | Current conclusion |
| --- | --- |
| Hard veto screen | No veto in six completed comparator fits; structural integration is reference-limited; first SIR localization failed |
| Statistically supported ranking | None |
| Descriptive-only differences | Costs, draw counts, local curvature and importance ESS |
| Default-readiness | Not established; corrected experimental profiles remain optional mechanics implementations |
| Next evidence needed | Finish replicated stopping/reference evaluation, refine full-model references, validate full-model prepared fits |

Post-run red-team note: short LGSSM/accumulation horizons can make every method
look adequate while harder targets remain unusable. A good local factor or high
importance ESS can miss a posterior region. Fresh-scramble reference instability
or posterior disagreement would overturn a usefulness interpretation. The
unchanged full five-model/five-method inventory remains open; no one-seed
pricing row is relabeled a completed scientific comparison.

## October 8 full-model preparation and conditional complete-fit execution

At 10:43 UTC, eleven of the 32 fresh D0 fits are complete and all eleven pass
their all-member posterior/reference screens. The twelfth fit is active. The
saved SIR mechanics recovery finished at 09:09 UTC: both members were verified,
exported and reloaded. Its last attempt used 5,275.21 seconds; earlier stopped
attempts remain charged. This completion establishes the saved mechanics
lifecycle, not a SIR posterior result.

The full-model trajectory price ended at 10:13 UTC. Each of its four model
workers reached its 900-second cap, for 3,602.34 seconds of enclosing occupied
wall time. Structural completed three local rows, PP-UKF all four local rows
and one prior row, PP-SGQF all four local rows and three prior rows, and SIR all
four local rows. Completed rows were healthy. Missing rows remain missing.
These are resource-limited prices, not complete comparisons or posterior checks.
SIR local epsilon .2 cost 84.72/195.17 seconds at L=3/9 for 32 transitions;
PP-SGQF local prices were approximately 37/84 seconds. The large shared-device
cost must be included in future complete-fit forecasts.

The independently implemented SIR likelihood and two-spacing score checks
passed (value error 6.82e-13; scaled score errors 1.94e-5 and 1.87e-7). All 275
declared wider support probes were finite. This finite check does not prove
global support validity. Its first integration used the correct standardized
measure, including the `3 log(.5)` Jacobian, and compared physical observation
**standard deviation**, not variance. It remained reference-limited, with
maximum final mean change .004416 SD versus .002 required. The new SIR
reference uses fresh scrambles at powers 12/14/16, a 3,600-second cap and the
same frozen proposal. A failed support-artifact serialization attempt consumed
5.93 seconds and is preserved; its `.tolist()` diagnostic-boundary repair
passed focused regressions.

The auxiliary Gaussian-mixture PP references improved their descriptive
refinement diagnostics but neither qualified at power 18:

| Numerical target | Integration worker seconds | Largest final uncertainty half-width / SD | Largest final order change / SD | Disposition |
| --- | ---: | ---: | ---: | --- |
| PP-UKF | 757.49 | .005836 | .003476 | Reference-limited |
| PP-SGQF | 533.87 | .002433 | .001761 | Reference-limited |

The requirement remains .002 for both measures. SGQF's new frozen-proposal
refinement uses powers 18/20 and eight fresh scrambles under 2,400 seconds.
UKF's largest uncertainty is the fifth source-coordinate second moment;
several scramble estimates differ despite large global importance ESS. This
localizes the precision problem to a tail-sensitive quantity, but does not
establish its cause or justify dropping the quantity. Structural's frozen
proposal refinement is also still running at power 22. No unfinished
reference can judge sampler accuracy.

The D1 full-model consumer is implemented and tested. It issues fresh immutable
fixed/randomized bindings from the existing checked geometry, calls the public
tuner, retains and reloads every verified member and applies the common
posterior controller to all 18 quantities directly from native source draws.
It preserves incomplete candidates, committed chunks and the two-cell planned
denominator. Nineteen focused tests cover numerical mappings, actual nonlinear
bindings, both public lifecycles, resource interruption, member accounting and
reference rejection/assessment. Two test-stage problems were repaired before
launch: an incorrect test-module import and use of the extra-quantity API for
quantities already supplied as model coordinates. They changed no sampler or
acceptance policy. Seven monitor tests pass. The command and exact test logs
are recorded in the active campaign's D1 section.

`N5/full-pp-sgqf-pricing-01` is now queued under the recorded 72,280-second
allocation. It waits for the completed GPU trajectory supervisor and a passing
matched PP-SGQF reference, then uses GPU 0; GPU 1 remains assigned to D0. The
24-hour passive monitor is `N5/continuation-watch-05`; it reports committed
posterior chunks and alerts for resource stops, reference limits and failed
children. It neither grants time nor retries numerical work.

The 10:38 budget snapshot charges 10.74 occupied worker-hours, reserves 16.68
remaining hours for the other active jobs and 20.08 for D1, with 24.34 hours
unallocated after the unchanged ten-minute overhead reserve. See
`N5/campaign-72h-audit/budget-ledger-20261008T103828.json`. Passive waiting and
monitoring are excluded; parallel numerical workers add, and child time is
not charged again on top of its parent. This snapshot precedes later progress.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue D0 | 11/32 fits complete, all eleven member screens pass | No completed-fit numerical veto so far | Remaining seeds and optional-stopping precision | Complete the frozen denominator and terminal paired report | Method superiority or general reliability |
| Close saved SIR mechanics recovery | Two members verified/exported/reloaded | No new numerical veto; prior caps charged | Posterior exploration remains untested | Use scoped local geometry and a qualified reference for posterior work | SIR posterior accuracy |
| Keep full-model references under refinement | Value/score parity passes; earlier integrals fail .002 | Reference limits veto accuracy interpretation | Tail-sensitive integral stability | Finish fresh frozen-proposal orders before adjudicating fits | Exact nonlinear posterior or rigorous integration error bound |
| Queue full PP-SGQF D1 | Public lifecycle and 18-quantity mechanics tests pass | Reference must qualify before numerical launch | Shared-device cost, posterior precision and global coverage | Execute both bounded cells when prerequisites pass | Fully tuned efficiency comparison or release |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Preserved reference limits and resource caps; no passing label for missing work |
| Statistically supported ranking | None; short-model replications are unfinished and D1 has one paired seed |
| Descriptive-only differences | Runtime, trajectory acceptance, importance ESS and reference-rung differences |
| Default-readiness | Not established; modern transition profiles remain experimental |
| Next evidence needed | Matched reference qualification, full-model stopped posterior agreement, replicated outcomes and the remaining method/model matrix |

Post-run red-team note: accurate local values and healthy short trajectories
can coexist with missing posterior regions; a high importance ESS can coexist
with unstable second moments. Independent tail quantities and fresh refinement
reduce these risks but cannot prove global coverage. D1's narrow measured grid
is a price and viability experiment. A failed or capped cell motivates its
classified next repair; it does not reject the sampler family or silently
relax the reference, warm-up or precision requirements.

### Terminal reference update and active state at 11:07 UTC

The refined PP-SGQF reference finished in 1,949 seconds including supervision.
Its final half-width and mean change are .00065254 and .00047042 reference SD.
SIR finished in 2,989 seconds including supervision, with .00007999 and
.00041861 respectively. Both pass the predeclared .002 empirical refinement
requirement for their respective numerical filter posteriors. Exact generative
posterior accuracy and rigorous integral-error bounds remain unproved.

Structural-reference-03 finished in approximately 7,695 seconds. Every
last-order mean change passes (maximum .001968), but source-square rho and
source-tail sigma half-widths remain .002190 and .002118, so this attempt stays
reference-limited. The latter has one visibly different scramble estimate and
a larger normalized weight; this explains the uncertainty estimate without
proving the cause. Phase E1 tests one further fixed-order refinement, powers
21/23 with fresh seed 130361008, capped at 16,200 CPU-worker seconds. PP-UKF E0
uses powers 18/20/22, fresh seed 120361008 and a 14,400-second cap. Both preserve
their own frozen proposals, all quantities and thresholds. A half-hour of
settled unused SIR-recovery allocation was reassigned to E; the total remains
72 hours. These are the last currently allocated enlargement attempts, not
renewable budgets or repeated-until-pass experiments. Their exact commands
and pre-run audits are in the campaign E0/E1 sections.

The first full PP-SGQF fixed-HMC fit started on GPU 0 at 10:44:49 UTC. The saved
runtime records verified pre-initialization memory growth and GPU/XLA execution.
Its first eleven tuning trial chunks are committed; twelve have been charged,
with the next in progress at the 11:06 snapshot. The randomized arm is queued
within the same fixed two-cell allocation. D0 on GPU 1 has completed 13/32 fits;
all thirteen pass every member's separate posterior and independent agreement
screen. Full-model results and the replicated comparison remain unfinished.

The passive monitor is now `N5/continuation-watch-06`. It retains earlier
terminal limitations and follows both new references and both GPU supervisors.
Eight watcher tests pass, including accurate distinction between charged and
committed tuning chunks. A saved checkpoint's `interrupted` work-item label is
a resumable snapshot, not evidence that the live worker failed. No numerical
retry or allocation extension is triggered by a heartbeat.

The 11:06 budget snapshot is
`N5/campaign-72h-audit/budget-ledger-20261008T110652.json`: 12.43 worker-hours
charged including active elapsed, 42.41 reserved for remaining active
allowances, and 16.99 unallocated after the unchanged 600-second startup/probe
reserve. This includes D1, E0 and E1 once; no supervisor-plus-child double
charge or reset. A contemporaneous source audit checks all 158 D0 and 148 D1
bound source/reference hashes and finds no mismatch.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit PP-SGQF and SIR references for scoped comparison | All declared refinement quantities pass | Native value/score and finite/reference checks pass | Empirical integration uncertainty and unobserved regions | Assess matched stopped posterior quantities | Rigorous integration bound or exact nonlinear posterior |
| Continue full PP-SGQF D1 | Qualified reference and public lifecycle tests; live tuning chunks committed | No numerical/harness veto observed | Full cost, warm-up and retained precision | Finish fixed and randomized cells within their caps | Posterior validity before results or method ranking |
| Refine structural and PP-UKF once more | Prior references fail unchanged quantity-specific precision | Failure preserved; no target/parity veto | Tail weights and scramble variation | Execute the two priced frozen-proposal refinements | Qualified references before their terminal checks |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Structural/UKF earlier references remain limited; current GPU fits have no recorded numerical veto |
| Statistically supported ranking | None; D0 and D1 are unfinished |
| Descriptive-only differences | Reference errors, observed costs, trial progress and observed delivery frequency |
| Default-readiness | Not established |
| Next evidence needed | Terminal D0 denominator/uncertainty report, all D1 member assessments, E0/E1 reference dispositions |

### D0 terminal audit and D1 precision limitation, 13:23 UTC

D0 finished at 12:27:11 UTC: all 32 planned fits completed, all 64 retained
members passed posterior checks and independent agreement screens, and the
terminal outcome audit reports no artifact errors. The occupied numerical
coordinator cost is 15,142.55 seconds (4.21 hours), including independent
assessments and excluding its 960-second prerequisite wait. Full report:
`N5/continuation-watch-06/terminal-replications.json`.

Each of the four target/method groups has eight successes in eight independent
complete fits. The predeclared working-binomial two-sided 95% interval is
[.6306, 1] for each group. This is evidence of viability in these frozen scopes,
not a high-reliability guarantee. Descriptive nominal mean intervals overlap
the numerical reference in 453/480 member-quantity records; these correlated
records are not independent replications and do not establish calibrated
coverage under sequential stopping. There is no supported method ranking.
The initial seed-1000 randomized LGSSM agreement failure remains preserved;
it is not erased or pooled into this separately declared fresh cohort.

The first full PP-SGQF fixed-HMC member (L=3, epsilon=.2) completed warm-up at
2,000 transitions per chain and reached 10,000 retained draws per chain.
It has no hard numerical veto; R-hat and bulk/tail ESS pass. Its retained
precision fails for the physical u/v parameters and their source square/tail
quantities. The worst lugsail MCSE/SD is .02558 for `second_4`, above .02.
Classify this as a posterior precision/count limitation, not an HMC tuning
failure, numerical crash or evidence against the method family. The frozen
10,000-draw cap and failed result remain unchanged. The second verified member
is sampling, with recent committed posterior chunks; the randomized arm remains
queued. Independent assessment of both members follows the fit's terminal
record. A later precision or geometry experiment must preserve these outcomes
and have its own scoped plan; there is no unrecorded extension or retuning.

PP-UKF E0 is on its last power-22 rung, with five of eight scrambles complete
at the snapshot; structural E1 is on power 23, with three of eight complete.
Both are within their fixed allowances, with no recorded numerical errors.
The passive monitor is current. All 158 D0 and 148 D1 bound source/reference
hashes still match. The budget snapshot
`N5/campaign-72h-audit/budget-ledger-20261008T132256.json` charges 20.57 hours,
reserves 21.14 hours for active jobs and leaves 30.12 hours unallocated after
the 600-second overhead reserve. D0's unused allocation was released once;
the 72-hour campaign grant is unchanged.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close D0 frozen replication | 32/32 complete fits and 64/64 members pass | No artifact or numerical veto | Eight seeds per group, fixed short-model datasets | Preserve terminal denominator and continue harder-model evidence | Reliability floor, sequential coverage or method superiority |
| Preserve first D1 member as precision-limited | Lugsail precision fails at retained count cap | No numerical veto; R-hat/ESS pass | Autocorrelation and finite MCSE estimation; reference assessment pending | Continue the second member and randomized arm under unchanged policy | Tuning rejection, sampler-family rejection or posterior delivery |
| Continue E0/E1 | Final reference rungs incomplete | No recorded numerical error | Remaining scramble variation | Finish their original bounded orders | Qualified references before terminal refinement checks |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No D0 numerical/artifact veto; first D1 member fails the posterior precision screen |
| Statistically supported ranking | None |
| Descriptive-only differences | Costs, pointwise interval overlap, MCSE differences and reference progress |
| Default-readiness | Not established |
| Next evidence needed | Both D1 cells' all-member results and independent assessments; terminal E0/E1 references |

Red-team note: 32 short-model successes cannot settle application-scale
geometry or stopping behavior. The D1 precision limit already shows that a
verified, healthy kernel can fail to deliver the requested precision within
the declared count budget. The remaining L=9 and randomized outcomes will
distinguish whether that limitation occurs across this small frozen grid;
none can be selected afterward to hide the failed member.

### N5 batch terminal audit, 9 October 2026

The batch finished at **17:08:59 UTC on 8 October**, or **01:08:59 China time
on 9 October**. D0, D1, E0 and E1 are all terminal, and no worker from this
batch remains active. This closes their allocated attempts; the modern-method
implementation and qualification program remains unfinished.

D0 preserved all 32 planned LGSSM/accumulation fits and all 64 members. Every
member passed the separate posterior and independent reference screens. The
four target/method groups each have eight successes in eight fits; their
working-iid-binomial 95% interval remains [.6306, 1]. The initial seed-1000
randomized LGSSM failure belongs to the earlier pricing cohort and remains
visible. No comparison winner or sequential-coverage claim follows from D0.

D1 ran fixed and randomized HMC on the full twenty-observation PP-SGQF
numerical posterior with the same prepared geometry, four starts, seed 4000
and epsilon .2. Both candidates in each fit passed tuning, export and reload.
All four members passed the declared warm-up checks after 2,000 transitions
per chain. The table reports every member; L is the maximum trajectory count
for randomized HMC.

| Method | L | Retained draws/chain | Posterior precision | Worst reference error screen | Combined posterior/reference outcome |
| --- | ---: | ---: | --- | ---: | --- |
| Fixed HMC | 3 | 10,000 | Insufficient at count cap | .088278 | Not established |
| Fixed HMC | 9 | 4,000 | Passed | .068961 | Passed |
| Randomized HMC | 3 | 10,000 | Insufficient at count cap | .125691 | Not established |
| Randomized HMC | 9 | 6,000 | Passed | .072658 | Passed |

The frozen reference screen is
`(abs(error) + 3*MCSE + reference_uncertainty) / reference_SD <= .10`
for all 18 quantities, together with the separate posterior checks. Fixed L=3
passes that numeric error bound but fails posterior precision, so it correctly
does not receive a combined passing label. Randomized L=3 fails both. All four
members pass retained R-hat and bulk/tail ESS; no hard numerical veto is
recorded. Fixed L=3 has six failing precision quantities and worst lugsail
MCSE/SD .025580 (`second_4`). Randomized L=3 has seventeen and worst .033114
(`upper_4`), against the unchanged .02 limit. This is a posterior precision/count
limitation, not a tuning failure or time-limit failure. All four tuning
qualifications remain intact. D1 therefore records 2/4 passing members,
2/2 fits with at least one passing member, and 0/2 fits with every member
passing. Neither failed member is removed from the denominator.

E0's PP-UKF reference passed at power 22: final maximum pointwise half-width
.00198457 and last-order mean change .00054130 reference SD, both below .002
across all eighteen quantities. It took 11,585.19 occupied worker-seconds
including supervision. This qualifies the empirical numerical-target reference
for a subsequent matched posterior comparison, not the UKF sampler itself.

E1's structural reference remains limited after power 23. Fourteen of fifteen
quantities pass; source-square rho has half-width .00270722 and mean change
.00204413 SD. It took 14,625.42 seconds including supervision. Its eight
rho-square scramble estimates range from .783906 to .790578. The latter
replication has maximum normalized weight .000967 and importance ESS about
639,446 out of 8,388,608 points. This identifies a quantity-sensitive integration
uncertainty; it does not establish its cause. Large global importance ESS is
insufficient to qualify the moment. The next repair is a proposal/tail-weight
diagnosis, not another blind sample enlargement. Earlier PP-SGQF and SIR
reference passes remain available for their matched numerical targets.

The static closeout audit was run with:

```bash
python docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/campaign-72h-audit/closeout-audit-20261009.py
```

It checks every D0/D1 saved member and independent assessment, verifies the
recorded GPU memory-growth policies for all 34 fits, and recomputes final-order
reference changes and half-width/SD ratios. All 158 D0 and 148 D1 source/input
bindings match. E0's 132 source plus four input hashes and E1's 131 source plus
three input hashes also match. No artifact inconsistency is found. Two initial
static audit assumptions were corrected to the actual saved schemas: `verified`
is a list of IDs, and structural refinement ratios are nested. Neither attempt
ran numerical work or changed evidence; their subsecond cost is covered by the
existing overhead reserve. No sampler code changed or new experiment launched
during closeout. The previously passing focused harness and watcher regressions
remain the implementation checks; the new audit verifies actual saved results.

Audit:
`N5/campaign-72h-audit/terminal-audit-20261008T181410.json`.
Settled ledger:
`N5/campaign-72h-audit/budget-ledger-20261008T181410.json`.
These supersede the 13:22 live snapshot for current accounting while preserving
it as history. The campaign has charged **96,550.07 seconds (26.82 hours)**,
including recorded tests and failed attempts, with no active reservations.
After the unchanged 600-second startup/probe/audit reserve,
**162,049.93 seconds (45.01 hours)** remain from the original 72-hour grant.
Final D0 and D1 coordinator costs, excluding passive waits and including small
supervisor overhead, are 15,143.43 and 23,051.44 seconds. This conservatively
exceeds their summed numerical-child charges by .88 and 1.68 seconds,
respectively; the two accounting measures are not added together. All unspent
attempt caps are released once.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close D0 replication | 32/32 fits; 64/64 posterior/reference members pass | No recorded numerical or artifact veto | Eight independent fits per target/method and operational stopping | Preserve results and complete nonlinear evidence | General reliability, coverage or method superiority |
| Close D1 with partial posterior delivery | 2/4 members; 0/2 all-member and 2/2 any-member fits pass | L=3 precision limits; no recorded numerical or harness failure | One paired seed, narrow grid and residual geometry | Keep all four qualified members and design fresh scoped duration/geometry comparisons | Tuning rejection or a speed ranking |
| Accept scoped E0 reference | All eighteen refinement quantities pass | Native parity and source/input checks pass | Empirical integration precision and unseen regions | Implement matched full UKF posterior assessment | Exact latent-state posterior or rigorous error bound |
| Retain E1 reference limitation | Source-square rho fails both refinement criteria | Reference failure vetoes structural agreement claims | Tail contributions and proposal representation | Bounded quantity-weight/proposal diagnosis with fresh validation after any repair | Structural HMC failure or rejection of the research direction |
| Continue the overall program | Batch settled; 45.01 worker-hours remain | No campaign-wide invalidity or exhausted-budget veto | Price and coverage of remaining model/method cells | Execute the refreshed phase sequence after each concrete design is priced | Completed N5, LAPS support, release or default readiness |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No new numerical/harness veto in D0/D1; D1 precision failures and E1 reference limitation remain explicit |
| Statistically supported ranking | None; neither D0 nor the single D1 seed establishes a winning method |
| Descriptive-only differences | Retained draw counts, runtimes, MCSE ratios, per-scramble weights and nominal interval overlap |
| Default-readiness | Not established; modern transition profiles remain experimental |
| Next evidence needed | Structural reference repair, full UKF/SIR member assessments, missing nonlinear methods and fresh replicated comparisons |

Post-run red-team note: the strongest alternative explanation for attractive
L=9 results is favorable behavior on this one geometry/seed rather than a
general duration or method advantage. Different starts, datasets or seeds may
overturn it. The weakest reference evidence is the unresolved structural
rho-square moment; no structural posterior-agreement claim is allowed using
it. These results reject two precision-limited member outcomes and one reference
attempt, not the harness, targets or research direction. The refreshed campaign
specifies the next discriminating repairs under the unchanged budget.

### E2/D2 implementation and live diagnostics, 9 October

The full-model fit and assessment consumers now support PP-UKF and SIR through
the existing public tuner and posterior controller. PP uses its six native
probit coordinates and eighteen physical/square/tail quantities. SIR uses three
native log parameters, maps physical rates with `[.1,18,10]*exp(theta)`, and
forms its source moments from `z=theta/.5`, matching the independent reference.
All nine SIR quantities enter posterior assessment. The new targets require
explicit candidate pairs and matched reference/geometry paths; they cannot
silently inherit the original PP-SGQF grid. No numerical target, sampler kernel,
tuning rule or posterior precision threshold changed.

Reference validation checks actual final two integration orders, every quantity,
target/dataset identity, parity and bound source files. Both saved PP-UKF and
SIR references validate under the extended consumer. Export/reload still
preserves every verified member, and the supervisor retains incomplete member
and fit outcomes. The queue can run an explicit target under its allocated cap;
the read-only watcher now accepts the current job list and reports diagnostic
failures and full SIR posterior/tuning progress.

Focused checks: structural contribution/mixture reporting passed nine tests in
2.57 seconds. The expanded full-model suite passed 37 cases; one source-identity
guard stopped a test because the fit script's cap validation was edited while
that test ran. The file was then frozen. That case, the new explicit-target
queue regression and all watcher tests passed together: eleven tests in 28.41
seconds. The genuine SIR affine test initially tried a shortened horizon and
was correctly rejected by SIR's fixed `[20,9]` observation requirement. It now
uses the full horizon and passed in 117.96 seconds; the failed fixture attempt
cost 3.65 seconds. The original expanded run's 282.61 seconds and both failed
checks remain charged. These tests establish mechanics and matched transforms,
not posterior accuracy or performance rankings.

E2 is running as `bayesfilter-modern-hmc-n5-structural-tail-01.service`, with
CPU/GPU visibility deliberately restricted to CPU. It replays the two declared
power-23 scrambles under a 4,200-second cap. Its NumPy reference code is explicitly
independent and diagnostic. It reports every original weight's contribution,
including component and source-rho partitions, and checks the native value/score
at the 64 dominant quantity-influence points. No replay counts as fresh evidence.
The supervisor preserves a failure or cap and creates no automatic third attempt.

D2a is running as `bayesfilter-modern-hmc-n5-current-price-01.service` on GPU 0,
with a total 3,600-second cap. The fresh trusted probe verified GPU availability
and memory growth. PP-UKF completed all eight measured calls with healthy
finite/status checks. Repeated 64-transition costs were 21.04/52.68 seconds for
fixed L=3/9 and 20.12/45.26 for randomized maximum-L=3/9; the first calls include
extra tracing/compilation. SIR follows on the same device. These measurements
inform fit allocation and cannot issue tuning or posterior qualification.
The monitor is `N5/continuation-watch-07/`; it neither grants extra time nor
changes numerical policy.

The 19:13 UTC budget snapshot is
`N5/campaign-72h-audit/budget-ledger-20261008T191344.json`: 27.46 hours charged
including live elapsed, 1.64 hours reserved for the remaining diagnostic caps,
and 42.73 hours unallocated after the unchanged 600-second overhead reserve.
The updated complete 25-cell inventory is
`N5/campaign-72h-audit/nonlinear-comparison-matrix-20261008T191436.json`.
It retains five scoped accumulation passes, two partial PP-SGQF deliveries,
four UKF/SIR cells awaiting full-fit allocation, five structural cells limited
by the reference, and nine unrun nonlinear comparator cells. No missing cell
is removed, and no screen pass implies a release or method ranking.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Use the expanded full-model harness | Matched transforms, public lifecycle and reference rejection regressions pass | Source/horizon guards exposed and rejected invalid test setups | Full GPU fit cost and posterior precision | Finish D2a pricing, then freeze target-specific fit budgets | Full UKF/SIR posterior validity |
| Continue E2 | Frozen replay in progress | No failure recorded at the live snapshot | Dominant weight locations and native/reference parity | Inspect both replay results before proposal repair | Reference qualification from replay |
| Continue D2a | UKF repeated-call measurements healthy; SIR active | No UKF numerical/status veto | Time variation, compilation and full posterior counts | Use measured complete-work estimates with bounded fit allowances | GPU throughput stability or sampler superiority |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No live diagnostic numerical failure observed; source/horizon test rejections remain engineering evidence |
| Statistically supported ranking | None |
| Descriptive-only differences | Short-call timings and eventual importance-contribution partitions |
| Default-readiness | Not established |
| Next evidence needed | Terminal E2 decomposition, terminal SIR timing, actual full-model stopped posterior outcomes |

### E2a resolution and E2b/D2b launch, 9 October

This entry supersedes the preceding live snapshot. E2 stopped after reproducing
scramble 3 in 1,469.181 seconds. The top 64 quantity-influence points carry
.00268757 of the normalized mass and .989632 of the descriptive squared
centered influence for source-square rho. These are selected diagnostics, not
a QMC variance estimator. Native/reference values agree within 1.10808e-9,
but the original central-difference steps gave scaled gradient errors .17645
and .00143037, exceeding the unchanged 2e-5 check. The failed receipt remains.

The predeclared E2a localization used the same saved coordinates and scalar
target. Five-point estimates at 1e-5 and 3e-6 give maximum scaled errors
1.07764e-5 and 8.91015e-8; their mutual discrepancy is 1.06872e-5. Both pass
the original tolerance. The central-difference ladder decreases consistently
to 4.49849e-7 at 1e-7. This supports coarse finite-difference truncation as the
cause of this local failure. It does not establish general score correctness
or resolve structural reference uncertainty. E2a cost 11.656 supervised seconds;
its full derivative arrays are in `N5/structural-tail-parity-01/diagnostic/result.json`.

D2a finished at 19:21:02 UTC in 966.487 supervised seconds. All sixteen
PP-UKF/SIR calls passed their numerical/status checks. Repeated-call costs per
64 transitions were 21.04/52.68 seconds for UKF fixed L=3/9 and 20.12/45.26
for randomized maximum L=3/9. SIR costs were 31.91/79.77 and 30.44/68.54 seconds,
respectively. First-call and repeated-call rows, raw tensors, target identities,
GPU runtime and allocator receipts remain in `N5/full-model-current-price-01/`.
These calls grant no tuning or posterior authority and support no method ranking.

The campaign's E2b/D2b section records the skeptical audit, measured runtime
scenarios, exact commands, criteria, stop conditions and allocation before launch.
Fourteen focused queue/watcher regressions passed in .25 seconds, including
retention of partial outcomes, stopping on execution failures, memory-growth
enforcement and hash-bound reuse of the checked first replay. Compilation and
whitespace checks passed. The actual first replay/parity source closure passed
145 hash checks before continuation. No target, integrator, tuning rule or
posterior criterion changed.

E2b started at 19:32:33 UTC as
`bayesfilter-modern-hmc-n5-structural-tail-02.service`, under 2,700 CPU-worker
seconds, output `N5/structural-tail-diagnostic-02/`. It replays only scramble 1,
calls the existing higher-order diagnostic on its top points, and compares it
with the saved first scramble. New independent replications remain zero.

D2b started at 19:32:50 UTC as
`bayesfilter-modern-hmc-n5-full-models-01.service`. It executes UKF then SIR,
fixed then randomized per target, with 24,000/36,000-second per-fit caps and
120,680 seconds reserved for all four fits, assessments and coordination.
The coordinator `N5/full-model-continuation-01/` points to
`N5/full-pp-ukf-pricing-01/` and `N5/full-sir-pricing-01/`. The first UKF fixed
fit is actively tuning; its GPU/XLA design verifies memory growth before
initialization and records the actual device. Each subsequent fit performs its
own resource probe. No unrelated process is stopped or numerical dependency
edited during these jobs.

`N5/continuation-watch-08/` records live checkpoints, child failures, posterior
chunks and terminal outcomes every 30 seconds. It is a read-only monitor, not
an autonomous policy extender. The authoritative pre-launch ledger is
`N5/campaign-72h-audit/budget-ledger-20261008T193213.json`: 99,432.85 seconds
charged, 123,380 reserved, 600 overhead reserved and 35,787.15 unallocated.
Nested worker receipts are not charged again. The total remains 259,200 seconds.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Resolve the first replay's derivative diagnostic | Both fixed five-point steps and mutual consistency pass | Native value/status and source checks pass | Evidence covers only the saved 64 points | Complete the second selected replay | General score correctness or a qualified structural reference |
| Start matched full UKF/SIR fits | Target-specific pricing and harness regressions pass | No D2a numerical veto; verified GPU memory growth | Posterior counts, contention and local-geometry coverage | Evaluate every qualified member with unchanged posterior/reference screens | Tuning or posterior admission from the price, superiority or release readiness |
| Continue structural diagnosis | First replay preserved, remaining replay active | Original coarse-step failure resolved locally | Tail representation and denominator variability | Interpret both selected decompositions before fresh proposal calibration | Independent validation from selected scrambles |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No native numerical failure found at the saved points or in D2a; original failed diagnostic preserved |
| Statistically supported ranking | None |
| Descriptive-only differences | GPU prices, central-difference step sensitivity and selected importance contributions |
| Default-readiness | Not established |
| Next evidence needed | Terminal second replay, all full UKF/SIR member outcomes, fresh proposal validation if indicated and missing nonlinear comparator cells |

Post-run red-team note: a passing local gradient diagnostic does not prove
global agreement, and stable short paths may still mix slowly. The next full
fits and the selected replay answer separate questions. A failed source check
or numerical disagreement would reopen the affected stage; insufficient posterior
precision or reference refinement triggers its planned repair without invalidating
the whole research direction.

### E2c/D2c repair and restart, 9 October, 09:19 China time

D2b stopped after its completed UKF fixed fit. Both candidates were tuning-
qualified and retained. L=3 passed warm-up, R-hat, ESS and numerical checks but
reached 10,000 retained transitions per chain with lugsail precision insufficient.
L=9 passed the separate posterior/reference screens after 2,000 warm-up and
3,000 retained transitions. Their numerical reference-error screens were .08938
and .06277 SD; the first still fails overall because posterior precision did
not pass. There are no numerical hard vetoes or artifact errors. Keep the
two-member denominator and the incomplete all-member fit outcome.

The randomized fit's preflight failed at 23:07:50 UTC: the trusted GPU probe
required GPU 1 once it became idle, while the queue still requested GPU 0.
No numerical worker, tuning or sampling started for that fit; neither SIR fit
started. This is a resource-selection failure, not evidence against randomized
HMC. D2b consumed 12,900.826 outer-supervisor seconds. Its original failed
preflight and queue receipts remain untouched.

E2b reproduced the second selected structural integral in 1,469.851 seconds,
then failed the original five-point step check. The discrepancy fell from
.000402422 at h=1e-5 to .00000328548 at h=3e-6, consistent with truncation
error, and native/reference values agreed within 2.17e-10. E2c checked all 128
saved points from both scrambles using the fixed smaller five-point steps
1e-6 and 3e-7, plus central differences at 1e-7. Maximum scaled errors were
1.47653e-7, 2.51044e-7 and 1.01567e-6; the two higher-order estimates differed
by at most 2.23585e-7. All pass the unchanged 2e-5 tolerance. E2c consumed
11.105 seconds. These checks resolve local parity, not global score correctness
or the structural reference's failed .002-SD refinement.

The two replays give source-square rho means .790578 and .783906. Their top64
points account for 98.96% and 77.59% of descriptive squared centered influence,
but only .269% and .127% of normalized mass. The |source-rho| in [2,3) band
accounts for 83.70% and 79.43% of squared influence. This motivates a proposal
with better representation of the influential tail region. It is neither a
QMC variance estimate nor proof of absent posterior mass. Any proposal repair
must train on fresh calibration draws and pass unchanged all-quantity reference
criteria on fresh validation scrambles. New independent replications from these
selected replays are zero. The static closeout is
`N5/campaign-72h-audit/structural-tail-closeout-20261009T012139.json`.

The new resource coordinator calls the trusted automatic GPU selector before
each fresh fit and records the physical UUID and hardware class. Its bounded
retry accepts only recognized resource/preference failures before a numerical
worker or artifact exists. Attempts and wait time consume the same cell cap.
It never retries numerical failures, migrates live chains, discards failed
candidates or changes the target, tuning or posterior policy. Existing shared
GPU admission still requires the recorded 4,096 MiB headroom and memory growth.
Other policy, driver and visibility failures stop for diagnosis.

Focused regressions passed **27 tests in 2.13 seconds**, including changing GPU
preference, both-busy shared admission, fail-closed driver/hardware conditions,
retry exclusion after numerical launch, decreasing per-cell remaining time,
preserved partial outcomes, nested checkpoint monitoring and independent
polynomial derivative checks. All 143 source hashes of the completed fixed fit
still match, and both empirical references validate. No bound numerical source
was edited. Compilation and whitespace checks passed.

D2c began at 01:19:25 UTC under
`bayesfilter-modern-hmc-n5-full-recovery-01.service`, output
`N5/full-model-recovery-01/`. UKF randomized is now tuning on physical GPU 1;
the runtime verifies memory growth and GPU/XLA execution. SIR fixed and randomized
follow sequentially. The original cell caps remain 24,000/36,000 seconds;
96,600 seconds cover the three remaining fits and assessment/coordination.
The repaired monitor is `N5/continuation-watch-09/`: it exposes the active nested
fit's actual checkpoints and marks the old unstarted SIR queue as terminal after
its parent failure, preserving the reason. It does not grant more compute or
change numerical criteria.

The ledger `N5/campaign-72h-audit/budget-ledger-20261009T011905.json` charges
113,816.758 seconds (31.6158 hours), reserves 96,600 seconds for D2c and the
unchanged 600 seconds for overhead, leaving 48,183.242 seconds (13.3842 hours)
unallocated. It settles E2b/D2b once and releases unused caps. The grant remains
259,200 aggregate CPU/GPU worker-wall seconds. The 25-cell matrix is refreshed
in `N5/campaign-72h-audit/nonlinear-comparison-matrix-20261009T012139.json`.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve UKF fixed result | 1/2 posterior/reference members pass; both tuning members retained | No numerical or artifact veto; L=3 precision insufficient | One seed and local geometry | Finish matched randomized and SIR fits | All-member delivery, broad tuning adequacy or superiority |
| Resolve selected-replay parity | All 128 saved points pass three fixed derivative checks | Values, statuses and unchanged source checks pass | Unsampled coordinates and reference integration variance | Fresh proposal calibration and validation | Qualified structural reference or global gradient proof |
| Repair and resume D2 | Device selection and bounded retry tests pass; randomized fit active | Original preflight failure preserved; numerical failures never auto-retry | Shared workload and posterior count requirements | Monitor and assess all three unfinished fits | Completion or release readiness |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No numerical failure in completed UKF fixed; the GPU preflight failure occurred before sampling |
| Statistically supported ranking | None |
| Descriptive-only differences | Single-fit draw counts and cost, selected influence concentration and finite-difference errors |
| Default-readiness | Not established |
| Next evidence needed | Remaining full fits, fresh structural proposal validation and missing nonlinear comparator cells |

Post-run red-team note: the fixed L=9 pass can reflect this seed and geometry;
it does not authorize eliminating L=3. The repaired launcher solves a known
resource-selection race, while new numerical or reference failures remain real
stops. The local derivative plateau supports diagnostic truncation/roundoff,
but the independent integral still lacks sufficient all-quantity precision.

### D2d scheduling transfer, 9 October, 10:04 China time

The owner requested use of GPU capacity. The serial queue imposed no scientific
dependency between UKF and SIR. A trusted 01:35 UTC snapshot had shown GPU 0
idle, but an unrelated process occupied it by the 01:57 recheck. GPU 1 had
about 31 GiB free with UKF actively computing. The existing shared-device
policy admitted SIR on physical GPU 1; this records resource availability,
not evidence that sharing improves throughput.

`bayesfilter-modern-hmc-n5-full-parallel-01.service` started at 02:04:36 UTC
using `scripts/parallelize_modern_hmc_full_models.py`, with the campaign plan's
exact command and the unchanged 96,600-second outer service limit. It paused
only old scheduler PID 656249. UKF supervisor 656262 and numerical worker 656269
continued with the same target, seed, chains, candidates, source closure and
24,000-second allowance. The new observer assesses that original fit after it
finishes and retires the old service. SIR fixed launched as numerical worker
691807 on GPU UUID `GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c`; SIR randomized
follows sequentially with its own fresh device selection. Each SIR fit keeps
its 36,000-second cap. No numerical fit was duplicated or restarted.
SIR's saved design verifies memory growth before TensorFlow initialization and
GPU execution; its log records XLA compilation. The first post-transfer monitor
snapshots show UKF tuning progress from 95 to 98 charged chunks. The launch
check is `N5/full-model-parallel-01/launch-verification.json`.

The first scheduling regression exposed a completion race in the new
supervisor's final status scan. After repair, **47 tests passed in 3.03 seconds**.
They include real stopped-parent/continuing-child behavior, safe scheduler
retirement, rollback on manifest and thread-setup failures, additive overlap
accounting, and the existing device-selection, retry and monitoring checks.
Both test attempts, including interpreter startup, cost 8.702 seconds and are
charged. Compile/whitespace checks passed; the completed and active UKF closures
each retain all 143 unchanged source hashes and unchanged reference hashes.
Only the new scheduling script, its tests and campaign records were changed.

The transfer ledger
`N5/campaign-72h-audit/budget-ledger-20261009T020436546444.json` charges
116,536.407 seconds, including D2c's 2,710.947-second prefix once. It transfers
93,889.053 seconds to D2d, reserves the same 600-second overhead, and leaves
48,174.540 seconds unallocated from the original 259,200-second grant. D2d
charges remaining UKF time plus SIR time, including overlap; it does not charge
the paused scheduler or nested workers again.

`N5/full-model-parallel-01/` holds the transfer and aggregate progress;
`N5/full-ukf-adopted-01/` observes the original UKF attempt, and
`N5/full-sir-parallel-01/` holds the SIR sequence. The active read-only monitor
is `N5/continuation-watch-10/`. Watcher 09 was stopped after watcher 10 started,
with a supersession receipt; its historical observations remain preserved.
The old queue manifest remains unchanged, accompanied by a transfer receipt.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Run the pending independent sequence concurrently | Scheduling regressions pass; original UKF continues and SIR has launched | Source/reference checks pass; no duplicate fit; memory growth required | Shared-device throughput and posterior count needs | Monitor both workers and assess every retained member | Faster completion, posterior success or release readiness |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No new numerical result; launch and source checks passed |
| Statistically supported ranking | None |
| Descriptive-only differences | GPU occupancy, elapsed times and checkpoint progress |
| Default-readiness | Unchanged; not established |
| Next evidence needed | Terminal UKF/SIR results and their separate posterior/reference assessments |

Post-launch red-team note: free VRAM is not unused compute. The launch can
reduce idle scheduling time while increasing each fit's elapsed time through
contention. The unchanged caps and cumulative budget remain binding; an
incomplete fit stays in the denominator. R-hat, ESS, MCSE and reference error
remain separate posterior diagnostics, with no tuning admission or ranking role.

### D2d terminal outcome and D2e closeout, 9 October, 23:19 China time

The SIR sequence finished at 13:53:27 UTC. Both fixed-HMC members passed the
declared posterior and matched-reference screens; L=3 used 8,000 retained
draws per chain and L=9 used 1,000, each after 2,000 warm-up draws. Randomized
L=9 passed after 3,000 retained draws; randomized L=3 reached 10,000 with
precision insufficient. Its R-hat, bulk/tail ESS, warm-up and numerical health
checks passed. All four tuning members stay retained. Single-seed draw counts
and costs provide no supported method ranking.

UKF randomized reached its original 24,000-second wall allowance at 07:59:26
UTC. L=3 finished with 4,000 warm-up and 10,000 retained draws but failed the
lugsail precision requirement; its other declared checks passed and its
persisted hard-veto list is empty. L=9 was interrupted after one committed
chunk, with no terminal posterior result. The overall fit result is absent.
Neither candidate is removed from the verified tuning set. No UKF randomized
posterior/reference pass is established; absence of its unfinished member's
terminal health report is not evidence that every attempted transition passed.

The observer separately failed during old-service retirement: TERM had already
removed the transient unit before `systemctl stop`, which returned exit code 5.
The launcher now accepts this race only after a second systemd check proves
MainPID zero, LoadState not-found, ActiveState inactive and no matching saved
scheduler process. Unexpected codes, a loaded/active unit and a live or replaced
process remain errors. **54 focused regressions passed in 3.17 seconds**; their
measured subprocess cost, including startup, was 4.527 seconds. Compilation
and whitespace checks passed. Numerical and reference source hashes of all
four original fits still match. No numerical or posterior source was changed.

The CPU-only versioned closeout is `N5/full-model-closeout-01/`, executed with
CUDA hidden using its preserved `closeout.py`. It records the retirement repair,
all original members, partial UKF progress and original terminal SIR assessments.
It does not manufacture the missing numerical result or claim an independent
UKF reference assessment. The original failed supervisor/watch receipts remain
unchanged. Across the four UKF/SIR fits, four of eight original members pass
posterior/reference screens and one of four fits passes every member. Those
counts describe this paired-seed campaign; they are not reliability estimates.

D2d spent 63,820.863 aggregate occupied seconds: 21,289.907 remaining UKF
observer seconds plus 42,530.956 SIR-sequence seconds. Its 42,533.279-second
calendar duration is not the compute charge. D2c's prior 2,710.947-second prefix
was not charged again. The settled ledger
`N5/campaign-72h-audit/budget-ledger-20261009T151939652244.json` charges
180,361.905 seconds (50.10 hours), releases the unused D2d reservation and leaves
78,238.095 seconds (21.73 hours) after the unchanged 600-second overhead reserve.
No numerical worker in this queue remains active. The nonlinear comparison
matrix has been refreshed with terminal, precision-limited and interrupted
dispositions rather than stale queued/running labels.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve SIR fixed as a scoped pass | Both original members pass posterior/reference | No recorded member or artifact veto | One seed and narrow proposal grid | Include in fresh matched replication/comparators | Superiority or release readiness |
| Preserve SIR randomized and UKF L=3 failures | Count caps reached with precision insufficient | Other terminal checks pass; no persisted hard veto | Longer duration or residual geometry may change precision | Predeclare duration/geometry study while retaining failures | Failed epsilon/L qualification or failed research direction |
| Close UKF randomized as interrupted | No completed all-member fit or L=9 terminal result | Wall cap; incomplete terminal diagnostics | Supported stateful continuation and remaining cost | Audit checkpoint-continuation support before a separate funded run | Posterior pass or complete health evidence |
| Repair transient-service cleanup | 54 focused tests pass; actual unit proven gone | Genuine cleanup and process-identity errors still reject | Future systemd state races | Use repaired retirement for later scheduling | Repair of UKF precision or sampling completeness |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No persisted numerical veto in finalized members; interrupted UKF L=9 remains unassessed |
| Statistically supported ranking | None |
| Descriptive-only differences | Draw counts, single-seed reference screens and shared-device elapsed times |
| Default-readiness | Not established; N5 and subsequent method phases remain open |
| Next evidence needed | Supported UKF continuation, declared precision study, structural proposal repair and missing nonlinear comparators |

Post-run red-team note: concurrent execution and parameter-specific mixing both
could explain the UKF wall cap; these runs cannot separate their effects or
establish faster throughput. The cleanup bug hid the partial outcome but did
not cause the original sample-count or wall-cap failures. Its repair changes
engineering correctness, not the scientific verdict.

### Additional 48-hour continuation, 10 October, in progress

The owner added 48 aggregate numerical-worker hours. The cumulative N5 ceiling
is now 120 hours (432,000 seconds); the settled 50.10-hour history is preserved.
The [supplement](bayesfilter-modern-hmc-n5-additional-48h-2026-10-10.md) contains
the evidence contract, numerical/default audit, source-preserving repair plan,
phase allocations and skeptical review. `N5/campaign-120h-audit/budget-ledger-03.json`
is the immutable reservation baseline; `N5/supplement-watch-02/` recomputes live
additive occupied costs without charging nested children twice. Passive test
waiting is separate. The observer records repair alerts and terminal outcomes;
it launches no automatic numerical retries.

D2f launched at 17:55:58 UTC on October 9 (01:55:58 China time on October 10).
`N5/ukf-member-continuation-01/` loads the original verified randomized UKF L=9
member and reconstructs its complete original policy, seeds, source identity,
GPU identity and reference. It copies the inactive checkpoint store under its
writer lock, replays the committed first 1,000 transitions and computes missing
chunks through the shared posterior controller. It does not rerun L=3, retune,
alter count/precision caps or manufacture the original missing fit result.
The first new chunk took 722.02 seconds; subsequent completed chunks took about
687–701 seconds. These are descriptive prices under current load, not a speed
comparison or duration guarantee. Terminal posterior and reference checks remain
pending. All 143 original numerical source hashes and the saved reference still
match. Memory growth and GPU/XLA are recorded in the new design.

Eight native continuation regressions passed, including same-seed uninterrupted
equivalence, skipped committed computation, original-store immutability,
seed/count/member/quantity mismatch rejection, corruption rejection, writer-lock
and existing-output rejection, and bounded process cleanup. The first run cost
50.786 seconds including startup. A subsequent combined mechanics check passed
15 tests; its unmeasured startup is covered by a conservative 120-second charge.

D3 retains the nine planned full UKF/SGQF/SIR comparator cells. The existing
NUTS diagnostic authority explicitly rejects `per_chain_step` filter-status
obligations. Those three cells are unsupported before numerical work; changing
the status policy would invalidate the comparison. Six SNAPER/MAMS cells use
the existing native authorities and full target-specific data/geometry/starts.
The short price hypotheses and their limits are declared before launch. The
CPU full-horizon regression first exceeded 240 seconds, then a rerun was
interrupted during setup; both are charged conservatively and retained as
failed/incomplete engineering attempts. No sampler verdict follows. The explicit
GPU 0/XLA integration run has passed SGQF and UKF model/geometry/status checks
and is testing SIR. Its 1,800-second cap remains enforced. A bounded coordinator
at `N5/full-comparator-prices-01/` waits for that result; only a successful
regression starts the six-cell price queue. Four queue/accounting regressions
pass, including failed-test rejection and passive-wait exclusion.

E3 launched at 18:48:49 UTC with a 12,000-second CPU-worker allowance.
`N5/structural-influence-reference-01/` uses two fresh power-15 calibration rounds,
with seeds 21261010 and 22261010, to fit an influence-weighted proposal for all
fifteen quantities. The fixed prior component and exact complete-mixture
importance denominator are preserved. Calibration finished in 12.239 seconds.
The frozen proposal is now evaluated with untouched eight-scramble power-19/21
validation, seed 230361010; the first order has completed. The unchanged .002-SD
precision/refinement screen remains decisive. The new proposal is a reference
sampling device, not a modified target or sampler geometry. Fifteen focused
reference controls pass, including exact Gaussian denominator checks, closed-form
influence weighting, positive covariance/prior preservation, invalid scale
rejection and component-seed separation. An earlier malformed two-dimensional
test fixture failed and was repaired to the required five dimensions; its failure
receipt and cost remain recorded.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue interrupted UKF L=9 | Original policy/member replay works; sampling active | Source/reference unchanged; terminal health pending | Remaining warm-up and precision | Complete unchanged controller and independent assessment | Original fit completion or posterior pass |
| Price supported modern nonlinear arms | Full SGQF/UKF wiring passes; SIR integration pending | Three NUTS cells unsupported for internal status | Target-specific pilot and runtime behavior | Start six-cell price only after regression success | Tuning qualification or method ranking |
| Repair structural reference proposal | Fresh calibration frozen; validation underway | Fifteen controls pass; final refinement pending | Tail coverage and scramble uncertainty | Assess every declared quantity at both orders | Exact nonlinear posterior or optimal proposal |
| Preserve resource accounting | New grant recorded without resetting old costs | Every active reservation bounded | Future contention and reference cost | Refresh after each terminal phase | Independent 48-hour grants per device |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Original completed-member outcomes preserved; new terminal results pending |
| Statistically supported ranking | None |
| Descriptive-only differences | Chunk costs, short prices and one-seed posterior outcomes |
| Default-readiness | Not established; methods remain experimental |
| Next evidence needed | D2f terminal assessment, D3 prices, E3 reference validation and a fresh full-fit/precision-study plan |

Interim red-team note: cheaper resumed chunks can reflect reduced contention,
not better mixing. Influence fitting can still miss rare tails; its unused
validation scrambles and unchanged quantity-level screen must decide whether it
helps. Source inspection also found that binding `_probe` calls scalar adapters
eagerly for four starts, repeated by nested preparation consumers; this explains
a plausible setup bottleneck but has not been separately profiled. A future
bounded preparation repair must preserve value/score/status parity and stable
compiled signatures, and must not edit numerical source closures while these
saved-member/reference continuations are active.

### D2f passed; D3 GPU integration passed and price queue recovered

At 19:06:50 UTC on October 9 (03:06:50 China time, October 10), D2f completed.
The unchanged randomized UKF L=9 member passed its separate posterior and
independent-reference checks: 2,000 warm-up and 5,000 retained draws per chain,
no hard vetoes, worst agreement screen .082159 against the unchanged .10 limit.
The new run replayed one committed chunk and computed six; original checkpoint
checksums still match. Its total additional occupied cost was 4,252.341 seconds,
including CPU reference assessment. L=3 remains precision-limited, and the
original overall fit result remains absent. A separately linked result now
supports five posterior/reference passes among the original eight UKF/SIR
candidates, with one of four all-member fits passing. These are descriptive
outcomes from the same original candidate inventory, not a reliability estimate.

All six full-horizon GPU/XLA regressions finished successfully in 1,398.899
subprocess seconds. The first queued price then failed before numerical work:
GPU 1 had become idle, so the local selection policy rejected GPU 0. The failed
attempt cost .318 occupied seconds and 900.039 passive waiting seconds. Its
manifest remains preserved. A fresh trusted inventory selected GPU 1; the
unchanged six-cell price queue launched there at 19:11:35 UTC into
`N5/full-comparator-prices-02/`. E3 remains in power-21 untouched validation on
CPU. Current bookkeeping is `campaign-120h-audit/budget-ledger-04.json`, live
monitoring is `supplement-watch-03/`, and the refreshed matrix is
`campaign-120h-audit/nonlinear-comparison-matrix-02.json`.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept scoped UKF L=9 continuation outcome | Posterior and reference screens pass | No recorded hard veto; source/store checks pass | One member and original paired seed | Preserve alongside L=3 failure in the full inventory | All-member pass, superiority or release readiness |
| Continue six supported comparator prices | Six full-model GPU/XLA integration tests pass | Prior device preflight failure preserved and resolved | Target-specific pilot/short-path behavior | Complete declared price cells on GPU 1 | Tuning or posterior qualification from a short price |
| Continue E3 | Calibration frozen, first validation order complete | Final refinement pending | Tail influence and replicate uncertainty | Complete power 21, retain every quantity and scramble | Reference qualification before terminal assessment |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Resumed UKF member has no hard veto and passes its declared posterior/reference screens |
| Statistically supported ranking | None |
| Descriptive-only differences | Five of eight original candidate outcomes, chunk costs and draw counts |
| Default-readiness | Not established |
| Next evidence needed | E3 reference validation, six supported D3 prices, then a concrete fresh full-fit/precision study |

Post-run red-team note: continuation establishes recovery and scoped posterior
delivery for the same member; it is not a new independent seed. The faster
chunks may reflect changed machine load. The owner excludes NUTS, so its old
filter-status limitation is historical context rather than a remaining gap.
Successful engineering tests do not by themselves establish posterior validity.


## October 10 owner scope correction: NUTS excluded

NUTS was incorrectly introduced as a diagnostic comparator. The owner excludes
it from BayesFilter, including diagnostic execution and comparison requirements.
The [scope repair](bayesfilter-nuts-exclusion-repair-2026-10-10.md) removes its
numerical bodies and retires the older NeuTra and Gaussian-benchmark execution
routes. Historical readers, results and costs remain. Current inventory has
twenty required cells and five separately preserved excluded rows; it lives at
`N5/nuts-exclusion-01/nonlinear-comparison-matrix.json`.

Final focused regressions: **27 passed, 4 opt-in full-model tests skipped**.
The initial discovery failure found a missed benchmark, which was then retired.
All three attempts remain charged: 52.127 seconds. All 153 inspected active
source dependencies still match; no ongoing numerical evidence was invalidated.
The chapter and full monograph build with resolved citations/references.

During this repair D3 finished all six SNAPER/MAMS cells and twelve short paths
without numerical-health vetoes, costing 1,408.462 occupied seconds. These are
engineering prices only; transferred MAMS settings and the short SNAPER pilot
still need target-specific fresh tuning before posterior execution. No method
ranking or release qualification follows. E3 structural validation remains
active. Observer 04 uses the new budget snapshot including the regression charge;
observer 03 is superseded. The master preserves D4's requirement for a concrete
full-fit/precision design after terminal price/reference review.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Enforce owner method scope | All execution exclusion and inventory checks pass | No executable TFP NUTS route found | Future additions require the discovery regression | Keep NUTS excluded | Sampler correctness from a scope test |
| Complete D3 prices | Six approved cells, twelve healthy short paths | No short-path numerical veto | Full tuning and posterior cost | Plan D4 under unchanged posterior checks | Ranking, posterior qualification or default readiness |
| Continue E3 | Frozen proposal and validation progress preserved | No source change | Final fifteen-quantity precision assessment pending | Preserve all scrambles and finish validation | Reference qualification before terminal assessment |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Twelve D3 short paths have no declared numerical-health veto |
| Statistically supported ranking | None |
| Descriptive-only differences | Short-path times and acceptance; no comparable full-fit cost study |
| Default-readiness | Not established |
| Next evidence needed | Terminal E3 assessment and predeclared D4 full fits/precision study |

Post-run red-team note: dropping NUTS is an owner scope correction, not a favorable
performance result. Every approved-method failure and original denominator stays
in the record. Healthy short paths cannot qualify transferred method settings.

## October 10 master refresh and D4a execution

The prior master was stale: it described terminal E3 as running, D3 cells as
queued, and D4 without an executable design. These have been repaired. The
[D4 plan](bayesfilter-modern-hmc-n5-method-proposals-2026-10-10.md) governs the
active continuation. Historical status/results and numerical source closures
were preserved. The fresh twenty-cell snapshot is
`N5/method-study-refresh-01/nonlinear-comparison-matrix.json`.

E3 finished at 19:50:23 UTC on October 9, taking 3693.275 occupied parent seconds,
already charged in observer04. All fifteen quantities pass, including the
previously limiting source-square rho: maximum pointwise half-width/SD is
.0018873181816258845; maximum last-order mean change/SD is
.0014673134087065422, both below the unchanged .002 criteria. Full-horizon
native/independent value-score parity passes. The new consumer recomputes the
criteria from the actual last two rungs, checks all eight scramble identities,
quantity chart, data, horizon100/dimension5, source and proposal inputs. It does
not rely only on a success label or import the NumPy reference into runtime.
This is empirical refinement for the finite UKF posterior, not a rigorous
integration bound or exact nonlinear-model posterior.

Skeptical audit identified two unresolved assumptions before further fits:
D3's 32-step SNAPER pilots all produced L1, and its MAMS epsilon.2 was transferred
from HMC. D4a prices longer discarded SNAPER proposals with a duration
neighborhood and nine native MAMS hypotheses for each of SGQF, UKF, SIR and
structural. Numeric choices and risks are explicit in its plan. These prices
issue no tuning/posterior authority. D4b must use fresh public measurement and
verification, retain every verified member, and assess posterior delivery
separately; its exact cohort/allocation is refreshed from terminal prices.

Focused CPU-only checks: **30 passed, 4 opt-in full-model checks skipped**, measured
26.83336242986843 parent seconds. Tests cover corrupt/limited references,
structural tail quantities, full structural chart/profile/status wiring,
reset-start native MAMS prices, vetoed-member continuation, original eight-cell
denominators across failed/timed-out workers, and the public trajectory-proposer
lifecycle. Gaussian fixtures establish mechanics, not structural inference.
Actual full-model GPU integration is the active eight-cell price queue.

Trusted readiness selected GPU0, with 18 MiB used and no compute process;
GPU1/2 were busy. Launch at 20:19:24 UTC uses service
`bayesfilter-modern-hmc-n5-method-proposals-01.service`, output
`N5/method-proposals-01/`, GPU/XLA, growth set before import and verified in each
worker. The first SGQF SNAPER cell is active. The recorded systemd launch uses
RuntimeMaxSec36060 and control-group cleanup; each child also has bounded process
group cleanup. Observer `bayesfilter-modern-hmc-n5-method-watch-01.service`
writes `N5/method-study-watch-01/` against the immutable D4 budget baseline.
It monitors and alerts; it does not extend policy, retune or launch retries.
No unattended D4b launch is implied.

Before this queue, 193430.28085593166 seconds were charged, including the new
focused tests. Its 36060-second reservation leaves 201909.71914406834 unallocated
seconds after the existing 600-second reserve. Prior NUTS-exclusion tests and
E3/D3 costs were not double charged. The baseline is
`N5/method-study-refresh-01/budget-ledger-d4-baseline.json`; the observer releases
unused caps at terminal settlement. The queue manifest preserves actual commands,
commit/environment, seeds, source/data/reference identity, resource probe and
all eight cell dispositions. Pilot and native records preserve every generated,
omitted and unstarted hypothesis and the full trace/work/status metadata.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept E3 as a scoped independent reference | Fifteen unchanged refinement criteria and parity pass | Source/input identity passes | Empirical randomized integration error; finite UKF target | Use for separate posterior assessment | Exact nonlinear inference, rigorous bound or sampler validity |
| Refresh master and execute D4a | Concrete plan self-audited; 30 focused tests pass | No preflight source/reference failure | Long adaptation and duration costs | Complete all eight proposal-price dispositions, then refresh D4b | Adequate adaptation or tuning membership from prices |
| Preserve posterior/tuning separation | Existing qualified members and failures unchanged | No R-hat/ESS/MCSE selection introduced | Full modern posterior delivery still unrun | Public fresh qualification and all-member consumer | Ranking, optional release or default readiness |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | E3 reference parity/integrity pass; D4a numerical outcomes pending |
| Statistically supported ranking | None |
| Descriptive-only differences | D3 acceptance, runtimes and one-step SNAPER proposals |
| Default-readiness | Not established |
| Next evidence needed | Target-specific proposals, fresh tuning, full posterior delivery, fair classical search and paired replication |

Red-team note: a precise importance reference does not validate an MCMC kernel;
a longer pilot may still fail or propose poor trajectories. A locally healthy
native path is not a mixing test. The guard against those alternative
explanations is the subsequent fresh public tuner and separate full posterior
assessment, not a more favorable price label. The weakest remaining evidence
is full nonlinear modern-member delivery and statistically supported comparison.

### D4a resource-selection repair and restart

GPU1 became idle during the first worker's setup. Inspection of the installed
probe confirmed its explicit rule `gpu0_forbidden_while_gpu1_is_available_and_idle`.
The fixed-device queue would therefore fail its next preflight despite capacity.
Attempt01 was stopped, not declared a mathematical failure; no tuning or
posterior artifact was issued. Its original consumer source is archived in
`N5/method-study-refresh-01/historical-source/` and 318.609382 seconds are
conservatively charged. Parent and child PIDs are gone. Systemd's delayed stop
reported `timeout` after inotify-watch exhaustion, despite both numerical PIDs
already being gone; this is supervisor cleanup evidence, not a numerical wall-cap
failure. The retry explicitly bounds service cleanup to15 seconds.

The repaired queue uses the trusted **automatic** probe before each cell and
records the selected device in its row, worker command and CUDA environment.
Only `no_idle_policy_permitted_gpu` permits the existing independent shared
headroom check; unexpected probe/driver failures still fail closed. The probe
omits inventory on this error, so the shared helper obtains its own trusted
inventory rather than relying on absent fields. No GPU setting, scientific
criterion, target, data, method, geometry or cumulative grant is changed.

Rerun regressions: **33 passed, 4 opt-in full-model tests skipped**, costing
27.22668807487935 parent seconds. These add changing-GPU selection, fail-closed
driver errors and actual child-group timeout cleanup to the earlier checks.
Both test attempts and the interrupted setup remain charged. Current charges
before the retry are 193776.11692600654 seconds. A fresh36060-second reservation
leaves 201563.88307399346 seconds unallocated after reserve; the prior unused
reservation was released exactly once.

The same eight-cell scientific study restarted in `N5/method-proposals-02/`,
initially on freshly admitted idle GPU1. Its service is
`bayesfilter-modern-hmc-n5-method-proposals-02.service`; read-only observer02
writes `N5/method-study-watch-02/` against
`N5/method-study-refresh-01/budget-ledger-d4-retry-baseline.json`. New
source-bound streams are recorded, not represented as a replay of attempt01.
The current matrix points to retry02 and preserves attempt01's failure pointer.
The main D4 plan includes this repair and its renewed skeptical self-audit:
resource selection changes between cells, whereas the scientific contract does
not. Full numerical outcomes and D4b allocation are still pending.

Live launch check: retry02 completed full SGQF preparation/reference preflight and
entered the discarded pilot. Its design records horizon20/dimension6, full
`per_chain_step` status policy for native prices, physical GPU memory growth
verified before initialization, host GPU1 and GPU execution. XLA compilation is
confirmed in its worker log. Both queue and observer are active with no recorded
failure. These establish launch/identity checks, not a completed pilot or price.

## D4a terminal result and full-set D4b1 follow-through, October 10

D4a is complete. Service and observer02 exited successfully. The queue ran
from20:27:54 to21:18:32 UTC, October9 (04:27:54–05:18:32 China time, October10),
charging3038.80259970529 parent seconds. All eight cells, sixty distinct planned
proposals and sixty measured short paths are present; none was vetoed or omitted.
The post-run source/reference hash audit passes for every cell. This closes
proposal pricing only; no tuning, posterior or performance claim follows.

| Target | SNAPER paths | MAMS paths | Occupied seconds, SNAPER / MAMS |
| --- | --- | --- | --- |
| Full PP-SGQF | 6/6 healthy | 9/9 healthy | 466.438 /393.040 |
| Full PP-UKF | 6/6 healthy | 9/9 healthy | 321.449 /279.096 |
| Full SIR | 6/6 healthy | 9/9 healthy | 325.678 /433.733 |
| Full structural UKF | 6/6 healthy | 9/9 healthy | 577.124 /240.571 |

The longer discarded SNAPER pilots proposed L2–8 on SGQF, L3–12 on UKF,
L1–3 on SIR and L5–23 on structural. Structural epsilon collapsed to roughly
.0030–.0038, with near-one short acceptance; that is a concern about proposal
quality, not evidence of good posterior exploration. Every MAMS neighborhood
remains a proposal requiring fresh qualification. These are descriptive findings;
there is no paired, uncertainty-supported performance comparison.

The terminal ledger charges196814.91952571183 seconds, with no active reservation,
leaving234585.08047428817 seconds (65.16 hours) after the unchanged600-second
reserve. Both earlier test attempts and the stopped setup are already included;
no historical costs were reset. Terminal integrity and refreshed twenty-cell
inventory are in `N5/method-full-fit-audit-01/d4a-terminal-audit.json` and
`nonlinear-comparison-matrix.json`, alongside `budget-ledger-d4a-terminal.json`.

The [D4 plan](bayesfilter-modern-hmc-n5-method-proposals-2026-10-10.md) now freezes
D4b1, the first complete SGQF proposal sets: all six SNAPER and nine MAMS members
enter fresh public tuning under seed4400; every verified member exports/reloads
and enters the unchanged separate posterior controller. No candidate was selected
by a posterior metric or by descriptive acceptance. The other45 priced proposals
remain explicitly awaiting later model cohorts. Runtime extrapolations cannot
guarantee the whole nonlinear matrix within the remaining budget, so this
bounded cohort preserves a substantial later-work/repair allowance.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close D4a pricing | Eight cells and sixty short paths complete | No recorded health veto; source/reference audit passes | Long-run adaptation and mixing | Fresh qualification of complete frozen sets | Tuning or posterior admission |
| Fund first SGQF cohort | Measured workload and declared target/method order | No invalid target or reference | Sequential stopping cost, seed variability | Full15-proposal public lifecycle under bounded allocation | Superiority or adequate baseline search |
| Keep structural proposal issue explicit | Finite tiny-step SNAPER proposals | No short numerical veto | Very short effective travel despite larger L | Later scoped tuning/posterior study; preserve all proposals | Adequate structural mixing from high acceptance |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | All60 short native paths passed the declared health checks |
| Statistically supported ranking | None |
| Descriptive-only differences | Pilot epsilon/duration, short acceptance and elapsed costs |
| Default-readiness | Not established |
| Next evidence needed | Fresh public qualification, all-member delivery, reference agreement, fair classical search and independent paired replications |

Post-run red-team note: the strongest alternative explanation for good prices is
that locally benign starting states and conservative step sizes conceal poor
global mixing. The structural result illustrates that possibility directly.
Separate full posterior/reference checks, including every failed member, are
the next discriminating evidence. More adaptation steps alone do not establish
better proposals, and a high short acceptance probability is not a ranking.

### D4b1 implementation, final checks and launch

The new consumer `scripts/run_modern_hmc_method_fit.py` reads every frozen
hypothesis, uses the public tuner with the actual family size, audits native
evidence, exports/reloads all verified members and runs the shared posterior
controller on each. The separate CPU diagnostic
`scripts/diagnose_modern_hmc_method_reference.py` preserves the full denominator,
including missing, vetoed and nonfinite posterior results. It uses the unchanged
quantity charts and reference screen; no feedback enters tuning membership.
The old two-member consumers and D4a source closures were not edited.

Initial checks:38 passed in141.77304956503212 parent seconds. Final review found
that a normal `partial_budget` tuner return also needed explicit resource-stop
classification, in addition to raised budget exceptions. That repair preserves
the original proposal inventory and candidate states. Shared-invalidity or
infrastructure-paused results stop dependent posterior work. All15 consumer
tests then passed in139.44133342104033 parent seconds. Across both runs there
are39 distinct focused checks, including three native public lifecycles on
Gaussian fixtures with SGQF, structural and SIR quantity dimensions, and actual
saved checks for all60 full-model price proposals. These are control/identity
checks, not independent posterior validation of those nonlinear models.

The renewed skeptical audit passes: source and target match, no stale
two-candidate budget remains, prices grant no membership, all15 hypotheses in
the funded cohort remain, member vetoes do not suppress peers, and partial
tuning cannot be mislabeled complete. The predefined one-seed workload study
does not support a comparative ranking or optional release.

Trusted probe selected idle GPU1. Launch at04:54:53 UTC (12:54:53 China time),
October10, uses `bayesfilter-modern-hmc-n5-method-full-fit-01.service` and
`N5/method-full-fit-01/`. It starts SGQF SNAPER's six-member cohort, with the
complete nine-member MAMS cohort next. Both the numerical queue and read-only
`bayesfilter-modern-hmc-n5-method-full-watch-01.service` are active; initial
preflight has no recorded error. The worker environment sets growth before
framework import; runtime verification and GPU/XLA provenance are preserved by
the numerical binding. No completed tuning or posterior result is yet claimed.

Actual commands, commit, source hashes, environment, seeds and hardware selection
are in the queue/worker manifests. The140760-second reservation includes the
12/27-hour fit caps, two120-second assessments, queue overhead and cleanup.
The140700-second queue allowance has a144000-second outer service guard, with
bounded process-group termination for each child. Costs are charged to the
parent once; the observer at `N5/method-full-fit-watch-01/` recomputes from
`N5/method-full-fit-audit-01/budget-ledger-d4b-baseline.json`. Charged work is
197096.1339086979 seconds before this cohort;93543.8660913021 seconds remain
unallocated after its reservation and the existing overhead reserve. All earlier
failed attempts remain charged. Numerical ceiling expiry preserves results and
triggers workload/budget diagnosis; it is not a posterior or tuning veto.

### D4b1 terminal cohort and reference diagnostic repair, 11 October

The serial queue ended at01:58:39 UTC with a `partial` manifest. Both numerical
cells exited normally. All six SNAPER and nine MAMS candidates passed fresh
tuning verification and export/reload. SNAPER passed posterior and separate
reference checks for all six. MAMS passed posterior checks for five; the first
had inconclusive warm-up and zero retained draws, and three reached 10,000
retained draws per chain without satisfying the declared precision target.
There was no recorded numerical hard veto. The original post-run MAMS reference
diagnostic failed when it called lugsail precision on the zero-draw tensor;
the original log and partial manifest remain preserved.

The [repair plan and result](bayesfilter-modern-hmc-d4b-reference-repair-2026-10-11.md)
records the skeptical audit, regression-first fix, CPU-only reassessment and
separate evidence roles. The repaired diagnostic handles zero to three draws
explicitly and still rejects malformed tensors. Four new cases reproduced the
failure before repair; after repair eight focused tests and all 22 consumer
tests passed. The assessment of saved MAMS results now preserves the nine
original identities: five agreement passes, three not established, one with
no retained draws. It does not modify any posterior run or tuned membership.
The old numerical consumer hashes still match; the diagnostic has an explicit
new checksum in `N5/method-reference-repair-01/repair-manifest.json`.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep SNAPER as a scoped viable arm | 6/6 verified, posterior and reference screens | No member veto | Single seed; baseline/search mismatch | Paired replication after budget audit | Superiority or release |
| Keep MAMS's five delivered peers | 9/9 verified, 5/9 posterior and agreement screens | Four delivery failures; no numerical hard veto | More exploration or sampling may change delivery | Diagnose failure modes under separate budgeted phase | Rejection of whole method |
| Close diagnostic repair | Original nine identities, correct dispositions, 22/22 consumer regressions | Corrupt shapes still fail closed | One saved cohort | Use repaired assessment and preserve original failed log | Changed scientific outcomes |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No candidate numerical hard veto; four MAMS posterior delivery vetoes, two with reference screens also above .10 |
| Statistically supported ranking | None |
| Descriptive-only differences | Acceptance, elapsed cost, retained count and per-member precision/reference ratios |
| Default-readiness | Not established |
| Next evidence needed | Feasible next target cohort, fair classical search and independent paired-seed replications |

The terminal observer charged272923.235 seconds including the queue. The repair
ledger conservatively adds143 seconds for all tests and CPU reassessment,
leaving158333.765 seconds (43.98 hours) after the existing600-second reserve.
The [fresh twenty-cell matrix](artifacts/modern-hmc-implementation-2026-10-07/N5/method-reference-repair-01/nonlinear-comparison-matrix.json)
records SNAPER's scoped pass and MAMS's partial delivery; the other45 priced
proposals remain unlaunched. This does not establish a method ranking or a
release. A successful SGQF candidate could be a local-mode result; only later
independent replicated and broader-model evidence can test that explanation.
