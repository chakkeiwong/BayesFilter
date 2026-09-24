# HMC repair master program

Updated 2026-09-23 at terminal M38. **The funded M31--M38 program is complete.**
The [execution result](bayesfilter-hmc-m31-m38-result-2026-09-23.md) records
repairs, tests, twelve GPU fits, the rebuilt guide and every open requirement.
R2/R3 confirmation remains underfunded; R4/R7 scientific validity remains open;
R5 awaits exact inputs. Completing this bounded program does not mean those
scientific requirements passed. The sections below preserve its reviewed
contract; they are not instructions to restart M31 or invent M39.

This revision replaces the rolling next-experiment approach. The exact former
master is preserved in [history through M30](bayesfilter-hmc-repair-master-history-through-m30-2026-09-23.md).
Historical "next" instructions and phase allocations there are not active.
The [analysis and skeptical review](bayesfilter-hmc-master-roadmap-audit-2026-09-23.md)
records the inspected code, evidence, planning defects and review limitations.
[Machine progress](artifacts/hmc-repair-master-2026-09-16/program-progress.json)
tracks the same requirements, dependencies and funding states.

| Phase | Terminal disposition |
| --- | --- |
| M31 | Source/design/cost audit complete; confirmation shortfalls explicit. |
| M32 | Lifecycle baseline and three additional exact GPU reuse pairs passed. |
| M33 | Posterior-policy wiring, estimator alignment and cached-fit identity repaired and tested. |
| M34 | Four paired full-pipeline defect controls and exact no-op audit complete; power unconfirmed. |
| M35 | Global indicator call chain and negative controls tested; exploration remains open. |
| M36 | Fresh inventory complete; exact consumer bundles still missing. |
| M37 | CPU/GPU training-freeze-retune composition passed; tiny posterior runs failed and map quality remains open. |
| M38 | Final affected tests, official guide build/inspection and resource reconciliation complete. |

Remaining allowance is 71,946.615 CPU and 74,041.939 GPU worker-seconds;
see the terminal ledger linked from machine progress. No worker remains active.

## Objective and completion

Finish a coherent tuning-to-posterior procedure with tested failure behavior,
explicit target/geometry requirements and measured limits. Preserve two public
tuners: `tune_hmc_kernel` for ordinary exact-value/score targets and
`tune_fixed_transport_hmc_kernel` for supported frozen maps. Both use the shared
candidate-set controller. The position-field branch remains conditional
mechanics and cannot acquire exact-score posterior authority through this plan.

Every verified epsilon/L pair stays retained. Fresh verification is required
for each exact candidate. R-hat, ESS, MCSE, reference agreement and validation
p-values never gate, rank or repair tuning membership. They can veto a
downstream posterior result. Posterior assessment may predeclare representative
members without deleting their siblings or counting siblings as independent fits.

The program has three separately reported completion levels:

* **Engineering complete:** R1, R6 and R8 below pass on their declared final
  source; numerical failures, caps, unavailable evidence and unsupported routes
  are reported correctly. Expected failing stress cases can pass this test.
* **Scientific cells validated:** only the named model/policy/source cells
  meeting R2--R4's declared statistical and numerical criteria. A general claim
  does not follow from a finite model matrix. An underfunded cell stays open.
* **Consumer/training extension complete:** R5 and R7 require their own exact
  inputs and downstream results. They are part of this roadmap but neither
  missing inputs nor training failure implies a defect in supplied-map tuning.

M38 reports each level. `scientific_gaps_closed` remains false while any required
scientific cell is failed, unavailable or underfunded. No further phase number
is created merely because a pilot was inconclusive. A genuinely new method or
target becomes a separately justified amendment, with the old requirement visible.

## Evidence baseline and requirements

Planning source is `e9fee584704a465fc6ab984e3a8cc53980f335d0`. Exclude unrelated
dirty Q20, training-protocol and governance files from numerical snapshots.
Each historical result retains its actual frozen source; the planning commit
does not relabel it. Preserve M22's frozen-Gaussian null result, M25's exact
AR(1) result, M28's supplied residual-map cells and M30's two-model reuse parity.

| ID | Current finding and required resolution | Primary evidence / responsible phase |
| --- | --- | --- |
| R1: unified procedure and candidate lifecycle | Implemented in inspected public paths; retain per-L epsilon evidence, all verified members, fair funded scheduling, distinct streams and replay checks under every supported branch. | Public dispatch, controller, numerical-binding, restart and authority regression matrix; M32/M38. |
| R2: equilibration and posterior precision | Readiness, lugsail and quantile MCSE exist. M25 full-fit stopped coverage failed some Gaussian/beta quantities; M26 successes are pilots. Separate initialization bias, finite-count variance estimation, member dependence and optional stopping. | Independent fixed-count and actual-stop complete fits, all planned outcomes; M31 design, M33 repair/calibration. |
| R3: complete-fit validation sensitivity | M22's frozen-kernel null screen is closed for that cell. Full preparation/tuning/posterior false-positive rate and quarter/half-SD defect power remain open. | Activated whole-fit defects, null/no-op controls and independent whole experiments; M31/M34. |
| R4: global exploration | Mixture runs can miss a mode despite local diagnostic success. Supplied whitened-funnel cells are already tested. | Mode-sensitive diagnostics and matched independent reference, numerical health, dispersed/local starts; M35. |
| R5: exact consumer | M24 repairs are tested; fresh nine-parameter bootstrap inputs and the earlier full-joint MIDAS reference are separate unresolved dependencies. | Exact source/data/prior/coordinate/reference inventory and fresh public consumer execution; M36. |
| R6: execution, maintainability and resource limits | M30's optional reuse is exact on Gaussian/beta-binomial pairs. Other targets, longer lifecycle behavior and legacy-module dependency boundaries need bounded checks. | Same-source parity, resource observations, failure/restart tests and focused extraction if warranted; M32/M38. |
| R7: upstream learned map | M23 repaired graph/batch prerequisites. No target-specific training-quality result follows from those checks or supplied-map tests. | Reviewed target-specific GPU training, frozen map, fresh tuning and model-coordinate posterior validation; M37. |
| R8: one guide and honest status | The official book and API reference describe the procedure; the old master mixed current and historical next steps. | Registry/code/book/examples/coverage agreement, rendered book inspection, requirement-by-requirement terminal report; M38. |

The [M25 result](bayesfilter-hmc-gap-closure-result-2026-09-22.md) is evidence of
real posterior deficiencies: Gaussian all-quantity stopped coverage failed;
beta-binomial median coverage failed; all eight rotated-Gaussian selected
posteriors hit precision caps. These outcomes cannot be closed by relabeling
them as engineering passes. The [M26 result](bayesfilter-hmc-m26-lifetime-and-policy-result-2026-09-22.md)
shows why larger windows/counts and predeclared sibling assessment are candidate
repairs. The [M30 result](bayesfilter-hmc-m30-runner-reuse-result-2026-09-23.md)
supports optional graph reuse, not statistical calibration or universal speedup.

## Fixed phase roadmap and dependencies

| Phase | Deliverable | Dependency / completion condition |
| --- | --- | --- |
| **M31** | One statistical design and funding decision for the whole remaining program. | M30 audited receipts; fixes experimental units, criteria, inventories, prices and funded/deferred cells before new HMC. |
| **M32** | Common engineering baseline, supported-route model matrix, reuse/lifecycle checks and bounded structural repair. | M31 contracts; all required positive and negative invariants pass; any excluded route is explicit. |
| **M33** | Equilibration/MCSE repair and stopped-versus-fixed calibration. | M31 design and M32 baseline; implementation and statistical completion recorded separately for every cell. |
| **M34** | Whole-fit defect activation and null/power assessment. | M31 design and M32 baseline; depends on M33 only if its proposed posterior policy is the declared test target. |
| **M35** | Base global exploration diagnostics and supplied-geometry study. | M31/M32; ordinary mixture failure is preserved. A learned-map extension uses this phase's diagnostics after M37. |
| **M36** | Exact MacroFinance integration and consumer reply. | Inventory can begin after M31; numerical run requires M32 and a qualified matching input bundle. Posterior-policy claims require corresponding M33 evidence. |
| **M37** | Separate learned-map training and downstream assessment. | M31/M32 and a target-specific training protocol; uses M35 diagnostics. It is not a prerequisite for supplied-map tuner completion. |
| **M38** | Final integration, official guide update, evidence coverage report and terminal disposition. | All prior work packages have an explicit disposition; unresolved requirements remain open. |

Execution order is M31, M32, then the funded M33/M34 cells with R2 first,
followed by M35. Run M36's input inventory and M37's protocol preparation early
so their dependencies are exposed before expensive work. Their execution need
not wait for unrelated calibration. M37 uses M35's base diagnostics for its
downstream learned-map arm; M35's base implementation therefore precedes M37
and there is no cycle. M38 integrates that extension with the base M35 result.
M38 always runs, even if an input or budget prevents scientific closure.
These are work dependencies, not instructions to launch extra agents.

### M31: resolve statistical design and total affordability

Retain the useful technical work in the [M31 design](bayesfilter-hmc-post-m30-next-phase-2026-09-23.md),
but its output must now populate this entire roadmap, not invent the next phase.
Produce `m31-r1/design-and-funding.md` and a resolved inventory JSON using the
existing validation design types. Each row specifies target/data/coordinates,
source policy, member rule, quantities, starts, estimator/counts, experimental
unit, fixed confirmation size, failure denominator, criteria, price and funding.

Compare the current design with at most two mathematical alternatives. Keep
stopped-functional coverage distinct from output-distribution sensitivity.
Inspect the technical sections, proofs/appendices and official code for Talts
SBC, Modrak test quantities and Gandy--Scott under the existing local paper
directory; preserve exact anchors. Independent fits, approximate thinning,
reversible stationary kernels and adaptive stopped output have different nulls.
Do not assert a theorem's assumptions merely because a diagnostic passed.

One specific alternative worth auditing on analytic targets is an exact
posterior-CDF test using one predetermined output per fresh fit. If the intended
conditional CDF is continuous, under the correct-output null
`P(F_y(X) <= u | y) = u`. For the normal-conjugate target,
`Z=(X-mu_y)/s_y` is standard normal under that null and has mean shift `d` under
the existing translated target. This local derivation avoids several rank
fits per dataset; it is not a general substitute for SBC, stopped-mean coverage,
or an empirical full-HMC power study. State dependency, stop and missing-output
limitations, test quantities and multiplicity before using it. The existing
terminal output rule is an object to validate, not proof of the required law.

R2 keeps the inherited **pointwise two-sided 95% exact binomial lower bound
at least .90**, for delivery and every declared mean/median coverage event.
Absent intervals are failures; available cap intervals and posterior delivery
are separate outcomes. Preserve requested-member slots. These are pointwise
screens, not simultaneous 95% coverage across the whole matrix or anytime-valid
posterior intervals. R3 keeps the inherited **null rejection upper bound .10**
and **defect detection lower bound .80** for its declared pointwise intervals;
the .25/.5 posterior-SD defects remain separate cells. These thresholds come
from M22/M26 and are screening objectives, not new universal defaults.

Fix each study size from operating characteristics, uncertainty and cost before
launch. A changed design gets a new identity and an explicit statement of
which original requirement it answers. Oracle simulations calibrate the
statistic only. If adequate confirmation cannot fit, price the shortfall and
mark the scientific cell underfunded; do not execute a token "confirmation".
Any small HMC run must answer an explicit implementation/activation question.

M31 is complete when every later cell has a criterion and dependency, every
funded run has a resolved inventory, and every unfunded/input-dependent cell
has an explicit shortfall or missing input. That is design completion, not
scientific closure. No new default or new transition family is selected here.

### M32: engineering baseline and bounded structure repair

Audit the actual path from public configuration through preparation, per-L
proposal, cohort scheduling, measurement, directional repair, fresh verification,
export/reload, cumulative posterior and restart. Inspect unsupported-option
rejection and every failure class as well as successful cases. Use
`HMC_TUNING_INTERFACE_CAPABILITIES` as the authoritative route inventory.

Extend existing tests only where a named invariant is missing. Required cases:
multiple valid epsilons at the same L; high/low acceptance with fresh child
verification; inconclusive evidence at its cap; nonfinite candidate versus
shared corruption; absent/malformed telemetry; R-hat missing/high/error without
membership changes; affordable peer scheduling; interrupted native calls and
single charging; source/data/map mismatch on reload; stale verified result after
shared failure; seed collision; warmup exclusion; unavailable versus deliberately
unassessed members; and abnormal process exit with completed numerical files.
Compare with an independent expected inventory rather than reproducing the
scheduler's decisions in the test oracle.

Broaden optional reuse checks to rotated/scaled geometry, a constrained target
and a supported frozen map. Keep Gaussian/beta controls from M30. Test cache
isolation across target/map/dtype/shape/trace/backend changes, repeated complete
fits and memory/function counts, cancellation and restart. Same-source static
and dynamic arms must preserve every numerical record and tensor under the
existing exact comparison; any justified backend tolerance is separately
specified and does not weaken that same-source contract. Normal shutdown and
bounded observed resource use are required. No finite run proves an asymptotic
memory bound. Reuse stays optional; a speed default needs another evidence bar.

The historical facade still supplies `_json_ready` to preparation and
`_HMCPhaseAttemptState` to mass adaptation. Audit these specific import
dependencies and other active facade edges. If a small shared utility/state
extraction removes a real cycle or isolates testable behavior, perform it with
public alias, serialization, source-closure and replay tests. Retain useful
legacy invariants through public tests. Do not delete historical readers,
rewrite the 20k-line facade wholesale or optimize unmeasured costs.

Output: named test inventory, changed-file/dependency map, exact paired records,
resource receipts and a frozen common source. A skipped required integration is
unavailable, not passed. No general posterior claim follows from M32.

### M33: diagnose and repair posterior stopping

Test three distinct mechanisms in order: finite-count MCSE under stationarity;
residual initialization bias and readiness; then coverage of actual stopped
intervals. Use existing exact AR(1) controls to expose dependence timescales,
without rerunning M25's closed cell merely to gain another pass. Add new
diagnostic controls only for an untested mechanism: delayed mean/scale drift,
short readiness windows, antithetic dependence, constant chains, insufficient
batches, infinite moments and exhausted warmup/retained caps.

Use independent fixed-length arms and actual public fits on Gaussian,
beta-binomial, rotated Gaussian and LGSSM location. Gaussian/beta are the first
confirmation targets because their failures and candidate repairs are already
localized. Rotated geometry tests member-specific precision; LGSSM tests the
data-bearing preparation path. Carry analytic quantities and reference
uncertainty correctly into each comparison. A Student-t/Cauchy stress row checks
finite-moment restrictions; it is not a requirement to estimate a nonexistent
Cauchy mean. Supplied funnel/banana model-coordinate quantities test nonlinear
reporting under their declared map and moment assumptions.

Repair selection follows the observed mechanism:

| Failure | Candidate repair to test in development | Unchanged requirement |
| --- | --- | --- |
| Recent window has little effective information | Jointly lengthen readiness window/minimum and cap; optionally use existing ESS floors/consecutive checks | No stationarity proof from a small R-hat or overlapping windows. |
| Finite-count long-run variance biased low | Test explicit longer batches or the existing autocorrelation estimator against independent analytic covariance | Positive lugsail value/minimum batch count alone cannot qualify precision. |
| Poor selected-member mixing | Predeclare additional identity-based member slots and target-specific allocations | Preserve every sibling; no truth/diagnostic selection of a favorable confirmation member. |
| Fixed intervals adequate but stopped intervals inadequate | Audit stopping dependence; test a separately frozen allocation/stopping policy | Wider intervals or lower precision requirements cannot silently replace the original objective. |
| Numerical-health veto | Localize target/map/transition and repair before statistical comparison | Do not repair invalid arithmetic by extending burn-in. |

At most two materially different posterior-policy hypotheses per failing cell
are evaluated under the funded development inventory (a convenience search cap,
not a statistical threshold). Existing M26 hypotheses count toward this cap if
continued unchanged; do not relabel the same pilot as a new discovery. Freeze
the surviving policy before untouched confirmation. No feasible survivor means
an explicit failed/underfunded cell, with the mechanism and next required evidence.

Report initialization/readiness outcome, retained precision, global diagnostic
availability, stopping reason and planned/completed fits separately. Confirm
the R2 screen on every declared quantity and member slot; document pointwise
versus joint uncertainty. No unqualified "burn-in sufficient" or universal
sample-count claim is permitted. Output repaired optional policy if supported,
calibration tables, raw intervals and a requirement-level decision.

### M34: complete-procedure defect sensitivity

Use the actual public preparation, broad candidate search, verification and
posterior route, not a frozen transition substituted for it. Preserve M22's
normal-conjugate `tau=2`, `sigma=1`, six-observation baseline and translated
targets with d=0, .25 and .5. Derive their density, score, center and scale from
the target; verify that each intended mutation reaches preparation and survives
serialization. Baseline and explicit no-op are separate controls. Development
can additionally diagnose an existing ignored-data or missing-Jacobian mutation
on its supported target; this cannot replace the subtle-shift requirement.

Implement only a mathematical design accepted by M31, with an independent
reference oracle and fixed quantity/multiplicity/output rules. Keep algorithm
failures, missing posterior draws and statistical rejection distinct. A crash
or NaN is not detection power. Conditional tests on successful fits cannot
establish unconditional null/power. Retain all planned experiments in reporting.

Report three separate accomplishments: mutation activation; defect detection
in a declared experiment; repeated-experiment null/power meeting R3. The first
two can close engineering subrequirements when full power is unaffordable;
they cannot set R3 to closed. Quantify the extra cost of repeated experiments
before committing the first confirmation run. A new statistic must carry its
own identity and calibration rather than inherit M22's Gaussian null result.

### M35: global exploration and supplied geometry

Use the existing separated-mixture law and both local and mode-dispersed start
regimes. Freeze the global quantities before sampling: mode probability and
appropriate tail/transition diagnostics. Check analytic mixture mass/CDF and
uncertainty against independent references. Do not infer occupancy precision
from a constant indicator or a few crossings. Global posterior acceptance
requires the declared quantitative reference agreement and uncertainty as well
as ordinary numerical-health, convergence and precision checks.

First retain a regression reproducing local-diagnostic success with a missed
mode. Correct reporting and monitor wiring are engineering completion; actual
exploration remains a separate scientific cell. Test a supported supplied
geometry when independently justified, with exact law/Jacobian and matched
model starts. M28's exact/residual-funnel results remain closed for their cells;
centered-funnel success is not reinstated as an ordinary-tuning requirement.

M35's base result is complete before this arm starts. M37 may then supply a
frozen target-specific map and fresh tuning for a learned-map extension, which
is reported under M37 and integrated at M38. If supported local HMC still fails
global exploration, report that limitation and the evidence; do not repeatedly
extend epsilon grids. A new tempering, replica-exchange or other transition
family is a material method extension and requires its own explicit decision.
There is no promise that a smooth bijection or ordinary HMC solves every
multimodal target within a fixed allowance.

### M36: exact MacroFinance integration

Perform one fresh bounded local inventory at the recorded consumer locations.
Distinguish (a) the reported nine-parameter, 48-observation bootstrap case and
(b) the earlier full-joint MIDAS posterior-reference case. For each, bind source,
prepared data, priors, coordinate transforms, adapter value/score, telemetry,
numerical mode and independent reference with uncertainty. Do not rerun the
consumed bootstrap attempt or equate a fixed-loadings block model to the joint law.

When a matching bundle is present, check density/score/transform consistency,
then run fresh public preparation, all-candidate verification, export/reload
and posterior assessment under the consumer's unchanged scientific contract.
Preserve candidate quotas and missing members explicitly. Diagnose a first
failing primitive before introducing adapter-specific domain recovery.

If matching inputs remain absent, write their precise inventory and required
fields in the consumer reply and mark the affected cell `awaiting_inputs`.
An integration fixture still tests the adapter interface but cannot close R5.
This dependency does not stop M32--M35 or M38. Write the reply locally; this
plan does not authorize sending external messages or modifying the consumer repo.

### M37: upstream learned-map quality

Use separate banana and separated-mixture protocols, not a transferred generic
training recipe. Before training, specify the exact density and scale checks,
batch-native target route, objective, architecture/capacity hypotheses,
optimizer/search ranges, seed/holdout policy, total update/attempt budget and
downstream criteria. Preserve affine/untrained and plain trained comparators;
an enhanced arm is justified only by a stated failure mechanism and available
budget. Existing training settings are baselines until target-specific evidence
supports them. Do not select maps on the final posterior/reference holdout.

Training must use trusted GPU/XLA, verified memory growth, real batches larger
than one and stable signatures without scalar row fallback or unapproved pfor.
Independent dataset generation uses the repository's CPU policy. Reuse M23's
graph/batch engineering checks but audit the actual selected training route.
Freeze each assessed map, verify inverse/log-Jacobian/target-score consistency,
retune every declared L and assess model-coordinate posterior quantities on
untouched streams. Loss reduction alone cannot establish useful geometry.

The current allocation is a protocol/readiness and bounded development ceiling,
not a promise to fund both serious training protocols. If the derived search
and downstream inventory does not fit, report the exact shortfall before
training. R7 remains open; supplied-map tuner completion is unaffected. A
successful map must pass the M35 global checks for the mixture, not only local
MCSE; if those checks are unavailable, the learned-map cell remains open.

### M38: terminal integration, guide and decision

Run the affected public unit/integration suite on the final owned source, with
both authoritative tuners, native automatic preparation, prepared binding,
fixed-map replay, supported option failures, counts/caps and restart. Reconcile
every study to its source and criteria. If a later edit changes a numerical
path, audit its effect and rerun affected checks; do not simply relabel an old
statistical result as final-source evidence. Unaffected old cells keep their
original scope. Exact cross-source replay needs saved numerical states/streams;
same configuration alone is insufficient.

Update the scientific guide in `docs/main.tex` through chapters 21b, 25 and 26b
as appropriate. Keep `docs/reference/hmc-tuning-interface.md` the aligned agent
API reference, not a competing procedure. Regenerate the interface inventory,
exercise examples, build the book, resolve references and inspect changed
rendered pages. Explain preparation, tuning qualification, equilibration,
precision and global exploration in distinct terms. Document tested policies,
unsupported scope and failure actions without turning plans into reader prose.

Publish a single terminal requirement table with source, model, criterion,
outcome, uncertainty, remaining dependency and next action. Allowed outcomes
are `closed_for_declared_scope`, `failed_requires_repair`, `underfunded`,
`awaiting_inputs` and `not_evaluated`; engineering completion is separate.
Reconcile worker time and live processes, preserve all failures and update the
master/progress record. Commit/merge/push only owned changes under the standing
authorization. M38 is the end of this roadmap, not permission to declare all
scientific gaps closed or automatically start an M39.

## Integration and model coverage

Use three evidence tiers: deterministic/unit checks for logic and formulae;
small real public-pipeline integrations for wiring/lifecycle; independent
replicated campaigns for statistical claims. Wider mechanics-only acceptance
bands in existing tiny fixtures never become production qualification evidence.

| Model / mechanism | Required integration or scientific question | Phase |
| --- | --- | --- |
| Isotropic and rotated/scaled Gaussian | Automatic versus supplied preparation, native grid, metric hints, candidate siblings, reuse, delayed readiness, precision caps and analytic truth | M32/M33 |
| Normal-conjugate and beta-binomial | Data/prior binding, transformations, whole-fit mutations, stopping and exact conditional reference | M32/M33/M34 |
| LGSSM location | Data-bearing state-space adapter through preparation, retained reload and posterior quantities | M32/M33 |
| Gamma/beta and Dirichlet | Support boundaries, Jacobian, two active versus three reported simplex coordinates, malformed transform rejection | M32 |
| Student-t and Cauchy | Heavy tails, moment restrictions, quantile/bounded quantities, honest unavailable mean precision | M32/M33 |
| Banana | Nonlinear supplied map and model-coordinate checks; separately trained-map quality | M32/M37 |
| Exact/residual-whitened funnel | Exact target/Jacobian/start roundtrip; preserve M28, extend only where new code changes behavior | M32/M35 |
| Separated mixture | Missed modes, constant mode indicators, two start regimes, matched reference; learned repair separately | M35/M37 |
| Pinned posteriordb regression/eight-schools | Independent real-model target/reference uncertainty and migration regression when matching fixtures are available | M32/M38 |
| Exact MacroFinance cases | Original consumer law and fresh matching inputs; no synthetic or block-model substitution | M36 |

Existing starting points are `tests/test_hmc_candidate_set_{tuning,execution}.py`,
`tests/test_hmc_whole_procedure_repair.py`,
`tests/test_hmc_candidate_runner_reuse.py`,
`tests/test_hmc_posterior_diagnostic_reference.py`,
`tests/test_neutra_sequential_hmc.py`, and
`tests/inference_validation/` (multimodel, transport routes, controller stopping,
sibling assessment, full-fit mutations, fit processes and reference tests).
M32 records which tests are actually collected/run and which requirement they
cover. One tiny fit per family is engineering coverage, never a reliability rate.
Do not multiply every option by every model without a reason; pairwise route/
feature coverage plus explicit risky combinations defines the engineering matrix.

## Compute, uncertainty and launch rules

The [M30 terminal ledger](artifacts/hmc-repair-master-2026-09-16/m30-r1/reconciliation-terminal.json)
leaves **74,427.314 CPU seconds (20.67 hours)** and **75,209.499 GPU seconds
(20.89 hours)**. This is the remaining campaign allowance, not a new allocation.
All earlier reservations are historical; only the following envelopes apply.

| Phase | CPU hours | GPU hours | Allocation role |
| --- | ---: | ---: | --- |
| M31 | 0.5 | 0.5 | Inherited design-development ceiling |
| M32 | 3 | 1.5 | Engineering tests and bounded paired integrations |
| M33 | 6 | 7 | Posterior repair/validation envelope, contingent on justified study size |
| M34 | 3 | 4 | Whole-fit activation/design/validation envelope, same contingency |
| M35 | 1 | 1.5 | Global diagnostics and bounded supplied-geometry evidence |
| M36 | 1 | 1 | Input inventory and matched integration if qualified/affordable |
| M37 | 1 | 1.5 | Target-specific protocol/readiness and development only if adequately budgeted |
| M38 | 1.5 | 0.5 | Final integration, documentation and audit |
| Common unallocated reserve | 3.6743 | 3.3915 | Localized repairs, cost uncertainty or justified redistribution |

The eight phase envelopes total **17 CPU / 17.5 GPU hours**. Except M31's
inherited cap, these are convenience scheduling ceilings, not measured costs,
statistical sample sizes or assurance of sufficient training. M31 may reallocate
them within the ledger after design/pricing, preserving M38 and a declared
repair reserve. Save changes once in machine progress and this table. Never
spend another phase's allocation silently, or allocate the same seconds twice.

The [planning arithmetic](artifacts/hmc-repair-master-2026-09-16/roadmap-review-2026-09-23/cost-and-design.json)
shows the current obstacle. At M30's single observed dynamic complete-fit costs:

| Fits per target | Minimum successes for inherited pointwise screen | Chance to pass if true success probability is .95 | Gaussian / beta GPU hours |
| --- | ---: | ---: | ---: |
| 64 | 63 | .164 | 6.16 / 5.08 |
| 128 | 122 | .541 | 12.32 / 10.15 |
| 256 | 240 | .855 | 24.65 / 20.30 |
| 384 | 358 | .951 | 36.97 / 30.45 |

The .95 success probability is a hypothesis, not estimated from pilot passes.
These probabilities apply to one binomial criterion; multiple required
quantities may lower the chance that all screens pass. Runtime is descriptive
and may vary. Two 384-fit targets alone project **67.42 GPU hours**, about
**46.53 hours more than the entire remaining GPU allowance**, before repairs,
full-fit power, real consumers or training. A 384-fit design is not universally
mandatory, but reducing its size cannot be defended solely by affordability.
No claim that all scientific closure is funded is supported at present.

Serious HMC/training uses `tfgpu`, trusted GPU/XLA and verified growth configured
before import. CPU reference/debug workers set `CUDA_VISIBLE_DEVICES=-1` before
framework import. No new packages/backends or scalar training fallback are
authorized. At most two numerical workers, and one GPU numerical worker, run
at once under this program; recheck actual device availability. Count outer
worker wall time once including failed attempts/startup/teardown; GPU charges
already include host work. Keep fresh versioned outputs under
`artifacts/hmc-repair-master-2026-09-16/mNN-rK/` and preserve prior runs.

Use the existing CLI (`python -m bayesfilter.testing.inference_validation
plan/run/report/assess`) and native checkpoints. M31/M32 must first check which
resolved engines support process isolation/reuse; unsupported SBC/power
execution cannot be enabled merely by copying a search-suite option. Add a
small tested capability when required, or keep the limitation explicit.
Every numerical launch records its exact command, environment, commit/owned
overlay, seeds, model/data version, device/memory/XLA status, caps, outputs and
plan/result paths. A command template is not an executed run manifest.

## Evidence roles, defaults and phase repair

| Diagnostic or event | Role and consequence |
| --- | --- |
| Density/score/Jacobian/transition or shared-evidence invalidity | Veto the affected experiment; localize and repair before using its outputs. |
| Candidate-local numerical or acceptance failure | Reject that candidate, preserve evidence and run the declared funded repair/peers. |
| Posterior health/global/reference failure or precision cap | Veto that posterior claim; execute the phase's specified diagnosis/repair. No deletion from the verified tuning set. |
| Complete-fit delivery/coverage or full-experiment null/power interval | Promotion criterion only for the exact frozen study and its denominator. |
| Pilot loss, runtime, ESS/MCSE differences, oracle power, training loss | Explanatory/development evidence unless the exact study explicitly assigns a further role. No candidate ranking. |
| Missing required exact input | Continuation veto for that dependent cell only; perform independent work. |
| Campaign exhaustion, unrepaired common invalidity, material method/target change | Stop affected execution; record what failed and whether additional resources or owner direction is required. |

Default audit: broad L `(3,5,9,13,18,25)` and acceptance bands are inherited
tuning policies; their finite scope can miss usable kernels. Existing
readiness R-hat 1.05, retained 1.01, recent-window/count rules, default zero
posterior ESS floors and one successful readiness check are owner/inherited
operational choices, not calibration results. Lugsail's square-root bandwidth,
20-batch floor, r=3 and c=.5 are implemented baseline choices with known
finite-count limitations. Existing .05 precision tolerances are target-specific
study criteria. M26 larger counts, M27 sibling rules and M30 graph reuse are
development/optional choices; passing one target never promotes another.
All proposed numeric changes must state provenance, failure mode and early check.

Between phases, reconcile receipts, classify findings, and update **all affected
requirement rows and dependent inventories**. A localized engineering repair
gets focused regression and a fresh attempt within the same phase. Changed
scientific hypotheses use fresh development evidence; untouched confirmation
is never reused to select a repair. Failed candidates do not terminate research
when the next declared repair addresses that failure. Budget-limited or invalid
confirmation remains open rather than being extended until it passes.

At each close, record decision, primary criterion, vetoes, uncertainty, next
action and limits, plus hard vetoes, supported rankings (usually none),
descriptive differences, default readiness and next evidence. Ask whether the
outcome invalidated the implementation/experiment or only the tested candidate.
Revisions change concrete run details inside this roadmap; they do not erase
its terminal criteria or replace it with another one-phase plan.

## Skeptical review of this revision

The [audit](bayesfilter-hmc-master-roadmap-audit-2026-09-23.md) accepts this
roadmap with explicit limits. It corrects wrong centered-funnel and fixed-kernel
baselines, separates engineering from calibration, preserves missing outcomes,
exposes budget and external-input dependencies, and reserves a final phase.
M31 completed the mathematical/source audit and froze the funded numerical
inventory. M37 completed bounded protocol and CPU/GPU composition checks;
adequate target-specific training and posterior validation remain separate
prerequisites for a learned-map quality claim. The terminal result preserves
those open requirements and the original scientific criteria.
