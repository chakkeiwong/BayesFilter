# Modern HMC rejection consistency repair

27 October 2026. The owner requests a commit and continued execution. The
completed diagnostic is committed as bf1986391. The remaining combined
diagnostic allocation is 17,452.518772122326 inclusive seconds; the separate
SIR allocation is untouched. Prior results and frozen sources remain unchanged.

## Question, evidence and boundaries

Can fixed/randomized HMC and fully refreshed MAMS report a numerical proposal
rejection consistently with their actual Metropolis selection, without hiding
invalid current states, finite target-status failures or broken bookkeeping?
The baseline is the completed source-isolated October 26 implementation. The
corrected TFP trace is a comparator for the rejection interface, not a license
to transplant its numerical integrator or q20 acceptance policy.

Pass requires unchanged successful trajectories and random streams, explicit
raw versus effective log ratios, failed proposals rejected with probability
zero, exact selected position/value/score checks, and preservation of all
failure records. Initial-state errors, malformed/missing trace fields, finite
status failures, broken MH selection and failed healthy-path parity veto the
repair. They stop numerical launch until resolved. Poor acceptance or movement
rejects the affected candidate, not the campaign. R-hat remains reporting-only.
No ranking, posterior convergence, default promotion, consumer admission or
release conclusion follows from these mechanics checks or the next tranche.

## Code trace and skeptical audit

ReusableModernHMCRunner already rejects paths when its sticky good flag fails,
but records the raw ratio (possibly NaN) and lacks selected-score telemetry.
The public health validator therefore cannot establish a safe numerical
rejection. Modern transitions re-evaluate the starting target every transition;
unlike TFP they do not carry a target/score cache between transitions. Comparing
adjacent values bitwise as if a cache existed would be the wrong baseline.
Validate each transition's actual pre-choice value/score and selection instead.

A finite invalid status may precede downstream overflow. Looking only at the
last NaN would misclassify this as a harmless numerical rejection. Preserve and
classify the first failure; finite-status failures remain vetoes even if later
arithmetic becomes non-finite. Do not change the integration path, stopping
policy, target or random streams in this repair. Numerical failures remain
diagnostic evidence; setting the effective ratio to minus infinity only
expresses the rejection the existing transition already performed.

The plan passes skeptical review with these constraints. Healthy-path parity,
mutation controls and the finite-status-then-NaN regression distinguish a
telemetry repair from a silent algorithm or admission-policy change. The real
model provides integration evidence; toy failures alone do not establish it.

## Execution and default audit

1. Add first-failure numerical classification, raw/effective ratios, pre-choice
   state/value/score and selected value/score traces. Validate shapes/dtypes,
   classification, selection and healthy-current invariants under a stable
   compiled health function. Fail closed for modern traces missing the fields.
2. Add CPU diagnostic tests (GPUs hidden) for HMC/MAMS, fixed/random duration,
   XLA/non-XLA, safe rejection at the first and later transitions, corrupt
   current state/value/score, missing evidence, finite-status-first failures,
   replay/chunk equivalence and unchanged healthy transitions. Run existing
   modern-kernel, TFP rejection and candidate/export integration regressions.
3. Inspect saved real-target failure traces. Update the tuning reference and
   LaTeX guide locally to explain the exact distinction. Preserve other tasks'
   dirty files; commit only this task's changes and required pre-existing
   modern-trace dependencies.
4. If checks pass, capture a fresh source copy. Run fresh GPU/XLA preflight and
   all-nine-pair prices, followed by a bounded full-design qualification tranche
   under the remaining allocation. Export/reload only validated members. Do not
   reuse prior tuning evidence or requalify old failures. Preserve checkpoints
   on a budget stop and settle the outer inclusive cost once.

Budget: tests initially capped at 900 seconds, an operational debugging ceiling
informed by the prior 107-test suite's 113-second measured cost, not a convergence
criterion. Actual diagnostic runtime is charged before launch. Preflight keeps
the inherited 600-second numerical ceiling, prices 1,800 seconds, and closeout
reserves 1,020 seconds. Queueing and all child stages consume the single balance.
The full previous run cost 33,911 seconds, so completion of this shorter tranche
is uncertain and must not be promised. No extra compute is authorized here.

The nine pairs, four starts, 64 transitions/trial, 16/32/64/128 evidence ladder,
multiplicity cap 24 and .55 qualification floor are inherited unchanged from
the reviewed diagnostic; none is a general default. Qualification root
(20261027,6101), pricing root (20261027,5100) and reload root (20261027,2001) are
predeclared reproducibility keys, not favorable-seed choices. New artifact root:
N5/rejection-consistency-20261027-01, with source hashes, commands, environments,
seeds, device/memory policy, run logs, test receipts and terminal result.

GPU sharing remains below 50 percent utilization with at least 4,096 MiB free
at launch, reassessed every 1,800 seconds; waiting checks every 60 seconds.
These inherited resource settings are hypotheses about safe sharing, not
scientific criteria. Use trusted execution, pre-import memory growth, TF/TFP,
stable signatures and XLA. No NUTS, pfor or runtime NumPy is introduced.

Before the numerical launch, refresh this plan with actual checks and budget;
at terminal closeout update the master with completed, partial or failed status
and the next justified action. Weak evidence cannot become a release claim.

## Prelaunch review and results

The focused working-tree suite passed 81 tests, and the public-controller,
state-space, export/reload and interruption suite passed 80 tests. Eight
healthy-path comparisons against the prior frozen implementation preserved
positions and every original trace field exactly. The two saved real-model
failures first failed at target evaluation with nonfinite value and score;
neither was a finite-status-first failure. Their original rejection decisions
and original evidence remain unchanged.

The scoped commit was also tested in a separate HEAD-plus-staged source tree:
60 tests passed without unrelated working-tree repairs. Its first import
attempt failed because the copied source lacked the prebuilt custom operation;
explicitly pointing to the existing shared operation directory repaired this
test environment. Two earlier fixture failures were repaired before the
81-test pass. All failed attempts are charged. The chapter builds to 47 pages;
the changed paragraph on PDF page 39 was visually inspected. Broader manuscript
edits belong to another task and are excluded from this commit.

The commit includes the required pre-existing first-failure instrumentation,
terminal-row preservation, safe-zero native-health reductions and acceptance
evidence handling. It excludes unrelated cache/chunk/schema changes, divergence
policy changes and other tasks' expanded guide sections. Execution sources will
be copied and individually hashed; the run manifest distinguishes those exact
working-tree dependencies from the scoped Git commit.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Proceed to fresh bounded diagnostic | Rejection/selection tests and healthy parity pass | No unresolved tested mechanics veto | GPU/real-target behavior and finite budget | Fresh preflight, prices and unchanged full-design tranche | Posterior validity or release readiness |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Invalid starts, finite-status-first failures, malformed evidence and selected-state corruption are rejected in tests. |
| Statistically supported ranking | None; no method-ranking experiment was run. |
| Descriptive-only differences | Historical candidate outcomes are localization evidence only. |
| Default-readiness | Not established. |
| Next evidence needed | Fresh GPU/XLA target checks and acceptance qualification, followed by separately funded posterior evidence. |

Red-team: selected-value consistency does not validate a wrong target evaluator,
and toy overflow tests cannot establish target-specific numerical behavior.
Healthy parity rules out the tested path changes only. The fresh real-target
preflight and source-isolated tranche answer the remaining local question.

Recorded diagnostics cost 487.526335490 seconds. The remaining
combined allowance is 16964.992436632 seconds, before any new
reservation. The separate SIR allowance remains untouched.
