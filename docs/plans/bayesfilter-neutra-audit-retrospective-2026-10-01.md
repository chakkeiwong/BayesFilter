# Why the first NeuTra repair review missed the remaining failures

The first audit found real defects, and the repair corrected several of them.
The failure was incomplete verification of the repair: changes to individual
functions and the controller's phase order were not tied to evidence that the
actual downstream consumer received and used the promised numerical behavior.
Some known issues therefore survived, and switching to discovered modes
introduced an initialization risk that was not checked at the HMC boundary.

This is an engineering retrospective requested by the owner, not another
training experiment. It compares the September 30 pipeline audit, repair plan,
source and tests with the completed causal audit. No new numerical runs,
thresholds, budgets or candidate promotions are involved. All numerical results
below come from the preserved September 30 evidence. The next work is to use
the concrete closure checks below when repairing the demonstrated defects.
These checks are requirements for that repair; this note does not report them
as implemented or passed.

The skeptical check for this retrospective is to avoid equating an unsuccessful
research candidate with an unsuccessful audit. The older documents explicitly
limited their conclusions and did not certify reliable training on every
target. In particular, an SMC teacher failing its independent reference check
is useful evidence. The narrower engineering failures are a mismatch between
promised and executable repairs, unchecked changes in the meaning of inputs,
and tests that establish less than the repair claim requires.

## What was missed, with concrete evidence

1. **An identified continuation defect was not closed.** The earlier pipeline
   audit, sections 7–8, already identified inadequate bridge calibration and
   progress that did not drive continuation. The new `fit` still ends its
   literal ladder at 8,192 updates
   (`bayesfilter/testing/neutra_warm_start_closure.py:305`) and returns
   `fresh_initialization_same_grid` when no candidate passes (line 334).
   `Controller.run` retries this phase without restoring the improving fit.
   Saving an optimizer checkpoint is not the same as exercising continuation
   from it. The later audit found 11 nonlinear fits with lower loss at the last
   rung; that descriptive evidence should have triggered the planned diagnosis,
   not been lost inside the common failure status.

2. **The mutation repair in the plan is absent from the executed teacher.**
   The repair plan promises separate particle/mutation repairs and lists
   mutation steps 4 → 16. `teacher` instead constructs
   `SMCConfig(particles,4,...)` at every repair level and explicitly reports
   `increase_particles_only` (closure lines 213 and 231). The repair did add
   pilots at beta 0, 0.5 and 1, but all are resampled from one 256-point proposal
   bank and produce one shared step size. Evaluating three bridge densities
   does not establish representative pilot coverage or useful movement.
   The later curvature repair correctly made a stiff pilot executable; its
   stiff-Gaussian test explicitly disclaims teacher quality. That limited
   success cannot close the proposal-overlap or mutation-mobility questions.

3. **A change of input meaning was not reviewed through the HMC call chain.**
   The target's old funnel representatives are scale-region points at
   V = -1, 0, 1 (`neutra_warm_start_targets_tf.py:127`). The repaired
   `qualify_selected` explicitly supplies discovered modes (closure line 431).
   `qualify` then calls the unchanged `initial_walkers`, which adds physical
   jitter of 0.2 to each coordinate. For this funnel, the discovered density
   mode is V = -9 with child standard deviation exp(-9). The repair plan's own
   later calibration analysis had already derived this narrow scale, but did
   not propagate its implication to HMC initialization. The map round-trip
   assertion only checks invertibility; it cannot establish suitable starts.
   The saved-map intervention subsequently demonstrated zero versus 0.999388
   mean one-step acceptance for recorded versus learned-map starts.

4. **The retry's stated intention was checked instead of its effective domain.**
   The worker increases work units, candidate allowance and refinement rounds
   (`scripts/run_neutra_warm_start_master.py:143`). The actual tuning consumer
   still fixes `fixed_grid_max_attempts=3`
   (`neutra_warm_start_qualification.py:81`). The funnel runs exhausted that
   downward-repair range after only 12 of 144 work units. A passing Gaussian
   retry did demonstrate a working retry route; it did not demonstrate that
   the route widens the epsilon boundary responsible for the funnel failure.

5. **A statistical rule was unit-tested without calibrating its interpretation.**
   `test_temporal_conflict_precedes_band_compatible_interval` in
   `tests/test_hmc_verification.py:480` deliberately expects inconclusive
   evidence when block means cross both practical-band boundaries. This tests
   implementation of the chosen rule, not its false-inconclusive frequency
   under stationary sampling. The causal audit supplied a recorded compatible
   global interval rejected by this rule, plus an illustrative stationary
   calculation. That calculation is not a calibration for correlated HMC;
   such calibration remains outstanding.

## Why the passing tests did not establish complete repair

The repair suite has useful tests for stale reuse, calibration failure,
reference separation, preserving independent branches and rejecting a broad
Gaussian fit. Those results should be retained.

However, `test_controller_repairs_fit_before_creating_fresh_final_reference`
replaces phase execution with a function that writes chosen pass/fail flags
(`tests/test_neutra_warm_start_repair.py:134`). It proves ordering and fresh
reference streams, but never invokes the teacher, trainer or HMC tuner. The
qualification codec test (`test_neutra_warm_start_master.py:120`) checks the
transport codec and batched adapter without calling `qualify`. Neither test
could detect the actual start bank or the effective epsilon repair domain.
The temporal-conflict test has a different limitation: it executes the rule,
but uses the rule itself as the expected scientific answer.

The recorded 55- and 67-test passes are therefore legitimate bounded regression
results. Treating them as evidence that all substantive repair promises were
implemented would be unsupported. The existing call-chain and default-audit
policies already require this distinction; additional approval ceremony is not
the remedy.

## Evidence required to close the next repair

Use one small table in the next repair plan linking each finding to its actual
consumer, its falsifying test and its saved result. Mark each row **open**,
**implemented but not verified**, or **verified for the stated scope**. A
deferred research question may remain open; it must not be reported as repaired.

| Repair claim | Required executable evidence |
|---|---|
| HMC uses suitable, reproducible starts | Invoke the real qualification entry point; capture the physical/latent states passed to the tuner and their source. Test funnel scale as well as Gaussian/mixture cases. Follow with a bounded proposal diagnostic on the repaired path. Round-trip success alone cannot pass this row. |
| Retry reaches the needed epsilon range | Exercise the real configuration-to-tuner path; record the effective domain and limits, then show an out-of-range failed pilot causes an actual proposal beyond the old boundary within budget. More configured work units alone cannot pass. |
| Improving fits can continue | Feed controlled improving and plateau histories through the real decision logic. Verify selected continuation restores parameters, Adam state and update count; plateau triggers the distinct repair, and exhausted budget preserves unfinished work. Do not prescribe continuation from one noisy loss decrease alone. |
| SMC has an effective mutation repair | Verify the actual teacher receives changed mutation controls at the relevant repair level. A bounded bridge diagnostic must assess pilot overlap, ancestry and movement on meaningful target scales, followed by the existing independent teacher accuracy check. Acceptance or weight ESS alone cannot pass. |
| The temporal rule distinguishes fluctuation from instability | Calibrate on stationary and deliberately changing reference sequences at the actual chain/block lengths, including relevant dependence. Preserve numerical and movement vetoes. Freeze any revised decision rule before fresh HMC verification; old candidates are not automatically admitted. |
| Training diagnostics remain inspectable | Persist the clipping counts and gradient totals already returned by training, and distinguish cap, plateau, numerical failure and shape failure in phase results. The consumer must use the declared repair trigger. |

For each boundary, test both a case that should proceed and a case that should
be rejected or repaired. Use independent mathematics or a declared reference
to specify the expected result. Mocks may control external cost or inject a
known failure; they must not replace the computation whose behavior is the
subject of the claim. Exercise the outer consumer to verify wiring and add the
smallest real numerical check needed to verify its meaning. Keep those two
evidence scopes distinct.

Before launching a broader matrix, run bounded checks that target the changed
mechanisms, including the funnel scale and mixture continuation cases. A
Gaussian success alone cannot validate transfer to either. After the first
repair executes, compare intended and actual parameter/state changes. If the
repair did not change the failing mechanism, classify that as a repair or
integration defect and fix dependent execution rather than spending every seed
on the same ineffective retry. A genuine failed candidate remains eligible for
the next declared scientific repair under the existing budget.

Review should ask whether these particular failures could still occur while
the proposed tests pass. Test counts, a refreshed status file and another
review signature do not answer that question. No finite audit guarantees every
future scientific outcome; these checks address the specific preventable
failure classes demonstrated here without creating a new approval process.

Evidence: [earlier pipeline audit](bayesfilter-neutra-warm-start-pipeline-audit-2026-09-30.md),
[repair plan and execution record](bayesfilter-neutra-warm-start-repair-master-2026-09-30.md),
[causal findings and numerical artifacts](bayesfilter-neutra-failure-causal-findings-2026-09-30.md).
