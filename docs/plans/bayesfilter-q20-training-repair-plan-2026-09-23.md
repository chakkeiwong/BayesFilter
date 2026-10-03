# q20 training repair and bounded continuation

Status: reviewed for implementation and execution, 2026-09-23. Owner request:
create, thoroughly review, and execute a plan closing the training gaps.

Execution update: implemented, 64 affected training regressions passed, and
11 checks passed in the frozen execution copy including complete Gaussian
estimation/replay. All three disposable GPU canaries passed. The four-arm
continuation completed; terminal review verified all update logs, checkpoint
checksums, lifetime counts, distinct comparisons and export parity. It used
3.086 aggregate worker hours. Scale saturation, residual geometry, seed/mode
coverage and fresh HMC evidence remain unresolved; see
[the execution note](bayesfilter-q20-training-repair-results-2026-09-23.md).

## Question and evidence contract

Can the existing beta-one NeuTra maps keep learning when their scheduler no
longer stops at trial eligibility, and do narrowly targeted clipping, capacity,
and batch repairs remain numerically viable? The target stays the saved q20,
T30, float64 UKF posterior approximation. The inference objective remains plain
NeuTra HMC or the tempered NeuTra ensemble, with identity latent mass.

The control is the selected saved width16, LR0.0005, two-stage map with its
complete Adam/RNG history. Cap10 is retained only as the observed control;
its near-universal intervention contradicts its stated outlier-guard role.
This is a bounded repair, **not hyperparameter optimization**. Loss differences
and geometry are descriptive; they cannot rank viable candidates, establish
mode coverage, or certify posterior estimates.

Engineering pass criteria: genuine distinct-checkpoint comparisons; preserved
lifetime counters and optimizer history; explicit state migration; no automatic
stop at trial nomination; alternative-map handling after failed tuning; bounded
parallel workers with sum-of-worker-time accounting; pilot labels that do not
claim calibration; successful tests through actual consumer paths.

Training viability requires finite target/score/parameters/Adam state, valid
target status, and frozen export forward/inverse/logdet/score parity. Invalid
source, data, state, or shared harness is a continuation veto. Candidate
deterioration or a failed clipping-role screen triggers the next declared
repair; it does not reject NeuTra. HMC remains a separate fresh tuning and
posterior assessment under the public fixed-transport tuner.

## Implementation phases and closure evidence

1. Repair scheduler and assessment history. Use transport-state identity to
   detect unchanged maps. Preserve the last distinct comparison and historical
   assessments during import. Training nomination alone cannot skip later rungs.
   Keep exact resume separate from changed-recipe warm starts.
2. Repair master fallback: try eligible maps in declared operational order,
   without loss ranking; after an exhausted map set request the next training
   rung, with fresh tuning stage identities. Budget exhaustion pauses honestly.
   Rename the 128-update calibration endpoint as a sanity pilot.
3. Permit explicitly configured positive tanh depth. Add a checked two-stage
   identity extension to a saved map: after the original stages, the two new
   identity maps and two reversals cancel. Keep s_max=2; check forward, inverse,
   logdet and score equivalence before updates. Preserve old Adam slots, zero
   the new slots, retain the Adam iteration, and record this warm-start rule.
   This changes the optimization problem, not the initial map. It does not prove
   adequate capacity or resolve saturation in existing layers.
4. Add a standard-library queue to the master, with one explicitly assigned
   available GPU per worker and isolated output/cache. Reserve the sum of all
   live worker caps before launch. Charge each worker's measured wall time,
   including failures; after lost supervision charge the full cap. Preserve
   attempt directories and normal source checks. No new approval-token system.
5. Run focused CPU/reference engineering regressions, then a short GPU sanity
   phase and the funded continuation below. Save complete checkpoints, common
   heldout losses, clipping counts, timing, source and device provenance.

## Numerical choices and provenance

| Choice | Provenance and purpose | Failure risk and early check |
| --- | --- | --- |
| width16, LR0.0005, tanh, s_max2, Adam(.9,.999,1e-7), prior-affine, float64 | Existing selected map; warm-start hypotheses, not tuned defaults | Existing scale saturation, noise and missed modes; keep telemetry and downstream checks |
| clipping repair | Maximum raw norm over the last128 accepted saved updates and the separate saved31 diagnostic batches | A local observed envelope, not a tail bound; fresh batches must show that ordinary updates are not mostly clipped; no objective-based threshold selection |
| depth4 | Derived function-preserving addition of two identity stages; minimum even extension with reversing permutations | New slots have zero moments at the old Adam time; check map equivalence and finite update canary; no exact optimizer-equivalence claim |
| batch32 control; batch128 repair | Existing batch and saved 4-to-1 regrouping that reduced noise descriptively | Larger batch may cost more or learn less per time; measure actual updates and timing, do not select by variance alone |
| 1024 added updates for control/clip/depth; 256 for batch128 | Same 32768 target rows per arm; 1024 is an existing rung increment and measured ~40–41 minutes on the old map | Equal rows are not equal compute or independent replications; report both, no ranking |
| checkpoint every128, assessment at start and end | Existing checkpoint spacing; removes repeated expensive validation | May miss an intermediate loss excursion; per-update numerical vetoes remain |
| common768-row bank, evaluation batch32, separate final bank | Existing smallest funded bank; fixed evaluation batching makes banks common across training batches | Reused-bank uncertainty is descriptive; untouched final bank is a screen, not posterior evidence |
| reliability32, rtol1e-9/atol1e-10 | Existing export checks | Local map parity only; CPU nonzero-map fixtures plus GPU checks |
| clipping-role majority screen | Earlier owner-requested cheap sanity contract: an occasional guard cannot modify most ordinary batches | Gross contradiction check, not a calibrated intervention probability |
| worker cap5400 seconds; at most two infrastructure attempts per arm | Engineering ceiling: >2x measured ~2460 seconds/1024, includes setup/validation; retries share each arm's cap | Slow/new device may need pause; never renew cap on retry |

The batch128 arm follows the first completed worker in the three-device queue.
Order is control, clipping repair, identity-depth repair, batch128 repair. All
repair arms start from the same saved state; none consumes a disposable canary
checkpoint. The common training stream is retained where batch shapes match.
No broad LR/architecture/temperature search is added.

Actual availability at launch: GPU0 was occupied, so the queue uses GPUs1 and2.
Actual fresh canaries: eight updates per repair, with128 heldout rows and the
same fixed evaluation batch32. All are disposable; production continuations
restore the original saved checkpoint. The repaired clip is
159.39995043193295. Every canary clipped zero updates.

## Source, budget, and execution

The live campaign05 balances at recovery are 155561.16818836093 campaign
seconds and 771.3191494950661 diagnostic seconds. Re-read before reservation.
The existing deadline is 2026-09-25 18:00 +08:00. Three GPUs do not triple the
allocation. The continuation reserves at most 21600 aggregate worker seconds
(four arm caps); required sanity checks use at most600 diagnostic seconds,
which are also charged to the campaign. Remaining allowance stays protected.
Routine CPU engineering tests are recorded separately from numerical research.

Freeze execution in a fresh /tmp source copy of the previously executed
/tmp/BayesFilter-q20-recovery-20260922-r2, overlay only reviewed repair files.
Verify the target/trainer numerical files against that preserved source. The
shared workspace has unrelated numerical changes, so it is not a valid basis
for silently transferring the old derivative evidence. Record the overlay,
Git commit, ordinary file hashes, exact worker commands, input checksums,
environment, seeds, device/growth/XLA records, wall time and result paths.

Output root: docs/plans/artifacts/q20-training-repair-2026-09-23, fresh versioned
campaign/attempt directories. Master entry: the production benchmark driver
repair-training mode with a concrete repair request. The request and manifest
will preserve the exact absolute source, input and output paths used.

Persistence: `scripts/q20_training_repair_watch.py` observes the ordinary
campaign lock every30 seconds, may resume the unchanged master once after its
owned process deadlines expire, and writes `campaign-01/tranche-summary.md`
after `result.json` exists. The installed user service is
`q20-training-repair-watch-20260923.service`, with a22500-second runtime ceiling:
21600 worker-cap seconds +600 diagnostic-cap seconds +300 seconds of polling
and termination margin. This is an engineering ceiling, not another compute
allocation. The master still enforces cumulative per-arm and total budgets.

## Skeptical review before implementation

Review found and corrected these design traps:

- A floor or trial nomination was being treated as training completion. Remove
  that transition and test continuation of an already eligible checkpoint.
- Imported identical maps cannot create an incremental loss or plateau. Keep
  history and require a distinct predecessor; absence means unassessed.
- Raising s_max at fixed weights changes the map. Use the derived identity
  extension, with explicit optimizer treatment and executable parity checks.
- A larger batch changes random blocks. Common validation uses a fixed batch32
  evaluator; comparisons across arms remain descriptive.
- A first-map failure says nothing about the remaining cohort. Test fallback
  and return-to-training wiring, not merely a helper's existence.
- Several GPUs may be occupied by other work. Respect fresh inventory and use
  only available devices; explicit selection must survive worker startup.
- A parallel wall clock understates cost. Reserve and settle individual workers,
  including setup, exceptions and interrupted attempts.
- Fresh core numerical changes would invalidate old derivative evidence. Freeze
  the inspected numerical implementation, without modifying unrelated work.
- The 128-update endpoint is a sanity pilot. Its name and returned metadata
  must say that calibration and posterior validity remain unestablished.

Review conclusion: the revised plan answers an engineering/training-repair
question within the available allocation. It deliberately does not promise
that one tranche closes the scientific capacity, mode-coverage, or convergence
questions. At tranche close record the actual result and next funded phase;
do not quietly declare all gaps closed or reject the method.
