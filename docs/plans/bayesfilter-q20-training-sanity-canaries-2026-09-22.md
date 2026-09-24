# q20 low-cost training sanity canaries

Owner request: check whether the current training settings are sensible at low
cost, explicitly without optimizing hyperparameters. This takes priority over
starting the larger calibration design. A canary can expose an obvious defect
or a setting that does not serve its stated purpose. Passing means only that
the specified problem was not detected in the tested scope. It does not mean
optimal, calibrated, converged, well-whitened, posterior-correct or production
ready. No hyperparameter search, candidate ranking or production-map update is
performed here.

## Contract and scope

Use the preserved r2 numerical source and current q20/T30 posterior, with the
selected width-16, LR-.0005, batch-32, cap-10 recipe. Inspect saved gradients,
Adam slots, histories and geometry first. Then run the unchanged recipe for
eight disposable updates from each of two states: exact prior initialization
and the selected saved checkpoint. The saved checkpoint remains untouched;
probe exports cannot be used by the master for estimation.

Outcome vocabulary:

- `no_issue_detected`: the named limited check passes; no adequacy claim.
- `review_needed`: a finite, valid observation indicates pressure/noise or
  sensitivity worth investigating; no automatic scientific rejection.
- `failed_check`: an invariant fails, or the setting contradicts its declared
  operational purpose. State which, rather than conflating them.
- `not_assessed`: the cheap evidence cannot answer this question.

Invalid target/status, nonfinite state, changed source/input, wrong checkpoint,
non-batched training, incorrect GPU allocation policy, contention or deadline
invalidate/stop the affected executable probe. Failed clipping-role or scale
pressure screens do not stop the disposable diagnostic: they cannot grant
promotion, but observing finite update behavior still answers a different
question. This is not a claim that the current clipping choice is acceptable.

## Canary definitions and numerical provenance

| Choice | Cheap measurement | Screen and meaning | What it cannot establish |
| --- | --- | --- | --- |
| Clipping | Saved batch norms and factors `min(1,C/norm)`; actual short-run clipping | If more than half of ordinary batches are clipped, fail the declared *occasional outlier guard* role. One half is the definition of majority, not an optimal clipping rate. Report attenuation and existing longer-history evidence. | Whether normalized-gradient training would work or which cap is best |
| Scale bounds | Per-output scales and derivative `1-(s/s_max)^2` on saved points | Flag a median derivative below .1: at least tenfold local attenuation. This is a transparent engineering warning, not a theorem of inadequate capacity. | Global expressiveness or benefit of increasing the cap |
| Hidden activation/init | Saved hidden saturation; verify exact prior map; compare hidden and output parameter movement over eight updates | Nonfinite or wrong initial map fails. Lack of any hidden movement after the bounded probe is a review trigger. Zero hidden gradient at the very first zero-output update is expected. | Good initialization, sufficient depth/width or seed robustness |
| Adam epsilon | Saved active-coordinate `sqrt(v)` versus epsilon; algebraic direction sensitivity to removing epsilon | Exclude structurally masked parameters. Flag over 10% relative change in the current moment direction; .1 is an engineering sensitivity alert. Record dormant coordinates separately. This is not a new optimizer setting. | Suitability of beta1/beta2 or long-run optimizer convergence |
| Batch size | Existing independent batch gradients; aggregate four 32-row gradients to obtain exact 128-row means at fixed weights | Report noise and norm changes on the same first896 points. Noise exceeding estimated mean norm is a review trigger, not a batch rejection rule: the true mean may be near zero. | Selection of128, an efficiency ranking or the population gradient |
| LR/actual update | Eight unchanged updates, finite parameters/slots, per-layer deltas and paired loss before/after | Invalid state fails. Zero overall movement needs review. A positive loss-change lower bound needs review; finite short-run behavior can only pass the local stability screen. | Optimal LR, sustained progress or convergence |
| Objective/map algebra | Reuse completed value/score/VJP checks; verify prior map and frozen checkpoint identity | Existing local numerical evidence remains scoped; exact restore and finite status are mandatory. | Global correctness or exact-model posterior accuracy |
| Stopping/selection/seed wiring | Read actual counters, distinct checkpoint identities and dispatch rules | Flag first-eligible selection, same-map increments and stopping while learning continues as already demonstrated design defects. | That a different recipe would cure these defects |
| Temperature spacing/moment transfer, architecture sufficiency, coverage | State the absent evidence explicitly | `not_assessed` by this short beta-one canary; do not manufacture a green light | These require focused later evidence if implicated |

The saved batch regrouping uses 28 complete existing batches, giving seven
128-row groups. All original individual-batch evidence is retained. The
comparison is descriptive, with no claim of statistical superiority from
seven groups. Adam epsilon sensitivity uses the stored moment direction
`m/(sqrt(v)+epsilon)`; common learning-rate/bias-correction factors cancel.
The epsilon-zero expression is only an algebraic limiting diagnostic, with
zero-moment/zero-variance coordinates reported as dormant, not a suggested
training configuration or reconstructed unclipped history.

Eight updates per state is a convenience cost cap intended to expose gross
local failures; it has no convergence interpretation. Batch32, optimizer and
architecture values are inherited unchanged. A fresh 128-row development bank
(four training-sized blocks) is a low-cost observation bank, reused only within
each before/after comparison and separate from training seeds. The inherited
normal multiplier1.959963984540054 produces an approximate descriptive interval
for paired changes; no sequential coverage, seed reliability or ranking is
claimed. Unresolved change is not failure and does not justify spending more
validation time merely to obtain a sign.

## Execution and budget

Driver:
`docs/plans/artifacts/q20-training-gap-investigation-2026-09-22/run_training_sanity_canaries.py`.
Execute with `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` under trusted GPU
permissions. The existing campaign supervisor runs one disposable worker,
selects one available device, enforces memory growth before TensorFlow import,
and charges wall time to the current diagnostic allocation. Pure repeated
numerical kernels remain batch-native float64 TensorFlow/XLA with stable
signatures. No scalar target fallback or pfor is used. Three concurrent GPUs
are unnecessary for this bounded check.

The cumulative stage cap is 360 seconds, a convenience spending ceiling, not a
runtime prediction. It is below the current 902.358-second diagnostic balance.
Allow at most two infrastructure attempts within that same cumulative cap;
the second may repair a localized harness problem, never search for favorable
stochastic evidence. The worker cooperatively stops before the external cap.
It preserves manifests, exact settings/commands/seeds, sampled points, losses,
updates, summaries and failures in a fresh campaign attempt. A brief report
and budget receipt live with this plan's investigation artifacts.

Pre-run skeptical audit: saved gradients are valid only at their original map;
they will not be reused as scores after updates. Each actual probe update
evaluates the real target anew. Priors and likelihood scaling remain fixed.
Initialization and warm-start probes answer different local questions and are
not competing candidates. Screen constants have explicit engineering meanings;
passing does not promote a map, and inconclusive loss differences remain
inconclusive. Existing broader master defects do not prevent this independent
diagnostic from answering its bounded questions. The plan passes this audit.

Executed result: see the [canary report](bayesfilter-q20-training-sanity-results-2026-09-22.md).
One attempt completed in 131.038409 supervisor seconds. The report preserves
the failed checks and limited passes; this plan has not become a calibration
or optimization certificate.
