# Generic NeuTra adaptive five-stage repair result (2026-08-15)

## Outcome

The generic controller now supports bounded held-out plateau scheduling and an
opt-in `carry_selected` Adam-state policy. The implementation and GPU/XLA
campaign completed correctly, but the proposed repair did not make the fixed
five-stage recipe general across the correlated Gaussian and banana controls.

At the matched 3,000-update ceiling, cold joint training passed the Gaussian
known-law gate in both seeds. Adaptive staging with reset moments and with
carried moments failed in both seeds. On banana, all three routes failed in both
seeds. Carrying Adam state did not repair either target and is not supported as
a default.

## Claimed and computed quantities

| Item | Status |
|---|---|
| Claimed target | A generic bounded adaptive stage-four scheduler and an exact selected-checkpoint Adam-state carry mechanism. |
| Quantity computed | Held-out reverse-KL checkpoint histories plus an untouched 131,072-draw known-law screen for coordinate means, second moments, and adjacent cross moments. |
| Equality of target and computation | Correct for the stated training-controller and known-law proposal diagnostics. Held-out loss was used only for scheduling and selection. |
| Supporting artifacts | `docs/plans/artifacts/neutra-generic-adaptive-five-stage-repair-2026-08-15/` |
| Not evaluated | HMC, SSL-LSTM posterior correctness, multimodal coverage, universal staging, and default readiness. |

## Implementation result

`bayesfilter/inference/neutra_staged_training.py` now provides
`NeuTraAdaptiveStagePolicy`. An adaptive phase records its hard update cap,
minimum updates, checkpoint patience, minimum improvement, LR-reduction factor,
maximum reductions, executed updates, checkpoint history, reductions, and stop
reason.

`train_neutra_five_stage` now accepts:

- `phase_reset`, the backward-compatible behavior; and
- `carry_selected`, which prebuilds Adam slots for the complete transport,
  restores identical incoming model/optimizer state for every LR candidate,
  updates only active variables, and carries the optimizer state paired with
  the selected model checkpoint.

Carry mode fails closed if active masks are not cumulative. Existing callers
continue to use fixed stages and phase-local optimizer resets unless they opt
into the new behavior.

## Run manifest

| Field | Value |
|---|---|
| Git commit recorded by run | `3030d86df9cb00346df82c7c19f015c09c7c6e1f` |
| Model runner | `docs/benchmarks/run_neutra_generic_adaptive_five_stage_model_2026_08_15.py` |
| Campaign runner | `docs/benchmarks/run_neutra_generic_adaptive_five_stage_campaign_2026_08_15.py` |
| Environment | TensorFlow `2.20.0`, campaign Python environment |
| Device | GPU 0, trusted TensorFlow/TFP GPU path |
| Memory policy | Growth enabled and verified before logical-device initialization |
| Numeric mode | float64, XLA JIT on, TF32 off |
| Batch | 4,096; batch-native; no scalar or row-mapped fallback |
| Architecture | Three dense-IAF blocks, width `(32,32)`, ELU, identical initialization family across arms |
| LR grid | `2e-4`, `5e-4`, `1e-3` |
| Staged ceiling | `100 + 300 + 3*100 + 2300 = 3000` selected-path updates |
| Cold ceiling | 3,000 updates |
| Adaptive policy | check every 100; minimum 400; patience 4; absolute delta `1e-5`; LR factor `0.5`; at most 3 reductions |
| Audit | 131,072 untouched proposal draws per cell, 99.9% separate intervals |
| Targets/seeds/arms | Gaussian and banana; seeds 0 and 1; adaptive reset, adaptive carry, cold |
| Campaign wall time | 698.50 seconds |
| Time cap | 2,700 seconds |
| Campaign result | `docs/plans/artifacts/neutra-generic-adaptive-five-stage-repair-2026-08-15/campaign_result.json` |
| Integrity | 50 campaign artifacts, all hashes verified |

The staged and cold arms each performed 9,000 optimizer updates while tuning
three LR candidates. Actual selected-path updates are reported separately. No
runtime-efficiency comparison is claimed.

## Results

| Target | Route | Gate passes | ESS fraction by seed | Ratio SD by seed | Joint behavior |
|---|---|---:|---|---|---|
| Gaussian | adaptive reset | 0/2 | 0.99577, 0.99556 | 0.06430, 0.06773 | 2,300/2,300 updates; cap reached |
| Gaussian | adaptive carry | 0/2 | 0.99555, 0.99504 | 0.06638, 0.07111 | 2,300/2,300 updates; cap reached |
| Gaussian | cold joint | 2/2 | 0.99651, 0.99714 | 0.05792, 0.05327 | 3,000/3,000 selected |
| Banana | adaptive reset | 0/2 | 0.88955, 0.87867 | 0.30365, 0.30123 | 2,300/2,300 updates; cap reached |
| Banana | adaptive carry | 0/2 | 0.88997, 0.88034 | 0.30331, 0.30247 | 2,300/2,300 updates; cap reached |
| Banana | cold joint | 0/2 | 0.98653, 0.88447 | 0.17831, 0.31013 | 3,000/3,000 selected |

ESS, ratio SD, losses, and LR reductions are descriptive. The predeclared
known-law screens determine viability.

### Gaussian

Both adaptive staged routes passed all adjacent cross-moment screens, so the
remaining error was not failure to learn the correlation pattern. They failed
small coordinate-location and scale screens:

- adaptive reset seed 0: 2/16 means and 3/16 second moments failed;
- adaptive reset seed 1: 4/16 means and 2/16 second moments failed;
- adaptive carry seed 0: 2/16 means and 4/16 second moments failed; and
- adaptive carry seed 1: 4/16 means and 2/16 second moments failed.

Typical failed means were about `0.01` from zero. Failed second moments were
about `0.985` or `1.02` instead of one. These are small absolute errors, but
they were outside the predeclared Monte Carlo intervals and repeated by failure
class across seeds. Cold joint passed every screen in both seeds.

All selected staged joint candidates used the terminal checkpoint. The selected
held-out loss was about `5.918`, while the historical 1,000-update cold route
was already about `5.901`. More joint time improved the staged proposal-law
diagnostics substantially but did not erase the path dependence introduced by
the fixed continuation allocation. Adam carry was descriptively slightly worse
than reset and does not explain the failure.

### Banana

Every route passed all coordinate means and adjacent cross moments. The hard
failure was marginal variance in the inverse-transformed latent coordinates.

The staged routes were highly reproducible:

- first latent second moment: `0.810-0.824` instead of `1`;
- second latent second moment: `1.079` instead of `1`; and
- one seed in each staged arm also narrowly failed coordinate 10 near `1.014`.

Stage one and stage two selected update zero in both banana seeds. The
controller correctly rejected those target-inappropriate changes, but the
selected path consequently used 2,600 rather than the 3,000-update ceiling.
All three nonlinear progressive blocks selected their 100th update and the
joint phase selected its 2,300th update. Thus the stage-four scheduler was still
improving at the cap; it did not diagnose a converged solution.

Cold joint seed 0 was much closer, with first latent second moment `0.942`, ESS
`0.9865`, and ratio SD `0.1783`. Cold seed 1 fell into the same distorted basin
as the staged routes, with first two second moments `0.821` and `1.082`. The
banana result is therefore seed-sensitive optimization failure. No route is
viable, and no statistically supported ranking is claimed.

## Mechanism verdict

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Fixed staging failed only because stage four was too short | Stage four grew from 300 to 2,300 updates and every selected candidate still hit the cap; Gaussian still failed 0/2 | Weakened; insufficient joint duration alone is not the full cause |
| Phase-local Adam resets caused the failure | Carry and reset produced nearly identical pass/fail and banana geometry | Rejected as the main cause under this configuration |
| The generic adaptive controller is broken | Mechanics tests, trusted GPU/XLA canary, finite campaign, exact-law harnesses, and artifacts all passed | Rejected |
| Dense IAF lacks Gaussian capacity | Identical cold architecture passed Gaussian 2/2 | Rejected |
| Banana is solved by more reverse-KL updates alone | All arms hit their cap; cold was strongly seed-dependent; staged variance distortion repeated | Unsupported |
| The fixed five-stage recipe generalizes across unimodal models | Failed Gaussian and banana after the repair | Rejected |

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep adaptive/carry mechanisms as optional generic API | Controller invariant tests and complete GPU/XLA campaign | No engineering veto | Other transports and schedules are untested | Retain experimental opt-in capability | Default training policy |
| Do not promote `carry_selected` | No target changed from fail to pass; Gaussian metrics were slightly worse descriptively | Known-law failures remain | Different Adam treatment for newly activated variables is untested | Leave off by default | Carry is universally harmful |
| Stop forcing five-stage continuation on Gaussian | Cold passed 2/2; both staged arms failed 0/2 | Repeated mean/variance failures | Whether a shorter target-specific continuation can pass | Select cold joint for Gaussian controls | Cold is universally superior |
| Keep banana unresolved | All arms failed 0/2 | First two latent variances failed | Initialization, architecture ordering, and objective basin remain candidates | Run a target-specific banana protocol with initialization/ordering arms | Architecture impossibility or scientific invalidity |
| Do not run HMC from these banana transports | Proposal known-law gate failed | Promotion veto fired | N/A | Repair proposal geometry first | HMC correctness |

## Inference status

| Evidence question | Status |
|---|---|
| Hard veto screen | Gaussian cold passed; all other target/route combinations failed their model-specific gate. |
| Statistically supported ranking | No continuous-metric ranking is supported. Replicated Gaussian pass/fail supports route viability classification for this target. |
| Descriptive-only differences | ESS, ratio SD, loss, runtime, LR reductions, and banana between-arm differences. |
| Default readiness | Not supported. |
| Next evidence needed | A target-specific banana training protocol; no further universal five-stage tuning campaign is justified. |

## Engineering, numerical, and scientific ledgers

| Ledger | Verdict |
|---|---|
| Engineering correctness | Supported: scheduler bounds, state restoration, cumulative-mask enforcement, phase isolation, validation non-mutation, GPU memory growth, XLA, and artifact hashes passed. |
| Numerical/training validity | Runs were finite and batch-native. Adaptive candidates reached the hard cap rather than a convergence stop. |
| Scientific interpretation | The repair mechanisms did not generalize the five-stage recipe. Gaussian cold remains viable; banana remains unresolved. |

## Post-run red team

The strongest alternative explanation for staged Gaussian failure is the fixed
ordering and early nonlinear continuation path, not lack of capacity or Adam
reset. A target-specific schedule that omits unnecessary early phases could
test this, but another universal stage recipe is not justified by current
evidence.

For banana, the strongest alternative explanation is initialization/basin
sensitivity. Cold seed 0 approached the correct law much more closely than cold
seed 1, while both staged policies reproducibly entered the same distorted
variance basin. A target-specific comparison of permutation/order,
identity-biased initialization, and a direct single-block banana-capable arm is
the smallest discriminating next experiment.

The weakest evidence is any claimed ordering among failed banana arms. Two
seeds and continuous metrics are insufficient for that ranking.
