# q20 training sanity canaries: results

The low-cost checks identify three immediate issues: cap 10 does not act as an
occasional outlier safeguard, the final observation-bias scale is under strong
bound pressure, and batch 32 has substantial gradient noise at the selected
map. They do not implicate every training setting. Adam epsilon has negligible
influence on the stored moment direction, and eight disposable updates from
both the saved map and exact prior initialization remain finite and move the
hidden parameters.

These are sanity checks, explicitly **not hyperparameter optimization**. No
new setting was selected, no recipe was declared calibrated and no probe
checkpoint was promoted for estimation. The purpose is to find gross local
problems cheaply and identify what deserves a focused repair.

The [predeclared canary plan](bayesfilter-q20-training-sanity-canaries-2026-09-22.md)
defines the checks, numerical alert meanings and limits. The existing master
remains paused. Its broader comparison/continuation defects remain separate
from these independently executed diagnostics.

## Results by choice

| Choice | Finding | Canary verdict and scope |
| --- | --- | --- |
| Clip norm10 | All 31 saved diagnostic batches clipped; median gradient multiplier .18059. All 16 new disposable updates also clipped. Historical direct runs clipped 99.32%–100% of updates. | **Failed stated role:** routine normalization rather than occasional outlier protection. Does not prove that clipping causes all training problems. |
| Scale-log bound 2 | Selected final observation-bias output median -1.966965; slope at median .03276, approximately 30.5-fold attenuation relative to slope1 at zero. | **Review needed:** strong parameterization pressure. Does not prove global inability to represent the target. |
| Batch32 | On matched 896 saved points, RMS batch-gradient variation 63.35 versus pooled mean-gradient norm 19.93. Regrouping gives variation 34.71 at 128; five of seven such batches still exceed cap 10. | **Review needed:** noisy gradients; changing batch alone does not reliably remove clipping. No batch or speed/accuracy ranking selected. |
| Adam epsilon 1e-7 | None of 616 structurally active coordinates has sqrt(v)<=epsilon; removing epsilon algebraically changes the current moment direction by .003325%. | **No issue detected in stored moments.** Does not calibrate beta1/beta2, future moments or temperature transfer. |
| LR .0005 at the saved map | Eight accepted, finite updates; nonzero updates in every recorded parameter tensor; paired mean loss change -.18796±.12208. | **No gross local instability detected.** One short trajectory cannot establish optimal LR or sustained convergence. |
| LR .0005 and initialization SD .02 at prior initialization | Exact prior-affine law checked, including reversal; eight finite updates; hidden parameters move; paired mean loss change -4.73602±1.11044. | **No wrong initial law or completely dormant hidden layers detected.** Initialization quality and seed reliability remain unassessed. |
| Hidden tanh activation | Saved per-layer fractions abs(activation)>.95 are 0, 0, 0 and .043625. | **No widespread hidden saturation detected on this saved bank.** This differs from the separate output-scale saturation. |
| Training stopping/selection | Previously inspected first-eligible selection, skipped nominee continuation, same-map increments and pilot/calibration label mismatch remain. | **Failed design checks.** These need implementation repair, not adjustment of numerical thresholds. |
| Architecture adequacy, posterior coverage, beta schedule, Adam transfer | The beta-one, one-recipe probes do not test these questions. | **Not assessed.** They must not inherit a pass from the other rows. |

The displayed loss uncertainty is the inherited normal-approximation half-width
on 128 paired development rows. It is descriptive. The probes use different
starting states and banks and are not a comparison of initialization versus
warm starts. No candidate superiority, confidence about training-seed
reliability or nominal sequential coverage is claimed.

The clipping rule is intentionally a gross contradiction test: if a majority
of ordinary batches are modified, the clipper is not merely treating unusual
outliers. The scale warning means at least tenfold local attenuation; the
epsilon alert would mean at least 10% change in the stored moment direction.
These are transparent engineering alerts, not derived optimal thresholds or
posterior tolerances. Passing one means only that its named problem was not
detected. Substantial noise near a stationary point is possible, so the batch
screen is a warning rather than a rejection.

## What the probes actually did

The script read the saved gradients, map geometry, full training state and
optimizer slots. It excluded 360 structurally masked parameters from the
epsilon check. The epsilon-zero expression was evaluated only as a mathematical
limit of the stored moment direction; the actual optimizer retained epsilon 1e-7.
The 32-to-128 regrouping averages four disjoint stored batch gradients at the
same parameters, so the pooled gradient is unchanged. It makes no new target
calls and provides no GPU throughput evidence for 128-row training.

Actual training probes used the selected width 16/two-stage map, LR .0005,
batch 32, cap 10 and existing Adam configuration unchanged. The saved-checkpoint
probe restored exact weights/moments and verified the saved frozen export.
It then used a separate diagnostic RNG stream. The initialization probe used
the actual prior initialization and checked its forward map and log determinant.
Both ran exactly eight disposable optimizer updates with fresh evaluations of
the real q20 target; cached scores were never used at changed weights.

Each state had 128 separate development base points evaluated before and after
the updates. Per-step actual update norms, variable movements, loss arrays,
base points, physical positions and disposable complete checkpoints were saved.
The saved-map update norms range .001556–.001961; prior-init update norms range
.005799–.007193. These are descriptive parameter-coordinate quantities, not
rates of posterior learning.

The repeated trainer, evaluator and map functions each traced once and used
float64 TensorFlow/XLA on host GPU 1, an RTX 4080 SUPER. The training target
remained batch-native. GPU memory growth was configured and verified before
initialization. TensorFlow allocator peak was 268,946,944 bytes. Original input
files and numerical settings were preserved; disposable results have an
explicit debug-only role and are not estimation inputs.

## Decision and next use

The useful next action is a focused repair of the clipping safeguard and
output-scale pressure, followed by the same cheap canaries on the repaired
recipe. A provisional clipping threshold can be proposed from separate pilot
norms and a declared intervention policy, then checked on fresh batches. Passing
on the same data used to choose its threshold is not independent evidence.
Both initialization and a learned state matter because gradient scales change
during training. This procedure checks whether the safeguard serves its
declared role; it does not optimize a threshold for loss or sampler efficiency.

Keep epsilon out of the immediate repair priority unless new evidence
implicates it. Do not select batch 128 merely because its descriptive variance
is smaller; its cost and actual learning were not tested. The finite eight-step
probes also do not justify ending training at eight steps or accepting a map
for HMC. Longer training and downstream evidence answer those different
questions.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject cap 10 as an occasional-outlier guard in this scope | Ordinary batches repeatedly clipped | Role contradiction; no numerical crash | Long-run harm and a suitable replacement policy | Propose a bounded repair and check fresh normal batches | Clipping is the sole cause or should always be removed |
| Review scale-bound pressure and batch noise | Measured attenuation and batch variability | No map/status failure in the short probes | Compensation by other stages; learning benefit of a repair | Focused parameterization/optimizer diagnostic | An optimal architecture, scale bound or batch size |
| Retain epsilon and LR as locally viable hypotheses | Epsilon effect small; short unchanged updates finite with movement | No finite/status/update veto | Other states, seeds and later learning | Continue proportionate monitoring while addressing demonstrated issues | Hyperparameters calibrated or posterior-ready |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No new target/status/finite-state veto; clipping-role and controller design checks fail |
| Statistically supported ranking | None; there was no hyperparameter search |
| Descriptive-only differences | Loss changes, update norms, batch regrouping and geometry statistics |
| Default-readiness | Unchanged; no recipe or probe map promoted |
| Next evidence needed | Fresh repaired-setting sanity checks, funded real training and downstream validity |

Post-run review: learning proceeded despite clipping, so the proposition that
cap 10 completely prevents learning is contradicted by these probes. It can
still distort or impair longer optimization, which was not tested. A bound
near saturation can be compensated elsewhere; the short run cannot distinguish
that possibility from inadequate scale capacity. The weakest evidence is one
short path per starting state, with no independent seed replication.

## Reproduction and budget

Command:
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/plans/artifacts/q20-training-gap-investigation-2026-09-22/run_training_sanity_canaries.py`.

The existing supervisor charged **131.038409 seconds**, below the 360-second
cumulative cap. There was one attempt, with sixteen disposable updates total.
The remaining recorded balances are **155,561.168 campaign seconds
(43.211 hours)** and **771.319 diagnostic seconds (12.855 minutes)**.

- [Structured result](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00009-training-sanity-canaries/worker/result.json)
- [Manifest, source/input hashes, exact settings and device evidence](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00009-training-sanity-canaries/worker/manifest.json)
- [Budget receipt](artifacts/q20-training-gap-investigation-2026-09-22/sanity-budget-receipt-01.json)

Reset: these canaries satisfy the requested cheap sanity inspection for the
named checks. They do not complete the larger calibration design. Keep the
failed, review-needed and not-assessed outcomes explicit, preserve the original
map, and do not rerun all checks after an unrelated edit. Recheck the affected
scope after a material setting or execution change.
