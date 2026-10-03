# q20 training continuation from the completed calibration

Status: `TRAINING_CONTINUATION_DIAGNOSTIC_COMPLETE`; finished September 18,
04:40:11 Asia/Shanghai. All eight maps reached 512 updates. The service exited
successfully and is inactive. The terminal review is below and in
`artifacts/ssl-lstm-q20-training-continuation-2026-09-18/training-review.json`.
Owner request: continue execution, September 18. This uses the settled September
17 allowance: 114407.59386372982 campaign seconds, including
38648.02608244992 diagnostic seconds. No allowance is renewed.

## Question and scope

Can the eight existing root-0 maps continue learning from 128 to the already
declared 512-update rung, and does all-update gradient clipping actually imply
small Adam parameter updates? The exact comparator is each map's own saved
128-update state, full Adam moments, RNG counter and beta. The immutable
execution source remains `/tmp/BayesFilter-q20-all-gpu-20260917`; current main
contains unrelated concurrent HMC work. No checkpoint scope is restamped.

Gate audit: the plain-NeuTra paper replication/enhancement gates are not closed
by this work. This is the owner's explicitly requested q20 engineering and
training diagnostic continuation. It cannot promote a paper baseline,
enhancement, learned map, posterior, whitening claim or production readiness.

The saved checkpoint is
`artifacts/ssl-lstm-q20-master-refresh-2026-09-17/campaign-all-gpu/attempts/00004-calibration/worker/data/cohort-00017.json`,
SHA-256 `e5943dd71142e8dc4adb9592ee772f61ce05647458d1389f1cb1e1554b0cb285`.
Its internal checksum, configuration, target, sources and complete training
states must match before numerical execution. Root 0 alone remains incomplete;
the other roots and continuation beta 1 are not silently removed from the full
protocol. Diagnostic checkpoints preserve ordinary training states for an
explicit later integration, but do not issue full-cohort admission.

## Evidence contract and decision roles

| Role | Evidence and action |
| --- | --- |
| Engineering pass | Exact complete-state restore; diagnostic gradient agrees with the actual compiled trainer's loss, raw norm and clipped Adam update within existing FP64 reliability tolerances; stable XLA graphs, batched target and finite state |
| Promotion veto | Any missing target/status, non-finite map/optimizer, failed forward/inverse/logdet/pullback check; no posterior or map promotion occurs in this phase regardless |
| Continuation veto | Invalid source/config/checksum, failed gradient/Adam equivalence, unsupported device/memory policy, competing numerical GPU worker, timeout or exhausted phase/campaign budget |
| Repair trigger | Deterioration on paired heldout losses; preserve that map and pause its extension rather than interpreting it as rejection of NeuTra |
| Explanatory | Raw/clipped gradient norm, actual update norm, matched one-step shadow updates, heldout means/intervals, loss variance, wall time and allocation |
| Next decision | Reprice the remaining cohort with completed work credited; retain all posterior/reference/tuning gates; no full campaign admission from partial pricing |

Heldout losses use the existing independent 768-row base-bank prefix and cached
beta-start/128-update losses. One new 768-row evaluation per candidate suffices
for this calibration question. The original 3072/12288 extensions remain in
the full protocol; this diagnostic reports unresolved precision rather than
expanding its bank. Reused-bank normal intervals are descriptive, not
anytime-valid confidence sequences or cross-candidate ranking evidence.

## Audit findings and mathematical checks

The actual trainer uses `mean(-target-logdet)` over a batch of 32. The target is
the declared log prior plus beta times the full T=30 log likelihood. Dividing
the target alone by the horizon would change the target; no such repair is
justified. The tape receives the analytic target score and differentiates the
batched transport and log determinant.

For raw parameter gradient g_t, the trainer supplies
u_t = min(1, C / ||g_t||) g_t to Adam. Adam has
m_t = b1 m_(t-1) + (1-b1) u_t and
v_t = b2 v_(t-1) + (1-b2) u_t^2. Its step is
-a_t m_t / (sqrt(v_t) + epsilon), with a_t carrying bias correction.
If every historical gradient and the current gradient are multiplied by the
same positive c, m becomes c m and v becomes c^2 v. Therefore the step is
-a_t m / (sqrt(v) + epsilon/c), not c times the original step.
MathDevMCP's symbolic check will be preserved with explicit nonzero denominator
assumptions. Batch-varying clipping factors are not constant scaling, so this
identity cannot prove clipping harmless or establish an optimal cap.

At each restored map, one diagnostic batch will obtain the raw gradient and
compare three Keras Adam updates on disposable copies: existing cap 10;
factor-two cap with moments rescaled consistently (local scale sensitivity);
and raw gradient with existing moments (an abrupt next-step switch only).
The third arm is not a reconstructed no-clipping training history. These
shadow copies never supply the continued map. The actual first continued step
must match the cap-10 shadow. The remaining steps use the unchanged trainer.
No cap is promoted from a one-step loss or update comparison. The inherited
cap's hypothesized protection against rare outlier gradients is still
uncalibrated; these measurements diagnose its effect, not its long-run safety.

The old assessment unnecessarily requires the baseline improvement interval
and the incremental interval both to have half-width <=.02. A baseline
interval [-120,-80] already establishes learning relative to -.04; requiring
its half-width <=.02 cannot add evidence of a current plateau. Repair the
assessment in main so the fine precision screen applies to the incremental
comparison. If that comparison is a plateau and the baseline still straddles
the learning threshold, expand validation. Keep the learning threshold,
incremental precision, two-plateau persistence and minimum-update floor.
Focused counterexamples must show that uncertainty about baseline learning,
deterioration and insufficient updates still prevent nomination. The saved
numerical source stays unchanged; this diagnostic is not a source migration.

## Execution, numerical provenance and budgets

The existing widths 16/32, learning rates .0005/.001, betas .5/1, root 0,
batch 32, IAF architecture, Adam settings and cap 10 are inherited development
hypotheses, not reviewed production defaults. Keep them to isolate continuation.
512 and checkpoint cadence 128 come from the existing rung schedule. The
single shadow batch per map is the smallest call-chain check, not a sample
size supporting a distributional conclusion. The factor-two shadow is a local
scale perturbation, not a nominated cap. Reliability rtol 1e-9/atol 1e-10 are
the existing FP64 engineering tolerances; failed parity stops execution.

Fresh all-device GPU selection precedes TensorFlow import in every worker.
Use TensorFlow/TFP FP64, strict eigensolver, XLA, batch-native target evaluation,
verified on-demand memory allocation, and contention checks between chunks.
CPU tests intentionally hide GPU devices. No pfor, scalar target fallback,
new numerical safeguards or numerical backend changes are permitted.

Artifact root:
`docs/plans/artifacts/ssl-lstm-q20-training-continuation-2026-09-18/`.
Use the existing `Campaign` supervisor for actual wall-time debits and process
tree deadlines. One bounded diagnostic driver supplies two stages: clipping
checks, then the 512-update continuation only after checks pass. Preserve
per-candidate checkpoints, original parent hashes, manifests, cache and errors.
The driver and the separate repaired assessment are hashed as additional
diagnostic sources; the frozen numerical source is checked again at exit.

Prelaunch allocation: at most 600 seconds for focused tests and a 120-second
setup/accounting envelope (engineering caps, not runtime estimates). Numerical
caps are computed from the measured batch/width/beta prices and recorded before
launch, with the inherited factor-two engineering reserve. The clipping stage
has at most 1200 seconds. The continuation has at most 25200 seconds, including
all eight maps, compilation, validation and teardown. All actual numerical
seconds in this diagnostic extension debit both balances. Thus the phase
cannot consume more than 27120 seconds including prelaunch allocations, leaving
at least 87287 campaign seconds and 11528 diagnostic seconds from its starting
allowance. A retry shares these phase caps; it does not receive a fresh budget.
Actual admission uses the lower measured quote where available.

Expected work beyond the checkpoint is 8*(512-128)=3072 optimizer updates and
8*768=6144 new validation rows. The measured update costs are about 3.4 seconds
per batch, so raw continuation is roughly three hours plus initialization and
validation. This estimate is descriptive and will be replaced by the explicit
quote; hardware placement and contention can change it. The full cohort still
needs pricing credit and downstream reservations before launch.

Exact commands and measured caps will be written to the request, manifest and
continuation record before launch. A durable systemd user service may supervise
the coordinator; it must have its own total deadline and retain worker logs.

## Skeptical audit and pre-mortem

Audit passed after correcting these proposed-plan flaws: interpreting clip
frequency as Adam step shrinkage; using uncertainty in old baseline improvement
to reject a precise current plateau; restarting current main against an old
scope; repricing a completed rung as fresh work; and calling one seed a full
cohort. Gradient/actual-step parity tests the executable path rather than just
an algebraic surrogate. Fresh GPU checks prevent stale capacity assumptions.
Learning can continue while the posterior remains multimodal or poorly covered;
loss and map algebra therefore cannot establish whitening. A one-step shadow
can miss rare gradients and long-run optimizer effects; cap promotion is
explicitly excluded. Failure of the current map triggers the stated repair,
not rejection of the scientific direction. Literature reinspection is not
needed for this unchanged reverse-KL target; the existing protocol's source
audit remains in force.

## Prelaunch checks and exact execution

The assessment/cache/restart checks passed (20 tests, 24.172036 seconds).
The diagnostic VJP/Adam equivalence, exact unchanged continuation and failed
probe stage-boundary checks passed (3 tests, 8.333857 seconds). GPU devices were
intentionally hidden. Syntax and whitespace checks passed. MathDevMCP/SymPy
proved the stated scalar scaling identity; the explicit positive scale and
epsilon ensure its denominators are nonzero. This does not prove clipping
harmless with changing scales.

The measured quote credits 1024 completed updates, eight first updates performed
by the clipping diagnostic, and 12288 cached validation rows. It forecasts
3064 further updates and 6144 new validation rows: 11327.283137 seconds raw,
22775 seconds reserved including the declared 120-second startup/teardown
reserve. Clipping has a separate 1200-second cap. Prelaunch debit is 152.505893
seconds, leaving 114255.087971 campaign and 38495.520190 diagnostic seconds.
The phase's remaining maximum debit is 23975 seconds; it cannot authorize the
full cohort or spend a predecessor allowance again.

Launch the prepared driver through the existing Campaign supervisor:

```text
systemd-run --user --unit=bayesfilter-q20-training-20260918-r1 --property=WorkingDirectory=/tmp/BayesFilter-q20-all-gpu-20260917 --property=RuntimeMaxSec=24035 --property=TimeoutStopSec=10 --property=KillMode=control-group --setenv=TF_FORCE_GPU_ALLOW_GROWTH=true --setenv=BAYESFILTER_PRELOAD_CUSTOM_OP=0 --setenv=TF_NUM_INTRAOP_THREADS=2 --setenv=TF_NUM_INTEROP_THREADS=2 --setenv=PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python /home/ubuntu/python/BayesFilter/docs/benchmarks/diagnose_q20_training_continuation_2026_09_18.py run --request /home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-training-continuation-2026-09-18/request.json --output-dir /home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-training-continuation-2026-09-18/campaign
```

The 24035-second service cap is the two measured/declared stage caps plus a
60-second coordinator envelope; each worker has a tighter deadline. The source
inventory still matches all 476 saved entries. The fixed-source numerical
workers use the historical assessment with `calibration_only=true`, which
forces nonpromotion; the tested decision repair in main is retained for a
later explicit source migration. No old evidence is relabeled as having used
the repaired assessment.

## GPU diagnostic result and active continuation

The clipping stage passed all eight restored maps on host GPU 1, selected from
the complete three-device inventory. TensorFlow 2.20.0/TFP 0.25.0, FP64, strict
eigensolver, TF32 enabled, XLA and verified memory growth were recorded. Every
gradient, shadow-Adam and actual-trainer graph was XLA with one trace and no
Python callbacks. All contention observations were empty. The actual trainer
and the cap-10 shadow had exactly equal observed parameter outputs; the loss,
raw norm and full optimizer-state checks also passed. All original checkpoint
and source checks passed. The stage charged 255.568375 seconds (the campaign
ledger owns the exact measured value).

The consistently doubled gradient/history arm changed the next update by
0.0019%–0.0093% relative norm, as expected when epsilon is small but nonzero.
Abruptly supplying the raw gradient with the existing clipped-gradient Adam
moments changed it by 15.6%–39.7%. These eight single-batch comparisons are
descriptive; they do not simulate an unclipped training history, establish
long-run non-harm of cap 10, or nominate a new cap. The numerical gradient and
optimizer call chain is working on the checked states. The remaining question
is sustained learning and eventual downstream usefulness.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue unchanged trained states | All eight matched gradient/Adam checks pass | No source, finite, graph, resource or memory-policy veto | Long-run cap behavior, coverage and learning | Automatic continuation to 512 with fresh cached-bank comparisons | Optimal clipping, whitening, convergence or production readiness |
| Keep the full campaign unadmitted | Updated remaining-cost quote still excludes downstream work | Funding/price gap remains | Adaptive training and downstream costs | Preserve partial training; complete repricing before broader launch | The full cohort fits the current reserve |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Checked states passed all declared mechanics and execution checks |
| Statistically supported ranking | None; no candidate or cap ranking was attempted |
| Descriptive differences | Gradient norms, one-step update changes and runtime only |
| Default readiness | Not established; cap safety and posterior behavior remain open |
| Next evidence needed | Further assessed training, independent roots and downstream posterior/reference checks |

The next worker started automatically from the new 129-update states under
the same numerical sources. It reselects an available GPU and verifies memory
growth before import/initialization. Its 22775-second maximum includes
cooperative checkpointing and termination. Live state and exact command are in
`artifacts/ssl-lstm-q20-training-continuation-2026-09-18/continuation.json`;
the campaign ledger has current charged time and the outstanding attempt cap.

Repricing with actual completed-work credit gives 68279.081 seconds raw
(136558.162 seconds reserved) for the remaining full cohort's 512 floor from
the original 128-update checkpoint, even assuming the first validation bank
resolves every decision. Conditional on this continuation completing, the
remaining floor is 56924.534 seconds raw (113849.068 reserved). Both exclude
all posterior/reference/tuning work. The diagnostic therefore does not make
the complete campaign affordable under the existing factor-two reserve.

Post-run red-team note for the clipping stage: the strongest alternative
explanation is that rare gradients or changing directions make clipping harmful
over many updates despite correct one-step mechanics. A matched longer branch
comparison could overturn the practical conclusion; the current evidence only
rejects the simple inference that clipping all gradients forces proportionally
tiny Adam steps. The weakest evidence is one diagnostic batch per map. Training
remains in progress, so no terminal learning conclusion is available yet.

## Terminal training review

The continuation finished successfully in 11488.353474 supervised seconds
(3 hours 11 minutes 28 seconds); clipping plus continuation charged
11743.921849 seconds. All 3072 new optimizer updates were accepted with finite
targets and optimizer states. Each of the eight maps now has 512 updates and
passed forward/inverse/log-determinant/pullback checks. The terminal review
verified both stage result/checkpoint checksums, all eight complete optimizer
states, preservation of the first 128-update histories, all 476 numerical
source hashes, and 24 cached validation prefixes. Memory growth remained
verified; recorded contention checks found no other numerical process.

Each paired 768-row validation comparison shows further learning. These are
changes from update 128 to update 512, where a negative loss change indicates
improvement. The intervals below are descriptive normal intervals on an
adaptively reused bank, not confidence sequences or cross-candidate rankings.

| Schedule | Width | Learning rate | Beta | Mean loss change | Interval half-width |
| --- | ---: | ---: | ---: | ---: | ---: |
| Continuation | 16 | .0005 | .5 | -72.7762 | 7.1064 |
| Continuation | 16 | .001 | .5 | -16.7054 | 1.2493 |
| Continuation | 32 | .0005 | .5 | -41.2720 | 3.1920 |
| Continuation | 32 | .001 | .5 | -2.5859 | .2840 |
| Direct | 16 | .0005 | 1 | -153.9882 | 13.0836 |
| Direct | 16 | .001 | 1 | -37.1756 | 2.6795 |
| Direct | 32 | .0005 | 1 | -78.2246 | 6.1447 |
| Direct | 32 | .001 | 1 | -9.5782 | .9350 |

All decisions remain `continue_training`; no plateau or deterioration was
observed at this resolution. Clipping affected 3044 of the 3072 new updates
(99.09%), so its long-run safety hypothesis remains open despite continued
learning. The serialized `minimum_updates_met=false` is intentional because
`calibration_only=true` forces nonpromotion; it does not mean the numerical
512-update floor was missed. There is still only one root, and the continuation
maps are still at beta .5. No map is development-eligible or posterior-qualified.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the completed diagnostic continuation | All eight maps reached 512 and exact-state/numerical checks passed | No finite, map, source, device or budget veto | One root, reused validation, incomplete temperature/cohort work | Preserve checkpoints and update the remaining-work plan | Whitening, convergence, candidate superiority or production readiness |
| Keep all eight candidates viable | Within-map intervals indicate further learning | No deterioration or map rejection | Sustained progress, clipping safety and posterior coverage | Continue only through a priced staged plan under remaining funds | Any one configuration is best |
| Do not launch the full campaign from this result | Complete downstream costs are missing; remaining floor reserve already exceeds balance | Funding condition unresolved | Training precision and posterior/reference costs | Reconcile the new checkpoints and price remaining work | That more money is mathematically necessary or thresholds may be relaxed |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | All declared execution, checkpoint and map screens passed |
| Statistically supported ranking | None; all configurations remain viable under current evidence |
| Descriptive-only differences | Paired loss changes, clipping counts, timing and architecture/LR differences |
| Default readiness | Not established; full roots, temperature continuation and downstream validation remain |
| Next evidence needed | Independently rooted assessed training and frozen-map HMC/reference checks under a complete cost plan |

The settled ledger leaves 102511.166122 seconds (28.4753 hours) of campaign
allowance, including 26751.598340 seconds (7.4310 hours) for diagnostics.
There are no outstanding attempts. Terminal inspection and documentation use
the already declared setup/accounting envelope; no new numerical work was run.
The now-realized remaining floor forecast is 56924.534 seconds raw and
113849.068 reserved, before HMC/reference/tuning. It exceeds the current
102511.166-second balance under the existing reservation rule, so it cannot
authorize a full launch. The result supports further staged research; it does
not reject NeuTra or require repeating completed updates.

Post-run red-team: reverse-KL loss can fall while a map misses posterior regions;
only actual downstream posterior/reference evidence can settle that concern.
The eight paths share one root and reuse their validation banks. Fresh roots
and untouched downstream checks could overturn their apparent viability. The
implementation and evidence are intact; the candidate training remains
unfinished rather than scientifically rejected.
