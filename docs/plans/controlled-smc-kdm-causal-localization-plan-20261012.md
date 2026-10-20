# KDM: separate pretraining, arithmetic and training-control effects

Updated 13 October 2026. Execution and terminal review of the initial and saved-case phases are complete. No experiment is active. This refines the next
action in the replication master. Existing campaign authorization remains;
the completed root-cause and coupled-trajectory attempt caps stay closed.

## Execution status after terminal review

See [the terminal result](controlled-smc-kdm-causal-localization-result-20261012.md)
and [terminal audit](artifacts/controlled-smc-kdm-replication-20260928/causal-localization-20261012/terminal-audit.json).
Initial attempts 01–06 and saved-case attempts 01–02 are terminal. No additional
launch is implicit in those exhausted attempt caps.

| Planned stage | Status |
|---|---|
|0: matched inputs and controls|Complete for the executed scope|
|1: fresh pretraining|One epoch completed; first feature flag replayed; later gradient at index 67 and unit01 validation remain open|
|2: crossed starts|Complete bounded 16-update comparison|
|3: numerical interventions|Selected CDF boundary causally closed; earlier small-radius and separate pretraining-gradient cases remain open|
|4: validation/controller|Identical-history replay and common 64-row evaluation complete; native-size selection sensitivity remains open|
|5: downstream qualification|Not completed; no new source-deviation repair or statistically justified final design|

The resolved local difference is author-intrinsic numerical sensitivity. Preserve
it without a scientific algorithm change. The harness observation repair is
verified by capture-off/on equality and 2,358 checks in each saved-case launch.
The terminal audit passes 155 checks. The final learned-checkpoint quality gap
remains visible under the common author evaluator.

Total charge 1,829.325689 seconds per resource. Remaining campaign CPU
217,986.948231 seconds; active GPU tranche 89,039.550787 seconds. The follow-up
sequence is in the refreshed master. Later launch narratives below are retained
as the pre-run record and do not imply an active process.

## Question and scope

Why did two independent TensorFlow MDPF training runs have worse final NLL and
RMSE than their author counterparts? Determine whether an incorrect translated
operation, different pretrained starts, numerical amplification, or stochastic
validation decisions contributes. A local discrepancy is not a root cause of
the final quality gap until a controlled intervention links the two.

Comparator: pinned released Younis author implementation at
b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18, using the saved data/configuration and
actual unit01/unit02 constructors and checkpoints. Both unfavorable units
remain in the analysis. This is replication; author-intrinsic failures remain
acceptable. No angle epsilon, optimizer redesign, new objective, or Task 3.

Current evidence: [root-cause result](controlled-smc-kdm-root-cause-trace-result-20261012.md)
and [coupled trajectory](controlled-smc-kdm-coupled-trajectory-result-20261012.md).
Stage transfer is correct in both units. All 569 pretrained parameters differ,
while 627 other parameters match. The previous trajectory used TF-produced
states in both engines. The prior source-stage pretraining fixture used two
updates from an already fitted checkpoint, not the fresh constructor.

## Evidence contract and diagnostic roles

| Role | Requirement |
|---|---|
| Primary localization criterion | Reproduce a discrepancy at the earliest affected operation with identical inputs/state; use one controlled intervention to test the proposed cause. A source deviation needs a source-grounded repair. |
| End-to-end closure criterion | After any repair, demonstrate native source-route execution and assess trained NLL/RMSE under a frozen, affordable uncertainty design. Existing KDM 0.01 nat NLL/1% RMSE criteria are unchanged; they are not local tensor tolerances. |
| Promotion veto | Incorrect source operation, additional TF invalid outcomes on matched author-valid inputs, or unresolved final quality evidence. These do not automatically veto the next repair phase. |
| Continuation veto | Corrupted/mismatched inputs, uncontrolled randomness, missing decisive diagnostics, budget exhaustion, or a changed scientific target. Stop the affected comparison and repair the harness when possible. |
| Repair trigger | A reproducible same-state/same-input discrepancy; unequal controller events for identical loss histories; a native sampler discrepancy not explained by different random mappings. |
| Explanatory only | Parameter distance, local gradient flags, angle radius, ancestor changes, validation variability, short-run NLL/RMSE and source-self perturbation spread. |
| Must not conclude | Local parity proves final quality; different weights imply a bad implementation; shared sensitivity permits unlimited error; FP64 agreement on different draws proves an FP32 cause. |

Heuristic-adversary scope: no method-superiority or general filtering claim is
made. The author is the appropriate replication comparator. Exact repeated
execution, unchanged-input controls, the initial checkpoint, and crossed starts
are diagnostic controls, not weaker replacement baselines. Preserve paired
author-intrinsic failures rather than silently deleting them.

## Ordered tests

### 0. Verify the experiment can distinguish its proposed causes

Use the existing TF pretraining replay entry point
`SourceWeightPretraining.update` and the actual source trainer operation.
Verify exported source computation against the uninstrumented source for the
same captured inputs. Identical repeats must reproduce their own result.
Record the consumer call chain, parameter names, trainable/frozen groups,
moment state, step counters, data order, masks and input hashes.

Matching seed numbers across libraries is insufficient. Generate/capture the
actual arrays once, save them, and assert equality before computation. For a
precision intervention, use the same FP32-representable constructor parameters,
data and base variates, promoted exactly to FP64; keep source-required internal
dtype rules explicit. Do not independently generate FP32 and FP64 noise.
Keep uniform draws strictly inside their required domain. This quantized common
input experiment is diagnostic, not an exact native random-number stream.

### 1. Locate the first pretraining discrepancy from fresh initialization

For each unit's actual common constructor, run 16 sequential weight-pretraining
updates in both engines, in FP32 and FP64. Use the same ordered training rows
and the same actual source-generated particle cloud at each update. Walk
successive batches, rather than replaying the same batch repeatedly. Preserve
source group learning rates, coupled weight decay, trainable groups and fresh
Adam state. Compare every update, not just the terminal parameter vector.

Record selected observations/states, cloud, features/logits, normalized weights,
per-example NLL, reduced loss, raw gradients, gradients including weight decay,
first/second moments and new parameters. Pretraining uses no gradient clip in
the checked source path. Inspect time-major masked-state ordering and reduction
weights, including the partial batch, before treating any gap as arithmetic.

At the first material flag, restore the identical saved source pre-update state
in both engines and repeat one update. This separates an operation mismatch
from accumulated parameter drift. If the fresh tests pass, extend once to one
complete source epoch, including partial batches and validation. Do not keep
extending short runs without a new discriminating question.

This deliberately holds clouds fixed. It does not certify the native TF cloud
generator: reuse the existing sampler evidence and separately verify any newly
instrumented sampling path against its native implementation. Do not claim
that a replay entry point has exercised a bypassed sampler.

Interpretation: a reproduced single-update mismatch identifies a localization
target. Close source agreement under common inputs, combined with divergence
under native inputs, motivates random-stream/controller tests; it does not
prove that random variation explains the final quality difference.

### 2. Cross checkpoint origin with engine

For each unfavorable unit, run the following four arms:

| Full-MDPF starting state | Author engine | TF engine |
|---|---|---|
| Actual author-pretrained checkpoint | A | B |
| Actual TF-pretrained checkpoint | C | D |

Compare A versus B and C versus D first. Both use identical full starting
parameters within the pair, zero Adam moments at the genuine stage boundary,
source rates, rows, masks and common base variates. Each engine samples from
its own evolving state. Never inject the other engine's sampled particles into
the evolving comparison. Test FP32 and FP64 with the same quantized base arrays.
Run 16 updates initially; record the initial functional outputs and the full
update diagnostics. A state intervention must never carry optimizer moments
from an unrelated checkpoint.

If engine differences persist for both starts, localize the operation. If
differences depend on the start, test numerical conditioning there. If engine
pairs agree while A/C and B/D differ similarly, initialization has a local
effect shared by both engines. This does not yet establish which start is
better or explain final converged quality. Evaluate functional losses and
particle predictions on a common development panel; parameter distance alone
does not answer that question.

### 3. Intervene on the first differing numerical operation

Run this only on the first flagged saved case, using a shared pre-update state.
Capture raw angle pairs/radii, layer activations, wrapped angle errors, weights,
CDFs, actual uniforms, distance to the adjacent CDF boundaries, ancestor
identities, rejection-sampling accept decisions and primitive counters. The
first gradient discrepancy precedes the previously observed ancestor switch;
start there instead of attributing every discrepancy to resampling.

Use matched FP32/FP64 inputs to distinguish arithmetic sensitivity from an
algebraic mismatch. If needed, evaluate a single saved case in TF GPU graph
mode without XLA, explicitly diagnostic only, to separate compiler arithmetic
from a source-operation difference. Do not switch the runtime default.

Keep two branches distinct: forcing the recorded ancestor for a one-step
counterfactual asks whether that switch explains the discrepancy; ordinary
own-state resampling remains the fidelity comparison. A forced branch cannot
qualify the filter. The common inverse-CDF diagnostic also cannot certify
native `stateless_categorical` mapping or reproduce a particular native failure.

For a source-self sensitivity panel, freeze in advance the saved cases,
coordinates, signs and amplitudes 0, 1e-8, 1e-7, 1e-6 and 1e-5. Include ordinary
and small-radius cases, and a separately labeled boundary case. Record the
actual representable change and the first altered activation; retain every
zero-effect result. Do not choose only perturbations that enlarge the envelope.
These probes explain conditioning; they do not define a final-error allowance.

Gradient localization must preserve IWSG's intended stop-gradient semantics.
For `exp(logq-stop_gradient(logq))`, recomputing both terms in an ordinary
finite difference gives a constant forward factor, not the estimator's intended
gradient. Compare the source/TF backward rules and saved-state adjoints, or
freeze detached terms explicitly for a local derivative check. Do not diagnose
a missing IWSG derivative from that invalid finite-difference comparison.

### 4. Separate validation noise from controller implementation

Replay each saved unit's validation-loss history through the actual author
scheduler/early-stopping implementation and the TF controller. Feed identical
scalars and initial controller state; compare rates, save-best decisions and
stopping epochs exactly, including equality and negative-loss cases. Extend
the existing controller tests to these histories rather than repeating only
generic fixtures. Missing history is an evidence gap, not permission to invent it.

Then evaluate the same available checkpoints on a fixed common development
panel and the same captured evaluation draws. Repeat a predeclared set of
eight draws to describe within-checkpoint evaluation variation. This estimates
evaluation randomness only, not training-seed uncertainty. Compare validation
selection margins with that variation. Intermediate checkpoints must actually
exist; if they do not, collect them in the next bounded diagnostic.

If selection is sensitive, compare independent versus shared validation draws
in a bounded diagnostic with training inputs held fixed. A fixed validation
panel or frozen source LR schedule is a causal intervention, not a change to
the author's native training procedure. Native confirmation must restore that
procedure. Differences already present with the schedule fixed require a
different explanation.

### 5. Link the diagnosed mechanism to final quality, then qualify

An identified translation defect gets the smallest source-grounded repair and
a saved-case regression. First show that the proposed intervention removes the
localized discrepancy without changing healthy matched outputs. Then use a
fresh unit and native source stopping/sampling to check downstream relevance.
If no implementation defect is found, retain that finding and plan an explicit
study of stochastic training variation; do not label the author paper faulty.

Before a new expensive full-training ladder, freeze an uncertainty/precision
design using training units as independent units. Repeated evaluation draws
and test sequences are nested observations, not extra training seeds. Assess
whether common random inputs actually reduce variance; matching seed labels
does not create an informative pair. Preserve source stopping, all selected
regimes and the native evaluation endpoint. Use an independent development
source-self sample to plan precision, not to accept the same inspected results.

For existing KDM limits, the eventual paired differences/ratios and their
predeclared uncertainty must support the requested no-worse statement. Do not
treat a nonsignificant difference, overlapping intervals, or a broad source-self
envelope as equivalence. If the required design exceeds the remaining budget,
report the bounded implementation result and unresolved statistical question.
Changing the qualification criteria is a separate substantive decision; it is
not an automatic consequence of a noisy result. Do not reinstate cSMC's retired
1%/5% criteria. Controlled-SMC's current bounded verdict is unchanged.

## Default audit, execution bounds and skeptical review

The initial localization phase is capped at four CPU wall-hours and four GPU
wall-hours, charged conservatively to both resources per supervisor wall time,
at most six launches including failed launches, at most one hour per launch.
These are caps, not runtime forecasts or a fresh compute grant. Last reconciled
balance: CPU 61.06 h and GPU tranche 25.24 h. Reconcile immediately before any
launch; deduct other campaign work. Full-training confirmation is excluded
from this diagnostic cap and requires a costed specification within the existing
remaining allocation. Do not reapprove unchanged authorized local retries.

| Material choice | Provenance/status | Risk and earliest check |
|---|---|---|
| Unit01/unit02 starts | Existing unfavorable development cases | Selected cases cannot estimate population quality; keep both and reserve fresh units for confirmation. |
| 16-update initial horizon | Localization convenience, extends prior trace | Long-run divergence can be missed; inspect every update and allow one full pretraining epoch if short parity passes. |
| FP32 source-mode settings; FP64 reference | Existing replication route | Precision changes also changing inputs would confound interpretation; assert exact common quantized inputs and record internal dtypes. |
| Source rates, weight decay, masks and group state | Pinned source baseline | Constructor/configuration mismatch can mimic arithmetic; assert actual groups and stage boundary before the first update. |
| Prior tensor flag scales, 2e-5 FP32 / 2e-9 FP64 against 1+abs(reference) | Inherited diagnostic conventions only | Not calibrated quality limits; save absolute/relative errors and local conditioning, investigate flags without treating them as final replication rejection. |
| Eight evaluation draws | Bounded descriptive diagnostic | Cannot establish training-seed variability or rank methods; report by checkpoint, retain zero differences. |
| Source angle normalization without added epsilon | Source replication baseline | Small-radius sensitivity is measured; expose radii and invalids, preserve the author object. Any numerics-changing safety study is separate from this replication target. |

Environment: author PyTorch CPU in `tf-gpu2`, with GPU intentionally hidden;
TF/TFP GPU/XLA in `tftwogpu`, TF32 disabled for this existing reference contract,
memory growth verified before device initialization, stable input signatures,
no pfor. Reference PyTorch remains an independent-author role. Reuse existing
compiled TF kernels; the host harness only orchestrates and records diagnostics.

Artifact root reserved by name only:
`docs/plans/artifacts/controlled-smc-kdm-replication-20260928/causal-localization-20261012/attemptNN/`.
Use fresh directories, source/input hashes, exact command, git/dirty provenance,
environment, seed and base-input identities, wall charges, per-step records and
first-discrepancy snapshots. Record failed launches and terminal decisions.
The drivers and literal commands are recorded below. Attempt05 is complete;
attempt06 is the final running launch of this initial diagnostic allocation.

Skeptical audit: the old proposal missed the constructor/pretraining boundary,
compared different draws across dtypes, and could mistake a diagnostic random
mapping or a short gradient flag for a native quality failure. This design
repairs those comparison weaknesses and distinguishes state, arithmetic and
controller effects. No proxy metric becomes qualification, no source-intrinsic
failure is erased, and no expensive ladder starts without a precision design.
Design passes this self-review. Execution readiness still depends on wiring and
checking the new harness; no independent reviewer or experiment has been used
to certify this plan. Strongest remaining risk: short common-input agreement
may fail to explain a long stochastic training outcome.


## Execution audit and fixed diagnostic choices

The actual fresh pretraining consumer is author
WeightModelTrainer.do_forward_pass (lines 107–167), with source optimizer groups,
no clipping and 0.001 coupled decay. The TF call chain is
SourceWeightPretraining.update -> _loss -> _apply_update. Retain already computed
features/logits/per-example NLL for observability without changing the objective.
The earlier fitted-checkpoint fixture does not answer the constructor question.

Use both unfavorable units, 16 successive batches initially, then extend an
agreeing pair once to all 5,000 training and 1,000 validation rows, including
8-row training and 232-row validation remainders. A source permutation uses
202610123 plus unit index. FP32 source clouds/data/constructors are exactly
promoted for FP64; their hashes must agree. This fixes precision confounding
but is not a native training-stream experiment. Replay the first flag from
identical source pre-update parameters and Adam moments before attributing it.

Source instrumentation must reproduce the uninstrumented consumer loss,
gradient and updated state. TF repeats after state restoration must agree.
Validation cannot mutate parameters/moments. Graphs use stable signatures for
four batch sizes and report tracing counts. Invalid controls stop the affected
comparison. Numerical flags trigger localization, not qualification changes.

Skeptical execution audit passes with these amendments: comparator and source
defaults are explicit, proxy metrics cannot close final quality, partial batches
and source controllers are included, inputs match across precision, and the
four-hour/resource, six-launch, one-hour/launch caps are enforced. Remaining
risk: deterministic local agreement cannot explain long stochastic outcomes.
No new full training ladder follows automatically from a local pass.

Literal launch (escalated GPU access, existing approved interpreter):
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B /home/chakwong/BayesFilter/docs/benchmarks/run_kdm_causal_localization.py --stage pretraining
```
The supervisor reconciles the campaign ledger, snapshots the driver and numerical
dependencies, enforces a one-hour timeout and deducts elapsed time from both
resources on success or failure. Remaining phases use the same supervisor with
their recorded stage name after wiring checks.

Execution repair audit (12 October): attempts 01–04 failed before any training update. Source numerical training parameters agree exactly after excluding the saved one-worker data-loader override. Read cloud dimension keys from verified YAML. Correct the diagnostic Adam-state collector to aggregate each parameter's own optimizer state (the original nested collector overwrote earlier groups with zeros). This changes observation only. FP64 extension is limited to the actual matched FP32 input scope. Preserve all failed manifests; two launches remain under the unchanged six-launch cap.

### Final bounded launch specification and audit

Attempt06 bundles independent saved-history controller replay (CPU reference,
run first) with crossed initial-checkpoint training and common development
evaluation (GPU/XLA TF, CPU author). This remains within the six-launch/one-hour
per-launch/four-hour resource caps. The four start roles each receive sixteen
updates in both precisions; source and TF moments begin at zero. The available
author/TF final-best checkpoints are evaluation-only roles, never mislabeled
as intermediate epochs. All eight checkpoint roles receive eight development
draws on the same first64 validation rows; NLL, posterior XY and XY error at
times 4/8/12/16 are descriptive only. The panel is selected before the run and
does not tune anything.

All normal and categorical base arrays are generated as FP32 and promoted for
FP64; von Mises rejection uniforms remain FP64 in both precisions, preserving
the author's precision choice. Hash the actual arrays after exact FP64
promotion and reject a mismatch. Common inverse-CDF categorical mapping is an
explicit diagnostic intervention. It cannot certify native RNG equivalence.
Each engine uses its own evolving cloud; sampler, shift-response, initialization,
source-repeat and first-flag same-state controls remain mandatory.

Audit: baseline is the pinned author callable and actual saved stage starts;
no heuristic or parameter-distance metric becomes a quality criterion.
Missing/invalid state, failed adapter controls, nonfinite outputs or timeout
stop this launch. Arithmetic flags preserve evidence and trigger same-state
replay. Stable signatures must each trace once. No model setting, optimizer
policy, learning-rate schedule, quality criterion or author failure is repaired
by this diagnostic. Sixteen updates and eight evaluation draws cannot certify
final trained quality. Same-state replay and CDF diagnostics may identify the
next operation probe; the larger sensitivity panel remains conditional on a
usable flagged case and is not claimed completed by saving one.


### Conditional operation probe after the crossed-start result

Attempts 01–06 are terminal. The initial six-launch cap remains closed.
Attempt06 exposed a materially different same-state FP32 trajectory at
unit01-tf/update2, whereas matched FP64 and both author-produced starts agree.
A one-step causal intervention is now warranted by ordered test3. This
explicit amendment allocates at most TWO saved-case launches, each at most
20 minutes, within the ORIGINAL four-hour/resource total and existing campaign
balance. It does not create additional hours or reopen either prior supervisor.
Initial phase charged 1,529.189673 seconds to each resource.

Terminal review found a diagnostic defect: ancestors and CDFs were recomputed
after the TF step. A separate XLA call can round a boundary differently from
the enclosing kernel. Therefore these reconstructed counts/margins are
provisional; losses, gradients, updates and actual sampled particles remain
valid observations. Preserve the evidence and fix observation before attributing
a particular ancestor switch.

Question: does the first categorical boundary explain this saved FP32
same-state divergence? Comparator: the actual author update at exactly the
saved pre-update state, data and common base variates. Retain an uninstrumented
TF replay, add optional direct in-kernel capture of the consumed ancestors/CDF/
uniforms, and compare capture-on/off before interpreting the trace. This is
pure observation; source/runtime numerical policy is unchanged.

Predeclared interventions: (a) natural own-state sampling, (b) replace only
the first actual disagreeing ancestor at its first time with the source choice,
(c) force all source ancestors as an explicitly diagnostic counterfactual.
Preserve all other operations. Select the first event by time/row/slot order;
do not search for the largest improvement. Report how much loss, gradient and
updated-parameter disagreement each removes. Source-self perturbations at that
boundary use signs +/- and amplitudes 0,1e-8,1e-7,1e-6,1e-5; retain every
rounded-to-zero result. This boundary-selected panel is explanatory only, not
a representative sensitivity envelope or acceptable-error rule.

Primary local criterion: unchanged controls reproduce saved source/TF results,
actual choices explain the first sampled-particle jump, and the specified
counterfactual tests that explanation. No final-quality or statistical ranking
criterion is added. Promotion veto: source-operation defect remains unresolved.
Continuation veto: baseline replay mismatch, source mutation, corrupted state,
missing consumed-choice diagnostics or compute cap exhaustion. Capturing values
may change XLA arithmetic; if so, classify this explicitly and restrict the
claim to the instrumented route. Do not hide or 'repair' the source boundary.

Assumption audit: inverse-CDF coupling is the existing diagnostic intervention,
not the native RNG route; the flagged case was selected for localization,
not population inference. Force-ancestor branches change the finite sampled
trajectory and cannot qualify replication. FP32 rounding of a requested
perturbation must be recorded. Python loops orchestrate independent cases;
the full filter/update stays a stable-signature TensorFlow GPU/XLA graph.
Count each helper's traces directly; a global TensorFlow warning is insufficient.
No NumPy numerical path or pfor is introduced.

Skeptical review: this probe answers the newly isolated boundary question.
The earlier logging violated the actual-call-chain evidence requirement and
must be corrected before attribution. Adding observations is a routine repair;
forcing ancestors and perturbing source randomness are labeled interventions,
not silent runtime changes. The narrow budget and unchanged scientific target
justify continuing under the user's existing execution instruction.
Artifact root: causal-localization-20261012/saved-case/attemptNN/.
Command (implemented and syntax-checked before launch):
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B
/home/chakwong/BayesFilter/docs/benchmarks/run_kdm_saved_case_intervention.py

Saved-case implementation review: source snapshot and quantized-base hashes are checked against attempt06; capture-off must reproduce the stored complete result exactly. Capture-on changes are explicitly reported. The source replay is unchanged and exact-repeat checked before signed perturbations. Counterfactuals select the first observed event, with no optimization or tolerance change. Syntax/scope checks passed. Ready for the bounded conditional launch.

### Terminal observation repair and second saved-case launch

Saved-case attempt01 completed: 2,358 checks pass, and capture-off reproduces
the saved full result exactly. The actual consumed first boundary is time4,
row52, slot1. Forcing its source ancestor reduces the gradient discrepancy
from0.03083752 to2.3842e-7 and the loss discrepancy to zero. Signed source-self
perturbations reproduce the natural TF discrepancy. This local mechanism is
shared sensitivity, not evidence of a defective translated gradient or a
final-quality acceptance envelope.

The terminal audit identified a real harness defect: attempt06 recomputed TF
ancestors/CDFs outside the consuming XLA graph. Its reconstructed boundary
diagnostics are provisional; original losses/gradients/parameters remain valid.
Repair the crossed-start consumer to retain its sampler's actual ancestor,
uniform and CDF. Validate the very same sampler factory through the saved-case
probe; capture-off must still reproduce the old complete result, and capture-on
must preserve numerical outputs. The second (last) conditional launch uses the
unchanged saved case and signed panel, within the existing20-minute cap.
No runtime algorithm, scientific comparator or tolerance changes.

Skeptical audit: the direct sampler call chain is exercised, with preserved
historical capture-off control; output equality is required before interpreting
new diagnostics. Do not repeat the full64-update ladder for a reporting fix.
The prior source-self experiment does not qualify trained quality or settle
the earlier small-radius gradient case. No additional campaign time is granted.
Before this launch, this phase has charged1,644.715917 seconds per resource;
campaign remaining CPU218,171.558004 and GPUtranche89,224.160560 seconds.
