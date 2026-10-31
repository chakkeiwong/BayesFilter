# KDM stage-isolation execution note

13 October 2026. The stage-isolation diagnostic is complete. Changing only the
pretrained initial measurement weights removed 95.26% of the saved unit01
MDPF NLL gap. Whole KDM replication remains open.

Attempt01 stopped during the CPU source audit before GPU training. The new
harness passed JSON-roundtripped distribution dimension keys to the author's
particle-cloud constructor. JSON changed integer indices to strings, causing
the author's tensor indexing to fail. This is a harness configuration defect,
not evidence of an algorithm or TensorFlow translation failure.

Repair: load the pinned native YAML training configuration, retaining integer
keys. Compare its JSON-normalized numerical settings against the saved
configuration, allowing only the already documented CPU loader worker count.
This is the same procedure used by the existing fresh-pretraining diagnostic.
No method, optimizer, data, seed, or acceptance rule changed. Source constants
and all imported author Python files are checked before accepting results.

Attempt01 charged 10.001016557 CPU wall seconds and zero GPU seconds. Its
manifest, traceback and source snapshot are preserved under
artifacts/controlled-smc-kdm-replication-20260928/stage-isolation-20261013/attempt01.
The phase retains its original caps and maximum of three attempts.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Repair harness and retry | Stage attribution not yet evaluated | Configuration exception before training | Whether pretrained state explains final gap | Check native config, run replay control, then isolated MDPF training | No KDM completion or population ranking |

| Inference status | Finding |
|---|---|
| Hard veto screen | Attempt01 harness invalid; no numerical candidate evaluated |
| Statistically supported ranking | None |
| Descriptive differences | Pending |
| Default readiness | Not assessed |
| Next evidence needed | Valid source audit and replay control before full training |

Post-run red-team check: the configuration failure says nothing about the
author algorithm. A passing repair smoke establishes only usable native
configuration. Full stage attribution still needs the controlled training and
native checkpoint evaluation specified in the plan.


Attempt02: source audit PASS (484 checks; 28 pinned imported/extracted author
files). The two saved pretraining NLL comparisons on eight shared clouds are
-0.00016066 and +0.00314893 nat (TF minus author). These differences describe
pretraining only and cannot settle the later optimization gap.

The GPU control then correctly rejected stale source-stage verification before
any training update. Current runtime differs from that verification by optional
sampling capture and added diagnostic fields. The new supervisor reruns the
existing verification before the replay control and pins the exact comparator
GPU (RTX 5080); the earlier launcher exposed both devices. Charged attempt02:
40.006865338 CPU wall seconds, 10.003685296 GPU seconds. No candidate failed.

A manual configuration smoke initially omitted the source environment's
LD_LIBRARY_PATH and failed to import matplotlib; the recorded source launch
environment fixes this without a package change. The finite cloud smoke then
passed in 1.721277367 CPU seconds. This is routine harness validation.

Historical next step after attempt02: attempt03, current source verification, two-epoch replay, unchanged
single MDPF intervention, then author-native evaluation. No conclusion of
replication completion, population ranking, or author-code defect is supported.


## Completed training and recovered evaluation

Attempt03 completed 477 MDPF epochs and 37,683 optimizer updates. Training
passed 6,680 checks with zero failures and stopped under the source early-stop
rule. Best epoch was 431 (zero-based). Every one of the four exercised graph
signatures remained at one trace throughout all epochs. Training used the
original RTX 5080, GPU/XLA FP32, TF32 off and verified memory growth.

Before the intervention, current source-stage verification passed 2,204 checks
and the replay control passed 1,863 checks: all 166 batch losses exactly
matched the saved TF comparator. Only the starting measurement-module weights
changed; data, stage index, TF seeds, Adam, rates, loss, TBPTT and stopping
remained fixed. This control supports attributing this trajectory change to
initial weights within the recorded execution environment.

The automatic author evaluation then failed at model initialization:
JSON had converted integer dimension keys to strings, causing
kde_particle_filter.py:592 to raise KeyError: 0. No validation loss had been
computed. The earlier cloud-construction repair had not covered this second
configuration consumer. This was our evaluator defect, not an author algorithm
failure.

The repair loads the pinned native YAML model and architecture configurations,
checks their JSON-normalized equality with the saved configurations, and
checks integer dimension keys before construction. The trained checkpoint was
preserved and training was not restarted. The plan explicitly added up to two
evaluation-only recovery attempts within the original phase caps. The first
recovery launcher invocation stopped before creating a run because it read an
incorrect campaign-status field; changing that field to the existing
running field repaired the preflight.

Command actually run for evaluation-only recovery:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/resume_kdm_stage_isolation_evaluation.py

Evaluation-repair01 completed on 2026-10-13 at 08:49:09 UTC (16:49:09 Hong Kong),
using the author's CPU reference environment with CUDA_VISIBLE_DEVICES=-1.
It charged 80.015312 CPU wall seconds and zero GPU seconds. Native evaluation
passed 260 checks, including configuration, complete validation coverage,
finite losses, native aggregation, checkpoint immutability and source
provenance. All 30 pinned author source files matched. There are 24 full-native
validation draws: eight for each saved checkpoint, each covering 1,000 rows.

| Saved unit01 checkpoint | Mean native NLL (nats) | SD across eight evaluation draws |
|---|---:|---:|
| Author best | 5.904294 | 0.024343 |
| TF trained from author pretrained weights | 5.939196 | 0.015413 |
| Old independently pretrained TF best | 6.641039 | 0.039318 |

The old TF-minus-author gap was 0.736745290 nat. The author-start TF-minus-author
gap was 0.034902066 nat, a 95.2627% reduction in this saved comparison. These are
checkpoint-conditional descriptive values. Evaluation draws repeat evaluation
of fixed checkpoints; they are not independent training replications. No
population ranking or fixed 0.01-nat/1% acceptance claim follows.

The evidence strongly localizes most of this unit01 final-training gap to the
pretrained starting weights. It does not establish why independent pretraining
produced those weights, nor identify an additional algorithm translation bug.
Similar isolated pretraining losses do not establish equivalent initial
conditions for later nonlinear training. A-MDPF and other training units were
not rerun by this intervention.

## Decision and next discriminating test

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Close the planned stage-isolation diagnostic | Unchanged replay passes; initial-weight intervention removes most of the saved unit01 gap | Accepted training and recovered evaluation have no numerical, coverage or provenance veto; original harness failure preserved | Origin of independent pretraining divergence and the remaining conditional gap | Trace both pretraining consumers; compare initialization, shared-cloud forward/gradient/Adam updates, validation and checkpoint selection before another full training run | Whole KDM completion, population noninferiority, author-paper defect or improved algorithm |

| Inference status | Finding |
|---|---|
| Hard veto screen | Original evaluation invalidated by configuration key types; repaired evaluation and completed training pass |
| Statistically supported ranking | None across independent training runs |
| Descriptive-only differences | NLL table and 95.26% gap reduction apply to this saved unit01 counterfactual |
| Default readiness | Not assessed; cSMC bounded fidelity stays accepted; KDM completion remains open |
| Next evidence needed | Explain independent pretraining differences with matched-state/source-call-chain checks and determine whether findings apply to A-MDPF and other units |

The next test should first establish equality of each pretraining consumer's
inputs, parameter initialization, optimizer state and validation-selection
semantics. Use shared clouds and matched parameters to test loss, gradients and
one-step updates. If these agree, trace short coupled trajectories and their
first consequential branch or selection divergence. Do not infer a code defect
from independently different random streams, and do not launch a population
training ladder to answer this source-replication question.

Post-run red-team review: the strongest alternative explanation is ordinary
stochastic and FP32 trajectory sensitivity interacting with the starting
weights, rather than a mistranslated pretraining operation. This single
intervention cannot separate those mechanisms. A matched-state update or
selection mismatch would establish a concrete repair target; agreement would
weaken that explanation. Another unit with a materially different response
would limit generalization of this unit01 result. The author evaluator is an
independent CPU reference and does not establish GPU production readiness.

## Evidence and budget

Evidence root:
artifacts/controlled-smc-kdm-replication-20260928/stage-isolation-20261013/.

The original attempt03 manifest, training diagnostics and failed evaluation
are retained. Evaluation-repair01 contains manifest.json, decision.json,
terminal-review.json, diagnostic/result.json, per-draw rows.json and
summaries.json, input hashes and source snapshots. The run manifests preserve
exact commands, environments, git commit, data/checkpoint hashes, seeds,
hardware, wall time and output paths. The selected trained checkpoint SHA-256
is b39244cc5c5aada6a24f50be1cd9cb615cb48d5352108e0f66f93af590db5468.

Total phase charge is 11250.946277 CPU wall seconds / 11090.917908 GPU seconds.
Within the 16200 CPU / 14400 GPU phase caps, 4949.053723 CPU / 3309.082092 GPU
seconds remain. The active allocation has 205682.324571 CPU / 77248.991988 GPU
seconds remaining under the tighter ledger/tranche limits; no conditional
additional allocation was added. No stage-isolation process remains active.
