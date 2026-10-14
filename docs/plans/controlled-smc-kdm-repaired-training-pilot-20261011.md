# Repaired KDM complete-training development pilot

11 October 2026. This continues the authorized replication campaign after the
mechanism and native-training checks. It changes no author method or acceptance
threshold. Run only after the current-runtime source-stage verification and
two-complete-epoch smoke pass.

## Question and evidence contract

Does the repaired TensorFlow optimizer complete the author's three-stage
bearings training protocol, and what downstream behavior does that produce?
The comparator is the completed pinned-author run
kdm-complete-training-20261007-03, not a different architecture or enhanced
filter. Its exported initial parameters and training/validation data are exactly
identical by checksum to kdm-fresh-training-cost-20261007-01. Reuse that source
run; retraining the same source unit adds no necessary evidence.

Run TF from those fresh constructor parameters through weight pretraining,
MDPF, and A-MDPF. Each stage loads its own predecessor's best validation
checkpoint. Preserve author stage caps 400/800/800, early stopping, scheduler,
batch sizes/partial batches, N=25, sparse labels, clipping, learning rates,
initialization and native sampling. Each backend follows its own randomness;
a matching integer seed does not imply common Torch/TF random draws. Never feed
realized author particles into an evolving TF training distribution.

Execution pass requires all three stages to end by the source controller or
source epoch cap; complete per-epoch data coverage; finite committed updates;
immutable validation; correct best-checkpoint transfers; current-runtime
verification; source/input integrity; and one graph trace per fixed batch shape.
Failure is a repair trigger unless the artifact is invalid or the total budget
is exhausted. Author-intrinsic failures are retained baseline outcomes.

After complete TF training, execute the existing common-author evaluator on
both source-trained and repaired-TF-trained checkpoints. The checkpoint
conversion must be exactly reversible; source files, model constants, metric
definitions, data and checkpoints are checked. Evaluate 256 development
trajectories, T=150, N=25 and two filter seeds, including the existing time,
large bearing residual and angle-seam subsets. Preserve every trajectory's
NLL/squared-error values. This measures learned parameter quality in a common
evaluator. It does not certify the native TF evaluation endpoint.

NLL/RMSE differences are descriptive for this one training pair. The unchanged
KDM acceptance margins are 0.01 nat and 1% RMSE, but a one-pair development
comparison cannot establish trained-unit no-worse performance or be recycled
as untouched qualification. Missing finite outputs, invalid checkpoints or
wrong metric/sample semantics veto interpretation. Training loss, runtime and
gradient proximity are explanatory only. No superiority, exact random-bit
identity, paper-wide replication, HMC or production claim is permitted.
Replication compares against the author; a new heuristic competition would
change this phase's question and is not a promotion requirement here.

## Assumptions, cost and stop conditions

The existing source unit is a baseline because inputs match, not because its
result is favorable. The native smoke and source-stage verification establish
mechanics only. The previous TF unit was pre-repair and is historical diagnostic
context, never current qualification. The TF scalar-precision repair is frozen;
no hyperparameter selection occurs on this development result.

Allocate at most 28,800 GPU wall seconds (8 h) and 1,800 CPU wall seconds
(30 min) from the existing campaign, at most two attempts. Prior TF complete
training took about 3.3 h; the cap allows variation under the same source
stopping law. A timeout remains incomplete, never a source-early-stop result.
The supervisor records a 30-second heartbeat and stops after 30 minutes without
progress, rejects concurrent phase runs, preserves each output directory,
accounts failed attempts, and hashes current code and inputs. Ordinary
environment/library failures may be repaired and retried within this allocation.

TensorFlow uses the existing tftwogpu environment, escalated RTX 5080 access,
verified memory growth, FP32/TF32 off and XLA. Common-author evaluation runs
isolated CPU-only in tf-gpu2 with the known conda C++ library path.
No environment mutation is needed.

## Review and execution

Skeptical audit: PASS conditional on the recorded stage-refresh pass and
unchanged runtime hashes. The baseline, exact input pairing, native sampling,
source stopping law, independent streams, historical-evidence scope, finite and
integrity vetoes, budget cap and downstream evaluator limitation are explicit.
This run answers whether the actual repaired full-training consumer executes
and produces a viable learned checkpoint; it cannot settle multi-seed acceptance.

Output root:
artifacts/controlled-smc-kdm-replication-20260928/repaired-training-20261011/.
Execute with the same approved dispatcher:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_replication_gap_localization.py --phase training-pilot --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/repaired-training-20261011/training-pilot-01

Result: controlled-smc-kdm-mechanism-result-20261011.md, linked to terminal
training/evaluation JSON and logs. Next, inspect the complete-run result,
check the native TF evaluation endpoint, and freeze a fresh paired multi-unit
qualification with uncertainty if these diagnostics support proceeding.
Controlled SMC's withdrawn percentage campaign remains retired.

## Prelaunch cost-readiness amendment (11 October, before repaired full run)

The current-runtime stage refresh passed 2,204/2,204 verification checks and
87/87 smoke checks, but its second full-data MDPF and A-MDPF epochs took
194 and 98 seconds. Historical TF median epochs were about 15 and 19 seconds.
That diagnostic omitted TF_NUM_INTEROP_THREADS=1 and
TF_NUM_INTRAOP_THREADS=1, which the old complete run set. Thread
oversubscription is a hypothesis, not a proven cause; workload contention
and other runtime differences remain possible.

Before full training, repeat the two-epoch full-data smoke with both thread
limits restored, same GPU, source fixtures, proof, XLA/TF32 settings and
memory growth. Charge at most 900 GPU seconds within the pilot's 28,800-second
cap, one fresh versioned output. Require all stage checks and project full-run
cost from the second-epoch times times the pinned author's completed epoch
counts (107/454/230). Launch a full run only if the projection is at most 80%
of the pilot GPU time then remaining. This is an operational feasibility
screen, not evidence of numerical accuracy or a guarantee of early stopping.
A cost failure stops the long launch for diagnosis; it is not a replication
verdict. Independent source and TF randomness remain unchanged.

Skeptical review: PASS for this bounded test. The old 3.3-hour TF estimate is
not a safe budget basis in view of the measured slowdown. This amendment checked that hidden assumption before committing to training.
The completed repeat used training-pilot-01; its manifest records 87/87
checks, 160.460316 GPU seconds, a projected 8,123.672899 GPU seconds, and a
READY_FOR_FULL decision. The source baseline, numerical criteria, data and
author method are unchanged.

## Reviewed execution after readiness

The next authorized command is the bounded full-training supervisor. It will
run the repaired TensorFlow complete training, independently audit its
terminal manifest, and evaluate the repaired and pinned author checkpoints on
the same development data. It has a fresh output directory and consumes the
remaining pilot budget only:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_replication_gap_localization.py --phase training-full --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/repaired-training-20261011/training-pilot-02

This is a development pair. A successful run cannot by itself establish
population no-worse behavior or close the KDM replication; those require the
terminal checks and, if still justified, a separately frozen multi-unit
comparison.
