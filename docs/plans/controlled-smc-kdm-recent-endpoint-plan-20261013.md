# Recent KDM checkpoints: native endpoint coverage

13 October 2026. Replication-only continuation under the existing campaign.
This phase evaluates fixed saved models; it performs no training or tuning.

## Intent and evidence contract

Question: does the native GPU/XLA filter evaluate recent MDPF/A-MDPF models
with valid distributions, complete sequences, and the author's metric semantics?
The comparator is pinned author b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.
Use both source-trained and TF-trained checkpoints from calibration units01/02,
plus the completed unit01 author-initialized MDPF intervention: nine models.
Use each unit's existing 256-trajectory, T=150 evaluation data, N=25, batch64,
FP32 with TF32 off, and seeds 202610082/202610083. This is development coverage.

The actual consumer must call SourceInitialCloud.sample and
BearingsNeuralReplay.native_evaluate. Include the initial cloud at t=0 and
all 149 returned times. Save per-state NLL and squared error, per-sequence
metrics, and the existing conditional slices. Check input/checkpoint hashes,
finite outputs, positive bandwidths, normalized nonnegative weights, all rows
and times, GPU placement, checkpoint immutability, and one trace per signature.
Replay the author's unchanged metric classes on the first two native-cloud
trajectories of every model's first seed. Retain the earlier arithmetic screen
2e-4*(1+abs(source value)); record actual errors. It is a localization screen,
not a likelihood-quality acceptance margin.

Passing the execution and metric checks closes this bounded endpoint question.
A mismatch is a repair trigger; source/data drift or corrupt/incomplete evidence
is a continuation veto. An intrinsic author failure is not a port defect.
Native stochastic NLL/RMSE differences and conditional slices are explanatory
only. These runs cannot establish training-population noninferiority, paper-wide
replication, superiority, default readiness, or HMC readiness. Heuristic
competition is outside this fixed-source translation question.

## Default audit and skeptical review

The source's evaluation metric is anchored in
src/evaluations/full_sequence_evaluations/full_sequence_evaluation_base.py:62-120
and src/loss_functions/loss_functions.py. The prior reviewed native-evaluation
plan and executable author replay established the operation; this phase changes
only checkpoint coverage. Source NLL includes its existing 1e-8 floor; RMSE takes
the square root within each trajectory before averaging trajectories. No new
ridge, clipping, damping, architecture, objective or hyperparameter is added.

Checkpoint selection is a frozen baseline with completed training/audit hashes;
stale or mislabelled states would invalidate the comparison, so verify hashes
before GPU import. Reused 256 rows and two seeds are convenience choices, not an
untouched claim set or a source-paper sample-size claim. Independent TF/Torch
streams cannot give pathwise equality; only same-cloud metric replay is judged
as numerical parity. Two-trace shapes or missing t=0 can silently change the
question; trace and shape checks expose them. Source and TF metric replay on the
same clouds checks evaluation arithmetic, not correctness of all filter draws.
Earlier sampler/step/gradient evidence remains necessary and is not rerun.

Skeptical audit: PASS before implementation/launch. Exact source comparator,
promotion scope, independent-stream limits, CPU/GPU roles, stop conditions and
artifact coverage are explicit. The bounded same-state intervention checkpoint
is included despite its evaluation-only recovery manifest; validate its actual
training checkpoint and successful recovery rather than treating the original
evaluation exception as a training failure. No independent-training ladder or
retired population-precision prerequisite is reintroduced.

## Execution and budget

Phase total: at most 900 GPU seconds and 300 CPU seconds, across two fresh
versioned attempts. Charge actual resource wall time exactly once to the
existing campaign. Current available CPU 205580.430427 / GPU 77151.070263
seconds; this phase reserves no new allocation. No concurrent campaign job.
Localized harness repairs may retry within this same phase cap; preserve every
attempt. Stop on the validity/provenance veto or exhausted cap.

Command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_recent_endpoint.py

Author preparation/replay uses tf-gpu2 with CUDA_VISIBLE_DEVICES=-1 before import.
Native evaluation uses escalated RTX5080, tftwogpu, stable GPU/XLA signatures,
verified memory growth, TF32 off and one host thread. Record commands, code/data
hashes, source pin, seeds, environments, hardware, allocator, timing and results.
Output: artifacts/controlled-smc-kdm-replication-20260928/recent-endpoint-20261013/.
Result: controlled-smc-kdm-recent-endpoint-result-20261013.md.

If this passes, evaluation coverage supplies no new repair target. The next
discriminating training experiment would be a separately reviewed, bounded
unit02 initial-weight intervention with exact original-driver replay. Do not
interpret an endpoint pass as explaining independently trained quality gaps.
