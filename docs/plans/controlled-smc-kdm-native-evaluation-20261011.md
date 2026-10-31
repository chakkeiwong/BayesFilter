# Native TensorFlow KDM evaluation diagnostic

11 October 2026. Follow-up to the repaired complete-training pilot. This is
replication-only development work under the existing campaign authorization.
It starts only after training-pilot-02, its independent audit and common-author
evaluation finish successfully. No live training source or frozen plan changes.

## Research intent and evidence contract

Question: does the actual GPU/XLA TensorFlow filter execute the author's full
T=150 bearings evaluation semantics on the completed source-trained and
repaired-TF-trained MDPF/A-MDPF checkpoints? Baseline: pinned author commit
b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18, the exact checkpoints and 256
trajectories from training-pilot-02/evaluation/diagnostic. Use N=25, batch64,
TF32 off, FP32, native independent TF random streams, two recorded seeds.

The endpoint must call BearingsNeuralReplay.native_evaluate and
SourceInitialCloud.sample, sharing the existing kernels. Retain the initial
cloud as t=0; native_evaluate returns t=1 through149. Use all150 time points.
Source NLL is -log(sum(weight*component_density)+1e-8). Source RMSE averages
squared x/y/wrapped-angle error within each trajectory, takes its square root,
then averages trajectories. Particle angle means use weighted sine/cosine.
Never substitute the training sparse-label loss for this evaluation metric.

Execution pass: completed training/audit/provenance, finite valid distributions,
all256 rows and150 times, unchanged checkpoint state, one graph trace per fixed
shape, GPU/XLA and verified memory growth. Save per-state NLL/squared errors,
per-trajectory summaries and existing conditional slices. On the first two
trajectories of each checkpoint's first seed, independently run the exact
pinned author metric classes on the saved native TF clouds, weights and bands.
Require pointwise agreement within 2e-4*(1+abs(source value)), and aggregate
agreement within the same bound. This is a conservative FP32 arithmetic
localization tolerance, not a scientific NLL/RMSE acceptance margin; failures
trigger metric diagnosis, not filter rejection. Record errors even when passing.

Native vs common-author stochastic outcomes use independent random draws.
Those differences are explanatory only, not exact pathwise parity or a ranking.
Passing this diagnostic establishes the checked call chain, metric semantics
and runtime validity. It cannot establish multi-trained-unit noninferiority,
paper-wide replication, default readiness, HMC or superiority. The unchanged
KDM0.01nat/1% margins belong to future untouched multi-unit qualification.
Author-intrinsic failures remain acceptable reference outcomes. A heuristic
competition would change this source-replication question and is not required.

## Assumptions, failure modes and review

- Checkpoint/data reuse: paired with completed common-author evaluation by SHA256;
  risk is stale inputs; reject hash/audit mismatch before GPU initialization.
- Batch64 and256 development rows: inherited evaluation convenience, not the
  paper's final sample size; exact row/time checks expose truncation.
- Different TF/Torch streams: required native execution, not common randomness;
  no causal backend ranking from raw endpoint differences.
- All-time metric: source evaluator lines77-115; omitting t=0 or taking one
  global square root would compute the wrong target. Source-class replay checks it.
- Unchanged initialization and1e-8 floor: source-prescribed, not new protections;
  no clipping/ridges/damping/optimizer or network change.
- Cheap validity and immutability checks are hard interpretation vetoes; runtime
  and conditional numerical differences are explanatory. No tuning on these data.

Paper/source anchors: Younis/Sudderth2023, IWSG equations14-15 and bearings
experiment; local paper lines447-480; pinned src/evaluations/full_sequence_evaluations/
full_sequence_evaluation_base.py:62-120; src/loss_functions/loss_functions.py:15-80,
145-300,313-351; src/models/particle_transformer.py:112-175. This diagnostic
checks the existing filter route, not a new estimator or training protocol.

Skeptical review: PASS. The comparator is the actual source metric on identical
saved outputs; matched-checkpoint stochastic evaluation is separately labeled.
Wrong baseline, proxy promotion, source/sample semantics, stale checkpoints,
hidden CPU fallback, retracing, cap overruns and incomplete artifacts have
explicit checks. No change to source numerical method is authorized.

## Budget and execution

Allocate at most1800 GPU wall seconds and300 CPU wall seconds in total across
two versioned attempts, from the existing campaign balances after the full
pilot is charged. No concurrent GPU launch with training. Stop on invalid
inputs, finite/probability/metric checks, source drift, or resource cap. Local
harness repairs may retry in a fresh directory within the same total cap.
A failure is a repair trigger; it does not reject the research direction.

Supervisor: docs/benchmarks/run_kdm_native_evaluation_diagnostic.py.
Plan: this file. Result: controlled-smc-kdm-mechanism-result-20261011.md.
Output root: artifacts/controlled-smc-kdm-replication-20260928/native-evaluation-20261011/.
First command:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_kdm_native_evaluation_diagnostic.py --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/native-evaluation-20261011/native-evaluation-01

GPU runs use trusted/escalated RTX5080 access, tftwogpu, growth and thread limits1.
Independent author preparation/metric checks use isolated CPU-only tf-gpu2.
Manifest records commands, hashes, seeds, environment, memory policy, wall time,
results and budget charge. Preserve the completed pilot and source unmodified.
