# CPU tree / GPU integration feasibility experiment

## Question and research intent

Owner request: plan, review and execute a substantial-model test of whether
CPU/XLA tree bookkeeping combined with GPU/XLA model and integration is useful.
This is a standalone engineering experiment. The running million-draw funnel
replication stays unchanged. The candidate is a resumable compiled CPU tree
controller calling the existing compiled GPU `grow_leaf`, including R2P local
refinement, reverse checks, model and autodiff. Python schedules device calls
and transfers only; no Python/NumPy numerical sampler or model is permitted.
This is separately XLA-compiled components, not one fused heterogeneous graph.

Baseline ladder: full CPU/XLA, full GPU/XLA, hybrid CPU tree/GPU leaf. A same-CPU
hybrid replay is an engineering reference, not a competing optimized baseline.
Reuse the existing R2P primitives and exactly the same tree traversal, RNG,
stopping, selection and weight arithmetic. Fixed model data remain on device.
Do not transfer the data or tree stack on every leaf.

Promotion criterion: retained-state/cache and discrete-work checks pass, large
model GPU gradient time is measurably below CPU time, and a paired 95% interval
for hybrid/full-GPU runtime is entirely below one. This nominates only a useful
optional execution design on this target. Full CPU remains a mandatory timing
baseline; report if it is faster. If intervals overlap one, no speed ranking.

Promotion veto: numerical mismatch, different discrete work in matched trials,
nonfinite retained state/cache, hidden host numerical fallback, retracing, or
GPU model failing the cost pilot. Continuation veto: device/allocator failure,
resource cap, source corruption, invalid target/score. Repair trigger: localized
scheduler/harness mismatch, bounded fix followed by focused test. Explanatory
only: leaf count, logical gradient count, bytes/round trips, compile time, peak
memory, transfer-only and value/gradient timings. No posterior convergence,
ESS superiority, production/default promotion, arbitrary-model advantage, or
claim of a mathematically GPU-only target is supported by this short test.

## Model and assumptions

Use synthetic Bayesian nonlinear regression with four independent chains, 128
latent bias coordinates, 128 hidden units, and initially 16,384 observations:
`h1=tanh(X + q/sqrt(N)); h2=tanh(h1 W); prediction=h2 a`,
`logp=-sum(q^2)/2-sum((prediction-y)^2)/2`. Independent Normal noise variance
one and standard Normal prior define the target. Generate X,W,a and observations
once with fixed stateless TensorFlow seeds; y uses q=0 plus Normal noise. W and
a scale by sqrt(width) to control forward variance. q/sqrt(N) is an explicit
coordinate scaling of the synthetic prior/likelihood, not transferred tuning
from the funnel. Every gradient recomputes parameter-dependent nonlinear hidden
activations and a large matrix product; no fixed sufficient-statistic reduction
is substituted. All four implementations use exactly this target and FP64.

This is a synthetic calibration model selected to exercise expensive GPU
linear algebra, not an existing trained scientific model. The model must earn
its GPU-beneficial description in the pilot. If N=16,384 is insufficient, test
N=32,768 then 65,536, at most three sizes; keep the first with lower measured GPU
gradient latency and confirm timing uncertainty in the final paired blocks.
No dtype/TF32 switch to manufacture a speedup. Fixed resident data and repeated
value/gradient work must remain visible in HLO. Maximum eight GiB device tensor
memory; fail on inability to verify growth.

Defaults/provenance: four chains and FP64 inherit the current replication as
controlled baselines. Width128 and the size ladder are convenience workload
choices; the gradient pilot is their early diagnostic. Seed20261029 and offsets
are reproducibility choices. Initial q=0.03*Normal, identity mass, epsilon0.2,
depth4, R2P delta0.3, cap4 and jitter0.2 are bounded engineering hypotheses, not
tuned posterior settings. Calibration uses a separate seed, inspects finite
cache, movement, exact replay and actual work; if no useful trajectories or
excessive refinement occur, reduce epsilon once to0.1 before freezing. Do not
change tuning per backend. A seven-seed paired timing design with alternating
execution order controls first-order thermal/order bias; CPU threads fixed4
(also report an 8-thread CPU sensitivity in a separate process if CPU loses and
budget permits). Compilation excluded from steady timing and reported apart.

## Implementation, checks and budget

1. Implement CPU pause/resume tree using existing numerical primitives; compile
   begin/advance/finish with stable signatures and XLA. GPU leaf is the existing
   `grow_leaf`. Keep device copies and host synchronization explicit.
2. Test same-CPU replay against native full transition for multiple seeds,
   chain masks, nonidentity metrics, deterministic and R2P variants, including
   reverse-zero endpoints. Check all returned fields and RNG.
3. GPU smoke and model pilot: check values/gradients vs CPU plus tiny independent
   analytic/finite-difference and row-isolation fixtures; no pfor. Inspect HLO,
   trace counts, memory and physical placement. Copy-only and evaluator round
   trips explain overhead but cannot promote hybrid execution.
4. Freeze smallest qualifying large size and calibration. Run seven paired
   seeds, two repeats each, initially two transitions per timing block. All
   variants start from the same state/key, force completion, and compare output
   fields/work after timing. Up to four transitions if cost permits. Use
   per-seed median log ratios and Student t(df6) 95% intervals (critical2.446912
   inherited from existing paired timing utility). Hardware sharing makes any
   ranking conditional on this session. No independent-model generalization.
5. Preserve source snapshot/hashes, git commit, commands/env, seeds, data
   checksums, HLO, per-trial timing, validity, device/memory reports and decisions.

One bounded campaign, at most45 minutes of benchmark wall time and two local
repair attempts; preparation/unit tests separate routine work. Sequential
backend timing, not competing benchmark jobs. Do not stop other jobs. Choose an
idle GPU using trusted inventory. GPU2 previously had Xid13 in the funnel run;
if it is the only idle GPU, explicitly retain that risk and require a fresh
smoke. Another CUDA execution error stops GPU testing; no automatic hardware
reset or retry loop. Use TF_FORCE_GPU_ALLOW_GROWTH=true before import and verify
growth. CPU references use explicit CPU placement in the mixed process and
standalone CPU tests hide GPUs.

Environment: /home/ubuntu/anaconda3/envs/tfgpu/bin/python, TF2.20, FP64, TF32off.
Versioned results: docs/benchmarks/artifacts/walnutpie-hybrid-large-20261029-01/.
Command: `python -m docs.benchmarks.walnutpie_gpu_tf.hybrid_benchmark --output
ROOT --gpu 2 --seconds 2700`. Result/report and manifest live under ROOT.

## Skeptical review before execution

Wrong-baseline risk: cheap funnel results do not choose placement for an
expensive model; require full CPU and GPU baselines and a model cost pilot.
Proxy risk: density timing alone does not decide sampler speed; compare full
transitions including actual transfers and Python dispatch. Hidden-method risk:
CPU/GPU graph nesting does not guarantee placement; use explicit separate
compiled boundaries and inspect HLO/device. A Python numerical tree rewrite
would confound the comparison; keep arithmetic and traversal in CPU/XLA, reuse
shared kernels, and demand same-CPU field/RNG replay. Workload-size selection
uses only model latency, not favorable hybrid timings. Short chains cannot
establish posterior quality. Full-GPU versus hybrid timing requires matched
work, not just equal transition count. Existing GPU failure and sharing remain
explicit limitations. Plan passes subject to these early checks.

## Execution checkpoint and explanatory follow-up

Both same-CPU replay tests pass (18.13 seconds), covering deterministic and R2P
variants, three seeds, nonidentity metric and varying chain steps. The first
16,384-row model pilot passes: approximately 0.249 seconds CPU versus 0.021
seconds GPU for a four-chain density/gradient evaluation, with maximum absolute
difference 6.94e-17. No larger workload is selected. Full kernels and hybrid
components each have one XLA trace; GPU HLO uses cuBLAS with no host sampler
callback. Paired full-transition trials proceed under the existing budget.

After main timing, measure resident-data raw buffer round trips and zero-work
compiled leaf dispatch as explanatory diagnostics (seven blocks of thirty
exchanges, matching the observed transition-block count). Repeat the seven
seed CPU baseline once at eight threads in a separate CPU-only process using
the exact recorded target, starts, steps and draw count. These are already
planned CPU-thread sensitivity and transfer diagnostics; they do not alter the
primary paired comparison or justify changing the implementation after seeing
results. Preserve them under ROOT/overhead and ROOT/cpu8. Subtract neither
microtiming from measured full-transition time: it is not a causal decomposition.
Remaining budget stays within 2,700 seconds aggregate benchmark wall time.

## Completion

Executed and terminally reviewed. Main experiment, CPU8 sensitivity and transfer
diagnostics completed with 486.27 seconds recorded benchmark wall time. All
fourteen paired blocks pass engineering parity, but hybrid/full-GPU time is
1.1381 (95% interval 1.1225--1.1539), so speed promotion fails on this target.
Full GPU remains the measured choice for the tested large model. No scientific
posterior/default claim and no change to the separate funnel replication.
Full result, decision tables, source verification and limitations:
`bayesfilter-walnutpie-hybrid-large-model-result-2026-10-29.md`.
