# KDM circular density and training-loss consumer

This is a bounded continuation of the replication master and gap-closure plan.
The question is whether TensorFlow can reproduce the released bearings
resampling-density, IWSG, and posterior-NLL consumers, including their different
normalization conventions. This is implementation parity, not a trained-filter
quality comparison. Both full replications remain incomplete.

## Source and mathematical target

Younis and Sudderth (2023), local paper
`.localresources/papers/younis-sudderth-2023-long-range-tracking.pdf`, equations
14–15 (text446–470), section5, and equation18 (text627–648), define stopped
mixture samples, importance-weight gradients, and the posterior NLL. Bearings
uses Normal position kernels and a von Mises angle kernel.

Author commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18, under
`.localresources/code/younis-mdpf-neurips-2023`:

- `src/models/kde_particle_filter/kde_particle_filter.py:690–751`: resampling
  weights/bandwidth selection, detached samples, and IWSG injection.
- `packages/kernel-density-estimator-bandwdidth-prediction/packages/kernel_density_estimation/kernel_density_estimator.py:14–57,232–267`:
  resampling concentration is 1/(bandwidth+1e-8); weights gain1e-8 and normalize.
- The same package's `kde_computer.py:26–97`: posterior concentration is
  1/bandwidth; supplied weights multiply component densities without this extra
  normalization or epsilon.
- `src/loss_functions/loss_functions.py:35–80`: posterior loss is
  -log(sum_i weight_i kernel_i(state)+1e-8).

For a circular dimension log kernel = kappa*cos(x-location)-log(2*pi)-log I0(kappa).
The implementation must preserve both source concentration rules and both weight
rules. A shared normalized log-density is wrong for the posterior-loss consumer.

## Evidence contract and intent ledger

Implement a batched TF product-Normal/von-Mises consumer with stable signatures,
XLA enabled, explicit dtype, and stopped replay samples. Compare with the actual
unmodified author resample_particles and KDE_NLL.compute_loss methods, loaded
by AST together with the real author KDE classes. The diagnostic supplies
constructor-independent settings; it is not the author's full neural model.
The author sampler supplies realized samples to both sides; this does not test
a TensorFlow circular random sampler.

Compare resampling log densities, injected weights, raw and normalized IWSG
expectation gradients, posterior densities, NLL values and gradients in all
particle locations, weights and bandwidths. Include ordinary, angular-seam,
small-weight, and far-tail/floor cases. Perturb inputs inside the same signatures
and require one trace. Check angular periodicity and invalid bandwidth, negative
weight and zero-mass rejection. Density normalization and gradient differences
are correctness checks; runtime and allocator use are explanatory only.

Primary pass: all comparisons within predeclared numerical bounds, finite valid
outputs, correct stopped-sample gradients, and no CPU fallback or extra traces.
Use float32 bound 128*machine_epsilon*(1+abs(reference)) for values/gradients;
float64 bound 1e-9*(1+abs(reference)). Pre-run inspection of the installed
reference torch/distributions/von_mises.py:16–99 found a two-branch polynomial
for log I0. Port that same finite formula to TensorFlow with a bounded Horner
loop. Record the installed file hash. This numerical choice is a replication
baseline; check positive finite concentrations and polynomial values. A bound
failure is a repair trigger,
never authority to increase the tolerance after viewing candidate error.

Additional loss-substitution control must show that the resampling density
cannot stand in for the posterior loss on unnormalized/tiny-weight fixtures.
This control is explanatory and prevents an untested call-chain claim.
Also call the actual source weigh_particles method between IWSG and NLL, with
constructed positive weight-model output and fixed downstream particles.
Compare loss and gradients in both upstream mixture and downstream posterior
parameters. Neural proposal/encoder and complete training are excluded.
Source hash mismatch, corrupted replay, unsupported source settings, timeout or
budget exhaustion stop the attempt. Numerical mismatch rejects this candidate
and triggers localized repair within the budget. No claim about training,
optimizer parity, unbiased finite-particle learning, autonomous sampling,
AppendixB reproduction, no-worse filtering quality or full replication follows.
Heuristic superiority is outside this deterministic translation question.

## Assumptions and skeptical audit

| Choice | Provenance/status | Failure mode and early diagnostic |
|---|---|---|
| Normal/Normal/von-Mises | Paper bearings model; source configs | Wrong concentration convention; inspect both actual density consumers |
| 1e-8 offsets | Released source | Silently correcting author numerics; compare actual endpoints |
| Constructed small clouds, batch2,25 particles | Mechanics diagnostic; source particle count | Mistaking these for training data; explicit fixture labels |
| float32 and float64 | Source dtype and diagnostic reference | Source default buffers downcast; set/report torch default dtype per arm |
| Fixed author samples | Matched-input gradient oracle | Claiming random sampler parity; explicitly excluded |
| Special-function tolerance | Matched installed PyTorch polynomial | Hiding a formula error; source inspection, concentration probes, seam/gradient checks |

Skeptical audit: this closes a missing consumer check only. It avoids the wrong
Gaussian-only baseline and does not treat loss parity as training replication.
Existing sequential checks used constructed AppendixB coefficients and SGD;
they do not already answer this circular bearings-loss question. No author
configuration, optimizer or numerical stabilization is changed.

## Execution and budget

Up to two GPU diagnostic attempts, at most180 seconds each (360 seconds total),
within the existing702.756496 GPU seconds. CPU author reference: at most120
seconds per attempt, conservatively include all child wall time in GPU charge.
No package installs. Use the existing mdpf-reference-20260924 CPU environment
with CUDA_VISIBLE_DEVICES=-1. TF runs in tftwogpu with escalated device access,
verified memory growth and TF32 disabled for this comparison. Unique outputs:
`docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-circular-20261006-01`
and -02 if needed. Preserve source snapshots, hashes, commands, environment,
seed20261006, logs, wall time, device/trace/memory evidence and full comparisons.
Charge attempts through the existing supplementary diagnostic ledger. Routine
compile and focused tests are separately recorded.

Next action: inspect the installed author-reference special-function formula,
implement the consumer and independent-source diagnostic, then execute it. A
successful result closes this bounded density/loss gap; full neural training
and checkpoint conversion remain separate work.

Pre-execution source-formula review: PASS. The source polynomial is available; no numerical results have been used to select the FP64 tolerance.

Prelaunch implementation review: PASS. Source resampling, weight normalization
and NLL call chains are exercised, with actual source bodies and real KDE
classes. Source hashes fail closed at runtime. Samples are explicitly replayed;
neural model outputs are constructed and cannot establish full training.
There are no NumPy, pfor or per-sample loops in the TF implementation; the
Horner loop has at most8 iterations and every endpoint has a stable signature.
TF GPU0 has approximately11GB free at preflight; the other GPU is not selected.
Initial sandbox-only compile attempt failed in bubblewrap, before Python;
escalated compile checks pass for all3 files.

Exact numerical command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B docs/benchmarks/diagnose_kdm_circular_tf.py --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-circular-20261006-01
Environment: CUDA_VISIBLE_DEVICES=0, TF_FORCE_GPU_ALLOW_GROWTH=true,
TF_NUM_INTRAOP_THREADS=2, TF_NUM_INTEROP_THREADS=2. Outer timeout180seconds.
The entire child wall time is conservatively GPU-charged, including CPU source.

Attempt01:311/315 checks pass; all157 FP64 checks pass. Four FP32
gradient checks fail. Largest relative discrepancy is0.307/598.854 (~0.0513%)
in the raw circular-bandwidth score. Source PyTorch Distribution.broadcast_all
expands concentration to[B,N] before logI0, so the backward program reduces
over samples before combining the density and normalizer derivatives for each
particle. The first translation computed one logI0 per batch/dimension and
reduced over particles before this cancellation. Its formula is algebraically
equal but the finite gradient program differs. Repair that graph to match the
source and preserve the source's sequential subtraction order. No tolerance,
sample, source, dtype or numerical objective changes. This is a localized
source translation repair, within the original two-attempt budget.
Attempt01 conservatively charged19.885164164938033 GPU seconds.
Hardware correction: CUDA index0 maps to RTX4080SUPER, despite nvidia-smi
listing RTX5080 first. Actual manifest records4080SUPER and memory growth;
the preflight free-memory statement referred to the other card. Keep the same
CUDA index for attempt02; this diagnostic's allocator peak is recorded.
The actual -02 command differs only in the versioned output suffix.

Repair review: PASS. The changed broadcasting is supported by the inspected
PyTorch VonMises constructor/broadcast_all call chain; no damping, clipping,
ridge or gradient replacement is introduced. The full-training gap remains.

Attempt02:311/315 checks pass, with the same four FP32 failures; the
source-broadcast repair is insufficient. Its17.879331667907536seconds are
charged. Both planned GPU attempts are complete; do not repeat GPU comparisons
without a new discriminating hypothesis.

Next bounded localization: replay the same saved author fixture through the
same candidate on CPU with jit_compile=False and CUDA_VISIBLE_DEVICES=-1,
maximum60CPU seconds, one attempt. Compare the same predeclared value/gradient
bounds. This tests whether removing XLA/GPU removes the discrepancy; it cannot
separate compiler from device effects or establish GPU/default readiness.
It does not relax the original criterion or replace the GPU path.
Store under kdm-circular-cpu-localization-20261006-01. The existing remaining
CPU allocation covers this explicit reference exception. Source/candidate
hashes, exact commands, traces, outputs and wall time are mandatory.
Skeptical localization review: PASS; it is an explanatory experiment with a
fixed implementation and existing fixture, not candidate tuning or admission.


Terminal state: both bounded GPU attempts and the CPU localization have
finished. Float64 consumer checks pass; the complete float32 screen fails.
The result, preserved comparisons and terminal engineering review are in
[the result note](controlled-smc-kdm-circular-consumer-result-20261006.md).
20 focused tests pass. The latest master and checkpoint retain both full
replications as incomplete and 664.9919998964178 GPU seconds remaining.
