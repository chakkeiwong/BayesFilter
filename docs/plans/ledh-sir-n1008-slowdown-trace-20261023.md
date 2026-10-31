# Trace the SIR slowdown at 1008 particles

23 October 2026. Active branch `ledh-refactor-with-policy-fix`, HEAD
`90a3acebd3965da5bdd64cb165dfdfe2d8cf0649`, with preserved working changes.

## Question and evidence contract

Explain the fresh N=1008 plain and conditional regressions relative to pre-merge,
and the different behavior of the physical mixture. The preceding eligible
measurement is `artifacts/ledh-sir-n1008-runtime-20261023/result.md`. No N=64
measurement, learned checkpoint, or calibration enters this investigation.

Trace the actual benchmark consumer through `make_moment_endpoint`,
`build_backward_guided_ledh_kernel`, the analytical recursion, flow substeps,
proposal density, and retained validity outputs. Inspect optimized GPU HLO from
the same N=1008 consumer. Static source assertions and fresh compiled graphs
must agree before calling a dependency an implemented cause.

Primary diagnostic: matched, synchronized endpoint latency with valid healthy
outputs and paired block intervals. Compare untouched pre-merge and current
snapshots and isolate two mechanisms in process-local diagnostic variants:

1. Replace only the flow determinant checker with zero determinant outputs and
   true masks. This is allowed ONLY for plain/conditional consumers whose
   importance ratios do not use those outputs; parity must demonstrate that.
2. Replace only `safe_cholesky(..., 'flow_innovation')` with an unchecked
   Cholesky and true mask. All other factorization calls remain untouched.
3. Combine those interventions to measure interaction, not to assume additive
   cost. The untouched current route remains the comparator.

These are deliberately incomplete validity checks, NOT repair candidates.
Record their loss of rejection on singular determinants/asymmetric innovations.
Do not modify algorithm source, select defaults, or promote diagnostic variants.
Numerical parity on healthy inputs does not establish equivalent safety.

## Intent ledger and assumptions

- Mechanism: newly consumed determinant/force validity keeps previously dead
  matrix arithmetic alive; innovation checks add dependent work in 24 substeps.
- Promotion criterion: none for algorithms; this is causal engineering diagnosis.
  Attribute a runtime effect only if healthy outputs pass existing 1e-8 absolute
  plus 1e-8 relative checks and paired 95% ratio interval excludes one.
- Promotion veto: any diagnostic variant is ineligible for runtime use regardless
  of its speed. No efficacy, score-accuracy, HMC, or default-readiness claim.
- Continuation veto: invalid untouched endpoint, changed source/input hashes,
  numerical disagreement, wrong GPU/environment, missing artifacts, budget limit.
- Repair trigger: bounded local harness/HLO export failure; preserve attempt and
  retry under the same total cap. Unsupported hypotheses may be rejected without
  ending investigation of other mechanisms.
- Explanatory diagnostics: HLO loop/factorization dependencies, call-chain
  assertions, per-arm timings, group size, memory, and live GPU process snapshots.
- Heuristic adversaries: unchanged pre-merge; unchanged current; diagnostic
  determinant removal; innovation-check removal; both. They isolate costs, not
  statistical algorithm quality. Evaluate each separately for plain/conditional
  and both datasets; physical-mixture baseline remains unmodified.
- Keep the fresh comparison's FP64, T=5, 24 flow steps, OT(1,5), analytical e0
  direction, reset/correction settings, N=K=1008, seeds and fresh inputs. These
  are comparability choices, not tuned defaults; early diagnostics are exact
  input hashes, compiled signatures and warm-call parity. No training or retuning.
- Single parameter cloud; N batches particles, not parameter proposals. Results
  do not establish FP32/TF32, long-horizon, or native parameter-batch performance.
- Source snapshots are pre-merge 4d6915cc and current from the preceding fresh
  campaign, verified before/after. Source versions may be inspected despite the
  disqualification of earlier measurements. No old numerical results are reused.

## Execution, budget, and skeptical review

Use `scripts/trace_sir_n1008_runtime.py`, sharing the exact existing benchmark
worker and its output materialization. Add only host-side diagnostic hooks and
optimized-HLO export to that worker; default behavior remains unchanged. Test
the evidence handling before launch. Each variant runs in an isolated process;
only one numerical request executes at once. Rotate sources/arms/datasets across
12 timing blocks, after separate warm calls. Bootstrap whole blocks (4096 draws,
pointwise 95% intervals); report effects as compiler-and-runtime interventions,
not exclusive kernel-time accounting. Include physical pre-merge/current controls.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, RTX 4080 SUPER
UUID `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, trusted GPU access, verified memory
growth, TF32 off, XLA on with `--xla_gpu_enable_command_buffer=`. Record devices,
memory policy, source/input/driver hashes, exact commands, wall time and seeds.
Poll GPU compute processes during blocks; do not stop unrelated processes.

Artifacts: `docs/plans/artifacts/ledh-sir-n1008-slowdown-trace-20261023/attempt-01/`.
Cap: 20 minutes elapsed across at most two attempts and five resident workers
(100 summed worker-minutes maximum). Stop workers on completion or failure.
HLO and healthy-output evidence are kept separately from diagnostic safety loss.

Pre-run skeptical audit: passes with explicit limits. Comparators and inputs are
fresh at N=1008; this tests the measured endpoint rather than a small primitive.
Changed compiler optimization can make intervention effects nonadditive, so
report joint and separate effects. Removing a guard cannot be a proposed fix;
the negative guard fixtures make that boundary executable. The conditional
proposal itself is identical across source snapshots, but its surrounding flow
and validity consumers differ. GPU sharing and compilation are possible
confounders: exclude compilation, interleave variants, and retain process polls.
All scientific controls stay fixed. A successful run explains engineering cost
only; it cannot rescue withdrawn scientific results or prove broader speed.

## Completed trace

The source and optimized-HLO trace completed in
`artifacts/ledh-sir-n1008-slowdown-trace-20261023/attempt-01/`. The 24 healthy
cross-source parity checks passed, the source snapshots remained unchanged, and
the focused harness had 8 passing tests. The trace identifies the regression
mechanism described in the result note. The intervention timings are retained
as compiler-structure diagnostics only: five XLA workers were resident on the
same target GPU, so those timings cannot assign an exclusive percentage of the
pre-merge/current gap to either guard family.
