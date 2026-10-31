# Exact-input neural precision localization — 7 October 2026

This is the planned localization follow-up to the failed fresh neural gradient
screen. It shares that phase's 600 CPU-second, 900 GPU-second and three-attempt
limits. Attempt 01 consumed 10.0999 CPU and 153.0542 GPU seconds; conservative
routine CPU accounting is 60 seconds. Reserve no additional compute allocation.

Question: when the failing float32 fixture's parameters, constants, initial
state, observations, targets, masks, samples and proposal noise are held exactly
fixed, do the actual author and TF derivatives agree in float64? The existing
double cases use different draws and do not answer this question.

Baseline: actual pinned author full-sequence trainer, KDE_NLL, TBPTT modulo 4,
neural model and Adam, with the preserved fresh-A first-update inputs replayed.
Both MDPF and A-MDPF; float32 control plus exact cast of those values to float64.
Only one update is needed: the observed veto concerns update zero, so there is
no accumulated optimizer-state confound.

Primary localization criterion: source float32 replay reproduces its saved
first update exactly (loss, gradients and post-Adam state); float64 author/TF
loss, outputs, gradients, clipping and Adam comparisons pass the unchanged
1e-9*(1+abs(source)) bound. This establishes only fixed-fixture high-precision
compatibility. Float32 discrepancies retain their unchanged 128-eps checks.
Finite/validity, source/input hashes, full replay-tape consumption, exact initial
parameter transfer, call-chain, device and tracing failures veto interpretation.
No finite-difference or float64 result is a universal gradient proof.

Report float32 source and TF errors relative to the same-input author float64
reference, the float64 inter-framework discrepancy, and float32-to-float32
errors on the originally failing tensors. Ratios and distances are explanatory,
not relaxed acceptance criteria or a statistical ranking. If double precision
also disagrees, inspect the first differing source operation. If double agrees,
investigate finite-arithmetic amplification using actual downstream error and
independently justified calibration; do not accept replication from this result.

Defaults and assumptions: the pinned historical checkpoint is diagnostic only;
the fixed development rows/seed are selected to explain a known failure, not to
qualify performance. Exact initial-state replay is appropriate for the first
transition because this source setup initializes from truth with frozen draws.
Require a float32 replay control to expose lost graph dependencies or changed
source semantics. Replaying samples preserves their source stopped-gradient
role. The frozen proposal/noise tapes separate backend arithmetic from RNG.

Skeptical review: PASS for localization. It fixes the actual defect in the prior
high-precision comparison: different random inputs. The source control can
invalidate the harness before GPU work. No runtime implementation, training
worker, model architecture, fitting rule or tolerance changes. No heuristic
performance ranking is attempted. Author full training continues under its own
bounded supervisor; this short one-thread reference/GPU diagnostic uses the
previously documented concurrency exception.

Execution: docs/benchmarks/run_kdm_neural_precision.py --output
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-angle-20261007-02
using /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B with escalated GPU
permissions. CPU child uses tf-gpu2 with GPU hidden and one thread. Artifacts
include full commands, environment, immutable input hashes, helper snapshots,
reference replay, GPU comparison, gradients and CPU/GPU wall-time accounting.
Result: controlled-smc-kdm-neural-precision-result-20261007.md.

Pre-mortem: returning captured initial values could remove a relevant derivative;
the exact source float32 control is mandatory. Casting only data but leaving
non-buffer network constants or parameters at different precision would confound
the result; reuse the audited model-construction path and compare transferred
values. No second update is interpreted.

Implementation review before launch: all three helpers compile. The source
helper copies every named parameter from the failing float32 case and replays
its data, initial state, 15 sampled clouds and 16 proposal innovations. Its
float32 control requires exact equality of captured outputs, loss, norm, raw
and clipped gradients, parameters and Adam state before GPU execution. The
GPU helper calls the unchanged actual update and preserves raw gradients for
the precision comparison. The supervisor verifies the failing fixture hash
against attempt 01 and counts that attempt plus 60 routine CPU seconds against
the shared phase caps. Float32 failure remains visible even when the stated
float64 localization criterion passes. Review verdict: PASS.
