# NeuTra HMC Robustness Phase S1 Subplan

Date: 2026-07-15  
Status: `EXECUTED_AND_CLOSED`

## Objective And Entry Conditions

Test one independent training seed `(20260715, 1203)` on the already validated
LGSSM fixture using the selected `wide_2x_lr5e3` recipe, fresh weights, 5,000
GPU/XLA batched steps, and the canonical shared sequential HMC controller.
C0-C2 and their focused tests must pass first.

## Evidence Contract

The question is training-seed viability on this fixture. The comparator is the
same fixture-bound tuned plain-HMC result already admitted for the prior
campaign. Training loss is explanatory only. Promotion requires finite/status
valid training and frozen parity, fixed-kernel tuning, warm-up readiness,
cumulative retained full convergence, plain-HMC agreement, and truth recovery.
Any nonfinite/status/energy-error health event is a hard candidate veto.

The run must archive the selected recipe reference, new seed, GPU memory-growth
status, XLA/device status, fresh payload hashes, commands, wall times, separate
warm-up and retained tensors, diagnostics, comparator identity, and result.
No population reliability, superiority, calibration, broad robustness,
production readiness, or new-fixture claim may be made.

## Defaults And Pre-Mortem

The recipe is a transferred winner on the same target and is held fixed to
isolate seed sensitivity. Its failure mode is seed-dependent optimization or
bad transport geometry; frozen parity and downstream HMC distinguish wiring
from transport quality. Step size must be freshly screened rather than copied
as a result. The existing comparator is valid only because observations and
target signature are unchanged.

## Commands, Budget, Stop, And Handoff

Run one 5,000-step trusted GPU/XLA training job with memory growth in a fresh S1
root, then CPU-hidden/XLA objective, tuning, sequential admission, and
confirmation jobs. Aggregate training budget is 30 minutes and HMC budget is
three hours, with one fresh-directory retry only for localized infrastructure
failure.

Write an S1 result and F0 subplan. A candidate veto is a scientific S1 result,
not a continuation veto for F0. Stop the program only for broken target,
comparator, shared harness, artifact corruption, unavailable trusted GPU, or
budget exhaustion.

Skeptical audit verdict: `PASS`. The baseline is target-matched, proxy metrics
cannot promote, failure classes and caps are explicit, and the artifact answers
the stated seed-specific question.
