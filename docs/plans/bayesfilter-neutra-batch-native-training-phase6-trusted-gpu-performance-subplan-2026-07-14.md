# Phase 6 Subplan: Trusted GPU Performance And Focused Repair

Date: 2026-07-14

## Phase Objective

Measure the certified exact batch target and full generic trainer on trusted
GPU/XLA, determine whether the former `~10.03 s/step` row-mapped bottleneck is
removed, and repair only identified performance/resource defects while
preserving all Phase 5 gates.

## Entry Conditions Inherited From Phase 5

- Correctness, status, objective gradient, deterministic state, graph, and
  dependency identity ledgers pass.
- CPU measurements are not GPU evidence.
- Total live trusted-GPU budget is at most 45 minutes in this phase.

## Required Artifacts

- A target-only GPU/XLA batch ladder under
  `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase6/attempt-*/`.
- One strict full-training five-step smoke at `B=128` after target admission.
- Exact commands, Git/environment/device provenance, compile/warm/steady timing,
  status counts, and artifact hashes.
- Phase 6 result and reviewed Phase 7 stability/protocol subplan.

## Predeclared Ladder

1. Escalated `nvidia-smi` and TensorFlow device probe.
2. Target-only batches `8`, `32`, `128`, `256`: one compile/warm call and three
   synchronized timed calls on identical truth-neighborhood construction.
3. Stop larger rungs on OOM, nonfinite output, invalid status, wrong device, or
   cumulative target-ladder live time over 20 minutes.
4. Run one strict five-step `B=128` full-training smoke in a fresh root if the
   `B=128` target rung passes.
5. If full training misses the aspirational `<=1.4 s/step` steady goal, profile
   in this order: duplicate compilation/calls, materialization, stationary solve,
   SVD recursion, flow/optimizer, and batch chunking. A miss is a repair trigger,
   not a correctness or research-direction veto.

## Required Checks

- Device and XLA provenance prove trusted GPU execution.
- Every timed call synchronizes by materializing a small scalar result.
- Warm/compile time is excluded from steady per-call/per-step rate.
- Target status is valid and finite for every timed row.
- Recheck one CPU-XLA/scalar parity test after any implementation repair.
- Preserve each failed attempt and use a fresh output root.
- Python compile, JSON validation, and `git diff --check` for harness changes.

## Evidence Contract

| Item | Phase contract |
| --- | --- |
| Question | Is the certified batch-native exact target practically usable for GPU NeuTra training, and where is any remaining bottleneck? |
| Baseline | Historical row-mapped `B=128` full training approximately `10.03 s/step`; descriptive only because code topology changed. |
| Primary engineering criterion | Valid trusted-GPU/XLA `B=128` target and five-step trainer complete with no fallback and measured steady rates. |
| Aspirational repair target | Full training `<=1.4 s/step` after warmup; miss triggers profiling but does not invalidate correctness. |
| Hard veto | Wrong device, non-XLA route, invalid/nonfinite target, parity regression after repair, OOM without lower rung, corrupted artifact, or GPU budget exhaustion. |
| Explanatory only | Batch scaling, compile time, memory, component timing, and comparison with historical DSGE rates. |
| Nonclaims | Speed and five steps do not establish recipe quality, posterior correctness, HMC readiness, convergence, or scientific validity. |

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Batches 8/32/128/256 | small scale ladder around decision batch 128 | 256 may exceed memory or add little information | stop-on-OOM and B=128 remains decision row | reviewed ladder |
| Three timed calls | bounded development budget | insufficient timing uncertainty | report individual timings; do not rank close rows statistically | diagnostic evidence |
| Five-step strict trainer | active target-specific harness | compile dominates or losses mislead | separate compile/warm; loss explanatory only | engineering smoke |
| `<=1.4 s/step` | master-program aspirational practical target | target may inherently exceed DSGE task cost | miss triggers component evidence, not abandonment | aspiration, not hard gate |

## Skeptical Subplan Audit

- Wrong baseline: primary result is absolute same-code target/training timing;
  historical row-mapped timing is descriptive context only.
- Unfair comparison: compile and steady timing are separated and every call is
  synchronized.
- Proxy promotion: speed cannot establish scientific or transport quality.
- Hidden resource risk: batch ladder and cumulative live-time stops are explicit.
- Artifact fitness: structured output contains device, XLA, status, per-call
  timings, environment, command, Git state, and source hashes.

Audit verdict: **PASS**. The ladder answers the engineering question within the
existing GPU budget and has a focused repair order.

## Forbidden Claims And Actions

- Do not use non-escalated GPU failures as environment evidence.
- Do not promote a batch or recipe using three timing observations.
- Do not start 100/500/5000-step training in Phase 6.
- Do not change target/status math for speed.

## Exact Next-Phase Handoff Conditions

Phase 7 starts when trusted GPU `B=128` target and five-step trainer are valid,
the measured rate supports a refreshed 100-step budget, and the Phase 7 subplan
retains target-specific recipe/heldout/downstream evidence boundaries.

## Stop Conditions

Stop only for a hard veto above, exhausted 45-minute GPU budget, or a repair
that would change target/method/hardware class or materially expand compute.
Performance misses alone trigger focused repair and continuation.

## Phase-End Procedure

1. Run required local checks.
2. Write Phase 6 result/close record.
3. Draft or refresh Phase 7 stability/protocol subplan.
4. Review Phase 7 suitability and continue when no real blocker exists.

