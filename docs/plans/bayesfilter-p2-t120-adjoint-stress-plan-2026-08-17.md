# P2 T=120 Adjoint-State Stress Plan (2026-08-17)

Status: `PLANNED` (queue item 4 of the relaunch memo
`bayesfilter-squared-tt-program-reset-memo-2026-08-17.md`; discharges the
P2A full-horizon obligation recorded in
`bayesfilter-p2a-cost-prototype-result-2026-08-17.md`).

## Question

Does the adjoint score engine's forward-trace + reverse-sweep remain
resource-feasible at the workhorse horizon T=120 (n=2 scale), and what is
the measured memory/wall overhead versus the value-only engine on the
identical program?

This is an ENGINEERING RESOURCE measurement, not a scientific claim: no
posterior, accuracy, rank, or superiority conclusion may be drawn.

## Mechanism under test

The adjoint engine checkpoints every traced ALS update (design, weights,
target, solution, previous core) across all T steps before the reverse
sweep. Trace memory grows linearly in T; the P2A note selected the adjoint
on flop grounds with the full-horizon memory question explicitly deferred.

## Evidence contract

- Baseline/comparator: `run_value_filter_branch_axis` (same program, no
  trace) on the same data, config, and process isolation.
- Primary criterion (feasibility gate, declared before execution):
  adjoint T=120 completes without error and peak RSS <= 16 GiB
  (machine-resource sanity bound, not a scientific threshold).
- Explanatory-only diagnostics: peak-RSS ratio adjoint/value, wall-time
  ratio, per-step wall trend, trace-entry count. Single run each:
  DESCRIPTIVE ONLY (no uncertainty claim, V12-compliant because values
  come from the measured artifact).
- Veto: non-finite value, exception, or gradient mismatch vs the T=4
  verified regime config (config identity asserted, not re-verified).
- Will NOT be concluded: HMC readiness, gradient accuracy at T=120 (no
  oracle here), memory scaling beyond n=2, GPU behavior.
- Artifact: `docs/benchmarks/artifacts/p2_t120_adjoint_stress_20260817/attempt01/`
  (fresh versioned dir; JSON per mode + manifest).

## Default audit

- Config = the I-P2-4-verified n=2 regime (deg 10, rank 2, qo 12,
  tau 1e-6, hw 4.0, ridge 1e-10, sweeps 2): reviewed default (verified at
  T=4 this session); provenance: test_p2_adjoint_engine_fd._config.
- Shift-family LGSSM adapter (p=n): convenience choice — the resource
  question is model-light; adapter cost is not the object measured.
- Peak RSS via ru_maxrss in SEPARATE subprocesses per mode: avoids
  peak-of-process contamination between arms.
- Failure mode: concurrent P1B ladder run inflates wall time. Mitigation:
  run only after the ladder finishes; record system load in the manifest;
  wall numbers are descriptive regardless.

## Commands

    CUDA_VISIBLE_DEVICES=-1 <tf-gpu python> \
      docs/benchmarks/run_p2_t120_adjoint_stress_20260817.py \
      --mode value   --output .../attempt01/value.json
    CUDA_VISIBLE_DEVICES=-1 <tf-gpu python> \
      docs/benchmarks/run_p2_t120_adjoint_stress_20260817.py \
      --mode adjoint --output .../attempt01/adjoint.json

## Stop conditions

Wall > 60 min per mode, peak RSS > 16 GiB, or any veto above -> stop,
record, and take the store-vs-recompute trade decision to the phase
review instead of patching silently.
