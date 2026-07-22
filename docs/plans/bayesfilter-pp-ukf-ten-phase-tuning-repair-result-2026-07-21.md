# BayesFilter PP-UKF Ten-Phase Tuning Repair Result

Date: 2026-07-21  
Plan: `docs/plans/bayesfilter-pp-ukf-ten-phase-tuning-repair-plan-2026-07-21.md`

## Decision

`PHASES_1_TO_7_EXECUTED_PHASE_8_RESOURCE_AND_BOOTSTRAP_VETO`

The retained-target replay defects were repaired and verified. The real PP-UKF
GPU/XLA target-cost benchmark passed, and a bounded canary produced a terminal
artifact. The canary did not admit tuning: bootstrap ended
`repair_budget_exhausted` without an acceptance-promoted kernel, and the
diagnostic downstream windowed stage closed with HMC/runtime hard vetoes.

The full serious tuning campaign was not rerun. The preflight projected about
8.73 GPU-hours from the prior observed route, exceeding the four-hour campaign
cap, and the new serious-path policy independently forbids continuing from a
non-promoted bootstrap kernel. Phases 8-10 are therefore correctly unexecuted.

## Repairs

- PP-UKF and its runtime wrappers now declare and preserve the flat retained
  batch contract.
- `BatchNativeBoundAdapter`, `FixedTransportValueScoreAdapter`, the bootstrap
  fixed-mass wrapper, and the affine warmup wrapper preserve an optional
  combined value/score/status protocol.
- Retained target health consumes that combined protocol once per batch and
  uses it for failure localization; adapters without it retain the legacy
  fallback.
- Serious public tuning now fails closed when bootstrap has no
  acceptance-promoted kernel. Geometry fallback remains available only to
  non-promoting diagnostic routes.
- A bounded GPU/XLA target-cost benchmark, prospective resource preflight, and
  timeout/heartbeat canary runner were added.

## Phase Results

| Phase | Result | Gate |
| --- | --- | --- |
| 1. Fixed identity freeze | Passed | Existing propagation/runtime invariants and focused tests passed |
| 2. Retained flat batching | Passed | PP-UKF rank-2 and wrapper propagation tests passed |
| 3. Combined status protocol | Passed | Parity, call-count, fallback, and failure-localization tests passed |
| 4. Real target-cost benchmark | Passed | GPU/XLA finite parity; 64-draw combined call mean `0.0704 s` |
| 5. Resource preflight | Full-run veto | Projected `8.73 h`; four-hour cap and 75% margin failed |
| 6. Bootstrap fallback repair | Passed | Serious route stops without acceptance-promoted bootstrap |
| 7. Bounded canary | Terminal hard veto | 32 diagnostic transitions completed; no admitted kernel |
| 8. Full tuning-only campaign | Not executed | Resource veto and bootstrap-promotion veto |
| 9. Terminal tuning admission | Not reached | No serious terminal tuning candidate exists |
| 10. Sequential HMC | Not executed | No admitted tuning artifact; sampling remained false |

## GPU Benchmark

Artifact:
`docs/plans/artifacts/bayesfilter-pp-ukf-ten-phase-repair-20260721-01/phase4-target-cost.json`

- Device: NVIDIA GeForce RTX 4080 SUPER through TensorFlow GPU/XLA.
- Memory growth: configured and verified before logical-device initialization.
- Frozen transport SHA-256:
  `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`.
- Target signature:
  `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`.
- Scalar, flat-batch, and combined status values/scores were finite and met the
  declared `1e-12` parity tolerance.
- Mean retained combined call for 64 logical draws: `0.070414 s`.

## Resource Preflight

Artifact:
`docs/plans/artifacts/bayesfilter-pp-ukf-ten-phase-repair-20260721-01/phase5-budget-preflight.json`

The preflight combined the new measured retained-health cost with the prior
observed repaired route of 100/1,000 transitions in approximately 2,820
seconds. Its conservative full-attempt projection was `31,444 s` (`8.73 h`).
This exceeds both the four-hour campaign cap and the `<=75%` launch margin.

The retained-health repair is real, but it is not the dominant remaining cost.
The HMC/bootstrap path still controls feasibility.

## Canary

Artifacts:

- `docs/plans/artifacts/bayesfilter-pp-ukf-ten-phase-repair-20260721-01/phase7-canary/run_manifest.json`
- `docs/plans/artifacts/bayesfilter-pp-ukf-ten-phase-repair-20260721-01/phase7-canary/tuning/hmc_kernel_tuning_result.json`

The canary used diagnostic budgets, fixed identity, GPU/XLA, memory growth, a
1,500-second timeout, a 30-second heartbeat, one attempt, and no sampling.

- Bootstrap: three rounds, oscillatory above/below acceptance behavior,
  `repair_budget_exhausted`, no acceptance-promoted kernel.
- Windowed diagnostic budget: 32 transitions completed across five segments.
- Terminal status: `hard_veto`.
- Windowed-stage vetoes included HMC/runtime and required telemetry validity
  failures. No fixed-mass step selection or verification was reached.
- The only recorded mass event was the initial identity artifact. No mass or
  coordinate mutation was admitted.
- `sampling_launched=false`.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retained-health repair | Batch parity and one-call protocol | Passed | Performance at larger unseen batch sizes | Keep the repaired path and regression tests | No tuning or scientific claim |
| Admit PP-UKF tuning kernel | Acceptance-promoted bootstrap plus terminal valid tuning | Failed | Whether bootstrap bracketing/step policy can be repaired for PP-UKF | Diagnose bootstrap oscillation and windowed-stage runtime telemetry with a new bounded plan | No candidate promotion or rejection of PP-UKF mathematics |
| Launch full tuning | Projection within four-hour cap | Vetoed | Transition rate after future bootstrap/runtime repair | Do not run the 1,000-transition serious campaign now | No resource expansion implied |
| Launch sequential HMC | Admitted tuning artifact | Correctly closed | None | Remain closed until independent tuning admission | No posterior or convergence result |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Retained-health engineering gates passed; bootstrap promotion and windowed HMC/runtime gates failed |
| Statistically supported ranking | None |
| Descriptive-only differences | Target-call times, bootstrap acceptance relations, transition counts, and wall time |
| Default readiness | Not established |
| Next evidence needed | A bounded repair of bootstrap bracketing and windowed runtime telemetry, then a new rate projection and serious tuning attempt |

## Research Question Guardian

- Harness invalidated: no. The GPU/XLA benchmark and bounded canary emitted
  structured artifacts.
- Retained-health implementation invalidated: no. Parity and call-count gates
  passed.
- PP-UKF target or mathematics invalidated: no. The observed failures are
  tuning/bootstrap/runtime evidence, not a mathematical contradiction.
- Current tuning candidate admitted: no.
- Research direction rejected: no. The next discriminating work is the
  bootstrap/windowed runtime repair, not another full rerun.

## Post-Run Red Team

The strongest alternative explanation is that the canary's diagnostic route
used a geometry fallback after its non-promoting bootstrap, which serious
tuning now forbids. That route was retained only to measure downstream behavior
and cannot be promoted. The weakest evidence is the resource projection, which
uses a prior long-run transition rate rather than a successful repaired serious
run; this uncertainty does not authorize a full run because both the resource
gate and the bootstrap-promotion gate currently fail.
