# KSC-UKF Gaussian-Sum Broad-Grid Tuning Result (attempt04)

Date: 2026-08-03  
Plan: `docs/plans/bayesfilter-ksc-ukf-neutra-hmc-continuation-plan-2026-08-02.md`  
Reset memo: `docs/plans/bayesfilter-ksc-ukf-neutra-hmc-reset-memo-2026-08-03.md`

## Decision Table

| Field | Status |
| --- | --- |
| Decision | `BROAD_GRID_TUNING_VIABLE_PAIR_SET`; complete viable pair set produced; sequential handoff gate **not** satisfied |
| Primary criterion | passed: 6/6 primaries and 1/1 planned coverage guard completed; exactly one viable primary `(L=25, eps~0.9896)` |
| Veto diagnostics | no hard rejections on any pair: no divergences, nonfinite values, movement, resonance, or telemetry failures |
| Main uncertainty | whether the handoff gate should count the compatible `L=24` coverage probe as a second candidate (union semantics, implemented) or as robustness evidence for the `L=25` primary (role semantics) |
| Next justified action | owner decision on the handoff contract before any sequential launch (options below) |
| Not concluded | no convergence, posterior-correctness, superiority, or default-readiness claim; no ranking among viable pairs |

## Run Manifest Summary

- Command: memo-specified broad-grid relaunch, unchanged; recorded in
  `broad-grid-attempt04/KSC-UKF-GAUSSIAN-SUM-T20/run_manifest.json`.
- Git commit: `efce62b5` (route, evidence, and manifest provenance fixes committed before launch).
- Environment: conda `tf-gpu`, TensorFlow GPU `/GPU:0` (RTX 4080 SUPER), XLA on, TF32 on, verified memory growth.
- Root seed: `(20260803, 2881)`; frozen transport SHA-256 `dbbaba37…e8bddb` verified at launch; live target signature `727718ec…` matched.
- Wall time: 159 s. Prior attempts 01–03 were blocked before process creation (no artifacts); attempt04 is the first execution.
- Tuning artifacts: `broad-grid-tuning/private_result.json`
  (`f2076c02…5232ec`), `broad-grid-tuning/public_result.json` (`5afa1920…c70ea9`).
  The private file is the only valid sequential-handoff input (the public file
  hides epsilon values and next-round candidates by design).

## Per-Pair Evidence (descriptive)

| L | tuned eps | 90% replication interval (acceptance) | Disposition |
| --- | --- | --- | --- |
| 3 | 1.0720 | (0.906, 0.943) | needs_higher_epsilon |
| 5 | 1.1070 | (0.846, 0.883) | needs_higher_epsilon |
| 9 | 1.0143 | (0.816, 0.888) | needs_higher_epsilon |
| 13 | 0.9582 | (0.872, 0.897) | needs_higher_epsilon |
| 18 | 0.9797 | (0.803, 0.846) | needs_higher_epsilon |
| **25** | **0.9896** | **(0.730, 0.816)** | **provisional_viable (primary)** |
| 24 (coverage, inherits eps) | 0.9896 | (0.717, 0.803) | provisional_viable (guard) |

Practical band (0.65, 0.75); classification is the reviewed
`replication_mean_t90_band_compatibility_v1` heuristic, three replications,
four chains, 65 discarded draws per screen.

## Why Sequential HMC Was Not Launched

The reviewed handoff gate in `run_neutra_broad_grid_sequential_cell` requires
the complete unranked primary-plus-coverage union (`next_round_candidates`) to
contain exactly one entry. Here the union has two: the `L=25` primary and its
compatible `L=24` same-epsilon coverage probe. Launching would fail closed at
`sequential handoff requires exactly one unranked viable pair`. The reset
memo's stop branch therefore fired: terminal result recorded, no sequential
launch, owner decision required.

The memo sentence "exactly one primary pair survives" is ambiguous against the
implemented union semantics; the implemented contract is authoritative for this
result. The coverage probe is not an independently tuned pair — it inherits the
primary's epsilon and exists as one-hop robustness evidence whose
"compatibility is never used to remove a compatible primary" (tuner docstring).
Whether it should be allowed to block the primary's handoff is exactly the open
contract question.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | all pairs pass; no hard rejection anywhere |
| Statistically supported ranking | none; `L=25` vs `L=24` intervals overlap heavily and no ranking is claimed |
| Descriptive-only differences | acceptance means/intervals per L, tuned epsilons, needs_higher_epsilon nominations for `L<=18` |
| Default readiness | not assessed |
| Next evidence needed | owner contract decision; then sequential HMC from this artifact, or a widened/narrowed grid round |

## Post-Run Red Team

- Strongest alternative explanation: the unique viable primary sits at the top
  edge of the grid (`L=25` is the maximum, and `L=26` was never probed because
  it lies outside the grid hull). With every tuned epsilon near 1.0, screen
  acceptance falls monotonically-ish with L, so `L>=26` pairs might also be
  compatible; "uniqueness" may partly reflect grid truncation rather than a
  distinguished optimum. The same edge survivor (`L=25`) occurred in the SVX-ZC
  campaign — a nomination signal, descriptive only.
- The compatible `L=24` probe supports a plateau reading (robust region), which
  weakens any knife-edge concern but is exactly what blocks the union gate.
- Screens use one burn-in step from fixed offset starts; screen acceptance can
  differ from stationary acceptance. This is the reviewed generic policy with
  declared limitations, descriptive only.

## Options For The Owner Decision

1. Amend the handoff contract so the gate requires exactly one viable
   **primary** (role-based, deterministic), with coverage-probe compatibility
   recorded as robustness health evidence that cannot block its parent. Then
   launch sequential HMC from this same attempt04 artifact. Requires a code
   change to the gate, a contract-test update, and a one-paragraph plan
   amendment; no retuning and no new tuning data.
2. Keep the union gate and run a follow-up grid round (for example, extend the
   L grid upward given the edge survivor, or a narrowing protocol) under a new
   plan section and fresh root.
3. Stop the KSC campaign at broad-grid evidence.

No option is authorized by this note; the choice changes a reviewed contract or
budget and belongs to the owner. The NeuTra route-ledger repair recorded in the
reset memo remains mandatory before any claim-bearing sequential result is
accepted, under every option.
