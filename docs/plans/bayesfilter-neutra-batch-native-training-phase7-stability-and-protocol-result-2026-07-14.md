# Phase 7 Result: Fresh Stability And Target-Specific Protocol

Date: 2026-07-14

## Outcome

**PASS_PHASE7_AND_NOMINATE_WITHOUT_RANKING.** All fresh batch-binding-v2
smokes, the 100-step stability rung, and all four 500-step screen arms passed
their engineering and numerical vetoes. The predeclared deterministic proxy
rule nominates `wide_2x_lr5e3` for fresh long-budget training. This is not a
statistically supported ranking and does not establish transport quality,
posterior correctness, HMC convergence, or default readiness.

## Research Intent Ledger

| Role | Phase 7 definition | Result |
| --- | --- | --- |
| Main question | Is the batch-native route stable enough for recipe screening, and which recipes remain viable? | yes; all four remain viable |
| Promotion criterion | none in this phase; the screen only nominates | no promotion performed |
| Promotion veto | invalid target status, nonfinite loss/gradient, wrong GPU/XLA/binding identity, stale/corrupt artifact | none fired |
| Continuation veto | target/harness invalidity, missing diagnostics, or exhausted budget | none fired |
| Repair trigger | localized harness/finalization failure inside unchanged scientific contract | finalizer admission audit repaired before use |
| Explanatory diagnostics | losses, gradient norms, heldout means/MCSEs, and runtimes | descriptive only |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` with a dirty multi-lane worktree |
| Environment | TensorFlow `2.19.1`, CUDA build, RTX 4080 SUPER, driver `591.86` |
| Device/XLA | trusted GPU/XLA, `float64`, batch `128`, one compiled training invocation per job |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Target | exact T=120, 18-parameter LGSSM, binding schema `bayesfilter.neutra.batch_native_target_binding.v2` |
| Binding closure | `6d5a05a65a15b5fb4378fc08547d5dfd22dc83705d31e7fd662b142df04732b5` |
| Smoke root | `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/fresh-protocol` |
| Stability root | `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/stability-100` |
| Screen root | `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/screen-500` |
| Finalizer command | `CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/finalize_lgssm_neutra_batch_native_screen.py --smoke-root .../fresh-protocol --screen-root .../screen-500` |

The CPU-hidden finalizer is a standard-library artifact inspection step. It did
not run training or target evaluation.

## Execution Results

All four five-step smokes passed with valid exact-target status, no floors or
fallbacks, GPU placement, XLA enabled, and binding v2.

The 100-step source-anchor stability job passed all vetoes:

- compiled program time: `18.9464 s`, or `0.1895 s/step`;
- total wall time: `205.9148 s`;
- heldout mean reverse KL: `74.7978`;
- heldout MCSE across eight common batches: `0.0923`.

All four 500-step arms passed:

| Recipe | Heldout mean | Heldout MCSE | Program time | Total wall |
| --- | ---: | ---: | ---: | ---: |
| `source_anchor_lr5e3` | 72.9451 | 0.0866 | 93.64 s | 279.23 s |
| `lower_lr1e3` | 74.3471 | 0.0850 | 88.46 s | 267.67 s |
| `shallow_2stage_lr5e3` | 73.3392 | 0.0775 | 86.09 s | 258.97 s |
| `wide_2x_lr5e3` | 72.7417 | 0.0776 | 82.99 s | 247.53 s |

The wide recipe's paired difference versus the source anchor was `-0.20345`
with paired MCSE `0.01322`. Under the predeclared deterministic selection rule,
that nominates `wide_2x_lr5e3`; it does not convert the eight-batch proxy into a
scientific ranking.

## Finalizer Repair

The initial unrun finalizer checked top-level pass/status and binding schema but
did not fail closed on enough identity fields. Before it was executed, the
finalizer was repaired to verify:

- exact job, recipe, seed, step, target, adapter, and campaign-contract identity;
- source artifact self-hashes plus checkpoint/progress/payload file hashes;
- consistent binding dependency closure across all eight rows;
- GPU placement, XLA, batch 128, one compiled invocation, and no fallback;
- NumPy/host-callback closure audit, progress cadence, and finite diagnostics;
- exact heldout seed order, status, mean, and MCSE recomputation; and
- immutable result writing with no overwrite.

Focused validation passed before finalization. A later Phase 8 policy audit
found that the first executed finalizer indirectly imported the diagnostic
parent campaign, and therefore NumPy, only to read smoke seeds. That first JSON
pair is preserved under `finalization-attempts/attempt-01-rejected-policy` and
is ineligible for handoff. The finalizer was repaired to import a frozen
standard-library seed constant, a subprocess test now proves its import leaves
both NumPy and TensorFlow unloaded, and accepted attempt 02 binds the finalizer
source file hash into its result. The nomination did not change.

Claude returned no output on two
minimal trusted health probes, so no substantive Claude verdict was available.
Under the repository proportional-review policy, reviewer unavailability was
recorded and the completed skeptical local review remained sufficient.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close Phase 7 | all fresh required jobs and finalization checks passed | no hard or continuation veto | screen is short and proxy-only | institutionalize the route and prepare fresh long-run handoff | no posterior/HMC/default claim |
| Nominate `wide_2x_lr5e3` | predeclared deterministic proxy rule applied | all candidates viable | ranking uncertainty not established by eight batches | two fresh 5,000-step seeds under a separate campaign budget | not best or superior |
| Preserve other recipes | all passed hard screens | none rejected | target-specific search is narrow | retain as viable alternatives | no causal architecture/LR conclusion |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | all four candidates passed |
| Statistically supported ranking | none |
| Descriptive-only differences | heldout means/MCSEs, losses, gradients, and runtimes |
| Default readiness | false |
| Next evidence needed | two fresh 5,000-step seeds, frozen-transport validation, and downstream posterior/HMC checks |

## Artifacts

- accepted final screen artifact hash: `sha256:e3f129316f2561c225ea47042b604085ecfd2664f835ccc74bcec3cec15840d5`;
- accepted selected recipe artifact hash: `sha256:00bf189dd8697c33b0378bda92a75d2df74d85ffb0e754f1df7c6dabcb216ac0`;
- accepted screen result file SHA-256: `c1baa7208d549fd905cb5f9beab876041201c9a6ea7953f1a4df68b60ca3e797`;
- accepted selected recipe file SHA-256: `1984c33142496ecbbd77ecaea17b1d3dc3320caa45a1b08aa947439ca7088c97`;
- finalizer source file SHA-256 bound into attempt 02: `856f40c0305cd132d60d3d7dfccfada11086c844d49ea3d27868c697ec7224e6`.

## Handoff

Phase 8 starts under
`docs/plans/bayesfilter-neutra-batch-native-training-phase8-institutionalization-closeout-subplan-2026-07-14.md`.
The 5,000-step training launch is not part of Phase 8 execution.
