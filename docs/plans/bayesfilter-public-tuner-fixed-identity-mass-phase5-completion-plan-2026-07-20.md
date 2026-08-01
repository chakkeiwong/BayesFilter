# Public-Tuner NeuTra Phase 5 Completion Plan

Date: `2026-07-20`

Status: `PLAN_READY_FOR_EXECUTION`

Parent plan: `docs/plans/bayesfilter-public-tuner-fixed-identity-mass-plan-2026-07-19.md`

## 1. Scope And Decision Boundary

This plan closes the remaining NeuTra campaign work after the authoritative
LGSSM Phase 4 second-seed result completed. It does not rerun or retune LGSSM.
It does not promote NeuTra as a repository-wide default and does not claim
universal posterior correctness, sampler superiority, filter exactness, or
production readiness.

The completion target is the four currently executable registry cells:

| Cell | Target route | Status |
| --- | --- | --- |
| `LGSSM-EXACT` | exact 18D deterministic LGSSM/Kalman target | complete in Phase 4; no rerun |
| `PP-UKF` | six-parameter predator-prey principal-square-root UKF | execute |
| `PP-SGQF` | six-parameter predator-prey fixed level-2 SGQF | execute |
| `SIR-SGQF` | three-parameter Austria SIR fixed level-2 SGQF | execute |
| `STR-UKF` | five-parameter structural principal-square-root UKF | execute |

The registry-blocked cells remain blocked inventory and must not be launched:
`SVX-SGQF`, `SVX-ZC`, `KSC-UKF`, `PP-ZC`, `STR-ZC`, `SIR-UKF`, and `SIR-ZC`.

## 2. Research Intent Ledger

| Field | Frozen decision |
| --- | --- |
| Main question | Do the remaining executable targets pass the same public fixed-identity NeuTra tuning, sampler-health, convergence, and truth-tail screens? |
| Candidate | Fresh target-specific batched dense-IAF transport followed by exact transformed-gradient HMC with public `tune_hmc_kernel` and `mass_policy="fixed_identity"`. |
| Comparator | Each cell's declared generating truth. Training loss, held-out reverse KL, acceptance, and runtime are explanatory diagnostics only. |
| Promotion criterion | Fresh final training, public fixed-identity tuning with acceptance in `[0.65, 0.75]`, valid sequential warm-up and retained diagnostics, no hard veto, and every declared parameter `p_truth >= 0.05`. |
| Hard vetoes | Target/transport signature drift, nonfinite target or transport values, non-batch-native training, identity-mass mutation, failed tuning acceptance, invalid status or energy telemetry, failed warm-up/retained R-hat or ESS caps, invalid artifact, output collision, or GPU/XLA/memory-growth contract failure. |
| Explanatory diagnostics | Recipe losses, held-out reverse KL, runtime, acceptance within the valid band, posterior moments, and tail magnitudes. They cannot establish correctness or rank candidates. |
| Marginal classification | `0.003 <= minimum p_truth < 0.05`; preserve the cell as evidence and nominate a later seed only under a new authorized continuation. |
| Severe classification | `minimum p_truth < 0.003`; preserve the negative result and do not retune on claim data. |
| Nonclaims | No universal NeuTra validity, distributional equivalence, sampler superiority, cross-model ranking, default readiness, or production readiness. |
| Artifact | Fresh versioned campaign root under `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/`, per-cell manifests/results, aggregate result, terminal review, and reset memo. |

## 3. Default And Assumption Audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- |
| Public tuner owns admission | Phase 3 migration and fixed-identity implementation | A private legacy tuner silently re-enters | Static route scan and tuning payload assertion | Required |
| `fixed_identity` in trained NeuTra coordinates | Owner policy and Phase 4 admitted replay artifact | Hidden empirical mass update changes the candidate | Mass policy/signature assertion across tuning handoff | Required |
| Target-specific recipes | Current registry recipes and prior target-specific protocols | Cross-model transfer is mistaken for a default | Disjoint screen/selection record per cell | Warm-start hypotheses only |
| Four 500-step screens and one 5,000-step final training | Parent campaign budget | Training cap hides an underfit transport | Progress, held-out status, frozen/trainable parity | Reviewed bounded budget |
| Four chains and sequential controller | Repository NeuTra HMC policy | Short fixed draws are misread as convergence | Warm-up and retained modern R-hat/ESS gates | Required |
| One seed per remaining cell | Owner cost policy | Stochastic tail miss is overinterpreted | Three-way truth-tail classification and explicit nonclaim | Diagnostic only |
| GPU/XLA, TF32, memory growth | Repository execution policy | Sandbox/device mismatch or unsafe allocation | Trusted GPU preflight and manifest fields | Required |
| Durable run state is diagnostic | Repair for stale-progress confusion | A progress snapshot is treated as scientific evidence | Terminal `result.json` precedence test | Required |

## 4. Evidence Contract And Stop Conditions

Before execution, verify the following exact contract:

1. Registry signatures, geometry hashes, recipes, seeds, and blocked inventory
   are unchanged from the inspected registry.
2. The fixed observed target data are shared because every stage targets the
   same posterior. Random training batches, common-heldout screening draws,
   final-training draws, tuner verification draws, warm-up draws, and retained
   claim draws use their declared stage-separated seeds. Heldout draws may
   nominate a recipe but cannot establish the downstream HMC claim.
3. The active route calls the public tuner and shared sequential controller;
   no private fixed-transport tuner or fixed-grid repair is admissible.
4. Each serious run records the commit, command, environment, target and
   transport hashes, GPU/memory-growth/XLA/TF32 settings, seeds, wall time,
   artifact paths, and plan path.
5. A completed `result.json` and manifest are authoritative over an older
   `run_state.json`; an incomplete root is never posterior evidence.

Stop the campaign for shared harness invalidity, target-signature drift,
artifact collision, GPU/XLA or memory-growth failure, a material contract or
budget change, or exhausted per-cell repair budget. A valid candidate failure
does not stop independent later cells.

## 5. Compute And Retry Budget

Hardware class is one visible NVIDIA GPU, sequential execution, TensorFlow/TFP,
float64 target evaluation where declared by the runner, XLA enabled, TF32
enabled, and memory growth configured before logical-device initialization.

Per cell:

- at most four 500-step recipe screens;
- one selected 5,000-step final training;
- one public native tuning call over the declared leapfrog ladder;
- up to 10,000 warm-up and 10,000 retained draws per chain;
- one localized infrastructure retry with the same target, method, seeds,
  hardware class, criteria, and budget.

No retry may retune on claim data, change the target, transfer a selected
setting from another cell as a promoted default, or overwrite an artifact.

## 6. Implementation Work Before Compute

1. Add atomic orchestration `run_state.json` records for launch, target/geometry
   validation, screen, final training, tuning admission, warm-up, retained
   sampling, terminal result, and exception paths. The state is diagnostic and
   must contain phase, status, timestamp, PID, cell, output root, last durable
   artifact, and exception/return information when available.
2. Update the campaign wrapper to stream each child stdout/stderr to a durable
  per-cell log under `launch-logs/`, write child PID/return code state, and classify a
   missing result only after the child has exited. Preserve child output tails
   in launch-failure records.
3. Add CPU-safe tests for terminal-result precedence, fresh-root enforcement,
   run-state atomicity, streamed child classification, public-tuner ownership,
   and blocked-cell exclusion.
4. Reconcile the stale Phase 4 blocked note with the completed second-seed
   artifact; preserve it as historical evidence and explicitly supersede it.

## 7. Execution Sequence

### Phase A: engineering repair

Run focused tests, compile checks, and a route/registry audit. Do not launch
serious sampling until these pass.

### Phase B: trusted preflight

Run escalated `nvidia-smi` and the repository TensorFlow GPU/XLA memory-growth
probe. Record the result under the fresh Phase 5 campaign root.

### Phase C: sequential cell execution

Launch `PP-UKF`, `PP-SGQF`, `SIR-SGQF`, and `STR-UKF` sequentially from a host
terminal, SSH session, host `tmux`, or equivalent host process boundary. The
campaign root must be fresh and versioned. Use streamed logs and durable state.

### Phase D: terminal interpretation

For each cell, classify hard vetoes first, then sampler validity, then truth
tail. Produce an aggregate decision table and inference-status table with rows
for hard veto screen, statistically supported ranking, descriptive-only
differences, default readiness, and next evidence needed. Do not rank cells
from one-seed descriptive metrics.

## 8. Required Closeout Artifacts

- this plan and its audit record;
- Phase 4 reconciliation result and reset memo;
- Phase 5 campaign `registry.json`, preflight artifact, per-cell `run_state.json`,
  `launch-logs/<cell>.log`, manifests, results, archives, and aggregate result;
- terminal review with post-run red-team note;
- final reset memo naming viable, marginal, failed, and blocked cells;
- explicit statement of what was not concluded.

## 9. Audit Verdict

`PASS_FOR_IMPLEMENTATION_AND_EXECUTION`.

The plan has a fixed scientific question, exact baseline and promotion
criteria, explicit vetoes and nonclaims, bounded compute, fresh artifact roots,
target-specific settings, blocked-cell handling, and a repair limited to
observability and process supervision. The plan does not authorize a new
scientific direction, a default change, or a rerun of the completed LGSSM
second seed.
