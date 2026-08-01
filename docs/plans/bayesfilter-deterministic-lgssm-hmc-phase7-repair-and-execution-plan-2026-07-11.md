# BayesFilter Deterministic LGSSM HMC Phase 7 Repair And Execution Plan

Date: 2026-07-11

Status: `BLOCKED_P7G_REFRESH_HASH_CONTRACT_MISMATCH`

## Purpose

Close the procedural gaps between the passed deterministic LGSSM Phase 6AA
kernel screen and runbook Phase 7, then execute Phase 7 under fixed rules. This
plan does not reopen the LGSSM target, prior, fixture, XLA score route, geometry,
mass-tuning policy, or HMC kernel-selection policy.

The current user instruction to execute this reviewed plan is the explicit
runtime approval for the plan-scoped Phase 6 replay-artifact refresh, tiny
Phase 7 smoke, and serious Phase 7 burn-in/retained-sampling run. It is not
approval for Phase 8 or NeuTra training.

## Execution Outcome

Execution reached P7G and stopped at the predeclared exact-hash continuation
veto. The Phase 6 private-replay refresh exited successfully and its own
kernel-tuning gate passed, but it did not reproduce the pinned public kernel,
private-loop kernel, or selected-trajectory hashes. The selected-step hash and
all pinned target/config/fixture/XLA/geometry/mass/adapter inputs remained
unchanged.

Inspection of the committed and refreshed private event artifacts found equal
selected HMC mechanics in the fields available in both artifacts. The hash
change is associated with current `handoff_screen_policy` provenance entering
the hashed stage payload and cascading through the trajectory and final-kernel
hashes. That evidence supports an engineering hash-contract/baseline-migration
blocker, not a sampler or target failure. It does not authorize silently
repinning the plan because exact hash equality was an explicit continuation
gate and the old full private replay payload does not exist for complete
byte-for-byte comparison.

P7H tiny smoke, P7I serious Phase 7, Phase 8, and NeuTra training were not
executed. The close record is
`docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase7-burnin-sampling-result-2026-07-09.md`.

## Authority And Existing State

- Runbook:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md`
- Phase 7 subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase7-burnin-sampling-subplan-2026-07-09.md`
- Phase 6AA result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6aa-svd-score-wiring-retry-result-2026-07-10.md`
- Existing deterministic config:
  `docs/benchmarks/configs/multidim_lgssm_serious_hmc_tuning_2026_07_09.json`
- Existing kernel artifact:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`

Pinned entry values:

| Field | Value |
| --- | --- |
| Git baseline | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Deterministic config hash | `sha256:1b5683e2f210e3976fca712ec2970f8327831596c0b67776316efbd0b6b46729` |
| Existing kernel artifact hash | `sha256:f8c94073b60a6458538537317e49ed683ad0c94b525cafc77cc4d01822badaa2` |
| Existing public final-kernel hash | `8ddf25a3b572893e19e814fad5ca5b6150718e36f760c159b47db1231d92ffff` |
| Existing private-loop kernel hash recorded by public handoff | `391558a9b5f4cdc1b9dff9a5e9bceba668dedded7298c1d8c76daea42f42039a` |
| Selected step hash | `ec7db59e51465eee95658167e1f7596e21d9ab0efdac11f54c2d397aa270ab40` |
| Selected trajectory hash | `6eaf7a563353b278a71dcfbe2515fda6d46c47ab2e38996b6b61fab1bbbd13b3` |

Concurrent LEDH/QR work is user-owned and out of scope. This plan must not edit,
revert, format, stage, or interpret those files.

## Direct Classification Of The Gap

Claimed Phase 7 input: an executable frozen HMC kernel selected by Phase 6.

Quantity currently persisted: a non-replayable public summary containing
kernel, step, trajectory, and mass hashes but no step size, leapfrog count, or
adapted-mass arrays.

Verdict: the Phase 6AA kernel screen is passed, but the persisted object is
wrong relative to the claimed executable Phase 7 input. The full private
kernel existed in memory during Phase 6 and BayesFilter already has a checked
private replay API; the serious driver discarded the private replay form when
serializing. This is an engineering handoff defect, not evidence against the
target or sampler.

## Research Intent Ledger

| Field | Declaration |
| --- | --- |
| Main question | Can the fixed Phase 6AA LGSSM HMC kernel complete deterministic burn-in and retained sampling under predeclared all-parameter R-hat and ESS gates? |
| Candidate/mechanism | The exact SVD/eigh XLA target and exact frozen kernel selected by the deterministic Phase 6 tune/verify/repair program. |
| Exact baseline/comparator | The existing Phase 6AA hashes and the unchanged `T=120`, 18-parameter prior-mean-truth fixture. There is no sampler-ranking comparator in Phase 7. |
| Expected failure mode | Slow mixing produces high rank-normalized split/folded R-hat or low bulk/tail ESS at a configured check or cap. |
| Promotion criterion | Phase 7 retained sampling passes every parameter's R-hat, bulk ESS, and tail ESS thresholds with finite chains and confirmed XLA execution. |
| Promotion veto | Any required diagnostic fails at the retained cap; Phase 8 must not start. |
| Continuation veto | Target/config/kernel hash mismatch, nonfinite state/sample/target/log-accept value, available nonzero divergence telemetry, XLA/JIT fallback, worker/process failure, artifact corruption, manual threshold/kernel/chain change, or machine wall-time cap. |
| Repair trigger | Test collection failure, replay serialization failure, worker orchestration failure, compile/resource failure, or a structured diagnostic cap failure. A diagnostic cap failure may motivate a separately reviewed tuning repair but cannot be repaired inside this run. |
| Explanatory diagnostics | Acceptance, per-check trajectories, runtime, compile count/time, worker resource provenance, posterior summaries reserved for Phase 8. |
| Must not be concluded | No posterior recovery until Phase 8, no sampler superiority, no production/default readiness, no GPU readiness, no DSGE claim, no NeuTra-training claim, and no broad scientific validity claim. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific/engineering question | Can fixed-kernel burn-in and retained sample count be governed without agent tuning and produce a Phase 8-ready retained artifact? |
| Baseline | The pinned Phase 6AA result and unchanged deterministic config, fixture, target, geometry, and mass inputs. |
| Primary pass criterion | A public Phase 7 artifact records `passed=true`; every parameter has rank-normalized split/folded `R_hat <= 1.01`, bulk ESS `>= 1000`, and tail ESS `>= 400`; all hard-veto checks pass; the private retained-sample hash verifies. |
| Promotion vetoes | High/nonfinite R-hat, low/nonfinite bulk or tail ESS at the retained cap, or missing all-parameter diagnostics. |
| Continuation vetoes | The continuation-veto list in the research intent ledger. |
| Explanatory only | Acceptance rate, check-to-check diagnostic movement, runtime, memory, compile timing, descriptive posterior means/SDs, and individual check values before the final pass/fail check. |
| What passing will not prove | Posterior truth recovery, calibrated uncertainty, sampler superiority, production readiness, default readiness, or model adequacy. |
| Preserving artifacts | This plan, reviewed plan verdict, Phase 7 config, refreshed public Phase 6 artifact, ignored private replay payload plus public hash, Phase 7 public JSON, ignored private retained samples plus public hash, log, result note, execution ledger, and run manifest. |

## Fixed Statistical Definitions

All diagnostics are computed in the 18 raw model-parameter coordinates after
mapping samples out of both HMC mass-coordinate transforms. Diagnostics on the
final latent HMC coordinates cannot satisfy the all-parameter contract.

For a draw-by-chain-by-parameter tensor:

1. Pool draws and chains separately for each parameter and assign average ranks
   to ties.
2. Rank-normalize with the Blom transform
   `z = Phi^-1((rank - 3/8) / (S + 1/4))`, where `S` is the pooled sample count.
3. Compute split R-hat on the rank-normalized values. Split each chain into its
   first and last `floor(draw_count / 2)` draws; discard the middle draw when
   the count is odd.
4. Compute folded rank-normalized split R-hat by replacing draws with
   `abs(draw - pooled_median)` before ranking and splitting.
5. The reported R-hat is the elementwise maximum of rank-normalized and folded
   rank-normalized split R-hat.
6. Bulk ESS is cross-chain ESS of the same split-chain rank-normalized tensor
   used for R-hat, using the initial positive-pair autocorrelation truncation
   implemented by TFP.
7. Tail ESS is the elementwise minimum of cross-chain ESS for the split-chain
   indicators `draw <= pooled_q05` and `draw >= pooled_q95`, using the same
   positive-pair truncation. Quantiles are linear-interpolation pooled
   quantiles computed before splitting.
8. Nonfinite input or diagnostic values fail closed. Every one of the 18
   parameter entries must pass; aggregate averages cannot substitute.

TensorFlow/TFP owns the diagnostic implementation. SciPy may be used only in an
independent test/reference calculation, not in the implementation path. The
rank/ESS reductions run after each XLA HMC chunk; they are not a non-XLA target
or sampler fallback.

### Burn-In Rule

- Four chains total.
- Initial burn-in: 2,000 transitions per chain.
- Check window: the latest 1,000 transitions per chain.
- Extension: 1,000 transitions per chain.
- Cap: 16,000 transitions per chain.
- At each check, apply all three diagnostic gates to only the latest fixed
  1,000-draw window. All burn-in diagnostic draws are discarded.
- Passing burn-in preserves only the final chain states for retained sampling.

This fixed-window choice avoids allowing early nonstationary burn-in draws to
dominate later checks and avoids silently treating burn-in draws as posterior
samples.

### Retained-Sampling Rule

- Start with 4,000 retained draws per chain.
- Check every 2,000 additional retained draws per chain.
- Accumulate all retained draws from the post-burn-in chain state.
- Cap at 40,000 retained draws per chain.
- No thinning, chain exclusion, threshold change, or manual extension.

## Deterministic Runtime Contract

| Field | Fixed value/policy |
| --- | --- |
| Device | CPU only, with `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import in every worker |
| Worker model | Two persistent spawned workers, two chains per worker |
| Total chains | Four, with stable global chain order `0..3` |
| HMC target | Existing TensorFlow/TFP SVD/eigh graph-status LGSSM posterior |
| HMC compilation | `tf.function(..., jit_compile=True)` only |
| Kernel | Exact private Phase 6 replay payload; no reconstruction from logs or hash guessing |
| Initial dispersion | Existing deterministic sequential-verifier pattern: four offsets evenly spaced from `-0.15` to `0.15` in final HMC coordinates, with alternating parameter signs |
| Root seed | `(20260711, 701)` |
| Chunk seeds | `(root0 + 100000 * stage_index + 1009 * check_index + 37 * worker_index, root1 + 10000 * stage_index + 101 * check_index + 17 * worker_index)`, with burn-in `stage_index=1`, retained sampling `stage_index=2`, and the discarded compile probe using burn-in `check_index=9999` so no executed chain chunk reuses its seed |
| CPU threads | Each worker records and receives `TF_NUM_INTRAOP_THREADS=8`, `TF_NUM_INTEROP_THREADS=1`, `OMP_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` on the 16-core/32-thread host |
| Worker state | Retained in memory across chunks within one process; no process-pool task migration |
| Resume | Not supported. Existing BayesFilter checkpoint inspection explicitly defers continuation. Interruption writes a blocker and requires a fresh run. |
| Machine wall-time cap | Eight hours for serious Phase 7. This is machine protection, not a scientific diagnostic. |

## Private/Public Artifact Boundary

The existing public tuning artifact remains non-replayable. A Phase 6 refresh
will additionally write an ignored private replay JSON containing the existing
BayesFilter private loop payload with adapted-mass arrays and mechanics. The
public artifact records only hashes and bounded provenance.

The Phase 7 run writes raw retained samples and final worker states only under
ignored `private_diagnostics`. The public Phase 7 JSON may contain parameter
names, per-parameter R-hat/ESS, finite/veto summaries, opaque artifact IDs,
hashes, byte counts, worker counts, seeds, PIDs, versions, and timing. It must
not contain raw samples, raw states, mass arrays, step size, leapfrog count, or
private paths.

## Phased Execution

### P7A: Governance And Plan Review

- Write this plan.
- Run the smallest one-path Claude read-only review.
- Revise until `VERDICT: AGREE`, or record bounded substitute review only if
  Claude is unavailable after the governed probe procedure.
- Do not edit implementation or execute HMC before this gate passes.

Gate: `PASS_PLAN_REVIEW`.

### P7B: Test Collection Repair

- Add the minimal local package marker required so `tests.*` resolves to this
  repository instead of the unrelated environment package.
- Run collect-only and focused public-API tests CPU-hidden.

Gate: the previously failing test module collects and focused tests run.

### P7C: Diagnostic Implementation

- Add the fixed rank-normalized split/folded R-hat and bulk/tail ESS reducer.
- Add synthetic tests for iid chains, shifted/nonmixing chains, autocorrelated
  chains, ties, odd draw counts, nonfinite values, coordinate ordering, and
  all-parameter threshold evaluation.
- Cross-check small fixtures against an independent reference calculation.

Gate: deterministic unit tests pass and no proxy replaces a required metric.

### P7D: Private Replay Serialization

- Extend only the serious driver serialization boundary to write the existing
  private `HMCKernelTuningResult`/loop replay form with full mass arrays.
- Bind it to config, fixture, target, geometry, mass, public kernel, and private
  loop hashes.
- Add tamper/mismatch and public-redaction tests.
- Add a narrow ignore rule for private replay and sample files.

Gate: round-trip through
`build_retained_frozen_kernel_hmc_adapter_from_tuning_payload` succeeds on a
test fixture; public artifacts remain redacted.

### P7E: Multicore Controller And Driver Stage

- Add a separate, versioned Phase 7 execution config pinned to the Phase 6
  inputs and thresholds.
- Add a `burnin_sampling` stage; keep JIT on by default with no non-JIT flag.
- Use two persistent spawned workers and exact state handoff between chunks.
- Map worker samples back to raw parameters before diagnostic aggregation by
  applying the final adapted-mass adapter's `latent_to_position` followed by
  its Phase 4/base mass adapter's `latent_to_position`. Assert the terminal
  adapter is the original 18-parameter LGSSM posterior adapter; one transform
  or an unknown transform depth is a hard veto.
- Implement fixed burn-in and retained rules, private sample persistence,
  public progress/result artifacts, structured errors, and the wall-time cap.

Gate: policy, seed, state-handoff, worker-partition, cap, artifact, and fail-closed
tests pass.

### P7F: Static, Unit, Integration, And Tiny Smoke Gates

Run CPU-hidden:

```text
python -m py_compile <touched Python files>
python -m pytest --collect-only -q tests/test_hmc_kernel_tuning_public_api.py
python -m pytest -q <focused Phase 7 tests>
git diff --check -- <Phase 7 touched files>
rg -n "GradientTape|jit_compile\s*=\s*False|jit_compile=False" <active Phase 7 runtime files>
```

Gate: all static and focused checks pass.

### P7G: Deterministic Phase 6 Replay-Artifact Refresh

Run the existing serious deterministic kernel-tuning stage once with the new
private serializer. This refresh is required because the current artifacts did
not preserve a signed private loop payload.

Before Phase 7, require exact equality of:

- deterministic config hash;
- fixture, XLA, geometry, and mass input hashes;
- selected step hash;
- selected trajectory hash;
- public final-kernel hash;
- private-loop final-kernel hash.

Any mismatch is a continuation veto. Do not choose between old and new kernels
manually and do not continue based only on similar acceptance.

Gate: `PASS_PRIVATE_REPLAY_REFRESH_HASH_MATCH`.

### P7H: Tiny Actual-Target Multicore XLA Smoke

Run a tiny actual-target, two-worker, four-chain XLA smoke with bounded
transitions and a separate `/tmp` output. It must exercise replay, worker
startup, XLA compilation, state handoff, the two-step raw-space mapping,
diagnostics, and structured closeout. It is engineering evidence only.

Gate: the smoke passes all engineering checks; no serious diagnostic or
scientific conclusion is inferred from it.

### P7I: Serious Phase 7 Execution

- Re-run the full preflight and artifact hash validation.
- Launch the serious CPU-hidden two-worker controller.
- Inspect bounded public progress only; full logs remain artifacts.
- Stop automatically at pass, configured diagnostic cap, hard veto, worker
  failure, artifact failure, or machine wall-time cap.

Gate to Phase 8: retained-sampling diagnostics pass all thresholds and the
private retained artifact verifies. Phase 8 remains unexecuted.

### P7J: Result, Manifest, Red-Team, And Handoff

- Write the required Phase 7 result note and update the visible execution
  ledger/runbook status.
- Include the run manifest, decision table, inference-status table, engineering
  correctness ledger, sampler-validity ledger, and scientific-interpretation
  ledger.
- Add a post-run red-team note: strongest alternative explanation, what would
  overturn the decision, weakest evidence, and exact next justified action.
- Run a one-path Claude result review when available before claiming a pass.

## Skeptical Plan Audit

| Risk | Audit result and control |
| --- | --- |
| Wrong baseline | Controlled by pinning the exact Phase 6AA config/input/kernel hashes. |
| Proxy promoted | Controlled: acceptance, smoke, compile success, and early checks are explanatory or engineering gates only. |
| Missing stop condition | Controlled by diagnostic caps, hard vetoes, process/artifact vetoes, and an eight-hour machine cap. |
| Unfair comparison | Not applicable: Phase 7 does not rank samplers or candidates. |
| Hidden diagnostic choice | Repaired by fixing rank, split/fold, quantile, ESS, window, aggregation, and nonfinite rules before runtime. |
| Stale context | Controlled by preflight re-hashing and branch/worktree recording. Concurrent LEDH changes are excluded. |
| Environment mismatch | Controlled by spawned-worker CPU hiding, TensorFlow/TFP version capture, XLA metadata, and no non-JIT fallback. |
| Artifact cannot answer question | Repaired by persisting exact private replay, raw-space retained samples, per-check required diagnostics, and hashes. |
| Public/private leak | Controlled by ignored private files, redaction tests, and public forbidden-field scans. |
| Run passes misleadingly | Controlled by all-parameter gates, raw-space diagnostics, folded R-hat, tail ESS, and Phase 8 recovery remaining separate. |

Audit verdict after substitute review:
`PASS_CONDITIONAL_ON_P7B_P7H_ENGINEERING_GATES`.
The previously proposed immediate Phase 7 runtime failed this audit because the
driver, executable handoff, required diagnostics, and multicore controller were
missing. This revised plan must not execute serious chains until those repairs
pass.

## Pre-Mortem

| How the run could mislead or fail | Cheap discriminator/control |
| --- | --- |
| Public hash is mistaken for an executable kernel | Replay API must validate the private payload and adapter/mass signatures. |
| Diagnostics pass in whitened coordinates but fail model parameters | Transform to raw parameter coordinates before every diagnostic. |
| Ordinary R-hat hides scale/tail problems | Use max rank-normalized/folded split R-hat plus bulk and tail ESS. |
| Burn-in appears worse forever because old draws accumulate | Use only the latest fixed predeclared burn-in window and discard it. |
| Workers restart and silently lose chain state | Persistent processes plus state-handoff unit/integration tests. |
| Multiprocessing duplicates seeds or chains | Predeclared seed function and global chain-index assertions. |
| A process crash leaves a plausible partial artifact | Atomic public writes, completion markers, worker return checks, and fail-closed status. |
| Phase 6 refresh selects a different kernel | Exact selected-hash and final-hash equality veto before Phase 7. |
| XLA silently falls back | Metadata and source scans must show JIT true and no fallback route. |
| Short-chain pass is overclaimed | Phase 7 only advances retained samples to Phase 8; recovery and broader claims remain untested. |

## Planned Run Manifest Fields

- Git commit and dirty-worktree paths, separated into in-scope and concurrent
  out-of-scope changes.
- Exact commands.
- Conda/environment identity, Python, TensorFlow, and TFP versions.
- CPU-hidden status, visible devices, worker count, PIDs, thread settings, and
  XLA/JIT status.
- Target/config/fixture/kernel signatures and hashes.
- All random seeds and deterministic seed derivation policy.
- Start/end time and wall time.
- Public and private artifact hashes, sizes, and opaque IDs.
- Plan, review, result, ledger, and log paths.

## Decision And Inference Tables Required In The Result

Decision table rows:

- Phase 7 decision;
- primary criterion status;
- promotion-veto status;
- continuation-veto status;
- main uncertainty;
- next justified action;
- what is not concluded.

Inference-status rows:

- hard-veto screen;
- viable candidate status;
- statistically supported ranking (`not applicable/no ranking attempted`);
- descriptive-only differences;
- default readiness;
- next evidence needed.

## Terminal Conditions

This plan is complete when either:

1. Phase 7 passes, private retained samples verify, the result review is closed,
   and the repository is handed off at the Phase 8 approval boundary; or
2. a structured blocker/result records the exact failed gate, preserved
   artifacts, classification, and next smallest repair, without starting or
   claiming Phase 8.
