# BayesFilter Deterministic LGSSM HMC Tuning Visible Execution Ledger

Date: 2026-07-09

## Status

`PHASE7_APPROVAL_BOUNDARY`

## Ledger

### 2026-07-09 - Phase 0 - PRECHECK

Evidence contract:

- Question: Does the launch plan enforce deterministic Python-owned tuning and
  preserve runtime/scientific boundaries?
- Baseline/comparator: Existing BayesFilter tuning APIs and user directive.
- Primary criterion: planning artifacts exist and launch review has no material
  blocker.
- Veto diagnostics: manual tuning allowed, non-XLA fallback allowed, missing
  phase stop conditions, unsupported claims, or runtime approval bypass.
- Non-claims: no implementation correctness, no HMC readiness, no convergence,
  no posterior recovery.

Actions:

- Create master program, phase subplans, runbook, stop handoff, and review
  bundle.

Artifacts:

- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-master-program-2026-07-09.md`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md`
- `docs/reviews/bayesfilter-deterministic-lgssm-hmc-tuning-launch-review-bundle-2026-07-09.md`

Gate status:

- `PASSED_WITH_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Execute Phase 1 read-only tool inventory.

### 2026-07-09 - Phase 1 - ASSESS_GATE

Evidence contract:

- Question: Which existing BayesFilter tools must the deterministic driver call?
- Baseline/comparator: `quadratic_geometry`, `mass_matrix`,
  `hmc_kernel_tuning`, `hmc_budget_ladder`, and `hmc_diagnostics`.
- Primary criterion: every tuning decision point is assigned to an existing
  Python API or explicitly deferred as a deterministic implementation gap.
- Veto diagnostics: agent-only tuning decision remains, missing quadratic
  initializer, missing mass/covariance handoff, non-XLA target path allowed.
- Non-claims: no tuning result, sampler convergence, or LGSSM recovery claim.

Actions:

- Inspected BayesFilter inference tuning exports and core tuning modules.
- Wrote Phase 1 inventory result.

Artifacts:

- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase1-tool-inventory-result-2026-07-09.md`

Gate status:

- `PASSED`

Next action:

- Execute Phase 2 deterministic config schema.

### 2026-07-09 - Phase 2 - ASSESS_GATE

Evidence contract:

- Question: Can the entire tuning/recovery run be determined by config plus
  code?
- Baseline/comparator: user requirement that tuning must be explicit Python
  with deterministic outcome.
- Primary criterion: config contains no manual post-result decision hook and
  all tuning knobs are predeclared.
- Veto diagnostics: missing seeds, missing caps, non-XLA fallback, unclear
  pass/fail thresholds, missing artifact paths.
- Non-claims: no target correctness or HMC success claim.

Actions:

- Created deterministic JSON config skeleton with `T=120`, fixed seeds,
  XLA-only target path, CPU-hidden sampling policy, geometry/mass/tuning tool
  bindings, burn-in/sampling extension rules, and final recovery gates.

Artifacts:

- `docs/benchmarks/configs/multidim_lgssm_serious_hmc_tuning_2026_07_09.json`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase2-config-schema-result-2026-07-09.md`

Gate status:

- `PASSED`

Next action:

- Execute Phase 3 deterministic fixture generation.

### 2026-07-09 - Phase 3 - ASSESS_GATE

Evidence contract:

- Question: Can the LGSSM fixture be generated deterministically from prior
  means?
- Baseline/comparator: Phase 2 config and lower-triangular LGSSM contract.
- Primary criterion: fixture hash stable and stationarity/shape checks pass.
- Veto diagnostics: nonstationary transition, missing initial distribution,
  nonfinite data, non-deterministic hash.
- Non-claims: no posterior recovery, HMC tuning, or scientific adequacy claim.

Actions:

- Added deterministic fixture stage to
  `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`.
- Added focused tests in
  `tests/test_deterministic_lgssm_hmc_tuning_driver.py`.
- Ran focused pytest and generated the `T=120` fixture.

Artifacts:

- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/fixture_T120_seed20260709_301.json`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase3-lgssm-fixture-result-2026-07-09.md`

Gate status:

- `PASSED`

Next action:

- Execute Phase 4 XLA value/score gate.

## Phase 4: XLA Value/Score Gate

Status: `PASSED`

Skeptical audit:

- The gate answers only XLA value/score admissibility for the Phase 3 fixture.
- Compile success is not treated as HMC tuning, convergence, recovery, GPU, or
  production evidence.
- Veto diagnostics were finite value/score, no runtime non-JIT fallback, no
  runtime autodiff tape, no unexpected retracing, and valid JSON artifact.

Actions:

- Added bounded HLO metadata capture to the deterministic driver.
- Ran the focused driver tests.
- Ran the configured `xla_score` stage with `CUDA_VISIBLE_DEVICES=-1`.
- Wrote Phase 4 result.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `5 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage xla_score
```

Result: passed; TensorFlow logged host XLA compilation.

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/xla_compile_gate.json
```

Result: passed.

```text
git diff --check -- docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py tests/test_deterministic_lgssm_hmc_tuning_driver.py docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/xla_compile_gate.json
```

Result: passed.

Artifacts:

- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/xla_compile_gate.json`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase4-xla-score-gate-result-2026-07-09.md`

Key artifact values:

- artifact hash:
  `sha256:f945e98ad2aac75cb2998e51855a717e9b9894384f254c29861e4911d35593a9`
- `jit_compile`: `true`
- `jit_compile_false_runtime_executed`: `false`
- `runtime_autodiff_tape_executed`: `false`
- score shape: `[18]`
- concrete function count: `1`
- compile plus first execute seconds: `7.762520305113867`
- warm execute seconds: `0.10597397992387414`
- HLO byte count: `3427939`
- vetoes: `[]`

Gate status:

- `PASSED`

Next action:

- Review Phase 4 result and Phase 5 subplan.
- If review passes, execute Phase 5 deterministic geometry and mass
  initialization. Do not start Phase 6 serious HMC tuning without explicit
  runtime approval.

Review:

- Claude review gate attempted with:
  `bash /home/chakwong/python/claudecodex/scripts/claude_review_gate.sh --cwd /home/chakwong/BayesFilter --review-name deterministic-lgssm-hmc-phase4-result-phase5-subplan --bundle /home/chakwong/BayesFilter/docs/reviews/bayesfilter-deterministic-lgssm-hmc-tuning-phase4-result-phase5-subplan-review-bundle-2026-07-09.md --probe-timeout 90 --timeout-seconds 120 --max-retries 1 --allow-bounded-fallback`
- Result: rejected by approval layer as external data-exfiltration risk. No
  workaround was attempted.
- Local substitute review:
  `docs/reviews/bayesfilter-deterministic-lgssm-hmc-tuning-phase4-result-phase5-subplan-codex-substitute-review-2026-07-09.md`
- Substitute verdict: `AGREE_WITH_LIMITATION`

Review limitation:

- This is weaker than Claude review and must not be called Claude convergence.

Handoff:

- Proceed to Phase 5 deterministic geometry/mass initialization only.
- Do not start Phase 6 serious HMC tuning without explicit runtime approval.

## Phase 5: Geometry And Mass Driver

Status: `PASSED`

Skeptical audit:

- The gate answers only whether existing BayesFilter tools can produce a
  deterministic geometry/mass initializer.
- Geometry and mass settings came from the fixed JSON config, not manual
  diagnostic editing.
- The pass does not support HMC convergence, posterior recovery, production,
  GPU, default, or scientific claims.

Actions:

- Added `geometry_mass` stage to
  `docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`.
- Bound the stage to `fit_low_rank_spd_quadratic_geometry` and
  `covariance_from_precision`.
- Added focused tests for deterministic geometry config/scale and mass
  conversion.
- Ran the configured Phase 5 stage.
- Wrote Phase 5 result.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `7 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage geometry_mass
```

Result: passed; geometry and mass artifacts written.

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/geometry.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/mass.json
```

Result: passed.

```text
git diff --check -- <Phase 4/5 modified files and artifacts>
```

Result: passed.

Artifacts:

- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/geometry.json`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/mass.json`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase5-geometry-mass-result-2026-07-09.md`

Key artifact values:

- geometry artifact hash:
  `sha256:e2b9531e86f85a662c4da26595e0ab082dd8a1a29d2dbb83b31b076bbf7683ac`
- geometry status: `usable`
- geometry vetoes: `[]`
- geometry holdout RMSE: `0.12444574204988731`
- geometry holdout threshold: `0.6641473363290465`
- geometry precision condition number: `1.9640743886787067`
- mass artifact hash:
  `sha256:92536fbd13e1ba89c53bfcc874355194b8d2d097ea498d22b5ccd7c318490d8e`
- mass vetoes: `[]`
- mass precision condition number: `1.9640743885607335`
- precision/covariance identity max error: `6.661338147750939e-16`

Gate status:

- `PASSED`

Next action:

- Review Phase 5 result and Phase 6 subplan.
- Stop for explicit runtime approval before any Phase 6 serious HMC tuning.

## Phase 6 / 6R: Kernel Tuning And Adapter Repair

Status: `BLOCKED`

Skeptical audit:

- The original bootstrap hard veto was an adapter contract failure, not
  posterior evidence.
- Phase 6R repaired batched value/score and target-status telemetry while
  preserving the same LGSSM target, fixture, mass artifact, CPU-hidden policy,
  and `use_xla=True` route.
- The post-repair rerun reached the fixed-mass candidate grid, then aborted
  during XLA CPU code generation before producing a final kernel artifact.
- The candidate log-accept hard vetoes remain active and must not be weakened.

Actions:

- Added adapter batch value/score and target-status telemetry support.
- Ran focused deterministic driver tests.
- Reran Phase 6 kernel tuning through the deterministic driver.
- Wrote Phase 6 blocked result and Phase 6S repair subplan.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `11 passed, 2 warnings`.

Artifacts:

- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6r-adapter-contract-repair-subplan-2026-07-09.md`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6s-fixed-mass-xla-compile-repair-subplan-2026-07-09.md`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json`

Key artifact values:

- bootstrap preflight: `passed`
- bootstrap hard vetoes: `[]`
- bootstrap XLA metadata: `true`
- windowed mass adapted signature:
  `eb30e03e82d353cc62ad5c33e0bd90c56cd92efa677e3aa2e7fe56949e8ac497`
- fixed-mass candidate grid completed candidates before abort: `2`
- completed candidate hard veto: `screen_log_accept_nonfinite_or_missing`
- latest progress stage: `fixed_mass_ladder_tune_call_start`

Gate status:

- `BLOCKED`

Next action:

- Execute Phase 6S compile-pressure repair only.
- Do not start Phase 7 burn-in or retained sampling.

## Phase 6S: Fixed-Mass Candidate Grid XLA Compile Repair

Status: `PASSED_ENGINEERING_REPAIR_PHASE6_STILL_BLOCKED`

Evidence contract:

- Question: Can Phase 6 reduce fixed-mass candidate-grid XLA compile pressure
  without changing the target, mass, deterministic tuner, or XLA-only rule?
- Baseline/comparator: blocked Phase 6 run that aborted during fixed-mass
  candidate 2 XLA CPU code generation.
- Primary criterion: focused tests pass and the rerun either produces a final
  kernel artifact or reaches a structured BayesFilter hard-veto/budget result
  instead of process abort.
- Veto diagnostics: non-XLA fallback, runtime `GradientTape`, manual HMC
  parameter choice, weakened log-accept hard vetoes, target/mass/fixture change,
  invalid artifacts, or process abort.
- Nonclaims: no posterior convergence, recovery, sampler superiority,
  production/default readiness, GPU readiness, or scientific claim.

Actions:

- Added a private shared reusable-runner cache hook to
  `run_fixed_mass_hmc_tuning_budget_ladder`.
- Added dynamic `num_leapfrog_steps` support to the fixed-mass reusable-runner
  contract path, matching the existing frozen-step trajectory pattern.
- Wired the joint `(L, epsilon)` candidate grid to share the private cache and
  use the dynamic-leapfrog route.
- Added focused test coverage for shared-cache dynamic-leapfrog reuse.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py
```

Result: `35 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `11 passed, 2 warnings`.

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/geometry.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/mass.json
```

Result: passed.

```text
git diff --check -- bayesfilter/inference/hmc_budget_ladder.py bayesfilter/inference/hmc_kernel_tuning.py tests/test_hmc_budget_ladder.py tests/test_deterministic_lgssm_hmc_tuning_driver.py docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6s-fixed-mass-xla-compile-repair-subplan-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md
```

Result: passed.

Review:

- Claude health probe returned `CLAUDE_PROBE_OK`.
- One-path bounded Claude review of the Phase 6S subplan returned
  `VERDICT: AGREE`.
- Review record:
  `docs/reviews/bayesfilter-deterministic-lgssm-hmc-tuning-phase6s-subplan-claude-review-2026-07-09.md`
- Claude did not inspect implementation; local tests carry the implementation
  evidence burden.

Gate status:

- `PASSED_FOR_STRUCTURED_RESULT_ONLY`

Next action:

- Execute Phase 6T log-accept telemetry root-cause repair.
- Do not start Phase 7 burn-in or retained sampling.

### Phase 6S Rerun Result

Actions:

- Reran Phase 6 kernel tuning through the deterministic driver with
  `CUDA_VISIBLE_DEVICES=-1`.
- The process exited with code `0` and produced structured result artifacts.
- Refreshed the Phase 6 result note to replace the stale XLA allocation-abort
  blocker with the current structured hard veto.
- Drafted the Phase 6T log-accept telemetry root-cause repair subplan.

Checks:

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json
```

Result: passed.

Artifacts:

- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_result.json`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6t-log-accept-telemetry-repair-subplan-2026-07-09.md`

Key artifact values:

- `kernel_tuning.json` artifact hash:
  `sha256:8b3c40faeabdcb71c41b4384ad0d1549f3f257a6254f3cde3bb7a76693c19970`
- `passed=false`
- `final_status=hard_veto`
- `xla_confirmed=true`
- hard veto: `screen_log_accept_nonfinite_or_missing`
- fixed-mass candidate completions: 14 across two bounded attempts
- fixed-mass candidate pass count: 0
- final kernel payload/hash: `null`

Gate status:

- `BLOCKED_STRUCTURED_HARD_VETO`

Next action:

- Execute Phase 6T after bounded subplan review.
- Do not start Phase 7 burn-in or retained sampling.

## Phase 6T: Log-Accept Telemetry Root-Cause Repair

Status: `SUBPLAN_DRAFTED`

Evidence contract:

- Question: Is the fixed-mass screen hard veto caused by true nonfinite
  log-acceptance evidence, or by missing telemetry propagation from runner
  diagnostics into budget-ladder diagnostics?
- Baseline/comparator: Phase 6S structured result with 14 of 14 candidates
  hard-vetoed by `screen_log_accept_nonfinite_or_missing`.
- Primary criterion: focused tests pass, the hard veto remains fail-closed for
  missing/nonfinite evidence, and a Phase 6 rerun either produces a final kernel
  artifact or a more specific structured hard-veto reason.
- Veto diagnostics: non-XLA fallback, runtime `GradientTape`, manual HMC
  parameter choice, treating missing telemetry as finite, target/mass/fixture
  change, invalid JSON artifacts, or process abort.
- Nonclaims: no posterior convergence, recovery, sampler superiority,
  production/default readiness, GPU readiness, or scientific claim.

Gate status:

- `READY_FOR_REVIEW`

Next action:

- Run a bounded one-path Claude read-only review of the Phase 6T subplan.
- If review agrees, implement the smallest fail-closed telemetry repair and
  rerun focused tests.

### Phase 6T Result

Status: `BLOCKED_TRUE_LOG_ACCEPT_MECHANICS_HARD_VETO`

Actions:

- Added fail-closed log-accept summary support to the budget ladder.
- Added private last-ladder-round diagnostic summaries to fixed-mass
  joint `(L, epsilon)` candidate events.
- Reran Phase 6 kernel tuning through the deterministic driver with
  `CUDA_VISIBLE_DEVICES=-1`, `use_xla=true`, and no non-XLA fallback.
- Refreshed the Phase 6 result note and added the Phase 6U mechanics
  diagnostic subplan to the runbook.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py
```

Result: `37 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: `16 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `11 passed, 2 warnings`.

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_result.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json
```

Result: passed.

Artifacts:

- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/private_diagnostics/hmc_tuning_events.jsonl`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6u-kernel-mechanics-diagnostic-subplan-2026-07-09.md`

Key artifact values:

- `kernel_tuning.json` artifact hash:
  `sha256:44c7cebd0a3cd09a4a41b75408d141bce6860ad95ee5a857e0626f042757960d`
- `kernel_tuning.json` file SHA-256:
  `3e7b11f0c4242ececc0f623253c513f968fd0d74784273e009f5cf8874dd89af`
- private event file SHA-256:
  `14896aed6d0885d02b5270b531d863e208af49c0adce2324ff20337fb3b3e173`
- latest instrumented fixed-mass candidate count: 14
- latest private log-accept source: full trace
- latest private screen log-accept finite count: 0 per candidate
- latest private screen log-accept nonfinite count: 500 per candidate
- latest private screen target-log-prob finite count: 500 per candidate
- latest private screen samples finite: true per candidate

Gate status:

- `BLOCKED_TRUE_LOG_ACCEPT_MECHANICS_HARD_VETO`

Next action:

- Execute Phase 6U after bounded one-path Claude review.
- Do not start Phase 7 burn-in or retained sampling.

## Phase 6U: Kernel Mechanics Nonfinite Log-Accept Diagnostic

Status: `BLOCKED_UNSTABLE_PROPOSED_HMC_TRANSITIONS`

Review:

- Claude health probe returned `CLAUDE_PROBE_OK`.
- One-path bounded Claude review of the Phase 6U subplan returned
  `VERDICT: AGREE`.

Actions:

- Added private trace summaries for proposed target log-probability and HMC
  log-acceptance correction.
- Threaded those summaries into budget-ladder diagnostics and fixed-mass
  private candidate events.
- Reran Phase 6 kernel tuning through the deterministic driver with
  `CUDA_VISIBLE_DEVICES=-1`, `use_xla=true`, and no non-XLA fallback.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py
```

Result: `38 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: `16 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_nonlinear_ssm_phase4_full_chain_hmc.py::test_phase4_tiny_full_chain_hmc_jit_returns_finite_samples_and_metadata
```

Result: `1 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `11 passed, 2 warnings`.

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_result.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json
```

Result: passed.

Artifacts:

- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/private_diagnostics/hmc_tuning_events.jsonl`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`

Key artifact values:

- `kernel_tuning.json` artifact hash:
  `sha256:733310aeeac8801779f33c8039e66eb73307e9dd9c2e7fe45d121d88835b73e2`
- `kernel_tuning.json` file SHA-256:
  `fb61514c4dde1ed36b0160ed63f2375c1d67c700db7e6340676d9aed42df696a`
- private event file SHA-256:
  `f368dc2b1a3c84ce2f15ac3fdffca0dfafa108e32d07445a2cfd3e5df9004912`
- latest fixed-mass screen log-accept finite count: 0 per candidate
- latest fixed-mass screen log-accept nonfinite count: 500 per candidate
- latest accepted/current target-log-prob finite count: 500 per candidate
- latest proposed target-log-prob finite count: 0 per candidate
- latest proposed target-log-prob nonfinite count: 500 per candidate
- latest log-acceptance-correction finite count: 0 per candidate
- latest log-acceptance-correction nonfinite count: 500 per candidate
- latest screen samples finite: true per candidate

Gate status:

- `BLOCKED_UNSTABLE_PROPOSED_HMC_TRANSITIONS`

Next action:

- Draft and review Phase 6V deterministic step/mass-scale policy audit and
  repair.
- Do not start Phase 7 burn-in or retained sampling.

## Phase 6V: Deterministic Step/Mass-Scale Policy Audit

Status: `BLOCKED_BUDGET_INCOMPLETE_NON_PROMOTING`

Skeptical audit:

- The old blocker was a true fixed-mass log-accept mechanics hard veto.
- The Phase 6V repair was scoped to an opt-in, serious-tuning-only private
  repair classification for isolated nonfinite proposal mechanics.
- The repair did not change the LGSSM fixture, prior, target, XLA-only runtime
  contract, or Phase 7 sampling boundary.

Actions:

- Added `repair_nonfinite_proposal_screen` as a fail-closed config flag.
- Threaded the flag from serious kernel tuning config through the fixed-mass
  step stage to the fixed-mass budget ladder.
- Added focused tests proving default behavior remains a hard veto and serious
  behavior treats the isolated screen as a private repair trigger.
- Reran Phase 6 kernel tuning with `CUDA_VISIBLE_DEVICES=-1`, `use_xla=true`,
  and no non-XLA fallback.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: `56 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `11 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `67 passed, 2 warnings`.

```text
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_result.json
python -m json.tool docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json
```

Result: passed.

Artifacts:

- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_result.json`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json`
- `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/private_diagnostics/hmc_tuning_events.jsonl`
- `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`

Key artifact values:

- `kernel_tuning.json` payload artifact hash:
  `sha256:8b3367c52619957080a6c28e51262cfed537b05c2d6d2d6b76a151d3da355484`
- `kernel_tuning.json` file SHA-256:
  `5bb6948c2bbc90038818bf98c4713d6d99983d593bd576c552810bbe622d482d`
- public result SHA-256:
  `3909d89b4f7cfbcb30899aee26f005cdc744c4b34ddf446826e3ab3268db4a28`
- public progress SHA-256:
  `510085bb385c32d7e35b410442571260722862df4243e635942b9e7aca77eb44`
- private event SHA-256:
  `1e818b8ef1d49ef8f5570f2ea6a4a3210af6e5836ccde29c68537eab346b5079`
- `passed=false`
- `final_status=budget_exhausted`
- `diagnostic_role=fixed_mass_step_budget_incomplete_non_promoting`
- `xla_confirmed=true`
- `hard_vetoes=[]`
- `final_kernel_hash=null`

Gate status:

- `BLOCKED_BUDGET_INCOMPLETE_NON_PROMOTING`

Next action:

- Draft and review Phase 6W fixed-mass final-local budget/timeout repair.
- Do not start Phase 7 burn-in or retained sampling.

## Phase 6W: Fixed-Mass Final-Local Budget/Timeout Repair

Status: `BLOCKED_XLA_CPU_LLVM_COMPILE_MEMORY`

Skeptical audit:

- The previous Phase 6V blocker was budget incomplete after selected-pair
  progress, not a mechanics hard veto.
- The Phase 6W repair used deterministic staged timeout policy only; it did
  not choose step size, leapfrog count, mass, or runtime mode manually.
- The rerun remained CPU-hidden and XLA-only.
- A process abort before refreshing `kernel_tuning.json` is a continuation
  blocker for Phase 6 and cannot be promoted.

Actions:

- Updated the serious deterministic driver to use the existing
  `HMCGeometryScaledBudgetTimingPolicy().staged_timeout_policy()`.
- Reviewed the Phase 6W subplan with bounded one-path Claude review:
  `VERDICT: AGREE`.
- Ran focused local checks.
- Reran Phase 6 kernel tuning with `CUDA_VISIBLE_DEVICES=-1`, `use_xla=true`,
  and no non-XLA fallback.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `11 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: `17 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `67 passed, 2 warnings`.

Artifacts:

- stale complete result:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
- updated progress:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json`
- updated private diagnostics:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/private_diagnostics/hmc_tuning_events.jsonl`
- Phase 6 result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`

Key artifact values:

- `kernel_tuning.json` remains stale from Phase 6V:
  `5bb6948c2bbc90038818bf98c4713d6d99983d593bd576c552810bbe622d482d`;
- progress SHA-256:
  `1da089a95ba23a60187271f6affde31c32deb6a913b050c3c3fd1088baa70bd2`;
- private event SHA-256:
  `2e46ad180748caf7ad64d4c796d2016746036642cdd1f5065e59dc691adbef74`;
- latest progress reports selected-pair progress before the crash.

Gate status:

- `BLOCKED_XLA_CPU_LLVM_COMPILE_MEMORY`

Next action:

- Draft, review, and execute Phase 6X fixed-mass XLA compile-memory reuse
  repair.
- Do not start Phase 7 burn-in or retained sampling.

## Phase 6X: Fixed-Mass XLA Compile-Memory Reuse Repair

Status: `BLOCKED_XLA_CPU_LLVM_COMPILE_MEMORY_TRAJECTORY_STAGE`

Skeptical audit:

- The Phase 6W crash happened after selected-pair progress, so a compile-cache
  scope repair was directly relevant.
- The Phase 6X implementation changed only reusable-runner cache lifetime
  across fixed-mass joint-grid rounds.
- The rerun remained CPU-hidden and XLA-only.
- A process abort before refreshing `kernel_tuning.json` remains a continuation
  blocker for Phase 6 and cannot be promoted.

Actions:

- Shared one dynamic-leapfrog reusable-runner cache across fixed-mass initial,
  edge-repair, and final-local grid rounds.
- Added focused test coverage for cross-round dynamic-runner reuse.
- Reran Phase 6 kernel tuning with `CUDA_VISIBLE_DEVICES=-1`, `use_xla=true`,
  and no non-XLA fallback.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: `18 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `68 passed, 2 warnings`.

```text
git diff --check -- bayesfilter/inference/hmc_kernel_tuning.py tests/test_hmc_kernel_tuning_fixed_mass_step.py docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md
```

Result: passed.

```text
rg -n "GradientTape|batch_jacobian|tape\.|jit_compile\s*=\s*False|jit_compile=False" bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
```

Result: no matches.

Artifacts:

- stale complete result:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
- updated progress:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json`
- updated private diagnostics:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/private_diagnostics/hmc_tuning_events.jsonl`
- Phase 6 result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- Phase 6Y subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6y-trajectory-xla-compile-memory-reuse-repair-subplan-2026-07-09.md`

Key artifact values:

- `kernel_tuning.json` remains stale from Phase 6V:
  `5bb6948c2bbc90038818bf98c4713d6d99983d593bd576c552810bbe622d482d`;
- public progress SHA-256:
  `c701cb887c7c7b8c452938d5e05caac7fe1eecdc2b07a2b90ebdef1332663563`;
- public result SHA-256:
  `3909d89b4f7cfbcb30899aee26f005cdc744c4b34ddf446826e3ab3268db4a28`;
- private event SHA-256:
  `e2473e79fbd6c99567c1d999cff006238df15d80aec07ece62cb1af1d6b198e6`;
- latest progress stage:
  `trajectory_candidate_call_start`;
- latest completed stage:
  `fixed_mass_step_complete`;
- final-local private summary reports selected-pair progress.

Gate status:

- `BLOCKED_XLA_CPU_LLVM_COMPILE_MEMORY_TRAJECTORY_STAGE`

Review:

- Claude review for Phase 6Y was attempted with the one-path subplan review
  gate and was rejected by the approval layer as external data-exfiltration
  risk.
- No workaround was attempted.
- A local Codex substitute review was launched; this is weaker than Claude
  review and must be recorded as such.

Next action:

- Execute Phase 6Y only after local substitute review is complete.

## Phase 6Y: Trajectory Handoff XLA Compile-Memory Reuse Repair

Status: `BLOCKED_MOVED_FORWARD`

Skeptical audit:

- The Phase 6X failure was a process abort at
  `trajectory_candidate_call_start`, so a private cache-handoff repair was
  targeted at redundant trajectory compile pressure only.
- The repair did not change the LGSSM target, prior, fixture, mass artifact,
  selected-kernel pass criteria, or XLA-only rule.
- The Phase 6Y result is not HMC readiness: it moved the blocker to final
  verification and still produced no final kernel payload/hash.

Review:

- Claude review for Phase 6Y was attempted with the one-path subplan review
  pattern, but the approval layer rejected the command as an external
  data-exfiltration risk. No workaround was attempted.
- A local Codex substitute review was used. Iteration 1 returned
  `VERDICT: REVISE`; iteration 2 returned `VERDICT: AGREE` after adding a
  negative static-contract-mismatch test and CPU-hidden result-manifest
  wording.
- This is weaker than Claude review and is not Claude convergence.

Actions:

- Added private fixed-mass reusable-runner cache handoff into the frozen-step
  trajectory selected-pair stage.
- Added tests proving reuse for matching static contracts, non-reuse for
  mismatched contracts, and no public payload leakage of live runner objects.
- Reran Phase 6 kernel tuning with GPU intentionally hidden.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: `20 passed`.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_deterministic_lgssm_hmc_tuning_driver.py
```

Result: `70 passed, 2 warnings`.

```text
git diff --check -- bayesfilter/inference/hmc_kernel_tuning.py tests/test_hmc_kernel_tuning_fixed_mass_step.py docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md
```

Result: passed.

```text
python -m py_compile bayesfilter/inference/hmc_kernel_tuning.py tests/test_hmc_kernel_tuning_fixed_mass_step.py
```

Result: passed.

```text
rg -n "GradientTape|batch_jacobian|tape\.|jit_compile\s*=\s*False|jit_compile=False" bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py
```

Result: no matches.

Serious rerun:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6y python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning
```

Result:

- GPU devices were intentionally hidden with `CUDA_VISIBLE_DEVICES=-1`.
- Attempt 1 fixed-mass final-local completed 3 of 3 candidates, with 2 viable.
- Selected fixed pair:
  `num_leapfrog_steps=4`, `step_size=0.6272447304363297`,
  `screen_acceptance_rate=0.686`.
- Frozen-step trajectory completed and passed:
  `screen_acceptance_rate=0.726`, `hard_vetoes=[]`,
  `selected_trajectory_hash=46b34400e68b6d18bed98a496d90ddf0f1d463055ab65533c24545f6f4b79cf2`.
- The process reached `verification_start`, then aborted with exit code `134`
  and XLA CPU LLVM allocation errors.

Artifacts:

- `kernel_tuning.json` remains stale from Phase 6V:
  `5bb6948c2bbc90038818bf98c4713d6d99983d593bd576c552810bbe622d482d`.
- public progress SHA-256:
  `8a97fb73ee83c52f854f60e3eb2f1216d7e9abbe26f565e25c98e7560d2cdbf6`.
- public result SHA-256:
  `3909d89b4f7cfbcb30899aee26f005cdc744c4b34ddf446826e3ab3268db4a28`.
- private event SHA-256:
  `782d8120e41fb300604070847437222a45392d4b2d2dd2288f04e127764a75ed`.

Gate status:

- `BLOCKED_XLA_CPU_LLVM_COMPILE_MEMORY_VERIFICATION_STAGE`

Next action:

- Execute Phase 6Z verification chunk compile-memory repair after bounded
  subplan review.
- Do not start Phase 7 burn-in or retained sampling.

## 2026-07-09T17:25:53Z - Phase 6Z Close

Action:

- Implemented deterministic final-verification chunk controls:
  `verification_chunk_max_results` and
  `verification_min_retained_results_for_pass`.
- Threaded the controls through the public kernel config, Phase 7 loop config,
  deterministic LGSSM driver config, and sequential R-hat verifier.
- Preserved budget-policy-owned total verification `max_results`, XLA-only
  execution, CPU-hidden HMC sample generation, and the minimum retained pass
  gate.

Checks:

- `CUDA_VISIBLE_DEVICES=-1 python -m py_compile bayesfilter/inference/hmc.py bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`:
  passed.
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_fixed_size_chunk_runner.py`:
  `17 passed, 2 warnings`.
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest --import-mode=importlib -q tests/test_hmc_kernel_tuning_outer_loop.py`:
  `66 passed`.
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest --import-mode=importlib -q tests/test_hmc_kernel_tuning_public_api.py`:
  `34 passed`.
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py`:
  `11 passed, 2 warnings`.
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest --import-mode=importlib -q tests/test_hmc_budget_ladder.py tests/test_hmc_kernel_tuning_fixed_mass_step.py tests/test_hmc_kernel_tuning_outer_loop.py tests/test_hmc_kernel_tuning_public_api.py tests/test_deterministic_lgssm_hmc_tuning_driver.py`:
  `170 passed, 2 warnings`.
- `git diff --check` on touched Phase 6Z implementation/test/driver/config
  files: passed.
- `rg -n "GradientTape|batch_jacobian|tape\.|jit_compile\s*=\s*False|jit_compile=False" bayesfilter/inference/hmc.py bayesfilter/inference/hmc_kernel_tuning.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`:
  no matches.

Serious rerun:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6z python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning
```

Result:

- GPU devices were intentionally hidden with `CUDA_VISIBLE_DEVICES=-1`.
- Host XLA compilation was observed.
- Attempt 1 reached a passed frozen-step trajectory handoff:
  `classification=passed_screen`, `diagnostic_role=trajectory_handoff_promotion_only`,
  `num_leapfrog_steps=4`.
- Public progress reached `current_stage=verification_start` for
  `attempt_index=1`, `budget=2000`.
- The process aborted with exit code `134` and XLA CPU LLVM allocation errors:
  `LLVM compilation error: Cannot allocate memory`,
  `allocateMappedMemory failed with error: Cannot allocate memory`, and
  `LLVM ERROR: Unable to allocate section memory!`.

Artifacts:

- `kernel_tuning.json` remains stale from Phase 6V:
  `5bb6948c2bbc90038818bf98c4713d6d99983d593bd576c552810bbe622d482d`.
- public result SHA-256:
  `3909d89b4f7cfbcb30899aee26f005cdc744c4b34ddf446826e3ab3268db4a28`.
- public progress SHA-256:
  `aa62e50b2c5f1e7a85f3e04ee1479dcadf04c7e763dc620a793add1a12343862`.
- private event SHA-256:
  `24321079dc3cc5302038665dbb6880c18db244a2298f9724109f47b667998cb7`.

Gate status:

- `BLOCKED_XLA_CPU_LLVM_COMPILE_MEMORY_VERIFICATION_STAGE`

Next action:

- Do not start Phase 7 burn-in or retained sampling.
- Draft Phase 6AA final-verification compile-memory reduction subplan.

## Phase 6Z: Verification Chunk XLA Compile-Memory Repair

Status: `SUBPLAN_DRAFTED`

Evidence contract:

- Question: Can final fixed-kernel verification avoid the Phase 6Y XLA CPU LLVM
  compile-memory abort by using a smaller deterministic fixed-size verification
  chunk while preserving the same total verification budget and pass gate?
- Baseline/comparator: Phase 6Y reached `verification_start` after a passed
  trajectory screen and then aborted with exit code `134`.
- Primary criterion: refreshed Phase 6 artifact has `passed=true`, confirmed
  XLA/JIT, no hard vetoes, and final kernel payload/hash; or preserves a more
  specific structured non-promoting blocker.
- Veto diagnostics: non-XLA fallback, runtime `GradientTape`, manual
  step/leapfrog/mass selection, target/prior/fixture change, invalid artifact,
  pass before minimum retained verification count, process abort, or Phase 7
  sampling.
- Nonclaims: no posterior convergence, recovery, sampler superiority,
  production/default readiness, GPU readiness, DSGE readiness, or scientific
  claim.

Artifacts:

- Phase 6Z subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6z-verification-chunk-xla-compile-memory-repair-subplan-2026-07-09.md`

Gate status:

- `READY_FOR_REVIEW`
- Do not start Phase 7 burn-in or retained sampling.

## Phase 6AA: SVD Score Wiring Retry

Status: `PASSED_KERNEL_HANDOFF_PHASE7_APPROVAL_REQUIRED`

Evidence contract:

- Question: after demoting the active QR derivative route and wiring
  SVD/eigh graph-status scoring, can the XLA value/score gate and Phase 6
  kernel-tuning gate be refreshed without the stale QR compile path?
- Baseline/comparator: Phase 6Z reached `verification_start` and then aborted
  with XLA CPU LLVM allocation errors before writing refreshed
  `kernel_tuning.json`.
- Primary criterion: refreshed XLA gate passes with `jit_compile=true`, finite
  value/score, and valid SVD status; if kernel tuning runs, refreshed
  `kernel_tuning.json` passes with confirmed XLA and no hard vetoes.
- Veto diagnostics: `jit_compile=false`, runtime `GradientTape`, active QR
  derivative route in the serious target/driver, invalid SVD status, nonfinite
  value/score, manual tuning, target/prior/fixture changes outside the SVD
  wiring repair, process abort without blocker artifact, or Phase 7 sampling.
- Nonclaims: no posterior convergence, recovery, sampler superiority,
  production/default readiness, GPU readiness, DSGE readiness, or scientific
  claim.

Artifacts:

- Phase 6AA subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6aa-svd-score-wiring-retry-subplan-2026-07-10.md`
- Runbook row added:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md`

Gate status:

- `PASSED_KERNEL_HANDOFF_PHASE7_APPROVAL_REQUIRED`
- Phase 7 remains unexecuted and requires explicit user approval.

Close record:

- Phase 6AA result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6aa-svd-score-wiring-retry-result-2026-07-10.md`
- Refreshed XLA gate:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/xla_compile_gate.json`
- Refreshed kernel tuning:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
- Public result:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_result.json`
- Public progress:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/hmc_kernel_tuning_progress.json`

Key artifact values:

- `xla_compile_gate.json`: `passed=true`, `jit_compile=true`,
  `finite_value=true`, `finite_score=true`, `target_status_valid=true`.
- XLA gate artifact hash:
  `sha256:8941b369f6280ebc3c124220a9bab21f6889228deb92121d63f2fefba3ea6842`.
- XLA gate file SHA-256:
  `8c54c60d7d51cf5ee3d04dfa32df036fc9616c0647e399813ca846e3812e0343`.
- `kernel_tuning.json`: `passed=true`, `final_status=passed`,
  `diagnostic_role=fresh_fixed_kernel_verification_passed`,
  `xla_confirmed=true`, `jit_compile=true`,
  `jit_compile_false_runtime_executed=false`,
  `runtime_autodiff_tape_executed=false`, `hard_vetoes=[]`.
- Kernel artifact hash:
  `sha256:f8c94073b60a6458538537317e49ed683ad0c94b525cafc77cc4d01822badaa2`.
- Kernel file SHA-256:
  `ee9f9308d055cb2482b1fbc2661fc2bd7fa21d7128a51902f49f237c98bddefa`.
- Final kernel hash:
  `8ddf25a3b572893e19e814fad5ca5b6150718e36f760c159b47db1231d92ffff`.
- Phase 7 public handoff kernel hash:
  `391558a9b5f4cdc1b9dff9a5e9bceba668dedded7298c1d8c76daea42f42039a`.
- Verification acceptance rate: `0.71325`.
- Public result file SHA-256:
  `2fc7a40c022465625e7855cda679a86a6735f7657b46f25db706d37bfef17d23`.
- Public progress file SHA-256:
  `db03bb242661889981c5898e34cc46995fa15f1235c558331aee76bd731d30bc`.
- Private event log SHA-256:
  `2e7499befaca4450d2109f8b55247a840682ee91949741182920014cb65f60fc`.

Checks:

- `python -m py_compile bayesfilter/testing/multidim_triangular_lgssm_tf.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`:
  passed.
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_multidim_triangular_lgssm_tf.py`:
  `9 passed`.
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py`:
  `11 passed`.
- `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_svd_linear_gaussian_score_tf.py`:
  `15 passed`.
- Forbidden-token scans on the active runtime files found no
  `GradientTape`, `batch_jacobian`, `tape.`, or `jit_compile=false` route.

Next action:

- Stop at the Phase 7 approval boundary.
- Do not start burn-in or retained sampling until the user explicitly approves
  Phase 7 runtime.
