# BayesFilter Fixed-Identity Tuner Repair and PP-UKF Rerun Plan

Date: 2026-07-21

## Objective

Repair the shared public HMC tuner so `mass_policy="fixed_identity"` is
actually fixed throughout the operational Phase-4 warmup, then rerun the
PP-UKF frozen-transport tuning-only campaign under the repaired code.

This is an engineering and sampler-mechanics campaign. It does not authorize
sequential posterior sampling, transport retraining, public release, or a
posterior/convergence claim.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Does the public tuner preserve identity mass when its configuration declares `fixed_identity`, and does PP-UKF reach a terminal tuning result after that repair? |
| Candidate/mechanism | Shared `tune_hmc_kernel` fixed-identity route; PP-UKF frozen NeuTra transport |
| Expected failure mode | Phase-4 inner warmup config defaults to `windowed_adaptive`, allowing covariance assessment and coordinate replacement despite the public fixed-identity declaration |
| Promotion criterion | All fixed-identity focused tests pass; a fresh PP-UKF tuning-only artifact has matching fixed-identity signatures, finite required diagnostics, and a terminal status |
| Promotion veto | Any mass signature mutation, metric decision/update in a fixed-identity run, non-finite target/HMC telemetry, source/transport identity mismatch, or missing terminal artifact |
| Continuation veto | Scientific contract, frozen transport, target, hardware class, privacy boundary, or total campaign budget would change; or the fresh run exhausts its declared budget without a safe repair |
| Repair trigger | Focused regression failure, source/transport mismatch, infrastructure interruption, or a localized serialization/runtime defect under the unchanged contract |
| Explanatory diagnostics | Acceptance, step size, runtime, target-status counts, and stage timing; these do not establish convergence or superiority |
| Nonclaims | No posterior correctness, convergence, sampler superiority, default readiness, HMC production readiness, or scientific PP-UKF claim |

## Evidence Contract

- Comparator: the declared PP-UKF frozen NeuTra target and identity mass, not a
  windowed/adaptive-mass alternative.
- Primary pass/fail: fixed-identity policy is propagated into the operational
  warmup config and no metric assessment/update or coordinate replacement is
  possible on that route.
- Required artifact: fresh output under
  `docs/plans/artifacts/bayesfilter-pp-ukf-fixed-identity-repair-rerun-20260721-01/`
  containing the tuning result, progress, run manifest, and frozen-transport
  input binding.
- A terminal `TUNING_SUCCEEDED` or `TUNING_FAILED` result is evidence about the
  repaired tuning attempt only. An interrupted run is infrastructure evidence,
  not candidate pass/fail evidence.

## Default and Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Fixed identity mass | PP-UKF public tuner and owner policy | The route explicitly declares `mass_policy="fixed_identity"` | Silent covariance adaptation changes the tuned target | Config propagation and no-update regression tests | Reviewed target policy |
| TensorFlow/TFP GPU/XLA | Repository defaults and PP-UKF route | Required execution path for this campaign | Sandbox/device or compile failure | Trusted GPU preflight and manifest | Reviewed default |
| Existing frozen PP-UKF transport | Prior completed training artifact | Avoids retraining and keeps target fixed | Hash or target-scope drift | Frozen payload SHA and source binding checks | Frozen input |
| Tuning-only mode | Prior PP-UKF repair guard | Prevents accidental sequential sampling | Runner accidentally launches sampling | Result/manifest `sampling_launched=false` check | Safety boundary |
| Fresh output root | Academic reproducibility policy | Prevents overwriting interrupted evidence | Confusing old and new evidence | Pre-launch path nonexistence check | Required |
| Current campaign cap | Prior bounded PP-UKF tuning campaign, reduced to one cell | Keeps compute bounded while testing the repair | Boundary interruption before terminal result | Progress heartbeat and terminal-artifact check | Bounded budget |

## Skeptical Plan Audit

The plan passes the required pre-execution audit with these limits:

1. The baseline is the PP-UKF fixed-identity target itself. A windowed-mass
   run would answer a different question and is not a comparator.
2. Acceptance and runtime are not promotion metrics; they are explanatory
   diagnostics. No ranking is attempted.
3. The plan has explicit stop conditions: source/target drift, mass mutation,
   non-finite telemetry, missing required artifact, contract change, or budget
   exhaustion.
4. The main hidden assumption, that copying the policy into the inner config is
   sufficient, is tested at both the config boundary and the runtime update
   branch. The runtime test must prove that `assess_metric_covariance` is not
   called and that the final metric signature equals the initial signature.
5. Existing failed/interrupted PP-UKF attempts cannot be upgraded. The rerun
   uses a fresh root and the unchanged transport and target.
6. The PP-UKF route may still fail for target, numerical, or infrastructure
   reasons after this repair. Such a result rejects the candidate attempt, not
   the PP-UKF research direction.

## Execution Steps

1. Add `mass_policy` to the `_windowed_mass_stage_internal_config` boundary and
   pass `cfg.mass_policy` from `run_hmc_windowed_mass_stage`.
2. Add focused tests for propagation and fixed-identity operational behavior;
   preserve the existing user changes in the HMC modules.
3. Run focused CPU-only tests and compile/diff checks.
4. Validate the frozen PP-UKF transport and current source bindings.
5. Launch one fresh PP-UKF tuning-only run with GPU/XLA and memory growth,
   recording the exact command, environment, seed, source hashes, wall time,
   and artifact paths.
6. Inspect the terminal result using the decision and inference-status tables
   below. Do not launch sequential sampling from this plan.

### Execution Audit Amendment

The first execution exposed a material plan flaw: the plan bounded transition
count but did not require a prospective wall-time/rate projection for the
repaired operational route. The PP-UKF attempt completed 100/1,000 transitions
in approximately 47 minutes and was stopped before terminal tuning. A future
continuation must add that projection and a wall-time cap before launch.

## Commands

Focused checks:

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_hmc_kernel_tuning_windowed_mass.py \
  tests/test_hmc_warmup.py \
  tests/test_hmc_kernel_tuning_public_api.py
python -m compileall -q bayesfilter/inference/hmc_kernel_tuning.py bayesfilter/inference/hmc_warmup.py
git diff --check
```

The exact trusted GPU PP-UKF command will be recorded in the result note after
the frozen transport and current source signatures are validated. It must use
the repository's `run_neutra_all_models_end_to_end.py --action validate-frozen
--cell PP-UKF --tuning-only` entry point and a fresh output root.

## Stop Conditions

- Do not run the PP-UKF campaign if focused tests fail.
- Do not use any prior tuning result as a selected kernel.
- Stop before sampling if tuning has no terminal result, has any fixed-mass
  mutation, or fails the frozen-input/source-binding checks.
- Stop and write an infrastructure result if the process boundary terminates
  the run before a terminal artifact.

## Planned Result Record

The execution result must include a decision table with primary criterion,
veto status, uncertainty, next action, and nonclaims, plus an inference-status
table covering hard veto, statistical ranking, descriptive differences,
default-readiness, and next evidence needed.
