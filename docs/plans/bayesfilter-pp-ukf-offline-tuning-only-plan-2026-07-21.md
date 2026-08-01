# PP-UKF Offline Fixed-Identity Tuning Plan

Date: 2026-07-21
Status: `EXECUTED_BLOCKED_ROUTE_CONTRACT`

## Scope And Boundary

This plan executes one fresh PP-UKF offline tuning-only run using the existing
public tuner. It does not retrain NeuTra, launch sequential HMC, consume a
claim partition, or promote PP-UKF as valid, converged, superior, or ready for
production.

Prior PP-UKF attempts under
`docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/`
remain historical evidence. Their bootstrap failures and managed-boundary
interruptions are not overwritten or treated as success.

## Research Intent Ledger

| Field | Frozen decision |
| --- | --- |
| Question | Can the current public tuner produce a valid fixed-identity PP-UKF frozen-kernel tuning artifact for the already trained transport? |
| Target | PP-UKF six-parameter predator-prey principal-square-root UKF adapter, target signature `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5` |
| Candidate | Existing frozen plain dense-IAF transport from the Phase 5 campaign, loaded without retraining |
| Tuner | `tune_hmc_kernel` through `run_neutra_frozen_transport_validation_cell`, serious public route, `mass_policy="fixed_identity"` |
| Comparator | None for promotion; acceptance band `[0.65, 0.75]` is a hard engineering screen, not a superiority comparison |
| Primary pass criterion | A complete tuning-only result with no hard veto, fixed-identity mass invariants, valid final frozen-kernel handoff, and repository artifacts |
| Hard vetoes | Hash/signature drift, target or transport load failure, GPU/memory-growth/XLA failure, nonfinite target/kernel diagnostics, bootstrap or verification hard veto, mass-signature mutation, stale/colliding output root, or missing terminal artifact |
| Explanatory diagnostics | Acceptance, runtime, candidate step/leapfrog values, repair triggers, and proposed-step telemetry; none establish convergence or posterior correctness |
| Nonclaims | No sequential HMC claim, posterior correctness claim, convergence claim, NeuTra quality claim, UKF exactness claim, sampler superiority claim, default-readiness claim, or production-readiness claim |

## Exact Scope Identity

- Model/cell: `PP-UKF`.
- Target signature: `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`.
- Frozen transport: `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/campaign-01/PP-UKF/final/segments/steps-004001-005000/frozen_transport.json`.
- Frozen transport SHA-256: `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`.
- Target dimension: 6.
- Backend: TensorFlow/TFP, GPU, XLA enabled, TF32 enabled, float64 target/transport path as declared by the existing PP-UKF adapter.
- Mass policy: fixed identity in the transformed coordinates; no empirical mass adaptation is admissible.
- Output root: `docs/plans/artifacts/bayesfilter-pp-ukf-offline-tuning-only-20260721-01`.

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- |
| Reuse frozen transport | Existing Phase 5 artifact and matching SHA-256 | Stale or cross-target transport | Hash and target-signature load checks | Frozen |
| Public tuner owns all tuning | Repository public API and merged route contract | Legacy/private tuner silently changes criteria | Tuning payload route and mass-policy assertions | Required |
| Fixed identity mass | Owner policy and PP-UKF repair scope | Hidden covariance update changes candidate | Mass signature invariants in each stage artifact | Required |
| Serious preset | Existing public tuner budget policy | Too-small screen hides bootstrap failure | Budget payload and terminal hard-veto checks | Reviewed bounded choice |
| Tuning-only mode | Existing frozen-validation API | Sampling accidentally launches | Config guard and `sampling_launched: false` result field | Required |

## Skeptical Plan Audit

- Wrong baseline risk: no cross-model tuning values are transferred; the only
  reused object is the PP-UKF frozen transport whose target signature and hash
  are checked.
- Proxy promotion risk: acceptance and runtime are explanatory or hard-screen
  diagnostics only. A pass means the engineering tuner completed, not that the
  PP-UKF posterior or HMC is valid.
- Missing stop condition risk: stop on any hard veto, artifact collision,
  target/hash drift, resource-policy failure, or missing terminal result. Do
  not retry with changed target, data, mass policy, or budget in this plan.
- Unfair comparison risk: no ranking or superiority claim is made.
- Stale-context risk: prior attempts are preserved and excluded from the fresh
  output root; the current commit is recorded in the new manifest.
- Environment risk: GPU visibility and memory growth are checked in trusted
  execution before TensorFlow work; CPU-only fallback is not allowed for this
  serious route.
- Artifact-answer risk: `result.json`, `run_manifest.json`, tuning artifact,
  progress, and run state answer whether the public tuner completed without
  launching sampling. A progress file alone is never treated as terminal
  evidence.

Audit verdict: `PASS_FOR_EXECUTION`.

## Compute Budget And Stop Conditions

- One PP-UKF tuning-only attempt under the serious public tuner budget.
- No sequential HMC draws and no retained posterior samples.
- One localized infrastructure retry is allowed only if the scientific scope,
  target, transport, mass policy, criteria, hardware class, and total budget
  remain unchanged; the retry must use another fresh root.
- A valid bootstrap or candidate failure is preserved as a failed tuning result
  and triggers analysis, not an automatic claim run.

## Commands

Trusted GPU preflight:

```bash
nvidia-smi
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python -c \
  'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
```

Tuning-only execution:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py \
  --action validate-frozen \
  --cell PP-UKF \
  --output-root docs/plans/artifacts/bayesfilter-pp-ukf-offline-tuning-only-20260721-01 \
  --frozen-transport \
  docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/campaign-01/PP-UKF/final/segments/steps-004001-005000/frozen_transport.json \
  --frozen-transport-sha256 \
  b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221 \
  --tuning-only
```

## Interpretation

Inspect hard vetoes first. If the run passes, classify it only as
`TUNING_ONLY_PASS`; it authorizes a separately planned untouched claim/HMC
phase but does not itself authorize or execute that phase. If it fails, retain
the result and diagnostics as a tuning failure; do not tune on claim data or
relax the fixed-identity route.

Execution result: the trusted GPU and XLA preflight passed, but the public
tuner was blocked before Phase 7 by the route contract because
`operational_interleaved_windowed_warmup_v2` is not validated with XLA. See
`docs/plans/bayesfilter-pp-ukf-offline-tuning-only-result-2026-07-21.md`.

## Required Result Note

After execution, record the actual command, commit, environment, GPU and
memory-policy fields, wall time, terminal status, hard vetoes, repair triggers,
artifact paths, and the next action in a reset/result note. Include the
strongest alternative explanation for failure and the evidence needed to
distinguish infrastructure failure from a PP-UKF tuning failure.
