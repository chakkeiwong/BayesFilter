# Phase 4 Subplan: Preserved-Transport LGSSM Validation

Status: `READY_AFTER_FOCUSED_CHECKS`

## Objective

Validate the repaired public fixed-identity tuner and sequential NeuTra HMC on
the exact 18D LGSSM target, reusing the fully trained current-target transport
without retraining or changing the scientific target.

## Entry Conditions

- Phase 3 closes `PASS_ENGINEERING_ROUTE`.
- Focused public-tuner, replay, adapter, and NeuTra orchestration tests pass.
- GPU/XLA and memory growth are available in a trusted execution context.
- The output root is fresh and versioned.
- The preserved transport file matches SHA-256
  `b0b89656b2503146556f50b4e5e3e0e6b9b63daf0673380043ccb046dd14877e`.
- A second sampling-seed attempt is admissible only if deterministic public
  retuning reproduces Attempt 01 final-kernel hash
  `e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc`
  exactly. A mismatch must stop before sequential sampling with
  `TUNING_REPLAY_HASH_MISMATCH`.

## Preserved Input

Path:
`docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-attempt-02/LGSSM-EXACT/final/segments/steps-004001-005000/frozen_transport.json`

This artifact completed 5,000 GPU/XLA optimizer steps, is bound to current
target signature
`bd40a828bc4916e5e09a8e6135f315ebc45c06844aed38a506d6296c2642557d`,
was selected from the target-specific four-recipe screen, passed the held-out
training criterion, and had exact frozen/trainable value-score parity. Its
historical `TUNING_FAILED` result came from the superseded specialized tuner and
does not invalidate the frozen transport as input to this repaired test.

## Research Intent Ledger

| Role | Decision |
| --- | --- |
| Main question | Does the preserved current-target transport pass canonical public tuning, adaptive warm-up, retained convergence, and the one-seed truth-tail criterion? |
| Candidate | The preserved 5,000-step wide dense-IAF transport with fixed identity mass in `z`. |
| Expected failure | The public tuner may fail acceptance/health admission, warm-up may fail folded/rank-normalized split R-hat by 10,000, retained sampling may fail R-hat/ESS by 10,000, or truth-tail may fail. |
| Promotion criterion | Public tuner passes; acceptance is in `[0.65,0.75]`; mass signature remains fixed identity; warm-up passes modern folded/rank-normalized R-hat; retained samples pass R-hat/ESS; every parameter has `p_truth >= 0.05`. |
| Promotion veto | Any failed acceptance, convergence, ESS, truth-tail, target, mass-lineage, or required-artifact gate. |
| Continuation veto | Invalid target/artifact, nonfinite target or gradient, mass mutation, broken replay lineage, missing diagnostics, GPU/XLA policy failure, or exhausted two-attempt campaign budget. |
| Repair trigger | Local harness, serialization, replay-lineage, or resource error under unchanged target/method/criteria/hardware/budget. Candidate statistical failure is recorded, not silently repaired by changing the method. |
| Explanatory only | Runtime, chosen step size/leapfrog count, per-check trajectories, and posterior descriptive summaries. |
| Forbidden conclusion | No universal validity, sampler superiority, distributional equivalence, or default-readiness claim. |

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Preserved wide transport | July 18 target-specific four-recipe screen | Training criterion may not imply HMC suitability | Public tuning and downstream convergence | Reviewed candidate, not default |
| Fixed identity mass in `z` | NeuTra coordinate contract and Phase 2 public option | Transport may not whiten sufficiently | Acceptance/health and convergence | Required experimental arm |
| Acceptance 0.70, band `[0.65,0.75]` | Repository sampler policy | Wrong tuning if route drifts | Public config and final handoff assertions | Fixed reviewed policy |
| Warm-up R-hat max 1.05 | Existing NeuTra sequential controller | 1,000-draw window may be noisy | Retain every chunk/check through 10,000 cap | Reviewed convergence policy |
| Retained R-hat max 1.01, ESS bulk 1000, tail 400 | Existing full convergence diagnostic | Short run may not meet evidence threshold | Extend in 1,000-draw chunks through 10,000 | Reviewed convergence policy |
| One seed | Owner diagnostic policy | Cannot support broad ranking or universal claim | Explicit nonclaim and marginal rerun rule | Diagnostic evidence only |

## Required Artifacts And Checks

- hash-bound frozen-transport input record;
- public tuning progress/result and exact final-kernel replay lineage;
- retained warm-up and posterior sample tensors, including all chunks;
- per-check folded/rank-normalized split R-hat, ESS, health, acceptance, and
  energy diagnostics;
- one-seed truth-tail table;
- run manifest with Git commit, command, environment, GPU/XLA/memory policy,
  seeds, wall time, input/output paths, plan, and result;
- Phase 4 result, next-subplan review, terminal review, and reset memo.

## Exact Launch

Attempt 01 used the following command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py \
  --action validate-frozen \
  --cell LGSSM-EXACT \
  --output-root docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt01 \
  --frozen-transport docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-attempt-02/LGSSM-EXACT/final/segments/steps-004001-005000/frozen_transport.json \
  --frozen-transport-sha256 b0b89656b2503146556f50b4e5e3e0e6b9b63daf0673380043ccb046dd14877e
```

Attempt 02 changes only the sequential warm-up and retained seeds. The public
tuner seed remains `(20260621, 8)`, and the exact kernel-hash gate proves that
the admitted step size, trajectory length, mass, and tuning handoff are
unchanged before sampling begins:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py \
  --action validate-frozen \
  --cell LGSSM-EXACT \
  --output-root docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt02 \
  --frozen-transport docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-attempt-02/LGSSM-EXACT/final/segments/steps-004001-005000/frozen_transport.json \
  --frozen-transport-sha256 b0b89656b2503146556f50b4e5e3e0e6b9b63daf0673380043ccb046dd14877e \
  --expected-tuning-final-kernel-hash e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc \
  --seed-offset 1000
```

## Budget, Repair, And Stop Conditions

Budget: one serious first-seed launch plus one second-seed launch when the
owner's marginal rule fires (`0.003 <= p_truth < 0.05`). The second seed keeps
the target, transport, public tuner seed, admitted final-kernel hash, tuning
policy, thresholds, and hardware unchanged. It is a pure sampling-seed
replication only after the exact final-kernel hash gate passes. A
localized harness repair may replace, but not add to, the remaining launch.
Every retry uses a fresh attempt root and preserves the failure, repair,
focused regression, and remaining budget.

Stop on a true continuation veto or exhausted budget. A failed candidate is a
Phase 4 negative result; do not change the target, transport, mass policy,
thresholds, hardware class, or evidence criterion to manufacture a pass.

At close: run focused checks, write the Phase 4 result, refresh Phase 5 only if
LGSSM mechanics validate, review that next step, and continue when no real
blocker remains.
