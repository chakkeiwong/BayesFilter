# NeuTra All-Executable-Models Attempt 01 Repair Review

Date: 2026-07-18

Campaign: `bayesfilter-neutra-all-executable-models-e2e-20260718`

Status: `PASS_TO_REAL_HMC_PREFLIGHT`

## Attempt 01 classification

Attempt 01 is preserved and must not be overwritten. LGSSM completed recipe
screening and fresh 5,000-step training, but its tuner reported
`acceptance_rate=null` for all leapfrog candidates. The first hard error was:

`target_status_telemetry missing required fields: floor_count_value, min_innovation_eigenvalue, innovation_condition_estimate`

This invalidates the harness, not NeuTra or the trained transport. The bound
batch target contained the required information; `BatchNativeBoundAdapter`
discarded it. The prior Phase 2 preflight injected a fake HMC chain and therefore
could not detect this integration failure. Attempt 01 later ended during PP-UKF
screening and has no aggregate result.

## Repairs

1. The bound adapter reuses BayesFilter's normalized batch value/status target
   and forwards its full mapping to HMC.
2. Preflight now invokes the real fixed-transport HMC runner and requires an
   error-free finite tuning row. Tiny budgets remain engineering evidence only.
3. Final training uses the reusable core segmented-training wrapper. Each
   segment has a fresh output directory, exact checkpoint lineage, unchanged
   scientific config hash, and terminal-only freeze.
4. Screening, held-out scoring, tuning, sampling, and convergence continue to
   use the existing BayesFilter implementations. No sampler, tuner, R-hat, ESS,
   or target-status validator was copied into the campaign runner.

## Checks

- `11 passed`: all-model registry, signature, binding, telemetry, policy, and
  duplication contracts.
- `4 passed`: uninterrupted/resumed/segmented deterministic training tests.
- Segmented and uninterrupted weights plus both Adam moment sets are bitwise
  equal on the deterministic fixture.
- `py_compile` and scoped `git diff --check` pass.
- The end-to-end runner has no NumPy import, direct TFP sampler construction,
  copied R-hat/ESS call, fake chain, or historical benchmark import.

## Skeptical relaunch audit

Wrong baseline and proxy promotion remain prevented by the plan. The repaired
preflight now answers the actual integration question that attempt 01 failed:
can the trained transport enter BayesFilter's real native tuning/HMC route with
complete status telemetry? Segmentation changes only infrastructure boundaries,
not the optimization trajectory or scientific budget. No acceptance,
convergence, truth recovery, or model ranking claim is supported until a fresh
serious attempt emits those artifacts.

Verdict: launch one fresh real-HMC GPU/XLA preflight. If it reports any HMC
exception, missing/nonfinite tuning statistic, GPU/XLA/memory-growth violation,
or target-status failure, stop before serious attempt 02. Otherwise rerun the
implementation/duplication audit once and launch attempt 02.
