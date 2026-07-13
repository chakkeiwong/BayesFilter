# Phase 6U Subplan: Kernel Mechanics Nonfinite Log-Accept Diagnostic

Date: 2026-07-09

## Phase Objective

Classify the true fixed-mass HMC mechanics source of the Phase 6 hard veto:
full trace `log_accept_ratio` is nonfinite for every screen transition while
accepted target log-probability and samples are finite. The goal is a bounded
diagnostic repair, not a pass relaxation and not a sampling run.

## Entry Conditions Inherited From Previous Phase

- Phase 5 geometry and mass artifacts passed.
- Phase 6R adapter repair passed focused tests.
- Phase 6S removed the XLA compile-abort blocker.
- Phase 6T proved the hard veto is not missing telemetry:
  - `log_accept_ratio_diagnostic_source="trace"`;
  - `log_accept_ratio_finite=false`;
  - latest fixed-mass screens have 500 nonfinite log-accept entries;
  - target log-probability and samples are finite.
- Phase 7 burn-in and retained sampling remain blocked.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6u-kernel-mechanics-diagnostic-subplan-2026-07-09.md`
- Refreshed Phase 6 result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- Updated runbook:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md`
- Updated code if needed:
  `bayesfilter/inference/hmc.py`,
  `bayesfilter/inference/hmc_budget_ladder.py`,
  `bayesfilter/inference/hmc_kernel_tuning.py`
- Focused tests:
  `tests/test_hmc_budget_ladder.py`
  and/or `tests/test_hmc_kernel_tuning_fixed_mass_step.py`
- Refreshed private diagnostic event stream if rerun:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning_public/private_diagnostics/hmc_tuning_events.jsonl`

## Required Checks / Tests / Reviews

- Add private-only HMC mechanics summaries sufficient to distinguish:
  - proposed target log-probability nonfinite;
  - current/accepted target log-probability nonfinite;
  - HMC log-acceptance correction nonfinite;
  - the final summed log-acceptance ratio nonfinite despite finite components.
- Preserve the hard veto when any required component is missing or nonfinite.
- Do not expose step size, leapfrog count, mass matrix, raw samples, raw states,
  or raw trace tensors in public artifacts.
- Run focused tests:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py`
  and
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_kernel_tuning_fixed_mass_step.py`.
- Run deterministic driver tests:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py`.
- Run `git diff --check` on touched files.
- Scan touched runtime files for forbidden non-XLA fallback and runtime
  `GradientTape` tokens.
- Use a bounded one-path Claude read-only review of this subplan before code
  execution when Claude is available.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Which HMC mechanics component makes the fixed-mass screen log-acceptance ratio nonfinite under the deterministic LGSSM kernel tuning gate? |
| Baseline/comparator | Phase 6T result with full trace log-accept evidence: all latest screens have 0 finite and 500 nonfinite log-accept entries while accepted target log-probability and samples are finite. |
| Primary pass criterion | Focused tests pass and a Phase 6 rerun produces private event summaries that classify the nonfinite source without relaxing the hard veto. |
| Veto diagnostics | Non-XLA fallback, runtime `GradientTape`, manual step/leapfrog/mass changes, public leakage of HMC mechanics, treating nonfinite log-accept as acceptable, invalid JSON artifacts, or process abort before structured result. |
| Explanatory diagnostics | Proposed target log-probability finite counts, accepted/current target finite counts, log-acceptance-correction finite counts, log-accept-ratio finite counts, target-status telemetry, candidate-grid progress. |
| Not concluded | No posterior convergence, posterior recovery, sampler superiority, production/default readiness, GPU readiness, or scientific claim. |
| Artifact preserving result | Refreshed Phase 6 result plus private `hmc_tuning_events.jsonl` summaries and file hashes. |

## Forbidden Claims / Actions

- Do not start Phase 7 burn-in or retained sampling.
- Do not run `jit_compile=False` or a non-XLA fallback.
- Do not manually choose new HMC budgets, step sizes, leapfrog counts, mass
  matrices, or acceptance thresholds.
- Do not demote the nonfinite log-accept hard veto to a repair trigger.
- Do not infer posterior failure or LGSSM target failure from this diagnostic
  alone.

## Exact Next-Phase Handoff Conditions

If Phase 6U identifies a fixable implementation or telemetry gap, write the
smallest next Phase 6 repair subplan and keep Phase 7 blocked.

If Phase 6U identifies deterministic step/mass scale failure without an
implementation bug, the next action is a deterministic tuning-policy repair
subplan, not manual tuning.

Phase 7 may start only after a refreshed Phase 6 result records a final kernel
payload/hash with `passed=true`, confirmed XLA/JIT, and no hard vetoes, followed
by separate explicit runtime approval.

## Stop Conditions

- The diagnostic requires changing the LGSSM target, fixture, prior, mass
  artifact, or deterministic tuning policy.
- The diagnostic requires non-XLA execution.
- The diagnostic cannot preserve public/private artifact boundaries.
- Focused tests fail.
- The rerun process aborts before writing a structured result artifact.

## Skeptical Audit

Pass. Phase 6T converted the blocker from a telemetry ambiguity into a true
HMC log-acceptance mechanics hard veto. This Phase 6U plan targets the next
unknown component directly and does not use proxy metrics as promotion
criteria, does not cross into sampling, and does not authorize manual tuning.
