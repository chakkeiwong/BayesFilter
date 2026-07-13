# Phase 6T Subplan: Log-Accept Telemetry Root-Cause Repair

Date: 2026-07-09

## Phase Objective

Repair or precisely classify the fixed-mass screen
`screen_log_accept_nonfinite_or_missing` blocker from Phase 6S without
weakening the hard veto and without changing the LGSSM target, fixture, mass
artifact, deterministic tuning policy, or XLA-only rule.

## Entry Conditions Inherited From Previous Phase

- Phase 5 geometry and mass artifacts passed.
- Phase 6R adapter repair passed focused tests.
- Phase 6S eliminated the XLA CPU code-generation process abort and produced a
  structured `kernel_tuning.json` result.
- The latest Phase 6 result has `xla_confirmed=true`, `passed=false`,
  `final_status=hard_veto`, no final kernel payload/hash, and hard veto
  `screen_log_accept_nonfinite_or_missing`.
- Phase 7 burn-in and retained sampling remain blocked.

## Required Artifacts

- This subplan:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6t-log-accept-telemetry-repair-subplan-2026-07-09.md`
- Refreshed Phase 6 result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md`
- Updated code if repair is needed:
  `bayesfilter/inference/hmc_budget_ladder.py`
  and/or narrowly related HMC diagnostic code.
- Focused tests:
  `tests/test_hmc_budget_ladder.py`
- Driver regression tests:
  `tests/test_deterministic_lgssm_hmc_tuning_driver.py`
- Refreshed result if the repair passes local checks:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`
  and an updated Phase 6 result note with a run manifest and decision table.

## Required Checks / Tests / Reviews

- Add or update focused tests proving:
  - full `trace["log_accept_ratio"]` remains the preferred diagnostic source;
  - if full trace and summary diagnostics disagree, the full trace governs and
    nonfinite trace values remain a hard veto;
  - missing full log-accept trace remains a hard veto unless an explicit finite
    summary diagnostic is present;
  - a summary diagnostic may mark log-accept finite only when finite-count,
    nonfinite-count, and max-absolute-finite evidence are internally
    consistent;
  - nonfinite summary evidence remains a hard veto;
  - no `jit_compile=False`, runtime `GradientTape`, or manual HMC parameter
    selection is introduced.
- Run focused tests:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_hmc_budget_ladder.py`
- Run deterministic driver tests:
  `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py`
- Validate JSON artifacts:
  `python -m json.tool` on progress/result artifacts.
- Run `git diff --check` on touched files.
- Use a bounded one-path Claude read-only review of this subplan when
  available. Claude is not an execution authority.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the fixed-mass screen hard veto caused by true nonfinite log-acceptance evidence, or by missing telemetry propagation from runner diagnostics into budget-ladder diagnostics? |
| Baseline/comparator | Phase 6S structured result with 14 of 14 candidates hard-vetoed by `screen_log_accept_nonfinite_or_missing`. |
| Primary pass criterion | Focused tests pass, the hard veto remains fail-closed for missing/nonfinite evidence, and a Phase 6 rerun either produces a final kernel artifact or a more specific structured hard-veto reason. |
| Veto diagnostics | Non-XLA fallback, runtime `GradientTape`, manual HMC parameter choice, treating missing telemetry as finite, changing target/mass/fixture, invalid JSON artifacts, or process abort before structured result. |
| Explanatory diagnostics | Trace keys, trace unavailability, log-accept finite/nonfinite counts, max finite log-accept magnitude, target-log-prob finite status, candidate-grid progress. |
| Not concluded | No posterior convergence, posterior recovery, sampler superiority, production/default readiness, GPU readiness, or scientific claim. |
| Artifact preserving result | Refreshed Phase 6 result and `kernel_tuning.json` / `hmc_kernel_tuning_progress.json` if rerun. |

## Forbidden Claims / Actions

- Do not run `jit_compile=False` or a non-XLA fallback.
- Do not manually choose step size, leapfrog count, candidate grids, burn-in
  budgets, or sample budgets outside BayesFilter-owned deterministic APIs.
- Do not convert missing log-accept telemetry into a pass.
- Do not ignore or downgrade nonfinite log-accept evidence.
- Do not start Phase 7 burn-in or retained sampling.

## Exact Next-Phase Handoff Conditions

Phase 7 remains blocked unless a refreshed Phase 6 result records a final
kernel payload/hash with `passed=true`, confirmed XLA/JIT, and no hard vetoes.

If Phase 6T only produces a more specific structured hard veto, the next step
is another Phase 6 repair subplan or a human decision about target/tuning
scope, not Phase 7.

## Stop Conditions

- Focused tests fail.
- The repair requires changing the LGSSM target, prior, fixture, mass artifact,
  or deterministic tuning policy.
- The repair requires non-XLA execution.
- The repair requires treating absent or nonfinite log-accept evidence as
  finite.
- The rerun process aborts before producing a structured result artifact.

## Skeptical Audit

Pass. The next action targets the actual observed blocker from the latest
structured result. It does not reinterpret a hard veto as sampler evidence, does
not relax the log-accept hard screen, and does not advance toward posterior
sampling without a valid kernel handoff.
