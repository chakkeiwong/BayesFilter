# HMC Rank-Normalized Tuning Verifier Repair Plan

Date: 2026-07-13

Status: `COMPLETE`

## Objective

Repair the deterministic HMC tuning verifier so that its R-hat handoff screen
uses the same modern statistic as the Phase 7 convergence screen:

`max(rank-normalized split R-hat, folded rank-normalized split R-hat)`.

The repair is code- and test-only. It does not rerun tuning, retry Phase 7,
alter the historical Phase 7 result, or authorize Phase 8 or NeuTra work.

## Entry Conditions

- The historical tuning verifier passed a classical split-free R-hat screen.
- Phase 7 later failed only the folded rank-normalized split R-hat component
  for eight parameters at the burn-in cap.
- The owner confirmed that tuning and testing must use the same modern
  rank-normalized split plus folded R-hat definition.
- The semantic-identity migration runbook is closed; any repair therefore
  requires a new plan rather than an implicit retry.

## Skeptical Plan Audit

| Risk | Audit result |
| --- | --- |
| Wrong baseline | The comparator is the existing Phase 7 implementation in `bayesfilter/inference/hmc_convergence.py`, not the historical tuning result. |
| Proxy promoted to scientific evidence | R-hat remains a tuning handoff screen here; no posterior-convergence claim is allowed. |
| Missing stop condition | Stop after shared-code equivalence and focused tests pass, or on an unresolved regression/API incompatibility. |
| Unfair comparison | A deterministic scale-mismatch fixture will exercise both tuning and Phase 7 diagnostics on identical draws. |
| Hidden assumption | Modern R-hat means the maximum of rank-normalized split and folded rank-normalized split components. |
| Stale context | The terminal Phase 7 result and current source implementations were inspected on 2026-07-13. |
| Environment mismatch | Checks are deliberate CPU-only diagnostics with `CUDA_VISIBLE_DEVICES=-1`; no GPU claim is made. |
| Artifact insufficiency | The result note must record code paths, regression outputs, commands, and test results. |

Audit verdict: `PASS_FOR_NARROW_CODE_REPAIR`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does tuning compute and threshold the exact same modern R-hat statistic as Phase 7? |
| Comparator | `rank_normalized_hmc_diagnostics` in `bayesfilter/inference/hmc_convergence.py`. |
| Primary criterion | Both paths consume one shared R-hat implementation and return the same maximum R-hat for identical draws. |
| Vetoes | Classical/split-free R-hat remains in the tuning path; folded R-hat is omitted; nonfinite diagnostics can pass; focused tests fail. |
| Explanatory only | Runtime and individual synthetic component values. |
| Not concluded | The historical kernel now passes tuning, Phase 7 convergence, posterior recovery, sampler superiority, or production/default readiness. |
| Preserving artifact | A result note under `docs/plans` plus focused test output. |

## Required Changes

1. Extract a shared rank-normalized split/folded R-hat summary from the Phase 7
   convergence module.
2. Make the full Phase 7 diagnostic consume that shared summary.
3. Replace the tuning verifier's classical split-free calculation with the
   shared summary while retaining its existing public aggregate fields.
4. Record the diagnostic definition and component maxima in tuning metadata.
5. Add a regression fixture with aligned chain locations and unequal chain
   scales. It must pass the location component, fail the folded component, and
   be rejected by both paths with the same maximum R-hat.

## Required Checks

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-rhat-repair \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -m pytest -q \
  tests/test_hmc_convergence.py \
  tests/test_hmc_fixed_size_chunk_runner.py

CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-rhat-repair \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -m pytest -q \
  tests/test_hmc_kernel_tuning_outer_loop.py \
  tests/test_deterministic_lgssm_hmc_phase7_tf.py

/home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -m py_compile \
  bayesfilter/inference/hmc.py bayesfilter/inference/hmc_convergence.py
```

## Forbidden Claims And Actions

- Do not rewrite historical tuning or Phase 7 artifacts.
- Do not describe the old tuning pass as a valid modern-R-hat pass.
- Do not rerun serious tuning or Phase 7 under this plan.
- Do not change acceptance, ESS, burn-in, retained-sample, or scientific gates.
- Do not infer that the selected kernel would fail a repaired tuning rerun; that
  remains untested until a separately planned run uses the repaired code.

## Handoff And Stop Conditions

On success, write a repair result that classifies the implementation bug as
fixed and states that all prior tuning artifacts predate the fix. Any new
tuning or serious sampling requires a separate experiment plan and fresh
artifacts.

Stop blocked if the shared statistic cannot preserve Phase 7 behavior, focused
tests reveal an unresolved contract conflict, or another lane changes an
in-scope file during this repair.
