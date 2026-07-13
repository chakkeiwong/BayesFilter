# Phase 6AA Subplan: SVD Score Wiring Retry

Date: 2026-07-10

## Phase Objective

Retry the deterministic LGSSM HMC tuning gate after demoting the
multidimensional LGSSM QR derivative route and wiring the active serious-HMC
target to the SVD/eigh graph-status score backend.

This phase must first refresh the XLA value/score gate. It may retry Phase 6
kernel tuning only if the refreshed XLA gate passes with `jit_compile=true`,
finite value/score, and valid SVD/eigh graph-status telemetry.

This is a compile/wiring retry only. It is not Phase 7 burn-in, retained
sampling, posterior recovery, HMC readiness, or scientific validation.

## Entry Conditions Inherited From Phase 6Z

- Phase 6Z reached final verification but aborted at `verification_start` with
  XLA CPU LLVM allocation errors before writing a refreshed `kernel_tuning.json`.
- `kernel_tuning.json` remains stale from Phase 6V and cannot support Phase 7.
- Phase 7 burn-in and retained sampling remain blocked.
- The active multidimensional LGSSM target has been rewired from
  `tf_qr_linear_gaussian_score` to
  `tf_svd_linear_gaussian_score_first_order_graph_status`.
- Focused SVD wiring tests have passed:
  `tests/test_multidim_triangular_lgssm_tf.py`,
  `tests/test_deterministic_lgssm_hmc_tuning_driver.py`, and
  `tests/test_svd_linear_gaussian_score_tf.py`.

## Required Artifacts

- This subplan.
- Updated runbook row for Phase 6AA.
- Updated execution ledger entry.
- SVD wiring result:
  `docs/plans/bayesfilter-multidim-lgssm-svd-score-wiring-demotion-result-2026-07-10.md`.
- Refreshed XLA gate artifact, if the gate runs:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/xla_compile_gate.json`.
- Refreshed Phase 6 artifact, if kernel tuning runs:
  `docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/kernel_tuning.json`.
- Phase 6AA result or blocker note:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6aa-svd-score-wiring-retry-result-2026-07-10.md`.

If either serious command aborts before writing its JSON artifact, the result
note must record the command, exit code, bounded stderr signature, last public
progress artifact hash when available, and whether Phase 7 remains blocked.

## Required Checks / Tests / Reviews

- Run static wiring checks before serious commands:
  - `python -m py_compile bayesfilter/testing/multidim_triangular_lgssm_tf.py docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py`
  - `git diff --check` on the Phase 6AA touched files.
  - scan active runtime files for `GradientTape`, `batch_jacobian`, `tape.`,
    `jit_compile=False`, and `jit_compile\s*=\s*False`.
  - scan the active LGSSM target/driver for stale
    `tensorflow_manual_lgssm_qr_score` and active
    `tf_qr_linear_gaussian_score` imports/calls.
- Run focused CPU-hidden tests:
  - `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_multidim_triangular_lgssm_tf.py`
  - `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_deterministic_lgssm_hmc_tuning_driver.py`
  - `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_svd_linear_gaussian_score_tf.py`
- Refresh the XLA gate only after checks pass:
  `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-svd-retry python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage xla_score`
- Retry kernel tuning only if the refreshed XLA gate has `passed=true`:
  `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-phase6-svd-retry python docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py --stage kernel_tuning`
- Claude read-only review is not required before the xla_score retry because
  the owner has explicitly instructed continuation and this phase is a narrow
  wiring/compile retry. If this phase produces a pass claim that would advance
  to Phase 7, stop at the Phase 7 approval boundary and review the result
  before any burn-in or retained sampling.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | After demoting the QR derivative route and wiring SVD/eigh graph-status scoring, can the XLA value/score gate and Phase 6 kernel-tuning gate be refreshed without the stale QR compile path? |
| Baseline/comparator | Phase 6Z with QR-derived serious target wiring: passed trajectory handoff, then XLA CPU LLVM allocation abort at final verification before refreshed `kernel_tuning.json`. |
| Primary pass criterion | Refreshed `xla_compile_gate.json` has `passed=true`, `jit_compile=true`, finite value/score, SVD graph-status valid; if kernel tuning runs, refreshed `kernel_tuning.json` has `passed=true`, XLA confirmed, no hard vetoes, and final kernel payload/hash. |
| Veto diagnostics | Any `jit_compile=False` fallback, runtime `GradientTape`, active QR derivative call in the serious LGSSM target/driver, invalid SVD status, nonfinite value/score, process abort without structured blocker, manual tuning, target/prior/fixture changes outside the SVD wiring repair, or Phase 7 sampling. |
| Explanatory diagnostics | XLA compile time, HLO/module size metadata, SVD min innovation eigenvalue, SVD condition estimate, target-status telemetry, last progress stage if kernel tuning aborts, and bounded XLA stderr signature. |
| Not concluded | No posterior convergence, posterior recovery, sampler superiority, production/default readiness, GPU readiness, DSGE readiness, or scientific claim. |
| Artifact preserving result | Phase 6AA result note plus refreshed JSON artifacts or structured blocker hashes. |

## Forbidden Claims / Actions

- Do not start Phase 7 burn-in or retained sampling.
- Do not run `jit_compile=False` or a non-XLA target-path fallback.
- Do not use runtime `GradientTape`.
- Do not manually tune step size, leapfrog count, mass matrix, budget, timeout,
  chain count, or evidence thresholds from observed diagnostics.
- Do not change the LGSSM prior, fixture, truth, mass artifact, R-hat threshold,
  total verification evidence gate, or final recovery criterion.
- Do not convert a compile crash, stale artifact, nonzero SVD status, or
  nonfinite score into a pass.
- Do not claim HMC readiness unless Phase 6 writes a refreshed `passed=true`
  kernel artifact and a separate Phase 7 approval/review boundary is satisfied.

## Exact Next-Phase Handoff Conditions

If the refreshed XLA gate fails or aborts, write the Phase 6AA blocker result
and stop. Phase 6 and Phase 7 remain blocked.

If the refreshed XLA gate passes but kernel tuning fails or aborts, write the
Phase 6AA blocker result with the last progress stage and stop. Phase 7 remains
blocked.

If kernel tuning writes a refreshed `passed=true` artifact with confirmed XLA,
no hard vetoes, and final kernel payload/hash, update the Phase 6 result and
stop at the Phase 7 approval boundary. Do not start burn-in or sampling without
explicit user approval.

## Stop Conditions

- Any required focused check fails.
- The refreshed XLA gate does not pass.
- The refreshed XLA gate or kernel tuning command aborts.
- The command writes an invalid JSON artifact or stale artifact only.
- SVD graph-status telemetry is invalid or missing.
- A QR derivative route remains active in the serious LGSSM target/driver.
- The next repair would require changing evidence thresholds, using non-XLA
  execution, or manually selecting tuning parameters.

## Skeptical Audit

Pass with constraints. This phase has a valid reason to rerun because the active
target wiring changed from QR derivative scoring to SVD/eigh graph-status
scoring. The misleading-pass risk is treating an XLA score-gate pass as HMC
readiness; the subplan blocks that by requiring a refreshed Phase 6 kernel
artifact before any Phase 7 boundary and forbidding burn-in/sampling inside
Phase 6AA.
