# Phase 2V Subplan: Longer Selected MAP-Local Screen

Date: 2026-07-09
Status: `DRAFT_PENDING_REVIEW`

## Phase Objective

Run a longer CPU-hidden finite/acceptance screen for the Phase 2U selected
MAP-local fixed kernel: `num_leapfrog_steps=2`, `step_size=0.785`,
trajectory length `1.57`.  This phase checks whether the selected short-screen
candidate remains finite and inside the acceptance envelope with more retained
draws before any GPU/XLA or posterior-agreement phase is considered.

This phase is not a posterior agreement phase, not a convergence certification,
and not a GPU/XLA/default-readiness phase.

## Entry Conditions

- Phase 2U passed and wrote:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-result-2026-07-09.md`.
- Phase 2U selected candidate 0 by the predeclared first-passing rule:
  - leapfrog steps: `2`;
  - step size: `0.785`;
  - trajectory length: `1.57`;
  - short-screen acceptance: `0.34375`.
- Phase 2U native divergence telemetry was unavailable for all candidates and
  did not support a zero-divergence claim.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2V harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py`
- Phase 2V tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py`
- Phase 2V JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2v_longer_selected_map_local_screen.log`
- Phase 2V result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-result-2026-07-09.md`

## Required Checks, Tests, And Reviews

- Review this subplan before runtime.  If Claude remains unavailable for
  repo-context review, use a fresh Codex substitute reviewer and record that it
  is weaker than full Claude material review.
- Run focused tests before runtime:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.py
```

- Run `git diff --check`.
- Create the quiet log directory before redirected runtime.
- Planned runtime command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2v_longer_selected_map_local_screen.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the Phase 2U selected MAP-local fixed kernel remain finite and inside the acceptance envelope with 128 retained draws and 8 burn-in draws? |
| Baseline/comparator | Phase 2U selected candidate 0 and its short-screen artifact. |
| Primary pass criterion | The selected kernel starts at `u_new = 0`, produces finite retained samples, finite target-log-prob trace, finite log-accept ratios, no positive native divergence if native telemetry is available, and acceptance strictly inside `(0.05, 0.99)`. |
| Veto diagnostics | Runtime error, invalid Phase 2U artifact, selected candidate mismatch, initial state not exactly `u_new = 0`, invalid MAP-local adapter, nonfinite initial target/score, nonfinite samples/target/log accept, positive native divergence when available, acceptance outside envelope, invalid artifact, timeout, or unsupported claim. |
| Explanatory diagnostics | Acceptance, log-accept summaries, target-log-prob summaries, sample ranges, runtime, and native-divergence availability status. |
| Not concluded | No posterior correctness, HMC readiness/convergence, zero divergences when native telemetry is unavailable, sampler superiority, statistically supported ranking, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2V JSON, Markdown, quiet log, result file, and refreshed handoff. |

## Fixed Runtime Settings

- `num_leapfrog_steps=2`
- `step_size=0.785`
- `trajectory_length_L_times_epsilon=1.57`
- `num_results=128`
- `num_burnin_steps=8`
- initial state `u_new = [0, 0, 0, 0]`, the Phase 2S MAP-local center in the
  HMC coordinate;
- seed `(20260709, 6401)`
- `trace_policy="standard"`
- `adaptation_policy="fixed_kernel_no_adaptation"`
- `chain_execution_mode="eager"`
- `CUDA_VISIBLE_DEVICES=-1`

## Forbidden Claims And Actions

- Do not run GPU/XLA in Phase 2V.
- Do not change defaults or public API behavior.
- Do not retune after seeing results.
- Do not treat unavailable native divergence telemetry as zero divergences.
- Do not use log-accept thresholds as native-divergence telemetry.
- Do not claim posterior correctness, HMC readiness/convergence, sampler
  superiority, statistically supported ranking, default readiness, or
  Zhao-Cui source faithfulness.

## Exact Next-Phase Handoff Conditions

If Phase 2V passes, write the Phase 2V result and draft a reviewed scalar
reference/posterior-agreement subplan.  Phase 2V finite/acceptance evidence
alone is not sufficient to draft or run Phase 3 GPU/XLA reproduction.  Phase 3
remains blocked until a later reviewed reference/posterior-agreement result
explicitly authorizes a GPU/XLA reproduction subplan and preserves all
nonclaims.

If Phase 2V fails, write a blocker/repair result and draft only a narrower
repair subplan if the failure is implementation or tuning-localization
evidence rather than target invalidity.

## Stop Conditions

Stop for invalid Phase 2U artifact, selected candidate mismatch, initial state
other than `u_new = 0`, invalid MAP-local affine adapter, runtime exception,
timeout, nonfinite target or samples, positive native divergence when
available, acceptance outside the envelope, review nonconvergence, or any need
to cross GPU, default-policy, model-file, source-faithfulness, or
scientific-claim boundaries.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the selected Phase 2U MAP-local candidate, not old Phase 1R or a post-hoc candidate. |
| Proxy metrics promoted | Finite/acceptance evidence remains a longer screen only, not posterior correctness or readiness. |
| Missing stop conditions | Artifact, selected-candidate, telemetry, acceptance, runtime, review, and claim-boundary stops are explicit. |
| Unfair comparison | No method ranking occurs. |
| Hidden assumptions | The selected candidate is first-passing by contract, not statistically best. |
| Stale context | Phase 2V must reload current Phase 2U JSON before runtime. |
| Environment mismatch | CPU-hidden non-XLA evidence cannot support GPU/XLA/default-readiness claims. |
| Artifact mismatch | JSON/Markdown/result/log paths and settings are predeclared. |

Audit status: `PASSED_FOR_REVIEW_ONLY`.  Runtime may begin only after review
converges.
