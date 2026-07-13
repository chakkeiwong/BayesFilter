# Phase 2AB Result: Transport Or Sequential Reference Pilot

Date: 2026-07-09
Status: `FAILED_NO_REFERENCE_NOMINATION_REPAIR_SUBPLAN_DRAFTED`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-subplan-2026-07-09.md`

Runtime JSON:
`docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_cpu_hidden_2026-07-09.json`

Runtime Markdown:
`docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_cpu_hidden_2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2AB did not nominate a sequential reference route for independent replication | Failed: the repaired cumulative-ESS schedule stalled at beta `0.3419540270406287` and did not reach beta `1.0` | Promotion vetoes: `temperature_increment_stalled`, `temperature_schedule_did_not_reach_beta_one`, `terminal_beta_not_one` | The result may indicate a narrow resampling/schedule policy defect or a deeper bridge mismatch; it is not evidence against the target or HMC mechanics | Draft Phase 2AC focused sequential-resampling repair subplan; stop before runtime until that subplan is reviewed/cleared | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered negatively for the reviewed Phase 2AB sequential pilot: it did not clear the nomination gate. |
| Candidate under test | CPU-hidden sequential tempering pilot in Phase 2S/2U `u_new`, base `N(0,I_4)`, 128 particles, adaptive beta, systematic resampling, and one RWM rejuvenation move per stage. |
| Baseline/comparator | Phase 2W/2X/2Z failed independent proposal branch and Phase 2Y proposal-family/global-geometry localization. |
| Primary criterion | Failed because beta did not reach `1.0`; no Phase 2AC replication nomination. |
| Veto diagnostics | Promotion vetoes fired for temperature stall and terminal beta below one. |
| Explanatory diagnostics | Temperature schedule, ESS trajectory, max weights, ancestor diversity, rejuvenation acceptance, and terminal weighted moments were recorded. |
| Not concluded | No reference validity, posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or source faithfulness. |

## Final Repaired Runtime Diagnostics

The final artifact is the run created at `2026-07-09T07:50:04.596318+00:00`.
Earlier Phase 2AB attempts were superseded by implementation repairs: first,
top-level decision veto reporting omitted promotion-gate failures; second, beta
selection used incremental rather than cumulative ESS.  Those were harness
defects, not scientific evidence, and focused tests were added before rerunning.

| Diagnostic | Value | Role |
| --- | --- | --- |
| Runtime exit | `0` | Artifact validity |
| Wall time | `59.509127364028245` seconds | Explanatory |
| Terminal beta | `0.3419540270406287` | Promotion veto |
| Stage count | `2` | Explanatory |
| Stage 0 beta | `0.2527170181274414` with ESS ratio `0.7000013562683505` | Explanatory |
| Stage 1 beta | `0.3419540270406287` with ESS ratio `0.5000012436397635` | Explanatory / stall localization |
| Terminal pre-final-resampling ESS | `64.00015918588973` | Promotion diagnostic |
| Terminal pre-final-resampling ESS ratio | `0.5000012436397635` | Promotion diagnostic |
| Terminal pre-final-resampling max weight | `0.06553946724924992` | Promotion diagnostic |
| Unique ancestor fraction | `1.0` | Promotion diagnostic |
| Aggregate rejuvenation acceptance | `0.625` | Promotion diagnostic |
| Candidate nominated | `False` | Primary Phase 2AB decision |

## Checks

| Check | Status |
| --- | --- |
| Claude review round 1 | `VERDICT: REVISE`; terminal measurement, provenance, beta bisection, and measurement-point issues patched. |
| Focused local review | `VERDICT: AGREE_FOR_IMPLEMENTATION_AND_CPU_HIDDEN_PILOT`. |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot.py` | Passed after final scheduler repair: `17 passed`. |
| `python -m py_compile docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_2026_07_09.py` | Passed. |
| `git diff --check` | Passed before final runtime. |
| Final Phase 2AB runtime command | Exited `0`; artifact decision failed nomination gate. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `52ee244498988e046a6356f926003b581103083b` |
| Git dirty status | Dirty; artifact records planned scalar HMC validation edits and unrelated user work. |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seed | `(20260709, 6801)` |
| Output artifacts | JSON, Markdown, and quiet log under `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2ab_transport_or_sequential_reference.log` |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Failed for Phase 2AB nomination because terminal beta did not reach one. |
| Reference validity | Not established. |
| HMC-reference agreement | Not assessed. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Temperature schedule, ESS trajectory, ancestor diversity, acceptance, terminal moments, and runtime. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed Phase 2AC focused sequential-resampling repair subplan or a reference-method blocker. |

## Post-Run Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | The failure may be a narrow resampling/schedule policy issue: the pilot reached the minimum ESS boundary with preserved ancestor diversity and acceptable rejuvenation acceptance, then could not take another admissible beta step. |
| What would overturn | A reviewed repair that resamples exactly at the minimum/fallback boundary and reaches beta `1.0` while preserving diversity and terminal weight gates, followed by independent replication. |
| Weakest evidence | One CPU-hidden seed, 128 particles, and repair-generated diagnostics; no uncertainty analysis and no valid reference claim. |

## Final Nonclaims

- No valid independent reference.
- No valid sequential reference.
- No HMC-vs-reference agreement.
- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
