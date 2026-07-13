# Phase 2AD Result: Diversity-Preserving Sequential Repair

Date: 2026-07-09
Status: `FAILED_NO_REFERENCE_NOMINATION_BLOCKER_OR_EXPANSION_TRIGGERED`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-subplan-2026-07-09.md`

Runtime JSON:
`docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.json`

Runtime Markdown:
`docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.md`

Quiet log:
`docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2ad_diversity_preserving_sequential_repair.log`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2AD did not nominate a sequential reference route for independent replication | Failed: diversity was preserved, but terminal beta stalled at `0.9712250668187553` and did not reach `1.0` | Promotion vetoes: `temperature_increment_stalled`, `temperature_schedule_did_not_reach_beta_one`, `terminal_beta_not_one` | The scalar bridge shows a local tradeoff: Phase 2AC reaches beta one by spending diversity; Phase 2AD preserves diversity by skipping the last low-diversity resampling draw but then cannot complete the bridge | Draft a reference-method blocker or expansion decision; keep Phase 3 GPU/XLA blocked | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered negatively for the Phase 2AD nomination gate. |
| Candidate under test | Same Phase 2AC target route, seed, particle count, beta thresholds, and RWM rejuvenation, except fallback resampling was accepted only if the realized systematic-resampling draw preserved projected root-ancestor fraction at least `0.25`. |
| Baseline/comparator | Final Phase 2AC artifact. |
| Primary criterion | Failed because beta did not reach `1.0`. |
| Veto diagnostics | Temperature progression vetoes fired after the final low-diversity draw was skipped. |
| Repair trigger | None inside this phase by reviewed subplan; write blocker or expansion decision. |
| Explanatory diagnostics | Stage schedule, projected diversity, resampling accept/skip decisions, ESS trajectory, max weights, ancestor diversity, acceptance, and runtime were recorded. |
| Not concluded | No reference validity, posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Diagnostics

| Diagnostic | Value | Role |
| --- | --- | --- |
| Runtime exit | `0` | Artifact validity |
| Wall time | `75.56550153600983` seconds | Explanatory |
| Terminal beta | `0.9712250668187553` | Promotion veto |
| Stage count | `6` | Explanatory |
| Terminal pre-final-resampling ESS ratio | `0.5000016429467714` | Promotion diagnostic passed |
| Terminal pre-final-resampling max weight | `0.03922774202134582` | Promotion diagnostic passed |
| Minimum adaptive post-temperature ESS ratio | `0.5000010206048837` | Promotion diagnostic passed |
| Unique ancestor fraction | `0.4140625` | Promotion diagnostic passed |
| Aggregate rejuvenation acceptance | `0.6458333333333334` | Promotion diagnostic passed |
| Candidate nominated | `False` | Primary decision |

## Stage Schedule

| Stage | Beta | Rule | ESS ratio | Max weight | Considered | Accepted | Projected diversity | Resampled | Skip reason | Unique ancestor fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `0.2527170181274414` | `bisection_largest_target_admissible_increment` | `0.7000013562683505` | `0.04017182623114065` | `False` | `False` | `None` | `False` | `not_at_resampling_boundary` | `1.0` |
| 1 | `0.3419540270406287` | `bisection_largest_minimum_admissible_increment` | `0.5000012436397635` | `0.06553946724924992` | `True` | `True` | `0.71875` | `True` | `None` | `0.71875` |
| 2 | `0.5811528746866764` | `bisection_largest_target_admissible_increment` | `0.7000006501357879` | `0.023202415520545847` | `False` | `False` | `None` | `False` | `not_at_resampling_boundary` | `0.71875` |
| 3 | `0.6735977385780526` | `bisection_largest_minimum_admissible_increment` | `0.5000010206048837` | `0.041457723731129006` | `True` | `True` | `0.4140625` | `True` | `None` | `0.4140625` |
| 4 | `0.8828990299946569` | `bisection_largest_target_admissible_increment` | `0.700000256650822` | `0.0255543659955664` | `False` | `False` | `None` | `False` | `not_at_resampling_boundary` | `0.4140625` |
| 5 | `0.9712250668187553` | `bisection_largest_minimum_admissible_increment` | `0.5000016429467714` | `0.03922774202134582` | `True` | `False` | `0.21875` | `False` | `projected_unique_root_ancestor_fraction_below_threshold` | `0.4140625` |

## Checks

| Check | Status |
| --- | --- |
| Claude review round 1 | `VERDICT: REVISE`; projected-diversity, terminal semantics, and exact regression-command issues patched. |
| Focused local review | `VERDICT: AGREE_FOR_IMPLEMENTATION_AND_CPU_HIDDEN_PILOT`. |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair.py` | Passed before runtime: `28 passed`. |
| `python -m py_compile docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_2026_07_09.py` | Passed before runtime. |
| `git diff --check` | Passed before runtime. |
| Final Phase 2AD runtime command | Exited `0`; artifact decision failed the beta-one nomination gate. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `52ee244498988e046a6356f926003b581103083b` |
| Git dirty status | Dirty; artifact records planned scalar validation edits and unrelated user work. |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seed | `(20260709, 6801)` |
| Output artifacts | JSON, Markdown, and quiet log under `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2ad_diversity_preserving_sequential_repair.log` |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Failed for nomination because terminal beta did not reach one. |
| Reference validity | Not established. |
| HMC-reference agreement | Not assessed. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Phase 2AD preserved unique ancestor diversity where Phase 2AC failed, but lost beta-one completion. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reference-method blocker or expansion decision. |

## Post-Run Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | The sequential bridge may need a more materially different rejuvenation/resampling/kernel strategy, not another local fallback rule. |
| What would overturn | A reviewed new reference-method design that reaches beta `1.0`, preserves diversity, passes terminal ESS/max-weight gates, and then replicates independently. |
| Weakest evidence | One CPU-hidden seed, 128 particles, and no uncertainty analysis; this is a diagnostic blocker for the current sequential branch, not evidence against the scalar target or HMC mechanics. |

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
