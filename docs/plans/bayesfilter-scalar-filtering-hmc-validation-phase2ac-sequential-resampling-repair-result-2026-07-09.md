# Phase 2AC Result: Sequential Resampling Repair

Date: 2026-07-09
Status: `FAILED_NO_REFERENCE_NOMINATION_SINGLE_DIVERSITY_REPAIR_TRIGGERED`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-subplan-2026-07-09.md`

Runtime JSON:
`docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.json`

Runtime Markdown:
`docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.md`

Quiet log:
`docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2ac_sequential_resampling_repair.log`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2AC did not nominate a sequential reference route for independent replication | Failed: beta reached `1.0`, terminal ESS and max-weight screens passed, but unique ancestor fraction was `0.21875` below the predeclared `0.25` threshold | Promotion veto: `unique_ancestor_fraction_below_threshold` | The resampling repair fixed the Phase 2AB beta stall descriptively, but diversity collapsed narrowly after three fallback-boundary resampling events | Draft Phase 2AD final focused diversity repair subplan; keep Phase 3 GPU/XLA blocked | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered negatively for the nomination gate, but the narrow beta-stall repair hypothesis was partially supported descriptively: the repaired schedule reached beta `1.0`. |
| Candidate under test | Same Phase 2AB target route, seed, particle count, thresholds, and RWM rejuvenation, except nonterminal resampling was forced after minimum-threshold fallback or near the ESS boundary. |
| Baseline/comparator | Final repaired Phase 2AB artifact, where beta stalled at `0.3419540270406287`. |
| Primary criterion | Failed because unique ancestor fraction was below threshold. |
| Veto diagnostics | One promotion veto fired: `unique_ancestor_fraction_below_threshold`. |
| Repair trigger | The subplan's single-screen repair trigger fired because beta reached `1.0` and only the diversity screen failed narrowly. |
| Explanatory diagnostics | Stage schedule, fallback resampling events, ESS trajectory, max weights, ancestor diversity, acceptance, and runtime were recorded. |
| Not concluded | No reference validity, posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Diagnostics

| Diagnostic | Value | Role |
| --- | --- | --- |
| Runtime exit | `0` | Artifact validity |
| Wall time | `80.28042639500927` seconds | Explanatory |
| Terminal beta | `1.0` | Promotion diagnostic passed |
| Stage count | `7` | Explanatory |
| Terminal pre-final-resampling ESS ratio | `0.9912539055044092` | Promotion diagnostic passed |
| Terminal pre-final-resampling max weight | `0.010002188339361427` | Promotion diagnostic passed |
| Minimum adaptive post-temperature ESS ratio | `0.5000010206048837` | Promotion diagnostic passed |
| Unique ancestor fraction | `0.21875` | Promotion veto |
| Aggregate rejuvenation acceptance | `0.6529017857142857` | Promotion diagnostic passed |
| Candidate nominated | `False` | Primary decision |

## Stage Schedule

| Stage | Beta | Rule | ESS ratio | Max weight | Resampled | Reason | Unique ancestor fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `0.2527170181274414` | `bisection_largest_target_admissible_increment` | `0.7000013562683505` | `0.04017182623114065` | `False` | `None` | `1.0` |
| 1 | `0.3419540270406287` | `bisection_largest_minimum_admissible_increment` | `0.5000012436397635` | `0.06553946724924992` | `True` | `minimum_threshold_fallback` | `0.71875` |
| 2 | `0.5811528746866764` | `bisection_largest_target_admissible_increment` | `0.7000006501357879` | `0.023202415520545847` | `False` | `None` | `0.71875` |
| 3 | `0.6735977385780526` | `bisection_largest_minimum_admissible_increment` | `0.5000010206048837` | `0.041457723731129006` | `True` | `minimum_threshold_fallback` | `0.4140625` |
| 4 | `0.8828990299946569` | `bisection_largest_target_admissible_increment` | `0.700000256650822` | `0.0255543659955664` | `False` | `None` | `0.4140625` |
| 5 | `0.9712250668187553` | `bisection_largest_minimum_admissible_increment` | `0.5000016429467714` | `0.03922774202134582` | `True` | `minimum_threshold_fallback` | `0.21875` |
| 6 | `1.0` | `terminal_beta_admissible` | `0.9912539055044092` | `0.010002188339361427` | `False` | `None` | `0.21875` |

## Checks

| Check | Status |
| --- | --- |
| Claude review round 1 | `VERDICT: REVISE`; missing runtime command/timeout patched. |
| Focused local review | `VERDICT: AGREE_FOR_IMPLEMENTATION_AND_CPU_HIDDEN_PILOT`. |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot.py` | Passed before runtime: `31 passed`. |
| `python -m py_compile docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_2026_07_09.py` | Passed before runtime. |
| `git diff --check` | Passed before runtime. |
| Final Phase 2AC runtime command | Exited `0`; artifact decision failed the diversity nomination gate. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `52ee244498988e046a6356f926003b581103083b` |
| Git dirty status | Dirty; artifact records planned scalar validation edits and unrelated user work. |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seed | `(20260709, 6801)` |
| Output artifacts | JSON, Markdown, and quiet log under `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2ac_sequential_resampling_repair.log` |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Passed for runtime finiteness and artifact validity. |
| Reference validity | Not established. |
| HMC-reference agreement | Not assessed. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Phase 2AC reached beta `1.0` where Phase 2AB stalled; this is descriptive repair evidence only. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed Phase 2AD final focused diversity repair subplan, or a reference-method blocker if no narrow repair is accepted. |

## Post-Run Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | Forced resampling may have repaired temperature progression by spending ancestor diversity too aggressively; the bridge can reach beta one but may not yet preserve enough independent support for a reference route. |
| What would overturn | A reviewed diversity repair that reaches beta `1.0` with unique ancestor fraction at least `0.25`, terminal ESS and max-weight gates passing, and finite target evaluations, followed by independent replication. |
| Weakest evidence | One CPU-hidden seed, 128 particles, and no uncertainty analysis; the beta-one improvement is not reference validity. |

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
