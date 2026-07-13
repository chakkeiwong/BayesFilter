# Phase 2Z Result: Proposal Strategy Pilot

Date: 2026-07-09
Status: `PASSED_ARTIFACT_VALIDITY_NO_CANDIDATE_NOMINATED`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Phase 2Z produced a valid proposal-pilot artifact but nominated no candidate for independent replication | Artifact validity passed; `candidate_nominated=False`; all four candidates failed ESS, ESS-ratio, and max-weight pilot screens | Final vetoes: `[]`; the first `4096` attempt timed out with no artifact and was repaired to `1024` per candidate under the subplan timeout escape hatch | The result is a finite pilot with no uncertainty analysis and smaller timeout-repaired sample size; it argues against another blind SNIS tweak but does not prove SNIS impossible | Draft and review a decision subplan to abandon the current SNIS reference branch or move to a transport/sequential reference method | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered negatively for the four fixed pilot candidates: none was worth independent SNIS replication under the pilot screen. |
| Baseline/comparator | Failed Phase 2W standard normal, failed Phase 2X shifted diagonal mixture, and Phase 2Y target-geometry localization. |
| Primary criterion | Passed for artifact validity: all candidates had finite target/proposal/log-weight evaluations and no hard vetoes. |
| Pilot nomination screen | Failed for every candidate: ESS, ESS ratio, and max normalized weight screens all failed. |
| Veto diagnostics | No invalid source artifact, nonfinite target/proposal/log weights, proposal replay mismatch, HMC-moment tuning, or unsupported claim. |
| Explanatory diagnostics | ESS, ESS ratio, max weight, nomination failures, top-weight rows, weighted moments, and runtime were recorded. |
| Not concluded | No posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or source faithfulness. |

## Runtime Artifacts

- JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json`
- Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2z_proposal_strategy_pilot.log`
- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_2026_07_09.py`
- Tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot.py`

## Candidate Outcomes

| Candidate | ESS | ESS ratio | Max weight | Nomination status |
| --- | --- | --- | --- | --- |
| `student_t_centered` | `21.01706760314462` | `0.02052448008119592` | `0.10525616575524867` | Failed ESS, ESS-ratio, and max-weight screens |
| `student_t_shifted` | `14.656246107251146` | `0.014312740339112447` | `0.2428262178992962` | Failed ESS, ESS-ratio, and max-weight screens |
| `anchor_mixture_student_t` | `26.071556547207543` | `0.025460504440632366` | `0.1112365984074613` | Failed ESS, ESS-ratio, and max-weight screens |
| `ridge_line_student_t` | `1.535816743965212` | `0.0014998210390285273` | `0.7997380494257067` | Failed ESS, ESS-ratio, and max-weight screens |

Pilot nomination thresholds were ESS `>= 256`, ESS ratio `>= 0.05`, and max
normalized weight `<= 0.05`.  No candidate met them.

## Runtime Repair Record

The first Phase 2Z runtime used the reviewed `4096` samples per candidate and
timed out with exit code `124` before writing JSON/Markdown artifacts.  That
timeout is a runtime-plan flaw, not proposal evidence.  The subplan allowed a
timeout-feasibility repair, so the pilot size was visibly repaired to `1024`
samples per candidate.  Candidate families, seeds, proposal formulas,
nomination thresholds, and claim boundaries were unchanged.

## Review And Boundary Record

Claude material review was not attempted again after the previous approval
layer rejection for external transfer of private repository context.  The
local Codex substitute review was updated after finding and fixing an
under-specified mixture-weight/allocation issue in the Phase 2Z draft.  The
substitute review is weaker than full Claude material review.

Phase 2Z does not clear Phase 3 GPU/XLA, HMC-readiness, default-policy, or
scientific-claim boundaries.

## Checks

| Check | Status |
| --- | --- |
| Phase 2Z subplan skeptical audit | Passed after patching fixed mixture weights, scales, seeds, and deterministic allocations |
| Claude review | Not attempted this turn; prior repo-context review gate blocked by approval layer |
| Codex substitute review | `VERDICT: AGREE` after patch; weaker than Claude |
| `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization.py` | Passed before repaired runtime: `11 passed` |
| `git diff --check` | Passed before repaired runtime |
| Initial Phase 2Z runtime with `4096` samples per candidate | Timed out with exit code `124`; no artifact; not proposal evidence |
| Repaired Phase 2Z runtime with `1024` samples per candidate | Exited `0`; artifact decision passed validity gate and nominated no candidate |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `52ee244498988e046a6356f926003b581103083b` |
| Git dirty status | Dirty; artifact records planned scalar HMC validation edits and unrelated user work |
| Command | `CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.md` |
| Environment | conda env `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU status | `CUDA_VISIBLE_DEVICES=-1`; CPU-hidden debug/reference exception; TensorFlow listed CPU only |
| JIT/TF32 | `jit_compile=False`; TF32 disabled by CPU-hidden debug contract |
| Seeds | `(20260709, 6701)`, `(20260709, 6702)`, `(20260709, 6703)`, `(20260709, 6704)` |
| Pilot sample count | `1024` per candidate after timeout repair |
| Wall time | `188.37495324999327` seconds |
| Plan/result paths | Master, Phase 2Z subplan, JSON, Markdown, quiet log, and this result file |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Passed for pilot artifact validity. |
| Candidate nomination | Failed for all candidates. |
| Reference validity | Not assessed; Phase 2Z is pilot nomination only. |
| HMC-reference agreement | Not assessed. |
| Statistically supported ranking | None. |
| Descriptive-only differences | ESS, ESS ratio, max weights, weighted moments, top weights, and runtime. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed decision to abandon the current SNIS branch or move to transport/sequential reference. |

## Post-Run Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | The smaller timeout-repaired pilot may miss a proposal that a larger or better designed adaptive/transport proposal would find; this does not prove SNIS impossible. |
| What would overturn | A reviewed transport/sequential reference or independently replicated proposal with fresh seeds passing ESS/weight screens. |
| Weakest evidence | Single pilot per candidate, no uncertainty analysis, and anchor/ridge candidates partially informed by failed-proposal diagnostics. |

## Final Nonclaims

- No valid independent reference.
- No HMC-vs-reference agreement.
- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.
