# Phase 2AC Subplan: Sequential Resampling Repair

Date: 2026-07-09
Status: `REVIEWED_READY_FOR_IMPLEMENTATION_AND_CPU_HIDDEN_PILOT`

## Phase Objective

Run a focused CPU-hidden repair of the Phase 2AB sequential tempering pilot by
testing whether resampling at the minimum/fallback ESS boundary prevents the
observed temperature stall and allows the bridge to reach beta `1.0`.

This is a narrow reference-method repair.  It does not run HMC and cannot
establish a valid reference, HMC-vs-reference agreement, posterior
correctness, HMC readiness, convergence, zero divergences, GPU/XLA readiness,
default readiness, or Zhao-Cui source faithfulness.

## Entry Conditions

- Phase 2AA abandoned blind independent SNIS tweaks for this scalar target for
  now and selected a sequential/transport reference branch.
- Phase 2AB was reviewed, implemented, and repaired for cumulative ESS
  scheduling and decision-veto reporting.
- Final Phase 2AB artifact was valid but did not nominate a reference route:
  beta stalled at `0.3419540270406287`, with ESS ratio
  `0.5000012436397635`, max weight `0.06553946724924992`, unique ancestor
  fraction `1.0`, and rejuvenation acceptance `0.625`.
- The narrow repair hypothesis is that the Phase 2AB resampling trigger was
  too conservative at the minimum/fallback ESS boundary.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2AC subplan review under `docs/reviews/`.
- If executed, harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_2026_07_09.py`
- If executed, tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair.py`
- If executed, runtime JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.json`
- If executed, runtime Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.md`
- If executed, quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2ac_sequential_resampling_repair.log`
- If executed, result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-result-2026-07-09.md`

## Required Checks, Tests, And Reviews

- Review this subplan before implementation or runtime.  Claude may be used as
  a read-only reviewer if safely available; otherwise record a local Codex
  substitute review as weaker than Claude.
- Before runtime, run focused tests for the repaired resampling policy and the
  Phase 2AB cumulative-ESS regression.
- Before runtime, run `python -m py_compile` on the Phase 2AC harness.
- Before runtime and before closeout, run `git diff --check`.

Runtime command, only after review and focused checks pass:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.md
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does resampling at the minimum/fallback ESS boundary repair the Phase 2AB beta stall enough to nominate a sequential reference route for independent replication? |
| Candidate under test | Same Phase 2AB sequential tempering target route, seed, particle count, beta thresholds, and RWM rejuvenation, except resampling is forced after any minimum-threshold fallback beta step or whenever post-temperature ESS ratio is within `1.0e-4` of the resampling threshold. |
| Baseline/comparator | Final repaired Phase 2AB artifact.  This is not a method ranking. |
| Primary criterion | Artifact valid; beta reaches `1.0` within `48` stages; finite target/base/log weights; terminal beta-1 pre-final-resampling ESS ratio `>= 0.50`; terminal max weight `<= 0.08`; unique ancestor fraction `>= 0.25`; aggregate rejuvenation acceptance in `[0.10, 0.90]`. Passing nominates an independent replication subplan only. |
| Promotion vetoes | Nonfinite target/base/log weights, beta not reaching `1.0`, terminal ESS or max-weight failure, unique-ancestor failure, acceptance failure, missing provenance, changed thresholds after seeing results, or unsupported HMC/posterior/GPU/default/source-faithfulness claim. |
| Continuation veto | The repair cannot build the same Phase 2U adapter, repeats a target-evaluation exception, times out before artifact, or fails broadly enough that no single next repair question remains. |
| Repair trigger | If beta reaches `1.0` but only one diversity or acceptance screen narrowly fails, draft a final focused repair; otherwise write a blocker/expansion decision. |
| Explanatory diagnostics | Stage schedule, whether a stage used minimum-threshold fallback, resampling counts, ESS trajectory, max weights, ancestor diversity, rejuvenation acceptance, terminal moments, and runtime. |
| What will not be concluded | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2AC JSON/Markdown/log/result artifacts and refreshed ledger/handoff. |

## Forbidden Claims And Actions

- Do not run HMC.
- Do not run GPU/XLA or proceed to Phase 3.
- Do not change defaults, public API behavior, package behavior, or model
  files.
- Do not claim a valid reference, posterior correctness, HMC readiness,
  convergence, zero divergences, sampler superiority, statistical ranking,
  GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness.
- Do not tune the repair using Phase 2V HMC moments.
- Do not run additional schedule variants after Phase 2AC without a new
  reviewed subplan.

## Exact Next-Phase Handoff Conditions

If Phase 2AC passes, draft a Phase 2AD independent sequential-reference
replication and limited HMC-agreement subplan with fresh seeds and Phase 3
still blocked.

If Phase 2AC fails to reach beta `1.0` or fails multiple validity/diversity
screens, draft a reference-method blocker or expansion decision subplan.

If Phase 2AC hits a source-artifact, adapter, nonfinite-target, or provenance
continuation veto, write a blocker result and stop.

## Stop Conditions

Stop before runtime for review nonconvergence, under-specified repair
mechanics, missing artifacts, unsupported claims, or any need to cross HMC,
GPU/XLA, model-file, default-policy, product, funding, or Zhao-Cui
source-faithfulness boundaries.

Stop after runtime for any promotion veto unless the artifact identifies a
single reviewed repair question; do not return to blind independent SNIS
tweaks without a new reviewed hypothesis.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the final repaired Phase 2AB artifact, not HMC success. |
| Proxy metrics promoted | ESS, max weights, and ancestor diversity can nominate replication only, not certify a reference or posterior. |
| Missing stop conditions | Runtime, artifact, review, claim, and boundary stops are explicit. |
| Unfair comparison | No superiority ranking is made; Phase 2AC tests one repair hypothesis. |
| Hidden assumptions | The repair assumes the Phase 2AB stall may be caused by resampling timing; this remains a hypothesis. |
| Stale context | The target route and source artifacts must match Phase 2AB/Phase 2U provenance. |
| Environment mismatch | CPU-hidden Phase 2AC cannot support GPU/XLA/default readiness. |
| Artifact mismatch | Required JSON/Markdown/log/result artifacts are predeclared. |
| Runtime feasibility | The runtime command uses `timeout 600`; timeout before artifact is a continuation veto, not scientific evidence. |

Audit status: `PASSED_AFTER_CLAUDE_REVIEW_ROUND_1_AND_FOCUSED_LOCAL_REVIEW`.
