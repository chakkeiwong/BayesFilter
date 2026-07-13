# Phase 2AD Subplan: Diversity-Preserving Sequential Repair

Date: 2026-07-09
Status: `REVIEWED_READY_FOR_IMPLEMENTATION_AND_CPU_HIDDEN_PILOT`

## Phase Objective

Run a final focused CPU-hidden repair of the Phase 2AC sequential tempering
route by testing whether a less aggressive, diversity-preserving resampling
policy can keep the beta-one repair while satisfying the predeclared ancestor
diversity screen.

This is still a narrow reference-method repair.  It does not run HMC and cannot
establish a valid reference, HMC-vs-reference agreement, posterior correctness,
HMC readiness, convergence, zero divergences, GPU/XLA readiness, default
readiness, or Zhao-Cui source faithfulness.

## Entry Conditions

- Phase 2AA abandoned blind independent SNIS tweaks for this scalar target for
  now and selected a sequential/transport reference branch.
- Phase 2AB sequential tempering did not reach beta `1.0`; it stalled at beta
  `0.3419540270406287`.
- Phase 2AC forced resampling at the minimum/fallback ESS boundary and reached
  beta `1.0`, with terminal ESS ratio `0.9912539055044092`, terminal max
  weight `0.010002188339361427`, minimum post-temperature ESS ratio
  `0.5000010206048837`, and aggregate rejuvenation acceptance
  `0.6529017857142857`.
- Phase 2AC failed only the unique ancestor fraction screen:
  `0.21875 < 0.25`.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2AD subplan review under `docs/reviews/`.
- If executed, harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_2026_07_09.py`
- If executed, tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair.py`
- If executed, runtime JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.json`
- If executed, runtime Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.md`
- If executed, quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2ad_diversity_preserving_sequential_repair.log`
- If executed, result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-result-2026-07-09.md`

## Required Checks, Tests, And Reviews

- Review this subplan before implementation or runtime.  Claude may be used as
  a read-only reviewer if safely available; otherwise record a local Codex
  substitute review as weaker than Claude.
- Before runtime, run focused tests for the Phase 2AD diversity policy and the
  Phase 2AC resampling-policy regression:
  `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair.py`
- Before runtime, run `python -m py_compile` on the Phase 2AD harness.
- Before runtime and before closeout, run `git diff --check`.

Runtime command, only after review and focused checks pass:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_cpu_hidden_2026-07-09.md
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can a less aggressive diversity-preserving resampling policy keep the Phase 2AC beta-one repair while restoring unique ancestor fraction to at least `0.25`? |
| Candidate under test | Same Phase 2AC target route, seed, particle count, beta thresholds, and RWM rejuvenation, except a fallback-boundary stage draws the exact systematic-resampling index vector it would use if accepted, computes the realized projected unique root-ancestor fraction `unique(ancestor_ids[indices]) / particle_count`, and performs that resampling only when the projected fraction is at least `0.25`; otherwise it skips resampling, keeps the cumulative log weights, and continues.  The RNG draw is consumed either way so the policy is deterministic for the fixed seed. |
| Baseline/comparator | Final Phase 2AC artifact.  This is not a method ranking. |
| Primary criterion | Artifact valid; beta reaches `1.0` within `48` stages; finite target/base/log weights; terminal beta-1 pre-final-resampling ESS ratio `>= 0.50`; terminal max weight `<= 0.08`; unique ancestor fraction `>= 0.25`; aggregate rejuvenation acceptance in `[0.10, 0.90]`. Passing nominates an independent replication subplan only. |
| Promotion vetoes | Nonfinite target/base/log weights, beta not reaching `1.0`, terminal ESS or max-weight failure, unique-ancestor failure, acceptance failure, missing provenance, changed thresholds after seeing results, or unsupported HMC/posterior/GPU/default/source-faithfulness claim. |
| Continuation veto | The repair cannot build the same Phase 2U adapter, repeats a target-evaluation exception, times out before artifact, or fails more than one promotion screen after Phase 2AC already isolated the diversity issue. |
| Repair trigger | None inside this phase.  If Phase 2AD fails, write a reference-method blocker or expansion decision rather than another unreviewed local policy tweak. |
| Explanatory diagnostics | Stage schedule, fallback stages, realized projected unique root-ancestor fraction for any considered resampling draw, whether the draw was accepted or skipped, ESS trajectory, max weights, ancestor diversity, rejuvenation acceptance, terminal moments, and runtime. |
| What will not be concluded | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2AD JSON/Markdown/log/result artifacts and refreshed ledger/handoff. |

## Forbidden Claims And Actions

- Do not run HMC.
- Do not run GPU/XLA or proceed to Phase 3.
- Do not change defaults, public API behavior, package behavior, or model
  files.
- Do not claim a valid reference, posterior correctness, HMC readiness,
  convergence, zero divergences, sampler superiority, statistical ranking,
  GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness.
- Do not tune the repair using Phase 2V HMC moments.
- Do not run additional schedule variants after Phase 2AD without a new
  reviewed subplan and explicit blocker/expansion rationale.

## Exact Next-Phase Handoff Conditions

If Phase 2AD passes, draft a Phase 2AE independent sequential-reference
replication and limited HMC-agreement subplan with fresh seeds and Phase 3
still blocked.

If Phase 2AD fails any promotion screen, write a reference-method blocker or
expansion decision subplan.  Do not proceed to Phase 3 GPU/XLA.

If Phase 2AD hits a source-artifact, adapter, nonfinite-target, or provenance
continuation veto, write a blocker result and stop.

## Measurement Semantics

- Root ancestor IDs are the original particle indices before any resampling.
- A projected resampling diversity fraction is
  `unique(ancestor_ids[resample_indices]) / particle_count` for the realized
  systematic-resampling draw generated at that stage.
- The resampling RNG draw is consumed before deciding whether to accept or skip
  the resample.  If skipped, particles, root ancestor IDs, target log
  probabilities, base log probabilities, and cumulative log weights are left
  unchanged before rejuvenation.
- Terminal beta `1.0` stages are not resampled in Phase 2AD.  Terminal ESS and
  max-weight diagnostics remain beta-one pre-final-resampling measurements.
- Unique ancestor fraction is measured after the last completed nonterminal
  resampling and rejuvenation stage, with no terminal resampling correction.

## Stop Conditions

Stop before runtime for review nonconvergence, under-specified repair
mechanics, missing artifacts, unsupported claims, or any need to cross HMC,
GPU/XLA, model-file, default-policy, product, funding, or Zhao-Cui
source-faithfulness boundaries.

Stop after runtime for any promotion veto and write a blocker/expansion
decision unless a new human-approved plan explicitly reopens this reference
branch.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the final Phase 2AC artifact, not HMC success. |
| Proxy metrics promoted | ESS, max weights, and ancestor diversity can nominate replication only, not certify a reference or posterior. |
| Missing stop conditions | Runtime, artifact, review, claim, and boundary stops are explicit. |
| Unfair comparison | No superiority ranking is made; Phase 2AD tests one diversity-preserving repair hypothesis. |
| Hidden assumptions | The repair assumes the Phase 2AC failure is caused by over-aggressive fallback resampling; this remains a hypothesis. |
| Stale context | The target route and source artifacts must match Phase 2AB/2AC/Phase 2U provenance. |
| Environment mismatch | CPU-hidden Phase 2AD cannot support GPU/XLA/default readiness. |
| Artifact mismatch | Required JSON/Markdown/log/result artifacts are predeclared. |
| Runtime feasibility | The runtime command uses `timeout 600`; timeout before artifact is a continuation veto, not scientific evidence. |

Audit status: `PASSED_AFTER_CLAUDE_REVIEW_ROUND_1_AND_FOCUSED_LOCAL_REVIEW`.
