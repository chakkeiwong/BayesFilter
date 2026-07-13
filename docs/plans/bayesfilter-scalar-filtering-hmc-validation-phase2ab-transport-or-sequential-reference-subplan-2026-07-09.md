# Phase 2AB Subplan: Transport Or Sequential Reference Pilot

Date: 2026-07-09
Status: `COMPLETED_FAILED_NO_REFERENCE_NOMINATION`

## Phase Objective

Build and run a small CPU-hidden sequential tempering reference pilot in the
Phase 2S/2U MAP-local `u_new` coordinate after Phase 2AA abandoned blind
independent SNIS tweaks for this scalar target for now.

The primary candidate is a sequential Monte Carlo / annealed-importance pilot
that bridges from the Phase 2S local Gaussian base density in `u_new` to the
scalar filtering target using adaptive inverse-temperature steps, resampling,
and predeclared random-walk rejuvenation moves.  This is a reference-viability
pilot, not posterior certification.

This phase may nominate a sequential reference route for independent
replication.  It does not, by itself, establish a valid reference,
HMC-vs-reference agreement, posterior correctness, HMC readiness,
convergence, zero divergences, GPU/XLA readiness, default readiness, or
Zhao-Cui source faithfulness.

## Entry Conditions

- Phase 2V produced a finite CPU-hidden selected-kernel HMC mechanics screen,
  but native divergence telemetry remained unavailable and no zero-divergence
  claim was made.
- Phase 2W fixed standard-normal independent SNIS failed reference-validity
  ESS and ESS-ratio gates.
- Phase 2X shifted-mixture independent SNIS failed reference-validity ESS and
  ESS-ratio gates.
- Phase 2Y found no affine-orientation or proposal-log-density replay bug and
  made proposal-family/global-geometry mismatch plausible, not proven.
- Phase 2Z piloted four heavier-tail or geometry-aware independent proposal
  families with finite evaluations and no hard vetoes, but nominated no
  candidate.
- Phase 2AA chose to stop blind independent SNIS tweaks for this target for
  now and to draft this sequential/transport reference plan.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_2026_07_09.py`
- Tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference.py`
- Runtime JSON:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_cpu_hidden_2026-07-09.json`
- Runtime Markdown:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2ab_transport_or_sequential_reference.log`
- Phase 2AB result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-result-2026-07-09.md`
- Review record under `docs/reviews/`.

## Required Checks, Tests, And Reviews

Before runtime:

- Review this subplan.  Claude may be used as a read-only reviewer if the
  approval layer permits bounded review.  Otherwise record a local Codex
  substitute review as weaker than Claude.
- Implement only the reviewed harness and tests.
- Run:
  `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot.py`
- Run:
  `python -m py_compile docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_2026_07_09.py`
- Run `git diff --check`.

Runtime command, only after review and focused checks pass:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_cpu_hidden_2026-07-09.md
```

After runtime:

- Write the Phase 2AB result with decision table, inference-status table, run
  manifest, post-run red-team note, and nonclaims.
- Draft or refresh the next subplan:
  - Phase 2AC independent sequential-reference replication / HMC agreement
    subplan if Phase 2AB nominates a viable route.
  - A blocker or reference-method redesign subplan if Phase 2AB fails validity
    or diversity gates.
- Review the next subplan before any further runtime.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific or engineering question | Can a sequential tempering reference pilot avoid the independent-SNIS weight collapse observed in Phase 2W/2X/2Z strongly enough to nominate a reference route for independent replication? |
| Candidate under test | A CPU-hidden sequential Monte Carlo / annealed-importance pilot in Phase 2S/2U `u_new`, bridging from base `q0(u)=N(0,I_4)` to the scalar filtering target with adaptive temperature increments, resampling, and predeclared random-walk rejuvenation.  Exact target route: `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py::build_phase2u_adapter(phase2s_payload).log_prob_and_grad`, with gradients ignored and source artifacts recorded. |
| Baseline/comparator | Failure baseline is Phase 2W fixed SNIS, Phase 2X shifted-mixture SNIS, Phase 2Z proposal pilots, and Phase 2Y geometry localization.  Phase 2AB does not rank methods against these baselines; it asks whether the sequential route clears its own viability gates. |
| Primary promotion/pass criterion | Runtime artifact is valid and the sequential pilot reaches beta `1.0` within `48` temperature stages with finite target/base/log-weight values, no missing required diagnostics, terminal `beta=1` pre-final-resampling ESS ratio `>= 0.50`, minimum adaptive post-temperature ESS ratio `>= 0.50`, terminal `beta=1` pre-final-resampling max normalized weight `<= 0.08`, final unique ancestor fraction `>= 0.25` measured after the last completed resampling/rejuvenation stage, and aggregate rejuvenation acceptance in `[0.10, 0.90]` over executed rejuvenation proposals. Passing nominates Phase 2AC replication only; it does not establish a valid reference by itself. |
| Promotion veto diagnostics | Missing or invalid source artifacts, stale target route, nonfinite target/base/log weights, temperature schedule not reaching beta `1.0`, terminal `beta=1` pre-final-resampling ESS ratio below `0.50`, any adaptive post-temperature ESS ratio below `0.50`, terminal `beta=1` pre-final-resampling max normalized weight above `0.08`, final unique ancestor fraction below `0.25`, aggregate rejuvenation acceptance outside `[0.10, 0.90]`, artifact missing required provenance, or unsupported HMC/posterior/GPU/default/source-faithfulness claim. |
| Continuation veto | Target adapter cannot be built from Phase 2S/2U artifacts, repeated nonfinite target values under bounded proposals, runtime timeout before artifact, or a result showing the sequential route lacks enough diversity to justify replication and no narrower repair question is stated. |
| Repair trigger | Schedule reaches beta `1.0` but misses only one diversity or acceptance screen, or diagnostics identify a narrow proposal-scale/resampling defect that can be repaired without changing the scientific question. |
| Explanatory diagnostics | Temperature schedule, incremental log-weight summaries, ESS trajectory, resampling count, ancestor diversity, rejuvenation acceptance by stage, terminal particle moments, optional descriptive comparison to Phase 2V HMC moments, wall time, and finite-count summaries. |
| What will not be concluded | No posterior correctness, no HMC readiness, no HMC convergence, no HMC-vs-reference agreement, no zero-divergence claim, no sampler superiority, no statistically supported ranking, no GPU/XLA production/default readiness, and no Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2AB JSON/Markdown/log artifacts, result file, review record, and refreshed ledger/handoff. |

## Fixed Runtime Design

| Item | Setting |
| --- | --- |
| Coordinate | Phase 2S/2U MAP-local `u_new`, dimension `4`. |
| Base density | `N(0, I_4)` in `u_new`. |
| Target log density | `build_phase2u_adapter(phase2s_payload).log_prob_and_grad` from `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py`; gradients ignored; source JSON artifacts and launch commit `52ee244498988e046a6356f926003b581103083b` recorded. |
| Particle count | `128` particles for the pilot. |
| Seed | `(20260709, 6801)`. |
| Temperature rule | Adaptive beta increments selected by bisection to keep post-temperature ESS ratio at least `0.70` when possible and never below `0.50`, capped at beta `1.0`; bisection uses at most `32` iterations, beta tolerance `1.0e-6`, minimum positive beta increment `1.0e-4`, and a fail-closed `temperature_increment_stalled` veto if no admissible increment exists. |
| Maximum stages | `48`. |
| Resampling | Systematic resampling when normalized ESS ratio is at or below `0.50` within beta tolerance; preserve ancestor IDs for diversity diagnostics. |
| Rejuvenation | One random-walk Metropolis move per stage with fixed Gaussian proposal scale `0.45` in `u_new`; the target is the current tempered density `(1-beta) log q0 + beta log p`. |
| HMC usage | None. |
| GPU/XLA usage | None; CPU-hidden debug/reference exception. |
| HMC moment comparison | Optional descriptive diagnostic only if reference viability passes; not a Phase 2AB promotion criterion. |

## Forbidden Claims And Actions

- Do not run HMC in Phase 2AB.
- Do not run GPU/XLA or proceed to Phase 3.
- Do not change defaults, package behavior, public API behavior, or model
  files.
- Do not claim the sequential pilot is an exact reference or posterior
  certificate.
- Do not claim HMC readiness, convergence, posterior correctness,
  zero divergences, sampler superiority, statistical ranking, GPU/XLA
  readiness, default readiness, or Zhao-Cui source faithfulness.
- Do not compare Phase 2AB to Phase 2W/2X/2Z as a superiority ranking; those
  artifacts are failure baselines and design motivation only.
- Do not use Phase 2V HMC moments to tune the sequential proposal.  Phase 2V
  moments may be recorded only as a descriptive downstream comparison after
  reference viability passes.

## Exact Next-Phase Handoff Conditions

If Phase 2AB passes all primary validity and diversity gates, draft Phase 2AC
as an independent sequential-reference replication and limited HMC-agreement
subplan with fresh seeds, a separate review, and Phase 3 GPU/XLA still blocked.

If Phase 2AB fails due to nonfinite target evaluations, invalid adapter state,
or missing provenance, write a blocker result and stop before more runtime.

If Phase 2AB reaches beta `1.0` but narrowly fails one diversity or
rejuvenation screen, draft a focused repair subplan only if the artifact
identifies a single discriminating repair question.

If Phase 2AB fails broadly, write a reference-method blocker or expansion
decision subplan.  Do not return to blind independent SNIS tweaks without a new
reviewed hypothesis.

## Stop Conditions

Stop before runtime for review nonconvergence, missing required source
artifacts, under-specified temperature/resampling/rejuvenation settings,
unsupported claims, or any need to cross HMC-runtime, GPU/XLA, default-policy,
model-file, product, funding, or Zhao-Cui source-faithfulness boundaries.

Stop after runtime for any promotion veto or continuation veto listed in the
evidence contract, missing JSON/Markdown/log artifacts, changed criteria after
seeing results, or inability to classify the result as nomination, focused
repair, broad failure, or blocker.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the failed independent SNIS branch and Phase 2Y diagnostic, not HMC success or GPU readiness. |
| Proxy metrics promoted | Sequential ESS/diversity/acceptance gates can nominate a route for replication only; they cannot certify posterior correctness or HMC readiness. |
| Missing stop conditions | Review, runtime, artifact, diversity, target-route, and boundary stops are explicit. |
| Unfair comparison | Phase 2AB is not ranked against Phase 2W/2X/2Z because the estimators and ESS semantics differ. |
| Hidden assumptions | Phase 2Y geometry mismatch remains plausible, not proven; the sequential route is a discriminating repair candidate. |
| Stale context | Source artifacts from Phase 2S/2U/2V/2W/2X/2Y/2Z must be loaded and provenance recorded. |
| Environment mismatch | CPU-hidden Phase 2AB cannot support GPU/XLA or default-readiness claims. |
| Runtime feasibility | The pilot uses `128` particles, at most `48` stages, and one rejuvenation proposal per stage to keep the expected number of target evaluations in the same rough order as Phase 2Z rather than a much larger infeasible run. |
| Artifact mismatch | JSON, Markdown, log, result, review, ledger, and handoff artifacts are predeclared. |

Audit status: `PASSED_AFTER_CLAUDE_REVIEW_REPAIR_AND_FOCUSED_LOCAL_REVIEW`.
