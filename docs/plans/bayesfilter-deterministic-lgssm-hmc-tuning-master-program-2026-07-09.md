# BayesFilter Deterministic LGSSM HMC Tuning Master Program

Date: 2026-07-09

Status: `DRAFT_LAUNCH_REVIEW`

## Objective

Build a deterministic BayesFilter-owned Python tuning program for serious
multidimensional identifiable LGSSM parameter-estimation HMC. Tuning decisions
must be made by explicit Python code using existing BayesFilter tuning tools,
not by an agent reading diagnostics and choosing the next run.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | Can BayesFilter run a reproducible serious HMC recovery test for an identifiable multidimensional stationary LGSSM using existing deterministic tuning tools? |
| Mechanism under test | Existing TensorFlow/TFP target, value/score path, quadratic geometry initializer, mass-matrix helpers, staged HMC kernel tuning, and deterministic burn-in/sampling controller. |
| Baseline/comparator | Prior-mean-generated LGSSM fixture with `T=120`; true parameters are the mean of the declared priors. |
| Promotion criterion | Final result artifact reports all parameters with `R_hat <= 1.01`, ESS floors satisfied, and posterior mean within `3 * posterior_sd` of truth. |
| Promotion veto | Nonfinite values, XLA/JIT fallback, forbidden GradientTape runtime, invalid geometry/mass artifact, missing hashes, failed target signature, R-hat/ESS failure at cap, or posterior recovery failure. |
| Continuation veto | Corrupt artifact, missing target implementation, non-deterministic output for fixed config/seeds, inability to run required local checks, or human-required runtime approval missing. |
| Repair trigger | Geometry rejection, mass regularization cap, kernel tuning repair status, burn-in/sampling cap pressure, low ESS, high R-hat, or posterior recovery misses. |
| Not concluded | No sampler superiority, default-readiness, production-readiness, scientific model adequacy, DSGE readiness, or public benchmark claim. |

## Minimal-Agent Policy

- Agents may write plans, code, tests, review bundles, and result summaries.
- Agents must not tune by reading partial results and choosing new budgets by
  hand.
- All tuning choices must be encoded in a deterministic Python driver and a
  versioned JSON config before runs start.
- The deterministic driver must own seed schedules, geometry configuration,
  mass conversion, kernel tuning, burn-in extension, retained-sample extension,
  pass/fail criteria, and artifact hashes.
- Claude is a read-only reviewer only; Claude cannot authorize runtime,
  hardware, product, funding, or scientific-claim boundaries.

## Existing BayesFilter Tools To Use

| Tool | Required use |
| --- | --- |
| `bayesfilter.inference.quadratic_geometry.fit_low_rank_spd_quadratic_geometry` | Quadratic initializer / local geometry candidate. |
| `bayesfilter.inference.mass_matrix.covariance_from_precision` | Convert accepted precision to mass covariance with explicit regularization report. |
| `bayesfilter.inference.hmc_kernel_tuning.HMCGeometryInitializationConfig` | Geometry-derived starting kernel and mass artifact conventions. |
| `bayesfilter.inference.hmc_kernel_tuning.HMCKernelTuningConfig.serious` | Serious staged kernel tuning config. |
| `bayesfilter.inference.hmc_kernel_tuning.tune_hmc_kernel` | One-call staged tuning / repair loop. |
| `bayesfilter.inference.hmc_budget_ladder.FixedMassHMCTuningBudgetLadderConfig` | Internal fixed-mass budget ladder where applicable. |
| `bayesfilter.inference.hmc_diagnostics` | Final diagnostic summaries and pass/fail checks. |

## Global Execution Rules

- Target-path runs must use `jit_compile=True` / `use_xla=True`.
- Any non-XLA HMC target-path execution is a hard veto, not a fallback.
- NeuTra training, if added by a later approved phase, is GPU-only.
- HMC sample generation is CPU-hidden and multicore unless a reviewed plan says
  otherwise.
- Runtime `GradientTape` is forbidden except for diagnostic/test-only paths.
- No serious run may start until its config, deterministic driver, expected
  artifacts, and approval boundary are recorded.

## Phase Index

| Phase | Name | Subplan | Required result artifact |
| --- | --- | --- | --- |
| 0 | Governance, Runbook, Review Gate | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase0-governance-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase0-governance-result-2026-07-09.md` |
| 1 | Tool Inventory And API Binding | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase1-tool-inventory-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase1-tool-inventory-result-2026-07-09.md` |
| 2 | Deterministic Config Schema | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase2-config-schema-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase2-config-schema-result-2026-07-09.md` |
| 3 | LGSSM Fixture Driver | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase3-lgssm-fixture-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase3-lgssm-fixture-result-2026-07-09.md` |
| 4 | XLA Value/Score Gate | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase4-xla-score-gate-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase4-xla-score-gate-result-2026-07-09.md` |
| 5 | Geometry And Mass Driver | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase5-geometry-mass-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase5-geometry-mass-result-2026-07-09.md` |
| 6 | Kernel Tuning Driver | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6-kernel-tuning-result-2026-07-09.md` |
| 7 | Burn-In And Sampling Controller | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase7-burnin-sampling-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase7-burnin-sampling-result-2026-07-09.md` |
| 8 | Serious Recovery Run | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase8-serious-recovery-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase8-serious-recovery-result-2026-07-09.md` |
| 9 | Closeout And Handoff | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase9-closeout-subplan-2026-07-09.md` | `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase9-closeout-result-2026-07-09.md` |

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Phase 2 fixes prior-mean truth and `T=120`; Phase 3 records generated fixture hash. |
| Proxy metric promotion | Geometry, mass, acceptance, compile success, and smoke checks cannot promote final recovery. |
| Missing stop conditions | Each subplan lists hard stop and repair triggers. |
| Agent tuning by hand | Phase 2 and Phase 7 require deterministic JSON config and coded extension rules. |
| Environment mismatch | Target-path HMC requires CPU-hidden sample generation with `use_xla=True`; GPU work requires trusted context. |
| Stale context | Phase 1 inventories actual exported APIs before implementation. |
| Artifact mismatch | Every driver phase writes JSON plus Markdown result and stable hashes. |

Initial audit verdict: `PASS_TO_PHASE0_ONLY`. The plan may create governance
artifacts and request read-only review. Serious runtime remains blocked until
the deterministic driver/config and explicit runtime approval exist.
