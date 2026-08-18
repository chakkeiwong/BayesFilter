# BayesFilter Tuning Streamline Refactor Execution Result

Date: 2026-08-18
Plan: `docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`
Status: `BAYESFILTER_PHASES_0_4_GREEN_CONSUMER_MIGRATION_BLOCKED`

## Scope And Evidence Contract

This run executes the reviewed tuning-streamline plan through the BayesFilter
structural, mass-matrix, posterior-oracle, and route-ledger gates, then runs the
current focused suites in MacroFinance and dsge_hmc. The question is whether
BayesFilter can expose exactly two active tuning interfaces without silently
changing target, mass, transport, or evidence semantics.

The promotion criterion is green BayesFilter contract/oracle evidence plus green
consumer migration suites. Hard vetoes are failed target/artifact/route
contracts, invalid mass or posterior evidence, missed route classification, or
consumer failures. Acceptance, ESS/R-hat, runtime, and candidate counts are
diagnostics unless used by the declared hard screen. This run does not claim
posterior correctness for MacroFinance or dsge_hmc, sampler superiority,
production readiness, or GPU readiness.

## Completed BayesFilter Gates

| Gate | Result | Status |
| --- | --- | --- |
| Compileall in `tfgpu`, CPU-hidden | BayesFilter and tests compile | PASS |
| Canonical contracts, mass, robust grid, fixed transport, public API, posterior oracle | 106 passed | PASS |
| Mass/geometry/windowed regression subset | 79 passed, 1 skipped, 203 deselected | PASS |
| Fixed-grid, route selection, handoff, robust broad-grid suite | 159 passed | PASS |
| AST route ledger | Exactly two active interfaces; no stale/unclassified entries | PASS |
| `git diff --check` | No whitespace errors | PASS |

The 106-test gate includes exact Gaussian value/score checks, affine
fixed-transport holdout, negative controls, four mass arms, fresh mass-specific
epsilon/L tuning, and analytic moment checks. Tests ran in `tfgpu` with
`CUDA_VISIBLE_DEVICES=-1`; this is CPU engineering evidence, not GPU evidence.

The same 106-test gate was rerun on 2026-08-18 in a trusted GPU-visible context with
`TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import. TensorFlow 2.20.0
reported both RTX 4080 SUPER devices and verified memory growth on both; the
gate again passed 106 tests in 209 seconds. This establishes GPU visibility and
allocator compliance for the run; it is not a per-test utilization benchmark
and does not show that the tuner is statistically superior.

## Consumer Gate Results

### MacroFinance

The current focused command collected 64 tests: 60 passed and 4 failed. The
failures are contract drift, not BayesFilter posterior failures:

1. Two tests expect historical `ccma_phase4y_stage_budget_v1`; the current
   repository-owned policy is `bayesfilter_hmc_emergency_stage_caps_v2`.
2. One public-artifact test rejects the string `0.25`, although that token is
   also legitimate budget metadata. The correct invariant is the explicit
   `public_summary_contains_mass_matrices == False` and
   `raw_matrix_values_publicized == False` fields.
3. One historical fixed-kernel test treats `chain_execution_mode` as a tuning
   control even though the current public runner reconstructs the same fixed
   kernel with a different execution detail.

The audited repairs were not applied: the external consumer patch operation was
rejected repeatedly by the patch-review service with a 502 response before
mutation, including single-file retries after the user requested continuation.
No MacroFinance files were changed by this run.

### dsge_hmc

The current focused command collected 47 passing tests and 3 failures. The
failures are:

1. Two relaunch tests expect the old 49-candidate fixed-transport grid and
   `shortest_leapfrog_acceptance_in_band_then_diagnostics`. The committed
   BayesFilter policy is 63 candidates with
   `eligible_trajectory_acceptance_in_band_then_rhat_convergence_then_ess`.
2. The Rotemberg bridge constructs `FlowFixedTransportAdapter` without the
   public `pullback_score`, `pullback_score_batch`, and Jacobian-score methods
   required by `FixedTransportValueScoreAdapter`.

The audited repairs were not applied because the same external patch operation
was rejected before mutation, including single-file retries after the user
requested continuation. No dsge_hmc files were changed by this run.
Archive tests remain excluded because of the known import-time segfault.

## Decision Table

| Decision | Primary criterion | Veto status | Next action | Nonclaim |
| --- | --- | --- | --- | --- |
| BayesFilter canonical interfaces | PASS | No BayesFilter hard veto | Keep two active interfaces and diagnostic compatibility routes | Does not prove universal adequacy |
| Mass tuning jointly with epsilon/L | PASS on oracle and mass contract | No stale-signature or target-health veto | Migrate consumers and rerun mass-specific suites | Does not rank mass arms as superior |
| MacroFinance migration | NOT PASS | Four consumer assertions fail | Apply audited repairs, rerun focused and full configured suites | 60/4 is not a migration pass |
| dsge_hmc migration | NOT PASS | Three consumer/bridge assertions fail | Apply audited repairs, rerun focused and configured suites | 47/3 is not a migration pass |
| Cleanup/quarantine | NOT STARTED | Cross-repo green prerequisite unmet | Do not remove shims or exports | No deletion authorized |

## Run Manifest

| Field | Value |
| --- | --- |
| Date | 2026-08-18 |
| Environment | `tfgpu`; Python 3.13.13; TensorFlow 2.20.0; TFP 0.25.0 |
| Device mode | CPU-only by explicit `CUDA_VISIBLE_DEVICES=-1` |
| TensorFlow allocator | `TF_FORCE_GPU_ALLOW_GROWTH=true` before import |
| BayesFilter source | Current dirty checkout; unrelated changes preserved |
| MacroFinance source | Current dirty checkout; no files changed by this run |
| dsge_hmc source | Current dirty checkout; no files changed by this run |
| Plan | `docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md` |
| Result | This file |

## Stop And Repair Record

The plan requires stopping before quarantine when a consumer contract is still
failing. That stop condition fired. The failure is localized to stale consumer
expectations and one missing bridge method family; it does not invalidate the
BayesFilter target, mass contract, posterior oracle, or route ledger. The next
attempt should apply only the listed repairs, rerun the identical focused
commands, then run each repository's configured full pytest paths. Only two
green cross-repository runs authorize Phase 7 quarantine and cleanup.
