# Phase 8 Subplan: Institutionalization And Program Closeout

Date: 2026-07-14

## Phase Objective

Make the certified batch-native target binding and strict dense-IAF trainer the
discoverable BayesFilter NeuTra training route, preserve fail-closed policy
tests and retired-route boundaries, transfer the operative knowledge into a
reset memo, and close the master program with separate engineering, numerical,
performance, and scientific ledgers. Prepare but do not execute the fresh
5,000-step campaign handoff.

## Entry Conditions Inherited From Phase 7

- Batch binding v2, exact target parity, status semantics, objective gradient,
  determinism, and trusted GPU/XLA execution are certified.
- All fresh Phase 7 smokes, stability, and screen arms passed.
- `wide_2x_lr5e3` is a proxy nominee only; all four recipes remain viable and no
  ranking is statistically supported.
- Screen weights cannot be reused for final training.
- The existing 5,000-step seeds remain unexecuted and require the refreshed
  handoff budget written by this phase.

## Required Artifacts

- Lazy public exports for the repository-issued batch binding and strict trainer.
- Public API tests and policy tests for batch-native admission.
- Focused tests proving the two obsolete LGSSM optimizer entry points remain
  retired before GPU/artifact side effects.
- Finalizer regression tests that revalidate the fresh Phase 7 evidence and
  bind selected-recipe identity to the final screen artifact.
- A knowledge-transfer/reset memo naming the live route, retired routes,
  evidence, performance, and exact next campaign.
- A fresh 5,000-step campaign handoff with measured compute budget and stop
  conditions, but no long training output.
- Phase 8 result and terminal master-program result.

## Implementation Scope

1. Export from `bayesfilter.inference` and top-level `bayesfilter`:
   `NeuTraBatchTargetBinding`, `InvalidNeuTraBatchTarget`, binding schema and
   helpers, `PlainDenseIAFTrainingConfig`, result/transport/error types, and
   `train_plain_dense_iaf`.
2. Preserve lazy top-level import behavior; TensorFlow may load only when a
   TensorFlow-backed NeuTra symbol is resolved.
3. Add focused public API and Phase 7 finalizer tests.
4. Verify the historical affine CPU fixture and Phase 16 bounded trainer still
   reject before optimizer/artifact side effects.
5. Write the reset memo, long-run handoff, and terminal ledgers.

The exact LGSSM batch adapter remains target-specific test/campaign
infrastructure. Phase 8 does not promote that fixture loader as a general public
API and does not change the numerical kernel or recipe.

## Required Checks

- Python compile checks for all touched modules/scripts.
- `tests/test_v1_public_api.py` including lazy-import regression.
- `tests/test_neutra_batching.py` and `tests/test_neutra_training.py`.
- exact adapter/materialization/kernel focused tests.
- retired-route tests in `tests/test_lgssm_neutra_training_tf.py` and
  `tests/test_neutra_gpu_bounded_training_tf.py`.
- new fresh-screen finalizer tests.
- existing Phase 5 certification matrix or a superset in isolated CPU-hidden
  processes.
- source audit for no NumPy, sample-axis mapping, or host callbacks in the live
  batch binding/trainer/adapter/kernel/materialization closure.
- no GPU run is required because Phase 8 changes exports, tests, and docs only;
  the trusted GPU behavior is unchanged and already evidenced in Phases 6-7.

## Evidence Contract

| Item | Phase contract |
| --- | --- |
| Engineering question | Can a caller discover and use the one admitted batch-native NeuTra training route without selecting retired fixtures? |
| Exact baseline | certified Phase 5 binding/trainer behavior and fresh Phase 7 artifacts |
| Pass criterion | exports resolve correctly, focused tests pass, retired routes fail before side effects, finalizer evidence revalidates |
| Promotion veto | public export changes behavior, import laziness breaks, retired route executes, policy test fails, or artifacts no longer validate |
| Explanatory only | test counts, doc hashes, and estimated future runtime |
| Nonclaims | Phase 8 does not establish posterior correctness, HMC convergence, recipe superiority, cross-model generalization, or default scientific readiness |
| Preserved result | Phase 8 result, reset memo, long-run handoff, and terminal program result under `docs/plans` |

## Fresh Long-Run Handoff Budget

Measured nominated-arm times are `82.99 s` compiled and `247.53 s` total for
500 steps. A linear 5,000-step estimate is approximately `13.8 min` compiled
and `16.6 min` wall per seed. The handoff therefore authorizes, but Phase 8 does
not launch:

- two sequential fresh seeds, `dense_seed1201` and `dense_seed1202`;
- 5,000 optimizer steps each, batch 128, selected recipe
  `wide_2x_lr5e3`, GPU/XLA/float64/binding v2;
- maximum aggregate compiled-program time `45 min`;
- maximum aggregate wall time `60 min`;
- at most one infrastructure resume per seed into a fresh output directory;
- no retry after a numerical/target veto without a focused repair result;
- no reuse of screen weights or screen optimizer state.

This budget is an engineering authorization for the next campaign, not evidence
that the result will pass downstream validation.

## Skeptical Subplan Audit

| Risk | Audit response |
| --- | --- |
| Wrong baseline | exports and tests preserve the already certified implementation; no alternate target/kernel is introduced |
| Proxy promotion | nominee language remains explicit; Phase 8 performs no scientific promotion |
| Hidden long run | 5,000-step execution is forbidden here and isolated in a separate handoff |
| Public API bloat | export only the generic batch binding/trainer, not the LGSSM campaign fixture |
| Import regression | retain lazy module resolution and run the no-TensorFlow-on-base-import test |
| Stale artifacts | finalizer tests recompute source-row and result-file hashes from the fresh Phase 7 roots |
| Retired route ambiguity | both legacy optimizer entry points must continue to raise before any artifact write |
| NumPy drift | live route closure source audit remains a hard check; diagnostic campaign NumPy is not promoted into the public trainer |
| Budget extrapolation error | use generous wall/program ceilings and stop conditions; estimate is planning only |

Audit verdict before execution: **PASS**. The phase closes discoverability and
knowledge-transfer gaps without changing target math, running unbudgeted GPU
work, or treating a proxy nominee as a scientific winner.

## Forbidden Claims And Actions

- Do not launch 5,000-step training or HMC in Phase 8.
- Do not reuse screen weights, checkpoint state, or optimizer moments.
- Do not call `wide_2x_lr5e3` best, superior, production-ready, or default-ready.
- Do not promote diagnostic NumPy campaign code as the training backend.
- Do not revive either retired legacy trainer.
- Do not change the exact target, SVD/eigh kernel, optimizer recipe, seeds, or
  downstream scientific criteria in this closeout phase.

## Exact Program Handoff Conditions

The master program closes when all required checks pass, live/retired route
boundaries are documented, the reset memo and fresh long-run handoff exist, and
the terminal result separates engineering readiness from unresolved scientific
validation. The next program may launch the two fresh 5,000-step seeds using
the handoff without repeating this migration.

## Stop Conditions

Stop only for a real public API behavior regression, failed batch-native
identity/policy test, executable retired route, corrupted Phase 7 evidence, or
an unrepairable mismatch with the certified target. Repair localized export,
test, or documentation defects inside this phase and rerun focused checks.

## Phase-End Procedure

1. Run the required local checks.
2. Write the Phase 8 result/close record.
3. Write the reset memo and refreshed long-run handoff.
4. Review terminal program suitability and close when no real blocker remains.

