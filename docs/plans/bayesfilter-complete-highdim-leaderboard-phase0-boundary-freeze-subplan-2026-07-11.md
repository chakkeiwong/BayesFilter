# Complete High-Dimensional Leaderboard Phase 0 Subplan

Date: 2026-07-11

Status: `DRAFT_REVIEW_REQUIRED`

## Material Review Question

Does this Phase 0 subplan and its declared freeze contract correctly bind the
six main rows, sidecar, baselines, current sources, missing-cell matrix, and
execution boundaries before Phase 1 begins?

## Phase Objective

Freeze the six-row main matrix, parameterized-SIR sidecar boundary, baseline
and source hashes, declared row metadata and LEDH execution seeds, current
missing-cell matrix, review roles, and execution authority before implementation
or GPU work. Byte-level canonical main-row target identities are a mandatory
Phase 1 pre-implementation gate, not a Phase 0 claim.

## Entry Conditions

- The owner requested execution of the revised complete-leaderboard program.
- Predator-prey FD resolution and generalized-SV manual Sinkhorn JVP bugs are
  repaired and their bounded root-cause artifacts validate.
- The worktree is dirty with another active lane; Phase 0 may create only the
  new `complete-highdim-leaderboard` namespace and dedicated scripts/tests.
- No Phase 1 implementation, GPU run, detached launch, merge-back, commit, or
  public release is authorized by this subplan.

## Normative Matrix

Algorithms, in this exact order:

1. `fixed_sgqf`
2. `ukf`
3. `zhao_cui_scalar_or_multistate`
4. `ledh_pfpf_ot`

Main rows, in this exact order:

| Row | T | N for LEDH only | Parameter order | Evaluation theta | Target observation policy |
| --- | ---: | ---: | --- | --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | 50 | 10000 | `phi1,phi2,phi3,q_scale,r_scale` | `0.72,0.55,0.35,0.35,0.45` | `lgssm_gaussian_observation_density` |
| `zhao_cui_sv_actual_nongaussian_T1000` | 1000 | 10000 | `gamma_unconstrained,log_beta` | `0.2533471031357997,-0.916290731874155` | `transformed_actual_sv_log_y_square` |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | 1000 | 10000 | `gamma_unconstrained,log_beta` | `0.2533471031357997,-0.916290731874155` | `ksc_log_chi_square_gaussian_mixture_surrogate` |
| `zhao_cui_spatial_sir_austria_j9_T20` | 20 | 10000 | `log_kappa_scale,log_nu_scale,log_obs_noise_scale` | `0,0,0` | `fixed_sir_infectious_components_gaussian_observation_density` |
| `zhao_cui_predator_prey_T20` | 20 | 10000 | `r,K,a,s,u,v` | `0.6,114,25,0.3,0.5,0.5` | `additive_gaussian_predator_prey` |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | 1008 | 10000 | `gamma_unconstrained,log_tau,mu` | `1.0824113944610982,-2.076793740349318,0` | `source_route_prior_mean_generalized_sv` |

The exact sidecar id is
`zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale`, with scope
`scoped_component_row`. It is not a main row.

Phase 0 freezes declared target metadata, not byte-level observation identity.
Phase 1 must generate and bind the canonical target signatures required by the
master before any harness edit or production execution. Failure of that
pre-gate is a continuation veto, not permission to infer identity from labels.

LEDH execution seeds are ordered exactly
`[81120, 81121, 81122, 81123, 81124]` for every main-row LEDH value/score/FD
pair and are not target-generation seeds. Target-generation identities are
separate: LGSSM `81100`, actual-SV and KSC-SV `81101`, predator-prey `81104`,
generalized-SV `81105`, and fixed Austria SIR has fixed source observations
with no synthetic dataset seed declared here. Non-LEDH deterministic methods
do not inherit LEDH execution seeds; any stochastic non-LEDH method must freeze
its own execution seed/configuration in its phase subplan.

## Sidecar Boundary

The sidecar has `T=20`, parameter order
`log_kappa_scale,log_nu_scale,log_obs_noise_scale`, and target scope
`local_complete_data_zhao_cui_sir_d18_component`. The July 3 source records:

| Algorithm | Frozen sidecar status |
| --- | --- |
| `fixed_sgqf` | `not_applicable_to_scoped_component_row` |
| `ukf` | `not_applicable_to_scoped_component_row` |
| `zhao_cui_scalar_or_multistate` | historical scoped component value/score exists, but remains outside current-program admission and outside the 24-cell completeness criterion |
| `ledh_pfpf_ot` | July 6 scoped diagnostic/status only; not a full observed-data score and not current admission |

Phase 0 does not freeze a four-way sidecar target signature or require sidecar
closure. Sidecar observations, evaluation coordinates, and seed applicability
are outside this complete-main-leaderboard program. The only binding rule is
that sidecar evidence remains labeled with its local-complete-data scope and
never enters a main row or the 24-cell totals.

## Normative Input Bindings

| Path | Expected SHA-256 | Role |
| --- | --- | --- |
| `docs/plans/bayesfilter-two-lane-highdim-leaderboard-results-2026-07-03.json` | `b44fd1ccc8a0132d45ea4f64925bd92930a17c11f7b62bc8f0a15f66631985e7` | sole starting-status source for all non-LEDH cells; candidates only |
| `docs/plans/bayesfilter-two-lane-highdim-ledh-inclusive-leaderboard-results-2026-07-06.json` | `57317fb8f0b4a55c3357a7014f1d68647278657b11843460f90e4f95383900d0` | LEDH historical status only; never current admission |
| `docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json` | `21e87489c8eb661db4b2e9b27cefb4e45e567a8c0bb4743ffd4f09feec3faf93` | LGSSM target/shape and historical forward evidence only |
| `docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json` | `38a7da0ef1f32f96e74d4f62676d823af2fbe1b4267d88dbfa0c39c4156ba9b8` | fixed-SIR target/shape and historical forward evidence only |
| `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json` | `17eaaf23302fa68e802eef686b167e4b31cc3dba755503f9b74343d2ca29ef45` | predator-prey target/shape and historical forward evidence only |
| `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json` | `3811268078d07e0ac4c2fcd9400af156a5918503e404937d516391ce0f034c16` | actual-SV target/shape and historical forward evidence only |
| `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json` | `5afb71144576bdb0070080f684b5d5b41f33de77889105b10bcd78e36b77dd77` | generalized-SV target/shape and historical forward evidence only |
| `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json` | `9883721faf8af9fbe96ef75c209f86eda5732aec6ca5e602980d4cf27338b3b6` | KSC-SV target/shape and historical forward evidence only |
| `bayesfilter/highdim/ledh_score_contract.py` | `aa15f058b30850c940b978491080893353c519c3ee31a344d0d42f20b81aeef3` | current source identity |
| `bayesfilter/ledh_fd_policy.py` | `32c20ab5467c464a32bd2f098b0a1f1c0e67765890007126349abc6434edd2b5` | current FD-only policy identity |
| `docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py` | `2bd7c4c62773657213ccd488c9e55b96f3f7d6d4a3b00a7aaf2a8fb070031d58` | current schema-v4 harness identity; known Phase 1 repair target |
| `docs/benchmarks/benchmark_two_lane_highdim_ledh_inclusive_results.py` | `dcc176e4e3533abfd609b27fc52db3dc3c608de27d88606e15f4ae8bb60bd365` | current historical builder identity; replacement target |
| `experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py` | `a9d680cc90ad59655a35268766213bb452d6ab703993918600148194364383fe` | current repaired Sinkhorn source identity |
| `docs/plans/bayesfilter-ledh-predator-generalized-fd-root-cause-repair-result-2026-07-11.md` | `42630b9ab97cdcb39d4ecd8c0fdc172647a63b86c5c3a478fd8efd23352f1fed` | current FD/JVP repair authority |

Any byte drift is a Phase 0 failure. Updating a hash after drift requires a
visible plan amendment and renewed review; the generator must not self-update
expected hashes.

## Normative Starting Matrix

For each of the first three main rows (LGSSM, actual-SV, KSC-SV), the three
non-LEDH algorithms are `frozen_nonledh_baseline_candidate`; LEDH is
`gap_current_source_five_seed_ledh_admission`.

For each of the last three main rows (fixed-SIR, predator-prey,
generalized-SV), all three non-LEDH algorithms are
`gap_target_matched_value_and_score_evaluator`; LEDH is
`gap_current_source_five_seed_ledh_admission`.

Therefore the exact totals are nine frozen candidates and 15 closure gaps.
Every main cell remains `current_program_admitted=false`. Non-LEDH cells must record
the July 3 source path/hash; LEDH cells must record the July 6 historical
source path/hash. The four sidecar cells retain the exact unadmitted
dispositions in the Sidecar Boundary table; they are not closure gaps and do
not affect the 24-cell totals.

## Authority Roles

- Codex is supervisor and executor.
- Claude is a read-only reviewer. `AGREE` is advisory and cannot override a
  local veto or human boundary; `REVISE` triggers visible repair.
- After two trusted health probes fail, a fresh Codex read-only substitute may
  review one exact path. Its agreement is weaker evidence and cannot alone
  approve Zhao-Cui source-faithfulness or final release.
- The human retains authority over target/scope/threshold changes, public or
  default changes, extensions/inventions, credentials/funding/packages,
  destructive actions, detached launch, merge-back, and scientific claims.

## Research Intent Ledger

| Field | Intent |
| --- | --- |
| Question | Are the exact targets and missing cells sufficiently frozen to plan implementation without silently changing the leaderboard? |
| Baselines | July 3 non-LEDH JSON; July 6 LEDH-inclusive JSON as historical status only; six July 7 LEDH target/shape and historical-forward artifacts; current score/FD/transport sources. |
| Primary criterion | A generated JSON freeze validates exact hashes, six main rows, four algorithms, 24 main cells, one scoped sidecar boundary, declared row shapes, ordered LEDH execution seeds, target-policy labels, parameter orders, evaluation theta, and the current missing-cell matrix. |
| Promotion veto | Missing/hash-drifted input, wrong row scope, duplicate/missing cell, wrong evaluation theta or coordinate order, sidecar counted as main, stale LEDH artifact treated as current admission, or current source identity omitted. |
| Continuation veto | Target contradiction, unavailable required artifact, unsafe dirty-work overlap, or nonconvergent material review. |
| Repair trigger | Missing metadata field in the Phase 0 checker or a review finding that can be fixed without changing the target matrix. |
| Explanatory only | Historical executed statuses and descriptive values; Phase 0 does not re-admit them. |
| Nonclaims | No evaluator correctness, GPU readiness, cell admission, complete leaderboard, ranking, or scientific claim. |

## Required Artifacts

- Master program:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-master-program-2026-07-11.md`
- Phase 0 freeze JSON:
  `docs/plans/artifacts/complete-highdim-leaderboard/phase0-boundary-freeze-2026-07-11.json`
- Phase 0 result:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-phase0-boundary-freeze-result-2026-07-11.md`
- Review-receipt manifest:
  `docs/reviews/bayesfilter-complete-highdim-leaderboard-phase0-review-receipts-2026-07-11.json`
- Next subplan:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-phase1-ledh-harness-subplan-2026-07-11.md`
- Visible runbook, ledger, stop handoff, detached plan, and supervisor prompt
  named in the master-program runbook set. The exact launcher-command manifest
  is reserved for the later launch-infrastructure gate and is not required for
  Phase 0 closure.
- Generator/checker:
  `scripts/build_complete_highdim_leaderboard_phase0_freeze.py`
- Independent literal-manifest auditor, which must not import the generator:
  `scripts/audit_complete_highdim_leaderboard_phase0_freeze.py`
- Focused test:
  `tests/test_complete_highdim_leaderboard_phase0_freeze.py`

Launch-only paths are reserved but are not Phase 0 pass artifacts:

- `scripts/complete_highdim_leaderboard_overnight_supervisor.sh`;
- `docs/plans/complete-highdim-leaderboard-exact-command-manifest-2026-07-11.json`.

They require their own implementation, tests, hashes, preflight, and human
launch approval after Phase 0 closes.

Review receipts:

- Claude availability:
  `docs/reviews/bayesfilter-complete-highdim-leaderboard-claude-availability-2026-07-11.md`;
- master review receipts:
  `docs/reviews/bayesfilter-complete-highdim-leaderboard-master-program-codex-substitute-review-iter<N>-2026-07-11.md`;
- Phase 0 review receipts:
  `docs/reviews/bayesfilter-complete-highdim-leaderboard-phase0-codex-substitute-review-iter<N>-2026-07-11.md`.

Each receipt must record reviewer type, exact reviewed path and SHA-256, exact
question, iteration, findings, terminal verdict, and receipt path. A separate
review-receipt manifest must record the final SHA-256 of every immutable
receipt, avoiding an impossible self-hash inside a receipt. Substitute receipts
must bind the two trusted Claude probe outcomes and state their weaker
authority. The Phase 0 result must enumerate the manifest and every receipt
used to claim convergence.

## Required Checks And Reviews

1. Generate the freeze JSON from repository inputs, never by copying expected
   output into the test.
2. Run the checker in `--check` mode.
3. Run the independent literal-manifest auditor against the stored JSON.
4. Run focused tests that contain literal expected identities and invoke the
   independent auditor; they must not derive expectations only from generator
   module constants.
5. Run `py_compile` and scoped `git diff --check`. Shell syntax checks for the
   reserved supervisor script occur at the separate launch-infrastructure gate.
6. Review the master program as one exact path with one exact governance
   question.
7. Review this subplan as one exact path with one exact Phase 0 question.
8. If either review returns `REVISE`, patch visibly and loop at most five times.

CPU-only Python checks must set `CUDA_VISIBLE_DEVICES=-1` before framework
import when applicable. Phase 0 itself does not initialize TensorFlow.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Exact question | Does the frozen matrix correctly separate six main rows from the sidecar and bind every baseline/source input needed by later phases? |
| Comparator | Bytes and normalized metadata of the specified repository artifacts. |
| Primary pass criterion | Generated freeze artifact and independent focused tests agree on all identities and hashes; material reviews agree; no veto fires. |
| Veto diagnostics | Hash mismatch, row/algorithm set mismatch, wrong declared target policy/shape/LEDH execution seeds/parameters, sidecar contamination, canonical-target pre-gate omitted from Phase 1, or unsafe source-worktree edit. |
| Explanatory diagnostics | Historical statuses and missing-cell counts. |
| What is not concluded | No numerical cell, implementation, GPU, review availability, or launch claim. |
| Result artifact | Phase 0 JSON and result Markdown. |

## Forbidden Claims And Actions

- Do not call the current July 6 output complete or current-source admitted.
- Do not infer that historical non-LEDH score provenance is correct merely
  because a cell is marked executed.
- Do not edit shared model runners, contracts, source artifacts, historical
  leaderboard artifacts, or other-lane files.
- Do not invoke GPU/CUDA or a detached process in Phase 0.
- Bounded read-only Claude review, or the documented substitute path after two
  trusted health-probe failures, is permitted after the local Phase 0 review
  preflight. It does not depend on launcher commands existing.
- Do not commit, push, merge, or delete files.

## Pre-Mortem

| Failure | Cheap discriminator |
| --- | --- |
| July 6 LEDH scores are accidentally treated as current | Freeze marks July 6 artifact `historical_status_only` and current admission count zero. |
| Parameterized SIR enters the main matrix | Assert exactly six main rows and a separate sidecar row id. |
| A target coordinate changes silently | Store exact parameter order, evaluation theta, target policy, and source hash for every LEDH row. |
| Dirty work is overwritten | Limit Phase 0 writes to the dedicated namespace and new scripts/tests. |
| Later plan assumes nonexistent evidence | Store a cell-by-cell starting status and required closure class. |
| July 6 composite silently becomes the non-LEDH baseline | Source every non-LEDH cell directly from the SHA-bound July 3 artifact, source only LEDH status from July 6, and validate the per-cell source path/hash. |

## Handoff Conditions

Advance to Phase 1 only if:

- Phase 0 JSON generation/check/tests pass;
- master and Phase 0 reviews converge;
- the Phase 0 result directly records all identities, hashes, limitations, and
  review status;
- the exact review-receipt manifest exists, validates every final receipt hash,
  and is enumerated by the Phase 0 result;
- the Phase 1 subplan is refreshed from the frozen JSON and reviewed;
- Phase 1 begins with the canonical-target signature pre-gate and forbids
  harness edits until that pre-gate passes;
- the visible and detached execution boundaries remain separate;
- exact launch commands and approvals remain unexecuted until their own gate.

## Stop Conditions

- Any required baseline/source artifact is missing or hash-unstable during the
  phase.
- The six-row target matrix conflicts with an admitted source artifact.
- The dedicated namespace overlaps unexpected concurrent edits.
- Review does not converge within five rounds.
- Continuing requires a human decision about row scope or target identity.
