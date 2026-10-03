# Claude Handoff: Ordinary HMC Migration-Debt And Plan Audit

Date: 2026-09-03

Requester: BayesFilter repository owner

Primary plan under review:
`docs/plans/bayesfilter-ordinary-hmc-migration-debt-trace-and-repair-plan-2026-09-02.md`

Self-review record:
`docs/plans/bayesfilter-ordinary-hmc-migration-debt-plan-review-2026-09-02.md`

Repository: `/home/ubuntu/python/BayesFilter`

Observed revision when this handoff was written:
`54201f5cd925ed15036bad8156606b812d53b045`

## Purpose

Perform an independent, source-grounded audit of the ordinary HMC tuning
trace, the migration-debt findings, and the proposed repair plan. The central
question is whether a future agent can identify the exact executable tuning
policy and evidence role from the public interface and resulting artifact, or
whether the current API, route registry, downstream consumers, and documents
permit the same frozen-epsilon/interface confusion to recur.

This memo requests review only. It does not authorize code edits, HMC/tuning
runs, benchmarks, package changes, network access, or a change to any numerical
default.

Amendment note (2026-09-03): the earlier Claude response was intentionally a
first invocation against the plan alone. Its verdict is valid only for plan
coherence; it did not inspect source code, the self-review, or downstream
consumers. The amended plan now records the source-verified distinction between
the deterministic ordinary default (operational warm-up/fixed-trajectory
selection) and the explicit legacy joint-grid branch, plus the confirmed gap
that public ordinary orchestration checks route support without requiring
`operational_authority`. Reviewers must preserve that distinction and must not
infer a runtime HMC result from static reachability.

## Required Review Mode

The repository policy requires the smallest exact path first. Each Claude
invocation must be read-only and bounded:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line: <one path>. Do not
edit, run commands, launch agents, or review the whole repo. Question: <one
question>. End with VERDICT: AGREE or VERDICT: REVISE.
```

Do not send a broad packet, pasted code bundle, whole-repository instruction,
or a list of paths in the first prompt. Start with the plan below. The first
response must not read this handoff's self-review record or any source path;
it must explicitly label all source-dependent questions as deferred. If the
answer needs more context, request the next exact path or line block and use a
new bounded invocation. The sequence below is an approved review order, not a
request to open everything in one invocation. A first-stage `AGREE` is not a
terminal code audit or numerical approval.

The worktree is intentionally dirty. Ignore unrelated changes and do not
recommend reverting them. Anchor every finding to the revision and path you
actually inspected. Some relevant guide, registry, generated-table, and
chapter files also have uncommitted overlays; distinguish `HEAD` from the
working-tree content rather than treating the overlay as committed evidence.
If line numbers moved, report the symbol and the observed line instead of
guessing.

The static review is bounded to the paths listed in this memo and the exact
symbols/blocks named for each path. For any external consumer scan, record the
explicit file list, exclusions (generated artifacts, virtual environments, and
unrelated repositories), command, and versioned output root. Do not expand to
a repo-wide or runtime search without a new bounded invocation.

## First Invocation

Review exactly:

`docs/plans/bayesfilter-ordinary-hmc-migration-debt-trace-and-repair-plan-2026-09-02.md`

Question:

> Is the plan internally coherent for a static source/documentation audit,
> does it separate confirmed facts from unresolved policy choices, and does it
> avoid authorizing numerical claims or an unreviewed default change? Check
> wrong-baseline assumptions, proxy metrics, stop conditions, fairness,
> hidden defaults, stale context, environment assumptions, and whether the
> proposed artifacts answer the stated question. Identify every material
> omission or unsupported claim with severity and exact section/line anchors.

Required final line: `VERDICT: AGREE` or `VERDICT: REVISE`.

## Sequential Source Review

Only continue to a path after the preceding response identifies a concrete
need for it or the owner explicitly requests the next step. Use one path per
invocation. For each path, ask Claude to compare the implementation with the
specific plan assertion and to classify each finding as `correct`, `wrong
relative to the stated target`, `unsupported`, or `not checked`.

### Dispatch and ordinary orchestration

1. `bayesfilter/inference/hmc_tuning_dispatch.py`

   Question: Does one public name dispatch materially different config
   variants and evidence contracts? Can a mechanics-only TensorFlow result be
   mistaken for ordinary artifact authority, and are forbidden combinations
   rejected?

2. `bayesfilter/inference/hmc_kernel_tuning.py`

   Review only the exact blocks containing `HMCKernelTuningConfig`,
   `run_hmc_windowed_mass_stage`, `run_hmc_fixed_mass_step_stage`,
   `_run_operational_fixed_mass_step_stage`, `_run_canonical_hmc_tuning`, and
   the Phase 7 candidate queue, plus the module header at lines 1-15. Do not
   ask for a whole-repository review.

Question: What branch does the default ordinary config actually execute?
   Confirm whether an operational warm-up deterministically forces the
   frozen-epsilon fixed-trajectory selector, and under what explicit route
   construction the `LEGACY_JOINT_L_EPSILON_ALGORITHM_ID` branch is reached.
   Determine whether the Phase 7 queue and direct final-kernel payload can
   carry that supported-but-non-promoting legacy route into an authority
   result, or whether a public `operational_authority` guard already prevents
   it. Are the policy IDs, module-header claim, and payload labels mutually
   consistent? Also determine whether the implicit `standard` preset's
   diagnostic role can issue or be mistaken for a serious authority artifact.
   Inspect the compatibility `tune_hmc_kernel` definition in this module and
   determine whether it creates an import-path/typing bypass. Check whether
   `use_xla=False` is an explicitly reviewed exception or an unguarded
   violation of the repository XLA-default policy. Also distinguish the route
   contract's operational stage authority from public artifact/scientific
   authority, and check the ordinary runtime's NumPy imports/calls against the
   diagnostic-only NumPy policy. If those calls are admitted-runtime use,
   state whether the route must remain non-admitting until repaired.

3. `bayesfilter/inference/hmc_kernel_selection.py`

   Review the exact blocks for `fixed_trajectory_candidate_values`,
   `FixedTrajectoryCandidate`, `FixedTrajectoryCandidateResult`, the
   operational selection loop, and exact-L retuning.

   Question: Are the bounded floor/anchor/double candidate set, shared
   epsilon, replication invariant, all-pass disposition, and post-nomination
   retune represented accurately? Which values are diagnostic screens rather
   than promotion or convergence evidence?

4. `bayesfilter/inference/hmc_tensorflow_tuning.py`

   Review the module header, `TensorFlowHMCKernelTuningConfig`, result payload,
   and `from_payload` validation.

   Question: Is this a mechanics/candidate handoff only? Does its public
   exposure through `tune_hmc_kernel` create an authority ambiguity, and which
   fields must be bound to distinguish it from ordinary tuning?

5. `bayesfilter/inference/hmc_tuning.py`

   Review the public-looking lower-level policies and diagnostic entry points,
   including `FixedTrajectoryTuningConfig`, `HMCTuningPolicy`, and the fixed
   mass/trajectory diagnostic helpers.

   Question: Which callers can reach these helpers without the public route,
   what controls do they own, and are their NumPy and nonclaim boundaries
   explicit enough to prevent accidental authority?

6. `bayesfilter/inference/hmc_budget_ladder.py`

   Review `FixedMassHMCTuningBudgetLadderConfig`, its defaults, result schema,
   and `run_fixed_mass_hmc_tuning_budget_ladder`.

   Question: Is this strictly a one-L diagnostic ladder, and can its payload or
   exports be mistaken for joint epsilon/L tuning or a canonical handoff?

7. `bayesfilter/inference/generic_hmc_tuning.py`

   Review `GenericHMCTuningConfig`, both orchestration functions, selection and
   artifact fields, and their nonclaims.

   Question: Does caller-owned Cartesian candidate selection remain reachable
   near claim-bearing consumers, and which fields would have to be removed or
   explicitly marked diagnostic?

8. `bayesfilter/inference/hmc_fixed_metric_grid_search.py`

   Review the fixed-metric grid constants, callback/adaptation boundary, and
   result authority fields.

   Question: Is its grid a historical/diagnostic comparator only, and does any
   code path emit language or artifacts that imply public tuning authority?

9. `bayesfilter/inference/hmc_operational_broad_grid.py`

   Review `ROUTE_ID`, candidate-grid construction, neighbor guards, and public
   result metadata.

   Question: Does this route share or diverge from the ordinary policy, and is
   that divergence visible to consumers and replay validators?

10. `bayesfilter/inference/hmc_robust_broad_grid.py`

    Review its per-L adaptation/repair, qualification settings, backend/XLA
    defaults, and route replacement metadata.

    Question: Is it a diagnostic replacement only, or can a caller mistake its
    richer evidence for the active ordinary route? Identify unsupported
    transferred defaults.

11. `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py`

    Question: Does the tiny-Gaussian historical route remain isolated from
    active imports and clearly non-claim-bearing?

### Route and evidence contracts

12. `bayesfilter/inference/tuning_contract.py`

   Review the active route registry, capability records, validation, and
   `require_active_hmc_tuning_route` blocks.

   Question: Does the registry classify policy/config variants, preset roles,
   and internal algorithm IDs, or only function names? Can the capability prose
   claim joint L/epsilon ownership while the default executes another policy?

13. `bayesfilter/hmc_route_contract.py`

   Question: Are operational, legacy joint-grid, and stage-level route IDs
   resolved and authorized consistently? Is the combination of a legacy or
   non-promoting algorithm with an active public authority route possible?
   Reconcile `promoted_default` payload language with route-contract language,
   and specify the smallest construction-only fail-closed test for a legacy
   public authority request. Do not treat `operational_authority=True` as
   artifact or scientific authority: inspect whether an independent
   `artifact_authority`/consumer-role decision exists and what invariant binds
   the fields.

14. `bayesfilter/inference/hmc_verification.py`

   Review `HMCAcceptancePolicy` and the relevant verification payload fields.

   Question: Are the chain/block/sample counts, target/bands, t-screen, R-hat,
   ESS, movement, divergence, and energy diagnostics assigned the roles the
   plan claims? Flag any metric that could be read as a convergence or
   superiority criterion without uncertainty evidence.

15. `bayesfilter/inference/hmc_tuning_artifacts.py`

   Review `private_start_bank_summary`, the tuning-artifact builder/validator,
   and `load_and_replay_hmc_tuning_artifact`.

   Question: Is start-bank identity and seed lineage included in selection and
   replay scope? Could candidate comparisons use different starts without the
   artifact showing it? Does the artifact validator bind the algorithm route
   strongly enough for an authority decision?

16. `bayesfilter/inference/hmc_artifact_identity.py`

   Review the repository-owned mass/adapter signature helpers used by ordinary
   tuning and replay.

   Question: Do the signatures cover every geometry and scope field that can
   change the selected transition, and are they computed from live values
   rather than caller-supplied labels?

17. `bayesfilter/inference/hmc_kernel_tuning.py`

   Review only the replay blocks containing
   `RetainedFrozenKernelAdapterReplayResult`,
   `_validated_retained_replay_header`, and
   `build_retained_frozen_kernel_hmc_adapter_from_tuning_payload`.

   Question: Does replay bind the resolved policy/config variant, algorithm
   ID, mass, epsilon, L, target/coordinates, backend/dtype, and source
   closure? Identify any missing invalidation field and whether caller-supplied
   identity can be forged.

### Inventory, exports, and tests

18. `bayesfilter/inference/__init__.py`

    Question: Which diagnostic, historical, stage, and compatibility helpers
    are re-exported beside the active tuners? Identify the smallest export or
    naming change that reduces accidental selection without breaking declared
    historical readers.

19. `bayesfilter/__init__.py`

    Question: Does the top-level lazy/export map expose the same ambiguous
    helpers, and can its public surface be tested against the role ledger?

20. `scripts/inventory_hmc_tuning_routes.py`

    Question: What does this scanner intentionally exclude? Would it pass when
    a caller imports a private helper, constructs `FullChainHMCConfig`, or
   selects a different typed config under the same public name? Specify the
   smallest extension that adds consumer/bypass classification without
   misclassifying legitimate mechanics or smoke routes. Require an
   `unknown_dynamic_import` classification for computed importlib/getattr,
   entry-point, or other unresolved indirection; it must block
   claim-adjacent admission until manually resolved.

21. `tests/test_hmc_tuning_contract.py`

    Question: Which authority and facade invariants are actually enforced,
    and which policy-identity or dispatch cases are untested?

22. `tests/test_hmc_tuning_documentation_contract.py`

   Question: Do these tests compare executable default policy with the guide,
   or only route counts, strings, and generated-file freshness? Verify that
   the focused commands are construction-only and cannot initialize an HMC
   runner or device; otherwise classify that claim as `not checked`. Propose
   exact construction-only tests for the identified gap.

### Reader-facing documents and generated views

23. `docs/reference/hmc-tuning-interface.md`

    Question: Does the guide state the exact ordinary default config variant,
    resolved policy ID, candidate set, adaptation relationship between L and
    epsilon, and authority/nonclaim status? Identify wording that is true only
    for an alternate or diagnostic branch.

24. `docs/generated/hmc_tuning_route_table.md`

    Question: Does the generated table expose enough policy detail to prevent
    a reader from interpreting "joint leapfrog-count and epsilon selection" as
    the operational default? Is the source of truth correctly identified?

25. `docs/chapters/ch21b_hmc_tuning_interfaces.tex`

   Question: Does the chapter preserve the same branch and evidence-role
   distinctions as the guide, without turning internal workflow labels into
   reader-facing claims?

26. `docs/main.tex`

    Question: Is the repaired chapter actually included in the manuscript at
    the intended location, and would a rendered build expose the ordinary
    policy guidance rather than only the generated table? Check ordering and
    build wiring; do not treat inclusion as proof of content correctness.

Optional historical context, only if the preceding documentation review finds
an XLA-default contradiction:

`docs/plans/bayesfilter_hmc_kernel_tuning_xla_parameter_repair_result_2026_06_22.md`

Question: Does this result clearly separate parameter-propagation evidence
from the current default policy, or could its ``Default non-XLA behavior``
wording be copied as active guidance? Require a supersession classification;
do not reinterpret the historical numerical checks.

### Downstream consumer audit

Inspect one exact file per invocation. Begin with the files below; add another
only when a finding requires it. For each, determine whether the file is
claim-bearing, candidate, mechanics-only, smoke/reference, or historical, and
whether its metadata matches its actual calls.

27. `../MacroFinance/mixed_frequency_tfp_c2_full_bayesfilter_hmc_tuning_v2_phase4_step_trajectory.py`

    Check its `run_fixed_mass_hmc_tuning_budget_ladder` and generic-orchestration
    calls, explicit epsilon/L/budget settings, and any field called tuning
    authority.

28. `../MacroFinance/cross_country_multi_asset_bayesfilter_owned_hmc_client.py`

    Check its `orchestrate_generic_hmc_tuning` calls, module-level claim that
    BayesFilter owns tuning, private imports, and raw full-chain calls.

29. `../dsge_hmc/scripts/run_rotemberg_round380_r16_tuning_repair.py`

    Check whether the budget ladder is merely diagnostic or is used to choose
    a claim-adjacent kernel, and whether the artifact says so.

30. `../dsge_hmc/scripts/run_rotemberg_public_explicit_state_fixed_mass_tuning.py`

    Check private imports, `use_xla`/CPU choices, raw HMC execution, and role
    metadata. Do not infer authority from the filename.

31. `../dsge_hmc/scripts/run_bgs_bayesfilter_phase08_stage_c_grid_tuning.py`

    Check fixed-metric/operational-broad grid use, policy identity, and
    whether it is historical, candidate, or claim-bearing.

32. `../MacroFinance/scripts/run_ccma_broad_fixed_metric_l_epsilon_search.py`

    Check fixed-metric grid ownership, candidate evidence, and downstream
    promotion language.

33. `../MacroFinance/daily_asset_midas_robust_broad_grid_tuning.py`

    Check whether its `tune_hmc_kernel` call resolves the ordinary default or
    a separately documented robust diagnostic policy, including `use_xla` and
    policy metadata.

34. One representative raw-mechanics script, selected from the preceding
    findings (for example
    `../dsge_hmc/scripts/run_bgs_full_estimation_hmc_stage.py`).

    Check whether `FullChainHMCConfig`/`run_full_chain_tfp_hmc` is explicitly
    classified as mechanics-only and whether its artifact forbids tuning or
    posterior claims.

35. `../MacroFinance/scripts/run_daily_asset_midas_phase14_c3_fixed_mass_trajectory_diagnostic.py`

    Check its direct calls to stage helpers and start-bank diagnostics. Verify
    that the file is explicitly diagnostic and cannot be read as a public
    ordinary-tuner handoff.

36. `../dsge_hmc/scripts/run_rotemberg_round380_r16_consolidated_program.py`

    Check its direct stage-helper references, route metadata, and whether the
    program bypasses the public dispatcher for any claim-adjacent decision.

Do not require migration of every raw mechanics smoke test. The audit should
instead identify the smallest set whose role is ambiguous or claim-adjacent.

## Cross-Path Questions Claude Must Resolve

After the bounded invocations, synthesize only the inspected evidence and
answer:

1. Is the statement "the ordinary public tuner owns joint epsilon and L"
   correct for the executable default, correct only as a capability goal, or
   wrong relative to the stated target?
2. Is the old operational selector still an active authority route, a private
   diagnostic handoff, or both depending on stage?
3. Is the joint-grid route genuinely legacy, genuinely promoted, or merely
   mislabeled? Name the exact source changes needed to make one answer true.
4. Does artifact/replay identity prove the branch that ran? If not, specify the
   minimum schema and test repair.
5. Which downstream files require migration now, and which should receive an
   explicit nonclaim/classification only?
6. Which NumPy imports and exports are actual admitted-runtime debt versus
   diagnostic/reference code?
7. Does the plan need a different baseline, criterion, budget, stop condition,
   or phase ordering before implementation?
8. Is the ordinary route's `use_xla=False` default compliant with the repository
   policy, or must the plan require XLA-on or a scope-bound exception record?
9. Does the compatibility definition of `tune_hmc_kernel` need to be removed,
   privatized, or schema-aligned before downstream migration can be considered
   complete?

10. Can a public ordinary call select a supported-but-non-promoting legacy
    algorithm and still reach `_phase7_direct_final_kernel_payload`? If so,
    identify the exact guard and construction-only regression required to make
    the authority boundary fail closed. Distinguish direct diagnostic stage
    tests from public authority construction.

For each answer, distinguish:

* hard source fact;
* derivation from the source;
* unresolved owner choice; and
* empirical question that cannot be answered without a future experiment.

## Required Claude Output

Return a concise but evidence-dense memo with:

* severity-ordered findings;
* exact file/symbol/line anchors for every material finding;
* the four classifications `correct`, `wrong relative to the stated target`,
  `unsupported`, or `not checked`;
* a table mapping each proposed phase to required repair, test, and artifact;
* explicit corrections to the plan, if any;
* unresolved owner decisions and numerical questions that must remain blocked;
* any backend-policy exception and import-path guard needed before authority can
  be issued;
* a statement of whether the static phase may proceed; and
* a final line exactly of the form `VERDICT: AGREE` or `VERDICT: REVISE`.

For every source-dependent item not yet inspected, use the literal
classification `not checked`; do not infer it from a filename, age, route
name, or prior artifact. The phase table must identify prerequisite ordering:
route/policy construction precedes artifact serialization and documentation
checks, and construction/replay/serialization success is an engineering
prerequisite, never promotion evidence. Constant-string dynamic imports must
be resolved by AST or manual call-site inspection; runtime HMC instrumentation
is outside this static review. Any recommendation to change XLA or numerical
defaults must remain an unresolved owner decision with a separate evidence
contract.

Computed `importlib`, `getattr`, entry-point, and other non-constant dynamic
paths must receive an explicit `unknown_dynamic_import`/`unresolved` row and
block claim-adjacent admission until manually classified. The method
comparison baseline ladder (naive, best tuned classical, plain proposed,
enhanced proposed) must be included in any future numerical plan where it is
applicable, or its inapplicability must be justified. Claims that a pytest or
renderer command is construction-only are `not checked` until the named
sources establish that they cannot initialize an HMC runner or device.

An `AGREE` verdict is valid only for the bounded question actually reviewed.
It must not be interpreted as numerical validation, posterior admission,
scientific superiority, or approval of a new ordinary default. If a required
path was not inspected, Claude must say `not checked` rather than infer it.

## Handoff Boundary

This memo is the audit request and review protocol. It is not an execution
manifest. After Claude's response, the owner/Codex should adjudicate any
material findings, update the plan and review record, and only then decide
whether to implement the static guards. A separate reviewed experiment plan is
required before any HMC or tuning run.
