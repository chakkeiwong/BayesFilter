# Ordinary HMC Migration-Debt Static Execution Note

Date: 2026-09-03

Plan executed:
`docs/plans/bayesfilter-ordinary-hmc-migration-debt-trace-and-repair-plan-2026-09-02.md`

Review records:

* `docs/plans/bayesfilter-ordinary-hmc-migration-debt-plan-review-2026-09-02.md`
* `docs/plans/bayesfilter-ordinary-hmc-migration-debt-amended-plan-codex-review-2026-09-03.md`
* `docs/plans/bayesfilter-ordinary-hmc-migration-debt-claude-audit-handoff-2026-09-02.md`
* `docs/plans/bayesfilter-ordinary-hmc-claude-audit-response-2026-09-03.md`

## Scope And Authorization

The owner authorized amendment, independent review, and execution of the
static/authority-boundary portion of the plan. The fresh Codex review was
read-only and returned `VERDICT: REVISE`; its source corrections were applied
before execution. The revised plan permits construction, serialization, AST
classification, documentation, and fail-closed route-guard work only.

No HMC transition, tuning run, benchmark, GPU initialization, numerical policy
selection, XLA-default change, NumPy migration, or promotion/default-readiness
claim was authorized or performed. A passing construction or serialization
check is an engineering prerequisite and is not sampler or scientific
evidence.

## Source Corrections Applied

The source trace established that the ordinary public default is
`ordinary_hmc` with algorithm ID
`operational_paired_fixed_trajectory_selection_v3`. Its operational path uses
windowed mass warm-up, a bounded floor/anchor/double trajectory screen, one
shared/frozen epsilon across three replications, and exact-L epsilon retuning.
The explicit `joint_l_epsilon_grid_fixed_mass_hmc` identifier is an alternate
legacy/non-promoting branch; direct stage tests do not establish it as the
public default.

The route contract previously accepted supported legacy IDs while callers could
ignore the non-promoting decision. The public ordinary facade now requires an
artifact-authoritative top-level route before it inspects adapter geometry or
runtime state. Legacy routes remain available to private diagnostic fixtures,
but an authority request raises `NonAuthoritativeHMCAlgorithmRoute`.

Route decisions now carry independent `operational_authority`,
`artifact_authority`, and `scientific_promotion_authority` fields. Ordinary
results and loop/artifact payloads expose a resolved-policy record. The current
ordinary runtime record explicitly remains non-claim-bearing because the
ordinary tuning family still has a known NumPy-policy violation; this is a
policy status, not a numerical result.

## Implemented Work

* Added the artifact-authority route guard and typed failure in
  `bayesfilter/hmc_route_contract.py`.
* Applied the guard at the public ordinary boundary in
  `bayesfilter/inference/hmc_kernel_tuning.py`; private diagnostic loop entry
  remains separate.
* Added resolved-policy fields to ordinary loop, result, artifact, and Phase 7
  summary payloads.
* Updated the capability registry text to describe the actual operational
  shared-epsilon/fixed-trajectory policy and to label the explicit joint grid
  as legacy diagnostic-only.
* Updated the reference guide, chapter, generated route tables, and semantic
  tests with the executable ordinary variant, policy ID, authority roles,
  construction-only inspection example, and XLA/NumPy qualification boundary.
* Corrected the ordinary module header and Phase 5 docstring so they describe
  the operational default and explicitly label the legacy joint grid
  diagnostic/non-promoting; added a regression assertion for that boundary.
* Added `scripts/audit_ordinary_hmc_migration_surface.py`, a standard-library
  bounded AST/import/policy scanner. It records branch reachability, consumer
  roles, dynamic indirection, NumPy call-chain findings, stale-policy findings,
  source revision, dirty-worktree hash, package versions, exact roots, and
  exclusions.
* Added focused scanner tests and route/dispatch/documentation regression tests.
* Added a narrow `.gitignore` rule for the reproducible local scanner output
  directory; authored plans and review notes remain tracked candidates.

## Static Audit Artifact

The scanner wrote its report under the ignored, versioned directory:

`docs/plans/artifacts/ordinary-hmc-migration-debt-2026-09-03/`

The directory contains the combined inventory, branch reachability,
consumer-role, NumPy call-chain, provenance, and Markdown trace files. The
normal scan completed with these counts:

| Finding | Count | Interpretation |
|---|---:|---|
| Relevant downstream consumer rows | 202 | Inventory input; not an admission result |
| Constant dynamic-import rows | 53 | Resolved by the bounded scanner |
| `unknown_dynamic_import` rows | 32 | Computed import/plugin paths requiring manual classification |
| `unresolved_dynamic_attribute` rows | 19 | Computed attribute paths requiring manual classification |
| BayesFilter NumPy runtime-candidate modules | 7 | Migration findings; claim authority remains disabled |
| Unqualified non-XLA policy findings | 0 | The active scan did not find an unqualified stale phrase |

The scanner's `--check` mode is intentionally fail-closed while the 32 plus 19
indirection rows remain unresolved. The nonzero result is an expected migration
blocker, not a test failure or evidence about HMC quality.

## Verification Commands And Results

All commands below used HEAD
`54201f5cd925ed15036bad8156606b812d53b045` plus the final uncommitted
worktree overlays; the worktree was left intact and its status hash is recorded
in the scanner provenance artifact.

* `pytest -q tests/test_hmc_route_contract.py` -> `17 passed`.
* `pytest -q tests/test_hmc_tuning_dispatch.py -k 'legacy_dispatch or tensor' --maxfail=1`
  -> `3 passed, 13 deselected`.
* `pytest -q tests/test_hmc_route_contract.py tests/test_hmc_tuning_dispatch.py tests/test_hmc_tuning_documentation_contract.py --maxfail=1`
  -> `47 passed` before adding the scanner tests to the combined command.
* `pytest -q tests/test_ordinary_hmc_migration_audit.py tests/test_hmc_tuning_documentation_contract.py --maxfail=1`
  -> `18 passed`.
* `pytest -q tests/test_hmc_route_contract.py tests/test_hmc_tuning_dispatch.py tests/test_hmc_tuning_documentation_contract.py tests/test_ordinary_hmc_migration_audit.py --maxfail=1`
  -> `51 passed`.
* `python scripts/inventory_hmc_tuning_routes.py --check` -> exit 0.
* `python scripts/render_hmc_tuning_interface_docs.py --check` -> exit 0.
* `python -m compileall -q bayesfilter/inference scripts/audit_ordinary_hmc_migration_surface.py scripts/inventory_hmc_tuning_routes.py scripts/render_hmc_tuning_interface_docs.py`
  -> exit 0.
* `git diff --check` and the plan whitespace check -> exit 0.
* The normal bounded downstream audit command -> exit 0 and wrote the files
  listed above.
* The same audit with `--check` -> nonzero by design because unresolved
  dynamic rows remain.

The reader-facing build also passed:

* `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/bayesfilter-hmc-docs-Ik2Rbt main.tex`
  -> exit 0; the temporary PDF contains 531 pages and includes Chapter 42,
  the repaired HMC tuning chapter. `pdftotext` inspection of the chapter pages
  confirmed the ordinary policy ID, shared/frozen-epsilon description, legacy
  rejection, and authority-role separation.

The build reports existing undefined cross-references/citations and numerous
overfull/underfull boxes in the wider monograph. They do not prevent this
chapter from compiling, but they remain reader-facing documentation debt and
were not silently treated as a clean-render quality verdict.

The focused pytest commands were inspected as construction/documentation
checks for this phase. They were not used as convergence, posterior, runtime,
or promotion evidence.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Route boundary repair | Legacy public authority request fails before adapter use | Passed focused guard tests | Other import paths may bypass the facade | Classify all downstream imports and aliases | No claim that every consumer is migrated |
| Resolved policy exposure | Payloads include variant, route, and separate authority roles | Passed construction/documentation checks | Replay behavior for all historical schemas remains to implement | Add schema/replay invalidation tests after Phase 3 completion | No sampler validity or posterior admission |
| Downstream inventory | Bounded AST report preserves every relevant row and unresolved indirection | Claim-adjacent unresolved rows are a continuation blocker | Manual role and dynamic target resolution remain open | Inspect the 32 computed imports and 19 attributes one path at a time | No inference from filename or artifact age |
| Ordinary runtime policy | NumPy/XLA mismatches are explicit in the payload and guide | Claim-bearing role remains vetoed | Owner has not selected the numerical/XLA repair | Write a separate reviewed policy/evidence plan | No approval of `use_xla=False` or a joint grid |

## Remaining Blockers And Required Follow-up

1. The seven ordinary tuning-family NumPy runtime candidates require a
   separate bounded migration plan or an explicitly reviewed exception. Do not
   issue claim-bearing authority while this status is unresolved.
2. Manually classify the 32 `unknown_dynamic_import` and 19
   `unresolved_dynamic_attribute` rows, including the MacroFinance and
   `dsge_hmc` claim-adjacent paths. A clean inventory is not the same as a
   clean consumer surface.
3. Complete Phase 3 replay/schema invalidation tests, including rejection of
   old payloads for authority use and explicit diagnostic-only loading.
4. Decide whether the ordinary public API remains typed under one name or is
   split into names. This is an owner decision, not an implication of the
   static trace.
5. Decide the ordinary numerical policy, primary selection criterion, and
   campaign budget. Then create a separate evidence-contract plan with the
   applicable naive/best-tuned-classical/plain-proposed/enhanced-proposed
   baseline ladder before any HMC run.
6. Revisit the broader monograph's unresolved references and layout warnings;
   the affected HMC chapter itself now compiles and has been text-inspected,
   but source inclusion and renderer freshness alone do not certify complete
   reader-facing quality.

## Post-Run Red Team

The strongest alternative explanation is that a downstream caller reaches a
different public-looking compatibility delegate or dynamic plugin target than
the bounded source set captured. The result would be overturned for the static
boundary if any such path can issue an authority-bearing artifact without the
new route/policy fields. The weakest evidence is the heuristic role
classification of external consumers and the uninspected historical replay
loaders; both are explicitly left open rather than promoted.

## Status

The static/authority-boundary portion of the amended plan is executed and
verified. Phase 4 consumer migration, Phase 7 NumPy cleanup, and Phase 8
numerical validation remain pending and blocked by the conditions above. No
scientific, convergence, HMC-readiness, superiority, or default-promotion
claim follows from this note.
