# Document Derivation Tree Audit

Target: `/home/chakwong/BayesFilter/docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex`
Search mode: `agent_guided`
Grounding policy: `strict`
Publication mode: `disabled`
Execution: `serial` with `1` worker(s)

## Executive Summary

- Selected source rows: `2`
- Semantic packets: `2`
- Proposition/context packets: `0`
- Context graphs: `2`
- Context graph statuses: `{'nearby_stated': 2, 'inferred_candidate': 11, 'missing': 2}`
- Typed repair obligations: `2`
- Typed repair obligation statuses: `{'typed_review': 1, 'blocked_on_missing_typed_assumptions': 1}`
- Ranked branches: `3`
- Effective promoted branches: `0`
- Raw promoted branches (diagnostic only): `0`
- Document-ready repair proposals: `0`
- Document gap reports: `2`
- Document partial-evidence reports: `0`
- Failure classifications: `{'branch_execution_pending': 2, 'formalization_blocked': 2, 'mathematical_blocked': 1}`
- Tool-grounded compiler statuses: `{'compiled': 2}`
- Tool-grounded compiler validation errors: `0`
- Parallel execution failures: `0`
- Blockers: `31`
- Missing focus labels: `['eq:robust-second-moment', 'eq:robust-mixture-score']`
- This report is generic and document-local; it is not tied to a card-NPV-specific plan.

## Tools Used

| Tool | Purpose | Status | Contract | Arguments |
| --- | --- | --- | --- | --- |
| `locate_equations_in_file` | Localize source rows in the exact target file. | `completed` | `equation_rows` | `{"root": "/home/chakwong/BayesFilter/docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01", "tex_path": "/home/chakwong/BayesFilter/docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex"}` |
| `extract_derivation_targets_for_label` | Bind each selected equation label to one complete validated source-owned obligation. | `quarantined` | `derivation_target_extraction_result` | `{"failure_count": 2, "fallback_to_locator_row": false, "requested_equation_labels": ["eq:robust-mode-derivatives", "eq:robust-change-measure", "eq:robust-second-moment", "eq:robust-mixture-score"], "selected_target_count": 2}` |
| `build_proposition_context_packet` | Localize proposition labels that are not display-equation rows and attach equation targets/context. | `not_needed` | `proposition_context_packet_result` | `{"context_target_count": 0, "focus_labels": ["eq:robust-mode-derivatives", "eq:robust-change-measure", "eq:robust-second-moment", "eq:robust-mixture-score"]}` |
| `build_semantic_work_packet` | Classify each target and generate full-display semantic packets, missing obligations, assumption sets, and derivation routes. | `completed` | `semantic_work_packet` | `{"selected_rows": 2}` |
| `assumptions_required` | Detect route-required assumptions before backend proof attempts. | `completed` | `assumption_discovery_result` | `{"selected_rows": 2}` |
| `build_local_context_graph` | Classify local source evidence as stated, nearby stated, inferred, missing, or unresolved before proposing repairs. | `completed` | `local_context_graph` | `{"context_graph_count": 2, "status_counts": {"inferred_candidate": 11, "missing": 2, "nearby_stated": 2}}` |
| `typed_repair_obligation_from_packet` | Convert context graph and semantic packet evidence into typed repair obligations before branch/report generation. | `completed` | `typed_repair_obligation` | `{"status_counts": {"blocked_on_missing_typed_assumptions": 1, "typed_review": 1}, "typed_repair_obligation_count": 2}` |
| `doctor_report` | Record external backend capability provenance. | `available` | `doctor_report` | `{"backend_env": "mathdevmcp-backends"}` |
| `can_derive_with_budget` | Run the external-tool-first branch controller on semantic packet targets. | `completed` | `derivation_search_tree_result` | `{"budget_profile": "standard", "execution_mode": "serial", "max_attempts": 1, "selected_rows": 2, "workers_used": 1}` |
| `rank_repair_branches` | Rank assumption branches by recorded backend evidence, blocker specificity, source support, closure strength, and non-minimality. | `completed` | `repair_branch_ranking_result` | `{"ranked_branch_count": 3}` |
| `tool_grounded_proposal_compiler` | Quarantine repair publication, compile legacy closure as partial evidence, and keep blocked branches as exact gap reports. | `completed` | `tool_grounded_proposal_compiler_result` | `{"gap_report_count": 2, "grounding_policy": "strict", "partial_evidence_count": 0, "repair_proposal_count": 0, "search_mode": "agent_guided", "validation_error_count": 0}` |
| `render_derivation_tree_report` | Render each derivation tree into structured evidence sections. | `completed` | `derivation_tree_report_result` | `{"rendered_trees": 2}` |

## Target Packets And Trees

### 1. `eq:robust-mode-derivatives`

- Location: `attempt05_observation_aware_tt_algorithm_note.tex > Stable coordinates and a physical defense against guide collapse > An observation-adapted positive quadrature repair > eq:robust-mode-derivatives > line 5269`
- Claim type: `theorem_proposition`
- Tree status: `budget_exhausted`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex', 'line_start': 5269, 'line_end': 5270, 'label': 'eq:robust-mode-derivatives', 'section_path': ['Stable coordinates and a physical defense against guide collapse', 'An observation-adapted positive quadrature repair'], 'start_byte': 257949, 'end_byte': 258048, 'source_digest': '1b8cba0c5ac6f37322e78a0115e304ef2275531c50ce1ff2927a5de3f3284be2', 'obligation_id': 'obl_ec17654bf7e3eccada2e5d4db241e71c3dab31b9f0ccad09e0fb33a44b4b10ad', 'obligation_digest': 'ec17654bf7e3eccada2e5d4db241e71c3dab31b9f0ccad09e0fb33a44b4b10ad', 'labels': ['eq:robust-mode-derivatives'], 'environment': 'align'}`
- Operators: `['equality']`
- Symbols: `{'latex_commands': ['\\odot', '\\succ'], 'bare_identifiers': ['H', 'S', 'c', 'diag', 'e', 'x']}`
- Context graph statuses: `{'nearby_stated': 1, 'inferred_candidate': 5}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca`
- Typed obligation status: `typed_review`
- Typed unresolved constructs: `['matrix_inverse', 'hamiltonian']`

Source row target:

```tex
H(x)&=S^{-1}+\tfrac12\operatorname{diag}(c\odot e^{-x})\succ0 .
 \label{eq:robust-mode-derivatives}
```

Full display target:

```tex
H(x)&=S^{-1}+\tfrac12\operatorname{diag}(c\odot e^{-x})\succ0 .
 \label{eq:robust-mode-derivatives}
```

Mathematically missing obligations:
- `formalized_local_obligation` (formalization_condition): A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Closes: Creates the next deterministic target for assumption discovery or proof audit.

Local context graph:
- `assumption_relevant_functions_differentiable` status `nearby_stated`
  Role: supports local derivative notation in the FOC route
  What: The relevant functions are differentiable.
  Why status: The condition is stated in the local paragraph/proposition context.
  Required next evidence: Use this as differentiability evidence, but do not treat it as integrability or derivative-expectation interchange evidence.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:5262-5286']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca`
  Diagnostic status: `typed_review`
  Encodability: `{'status': 'candidate', 'candidate_backends': ['sage', 'lean', 'human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
  Unresolved constructs: `['matrix_inverse', 'hamiltonian']`
  Route hints: `[{'backend': 'sage', 'suitability': 'diagnostic_candidate', 'reason': 'Matrix-oriented notation may benefit from optional Sage/numeric diagnostics when safely encoded.'}, {'backend': 'lean', 'suitability': 'formalization_candidate', 'reason': 'Typed notation may be formalized manually and checked by Lean.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}]`
  Assumption statuses:
  - `assumption_relevant_functions_differentiable` status `nearby_stated`: The relevant functions are differentiable.
  Boundary: Typed repair obligations are diagnostic routing artifacts; they are not proof certificates or backend encodings.

Tool-grounded proposal compiler:
- Contract: `tool_grounded_proposal_compiler_result`
- Status: `compiled`
- Publication mode: `disabled`
- Grounding policy: `strict`
- Repair proposal count: `0`
- Gap report count: `1`
- Partial-evidence count: `0`
- Boundary: Document repair publication is disabled. Legacy backend closure is partial evidence only; blocked paths are gap reports and all candidate edit text remains non-applicable.
- Compiled items:
  - `document_gap_report_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca', 'branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `attempt05_observation_aware_tt_algorithm_note.tex > Stable coordinates and a physical defense against guide collapse > An observation-adapted positive quadrature repair > eq:robust-mode-derivatives > line 5269`
  - Context branch: `branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions [] block constructs ['matrix_inverse', 'hamiltonian'].
  - Why this is a derivation problem: No missing typed assumptions were detected by the bounded typed IR builder. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. The source span contains macros `['\\odot', '\\succ']` whose mathematical types and backend names are not fixed.
  - Candidate assumption set that remains blocked:
    - Define every symbol, domain, and operator in the cited source line.
    - Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.
    - Rerun the relevant assumption/proof audit after the typed obligation exists.
  - Candidate derivation route that remains blocked:
    - Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit.
  - Exact blockers before this can become a repair proposal:
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: lean stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `branch_bound_backend_execution_required`: This assumption branch has no branch-bound backend request/result evidence. Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
  - Why no proposed edit is emitted: The serialization-only context branch is reported as a gap because backend formalization or translation remains incomplete; the exact branch-bound backend action has not been executed.
  - Backend evidence status: `typed_translation_blocked`
  - Validation: `typed_translation_blocked` from `strict_proposal_gate`
  - Non-claims: `['This is a gap report, not a repair proposal.', 'No proposed edit should be applied until the remaining blockers are closed by source evidence or a certifying backend.', 'The context branch is a serialization aid, not a scientific winner, global optimum, or minimal route.']`

Possible sufficient assumption sets:
- `typed_obligation_first`: Makes the abstention inspectable by deterministic tooling.
  - Define every symbol, domain, and operator in the cited source line.
  - Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.
  - Rerun the relevant assumption/proof audit after the typed obligation exists.

Branch ranking:
- Contract: `repair_branch_ranking_result`
- Nondominated branches: `['branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_formalization_required', 'target_ids': ['branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first'], 'ledger_entry_ids': ['ledger_0a77d5c2659a17d257b750952d9dca54cda53d3897955bd01cc168f8895eea54'], 'prerequisites': ['scope_bound:ledger_0a77d5c2659a17d257b750952d9dca54cda53d3897955bd01cc168f8895eea54'], 'launch_vetoes': ['ledger_1af945bb3e84f52dfa7cee3ba5bd9cb872efc036f3743f6a72de93b08158705a', 'ledger_205a8405d7469460f1538f59190a383404838668caec79ad30f04b3ccdd3a94c'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_5e325951833073d4834578c4306a1d0dae8e994638ecd288a5d327b1cb1bf061'}`
- Serialization position `1`: `branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_0a77d5c2659a17d257b750952d9dca54cda53d3897955bd01cc168f8895eea54', 'ledger_1af945bb3e84f52dfa7cee3ba5bd9cb872efc036f3743f6a72de93b08158705a', 'ledger_205a8405d7469460f1538f59190a383404838668caec79ad30f04b3ccdd3a94c'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca']`
  - Typed unresolved constructs: `['matrix_inverse', 'hamiltonian']`
  - Typed encodability: `{'status': 'candidate', 'candidate_backends': ['sage', 'lean', 'human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['Define every symbol, domain, and operator in the cited source line.', 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'Rerun the relevant assumption/proof audit after the typed obligation exists.']
  - Route under assumptions:
    - Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Convert the cited line into a typed obligation before proposing a document edit.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Build a typed symbol map from every LaTeX macro in the target to backend variables or definitions.
    - `blocker` status `blocking`: LaTeX macros must be translated into backend symbols before execution.
    - `blocker` status `blocking`: sympy stub is not yet a certifying formalization.
  - External-tool ledger: `['sympy:available', 'sage:available', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\odot', '\\succ']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Formalize local obligation`: Convert the cited line into a typed obligation before proposing a document edit.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:robust-mode-derivatives` at attempt05_observation_aware_tt_algorithm_note.tex > Stable coordinates and a physical defense against guide collapse > An observation-adapted positive quadrature repair > eq:robust-mode-derivatives > line 5269, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification.

Remaining blockers:
- `blocker_lean_source_required` (formalization_required)
  Problem: Lean certification was selected but no Lean source was supplied.
  Why: Direct Lean checking requires an explicit Lean statement/proof artifact.
  Required next evidence: Supply Lean source or a formalization branch before Lean certification.
- `blocker_sympy_algebra_attempt` (adapter_diagnostic)
  Problem: sympy did not certify or refute the target.
  Why: Expression contains syntax outside the conservative router grammar.
  Required next evidence: Provide a certifying backend result, concrete counterexample, formalization, or stronger assumption set.
- `blocker_budget_exhausted` (budget_exhausted)
  Problem: The controller exhausted its attempt budget before proof or refutation.
  Why: Some scheduled evidence actions were not attempted within the selected budget profile.
  Required next evidence: Increase budget, provide a stronger formalization, or inspect exhausted actions.
- `blocker_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\odot', '\\succ']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_robust_mode_derivatives_ec17654bf7e3ecca_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 2. `eq:robust-change-measure`

- Location: `attempt05_observation_aware_tt_algorithm_note.tex > Stable coordinates and a physical defense against guide collapse > An observation-adapted positive quadrature repair > eq:robust-change-measure > line 5302`
- Claim type: `theorem_proposition`
- Tree status: `budget_exhausted`
- Promotion guard: `can_promote=False`
- Semantic domains: `['conditional_expectation']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex', 'line_start': 5302, 'line_end': 5305, 'label': 'eq:robust-change-measure', 'section_path': ['Stable coordinates and a physical defense against guide collapse', 'An observation-adapted positive quadrature repair'], 'start_byte': 259621, 'end_byte': 259764, 'source_digest': '1b8cba0c5ac6f37322e78a0115e304ef2275531c50ce1ff2927a5de3f3284be2', 'obligation_id': 'obl_4260424ec6a1a83b679bd5d7da8038d43998f271b96e693311026c34bfee7ea9', 'obligation_digest': '4260424ec6a1a83b679bd5d7da8038d43998f271b96e693311026c34bfee7ea9', 'labels': ['eq:robust-change-measure'], 'environment': 'equation'}`
- Operators: `['equality', 'conditional_bar', 'integral']`
- Symbols: `{'latex_commands': ['\\rho'], 'bare_identifiers': ['Bu', 'L', 'du', 'dx', 'g', 'h', 'p', 'q', 'u', 'x', 'y']}`
- Context graph statuses: `{'inferred_candidate': 6, 'missing': 2, 'nearby_stated': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['conditional', 'conditional_law', 'route_assumption_denominator_is_nonzero']`

Source row target:

```tex
\int h(x)p^-(x)g(y\mid x)\,dx
 =\int h(x_*+Bu)
   \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du .
 \label{eq:robust-change-measure}
```

Full display target:

```tex
\int h(x)p^-(x)g(y\mid x)\,dx
 =\int h(x_*+Bu)
   \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du .
 \label{eq:robust-change-measure}
```

Mathematically missing obligations:
- `conditional_law_defined` (probability_condition): A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Closes: Makes the expectation operator well defined.
- `measurable_integrable_payoff_terms` (integrability_condition): Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Closes: Turns the displayed expression into a finite scalar equality.
- `conditioning_information_defined` (information_condition): A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Closes: Fixes the scope of the conditional expectation used in the derivation.

Local context graph:
- `requirement_conditional_law_defined` status `missing`
  Role: well-definedness condition for conditional expectation
  What: A conditional law for the expectation is defined.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: Cite or add the transition kernel/probability law used by the conditional expectation.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:5302-5305']`
- `requirement_conditional_integrability` status `nearby_stated`
  Role: finite-scalar condition for expectation-valued equations
  What: Random terms inside the conditional expectation are measurable and integrable.
  Why status: The condition is stated in nearby local context, not in the target span.
  Required next evidence: Cite or add measurability and finite conditional first-moment/dominated-envelope conditions.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:5302-5305']`
- `route_assumption_denominator_is_nonzero` status `missing`
  Role: route-required assumption from assumption_discovery
  What: denominator is nonzero
  Why status: The low-level route detector marked this assumption as missing.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:5302-5305']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined', 'route_assumption_denominator_is_nonzero'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['conditional', 'conditional_law', 'route_assumption_denominator_is_nonzero']`
  Route hints: `[{'backend': 'lean', 'suitability': 'formalization_candidate', 'reason': 'Typed notation may be formalized manually and checked by Lean.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
  - `route_assumption_denominator_is_nonzero` status `missing`: denominator is nonzero
  - `requirement_conditional_integrability` status `nearby_stated`: Random terms inside the conditional expectation are measurable and integrable.
  Boundary: Typed repair obligations are diagnostic routing artifacts; they are not proof certificates or backend encodings.

Tool-grounded proposal compiler:
- Contract: `tool_grounded_proposal_compiler_result`
- Status: `compiled`
- Publication mode: `disabled`
- Grounding policy: `strict`
- Repair proposal count: `0`
- Gap report count: `1`
- Partial-evidence count: `0`
- Boundary: Document repair publication is disabled. Legacy backend closure is partial evidence only; blocked paths are gap reports and all candidate edit text remains non-applicable.
- Compiled items:
  - `document_gap_report_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_robust_change_measure_4260424ec6a1a83b', 'branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation', 'actionable_abstention:conditional_expectation', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_sympy', 'blocker_formalization_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_sage', 'blocker_formalization_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_lean', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `attempt05_observation_aware_tt_algorithm_note.tex > Stable coordinates and a physical defense against guide collapse > An observation-adapted positive quadrature repair > eq:robust-change-measure > line 5302`
  - Context branch: `branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation`
  - Context selection authority: `serialization_only_nondominated_context`
  - Nondominated branches: `['branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation', 'branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['requirement_conditional_law_defined', 'route_assumption_denominator_is_nonzero'] block constructs ['conditional', 'conditional_law', 'route_assumption_denominator_is_nonzero'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed. Typed encodability is blocked by `['requirement_conditional_law_defined', 'route_assumption_denominator_is_nonzero']`.
  - Missing or unresolved assumptions:
    - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
    - `route_assumption_denominator_is_nonzero` status `missing`: denominator is nonzero
  - Candidate assumption set that remains blocked:
    - The conditioned shock or path has finite support.
    - Every payoff/value term inside the expectation is finite at each support point.
    - The conditioning state or information set is explicitly defined.
    - The conditioning object `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du` is defined as a sigma-field, information set, state, or conditioning variable for this equality.
  - Candidate derivation route that remains blocked:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Exact blockers before this can become a repair proposal:
    - `conditioning_scope_translation_required`: The conditional bar has no backend-level conditioning object yet. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `conditional_law_translation_required`: The conditional law required by the expectation is not stated as an encodable object. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `missing_domain_or_assumption_required`: The branch still has missing or unresolved typed assumptions. Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: lean stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `branch_bound_backend_execution_required`: This assumption branch has no branch-bound backend request/result evidence. Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
  - Source refs for missing/unresolved evidence: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex > eq:robust-change-measure > line 5302-5305']`
  - Why no proposed edit is emitted: The serialization-only context branch is reported as a gap because typed mathematical assumptions or domain conditions remain unresolved; backend formalization or translation remains incomplete; the exact branch-bound backend action has not been executed.
  - Backend evidence status: `typed_translation_blocked`
  - Validation: `typed_translation_blocked` from `strict_proposal_gate`
  - Non-claims: `['This is a gap report, not a repair proposal.', 'No proposed edit should be applied until the remaining blockers are closed by source evidence or a certifying backend.', 'The context branch is a serialization aid, not a scientific winner, global optimum, or minimal route.']`

Possible sufficient assumption sets:
- `finite_state_conditional_expectation`: The expectation becomes a finite weighted sum.
  - The conditioned shock or path has finite support.
  - Every payoff/value term inside the expectation is finite at each support point.
  - The conditioning state or information set is explicitly defined.
- `kernel_integrability_condition`: The expectation is a well-defined finite conditional integral.
  - A conditional kernel or probability law is fixed for the random object.
  - All random terms inside the expectation are measurable under that law.
  - Those terms are dominated by an integrable envelope or have finite conditional first moments.

Branch ranking:
- Contract: `repair_branch_ranking_result`
- Nondominated branches: `['branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation', 'branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition']`
- Unique top branch, only if one exists: `None`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'blocked_for_human_or_formalization_choice', 'target_ids': ['branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation', 'branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition'], 'branch_ids': ['branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation', 'branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition'], 'ledger_entry_ids': ['unresolved_branch_choice'], 'prerequisites': ['resolve_nondominated_branch_choice'], 'launch_vetoes': ['ledger_3f54b5129b294b17a196d3ab22d3f9608b481cb74c730b5598d0feaffcf562bc', 'ledger_50bc88e2e05a26d2688ae47dd7c60f26d80a85b3113295722ecd6e86840505fe', 'ledger_72688d4e007a213a564b3842bdf4f649713680904b7a61f50ae852e85fcc46c3', 'ledger_7a99c72d96a873b707091eec7ecc8eccf56f33639396a662ede44193d6ffaf82', 'ledger_978b8b77506e27038b363bd9d1388f80024699a0fb4b2a92e09cefe4f1e9714a', 'ledger_c46092d4418aa03c293c2ba1a4678dec550ee72fa9341678444ba29250d4964d', 'ledger_c54fc872f904fa224161cb22ac4a5c439d6219121b04f00877e0aa3f377aee2b', 'ledger_c70bf253090c6cb5eede2ca93ee82a68b8edefee7f00ad4e614bab16d43af74c', 'ledger_cac0cc0f2ffd2264cdf38c668fa38c4ae64b176073968c260b8ca2bfe32f300a', 'ledger_cfd6fe7e2199f5c332f6e541c6ee5ea2a203ef5562e8bfa0e2a315770901801c', 'ledger_de007efae25f7b0086b20ddfb13f7fc751bec6443eee861744f36b4b195dba82', 'ledger_fcbc201f0946a0681fd233fce5a1f45eb9b0a3f8e694f4537cf991298ab49b88'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_choice_record', 'schema_version': 'p06_formalization_choice@1', 'binding_fields': ['branch_ids', 'target_id'], 'path_role': 'decision_blocker'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_b04e9789001ce1dcb8ac0a5ce8b30b7b7bfb1942267b31ec359352d2c52741d1'}`
- Serialization position `1`: `branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_50bc88e2e05a26d2688ae47dd7c60f26d80a85b3113295722ecd6e86840505fe', 'ledger_978b8b77506e27038b363bd9d1388f80024699a0fb4b2a92e09cefe4f1e9714a', 'ledger_c46092d4418aa03c293c2ba1a4678dec550ee72fa9341678444ba29250d4964d', 'ledger_c70bf253090c6cb5eede2ca93ee82a68b8edefee7f00ad4e614bab16d43af74c', 'ledger_cac0cc0f2ffd2264cdf38c668fa38c4ae64b176073968c260b8ca2bfe32f300a', 'ledger_cfd6fe7e2199f5c332f6e541c6ee5ea2a203ef5562e8bfa0e2a315770901801c'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'The conditioned shock or path has finite support.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Every payoff/value term inside the expectation is finite at each support point.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'The conditioning state or information set is explicitly defined.', 'status': 'candidate'}, {'id': 'legacy_assumption_4', 'statement': 'The conditioning object `x)\\,dx = \\int h(x_*+Bu) \\frac{p^-(x_*+Bu)g(y\\mid x_*+Bu)}{q_L(x_*+Bu)}\\rho(u)\\,du` is defined as a sigma-field, information set, state, or conditioning variable for this equality.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.
- Serialization position `2`: `branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_3f54b5129b294b17a196d3ab22d3f9608b481cb74c730b5598d0feaffcf562bc', 'ledger_72688d4e007a213a564b3842bdf4f649713680904b7a61f50ae852e85fcc46c3', 'ledger_7a99c72d96a873b707091eec7ecc8eccf56f33639396a662ede44193d6ffaf82', 'ledger_c54fc872f904fa224161cb22ac4a5c439d6219121b04f00877e0aa3f377aee2b', 'ledger_de007efae25f7b0086b20ddfb13f7fc751bec6443eee861744f36b4b195dba82', 'ledger_fcbc201f0946a0681fd233fce5a1f45eb9b0a3f8e694f4537cf991298ab49b88'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'A conditional kernel or probability law is fixed for the random object.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'All random terms inside the expectation are measurable under that law.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Those terms are dominated by an integrable envelope or have finite conditional first moments.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b']`
  - Typed unresolved constructs: `['conditional', 'conditional_law', 'route_assumption_denominator_is_nonzero']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined', 'route_assumption_denominator_is_nonzero'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['The conditioned shock or path has finite support.', 'Every payoff/value term inside the expectation is finite at each support point.', 'The conditioning state or information set is explicitly defined.', 'The conditioning object `x)\\,dx = \\int h(x_*+Bu) \\frac{p^-(x_*+Bu)g(y\\mid x_*+Bu)}{q_L(x_*+Bu)}\\rho(u)\\,du` is defined as a sigma-field, information set, state, or conditioning variable for this equality.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 4 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_law_defined', 'route_assumption_denominator_is_nonzero']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\rho']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`
- `branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b']`
  - Typed unresolved constructs: `['conditional', 'conditional_law', 'route_assumption_denominator_is_nonzero']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined', 'route_assumption_denominator_is_nonzero'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['A conditional kernel or probability law is fixed for the random object.', 'All random terms inside the expectation are measurable under that law.', 'Those terms are dominated by an integrable envelope or have finite conditional first moments.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_law_defined', 'route_assumption_denominator_is_nonzero']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\rho']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Define conditional law`: Specify the kernel or conditional distribution used by the expectation.
- `Check integrability`: Verify each random payoff, value, or derivative term has a finite conditional expectation.
- `Use expectation as scalar`: Only after those checks should the equality be treated as a scalar derivation step.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `missing_assumptions`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:robust-change-measure` at attempt05_observation_aware_tt_algorithm_note.tex > Stable coordinates and a physical defense against guide collapse > An observation-adapted positive quadrature repair > eq:robust-change-measure > line 5302, add an assumptions paragraph: "For this displayed equality, assume: The conditioned shock or path has finite support. Every payoff/value term inside the expectation is finite at each support point. The conditioning state or information set is explicitly defined. The conditioning object `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du` is defined as a sigma-field, information set, state, or conditioning variable for this equality. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
- `patch_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:robust-change-measure` at attempt05_observation_aware_tt_algorithm_note.tex > Stable coordinates and a physical defense against guide collapse > An observation-adapted positive quadrature repair > eq:robust-change-measure > line 5302, add an assumptions paragraph: "For this displayed equality, assume: A conditional kernel or probability law is fixed for the random object. All random terms inside the expectation are measurable under that law. Those terms are dominated by an integrable envelope or have finite conditional first moments. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x)\,dx = \int h(x_*+Bu) \frac{p^-(x_*+Bu)g(y\mid x_*+Bu)}{q_L(x_*+Bu)}\rho(u)\,du`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.

Remaining blockers:
- `blocker_lean_source_required` (formalization_required)
  Problem: Lean certification was selected but no Lean source was supplied.
  Why: Direct Lean checking requires an explicit Lean statement/proof artifact.
  Required next evidence: Supply Lean source or a formalization branch before Lean certification.
- `blocker_sympy_algebra_attempt` (adapter_diagnostic)
  Problem: sympy did not certify or refute the target.
  Why: The target has missing route-required assumptions.
  Required next evidence: Provide a certifying backend result, concrete counterexample, formalization, or stronger assumption set.
- `blocker_budget_exhausted` (budget_exhausted)
  Problem: The controller exhausted its attempt budget before proof or refutation.
  Why: Some scheduled evidence actions were not attempted within the selected budget profile.
  Required next evidence: Increase budget, provide a stronger formalization, or inspect exhausted actions.
- `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_law_defined', 'route_assumption_denominator_is_nonzero']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\rho']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_finite_state_conditional_expectation_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_law_defined', 'route_assumption_denominator_is_nonzero']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\rho']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_kernel_integrability_condition_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_conditional_law_defined` (probability_condition)
  Problem: A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Required next evidence: Makes the expectation operator well defined.
- `blocker_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_measurable_integrable_payoff_terms` (integrability_condition)
  Problem: Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Required next evidence: Turns the displayed expression into a finite scalar equality.
- `blocker_semantic_packet_eq_robust_change_measure_4260424ec6a1a83b_conditioning_information_defined` (information_condition)
  Problem: A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Required next evidence: Fixes the scope of the conditional expectation used in the derivation.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

## Non-Claims

- `document_tree_audit_not_document_proof`: This workflow is a semantic gap and tree-evidence report; it does not prove the whole document.
- `semantic_packets_not_certificates`: Missing obligations, assumption sets, and derivation routes are deterministic guidance, not proof certificates.
- `proof_search_not_final_certificate`: LeanDojo, Pantograph, retrieval, route plans, and static extraction are diagnostic until direct Lean or another certifying backend checks the scoped target.
- `document_repair_publication_quarantined`: No returned candidate is an applicable document edit while publication mode is disabled.
