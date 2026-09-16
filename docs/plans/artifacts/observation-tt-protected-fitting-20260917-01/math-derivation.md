# Document Derivation Tree Audit

Target: `/home/chakwong/BayesFilter/docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex`
Search mode: `agent_guided`
Grounding policy: `strict`
Publication mode: `disabled`
Execution: `serial` with `1` worker(s)

## Executive Summary

- Selected source rows: `4`
- Semantic packets: `4`
- Proposition/context packets: `0`
- Context graphs: `4`
- Context graph statuses: `{'inferred_candidate': 14, 'unresolved': 1, 'nearby_stated': 4, 'missing': 3}`
- Typed repair obligations: `4`
- Typed repair obligation statuses: `{'blocked_on_missing_typed_assumptions': 3, 'needs_assumptions': 1}`
- Ranked branches: `5`
- Effective promoted branches: `0`
- Raw promoted branches (diagnostic only): `0`
- Document-ready repair proposals: `0`
- Document gap reports: `4`
- Document partial-evidence reports: `0`
- Failure classifications: `{'branch_execution_pending': 4, 'formalization_blocked': 4, 'mathematical_blocked': 3}`
- Tool-grounded compiler statuses: `{'compiled': 4}`
- Tool-grounded compiler validation errors: `0`
- Parallel execution failures: `0`
- Blockers: `54`
- Missing focus labels: `['eq:a11-blend', 'eq:a11-floor', 'eq:a11-amplitude', 'eq:a11-design', 'eq:a11-capacity', 'eq:a11-mixture', 'eq:a11-weights', 'eq:a11-ess', 'eq:a11-epsilon', 'eq:a11-mixture-score']`
- This report is generic and document-local; it is not tied to a card-NPV-specific plan.

## Tools Used

| Tool | Purpose | Status | Contract | Arguments |
| --- | --- | --- | --- | --- |
| `locate_equations_in_file` | Localize source rows in the exact target file. | `completed` | `equation_rows` | `{"root": "/home/chakwong/BayesFilter/docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01", "tex_path": "/home/chakwong/BayesFilter/docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex"}` |
| `extract_derivation_targets_for_label` | Bind each selected equation label to one complete validated source-owned obligation. | `quarantined` | `derivation_target_extraction_result` | `{"failure_count": 10, "fallback_to_locator_row": false, "requested_equation_labels": ["eq:a11-blend", "eq:a11-floor", "eq:a11-amplitude", "eq:a11-design", "eq:a11-capacity", "eq:a11-projection", "eq:a11-conditional", "eq:a11-mixture", "eq:a11-weights", "eq:a11-ess", "eq:a11-epsilon", "eq:a11-chart-derivative", "eq:a11-tt-score", "eq:a11-mixture-score"], "selected_target_count": 4}` |
| `build_proposition_context_packet` | Localize proposition labels that are not display-equation rows and attach equation targets/context. | `not_needed` | `proposition_context_packet_result` | `{"context_target_count": 0, "focus_labels": ["eq:a11-blend", "eq:a11-floor", "eq:a11-amplitude", "eq:a11-design", "eq:a11-capacity", "eq:a11-projection", "eq:a11-conditional", "eq:a11-mixture", "eq:a11-weights", "eq:a11-ess", "eq:a11-epsilon", "eq:a11-chart-derivative", "eq:a11-tt-score", "eq:a11-mixture-score"]}` |
| `build_semantic_work_packet` | Classify each target and generate full-display semantic packets, missing obligations, assumption sets, and derivation routes. | `completed` | `semantic_work_packet` | `{"selected_rows": 4}` |
| `assumptions_required` | Detect route-required assumptions before backend proof attempts. | `completed` | `assumption_discovery_result` | `{"selected_rows": 4}` |
| `build_local_context_graph` | Classify local source evidence as stated, nearby stated, inferred, missing, or unresolved before proposing repairs. | `completed` | `local_context_graph` | `{"context_graph_count": 4, "status_counts": {"inferred_candidate": 14, "missing": 3, "nearby_stated": 4, "unresolved": 1}}` |
| `typed_repair_obligation_from_packet` | Convert context graph and semantic packet evidence into typed repair obligations before branch/report generation. | `completed` | `typed_repair_obligation` | `{"status_counts": {"blocked_on_missing_typed_assumptions": 3, "needs_assumptions": 1}, "typed_repair_obligation_count": 4}` |
| `doctor_report` | Record external backend capability provenance. | `available` | `doctor_report` | `{"backend_env": "mathdevmcp-backends"}` |
| `can_derive_with_budget` | Run the external-tool-first branch controller on semantic packet targets. | `completed` | `derivation_search_tree_result` | `{"budget_profile": "balanced", "execution_mode": "serial", "max_attempts": 2, "selected_rows": 4, "workers_used": 1}` |
| `rank_repair_branches` | Rank assumption branches by recorded backend evidence, blocker specificity, source support, closure strength, and non-minimality. | `completed` | `repair_branch_ranking_result` | `{"ranked_branch_count": 5}` |
| `tool_grounded_proposal_compiler` | Quarantine repair publication, compile legacy closure as partial evidence, and keep blocked branches as exact gap reports. | `completed` | `tool_grounded_proposal_compiler_result` | `{"gap_report_count": 4, "grounding_policy": "strict", "partial_evidence_count": 0, "repair_proposal_count": 0, "search_mode": "agent_guided", "validation_error_count": 0}` |
| `render_derivation_tree_report` | Render each derivation tree into structured evidence sections. | `completed` | `derivation_tree_report_result` | `{"rendered_trees": 4}` |

## Target Packets And Trees

### 1. `eq:a11-projection`

- Location: `attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > What the pair-TT regression is approximating > eq:a11-projection > line 5870`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex', 'line_start': 5870, 'line_end': 5872, 'label': 'eq:a11-projection', 'section_path': ['Recovering fitting accuracy without removing the physical defense', 'What the pair-TT regression is approximating'], 'start_byte': 289486, 'end_byte': 289573, 'source_digest': '638a3ad7bfefc66e9e6f0ed94dc92a02cabeed7a917e62b1d1a7c65812e20dfb', 'obligation_id': 'obl_934d1325a86aa13f6b564782756091910b10f8e7c7107299dcef07559ebfd21d', 'obligation_digest': '934d1325a86aa13f6b564782756091910b10f8e7c7107299dcef07559ebfd21d', 'labels': ['eq:a11-projection'], 'environment': 'equation'}`
- Operators: `['equality']`
- Symbols: `{'latex_commands': ['\\Pi'], 'bare_identifiers': ['G', 'b', 'h', 'pb']}`
- Context graph statuses: `{'inferred_candidate': 2}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_a11_projection_934d1325a86aa13f`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['conditional']`

Source row target:

```tex
\|b_G-h_G\|_2^2=
 \|b_G-\Pi_pb_G\|_2^2+\|\Pi_pb_G-h_G\|_2^2,
 \label{eq:a11-projection}
```

Full display target:

```tex
\|b_G-h_G\|_2^2=
 \|b_G-\Pi_pb_G\|_2^2+\|\Pi_pb_G-h_G\|_2^2,
 \label{eq:a11-projection}
```

Mathematically missing obligations:
- `formalized_local_obligation` (formalization_condition): A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Closes: Creates the next deterministic target for assumption discovery or proof audit.

Local context graph:

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_a11_projection_934d1325a86aa13f`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': ['conditional'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['conditional']`
  Route hints: `[{'backend': 'lean', 'suitability': 'formalization_candidate', 'reason': 'Typed notation may be formalized manually and checked by Lean.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}]`
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
  - `document_gap_report_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_a11_projection_934d1325a86aa13f', 'branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_a11_projection_934d1325a86aa13f']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > What the pair-TT regression is approximating > eq:a11-projection > line 5870`
  - Context branch: `branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions [] block constructs ['conditional'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. The source span contains macros `['\\Pi']` whose mathematical types and backend names are not fixed.
  - Candidate assumption set that remains blocked:
    - Define every symbol, domain, and operator in the cited source line.
    - Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.
    - Rerun the relevant assumption/proof audit after the typed obligation exists.
  - Candidate derivation route that remains blocked:
    - Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit.
  - Exact blockers before this can become a repair proposal:
    - `conditioning_scope_translation_required`: The conditional bar has no backend-level conditioning object yet. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
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
- Nondominated branches: `['branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_formalization_required', 'target_ids': ['branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first'], 'ledger_entry_ids': ['ledger_46016bd5ef2add9f5846ed9f921ee2b88b5ef84c3d3fe0be42eb07176569317d'], 'prerequisites': ['scope_bound:ledger_46016bd5ef2add9f5846ed9f921ee2b88b5ef84c3d3fe0be42eb07176569317d'], 'launch_vetoes': ['ledger_7b6d7c62d38099277e41052ba73721991cfcfc69dd33fef8ce1d4fb96755f661', 'ledger_e175b7b27a42273d38fa8a63b9ea8e5f4cb88fbfed78c7ee44e641156f82fd06', 'ledger_f0d1e490a43f0fcfbab9edbf91da202143ff6aef0573ab0986e7c146a67be05f'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_ac86602453c5381bb74df6f77571a0cb9323f7f0fcfa67fe6149b0dcf44c135d'}`
- Serialization position `1`: `branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_46016bd5ef2add9f5846ed9f921ee2b88b5ef84c3d3fe0be42eb07176569317d', 'ledger_7b6d7c62d38099277e41052ba73721991cfcfc69dd33fef8ce1d4fb96755f661', 'ledger_e175b7b27a42273d38fa8a63b9ea8e5f4cb88fbfed78c7ee44e641156f82fd06', 'ledger_f0d1e490a43f0fcfbab9edbf91da202143ff6aef0573ab0986e7c146a67be05f'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_a11_projection_934d1325a86aa13f']`
  - Typed unresolved constructs: `['conditional']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': ['conditional'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
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
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Treat the conditioning object as the argument of a conditional transition kernel and bind the kernel to the expectation.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Build a typed symbol map from every LaTeX macro in the target to backend variables or definitions.
  - External-tool ledger: `['sympy:available', 'sage:available', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\Pi']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Formalize local obligation`: Convert the cited line into a typed obligation before proposing a document edit.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `unknown`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:a11-projection` at attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > What the pair-TT regression is approximating > eq:a11-projection > line 5870, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification.

Remaining blockers:
- `blocker_lean_source_required` (formalization_required)
  Problem: Lean certification was selected but no Lean source was supplied.
  Why: Direct Lean checking requires an explicit Lean statement/proof artifact.
  Required next evidence: Supply Lean source or a formalization branch before Lean certification.
- `blocker_sympy_algebra_attempt` (adapter_diagnostic)
  Problem: sympy did not certify or refute the target.
  Why: No bounded derivation or refutation was found.
  Required next evidence: Provide a certifying backend result, concrete counterexample, formalization, or stronger assumption set.
- `blocker_bounded_counterexample_attempt` (adapter_diagnostic)
  Problem: sympy_finite_domain did not certify or refute the target.
  Why: Expression is outside the conservative scalar grammar.
  Required next evidence: Provide a certifying backend result, concrete counterexample, formalization, or stronger assumption set.
- `blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\Pi']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_a11_projection_934d1325a86aa13f_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_a11_projection_934d1325a86aa13f_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 2. `eq:a11-conditional`

- Location: `attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > A normalized conditional and its physical correction > eq:a11-conditional > line 5901`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['conditional_expectation']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex', 'line_start': 5901, 'line_end': 5904, 'label': 'eq:a11-conditional', 'section_path': ['Recovering fitting accuracy without removing the physical defense', 'A normalized conditional and its physical correction'], 'start_byte': 290940, 'end_byte': 291060, 'source_digest': '638a3ad7bfefc66e9e6f0ed94dc92a02cabeed7a917e62b1d1a7c65812e20dfb', 'obligation_id': 'obl_2197dedc72a07b751d4dc74f4c3684e77ccdebef9d94b41a809171835a80fefa', 'obligation_digest': '2197dedc72a07b751d4dc74f4c3684e77ccdebef9d94b41a809171835a80fefa', 'labels': ['eq:a11-conditional'], 'environment': 'equation'}`
- Operators: `['equality', 'conditional_bar']`
- Symbols: `{'latex_commands': ['\\det', '\\tau', '\\varphi'], 'bare_identifiers': ['L', 'TT', 'a', 'd', 'h', 'q', 't', 'u', 'v', 'x', 'z']}`
- Context graph statuses: `{'inferred_candidate': 4, 'unresolved': 1, 'nearby_stated': 1, 'missing': 2}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_a11_conditional_2197dedc72a07b75`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['determinant', 'conditional', 'conditional_law', 'route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet']`

Source row target:

```tex
q_t^{TT}(x\mid z)=
 \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t}
 \frac{\varphi_d(u)}{|\det L_t|}
 \label{eq:a11-conditional}
```

Full display target:

```tex
q_t^{TT}(x\mid z)=
 \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t}
 \frac{\varphi_d(u)}{|\det L_t|}
 \label{eq:a11-conditional}
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
- `requirement_conditional_law_defined` status `unresolved`
  Role: well-definedness condition for conditional expectation
  What: A conditional law for the expectation is defined.
  Why status: The local source contains related notation or a proof step, but not the required condition itself.
  Required next evidence: Cite or add the transition kernel/probability law used by the conditional expectation.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:5901-5904']`
- `requirement_conditional_integrability` status `nearby_stated`
  Role: finite-scalar condition for expectation-valued equations
  What: Random terms inside the conditional expectation are measurable and integrable.
  Why status: The condition is stated in nearby local context, not in the target span.
  Required next evidence: Cite or add measurability and finite conditional first-moment/dominated-envelope conditions.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:5901-5904']`
- `route_assumption_denominator_is_nonzero` status `missing`
  Role: route-required assumption from assumption_discovery
  What: denominator is nonzero
  Why status: The low-level route detector marked this assumption as missing.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:5901-5904']`
- `route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet` status `missing`
  Role: route-required assumption from assumption_discovery
  What: matrix operand is square with valid determinant domain, usually positive definite for logdet
  Why status: The low-level route detector marked this assumption as missing.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:5901-5904']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_a11_conditional_2197dedc72a07b75`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['sage', 'lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet', 'requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['determinant', 'conditional', 'conditional_law', 'route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet']`
  Route hints: `[{'backend': 'sage', 'suitability': 'diagnostic_candidate', 'reason': 'Matrix-oriented notation may benefit from optional Sage/numeric diagnostics when safely encoded.'}, {'backend': 'lean', 'suitability': 'formalization_candidate', 'reason': 'Typed notation may be formalized manually and checked by Lean.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `route_assumption_denominator_is_nonzero` status `missing`: denominator is nonzero
  - `route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet` status `missing`: matrix operand is square with valid determinant domain, usually positive definite for logdet
  - `requirement_conditional_law_defined` status `unresolved`: A conditional law for the expectation is defined.
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
  - `document_gap_report_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_a11_conditional_2197dedc72a07b75', 'branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation', 'actionable_abstention:conditional_expectation', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_a11_conditional_2197dedc72a07b75']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_sympy', 'blocker_formalization_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_sage', 'blocker_formalization_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_lean', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > A normalized conditional and its physical correction > eq:a11-conditional > line 5901`
  - Context branch: `branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation`
  - Context selection authority: `serialization_only_nondominated_context`
  - Nondominated branches: `['branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation', 'branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet', 'requirement_conditional_law_defined'] block constructs ['determinant', 'conditional', 'conditional_law', 'route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed. Typed encodability is blocked by `['route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet', 'requirement_conditional_law_defined']`.
  - Missing or unresolved assumptions:
    - `route_assumption_denominator_is_nonzero` status `missing`: denominator is nonzero
    - `route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet` status `missing`: matrix operand is square with valid determinant domain, usually positive definite for logdet
    - `requirement_conditional_law_defined` status `unresolved`: A conditional law for the expectation is defined.
  - Candidate assumption set that remains blocked:
    - The conditioned shock or path has finite support.
    - Every payoff/value term inside the expectation is finite at each support point.
    - The conditioning state or information set is explicitly defined.
    - The conditioning object `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}` is defined as a sigma-field, information set, state, or conditioning variable for this equality.
  - Candidate derivation route that remains blocked:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`.
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
  - Source refs for missing/unresolved evidence: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex > eq:a11-conditional > line 5901-5904']`
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
- Nondominated branches: `['branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation', 'branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition']`
- Unique top branch, only if one exists: `None`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'blocked_for_human_or_formalization_choice', 'target_ids': ['branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation', 'branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition'], 'branch_ids': ['branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation', 'branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition'], 'ledger_entry_ids': ['unresolved_branch_choice'], 'prerequisites': ['resolve_nondominated_branch_choice'], 'launch_vetoes': ['ledger_062b6a658bb48254d68c4c354b4218f2118eda9bd099fb3ae6d22c0a0f66ebc7', 'ledger_0e30882fc65a727db50c6d082d4cc22f1fbd1ec87b94da38d612276621af5606', 'ledger_361dcbaf6e10913e7b1f151a09847db4915ec4d99b9ac3698c159f177df4f82e', 'ledger_431f8c6960ccc6c926c9b445457144f568997d22b4132cd409b8c5805597e48d', 'ledger_66f0611b710e91e735587687b45fef63b61599a6a7a0178410acff2d6962e057', 'ledger_69e7282997966b71f611b9ef39e264ade769f848a92d037440b3f203dceedb91', 'ledger_6a56b3ad7001c755470e3c773ff35be743028649e2088b2b7699b15be72998ee', 'ledger_6f6ac950c24669e020fb1c741433d2902a0173aa284139eea14e4efcdff0f2e3', 'ledger_8f575623f3fb7eb8418e6e183e6e3dd4a80eddd4c820238ba53b8c7997a2f01b', 'ledger_95b0608137ca69ba51e280e33c4c3fee5abd754dd8a3056dc7f3c8f919b39db5', 'ledger_b0c53d4aacf7eb2ab4e38787c55ddcaade24b84a827347db43730437f18da12b', 'ledger_cebd26b901ddc70d6b0e4d08f9696d8c43d744c9f24ee2a59b8a8714338022dc'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_choice_record', 'schema_version': 'p06_formalization_choice@1', 'binding_fields': ['branch_ids', 'target_id'], 'path_role': 'decision_blocker'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_f79ab351efd75ba4ed1fe852def9a387dc885e374466be5e72b69fc712768814'}`
- Serialization position `1`: `branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_361dcbaf6e10913e7b1f151a09847db4915ec4d99b9ac3698c159f177df4f82e', 'ledger_431f8c6960ccc6c926c9b445457144f568997d22b4132cd409b8c5805597e48d', 'ledger_66f0611b710e91e735587687b45fef63b61599a6a7a0178410acff2d6962e057', 'ledger_95b0608137ca69ba51e280e33c4c3fee5abd754dd8a3056dc7f3c8f919b39db5', 'ledger_b0c53d4aacf7eb2ab4e38787c55ddcaade24b84a827347db43730437f18da12b', 'ledger_cebd26b901ddc70d6b0e4d08f9696d8c43d744c9f24ee2a59b8a8714338022dc'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'The conditioned shock or path has finite support.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Every payoff/value term inside the expectation is finite at each support point.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'The conditioning state or information set is explicitly defined.', 'status': 'candidate'}, {'id': 'legacy_assumption_4', 'statement': 'The conditioning object `z) = \\frac{h_t(u,v)^2+\\tau_t}{a_t(v)+\\tau_t} \\frac{\\varphi_d(u)}{\|\\det L_t\|}` is defined as a sigma-field, information set, state, or conditioning variable for this equality.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.
- Serialization position `2`: `branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_062b6a658bb48254d68c4c354b4218f2118eda9bd099fb3ae6d22c0a0f66ebc7', 'ledger_0e30882fc65a727db50c6d082d4cc22f1fbd1ec87b94da38d612276621af5606', 'ledger_69e7282997966b71f611b9ef39e264ade769f848a92d037440b3f203dceedb91', 'ledger_6a56b3ad7001c755470e3c773ff35be743028649e2088b2b7699b15be72998ee', 'ledger_6f6ac950c24669e020fb1c741433d2902a0173aa284139eea14e4efcdff0f2e3', 'ledger_8f575623f3fb7eb8418e6e183e6e3dd4a80eddd4c820238ba53b8c7997a2f01b'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'A conditional kernel or probability law is fixed for the random object.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'All random terms inside the expectation are measurable under that law.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Those terms are dominated by an integrable envelope or have finite conditional first moments.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_a11_conditional_2197dedc72a07b75']`
  - Typed unresolved constructs: `['determinant', 'conditional', 'conditional_law', 'route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['sage', 'lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet', 'requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['The conditioned shock or path has finite support.', 'Every payoff/value term inside the expectation is finite at each support point.', 'The conditioning state or information set is explicitly defined.', 'The conditioning object `z) = \\frac{h_t(u,v)^2+\\tau_t}{a_t(v)+\\tau_t} \\frac{\\varphi_d(u)}{\|\\det L_t\|}` is defined as a sigma-field, information set, state, or conditioning variable for this equality.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 4 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet', 'requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\det', '\\tau', '\\varphi']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`
- `branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_a11_conditional_2197dedc72a07b75']`
  - Typed unresolved constructs: `['determinant', 'conditional', 'conditional_law', 'route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['sage', 'lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet', 'requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['A conditional kernel or probability law is fixed for the random object.', 'All random terms inside the expectation are measurable under that law.', 'Those terms are dominated by an integrable envelope or have finite conditional first moments.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet', 'requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\det', '\\tau', '\\varphi']` whose mathematical types and backend names are not fixed.
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
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:a11-conditional` at attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > A normalized conditional and its physical correction > eq:a11-conditional > line 5901, add an assumptions paragraph: "For this displayed equality, assume: The conditioned shock or path has finite support. Every payoff/value term inside the expectation is finite at each support point. The conditioning state or information set is explicitly defined. The conditioning object `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}` is defined as a sigma-field, information set, state, or conditioning variable for this equality. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
- `patch_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:a11-conditional` at attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > A normalized conditional and its physical correction > eq:a11-conditional > line 5901, add an assumptions paragraph: "For this displayed equality, assume: A conditional kernel or probability law is fixed for the random object. All random terms inside the expectation are measurable under that law. Those terms are dominated by an integrable envelope or have finite conditional first moments. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `z) = \frac{h_t(u,v)^2+\tau_t}{a_t(v)+\tau_t} \frac{\varphi_d(u)}{\|\det L_t\|}`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
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
- `blocker_bounded_counterexample_attempt` (adapter_diagnostic)
  Problem: sympy_finite_domain did not certify or refute the target.
  Why: Expression is outside the conservative scalar grammar.
  Required next evidence: Provide a certifying backend result, concrete counterexample, formalization, or stronger assumption set.
- `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet', 'requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\det', '\\tau', '\\varphi']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_finite_state_conditional_expectation_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero', 'route_assumption_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet', 'requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\det', '\\tau', '\\varphi']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_a11_conditional_2197dedc72a07b75_kernel_integrability_condition_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_a11_conditional_2197dedc72a07b75_conditional_law_defined` (probability_condition)
  Problem: A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Required next evidence: Makes the expectation operator well defined.
- `blocker_semantic_packet_eq_a11_conditional_2197dedc72a07b75_measurable_integrable_payoff_terms` (integrability_condition)
  Problem: Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Required next evidence: Turns the displayed expression into a finite scalar equality.
- `blocker_semantic_packet_eq_a11_conditional_2197dedc72a07b75_conditioning_information_defined` (information_condition)
  Problem: A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Required next evidence: Fixes the scope of the conditional expectation used in the derivation.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

### 3. `eq:a11-chart-derivative`

- Location: `attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > Local analytical derivatives and the experiment they support > eq:a11-chart-derivative > line 6026`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex', 'line_start': 6026, 'line_end': 6027, 'label': 'eq:a11-chart-derivative', 'section_path': ['Recovering fitting accuracy without removing the physical defense', 'Local analytical derivatives and the experiment they support'], 'start_byte': 296632, 'end_byte': 296703, 'source_digest': '638a3ad7bfefc66e9e6f0ed94dc92a02cabeed7a917e62b1d1a7c65812e20dfb', 'obligation_id': 'obl_a84160360037550486e406f56a88b9df43162d45318a19b6a957670df48492e2', 'obligation_digest': 'a84160360037550486e406f56a88b9df43162d45318a19b6a957670df48492e2', 'labels': ['eq:a11-chart-derivative'], 'environment': 'align'}`
- Operators: `['equality']`
- Symbols: `{'latex_commands': ['\\dot'], 'bare_identifiers': ['L', 'm', 't', 'tu', 'u']}`
- Context graph statuses: `{'nearby_stated': 1, 'inferred_candidate': 4}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_a11_chart_derivative_a841603600375504`
- Typed obligation status: `needs_assumptions`
- Typed unresolved constructs: `['matrix_inverse']`

Source row target:

```tex
\dot u&=-L_t^{-1}(\dot m_t+\dot L_tu),
 \label{eq:a11-chart-derivative}
```

Full display target:

```tex
\dot u&=-L_t^{-1}(\dot m_t+\dot L_tu),
 \label{eq:a11-chart-derivative}
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
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:6011-6016', 'docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:6018-6067']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_a11_chart_derivative_a841603600375504`
  Diagnostic status: `needs_assumptions`
  Encodability: `{'status': 'candidate', 'candidate_backends': ['sage', 'human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
  Unresolved constructs: `['matrix_inverse']`
  Route hints: `[{'backend': 'sage', 'suitability': 'diagnostic_candidate', 'reason': 'Matrix-oriented notation may benefit from optional Sage/numeric diagnostics when safely encoded.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}]`
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
  - `document_gap_report_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_a11_chart_derivative_a841603600375504', 'branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_missing_domain_constraints', 'typed_repair_obligation_semantic_packet_eq_a11_chart_derivative_a841603600375504']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_missing_domain_constraints', 'blocker_formalization_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > Local analytical derivatives and the experiment they support > eq:a11-chart-derivative > line 6026`
  - Context branch: `branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions [] block constructs ['matrix_inverse'].
  - Why this is a derivation problem: No missing typed assumptions were detected by the bounded typed IR builder. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. The source span contains macros `['\\dot']` whose mathematical types and backend names are not fixed. Matrix inverse or solve notation requires an invertible operand; positive definiteness is one structured sufficient condition.
  - Candidate assumption set that remains blocked:
    - Define every symbol, domain, and operator in the cited source line.
    - Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.
    - Rerun the relevant assumption/proof audit after the typed obligation exists.
  - Candidate derivation route that remains blocked:
    - Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit.
  - Exact blockers before this can become a repair proposal:
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `missing_domain_or_shape_required`: The backend translation lacks required domain, dimension, or conformability constraints. Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: lean stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `branch_bound_backend_execution_required`: This assumption branch has no branch-bound backend request/result evidence. Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
  - Why no proposed edit is emitted: The serialization-only context branch is reported as a gap because typed mathematical assumptions or domain conditions remain unresolved; backend formalization or translation remains incomplete; the exact branch-bound backend action has not been executed.
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
- Nondominated branches: `['branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_formalization_required', 'target_ids': ['branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first'], 'ledger_entry_ids': ['ledger_234cadea8c2818e29a9a1f9619900bb7214dd065fd4dbf6460c0b72d6b47bb7a'], 'prerequisites': ['scope_bound:ledger_234cadea8c2818e29a9a1f9619900bb7214dd065fd4dbf6460c0b72d6b47bb7a'], 'launch_vetoes': ['ledger_4b6b38bc7f633ad08c9247cbec8e406f0b576d5f908f08f31d8ccf0c23845ec4', 'ledger_957eb87130b53e5a2ff7fbb93c85398e3e633f505c6eb827ec7ada49f5c681fb', 'ledger_9a855b304956f3a9d59db6d06908e53c188e49db4c672106c7b7145a6cb46fc5'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_8f4762f03064cf0944259ecf75f0027b28806b50db53414a153215e3fa7ce84f'}`
- Serialization position `1`: `branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_234cadea8c2818e29a9a1f9619900bb7214dd065fd4dbf6460c0b72d6b47bb7a', 'ledger_4b6b38bc7f633ad08c9247cbec8e406f0b576d5f908f08f31d8ccf0c23845ec4', 'ledger_957eb87130b53e5a2ff7fbb93c85398e3e633f505c6eb827ec7ada49f5c681fb', 'ledger_9a855b304956f3a9d59db6d06908e53c188e49db4c672106c7b7145a6cb46fc5'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_a11_chart_derivative_a841603600375504']`
  - Typed unresolved constructs: `['matrix_inverse']`
  - Typed encodability: `{'status': 'candidate', 'candidate_backends': ['sage', 'human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
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
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Add explicit scalar, vector, matrix, and conformability declarations before backend translation.
    - `blocker` status `blocking`: LaTeX macros must be translated into backend symbols before execution.
  - External-tool ledger: `['sympy:available', 'sage:available', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dot']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Matrix inverse or solve notation requires an invertible operand; positive definiteness is one structured sufficient condition.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Formalize local obligation`: Convert the cited line into a typed obligation before proposing a document edit.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:a11-chart-derivative` at attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > Local analytical derivatives and the experiment they support > eq:a11-chart-derivative > line 6026, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
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
- `blocker_bounded_counterexample_attempt` (adapter_diagnostic)
  Problem: sympy_finite_domain did not certify or refute the target.
  Why: Expression is outside the conservative scalar grammar.
  Required next evidence: Provide a certifying backend result, concrete counterexample, formalization, or stronger assumption set.
- `blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dot']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Matrix inverse or solve notation requires an invertible operand; positive definiteness is one structured sufficient condition.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_a11_chart_derivative_a841603600375504_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_a11_chart_derivative_a841603600375504_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 4. `eq:a11-tt-score`

- Location: `attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > Local analytical derivatives and the experiment they support > eq:a11-tt-score > line 6033`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['shape_conformability']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex', 'line_start': 6033, 'line_end': 6037, 'label': 'eq:a11-tt-score', 'section_path': ['Recovering fitting accuracy without removing the physical defense', 'Local analytical derivatives and the experiment they support'], 'start_byte': 296953, 'end_byte': 297146, 'source_digest': '638a3ad7bfefc66e9e6f0ed94dc92a02cabeed7a917e62b1d1a7c65812e20dfb', 'obligation_id': 'obl_a259c3b9dd08f680ab29dae4da1a89505364069225986e7ae7133f86c52406dc', 'obligation_digest': 'a259c3b9dd08f680ab29dae4da1a89505364069225986e7ae7133f86c52406dc', 'labels': ['eq:a11-tt-score'], 'environment': 'equation'}`
- Operators: `['equality', 'derivative', 'transpose']`
- Symbols: `{'latex_commands': ['\\dot', '\\log', '\\tau', '\\theta'], 'bare_identifiers': ['L', 'TT', 'a', 'h', 'q', 't', 'tr', 'u']}`
- Context graph statuses: `{'nearby_stated': 2, 'inferred_candidate': 4, 'missing': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['derivative', 'matrix_inverse', 'trace', 'transpose', 'route_assumption_denominator_is_nonzero']`

Source row target:

```tex
\partial_\theta\log q_t^{TT}=
 \frac{2h_t\dot h_t+\dot\tau_t}{h_t^2+\tau_t}
 -\frac{\dot a_t+\dot\tau_t}{a_t+\tau_t}
 -u^\top\dot u-\operatorname{tr}(L_t^{-1}\dot L_t),
 \label{eq:a11-tt-score}
```

Full display target:

```tex
\partial_\theta\log q_t^{TT}=
 \frac{2h_t\dot h_t+\dot\tau_t}{h_t^2+\tau_t}
 -\frac{\dot a_t+\dot\tau_t}{a_t+\tau_t}
 -u^\top\dot u-\operatorname{tr}(L_t^{-1}\dot L_t),
 \label{eq:a11-tt-score}
```

Mathematically missing obligations:
- `dimension_declarations` (shape_condition): Dimensions for every vector, matrix, and transposed object in the expression.
  Why: Matrix products and transposes are undefined unless the operands have conformable dimensions.
  Closes: Makes the matrix expression syntactically and semantically well formed.
- `scalar_vector_matrix_roles` (type_condition): A scalar/vector/matrix role for each ambiguous symbol.
  Why: The same notation can denote scalars, vectors, or matrices; the product type changes with that role.
  Closes: Prevents a shape-compatible expression from being misread as a different object.

Local context graph:
- `assumption_relevant_functions_differentiable` status `nearby_stated`
  Role: supports local derivative notation in the FOC route
  What: The relevant functions are differentiable.
  Why status: The condition is stated in the local paragraph/proposition context.
  Required next evidence: Use this as differentiability evidence, but do not treat it as integrability or derivative-expectation interchange evidence.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:6011-6016', 'docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:6018-6067']`
- `route_assumption_denominator_is_nonzero` status `missing`
  Role: route-required assumption from assumption_discovery
  What: denominator is nonzero
  Why status: The low-level route detector marked this assumption as missing.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:6033-6037']`
- `route_assumption_target_function_is_differentiable_on_the_stated_domain` status `nearby_stated`
  Role: route-required assumption from assumption_discovery
  What: target function is differentiable on the stated domain
  Why status: The condition is stated in the local paragraph/proposition context. This reconciles the low-level route requirement with local source evidence.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:6011-6016', 'docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex:6018-6067']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['sage', 'human_review'], 'blocked_by_assumption_ids': ['route_assumption_denominator_is_nonzero'], 'unsupported_constructs': [], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['derivative', 'matrix_inverse', 'trace', 'transpose', 'route_assumption_denominator_is_nonzero']`
  Route hints: `[{'backend': 'sage', 'suitability': 'diagnostic_candidate', 'reason': 'Matrix-oriented notation may benefit from optional Sage/numeric diagnostics when safely encoded.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `route_assumption_denominator_is_nonzero` status `missing`: denominator is nonzero
  - `assumption_relevant_functions_differentiable` status `nearby_stated`: The relevant functions are differentiable.
  - `route_assumption_target_function_is_differentiable_on_the_stated_domain` status `nearby_stated`: target function is differentiable on the stated domain
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
  - `document_gap_report_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_a11_tt_score_a259c3b9dd08f680', 'branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract', 'actionable_abstention:shape_conformability', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_domain_constraints', 'typed_repair_obligation_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_domain_constraints', 'blocker_formalization_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_sympy', 'blocker_formalization_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_sage', 'blocker_formalization_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_lean', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > Local analytical derivatives and the experiment they support > eq:a11-tt-score > line 6033`
  - Context branch: `branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['route_assumption_denominator_is_nonzero'] block constructs ['derivative', 'matrix_inverse', 'trace', 'transpose', 'route_assumption_denominator_is_nonzero'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `Makes the expression well typed before any algebraic or proof audit.` by making the operators and objects in the displayed equality well-defined before backend certification. Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`. The source span contains macros `['\\dot', '\\log', '\\tau', '\\theta']` whose mathematical types and backend names are not fixed. Matrix inverse or solve notation requires an invertible operand; positive definiteness is one structured sufficient condition.; Trace requires a square matrix operand.; Transpose and matrix product notation require conformable dimensions.
  - Missing or unresolved assumptions:
    - `route_assumption_denominator_is_nonzero` status `missing`: denominator is nonzero
  - Candidate assumption set that remains blocked:
    - Declare the dimension of every matrix and vector appearing in the product.
    - State that adjacent matrix products are conformable.
    - State whether transpose notation denotes an inner product, outer product, or matrix transpose.
  - Candidate derivation route that remains blocked:
    - Assign dimensions: Map each symbol to a scalar, vector, or matrix with explicit dimensions.
    - Check products: Verify adjacent dimensions match and the final expression has the claimed scalar/vector/matrix type.
  - Exact blockers before this can become a repair proposal:
    - `missing_domain_or_assumption_required`: The branch still has missing or unresolved typed assumptions. Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `missing_domain_or_shape_required`: The backend translation lacks required domain, dimension, or conformability constraints. Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: lean stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `branch_bound_backend_execution_required`: This assumption branch has no branch-bound backend request/result evidence. Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
  - Source refs for missing/unresolved evidence: `['docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.tex > eq:a11-tt-score > line 6033-6037']`
  - Why no proposed edit is emitted: The serialization-only context branch is reported as a gap because typed mathematical assumptions or domain conditions remain unresolved; backend formalization or translation remains incomplete; the exact branch-bound backend action has not been executed.
  - Backend evidence status: `typed_translation_blocked`
  - Validation: `typed_translation_blocked` from `strict_proposal_gate`
  - Non-claims: `['This is a gap report, not a repair proposal.', 'No proposed edit should be applied until the remaining blockers are closed by source evidence or a certifying backend.', 'The context branch is a serialization aid, not a scientific winner, global optimum, or minimal route.']`

Possible sufficient assumption sets:
- `explicit_dimension_contract`: Makes the expression well typed before any algebraic or proof audit.
  - Declare the dimension of every matrix and vector appearing in the product.
  - State that adjacent matrix products are conformable.
  - State whether transpose notation denotes an inner product, outer product, or matrix transpose.

Branch ranking:
- Contract: `repair_branch_ranking_result`
- Nondominated branches: `['branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_missing_domain_or_shape_required', 'target_ids': ['branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract'], 'branch_ids': ['branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract'], 'ledger_entry_ids': ['ledger_0ae0619c7ecf8f66e3b11c820b5aa1e85e9a8d495fc899072e1cc738bc55d727'], 'prerequisites': ['scope_bound:ledger_0ae0619c7ecf8f66e3b11c820b5aa1e85e9a8d495fc899072e1cc738bc55d727'], 'launch_vetoes': ['ledger_11a91f668f8dbb148916f59308f8c439f63cbbff2f40efa957902b7f3a76d3f4', 'ledger_635ea8be997c218b9512d5c91112b7ca476b51176b6e574b8567fdf2fbcaad6d', 'ledger_e6a1ba08e6523bee52e70215524646b2278f1ea9bc82ffe8bdc5bc84997e2c1d', 'ledger_f6e74ba267f4ae8791542cf2b1e40508930cea0dacdfff4e33309f20572ab2d9'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'missing_domain_or_shape_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_4afe8e6506b7c2b992d5b4a33e32dda2f509f645d85f60c6c226f04d5e4b0e54'}`
- Serialization position `1`: `branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_0ae0619c7ecf8f66e3b11c820b5aa1e85e9a8d495fc899072e1cc738bc55d727', 'ledger_11a91f668f8dbb148916f59308f8c439f63cbbff2f40efa957902b7f3a76d3f4', 'ledger_635ea8be997c218b9512d5c91112b7ca476b51176b6e574b8567fdf2fbcaad6d', 'ledger_e6a1ba08e6523bee52e70215524646b2278f1ea9bc82ffe8bdc5bc84997e2c1d', 'ledger_f6e74ba267f4ae8791542cf2b1e40508930cea0dacdfff4e33309f20572ab2d9'], 'covered_obligation_ids': ['dimension_declarations', 'scalar_vector_matrix_roles'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Declare the dimension of every matrix and vector appearing in the product.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'State that adjacent matrix products are conformable.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'State whether transpose notation denotes an inner product, outer product, or matrix transpose.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract` status `blocked_before_backend_certification`
  - Closes obligations: `['dimension_declarations', 'scalar_vector_matrix_roles']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680']`
  - Typed unresolved constructs: `['derivative', 'matrix_inverse', 'trace', 'transpose', 'route_assumption_denominator_is_nonzero']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['sage', 'human_review'], 'blocked_by_assumption_ids': ['route_assumption_denominator_is_nonzero'], 'unsupported_constructs': [], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `Makes the expression well typed before any algebraic or proof audit.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['Declare the dimension of every matrix and vector appearing in the product.', 'State that adjacent matrix products are conformable.', 'State whether transpose notation denotes an inner product, outer product, or matrix transpose.']
  - Route under assumptions:
    - Assign dimensions: Map each symbol to a scalar, vector, or matrix with explicit dimensions.
    - Check products: Verify adjacent dimensions match and the final expression has the claimed scalar/vector/matrix type.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Map each symbol to a scalar, vector, or matrix with explicit dimensions.
    - `derivation_split` status `diagnostic_route`: Verify adjacent dimensions match and the final expression has the claimed scalar/vector/matrix type.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Split the blocker into an explicit typed obligation and ask for the smallest backend-checkable subclaim.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Build a typed symbol map from every LaTeX macro in the target to backend variables or definitions.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dot', '\\log', '\\tau', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Matrix inverse or solve notation requires an invertible operand; positive definiteness is one structured sufficient condition.; Trace requires a square matrix operand.; Transpose and matrix product notation require conformable dimensions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Assign dimensions`: Map each symbol to a scalar, vector, or matrix with explicit dimensions.
- `Check products`: Verify adjacent dimensions match and the final expression has the claimed scalar/vector/matrix type.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `missing_assumptions`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:a11-tt-score` at attempt05_observation_aware_tt_algorithm_note.tex > Recovering fitting accuracy without removing the physical defense > Local analytical derivatives and the experiment they support > eq:a11-tt-score > line 6033, add an assumptions paragraph: "For this displayed equality, assume: Declare the dimension of every matrix and vector appearing in the product. State that adjacent matrix products are conformable. State whether transpose notation denotes an inner product, outer product, or matrix transpose. Under these assumptions, the derivation route is: Assign dimensions: Map each symbol to a scalar, vector, or matrix with explicit dimensions. Check products: Verify adjacent dimensions match and the final expression has the claimed scalar/vector/matrix type."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `Makes the expression well typed before any algebraic or proof audit.` by making the operators and objects in the displayed equality well-defined before backend certification.

Remaining blockers:
- `blocker_lean_source_required` (formalization_required)
  Problem: Lean certification was selected but no Lean source was supplied.
  Why: Direct Lean checking requires an explicit Lean statement/proof artifact.
  Required next evidence: Supply Lean source or a formalization branch before Lean certification.
- `blocker_sympy_algebra_attempt` (adapter_diagnostic)
  Problem: sympy did not certify or refute the target.
  Why: The target has missing route-required assumptions.
  Required next evidence: Provide a certifying backend result, concrete counterexample, formalization, or stronger assumption set.
- `blocker_bounded_counterexample_attempt` (adapter_diagnostic)
  Problem: sympy_finite_domain did not certify or refute the target.
  Why: Expression is outside the conservative scalar grammar.
  Required next evidence: Provide a certifying backend result, concrete counterexample, formalization, or stronger assumption set.
- `blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dot', '\\log', '\\tau', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Matrix inverse or solve notation requires an invertible operand; positive definiteness is one structured sufficient condition.; Trace requires a square matrix operand.; Transpose and matrix product notation require conformable dimensions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_explicit_dimension_contract_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_dimension_declarations` (shape_condition)
  Problem: Dimensions for every vector, matrix, and transposed object in the expression.
  Why: Matrix products and transposes are undefined unless the operands have conformable dimensions.
  Required next evidence: Makes the matrix expression syntactically and semantically well formed.
- `blocker_semantic_packet_eq_a11_tt_score_a259c3b9dd08f680_scalar_vector_matrix_roles` (type_condition)
  Problem: A scalar/vector/matrix role for each ambiguous symbol.
  Why: The same notation can denote scalars, vectors, or matrices; the product type changes with that role.
  Required next evidence: Prevents a shape-compatible expression from being misread as a different object.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

## Non-Claims

- `document_tree_audit_not_document_proof`: This workflow is a semantic gap and tree-evidence report; it does not prove the whole document.
- `semantic_packets_not_certificates`: Missing obligations, assumption sets, and derivation routes are deterministic guidance, not proof certificates.
- `proof_search_not_final_certificate`: LeanDojo, Pantograph, retrieval, route plans, and static extraction are diagnostic until direct Lean or another certifying backend checks the scoped target.
- `document_repair_publication_quarantined`: No returned candidate is an applicable document edit while publication mode is disabled.
