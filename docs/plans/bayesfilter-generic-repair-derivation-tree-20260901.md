# Document Derivation Tree Audit

Target: `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex`
Search mode: `agent_guided`
Grounding policy: `strict`
Publication mode: `disabled`
Execution: `serial` with `1` worker(s)

## Executive Summary

- Selected source rows: `3`
- Semantic packets: `3`
- Proposition/context packets: `0`
- Context graphs: `3`
- Context graph statuses: `{'inferred_candidate': 11, 'missing': 2, 'nearby_stated': 1}`
- Typed repair obligations: `3`
- Typed repair obligation statuses: `{'blocked_on_missing_typed_assumptions': 3}`
- Ranked branches: `3`
- Effective promoted branches: `0`
- Raw promoted branches (diagnostic only): `0`
- Document-ready repair proposals: `0`
- Document gap reports: `3`
- Document partial-evidence reports: `0`
- Failure classifications: `{'branch_execution_pending': 3, 'formalization_blocked': 3, 'mathematical_blocked': 2}`
- Tool-grounded compiler statuses: `{'compiled': 3}`
- Tool-grounded compiler validation errors: `0`
- Parallel execution failures: `0`
- Blockers: `30`
- Missing focus labels: `['eq:generic-dmis-expectation', 'eq:generic-defensive-bound', 'eq:generic-affine-contract']`
- This report is generic and document-local; it is not tied to a card-NPV-specific plan.

## Tools Used

| Tool | Purpose | Status | Contract | Arguments |
| --- | --- | --- | --- | --- |
| `locate_equations_in_file` | Localize source rows in the exact target file. | `completed` | `equation_rows` | `{"root": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05", "tex_path": "docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex"}` |
| `extract_derivation_targets_for_label` | Bind each selected equation label to one complete validated source-owned obligation. | `quarantined` | `derivation_target_extraction_result` | `{"failure_count": 3, "fallback_to_locator_row": false, "requested_equation_labels": ["eq:generic-dmis-expectation", "eq:generic-cv-unbiased", "eq:generic-cv-variance", "eq:generic-defensive-bound", "eq:generic-cv-tangent", "eq:generic-affine-contract"], "selected_target_count": 3}` |
| `build_proposition_context_packet` | Localize proposition labels that are not display-equation rows and attach equation targets/context. | `not_needed` | `proposition_context_packet_result` | `{"context_target_count": 0, "focus_labels": ["eq:generic-dmis-expectation", "eq:generic-cv-unbiased", "eq:generic-cv-variance", "eq:generic-defensive-bound", "eq:generic-cv-tangent", "eq:generic-affine-contract"]}` |
| `build_semantic_work_packet` | Classify each target and generate full-display semantic packets, missing obligations, assumption sets, and derivation routes. | `completed` | `semantic_work_packet` | `{"selected_rows": 3}` |
| `assumptions_required` | Detect route-required assumptions before backend proof attempts. | `completed` | `assumption_discovery_result` | `{"selected_rows": 3}` |
| `build_local_context_graph` | Classify local source evidence as stated, nearby stated, inferred, missing, or unresolved before proposing repairs. | `completed` | `local_context_graph` | `{"context_graph_count": 3, "status_counts": {"inferred_candidate": 11, "missing": 2, "nearby_stated": 1}}` |
| `typed_repair_obligation_from_packet` | Convert context graph and semantic packet evidence into typed repair obligations before branch/report generation. | `completed` | `typed_repair_obligation` | `{"status_counts": {"blocked_on_missing_typed_assumptions": 3}, "typed_repair_obligation_count": 3}` |
| `doctor_report` | Record external backend capability provenance. | `available` | `doctor_report` | `{"backend_env": "mathdevmcp-backends"}` |
| `can_derive_with_budget` | Run the external-tool-first branch controller on semantic packet targets. | `completed` | `derivation_search_tree_result` | `{"budget_profile": "standard", "execution_mode": "serial", "max_attempts": 2, "selected_rows": 3, "workers_used": 1}` |
| `rank_repair_branches` | Rank assumption branches by recorded backend evidence, blocker specificity, source support, closure strength, and non-minimality. | `completed` | `repair_branch_ranking_result` | `{"ranked_branch_count": 3}` |
| `tool_grounded_proposal_compiler` | Quarantine repair publication, compile legacy closure as partial evidence, and keep blocked branches as exact gap reports. | `completed` | `tool_grounded_proposal_compiler_result` | `{"gap_report_count": 3, "grounding_policy": "strict", "partial_evidence_count": 0, "repair_proposal_count": 0, "search_mode": "agent_guided", "validation_error_count": 0}` |
| `render_derivation_tree_report` | Render each derivation tree into structured evidence sections. | `completed` | `derivation_tree_report_result` | `{"rendered_trees": 3}` |

## Target Packets And Trees

### 1. `eq:generic-cv-unbiased`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-cv-unbiased > line 1933`
- Claim type: `statistical_estimator`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex', 'line_start': 1933, 'line_end': 1934, 'label': 'eq:generic-cv-unbiased', 'section_path': ['A Coherent Proposal and Testing Program', 'The model-independent repair actually implemented'], 'start_byte': 86192, 'end_byte': 86280, 'source_digest': '59a2c6b257ed1eb08f736f493f5ca09752e78db5beec06ec28eb7ca70edfab27', 'obligation_id': 'obl_23a11ad7bc4a452f473c720d95ea690f4fe320f96a658eb8f5fd89b5996c7273', 'obligation_digest': '23a11ad7bc4a452f473c720d95ea690f4fe320f96a658eb8f5fd89b5996c7273', 'labels': ['eq:generic-cv-unbiased'], 'environment': 'align'}`
- Operators: `['equality', 'integral']`
- Symbols: `{'latex_commands': ['\\gamma', '\\rm'], 'bare_identifiers': ['CV', 'E', 'Z', 'dx', 'x']}`
- Context graph statuses: `{'inferred_candidate': 4}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['expectation']`

Source row target:

```tex
\mathbb E[\widehat Z_{\rm CV}]&=\int\gamma(x)\,dx,
       \label{eq:generic-cv-unbiased}
```

Full display target:

```tex
\mathbb E[\widehat Z_{\rm CV}]&=\int\gamma(x)\,dx,
       \label{eq:generic-cv-unbiased}
```

Mathematically missing obligations:
- `formalized_local_obligation` (formalization_condition): A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Closes: Creates the next deterministic target for assumption discovery or proof audit.

Local context graph:

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': ['expectation'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['expectation']`
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
  - `document_gap_report_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f', 'branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-cv-unbiased > line 1933`
  - Context branch: `branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions [] block constructs ['expectation'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target. The source span contains macros `['\\gamma', '\\rm']` whose mathematical types and backend names are not fixed.
  - Candidate assumption set that remains blocked:
    - Define every symbol, domain, and operator in the cited source line.
    - Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.
    - Rerun the relevant assumption/proof audit after the typed obligation exists.
  - Candidate derivation route that remains blocked:
    - Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit.
  - Exact blockers before this can become a repair proposal:
    - `conditional_expectation_translation_required`: The expectation operator cannot be translated as a scalar algebraic expression yet. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
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
- Nondominated branches: `['branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_conditional_expectation_translation_required', 'target_ids': ['branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first'], 'ledger_entry_ids': ['ledger_083776b97d68896fa37cf56de38b3428eb8614bbc50131c1fd56a41488cd7a40'], 'prerequisites': ['scope_bound:ledger_083776b97d68896fa37cf56de38b3428eb8614bbc50131c1fd56a41488cd7a40'], 'launch_vetoes': ['ledger_0a108657bf8af156844dd20e681194981809c486bd0ed49f771ae75cf6d20fca', 'ledger_d33d95755346dbb5cc81ec2597b86946f5759d5aae2c9a2f69c5f897a0f89a4f', 'ledger_f15fcd2e8634b0046e08c5444f6a14c2aaf982d7a80a3ac868023a3b71013083'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'conditional_expectation_translation_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_b10c49fa3c6567106f518ff2cfbacc53d68a33d6a40e320199432e55d9127f38'}`
- Serialization position `1`: `branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_083776b97d68896fa37cf56de38b3428eb8614bbc50131c1fd56a41488cd7a40', 'ledger_0a108657bf8af156844dd20e681194981809c486bd0ed49f771ae75cf6d20fca', 'ledger_d33d95755346dbb5cc81ec2597b86946f5759d5aae2c9a2f69c5f897a0f89a4f', 'ledger_f15fcd2e8634b0046e08c5444f6a14c2aaf982d7a80a3ac868023a3b71013083'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f']`
  - Typed unresolved constructs: `['expectation']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': ['expectation'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
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
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Split the blocker into an explicit typed obligation and ask for the smallest backend-checkable subclaim.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Build a typed symbol map from every LaTeX macro in the target to backend variables or definitions.
    - `blocker` status `blocking`: The expectation operator cannot be translated as a scalar algebraic expression yet.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_conditional_expectation_translation_required` (conditional_expectation_translation_required): The expectation operator cannot be translated as a scalar algebraic expression yet.
      Why: A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\gamma', '\\rm']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
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
- `patch_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:generic-cv-unbiased` at attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-cv-unbiased > line 1933, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
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
- `blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_conditional_expectation_translation_required` (conditional_expectation_translation_required)
  Problem: The expectation operator cannot be translated as a scalar algebraic expression yet.
  Why: A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\gamma', '\\rm']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_generic_cv_unbiased_23a11ad7bc4a452f_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 2. `eq:generic-cv-variance`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-cv-variance > line 1935`
- Claim type: `statistical_estimator`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex', 'line_start': 1935, 'line_end': 1938, 'label': 'eq:generic-cv-variance', 'section_path': ['A Coherent Proposal and Testing Program', 'The model-independent repair actually implemented'], 'start_byte': 86285, 'end_byte': 86464, 'source_digest': '59a2c6b257ed1eb08f736f493f5ca09752e78db5beec06ec28eb7ca70edfab27', 'obligation_id': 'obl_35335750113c57844cb81b55c9c94af69f6e023714d51aaa57c442e947c79a5b', 'obligation_digest': '35335750113c57844cb81b55c9c94af69f6e023714d51aaa57c442e947c79a5b', 'labels': ['eq:generic-cv-variance'], 'environment': 'align'}`
- Operators: `['equality', 'integral']`
- Symbols: `{'latex_commands': ['\\gamma', '\\rm'], 'bare_identifiers': ['CV', 'H', 'Var', 'Z', 'dx', 'h', 'q']}`
- Context graph statuses: `{'inferred_candidate': 4, 'missing': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_generic_cv_variance_35335750113c5784`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['route_assumption_denominator_is_nonzero']`

Source row target:

```tex
\operatorname{Var}(\widehat Z_{\rm CV})
    &=\frac1N\left\{\int\frac{(\gamma-h)^2}{q}\,dx
       -\left(\int\gamma\,dx-Z_H\right)^2\right\}.
       \label{eq:generic-cv-variance}
```

Full display target:

```tex
\operatorname{Var}(\widehat Z_{\rm CV})
    &=\frac1N\left\{\int\frac{(\gamma-h)^2}{q}\,dx
       -\left(\int\gamma\,dx-Z_H\right)^2\right\}.
       \label{eq:generic-cv-variance}
```

Mathematically missing obligations:
- `formalized_local_obligation` (formalization_condition): A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Closes: Creates the next deterministic target for assumption discovery or proof audit.

Local context graph:
- `route_assumption_denominator_is_nonzero` status `missing`
  Role: route-required assumption from assumption_discovery
  What: denominator is nonzero
  Why status: The low-level route detector marked this assumption as missing.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex:1935-1938']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_generic_cv_variance_35335750113c5784`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': ['route_assumption_denominator_is_nonzero'], 'unsupported_constructs': [], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['route_assumption_denominator_is_nonzero']`
  Route hints: `[{'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `route_assumption_denominator_is_nonzero` status `missing`: denominator is nonzero
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
  - `document_gap_report_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_generic_cv_variance_35335750113c5784', 'branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_generic_cv_variance_35335750113c5784']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-cv-variance > line 1935`
  - Context branch: `branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['route_assumption_denominator_is_nonzero'] block constructs ['route_assumption_denominator_is_nonzero'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`. The source span contains macros `['\\gamma', '\\rm']` whose mathematical types and backend names are not fixed.
  - Missing or unresolved assumptions:
    - `route_assumption_denominator_is_nonzero` status `missing`: denominator is nonzero
  - Candidate assumption set that remains blocked:
    - Define every symbol, domain, and operator in the cited source line.
    - Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.
    - Rerun the relevant assumption/proof audit after the typed obligation exists.
  - Candidate derivation route that remains blocked:
    - Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit.
  - Exact blockers before this can become a repair proposal:
    - `missing_domain_or_assumption_required`: The branch still has missing or unresolved typed assumptions. Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: lean stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `branch_bound_backend_execution_required`: This assumption branch has no branch-bound backend request/result evidence. Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
  - Source refs for missing/unresolved evidence: `['docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex > eq:generic-cv-variance > line 1935-1938']`
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
- Nondominated branches: `['branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_missing_domain_or_assumption_required', 'target_ids': ['branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first'], 'ledger_entry_ids': ['ledger_86e1f8a9365277204be5c286b84f74b206ee84b2dc6dd455fdd878a352659fd6'], 'prerequisites': ['scope_bound:ledger_86e1f8a9365277204be5c286b84f74b206ee84b2dc6dd455fdd878a352659fd6'], 'launch_vetoes': ['ledger_b34bc400b83fe1e065ea3912387a0da6b7e13bbdf3eaa3a3dac32b19fd37a97b', 'ledger_cf99c32691c39e50fabb207d54ec0c0e0538bbea6291969eb8345b1d5afaf0a4', 'ledger_ddb69d5b599640a356049624729a2eae8ae90bba343ba22bab657560a866a9e6'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'missing_domain_or_assumption_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_c50d30535ec75c1096470af5bc18fc7c4486207559a05ec5c06534457ce37c4d'}`
- Serialization position `1`: `branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_86e1f8a9365277204be5c286b84f74b206ee84b2dc6dd455fdd878a352659fd6', 'ledger_b34bc400b83fe1e065ea3912387a0da6b7e13bbdf3eaa3a3dac32b19fd37a97b', 'ledger_cf99c32691c39e50fabb207d54ec0c0e0538bbea6291969eb8345b1d5afaf0a4', 'ledger_ddb69d5b599640a356049624729a2eae8ae90bba343ba22bab657560a866a9e6'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_generic_cv_variance_35335750113c5784']`
  - Typed unresolved constructs: `['route_assumption_denominator_is_nonzero']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': ['route_assumption_denominator_is_nonzero'], 'unsupported_constructs': [], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
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
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Split the blocker into an explicit typed obligation and ask for the smallest backend-checkable subclaim.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Build a typed symbol map from every LaTeX macro in the target to backend variables or definitions.
    - `blocker` status `blocking`: The branch still has missing or unresolved typed assumptions.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\gamma', '\\rm']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Formalize local obligation`: Convert the cited line into a typed obligation before proposing a document edit.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `missing_assumptions`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:generic-cv-variance` at attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-cv-variance > line 1935, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification.

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
- `blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\gamma', '\\rm']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_generic_cv_variance_35335750113c5784_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_generic_cv_variance_35335750113c5784_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 3. `eq:generic-cv-tangent`

- Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-cv-tangent > line 2000`
- Claim type: `statistical_estimator`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex', 'line_start': 2000, 'line_end': 2004, 'label': 'eq:generic-cv-tangent', 'section_path': ['A Coherent Proposal and Testing Program', 'The model-independent repair actually implemented'], 'start_byte': 89185, 'end_byte': 89363, 'source_digest': '59a2c6b257ed1eb08f736f493f5ca09752e78db5beec06ec28eb7ca70edfab27', 'obligation_id': 'obl_d092564592a70cc25fb733e7f81d98da063e3e4efcaa48c428ae7fe0f35d3664', 'obligation_digest': 'd092564592a70cc25fb733e7f81d98da063e3e4efcaa48c428ae7fe0f35d3664', 'labels': ['eq:generic-cv-tangent'], 'environment': 'equation'}`
- Operators: `['equality', 'summation']`
- Symbols: `{'latex_commands': ['\\dot', '\\ell', '\\gamma', '\\rm'], 'bare_identifiers': ['CV', 'H', 'Z', 'b', 'bank', 'h', 'i', 'q']}`
- Context graph statuses: `{'nearby_stated': 1, 'inferred_candidate': 3, 'missing': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['route_assumption_denominator_is_nonzero']`

Source row target:

```tex
\label{eq:generic-cv-tangent}
  \dot{\widehat Z}_{\rm CV,bank}
  =\dot Z_H+\sum_i\frac{b_i}{q_i}
    \left[(\dot\gamma_i-\dot h_i)
          -(\gamma_i-h_i)\dot\ell_{q,i}\right].
```

Full display target:

```tex
\label{eq:generic-cv-tangent}
  \dot{\widehat Z}_{\rm CV,bank}
  =\dot Z_H+\sum_i\frac{b_i}{q_i}
    \left[(\dot\gamma_i-\dot h_i)
          -(\gamma_i-h_i)\dot\ell_{q,i}\right].
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
  Source refs: `['docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex:2018-2027']`
- `route_assumption_denominator_is_nonzero` status `missing`
  Role: route-required assumption from assumption_discovery
  What: denominator is nonzero
  Why status: The low-level route detector marked this assumption as missing.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex:2000-2004']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': ['route_assumption_denominator_is_nonzero'], 'unsupported_constructs': [], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['route_assumption_denominator_is_nonzero']`
  Route hints: `[{'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `route_assumption_denominator_is_nonzero` status `missing`: denominator is nonzero
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
  - `document_gap_report_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_generic_cv_tangent_d092564592a70cc2', 'branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-cv-tangent > line 2000`
  - Context branch: `branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['route_assumption_denominator_is_nonzero'] block constructs ['route_assumption_denominator_is_nonzero'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`. The source span contains macros `['\\dot', '\\ell', '\\gamma', '\\rm']` whose mathematical types and backend names are not fixed.
  - Missing or unresolved assumptions:
    - `route_assumption_denominator_is_nonzero` status `missing`: denominator is nonzero
  - Candidate assumption set that remains blocked:
    - Define every symbol, domain, and operator in the cited source line.
    - Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.
    - Rerun the relevant assumption/proof audit after the typed obligation exists.
  - Candidate derivation route that remains blocked:
    - Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit.
  - Exact blockers before this can become a repair proposal:
    - `missing_domain_or_assumption_required`: The branch still has missing or unresolved typed assumptions. Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: lean stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `branch_bound_backend_execution_required`: This assumption branch has no branch-bound backend request/result evidence. Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
  - Source refs for missing/unresolved evidence: `['docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex > eq:generic-cv-tangent > line 2000-2004']`
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
- Nondominated branches: `['branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_formalization_required', 'target_ids': ['branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first'], 'ledger_entry_ids': ['ledger_18e3b05b878af6f2ddc031c34853a6b656bc4e9ba55f4b1371746fa059e70fa4'], 'prerequisites': ['scope_bound:ledger_18e3b05b878af6f2ddc031c34853a6b656bc4e9ba55f4b1371746fa059e70fa4'], 'launch_vetoes': ['ledger_28b523786093d526120f1260873a478d97797cdfd2c3dd965ce4616a94f4422f', 'ledger_71ec01d779f166e3a27d43c115b812f6abd71e3a2b2f0a546f0cc4a9c8cfa5a6', 'ledger_8a0df5b3fc2259d2cfad7dcc1cf485a9b11fb0c02172ce22809167a0bf0e353c'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_0914eb35cf2663f771fd12b7978883c1a4b03e1154e8962eb3b0fb2d2983bfe5'}`
- Serialization position `1`: `branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_18e3b05b878af6f2ddc031c34853a6b656bc4e9ba55f4b1371746fa059e70fa4', 'ledger_28b523786093d526120f1260873a478d97797cdfd2c3dd965ce4616a94f4422f', 'ledger_71ec01d779f166e3a27d43c115b812f6abd71e3a2b2f0a546f0cc4a9c8cfa5a6', 'ledger_8a0df5b3fc2259d2cfad7dcc1cf485a9b11fb0c02172ce22809167a0bf0e353c'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2']`
  - Typed unresolved constructs: `['route_assumption_denominator_is_nonzero']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': ['route_assumption_denominator_is_nonzero'], 'unsupported_constructs': [], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
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
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Split the blocker into an explicit typed obligation and ask for the smallest backend-checkable subclaim.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Build a typed symbol map from every LaTeX macro in the target to backend variables or definitions.
    - `blocker` status `blocking`: The branch still has missing or unresolved typed assumptions.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dot', '\\ell', '\\gamma', '\\rm']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Formalize local obligation`: Convert the cited line into a typed obligation before proposing a document edit.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `missing_assumptions`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:generic-cv-tangent` at attempt05_n4_failure_analysis.tex > A Coherent Proposal and Testing Program > The model-independent repair actually implemented > eq:generic-cv-tangent > line 2000, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification.

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
- `blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dot', '\\ell', '\\gamma', '\\rm']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_generic_cv_tangent_d092564592a70cc2_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

## Non-Claims

- `document_tree_audit_not_document_proof`: This workflow is a semantic gap and tree-evidence report; it does not prove the whole document.
- `semantic_packets_not_certificates`: Missing obligations, assumption sets, and derivation routes are deterministic guidance, not proof certificates.
- `proof_search_not_final_certificate`: LeanDojo, Pantograph, retrieval, route plans, and static extraction are diagnostic until direct Lean or another certifying backend checks the scoped target.
- `document_repair_publication_quarantined`: No returned candidate is an applicable document edit while publication mode is disabled.
