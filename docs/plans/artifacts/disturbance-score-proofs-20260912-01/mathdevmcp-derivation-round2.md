# Document Derivation Tree Audit

Target: `/home/chakwong/BayesFilter/docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex`
Search mode: `agent_guided`
Grounding policy: `strict`
Publication mode: `disabled`
Execution: `serial` with `1` worker(s)

## Executive Summary

- Selected source rows: `16`
- Semantic packets: `16`
- Proposition/context packets: `0`
- Context graphs: `16`
- Context graph statuses: `{'nearby_stated': 18, 'inferred_candidate': 71, 'missing': 18, 'stated': 1}`
- Typed repair obligations: `16`
- Typed repair obligation statuses: `{'blocked_on_missing_typed_assumptions': 12, 'needs_assumptions': 2, 'ready_for_backend': 2}`
- Ranked branches: `24`
- Effective promoted branches: `0`
- Raw promoted branches (diagnostic only): `0`
- Document-ready repair proposals: `0`
- Document gap reports: `16`
- Document partial-evidence reports: `0`
- Failure classifications: `{'branch_execution_pending': 16, 'formalization_blocked': 16, 'mathematical_blocked': 13}`
- Tool-grounded compiler statuses: `{'compiled': 16}`
- Tool-grounded compiler validation errors: `0`
- Parallel execution failures: `0`
- Blockers: `275`
- Missing focus labels: `['eq:disturbance-state-map', 'eq:disturbance-model', 'eq:disturbance-complete-score', 'eq:disturbance-importance-pair', 'eq:disturbance-importance-identities', 'eq:disturbance-apf-weight', 'eq:disturbance-particle-measure', 'eq:disturbance-particle-derivative', 'eq:disturbance-particle-induction-step', 'eq:ratio-bias-counterexample', 'eq:disturbance-control-variate', 'eq:disturbance-reference-control', 'eq:disturbance-rao-blackwell', 'eq:disturbance-moving-line-score']`
- This report is generic and document-local; it is not tied to a card-NPV-specific plan.

## Tools Used

| Tool | Purpose | Status | Contract | Arguments |
| --- | --- | --- | --- | --- |
| `locate_equations_in_file` | Localize source rows in the exact target file. | `completed` | `equation_rows` | `{"root": "/home/chakwong/BayesFilter/docs/papers/ledh_younis_kdm_score", "tex_path": "/home/chakwong/BayesFilter/docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex"}` |
| `extract_derivation_targets_for_label` | Bind each selected equation label to one complete validated source-owned obligation. | `quarantined` | `derivation_target_extraction_result` | `{"failure_count": 14, "fallback_to_locator_row": false, "requested_equation_labels": ["eq:disturbance-state-map", "eq:disturbance-model", "eq:disturbance-joint-integrand", "eq:disturbance-tangent", "eq:disturbance-shock-total-derivative", "eq:disturbance-complete-score", "eq:disturbance-score-identity", "eq:disturbance-differentiate-integral", "eq:disturbance-proposal-law", "eq:disturbance-importance-pair", "eq:disturbance-importance-identities", "eq:disturbance-apf-weight", "eq:disturbance-apf-conditional-normalizer", "eq:disturbance-history-kernel", "eq:disturbance-particle-measure", "eq:disturbance-particle-unbiasedness", "eq:disturbance-particle-derivative", "eq:disturbance-particle-induction-step", "eq:disturbance-analytical-score-recursion", "eq:ratio-bias-counterexample", "eq:control-known-integral", "eq:disturbance-control-variate", "eq:control-variate-coefficient", "eq:control-variate-variance", "eq:disturbance-reference-control", "eq:disturbance-reference-residual", "eq:disturbance-conditional-score", "eq:disturbance-rao-blackwell", "eq:disturbance-moving-line-score", "eq:disturbance-moving-line-likelihood"], "selected_target_count": 16}` |
| `build_proposition_context_packet` | Localize proposition labels that are not display-equation rows and attach equation targets/context. | `not_needed` | `proposition_context_packet_result` | `{"context_target_count": 0, "focus_labels": ["eq:disturbance-state-map", "eq:disturbance-model", "eq:disturbance-joint-integrand", "eq:disturbance-tangent", "eq:disturbance-shock-total-derivative", "eq:disturbance-complete-score", "eq:disturbance-score-identity", "eq:disturbance-differentiate-integral", "eq:disturbance-proposal-law", "eq:disturbance-importance-pair", "eq:disturbance-importance-identities", "eq:disturbance-apf-weight", "eq:disturbance-apf-conditional-normalizer", "eq:disturbance-history-kernel", "eq:disturbance-particle-measure", "eq:disturbance-particle-unbiasedness", "eq:disturbance-particle-derivative", "eq:disturbance-particle-induction-step", "eq:disturbance-analytical-score-recursion", "eq:ratio-bias-counterexample", "eq:control-known-integral", "eq:disturbance-control-variate", "eq:control-variate-coefficient", "eq:control-variate-variance", "eq:disturbance-reference-control", "eq:disturbance-reference-residual", "eq:disturbance-conditional-score", "eq:disturbance-rao-blackwell", "eq:disturbance-moving-line-score", "eq:disturbance-moving-line-likelihood"]}` |
| `build_semantic_work_packet` | Classify each target and generate full-display semantic packets, missing obligations, assumption sets, and derivation routes. | `completed` | `semantic_work_packet` | `{"selected_rows": 16}` |
| `assumptions_required` | Detect route-required assumptions before backend proof attempts. | `completed` | `assumption_discovery_result` | `{"selected_rows": 16}` |
| `build_local_context_graph` | Classify local source evidence as stated, nearby stated, inferred, missing, or unresolved before proposing repairs. | `completed` | `local_context_graph` | `{"context_graph_count": 16, "status_counts": {"inferred_candidate": 71, "missing": 18, "nearby_stated": 18, "stated": 1}}` |
| `typed_repair_obligation_from_packet` | Convert context graph and semantic packet evidence into typed repair obligations before branch/report generation. | `completed` | `typed_repair_obligation` | `{"status_counts": {"blocked_on_missing_typed_assumptions": 12, "needs_assumptions": 2, "ready_for_backend": 2}, "typed_repair_obligation_count": 16}` |
| `doctor_report` | Record external backend capability provenance. | `available` | `doctor_report` | `{"backend_env": "mathdevmcp-backends"}` |
| `can_derive_with_budget` | Run the external-tool-first branch controller on semantic packet targets. | `completed` | `derivation_search_tree_result` | `{"budget_profile": "standard", "execution_mode": "serial", "max_attempts": 2, "selected_rows": 16, "workers_used": 1}` |
| `rank_repair_branches` | Rank assumption branches by recorded backend evidence, blocker specificity, source support, closure strength, and non-minimality. | `completed` | `repair_branch_ranking_result` | `{"ranked_branch_count": 24}` |
| `tool_grounded_proposal_compiler` | Quarantine repair publication, compile legacy closure as partial evidence, and keep blocked branches as exact gap reports. | `completed` | `tool_grounded_proposal_compiler_result` | `{"gap_report_count": 16, "grounding_policy": "strict", "partial_evidence_count": 0, "repair_proposal_count": 0, "search_mode": "agent_guided", "validation_error_count": 0}` |
| `render_derivation_tree_report` | Render each derivation tree into structured evidence sections. | `completed` | `derivation_tree_report_result` | `{"rendered_trees": 16}` |

## Target Packets And Trees

### 1. `eq:disturbance-joint-integrand`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-joint-integrand > line 1440`
- Claim type: `definition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['conditional_expectation']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1440, 'line_end': 1443, 'label': 'eq:disturbance-joint-integrand', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 72567, 'end_byte': 72736, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_f0bd884aa1c60d2e644432cbf728dc34e9c661ce27276bcd3fb02cee1d886043', 'obligation_digest': 'f0bd884aa1c60d2e644432cbf728dc34e9c661ce27276bcd3fb02cee1d886043', 'labels': ['eq:disturbance-joint-integrand'], 'environment': 'equation'}`
- Operators: `['equality', 'conditional_bar']`
- Symbols: `{'latex_commands': ['\\pi', '\\rho', '\\theta'], 'bare_identifiers': ['T', 'g', 'r', 't', 'u', 'v', 'x', 'y', 'z']}`
- Context graph statuses: `{'nearby_stated': 2, 'inferred_candidate': 3, 'missing': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['conditional', 'conditional_law']`

Source row target:

```tex
\pi_\theta(z;y)=\rho_{0,\theta}(v_0)
        \prod_{t=1}^T r_{t,\theta}(u_t\mid x_{t-1})
        g_{t,\theta}(y_t\mid x_t)
        \label{eq:disturbance-joint-integrand}
```

Full display target:

```tex
\pi_\theta(z;y)=\rho_{0,\theta}(v_0)
        \prod_{t=1}^T r_{t,\theta}(u_t\mid x_{t-1})
        g_{t,\theta}(y_t\mid x_t)
        \label{eq:disturbance-joint-integrand}
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
- `assumption_relevant_functions_differentiable` status `nearby_stated`
  Role: supports local derivative notation in the FOC route
  What: The relevant functions are differentiable.
  Why status: The condition is stated in the local paragraph/proposition context.
  Required next evidence: Use this as differentiability evidence, but do not treat it as integrability or derivative-expectation interchange evidence.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1426-1450', 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1452-1456']`
- `requirement_conditional_law_defined` status `missing`
  Role: well-definedness condition for conditional expectation
  What: A conditional law for the expectation is defined.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: Cite or add the transition kernel/probability law used by the conditional expectation.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1440-1443']`
- `requirement_conditional_integrability` status `nearby_stated`
  Role: finite-scalar condition for expectation-valued equations
  What: Random terms inside the conditional expectation are measurable and integrable.
  Why status: The condition is stated in nearby local context, not in the target span.
  Required next evidence: Cite or add measurability and finite conditional first-moment/dominated-envelope conditions.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1440-1443']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['conditional', 'conditional_law']`
  Route hints: `[{'backend': 'lean', 'suitability': 'formalization_candidate', 'reason': 'Typed notation may be formalized manually and checked by Lean.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
  - `assumption_relevant_functions_differentiable` status `nearby_stated`: The relevant functions are differentiable.
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e', 'branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation', 'actionable_abstention:conditional_expectation', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_lean', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-joint-integrand > line 1440`
  - Context branch: `branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation`
  - Context selection authority: `serialization_only_nondominated_context`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['requirement_conditional_law_defined'] block constructs ['conditional', 'conditional_law'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed. Typed encodability is blocked by `['requirement_conditional_law_defined']`.
  - Missing or unresolved assumptions:
    - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
  - Candidate assumption set that remains blocked:
    - The conditioned shock or path has finite support.
    - Every payoff/value term inside the expectation is finite at each support point.
    - The conditioning state or information set is explicitly defined.
    - The conditioning object `x_{t-1}) g_{t,\theta}(y_t\mid x_t)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.
  - Candidate derivation route that remains blocked:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`.
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
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:disturbance-joint-integrand > line 1440-1443']`
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition']`
- Unique top branch, only if one exists: `None`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'blocked_for_human_or_formalization_choice', 'target_ids': ['branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition'], 'ledger_entry_ids': ['unresolved_branch_choice'], 'prerequisites': ['resolve_nondominated_branch_choice'], 'launch_vetoes': ['ledger_058ee007a717a68cc53169cae1737268bf633da16ad7e9e937695eb59c066c96', 'ledger_0c50d07f578a088a5ee1231528634903ae04dd0882c1997d5e2ef7a371f20225', 'ledger_2894cf58e4007604e3d62b6af5eac961e34fb846c44f9dca251bc045c1e8d857', 'ledger_3d1aa29ccfa6d121c7068fe0aeef67397935b3c7881e240a70dd5e8b0d6c3876', 'ledger_4be9abdf6cca4c5a2fe117fa9543123f19e6c17aa60c63d1ba67804c158f6396', 'ledger_6aa8008c4b92e519b9edbc49d01c07af8eb59d43a412faf40306ee184d6b4cd8', 'ledger_6eea1b3eea22bc3816893d64a938084f2681d7ceeab998739766b24e53b9af15', 'ledger_6f9a6f006d109e539ccf5786db93e0960cc1eb3b53e2d8aeff3b72e88c1c43ad', 'ledger_ae160057f2895cff44eabead3748719ce6b4890bbd7da9372ef1b0aa6fec7775', 'ledger_b4b24b2d740a5ea32f4db1cd9cd1280facd9815546dfef5bd73b9f3e90054a23', 'ledger_c6538d972e1cb80062c679ba4fa28b7551c332c71d134dbf96b663b1e4b82f24', 'ledger_fcd4d75fafd8fc90a20dddebfeea52a9d69c990601ba4469d812bf89d3715c66'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_choice_record', 'schema_version': 'p06_formalization_choice@1', 'binding_fields': ['branch_ids', 'target_id'], 'path_role': 'decision_blocker'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_fc49c5f0a658f4be93d9377fa6500dc69489bf6add24f7b81d30a2b0221e4f92'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_058ee007a717a68cc53169cae1737268bf633da16ad7e9e937695eb59c066c96', 'ledger_2894cf58e4007604e3d62b6af5eac961e34fb846c44f9dca251bc045c1e8d857', 'ledger_3d1aa29ccfa6d121c7068fe0aeef67397935b3c7881e240a70dd5e8b0d6c3876', 'ledger_6f9a6f006d109e539ccf5786db93e0960cc1eb3b53e2d8aeff3b72e88c1c43ad', 'ledger_ae160057f2895cff44eabead3748719ce6b4890bbd7da9372ef1b0aa6fec7775', 'ledger_b4b24b2d740a5ea32f4db1cd9cd1280facd9815546dfef5bd73b9f3e90054a23'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'The conditioned shock or path has finite support.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Every payoff/value term inside the expectation is finite at each support point.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'The conditioning state or information set is explicitly defined.', 'status': 'candidate'}, {'id': 'legacy_assumption_4', 'statement': 'The conditioning object `x_{t-1}) g_{t,\\theta}(y_t\\mid x_t)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.
- Serialization position `2`: `branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_0c50d07f578a088a5ee1231528634903ae04dd0882c1997d5e2ef7a371f20225', 'ledger_4be9abdf6cca4c5a2fe117fa9543123f19e6c17aa60c63d1ba67804c158f6396', 'ledger_6aa8008c4b92e519b9edbc49d01c07af8eb59d43a412faf40306ee184d6b4cd8', 'ledger_6eea1b3eea22bc3816893d64a938084f2681d7ceeab998739766b24e53b9af15', 'ledger_c6538d972e1cb80062c679ba4fa28b7551c332c71d134dbf96b663b1e4b82f24', 'ledger_fcd4d75fafd8fc90a20dddebfeea52a9d69c990601ba4469d812bf89d3715c66'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'A conditional kernel or probability law is fixed for the random object.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'All random terms inside the expectation are measurable under that law.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Those terms are dominated by an integrable envelope or have finite conditional first moments.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e']`
  - Typed unresolved constructs: `['conditional', 'conditional_law']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['The conditioned shock or path has finite support.', 'Every payoff/value term inside the expectation is finite at each support point.', 'The conditioning state or information set is explicitly defined.', 'The conditioning object `x_{t-1}) g_{t,\\theta}(y_t\\mid x_t)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 4 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\pi', '\\rho', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`
- `branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e']`
  - Typed unresolved constructs: `['conditional', 'conditional_law']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['A conditional kernel or probability law is fixed for the random object.', 'All random terms inside the expectation are measurable under that law.', 'Those terms are dominated by an integrable envelope or have finite conditional first moments.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\pi', '\\rho', '\\theta']` whose mathematical types and backend names are not fixed.
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
- `sympy_algebra_attempt` with `sympy`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-joint-integrand` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-joint-integrand > line 1440, add an assumptions paragraph: "For this displayed equality, assume: The conditioned shock or path has finite support. Every payoff/value term inside the expectation is finite at each support point. The conditioning state or information set is explicitly defined. The conditioning object `x_{t-1}) g_{t,\theta}(y_t\mid x_t)` is defined as a sigma-field, information set, state, or conditioning variable for this equality. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
- `patch_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-joint-integrand` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-joint-integrand > line 1440, add an assumptions paragraph: "For this displayed equality, assume: A conditional kernel or probability law is fixed for the random object. All random terms inside the expectation are measurable under that law. Those terms are dominated by an integrable envelope or have finite conditional first moments. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid x_t)`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.

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
- `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\pi', '\\rho', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_finite_state_conditional_expectation_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\pi', '\\rho', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_kernel_integrability_condition_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_conditional_law_defined` (probability_condition)
  Problem: A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Required next evidence: Makes the expectation operator well defined.
- `blocker_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_measurable_integrable_payoff_terms` (integrability_condition)
  Problem: Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Required next evidence: Turns the displayed expression into a finite scalar equality.
- `blocker_semantic_packet_eq_disturbance_joint_integrand_f0bd884aa1c60d2e_conditioning_information_defined` (information_condition)
  Problem: A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Required next evidence: Fixes the scope of the conditional expectation used in the derivation.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

### 2. `eq:disturbance-tangent`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-tangent > line 1463`
- Claim type: `definition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1463, 'line_end': 1465, 'label': 'eq:disturbance-tangent', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 73689, 'end_byte': 73822, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_eb230395c8aada9c1978318ecc59e4d11d1f284d9ee2dda1b709c69aa44c3ba0', 'obligation_digest': 'eb230395c8aada9c1978318ecc59e4d11d1f284d9ee2dda1b709c69aa44c3ba0', 'labels': ['eq:disturbance-tangent'], 'environment': 'align'}`
- Operators: `['equality', 'derivative']`
- Symbols: `{'latex_commands': ['\\dot', '\\theta'], 'bare_identifiers': ['F', 'J', 't', 'u', 'x', 'xF']}`
- Context graph statuses: `{'nearby_stated': 2, 'inferred_candidate': 6}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c`
- Typed obligation status: `needs_assumptions`
- Typed unresolved constructs: `['derivative']`

Source row target:

```tex
\dot x_t&=\partial_\theta F_{t,\theta}(x_{t-1},u_t)
       +J_xF_{t,\theta}(x_{t-1},u_t)\dot x_{t-1}.
 \label{eq:disturbance-tangent}
```

Full display target:

```tex
\dot x_t&=\partial_\theta F_{t,\theta}(x_{t-1},u_t)
       +J_xF_{t,\theta}(x_{t-1},u_t)\dot x_{t-1}.
 \label{eq:disturbance-tangent}
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
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1452-1456']`
- `route_assumption_target_function_is_differentiable_on_the_stated_domain` status `nearby_stated`
  Role: route-required assumption from assumption_discovery
  What: target function is differentiable on the stated domain
  Why status: The condition is stated in the local paragraph/proposition context. This reconciles the low-level route requirement with local source evidence.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1452-1456']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c`
  Diagnostic status: `needs_assumptions`
  Encodability: `{'status': 'candidate', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
  Unresolved constructs: `['derivative']`
  Route hints: `[{'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}]`
  Assumption statuses:
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_tangent_eb230395c8aada9c', 'branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_missing_domain_constraints', 'typed_repair_obligation_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_missing_domain_constraints', 'blocker_formalization_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-tangent > line 1463`
  - Context branch: `branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions [] block constructs ['derivative'].
  - Why this is a derivation problem: No missing typed assumptions were detected by the bounded typed IR builder. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. The source span contains macros `['\\dot', '\\theta']` whose mathematical types and backend names are not fixed. Derivative or gradient notation requires differentiability assumptions.
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_branch_bound_backend_execution_required', 'target_ids': ['branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first'], 'ledger_entry_ids': ['ledger_238dcfa761f60d71b1803ae70fe6669ec4927b266b6d95a156944ea336fd07aa'], 'prerequisites': ['scope_bound:ledger_238dcfa761f60d71b1803ae70fe6669ec4927b266b6d95a156944ea336fd07aa'], 'launch_vetoes': ['ledger_6250dc03d7fa7fbf5dd60721a8686a1252498d28058784ef535b5baaed04d3a2', 'ledger_b9f7a220e3707eac60960a90c8176f928d5b0f0105f9cf3504f13a606d27862f', 'ledger_e27d58a5d9f1e333f4d1f132214e08420d59509418a05e566847f4f556a18f1b'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'branch_bound_backend_execution_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_1e65e87f492e93c04a253cee8fca7eec8c4bb746388d33a8b79d30dffdf8d6ed'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_238dcfa761f60d71b1803ae70fe6669ec4927b266b6d95a156944ea336fd07aa', 'ledger_6250dc03d7fa7fbf5dd60721a8686a1252498d28058784ef535b5baaed04d3a2', 'ledger_b9f7a220e3707eac60960a90c8176f928d5b0f0105f9cf3504f13a606d27862f', 'ledger_e27d58a5d9f1e333f4d1f132214e08420d59509418a05e566847f4f556a18f1b'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c']`
  - Typed unresolved constructs: `['derivative']`
  - Typed encodability: `{'status': 'candidate', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
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
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dot', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Formalize local obligation`: Convert the cited line into a typed obligation before proposing a document edit.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `missing_assumptions`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-tangent` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-tangent > line 1463, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
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
- `blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dot', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_tangent_eb230395c8aada9c_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 3. `eq:disturbance-shock-total-derivative`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-shock-total-derivative > line 1469`
- Claim type: `definition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['conditional_expectation']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1469, 'line_end': 1472, 'label': 'eq:disturbance-shock-total-derivative', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 73917, 'end_byte': 74129, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_8a77d391436af760665264fe29489bdd58f3ef6691626cc40464b2a6cb0c4a6e', 'obligation_digest': '8a77d391436af760665264fe29489bdd58f3ef6691626cc40464b2a6cb0c4a6e', 'labels': ['eq:disturbance-shock-total-derivative'], 'environment': 'equation'}`
- Operators: `['equality', 'conditional_bar', 'derivative']`
- Symbols: `{'latex_commands': ['\\dot', '\\log', '\\mathsf', '\\nabla', '\\theta'], 'bare_identifiers': ['D', 'T', 'r', 't', 'u', 'x']}`
- Context graph statuses: `{'nearby_stated': 3, 'inferred_candidate': 6, 'missing': 3}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['derivative', 'conditional', 'integrability', 'derivative_expectation_interchange', 'conditional_law']`

Source row target:

```tex
D_\theta\log r_{t,\theta}(u_t\mid x_{t-1})
 =\partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1})
  +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1}).
 \label{eq:disturbance-shock-total-derivative}
```

Full display target:

```tex
D_\theta\log r_{t,\theta}(u_t\mid x_{t-1})
 =\partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1})
  +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1}).
 \label{eq:disturbance-shock-total-derivative}
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
- `assumption_relevant_functions_differentiable` status `nearby_stated`
  Role: supports local derivative notation in the FOC route
  What: The relevant functions are differentiable.
  Why status: The condition is stated in the local paragraph/proposition context.
  Required next evidence: Use this as differentiability evidence, but do not treat it as integrability or derivative-expectation interchange evidence.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1452-1456']`
- `requirement_conditional_law_defined` status `nearby_stated`
  Role: well-definedness condition for conditional expectation
  What: A conditional law for the expectation is defined.
  Why status: The condition is stated in nearby local context, not in the target span.
  Required next evidence: Cite or add the transition kernel/probability law used by the conditional expectation.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1469-1472']`
- `requirement_conditional_integrability` status `missing`
  Role: finite-scalar condition for expectation-valued equations
  What: Random terms inside the conditional expectation are measurable and integrable.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: Cite or add measurability and finite conditional first-moment/dominated-envelope conditions.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1469-1472']`
- `requirement_expectation_derivative_interchange` status `missing`
  Role: justifies replacing the derivative of expected continuation value with expected value derivatives
  What: Differentiation may pass through the conditional expectation.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: State a finite-state sum route or a dominated/Leibniz interchange condition for the continuation-value derivatives.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1469-1472']`
- `requirement_choice_independent_transition_law` status `missing`
  Role: rules out omitted transition-kernel derivative terms in the FOC
  What: The conditional law does not add choice-derivative terms.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: State that the conditional law of `z'` given `z` is independent of `k'` and `b'`, or include the missing kernel derivative terms.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1469-1472']`
- `route_assumption_target_function_is_differentiable_on_the_stated_domain` status `nearby_stated`
  Role: route-required assumption from assumption_discovery
  What: target function is differentiable on the stated domain
  Why status: The condition is stated in the local paragraph/proposition context. This reconciles the low-level route requirement with local source evidence.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1452-1456']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization', 'lean'], 'blocked_by_assumption_ids': ['requirement_choice_independent_transition_law', 'requirement_conditional_integrability', 'requirement_expectation_derivative_interchange'], 'unsupported_constructs': ['conditional', 'integrability', 'derivative_expectation_interchange', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['derivative', 'conditional', 'integrability', 'derivative_expectation_interchange', 'conditional_law']`
  Route hints: `[{'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}, {'backend': 'lean', 'suitability': 'formalization_candidate_after_assumptions', 'reason': 'Derivative-under-expectation can only be checked after the interchange theorem assumptions are stated.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `requirement_choice_independent_transition_law` status `missing`: The conditional law does not add choice-derivative terms.
  - `requirement_conditional_integrability` status `missing`: Random terms inside the conditional expectation are measurable and integrable.
  - `requirement_expectation_derivative_interchange` status `missing`: Differentiation may pass through the conditional expectation.
  - `assumption_relevant_functions_differentiable` status `nearby_stated`: The relevant functions are differentiable.
  - `requirement_conditional_law_defined` status `nearby_stated`: A conditional law for the expectation is defined.
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760', 'branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation', 'actionable_abstention:conditional_expectation', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_domain_constraints', 'typed_repair_obligation_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_domain_constraints', 'blocker_formalization_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_lean', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-shock-total-derivative > line 1469`
  - Context branch: `branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation`
  - Context selection authority: `serialization_only_nondominated_context`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['requirement_choice_independent_transition_law', 'requirement_conditional_integrability', 'requirement_expectation_derivative_interchange'] block constructs ['derivative', 'conditional', 'integrability', 'derivative_expectation_interchange', 'conditional_law'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite. The backend needs a finite-state sum route or a dominated/Leibniz interchange condition before this derivation step can be checked.
  - Missing or unresolved assumptions:
    - `requirement_choice_independent_transition_law` status `missing`: The conditional law does not add choice-derivative terms.
    - `requirement_conditional_integrability` status `missing`: Random terms inside the conditional expectation are measurable and integrable.
    - `requirement_expectation_derivative_interchange` status `missing`: Differentiation may pass through the conditional expectation.
  - Candidate assumption set that remains blocked:
    - The conditioned shock or path has finite support.
    - Every payoff/value term inside the expectation is finite at each support point.
    - The conditioning state or information set is explicitly defined.
    - The conditioning object `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})` is defined as a sigma-field, information set, state, or conditioning variable for this equality.
  - Candidate derivation route that remains blocked:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Exact blockers before this can become a repair proposal:
    - `conditioning_scope_translation_required`: The conditional bar has no backend-level conditioning object yet. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `integrability_translation_required`: Integrability of the random payoff/value terms is not established. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `derivative_expectation_interchange_required`: The derivative-under-expectation step is not justified as an encodable theorem instance. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `conditional_law_translation_required`: The conditional law required by the expectation is not stated as an encodable object. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `missing_domain_or_assumption_required`: The branch still has missing or unresolved typed assumptions. Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `missing_domain_or_shape_required`: The backend translation lacks required domain, dimension, or conformability constraints. Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:disturbance-shock-total-derivative > line 1469-1472']`
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition']`
- Unique top branch, only if one exists: `None`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'blocked_for_human_or_formalization_choice', 'target_ids': ['branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition'], 'ledger_entry_ids': ['unresolved_branch_choice'], 'prerequisites': ['resolve_nondominated_branch_choice'], 'launch_vetoes': ['ledger_060e10f3506182133814416072ecce77b57f02e9e35f42169ca26a9980b5228d', 'ledger_1afe8b4ca5ddc928d431a7ca994a61388f36243f52d50b8b25043bfe6c8d41d7', 'ledger_312b0e5db18085386e56288bfbe1e59b721b59b703945b8c39c9911f666d078b', 'ledger_40a723d1214030be9aa9a0ce5f783dc807fc27a8690e8d1286af8dc73a141dbd', 'ledger_5b5fe6c8666c7c8740407b84483fbc90dfdfd2017e197126d0db8d5ffe51c145', 'ledger_687b8b1728d3d0d13f01cbf50104aecf88e74b46e57437169fce6d2f5dfa605a', 'ledger_72a64493f58d5f7223b0f965aa1155a6a7ac796985f993d124297acb0c9571df', 'ledger_72cadafb863b74f70032cc9b90a674f59a904c183e3baa12608bdde4eb9b76da', 'ledger_7f254e9deb0d3f0de31323072b5ffb4edd561c59f787e456a10c64aa9decfc98', 'ledger_818d22733a0da345372bc1c567ef0df7b15b80c13a9c0d04329faad95ea33cc2', 'ledger_898000faecd919722119ea874ca4f86d117b467ac22a2cbcd661e4c7eb08360e', 'ledger_8cbd9c7bdb3114cedef71146de221f26dd1766b295c805e42a7749f9da800705', 'ledger_908f36030b560eaff665cc5494db79b61081711c11505ab63294a80e6c97d21e', 'ledger_a5650e81a9dea2c0d73a81963259b58a338b2bcfbc5171f6c432185ab1cdd91c', 'ledger_b3f162a6010a87dc198c0c3be4af519904dea2553d9c95c2b908a1d2fa936846', 'ledger_dd9b06806215eefa612a91f30b3494abbf5a99983ca26ee6580f9d99f224f2e0', 'ledger_f06f1587961356f09471df2e95c749893894c4c02cb64b0fe6c1f8906550f515', 'ledger_f442af5312447eb8099a7a741959eb2bf325acea8c4e548cac1cda155c15d1b1'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_choice_record', 'schema_version': 'p06_formalization_choice@1', 'binding_fields': ['branch_ids', 'target_id'], 'path_role': 'decision_blocker'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_ed87edbbb31015274fa66491f207abdf424a0eeee7548615c3ade8d127a41312'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_060e10f3506182133814416072ecce77b57f02e9e35f42169ca26a9980b5228d', 'ledger_1afe8b4ca5ddc928d431a7ca994a61388f36243f52d50b8b25043bfe6c8d41d7', 'ledger_7f254e9deb0d3f0de31323072b5ffb4edd561c59f787e456a10c64aa9decfc98', 'ledger_898000faecd919722119ea874ca4f86d117b467ac22a2cbcd661e4c7eb08360e', 'ledger_8cbd9c7bdb3114cedef71146de221f26dd1766b295c805e42a7749f9da800705', 'ledger_908f36030b560eaff665cc5494db79b61081711c11505ab63294a80e6c97d21e', 'ledger_a5650e81a9dea2c0d73a81963259b58a338b2bcfbc5171f6c432185ab1cdd91c', 'ledger_b3f162a6010a87dc198c0c3be4af519904dea2553d9c95c2b908a1d2fa936846', 'ledger_f442af5312447eb8099a7a741959eb2bf325acea8c4e548cac1cda155c15d1b1'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'The conditioned shock or path has finite support.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Every payoff/value term inside the expectation is finite at each support point.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'The conditioning state or information set is explicitly defined.', 'status': 'candidate'}, {'id': 'legacy_assumption_4', 'statement': 'The conditioning object `x_{t-1}) = \\partial_\\theta\\log r_{t,\\theta}(u_t\\mid x_{t-1}) +\\dot x_{t-1}^{\\mathsf T}\\nabla_x\\log r_{t,\\theta}(u_t\\mid x_{t-1})` is defined as a sigma-field, information set, state, or conditioning variable for this equality.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.
- Serialization position `2`: `branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_312b0e5db18085386e56288bfbe1e59b721b59b703945b8c39c9911f666d078b', 'ledger_40a723d1214030be9aa9a0ce5f783dc807fc27a8690e8d1286af8dc73a141dbd', 'ledger_5b5fe6c8666c7c8740407b84483fbc90dfdfd2017e197126d0db8d5ffe51c145', 'ledger_687b8b1728d3d0d13f01cbf50104aecf88e74b46e57437169fce6d2f5dfa605a', 'ledger_72a64493f58d5f7223b0f965aa1155a6a7ac796985f993d124297acb0c9571df', 'ledger_72cadafb863b74f70032cc9b90a674f59a904c183e3baa12608bdde4eb9b76da', 'ledger_818d22733a0da345372bc1c567ef0df7b15b80c13a9c0d04329faad95ea33cc2', 'ledger_dd9b06806215eefa612a91f30b3494abbf5a99983ca26ee6580f9d99f224f2e0', 'ledger_f06f1587961356f09471df2e95c749893894c4c02cb64b0fe6c1f8906550f515'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'A conditional kernel or probability law is fixed for the random object.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'All random terms inside the expectation are measurable under that law.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Those terms are dominated by an integrable envelope or have finite conditional first moments.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760']`
  - Typed unresolved constructs: `['derivative', 'conditional', 'integrability', 'derivative_expectation_interchange', 'conditional_law']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization', 'lean'], 'blocked_by_assumption_ids': ['requirement_choice_independent_transition_law', 'requirement_conditional_integrability', 'requirement_expectation_derivative_interchange'], 'unsupported_constructs': ['conditional', 'integrability', 'derivative_expectation_interchange', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['The conditioned shock or path has finite support.', 'Every payoff/value term inside the expectation is finite at each support point.', 'The conditioning state or information set is explicitly defined.', 'The conditioning object `x_{t-1}) = \\partial_\\theta\\log r_{t,\\theta}(u_t\\mid x_{t-1}) +\\dot x_{t-1}^{\\mathsf T}\\nabla_x\\log r_{t,\\theta}(u_t\\mid x_{t-1})` is defined as a sigma-field, information set, state, or conditioning variable for this equality.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 4 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_integrability_translation_required` (integrability_translation_required): Integrability of the random payoff/value terms is not established.
      Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_derivative_expectation_interchange_required` (derivative_expectation_interchange_required): The derivative-under-expectation step is not justified as an encodable theorem instance.
      Why: The backend needs a finite-state sum route or a dominated/Leibniz interchange condition before this derivation step can be checked.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_choice_independent_transition_law', 'requirement_conditional_integrability', 'requirement_expectation_derivative_interchange']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dot', '\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`
- `branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760']`
  - Typed unresolved constructs: `['derivative', 'conditional', 'integrability', 'derivative_expectation_interchange', 'conditional_law']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization', 'lean'], 'blocked_by_assumption_ids': ['requirement_choice_independent_transition_law', 'requirement_conditional_integrability', 'requirement_expectation_derivative_interchange'], 'unsupported_constructs': ['conditional', 'integrability', 'derivative_expectation_interchange', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['A conditional kernel or probability law is fixed for the random object.', 'All random terms inside the expectation are measurable under that law.', 'Those terms are dominated by an integrable envelope or have finite conditional first moments.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_integrability_translation_required` (integrability_translation_required): Integrability of the random payoff/value terms is not established.
      Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_derivative_expectation_interchange_required` (derivative_expectation_interchange_required): The derivative-under-expectation step is not justified as an encodable theorem instance.
      Why: The backend needs a finite-state sum route or a dominated/Leibniz interchange condition before this derivation step can be checked.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_choice_independent_transition_law', 'requirement_conditional_integrability', 'requirement_expectation_derivative_interchange']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dot', '\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Define conditional law`: Specify the kernel or conditional distribution used by the expectation.
- `Check integrability`: Verify each random payoff, value, or derivative term has a finite conditional expectation.
- `Use expectation as scalar`: Only after those checks should the equality be treated as a scalar derivation step.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `missing_assumptions`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-shock-total-derivative` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-shock-total-derivative > line 1469, add an assumptions paragraph: "For this displayed equality, assume: The conditioned shock or path has finite support. Every payoff/value term inside the expectation is finite at each support point. The conditioning state or information set is explicitly defined. The conditioning object `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})` is defined as a sigma-field, information set, state, or conditioning variable for this equality. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
- `patch_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-shock-total-derivative` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-shock-total-derivative > line 1469, add an assumptions paragraph: "For this displayed equality, assume: A conditional kernel or probability law is fixed for the random object. All random terms inside the expectation are measurable under that law. Those terms are dominated by an integrable envelope or have finite conditional first moments. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) = \partial_\theta\log r_{t,\theta}(u_t\mid x_{t-1}) +\dot x_{t-1}^{\mathsf T}\nabla_x\log r_{t,\theta}(u_t\mid x_{t-1})`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
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
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_integrability_translation_required` (integrability_translation_required)
  Problem: Integrability of the random payoff/value terms is not established.
  Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_derivative_expectation_interchange_required` (derivative_expectation_interchange_required)
  Problem: The derivative-under-expectation step is not justified as an encodable theorem instance.
  Why: The backend needs a finite-state sum route or a dominated/Leibniz interchange condition before this derivation step can be checked.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_choice_independent_transition_law', 'requirement_conditional_integrability', 'requirement_expectation_derivative_interchange']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dot', '\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_finite_state_conditional_expectation_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_integrability_translation_required` (integrability_translation_required)
  Problem: Integrability of the random payoff/value terms is not established.
  Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_derivative_expectation_interchange_required` (derivative_expectation_interchange_required)
  Problem: The derivative-under-expectation step is not justified as an encodable theorem instance.
  Why: The backend needs a finite-state sum route or a dominated/Leibniz interchange condition before this derivation step can be checked.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_choice_independent_transition_law', 'requirement_conditional_integrability', 'requirement_expectation_derivative_interchange']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dot', '\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_kernel_integrability_condition_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_conditional_law_defined` (probability_condition)
  Problem: A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Required next evidence: Makes the expectation operator well defined.
- `blocker_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_measurable_integrable_payoff_terms` (integrability_condition)
  Problem: Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Required next evidence: Turns the displayed expression into a finite scalar equality.
- `blocker_semantic_packet_eq_disturbance_shock_total_derivative_8a77d391436af760_conditioning_information_defined` (information_condition)
  Problem: A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Required next evidence: Fixes the scope of the conditional expectation used in the derivation.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

### 4. `eq:disturbance-score-identity`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-score-identity > line 1492`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['conditional_expectation']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1492, 'line_end': 1494, 'label': 'eq:disturbance-score-identity', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 74794, 'end_byte': 74931, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_d38232597f96b2fa4ce878dbdc0e6d78299254065821c35b6f4bdfbb64b893ee', 'obligation_digest': 'd38232597f96b2fa4ce878dbdc0e6d78299254065821c35b6f4bdfbb64b893ee', 'labels': ['eq:disturbance-score-identity'], 'environment': 'align'}`
- Operators: `['equality', 'conditional_bar']`
- Symbols: `{'latex_commands': ['\\log', '\\mathsf', '\\nabla', '\\theta'], 'bare_identifiers': ['E', 'T', 'Z', 'h', 'y']}`
- Context graph statuses: `{'nearby_stated': 2, 'inferred_candidate': 6, 'missing': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['derivative', 'conditional', 'integrability']`

Source row target:

```tex
\nabla_\theta\log Z(\theta;y)
 &=\mathbb E_\theta\!\left[h_\theta(\mathsf Z;y)\mid y_{1:T}\right],
 \label{eq:disturbance-score-identity}
```

Full display target:

```tex
\nabla_\theta\log Z(\theta;y)
 &=\mathbb E_\theta\!\left[h_\theta(\mathsf Z;y)\mid y_{1:T}\right],
 \label{eq:disturbance-score-identity}
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
- `assumption_relevant_functions_differentiable` status `nearby_stated`
  Role: supports local derivative notation in the FOC route
  What: The relevant functions are differentiable.
  Why status: The condition is stated in the local paragraph/proposition context.
  Required next evidence: Use this as differentiability evidence, but do not treat it as integrability or derivative-expectation interchange evidence.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1500-1521']`
- `requirement_conditional_law_defined` status `nearby_stated`
  Role: well-definedness condition for conditional expectation
  What: A conditional law for the expectation is defined.
  Why status: The condition is stated in nearby local context, not in the target span.
  Required next evidence: Cite or add the transition kernel/probability law used by the conditional expectation.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1492-1494']`
- `requirement_conditional_integrability` status `missing`
  Role: finite-scalar condition for expectation-valued equations
  What: Random terms inside the conditional expectation are measurable and integrable.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: Cite or add measurability and finite conditional first-moment/dominated-envelope conditions.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1492-1494']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_integrability'], 'unsupported_constructs': ['conditional', 'integrability'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['derivative', 'conditional', 'integrability']`
  Route hints: `[{'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `requirement_conditional_integrability` status `missing`: Random terms inside the conditional expectation are measurable and integrable.
  - `assumption_relevant_functions_differentiable` status `nearby_stated`: The relevant functions are differentiable.
  - `requirement_conditional_law_defined` status `nearby_stated`: A conditional law for the expectation is defined.
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa', 'branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation', 'actionable_abstention:conditional_expectation', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_domain_constraints', 'typed_repair_obligation_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_domain_constraints', 'blocker_formalization_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_lean', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-score-identity > line 1492`
  - Context branch: `branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation`
  - Context selection authority: `serialization_only_nondominated_context`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['requirement_conditional_integrability'] block constructs ['derivative', 'conditional', 'integrability'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite. Typed encodability is blocked by `['requirement_conditional_integrability']`.
  - Missing or unresolved assumptions:
    - `requirement_conditional_integrability` status `missing`: Random terms inside the conditional expectation are measurable and integrable.
  - Candidate assumption set that remains blocked:
    - The conditioned shock or path has finite support.
    - Every payoff/value term inside the expectation is finite at each support point.
    - The conditioning state or information set is explicitly defined.
    - The conditioning object `y_{1:T}` is defined as a sigma-field, information set, state, or conditioning variable for this equality.
  - Candidate derivation route that remains blocked:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `y_{1:T}`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `y_{1:T}`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Exact blockers before this can become a repair proposal:
    - `conditioning_scope_translation_required`: The conditional bar has no backend-level conditioning object yet. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `integrability_translation_required`: Integrability of the random payoff/value terms is not established. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `missing_domain_or_assumption_required`: The branch still has missing or unresolved typed assumptions. Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `missing_domain_or_shape_required`: The backend translation lacks required domain, dimension, or conformability constraints. Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: lean stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:disturbance-score-identity > line 1492-1494']`
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition']`
- Unique top branch, only if one exists: `None`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'blocked_for_human_or_formalization_choice', 'target_ids': ['branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition'], 'ledger_entry_ids': ['unresolved_branch_choice'], 'prerequisites': ['resolve_nondominated_branch_choice'], 'launch_vetoes': ['ledger_025a3ca7d25f6795842ed66f1ea55dac1c4fc821868a26a40c52f81f3ba9f77f', 'ledger_14f5e0d8fe493e06bcc693d3e3d129de904ce26d4fb7d29b40a183b0a6e2d74d', 'ledger_2babaf76bfbd2f0c3a3d78202af22da63d4103b6f0838d302cdee61d83966431', 'ledger_3ac2de9951576778c462485a8eb7b016fe71d3a446bbb99c883955197511b8ff', 'ledger_6eb699010d2fbd7e018d5e695aa4a8d627a6621cfa9ed0b9a8857c9f39cc5132', 'ledger_701abaff14ab9a53bc78f46413e7fcfb24cc39921530aa348acf173b894ebfc0', 'ledger_74bdf69fd714af5c0454924e453307381f4c5da49d66cd4f57a4ab4b9c315214', 'ledger_7af6d1f5f8720fb9879220aca1a7fccbd5515d56e8804dc4ec0851afba119e95', 'ledger_96097b518ff5ef690b3da5ffc38874982a4dd53405e4b023c12394d2d8f4d2c8', 'ledger_9c280d87c4bf37432ca9ba7509e5f27b18220f15bb584228bc39b6b96ddf875c', 'ledger_9e68ef65d85b0b069fdff4671b1fd962660fa820b7bdad654ced1a67fbc11218', 'ledger_d962b53458b43cee60690d8811f78d6b55bdc12fb390961d6a812dbad99e5733', 'ledger_e5ec770cf590fe628f9736dd483fc84a13ef1aa4f7701c2e0a383d13a0b3ab81', 'ledger_ecc5bf20fc40556f8ac6678216f8ba13bc7e5f1e9de672976b19d37eda7bd1f2'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_choice_record', 'schema_version': 'p06_formalization_choice@1', 'binding_fields': ['branch_ids', 'target_id'], 'path_role': 'decision_blocker'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_714a8fab6a3d1ee4cdc1e9d34a1049e19293c62f98d5553a2d3becff621862b0'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_025a3ca7d25f6795842ed66f1ea55dac1c4fc821868a26a40c52f81f3ba9f77f', 'ledger_2babaf76bfbd2f0c3a3d78202af22da63d4103b6f0838d302cdee61d83966431', 'ledger_3ac2de9951576778c462485a8eb7b016fe71d3a446bbb99c883955197511b8ff', 'ledger_7af6d1f5f8720fb9879220aca1a7fccbd5515d56e8804dc4ec0851afba119e95', 'ledger_96097b518ff5ef690b3da5ffc38874982a4dd53405e4b023c12394d2d8f4d2c8', 'ledger_9c280d87c4bf37432ca9ba7509e5f27b18220f15bb584228bc39b6b96ddf875c', 'ledger_d962b53458b43cee60690d8811f78d6b55bdc12fb390961d6a812dbad99e5733'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'The conditioned shock or path has finite support.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Every payoff/value term inside the expectation is finite at each support point.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'The conditioning state or information set is explicitly defined.', 'status': 'candidate'}, {'id': 'legacy_assumption_4', 'statement': 'The conditioning object `y_{1:T}` is defined as a sigma-field, information set, state, or conditioning variable for this equality.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.
- Serialization position `2`: `branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_14f5e0d8fe493e06bcc693d3e3d129de904ce26d4fb7d29b40a183b0a6e2d74d', 'ledger_6eb699010d2fbd7e018d5e695aa4a8d627a6621cfa9ed0b9a8857c9f39cc5132', 'ledger_701abaff14ab9a53bc78f46413e7fcfb24cc39921530aa348acf173b894ebfc0', 'ledger_74bdf69fd714af5c0454924e453307381f4c5da49d66cd4f57a4ab4b9c315214', 'ledger_9e68ef65d85b0b069fdff4671b1fd962660fa820b7bdad654ced1a67fbc11218', 'ledger_e5ec770cf590fe628f9736dd483fc84a13ef1aa4f7701c2e0a383d13a0b3ab81', 'ledger_ecc5bf20fc40556f8ac6678216f8ba13bc7e5f1e9de672976b19d37eda7bd1f2'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'A conditional kernel or probability law is fixed for the random object.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'All random terms inside the expectation are measurable under that law.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Those terms are dominated by an integrable envelope or have finite conditional first moments.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa']`
  - Typed unresolved constructs: `['derivative', 'conditional', 'integrability']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_integrability'], 'unsupported_constructs': ['conditional', 'integrability'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['The conditioned shock or path has finite support.', 'Every payoff/value term inside the expectation is finite at each support point.', 'The conditioning state or information set is explicitly defined.', 'The conditioning object `y_{1:T}` is defined as a sigma-field, information set, state, or conditioning variable for this equality.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `y_{1:T}`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `y_{1:T}`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 4 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `y_{1:T}`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `y_{1:T}`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_integrability_translation_required` (integrability_translation_required): Integrability of the random payoff/value terms is not established.
      Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_integrability']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`
- `branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa']`
  - Typed unresolved constructs: `['derivative', 'conditional', 'integrability']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_integrability'], 'unsupported_constructs': ['conditional', 'integrability'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['A conditional kernel or probability law is fixed for the random object.', 'All random terms inside the expectation are measurable under that law.', 'Those terms are dominated by an integrable envelope or have finite conditional first moments.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `y_{1:T}`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `y_{1:T}`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `y_{1:T}`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `y_{1:T}`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_integrability_translation_required` (integrability_translation_required): Integrability of the random payoff/value terms is not established.
      Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_integrability']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Define conditional law`: Specify the kernel or conditional distribution used by the expectation.
- `Check integrability`: Verify each random payoff, value, or derivative term has a finite conditional expectation.
- `Use expectation as scalar`: Only after those checks should the equality be treated as a scalar derivation step.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-score-identity` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-score-identity > line 1492, add an assumptions paragraph: "For this displayed equality, assume: The conditioned shock or path has finite support. Every payoff/value term inside the expectation is finite at each support point. The conditioning state or information set is explicitly defined. The conditioning object `y_{1:T}` is defined as a sigma-field, information set, state, or conditioning variable for this equality. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `y_{1:T}`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `y_{1:T}`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
- `patch_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-score-identity` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-score-identity > line 1492, add an assumptions paragraph: "For this displayed equality, assume: A conditional kernel or probability law is fixed for the random object. All random terms inside the expectation are measurable under that law. Those terms are dominated by an integrable envelope or have finite conditional first moments. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `y_{1:T}`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `y_{1:T}`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.

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
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_integrability_translation_required` (integrability_translation_required)
  Problem: Integrability of the random payoff/value terms is not established.
  Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_integrability']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_finite_state_conditional_expectation_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_integrability_translation_required` (integrability_translation_required)
  Problem: Integrability of the random payoff/value terms is not established.
  Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_integrability']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_kernel_integrability_condition_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_conditional_law_defined` (probability_condition)
  Problem: A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Required next evidence: Makes the expectation operator well defined.
- `blocker_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_measurable_integrable_payoff_terms` (integrability_condition)
  Problem: Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Required next evidence: Turns the displayed expression into a finite scalar equality.
- `blocker_semantic_packet_eq_disturbance_score_identity_d38232597f96b2fa_conditioning_information_defined` (information_condition)
  Problem: A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Required next evidence: Fixes the scope of the conditional expectation used in the derivation.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

### 5. `eq:disturbance-differentiate-integral`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-differentiate-integral > line 1506`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1506, 'line_end': 1507, 'label': 'eq:disturbance-differentiate-integral', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 75344, 'end_byte': 75466, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_c31f523f71e4fb1b4cc89f165514fc0311360d0a1bb2bfc7cad5aced8752c39c', 'obligation_digest': 'c31f523f71e4fb1b4cc89f165514fc0311360d0a1bb2bfc7cad5aced8752c39c', 'labels': ['eq:disturbance-differentiate-integral'], 'environment': 'equation'}`
- Operators: `['equality', 'integral']`
- Symbols: `{'latex_commands': ['\\dd', '\\lambda', '\\nabla', '\\pi', '\\theta'], 'bare_identifiers': ['Z', 'y', 'z']}`
- Context graph statuses: `{'stated': 1, 'inferred_candidate': 4}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b`
- Typed obligation status: `needs_assumptions`
- Typed unresolved constructs: `['derivative']`

Source row target:

```tex
\nabla_\theta Z(\theta;y)=\int \nabla_\theta\pi_\theta(z;y)\,\dd\lambda(z).
 \label{eq:disturbance-differentiate-integral}
```

Full display target:

```tex
\nabla_\theta Z(\theta;y)=\int \nabla_\theta\pi_\theta(z;y)\,\dd\lambda(z).
 \label{eq:disturbance-differentiate-integral}
```

Mathematically missing obligations:
- `formalized_local_obligation` (formalization_condition): A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Closes: Creates the next deterministic target for assumption discovery or proof audit.

Local context graph:
- `assumption_relevant_functions_differentiable` status `stated`
  Role: supports local derivative notation in the FOC route
  What: The relevant functions are differentiable.
  Why status: The condition is stated in the target source span.
  Required next evidence: Use this as differentiability evidence, but do not treat it as integrability or derivative-expectation interchange evidence.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1506-1507']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b`
  Diagnostic status: `needs_assumptions`
  Encodability: `{'status': 'candidate', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
  Unresolved constructs: `['derivative']`
  Route hints: `[{'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}]`
  Assumption statuses:
  - `assumption_relevant_functions_differentiable` status `stated`: The relevant functions are differentiable.
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b', 'branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_missing_domain_constraints', 'typed_repair_obligation_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_missing_domain_constraints', 'blocker_formalization_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-differentiate-integral > line 1506`
  - Context branch: `branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions [] block constructs ['derivative'].
  - Why this is a derivation problem: No missing typed assumptions were detected by the bounded typed IR builder. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. The source span contains macros `['\\dd', '\\lambda', '\\nabla', '\\pi', '\\theta']` whose mathematical types and backend names are not fixed. Derivative or gradient notation requires differentiability assumptions.
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_macro_translation_required', 'target_ids': ['branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first'], 'ledger_entry_ids': ['ledger_3b2f20ea3d02ff184c5faa3f8de65bc04af842ff6057f85b37327e06af9c4c6f'], 'prerequisites': ['scope_bound:ledger_3b2f20ea3d02ff184c5faa3f8de65bc04af842ff6057f85b37327e06af9c4c6f'], 'launch_vetoes': ['ledger_424eb42a78471646dba05d3197f4c23978b8e16ca9237ebdfe78c74a61be097b', 'ledger_f945400cc115307f3f70620602eaa975be911e70e25fd824da407ea9f005fc9c', 'ledger_fa1625c385a8c3f061ee583584f811900d2b8ea6cc0df0584acb016ab641386c'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'macro_translation_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_2fef6e47d7ad31f6ebc7bbb5bd1c12d72ae9a7115f92de598398ade32ec225a1'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_3b2f20ea3d02ff184c5faa3f8de65bc04af842ff6057f85b37327e06af9c4c6f', 'ledger_424eb42a78471646dba05d3197f4c23978b8e16ca9237ebdfe78c74a61be097b', 'ledger_f945400cc115307f3f70620602eaa975be911e70e25fd824da407ea9f005fc9c', 'ledger_fa1625c385a8c3f061ee583584f811900d2b8ea6cc0df0584acb016ab641386c'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b']`
  - Typed unresolved constructs: `['derivative']`
  - Typed encodability: `{'status': 'candidate', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
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
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dd', '\\lambda', '\\nabla', '\\pi', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
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
- `patch_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-differentiate-integral` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-differentiate-integral > line 1506, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
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
- `blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dd', '\\lambda', '\\nabla', '\\pi', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_differentiate_integral_c31f523f71e4fb1b_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 6. `eq:disturbance-proposal-law`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-proposal-law > line 1538`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['conditional_expectation']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1538, 'line_end': 1540, 'label': 'eq:disturbance-proposal-law', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 77029, 'end_byte': 77144, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_2b753379924ee962b5ba801abf0050a6293817a6b0a639f827743eb6afd48643', 'obligation_digest': '2b753379924ee962b5ba801abf0050a6293817a6b0a639f827743eb6afd48643', 'labels': ['eq:disturbance-proposal-law'], 'environment': 'equation'}`
- Operators: `['equality', 'conditional_bar']`
- Symbols: `{'latex_commands': ['\\theta'], 'bare_identifiers': ['Q', 'T', 'q', 't', 'u', 'v', 'x', 'y', 'z']}`
- Context graph statuses: `{'nearby_stated': 1, 'inferred_candidate': 4, 'missing': 2}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['conditional', 'conditional_law', 'integrability']`

Source row target:

```tex
Q_\theta(z)=q_{0,\theta}(v_0)\prod_{t=1}^T
 q_{t,\theta}(u_t\mid x_{t-1},y_t),
 \label{eq:disturbance-proposal-law}
```

Full display target:

```tex
Q_\theta(z)=q_{0,\theta}(v_0)\prod_{t=1}^T
 q_{t,\theta}(u_t\mid x_{t-1},y_t),
 \label{eq:disturbance-proposal-law}
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
- `assumption_relevant_functions_differentiable` status `nearby_stated`
  Role: supports local derivative notation in the FOC route
  What: The relevant functions are differentiable.
  Why status: The condition is stated in the local paragraph/proposition context.
  Required next evidence: Use this as differentiability evidence, but do not treat it as integrability or derivative-expectation interchange evidence.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1563-1578']`
- `requirement_conditional_law_defined` status `missing`
  Role: well-definedness condition for conditional expectation
  What: A conditional law for the expectation is defined.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: Cite or add the transition kernel/probability law used by the conditional expectation.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1538-1540']`
- `requirement_conditional_integrability` status `missing`
  Role: finite-scalar condition for expectation-valued equations
  What: Random terms inside the conditional expectation are measurable and integrable.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: Cite or add measurability and finite conditional first-moment/dominated-envelope conditions.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1538-1540']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_integrability', 'requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law', 'integrability'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['conditional', 'conditional_law', 'integrability']`
  Route hints: `[{'backend': 'lean', 'suitability': 'formalization_candidate', 'reason': 'Typed notation may be formalized manually and checked by Lean.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `requirement_conditional_integrability` status `missing`: Random terms inside the conditional expectation are measurable and integrable.
  - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_proposal_law_2b753379924ee962', 'branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation', 'actionable_abstention:conditional_expectation', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_lean', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-proposal-law > line 1538`
  - Context branch: `branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation`
  - Context selection authority: `serialization_only_nondominated_context`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['requirement_conditional_integrability', 'requirement_conditional_law_defined'] block constructs ['conditional', 'conditional_law', 'integrability'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed. Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
  - Missing or unresolved assumptions:
    - `requirement_conditional_integrability` status `missing`: Random terms inside the conditional expectation are measurable and integrable.
    - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
  - Candidate assumption set that remains blocked:
    - The conditioned shock or path has finite support.
    - Every payoff/value term inside the expectation is finite at each support point.
    - The conditioning state or information set is explicitly defined.
    - The conditioning object `x_{t-1},y_t)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.
  - Candidate derivation route that remains blocked:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1},y_t)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1},y_t)`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Exact blockers before this can become a repair proposal:
    - `conditioning_scope_translation_required`: The conditional bar has no backend-level conditioning object yet. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `conditional_law_translation_required`: The conditional law required by the expectation is not stated as an encodable object. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `integrability_translation_required`: Integrability of the random payoff/value terms is not established. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `missing_domain_or_assumption_required`: The branch still has missing or unresolved typed assumptions. Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: lean stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:disturbance-proposal-law > line 1538-1540']`
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition']`
- Unique top branch, only if one exists: `None`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'blocked_for_human_or_formalization_choice', 'target_ids': ['branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition'], 'ledger_entry_ids': ['unresolved_branch_choice'], 'prerequisites': ['resolve_nondominated_branch_choice'], 'launch_vetoes': ['ledger_01112429a4c85f85148d23fb811e692cc742a17ddcaa26180e8e1030cfa42c2f', 'ledger_28ab3e44af9314918094bd81d687e1e2a8af399bc3bfd744ffdf2fdc0ee48bd9', 'ledger_2ae7466b41df593ba0229d6cb71941d50dae69770399a186499495601ace4251', 'ledger_43ccb53f1a71baa18e4ffe8f2bdae461d95452778a792d5192c10e62c3206c99', 'ledger_588ee9c599d7d819f3d5ab9406869acbfeac14386a620b2215151f522b8550d2', 'ledger_5e6e3c4b2269bd8dbc7339c358d7a43ea59d7966b7bd1acfc6eb7407fcc93547', 'ledger_844541f8e16ab24ddff129c095d5192d983fd78d9b03e7ebea37bcfa959253e4', 'ledger_93f79c34c5891f6812faa23c8bd437ee4100401d0c633eb3008a1002444e493b', 'ledger_ac9a7132150309efbeef58fa4126bd0acc97badacb1cd3f5be7034b6ff714e0e', 'ledger_d21ed8eac6eea18693b59694de2dada498914de326d74f8c2fa5841cf7c5d2a3', 'ledger_dd23a445db3022b84d7e9ac68b039b7f55de6df710b4b8571924807f8f6fdb2c', 'ledger_f75788a41f58de1028882bcee01f9de3fa6dbfafa185d91ebcf7e6eeff588e37', 'ledger_f7a1d51a3a40a711b723437dd42b4e42df0c0a9fd3d1e04b520b0e6ebca97e07', 'ledger_ff368a32f26fd6831b2f0e7429a0809f66ad15869851820d456f7297b121ce2d'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_choice_record', 'schema_version': 'p06_formalization_choice@1', 'binding_fields': ['branch_ids', 'target_id'], 'path_role': 'decision_blocker'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_fd4073134b0157ef9a9b4ea3f3814a9c161729f8a7e7195e384f892ebd98906a'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_2ae7466b41df593ba0229d6cb71941d50dae69770399a186499495601ace4251', 'ledger_43ccb53f1a71baa18e4ffe8f2bdae461d95452778a792d5192c10e62c3206c99', 'ledger_5e6e3c4b2269bd8dbc7339c358d7a43ea59d7966b7bd1acfc6eb7407fcc93547', 'ledger_93f79c34c5891f6812faa23c8bd437ee4100401d0c633eb3008a1002444e493b', 'ledger_d21ed8eac6eea18693b59694de2dada498914de326d74f8c2fa5841cf7c5d2a3', 'ledger_f75788a41f58de1028882bcee01f9de3fa6dbfafa185d91ebcf7e6eeff588e37', 'ledger_f7a1d51a3a40a711b723437dd42b4e42df0c0a9fd3d1e04b520b0e6ebca97e07'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'The conditioned shock or path has finite support.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Every payoff/value term inside the expectation is finite at each support point.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'The conditioning state or information set is explicitly defined.', 'status': 'candidate'}, {'id': 'legacy_assumption_4', 'statement': 'The conditioning object `x_{t-1},y_t)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.
- Serialization position `2`: `branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_01112429a4c85f85148d23fb811e692cc742a17ddcaa26180e8e1030cfa42c2f', 'ledger_28ab3e44af9314918094bd81d687e1e2a8af399bc3bfd744ffdf2fdc0ee48bd9', 'ledger_588ee9c599d7d819f3d5ab9406869acbfeac14386a620b2215151f522b8550d2', 'ledger_844541f8e16ab24ddff129c095d5192d983fd78d9b03e7ebea37bcfa959253e4', 'ledger_ac9a7132150309efbeef58fa4126bd0acc97badacb1cd3f5be7034b6ff714e0e', 'ledger_dd23a445db3022b84d7e9ac68b039b7f55de6df710b4b8571924807f8f6fdb2c', 'ledger_ff368a32f26fd6831b2f0e7429a0809f66ad15869851820d456f7297b121ce2d'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'A conditional kernel or probability law is fixed for the random object.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'All random terms inside the expectation are measurable under that law.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Those terms are dominated by an integrable envelope or have finite conditional first moments.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962']`
  - Typed unresolved constructs: `['conditional', 'conditional_law', 'integrability']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_integrability', 'requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law', 'integrability'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['The conditioned shock or path has finite support.', 'Every payoff/value term inside the expectation is finite at each support point.', 'The conditioning state or information set is explicitly defined.', 'The conditioning object `x_{t-1},y_t)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1},y_t)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1},y_t)`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 4 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1},y_t)`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1},y_t)`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_integrability_translation_required` (integrability_translation_required): Integrability of the random payoff/value terms is not established.
      Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_integrability', 'requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`
- `branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962']`
  - Typed unresolved constructs: `['conditional', 'conditional_law', 'integrability']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_integrability', 'requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law', 'integrability'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['A conditional kernel or probability law is fixed for the random object.', 'All random terms inside the expectation are measurable under that law.', 'Those terms are dominated by an integrable envelope or have finite conditional first moments.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1},y_t)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1},y_t)`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1},y_t)`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1},y_t)`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_integrability_translation_required` (integrability_translation_required): Integrability of the random payoff/value terms is not established.
      Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_integrability', 'requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\theta']` whose mathematical types and backend names are not fixed.
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
- `sympy_algebra_attempt` with `sympy`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-proposal-law` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-proposal-law > line 1538, add an assumptions paragraph: "For this displayed equality, assume: The conditioned shock or path has finite support. Every payoff/value term inside the expectation is finite at each support point. The conditioning state or information set is explicitly defined. The conditioning object `x_{t-1},y_t)` is defined as a sigma-field, information set, state, or conditioning variable for this equality. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1},y_t)`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1},y_t)`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
- `patch_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-proposal-law` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-proposal-law > line 1538, add an assumptions paragraph: "For this displayed equality, assume: A conditional kernel or probability law is fixed for the random object. All random terms inside the expectation are measurable under that law. Those terms are dominated by an integrable envelope or have finite conditional first moments. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1},y_t)`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1},y_t)`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.

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
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_integrability_translation_required` (integrability_translation_required)
  Problem: Integrability of the random payoff/value terms is not established.
  Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_integrability', 'requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_finite_state_conditional_expectation_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_integrability_translation_required` (integrability_translation_required)
  Problem: Integrability of the random payoff/value terms is not established.
  Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_integrability', 'requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_kernel_integrability_condition_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_conditional_law_defined` (probability_condition)
  Problem: A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Required next evidence: Makes the expectation operator well defined.
- `blocker_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_measurable_integrable_payoff_terms` (integrability_condition)
  Problem: Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Required next evidence: Turns the displayed expression into a finite scalar equality.
- `blocker_semantic_packet_eq_disturbance_proposal_law_2b753379924ee962_conditioning_information_defined` (information_condition)
  Problem: A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Required next evidence: Fixes the scope of the conditional expectation used in the derivation.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

### 7. `eq:disturbance-apf-conditional-normalizer`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-apf-conditional-normalizer > line 1605`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['conditional_expectation']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1605, 'line_end': 1609, 'label': 'eq:disturbance-apf-conditional-normalizer', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 79959, 'end_byte': 80188, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_8e65c73848f18ba97f87660a019d2a60fc9def5e67e5c31d7f563d1a24fe4453', 'obligation_digest': '8e65c73848f18ba97f87660a019d2a60fc9def5e67e5c31d7f563d1a24fe4453', 'labels': ['eq:disturbance-apf-conditional-normalizer'], 'environment': 'equation'}`
- Operators: `['equality', 'conditional_bar', 'summation', 'integral']`
- Symbols: `{'latex_commands': ['\\dd', '\\lambda', '\\omega', '\\theta'], 'bare_identifiers': ['E', 'F', 'N', 'W', 'a', 'g', 'i', 'r', 't', 'u', 'x', 'y']}`
- Context graph statuses: `{'inferred_candidate': 5, 'missing': 1, 'nearby_stated': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['expectation', 'conditional', 'conditional_law']`

Source row target:

```tex
\mathbb E[\omega_t^i\mid\mathcal F_{t-1}]
 =\sum_{a=1}^N W_{t-1}^a
   \int r_{t,\theta}(u\mid x_{t-1}^a)
        g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1}^a,u))\,\dd\lambda_t(u).
 \label{eq:disturbance-apf-conditional-normalizer}
```

Full display target:

```tex
\mathbb E[\omega_t^i\mid\mathcal F_{t-1}]
 =\sum_{a=1}^N W_{t-1}^a
   \int r_{t,\theta}(u\mid x_{t-1}^a)
        g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1}^a,u))\,\dd\lambda_t(u).
 \label{eq:disturbance-apf-conditional-normalizer}
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
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1605-1609']`
- `requirement_conditional_integrability` status `nearby_stated`
  Role: finite-scalar condition for expectation-valued equations
  What: Random terms inside the conditional expectation are measurable and integrable.
  Why status: The condition is stated in nearby local context, not in the target span.
  Required next evidence: Cite or add measurability and finite conditional first-moment/dominated-envelope conditions.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1605-1609']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined'], 'unsupported_constructs': ['expectation', 'conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['expectation', 'conditional', 'conditional_law']`
  Route hints: `[{'backend': 'lean', 'suitability': 'formalization_candidate', 'reason': 'Typed notation may be formalized manually and checked by Lean.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9', 'branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation', 'actionable_abstention:conditional_expectation', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_lean', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-apf-conditional-normalizer > line 1605`
  - Context branch: `branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation`
  - Context selection authority: `serialization_only_nondominated_context`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['requirement_conditional_law_defined'] block constructs ['expectation', 'conditional', 'conditional_law'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification. A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  - Missing or unresolved assumptions:
    - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
  - Candidate assumption set that remains blocked:
    - The conditioned shock or path has finite support.
    - Every payoff/value term inside the expectation is finite at each support point.
    - The conditioning state or information set is explicitly defined.
    - The conditioning object `\mathcal F_{t-1}` is defined as a sigma-field, information set, state, or conditioning variable for this equality.
  - Candidate derivation route that remains blocked:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Exact blockers before this can become a repair proposal:
    - `conditional_expectation_translation_required`: The expectation operator cannot be translated as a scalar algebraic expression yet. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `conditioning_scope_translation_required`: The conditional bar has no backend-level conditioning object yet. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `conditional_law_translation_required`: The conditional law required by the expectation is not stated as an encodable object. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `missing_domain_or_assumption_required`: The branch still has missing or unresolved typed assumptions. Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: lean stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:disturbance-apf-conditional-normalizer > line 1605-1609']`
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition']`
- Unique top branch, only if one exists: `None`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'blocked_for_human_or_formalization_choice', 'target_ids': ['branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition'], 'ledger_entry_ids': ['unresolved_branch_choice'], 'prerequisites': ['resolve_nondominated_branch_choice'], 'launch_vetoes': ['ledger_096d80fd7fcf41c70bf69f69efc5fd02a95a3bde6e12dcb120870a18f780b7cb', 'ledger_39f13aa57635eb70f238553949199a573223ce1031fc0f818961bd9cc6630cc4', 'ledger_3f18ab0c7a8c4f6de202cb60933b9480bb1b3e47c2564cd808fb66fe4d02c151', 'ledger_49e456ac8ec461dd117ffc9a07e1fc86be004865763f0b4d7f4bfe6f0520ea46', 'ledger_4ea94077fe2274478bce1a0a8fc12bec643fc8bf0f78beefb85397eb642a43b3', 'ledger_74e97181d5c0cbf5f5abe5ad591e52e4a13a6a850a82b0353ab705b237cbde70', 'ledger_7b1317144a56f207980f6610d25de2bb5fe2553083a045e8f2586c3fb3274aa0', 'ledger_7bf1148ecdc8abeda3d5cf4a0b538c5287d82aa57ae6ba142552fa5807504729', 'ledger_a2af1e58a671c5c71b7729e5dd9dcc4465f8ef6c7d58bb0cdf0b5cf4a0c4f2bf', 'ledger_c65ab17adf641647f7d09bf993fb4d628d6a042e958e878fbbd6fafb101c83f5', 'ledger_e908ee7764462cfc3210c43c22a4c69aaabead11ec748b730be729648e1f8a9d', 'ledger_f46ff9ec6675cfd6a148cf2022a3c56ddb343be05c073c81d455421e7c2630ec', 'ledger_f497a3dbb71ba898f30c31a8c27269252de363ae622df426db2d6493bcc9ce02', 'ledger_f81dcad9d81cc1b2772f523738902612fff403f9c84dc0de71a6c14176131a34'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_choice_record', 'schema_version': 'p06_formalization_choice@1', 'binding_fields': ['branch_ids', 'target_id'], 'path_role': 'decision_blocker'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_da306ba8ed5a408e2121c1bec643f1db232687a10102ae35ff7142f31c16084d'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_096d80fd7fcf41c70bf69f69efc5fd02a95a3bde6e12dcb120870a18f780b7cb', 'ledger_3f18ab0c7a8c4f6de202cb60933b9480bb1b3e47c2564cd808fb66fe4d02c151', 'ledger_49e456ac8ec461dd117ffc9a07e1fc86be004865763f0b4d7f4bfe6f0520ea46', 'ledger_4ea94077fe2274478bce1a0a8fc12bec643fc8bf0f78beefb85397eb642a43b3', 'ledger_7bf1148ecdc8abeda3d5cf4a0b538c5287d82aa57ae6ba142552fa5807504729', 'ledger_a2af1e58a671c5c71b7729e5dd9dcc4465f8ef6c7d58bb0cdf0b5cf4a0c4f2bf', 'ledger_e908ee7764462cfc3210c43c22a4c69aaabead11ec748b730be729648e1f8a9d'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'The conditioned shock or path has finite support.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Every payoff/value term inside the expectation is finite at each support point.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'The conditioning state or information set is explicitly defined.', 'status': 'candidate'}, {'id': 'legacy_assumption_4', 'statement': 'The conditioning object `\\mathcal F_{t-1}` is defined as a sigma-field, information set, state, or conditioning variable for this equality.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.
- Serialization position `2`: `branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_39f13aa57635eb70f238553949199a573223ce1031fc0f818961bd9cc6630cc4', 'ledger_74e97181d5c0cbf5f5abe5ad591e52e4a13a6a850a82b0353ab705b237cbde70', 'ledger_7b1317144a56f207980f6610d25de2bb5fe2553083a045e8f2586c3fb3274aa0', 'ledger_c65ab17adf641647f7d09bf993fb4d628d6a042e958e878fbbd6fafb101c83f5', 'ledger_f46ff9ec6675cfd6a148cf2022a3c56ddb343be05c073c81d455421e7c2630ec', 'ledger_f497a3dbb71ba898f30c31a8c27269252de363ae622df426db2d6493bcc9ce02', 'ledger_f81dcad9d81cc1b2772f523738902612fff403f9c84dc0de71a6c14176131a34'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'A conditional kernel or probability law is fixed for the random object.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'All random terms inside the expectation are measurable under that law.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Those terms are dominated by an integrable envelope or have finite conditional first moments.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9']`
  - Typed unresolved constructs: `['expectation', 'conditional', 'conditional_law']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined'], 'unsupported_constructs': ['expectation', 'conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['The conditioned shock or path has finite support.', 'Every payoff/value term inside the expectation is finite at each support point.', 'The conditioning state or information set is explicitly defined.', 'The conditioning object `\\mathcal F_{t-1}` is defined as a sigma-field, information set, state, or conditioning variable for this equality.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 4 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Split the blocker into an explicit typed obligation and ask for the smallest backend-checkable subclaim.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_expectation_translation_required` (conditional_expectation_translation_required): The expectation operator cannot be translated as a scalar algebraic expression yet.
      Why: A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dd', '\\lambda', '\\omega', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`
- `branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9']`
  - Typed unresolved constructs: `['expectation', 'conditional', 'conditional_law']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined'], 'unsupported_constructs': ['expectation', 'conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['A conditional kernel or probability law is fixed for the random object.', 'All random terms inside the expectation are measurable under that law.', 'Those terms are dominated by an integrable envelope or have finite conditional first moments.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Split the blocker into an explicit typed obligation and ask for the smallest backend-checkable subclaim.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditional_expectation_translation_required` (conditional_expectation_translation_required): The expectation operator cannot be translated as a scalar algebraic expression yet.
      Why: A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dd', '\\lambda', '\\omega', '\\theta']` whose mathematical types and backend names are not fixed.
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
- `sympy_algebra_attempt` with `sympy`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-apf-conditional-normalizer` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-apf-conditional-normalizer > line 1605, add an assumptions paragraph: "For this displayed equality, assume: The conditioned shock or path has finite support. Every payoff/value term inside the expectation is finite at each support point. The conditioning state or information set is explicitly defined. The conditioning object `\mathcal F_{t-1}` is defined as a sigma-field, information set, state, or conditioning variable for this equality. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
- `patch_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-apf-conditional-normalizer` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-apf-conditional-normalizer > line 1605, add an assumptions paragraph: "For this displayed equality, assume: A conditional kernel or probability law is fixed for the random object. All random terms inside the expectation are measurable under that law. Those terms are dominated by an integrable envelope or have finite conditional first moments. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `\mathcal F_{t-1}`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.

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
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_expectation_translation_required` (conditional_expectation_translation_required)
  Problem: The expectation operator cannot be translated as a scalar algebraic expression yet.
  Why: A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dd', '\\lambda', '\\omega', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_finite_state_conditional_expectation_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditional_expectation_translation_required` (conditional_expectation_translation_required)
  Problem: The expectation operator cannot be translated as a scalar algebraic expression yet.
  Why: A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dd', '\\lambda', '\\omega', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_kernel_integrability_condition_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_conditional_law_defined` (probability_condition)
  Problem: A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Required next evidence: Makes the expectation operator well defined.
- `blocker_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_measurable_integrable_payoff_terms` (integrability_condition)
  Problem: Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Required next evidence: Turns the displayed expression into a finite scalar equality.
- `blocker_semantic_packet_eq_disturbance_apf_conditional_normalizer_8e65c73848f18ba9_conditioning_information_defined` (information_condition)
  Problem: A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Required next evidence: Fixes the scope of the conditional expectation used in the derivation.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

### 8. `eq:disturbance-history-kernel`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-history-kernel > line 1638`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['conditional_expectation']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1638, 'line_end': 1642, 'label': 'eq:disturbance-history-kernel', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 81460, 'end_byte': 81665, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_0988992a073e5eb079a62603f27fab42967355fcf969a04bf3d9df07e74edb9f', 'obligation_digest': '0988992a073e5eb079a62603f27fab42967355fcf969a04bf3d9df07e74edb9f', 'labels': ['eq:disturbance-history-kernel'], 'environment': 'equation'}`
- Operators: `['equality', 'conditional_bar', 'integral']`
- Symbols: `{'latex_commands': ['\\dd', '\\lambda', '\\theta', '\\varphi'], 'bare_identifiers': ['F', 'K', 'g', 'r', 't', 'u', 'x', 'y', 'z']}`
- Context graph statuses: `{'inferred_candidate': 5, 'missing': 1, 'nearby_stated': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['conditional', 'conditional_law']`

Source row target:

```tex
(K_{t,\theta}\varphi)(z_{0:t-1})
 =\int r_{t,\theta}(u\mid x_{t-1})
       g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u))
       \varphi(z_{0:t-1},u)\,\dd\lambda_t(u).
 \label{eq:disturbance-history-kernel}
```

Full display target:

```tex
(K_{t,\theta}\varphi)(z_{0:t-1})
 =\int r_{t,\theta}(u\mid x_{t-1})
       g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u))
       \varphi(z_{0:t-1},u)\,\dd\lambda_t(u).
 \label{eq:disturbance-history-kernel}
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
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1638-1642']`
- `requirement_conditional_integrability` status `nearby_stated`
  Role: finite-scalar condition for expectation-valued equations
  What: Random terms inside the conditional expectation are measurable and integrable.
  Why status: The condition is stated in nearby local context, not in the target span.
  Required next evidence: Cite or add measurability and finite conditional first-moment/dominated-envelope conditions.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1638-1642']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['conditional', 'conditional_law']`
  Route hints: `[{'backend': 'lean', 'suitability': 'formalization_candidate', 'reason': 'Typed notation may be formalized manually and checked by Lean.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0', 'branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation', 'actionable_abstention:conditional_expectation', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_lean', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-history-kernel > line 1638`
  - Context branch: `branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation`
  - Context selection authority: `serialization_only_nondominated_context`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['requirement_conditional_law_defined'] block constructs ['conditional', 'conditional_law'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed. Typed encodability is blocked by `['requirement_conditional_law_defined']`.
  - Missing or unresolved assumptions:
    - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
  - Candidate assumption set that remains blocked:
    - The conditioned shock or path has finite support.
    - Every payoff/value term inside the expectation is finite at each support point.
    - The conditioning state or information set is explicitly defined.
    - The conditioning object `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.
  - Candidate derivation route that remains blocked:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`.
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
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:disturbance-history-kernel > line 1638-1642']`
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition']`
- Unique top branch, only if one exists: `None`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'blocked_for_human_or_formalization_choice', 'target_ids': ['branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition'], 'ledger_entry_ids': ['unresolved_branch_choice'], 'prerequisites': ['resolve_nondominated_branch_choice'], 'launch_vetoes': ['ledger_0027d11289e9a9330d0067a105e0e124393000f332e926769ff9681be6c0ce33', 'ledger_14802e2f284c68666bb08192e4bf906b232236f5effa4bf07d39f1a38cf24f73', 'ledger_63bb6c65ea80b5a64d3821edf739fbf0e3fba7fd9be07d6bfab5469d26b3907d', 'ledger_66c7db61d13dbd4b10fc780e8ada3c22c681cb70809e4940f14e575227d70f63', 'ledger_74ec53bf96b0c5eaec804679a916efb13e7bfd611a9e02d7dcd06349be95e0cb', 'ledger_88358beb5f9700554f98daff7fe6b62ec5beb01223874ed46b6863d10d38bf3c', 'ledger_9d6f5fd500bf039a3ed6a5823fd3e677510bbe9b2c615243ce8fd6b5087a9db1', 'ledger_a33a76df58e6df8c4ea34313461ac6b2924cffc91bd5226faff36b7bd8c90803', 'ledger_ab8ec817ad91358626613dfb88446e24a2af1b840c985b51ac116bd76af09247', 'ledger_c1087d3a5739d6245fd2a2b116f82f38fa84137327e29a9b718162c11c5931c0', 'ledger_ddb3d4cc5a139819074138c68f0f2741a8dfbfa222420e1017a2293d11e06ec7', 'ledger_e5bad05bc303113b90c8b1a252979b1b75eb8c1d1a35031b3bc4407dc4ecee06'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_choice_record', 'schema_version': 'p06_formalization_choice@1', 'binding_fields': ['branch_ids', 'target_id'], 'path_role': 'decision_blocker'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_6b556d9d5fea1ae29d3d8fed85df2b2ab41cd43fd19083e43f4f6050047d1108'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_0027d11289e9a9330d0067a105e0e124393000f332e926769ff9681be6c0ce33', 'ledger_14802e2f284c68666bb08192e4bf906b232236f5effa4bf07d39f1a38cf24f73', 'ledger_63bb6c65ea80b5a64d3821edf739fbf0e3fba7fd9be07d6bfab5469d26b3907d', 'ledger_66c7db61d13dbd4b10fc780e8ada3c22c681cb70809e4940f14e575227d70f63', 'ledger_9d6f5fd500bf039a3ed6a5823fd3e677510bbe9b2c615243ce8fd6b5087a9db1', 'ledger_c1087d3a5739d6245fd2a2b116f82f38fa84137327e29a9b718162c11c5931c0'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'The conditioned shock or path has finite support.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Every payoff/value term inside the expectation is finite at each support point.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'The conditioning state or information set is explicitly defined.', 'status': 'candidate'}, {'id': 'legacy_assumption_4', 'statement': 'The conditioning object `x_{t-1}) g_{t,\\theta}(y_t\\mid F_{t,\\theta}(x_{t-1},u)) \\varphi(z_{0:t-1},u)\\,\\dd\\lambda_t(u)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.
- Serialization position `2`: `branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_74ec53bf96b0c5eaec804679a916efb13e7bfd611a9e02d7dcd06349be95e0cb', 'ledger_88358beb5f9700554f98daff7fe6b62ec5beb01223874ed46b6863d10d38bf3c', 'ledger_a33a76df58e6df8c4ea34313461ac6b2924cffc91bd5226faff36b7bd8c90803', 'ledger_ab8ec817ad91358626613dfb88446e24a2af1b840c985b51ac116bd76af09247', 'ledger_ddb3d4cc5a139819074138c68f0f2741a8dfbfa222420e1017a2293d11e06ec7', 'ledger_e5bad05bc303113b90c8b1a252979b1b75eb8c1d1a35031b3bc4407dc4ecee06'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'A conditional kernel or probability law is fixed for the random object.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'All random terms inside the expectation are measurable under that law.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Those terms are dominated by an integrable envelope or have finite conditional first moments.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0']`
  - Typed unresolved constructs: `['conditional', 'conditional_law']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['The conditioned shock or path has finite support.', 'Every payoff/value term inside the expectation is finite at each support point.', 'The conditioning state or information set is explicitly defined.', 'The conditioning object `x_{t-1}) g_{t,\\theta}(y_t\\mid F_{t,\\theta}(x_{t-1},u)) \\varphi(z_{0:t-1},u)\\,\\dd\\lambda_t(u)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 4 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dd', '\\lambda', '\\theta', '\\varphi']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`
- `branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0']`
  - Typed unresolved constructs: `['conditional', 'conditional_law']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['lean', 'human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['A conditional kernel or probability law is fixed for the random object.', 'All random terms inside the expectation are measurable under that law.', 'Those terms are dominated by an integrable envelope or have finite conditional first moments.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dd', '\\lambda', '\\theta', '\\varphi']` whose mathematical types and backend names are not fixed.
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
- `sympy_algebra_attempt` with `sympy`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-history-kernel` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-history-kernel > line 1638, add an assumptions paragraph: "For this displayed equality, assume: The conditioned shock or path has finite support. Every payoff/value term inside the expectation is finite at each support point. The conditioning state or information set is explicitly defined. The conditioning object `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)` is defined as a sigma-field, information set, state, or conditioning variable for this equality. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
- `patch_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-history-kernel` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-history-kernel > line 1638, add an assumptions paragraph: "For this displayed equality, assume: A conditional kernel or probability law is fixed for the random object. All random terms inside the expectation are measurable under that law. Those terms are dominated by an integrable envelope or have finite conditional first moments. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}) g_{t,\theta}(y_t\mid F_{t,\theta}(x_{t-1},u)) \varphi(z_{0:t-1},u)\,\dd\lambda_t(u)`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.

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
- `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dd', '\\lambda', '\\theta', '\\varphi']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_finite_state_conditional_expectation_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dd', '\\lambda', '\\theta', '\\varphi']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_kernel_integrability_condition_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_conditional_law_defined` (probability_condition)
  Problem: A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Required next evidence: Makes the expectation operator well defined.
- `blocker_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_measurable_integrable_payoff_terms` (integrability_condition)
  Problem: Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Required next evidence: Turns the displayed expression into a finite scalar equality.
- `blocker_semantic_packet_eq_disturbance_history_kernel_0988992a073e5eb0_conditioning_information_defined` (information_condition)
  Problem: A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Required next evidence: Fixes the scope of the conditional expectation used in the derivation.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

### 9. `eq:disturbance-particle-unbiasedness`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-particle-unbiasedness > line 1667`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1667, 'line_end': 1668, 'label': 'eq:disturbance-particle-unbiasedness', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 82816, 'end_byte': 82911, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_64f453511b8fc4499e8eaab15cd7647801b1fa6a1c641ca5d056d80de57c7692', 'obligation_digest': '64f453511b8fc4499e8eaab15cd7647801b1fa6a1c641ca5d056d80de57c7692', 'labels': ['eq:disturbance-particle-unbiasedness'], 'environment': 'equation'}`
- Operators: `['equality']`
- Symbols: `{'latex_commands': ['\\gamma', '\\varphi'], 'bare_identifiers': ['E', 'N', 't']}`
- Context graph statuses: `{'inferred_candidate': 5}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['expectation']`

Source row target:

```tex
\mathbb E[\gamma_t^N(\varphi)]=\gamma_t(\varphi).
 \label{eq:disturbance-particle-unbiasedness}
```

Full display target:

```tex
\mathbb E[\gamma_t^N(\varphi)]=\gamma_t(\varphi).
 \label{eq:disturbance-particle-unbiasedness}
```

Mathematically missing obligations:
- `formalized_local_obligation` (formalization_condition): A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Closes: Creates the next deterministic target for assumption discovery or proof audit.

Local context graph:

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449`
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449', 'branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-particle-unbiasedness > line 1667`
  - Context branch: `branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions [] block constructs ['expectation'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target. The source span contains macros `['\\gamma', '\\varphi']` whose mathematical types and backend names are not fixed.
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_formalization_required', 'target_ids': ['branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first'], 'ledger_entry_ids': ['ledger_163989bbe8eddbf7232cd22e21e9b223ce2670f14aa873ba1b1d616a26cd1ac6'], 'prerequisites': ['scope_bound:ledger_163989bbe8eddbf7232cd22e21e9b223ce2670f14aa873ba1b1d616a26cd1ac6'], 'launch_vetoes': ['ledger_645b84d5c43f2db33e3a61199b98f0aa86a68de0a6a5162fafebb62238fb319d', 'ledger_c336221f6a5e6aa9e2ef9b0f8cf1080cd6b528174c758f97a1f4957bed9eb34d', 'ledger_f42dfefdcbb404c43913cb4da53371c868a4077485260d55251d9d881c4a03b1'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_d2bb764c6bfb4bbac24998f41ca5e8fc52bcb6615a9ef8646a02d26d74c1ddae'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_163989bbe8eddbf7232cd22e21e9b223ce2670f14aa873ba1b1d616a26cd1ac6', 'ledger_645b84d5c43f2db33e3a61199b98f0aa86a68de0a6a5162fafebb62238fb319d', 'ledger_c336221f6a5e6aa9e2ef9b0f8cf1080cd6b528174c758f97a1f4957bed9eb34d', 'ledger_f42dfefdcbb404c43913cb4da53371c868a4077485260d55251d9d881c4a03b1'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449']`
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
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_conditional_expectation_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_conditional_expectation_translation_required` (conditional_expectation_translation_required): The expectation operator cannot be translated as a scalar algebraic expression yet.
      Why: A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\gamma', '\\varphi']` whose mathematical types and backend names are not fixed.
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
- `patch_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-particle-unbiasedness` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-particle-unbiasedness > line 1667, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
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
- `blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_conditional_expectation_translation_required` (conditional_expectation_translation_required)
  Problem: The expectation operator cannot be translated as a scalar algebraic expression yet.
  Why: A backend needs the probability law, measurable random terms, and finite expectation before it can encode the target.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\gamma', '\\varphi']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_particle_unbiasedness_64f453511b8fc449_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 10. `eq:disturbance-analytical-score-recursion`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-analytical-score-recursion > line 1720`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['conditional_expectation']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1720, 'line_end': 1727, 'label': 'eq:disturbance-analytical-score-recursion', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 85029, 'end_byte': 85374, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_fbdb3266296546c9938bc10632dd7b8ad5976c6f1248586d3661323c71c9adbb', 'obligation_digest': 'fbdb3266296546c9938bc10632dd7b8ad5976c6f1248586d3661323c71c9adbb', 'labels': ['eq:disturbance-analytical-score-recursion'], 'environment': 'align'}`
- Operators: `['equality', 'conditional_bar', 'derivative']`
- Symbols: `{'latex_commands': ['\\dot', '\\log', '\\mathsf', '\\nabla', '\\theta'], 'bare_identifiers': ['T', 'a', 'g', 'i', 'r', 's', 't', 'u', 'x', 'y']}`
- Context graph statuses: `{'inferred_candidate': 6, 'missing': 4, 'nearby_stated': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['derivative', 'conditional', 'conditional_law', 'derivative_expectation_interchange', 'differentiability']`

Source row target:

```tex
s_t^i={}&s_{t-1}^a
 +\partial_\theta\log r_{t,\theta}(u_t^i\mid x_{t-1}^a)
 +(\dot x_{t-1}^a)^{\mathsf T}
          \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a)
 \nonumber\\
 &+\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i)
 +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i).
 \label{eq:disturbance-analytical-score-recursion}
```

Full display target:

```tex
s_t^i={}&s_{t-1}^a
 +\partial_\theta\log r_{t,\theta}(u_t^i\mid x_{t-1}^a)
 +(\dot x_{t-1}^a)^{\mathsf T}
          \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a)
 \nonumber\\
 &+\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i)
 +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i).
 \label{eq:disturbance-analytical-score-recursion}
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
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1720-1727']`
- `requirement_conditional_integrability` status `nearby_stated`
  Role: finite-scalar condition for expectation-valued equations
  What: Random terms inside the conditional expectation are measurable and integrable.
  Why status: The condition is stated in nearby local context, not in the target span.
  Required next evidence: Cite or add measurability and finite conditional first-moment/dominated-envelope conditions.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1720-1727']`
- `requirement_expectation_derivative_interchange` status `missing`
  Role: justifies replacing the derivative of expected continuation value with expected value derivatives
  What: Differentiation may pass through the conditional expectation.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: State a finite-state sum route or a dominated/Leibniz interchange condition for the continuation-value derivatives.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1720-1727']`
- `requirement_choice_independent_transition_law` status `missing`
  Role: rules out omitted transition-kernel derivative terms in the FOC
  What: The conditional law does not add choice-derivative terms.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: State that the conditional law of `z'` given `z` is independent of `k'` and `b'`, or include the missing kernel derivative terms.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1720-1727']`
- `route_assumption_target_function_is_differentiable_on_the_stated_domain` status `missing`
  Role: route-required assumption from assumption_discovery
  What: target function is differentiable on the stated domain
  Why status: The low-level route detector marked this assumption as missing.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1720-1727']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization', 'lean'], 'blocked_by_assumption_ids': ['requirement_choice_independent_transition_law', 'requirement_conditional_law_defined', 'requirement_expectation_derivative_interchange', 'route_assumption_target_function_is_differentiable_on_the_stated_domain'], 'unsupported_constructs': ['conditional', 'conditional_law', 'derivative_expectation_interchange'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['derivative', 'conditional', 'conditional_law', 'derivative_expectation_interchange', 'differentiability']`
  Route hints: `[{'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}, {'backend': 'lean', 'suitability': 'formalization_candidate_after_assumptions', 'reason': 'Derivative-under-expectation can only be checked after the interchange theorem assumptions are stated.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `requirement_choice_independent_transition_law` status `missing`: The conditional law does not add choice-derivative terms.
  - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
  - `requirement_expectation_derivative_interchange` status `missing`: Differentiation may pass through the conditional expectation.
  - `route_assumption_target_function_is_differentiable_on_the_stated_domain` status `missing`: target function is differentiable on the stated domain
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9', 'branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation', 'actionable_abstention:conditional_expectation', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_domain_constraints', 'typed_repair_obligation_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_domain_constraints', 'blocker_formalization_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_lean', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-analytical-score-recursion > line 1720`
  - Context branch: `branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation`
  - Context selection authority: `serialization_only_nondominated_context`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['requirement_choice_independent_transition_law', 'requirement_conditional_law_defined', 'requirement_expectation_derivative_interchange', 'route_assumption_target_function_is_differentiable_on_the_stated_domain'] block constructs ['derivative', 'conditional', 'conditional_law', 'derivative_expectation_interchange', 'differentiability'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed. The backend needs a finite-state sum route or a dominated/Leibniz interchange condition before this derivation step can be checked.
  - Missing or unresolved assumptions:
    - `requirement_choice_independent_transition_law` status `missing`: The conditional law does not add choice-derivative terms.
    - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
    - `requirement_expectation_derivative_interchange` status `missing`: Differentiation may pass through the conditional expectation.
    - `route_assumption_target_function_is_differentiable_on_the_stated_domain` status `missing`: target function is differentiable on the stated domain
  - Candidate assumption set that remains blocked:
    - The conditioned shock or path has finite support.
    - Every payoff/value term inside the expectation is finite at each support point.
    - The conditioning state or information set is explicitly defined.
    - The conditioning object `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.
  - Candidate derivation route that remains blocked:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Exact blockers before this can become a repair proposal:
    - `conditioning_scope_translation_required`: The conditional bar has no backend-level conditioning object yet. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `conditional_law_translation_required`: The conditional law required by the expectation is not stated as an encodable object. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `derivative_expectation_interchange_required`: The derivative-under-expectation step is not justified as an encodable theorem instance. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `missing_domain_or_assumption_required`: The branch still has missing or unresolved typed assumptions. Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `missing_domain_or_shape_required`: The backend translation lacks required domain, dimension, or conformability constraints. Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:disturbance-analytical-score-recursion > line 1720-1727']`
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition']`
- Unique top branch, only if one exists: `None`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'blocked_for_human_or_formalization_choice', 'target_ids': ['branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition'], 'ledger_entry_ids': ['unresolved_branch_choice'], 'prerequisites': ['resolve_nondominated_branch_choice'], 'launch_vetoes': ['ledger_09316758dcc84dd69be0cb48c9171014d418514f4de4f0bfc38a0cdf059f783d', 'ledger_10b50c31fa846eb42058574dad72e4b15b3b201801a4a77bd8b8db08abf2a66f', 'ledger_1fa72878ce9a2d7d8a5a69726654988067dbf5c2a73cf55f5458eef306f0eeae', 'ledger_217961c995177929764ef7c92c8b6a1820568054f15ae41f9a1a3bf86ce63368', 'ledger_2675d9fd7c5e6ed560316fef79087774780b1b36a3218921b7e7a3a9ec57bb03', 'ledger_388f12794f91cee650a4c073297ffa1de9266acb579749c63d0fb5dae501dae2', 'ledger_46768a9265097da2cd96d916fb1e819f0120b1133421a34f1936909fcbd50697', 'ledger_733bf44b945ffe92f45dd779e06e4759efdd90f7d4febacaab9203a48261f43d', 'ledger_7a65469d0f424968d6bfebb161167f7c1b83960491d6cdab07cb16a63dbd1c21', 'ledger_8084a0d05e044f82ece6288e7fb68d621105ceeb996f375763cc301efde7c35c', 'ledger_8095f604dcd59521aadfcb0cfabc039a2ad724d07cf3cb8d86df92a5336fb34e', 'ledger_896cd7dc01d6c30382ace6a613952fdcc255c6e9c18152ca97b3ddf1632199c2', 'ledger_8edcb40b12e0759577e8ace6de74a582ea88ed09e01304a40ab29f53cc9e2738', 'ledger_bd41954f78245b002e1c0f1e7832f7d78e62d4b2874f2c1a853ce1a19912ef40', 'ledger_da50266f92c1bddccf6f4e06a492749749cd1772f48577d270fc57ba85c8a077', 'ledger_f04ae311e66de1d99fbbe2468ed862b8bb8e024ad1b7f635628b9320ba7f7f9f'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_choice_record', 'schema_version': 'p06_formalization_choice@1', 'binding_fields': ['branch_ids', 'target_id'], 'path_role': 'decision_blocker'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_8a70f5e9b692e844029074e015563578d3cb5559d7a50f40e7a0c4758e84b910'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_10b50c31fa846eb42058574dad72e4b15b3b201801a4a77bd8b8db08abf2a66f', 'ledger_1fa72878ce9a2d7d8a5a69726654988067dbf5c2a73cf55f5458eef306f0eeae', 'ledger_2675d9fd7c5e6ed560316fef79087774780b1b36a3218921b7e7a3a9ec57bb03', 'ledger_733bf44b945ffe92f45dd779e06e4759efdd90f7d4febacaab9203a48261f43d', 'ledger_8084a0d05e044f82ece6288e7fb68d621105ceeb996f375763cc301efde7c35c', 'ledger_8edcb40b12e0759577e8ace6de74a582ea88ed09e01304a40ab29f53cc9e2738', 'ledger_da50266f92c1bddccf6f4e06a492749749cd1772f48577d270fc57ba85c8a077', 'ledger_f04ae311e66de1d99fbbe2468ed862b8bb8e024ad1b7f635628b9320ba7f7f9f'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'The conditioned shock or path has finite support.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Every payoff/value term inside the expectation is finite at each support point.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'The conditioning state or information set is explicitly defined.', 'status': 'candidate'}, {'id': 'legacy_assumption_4', 'statement': 'The conditioning object `x_{t-1}^a) +(\\dot x_{t-1}^a)^{\\mathsf T} \\nabla_x\\log r_{t,\\theta}(u_t^i\\mid x_{t-1}^a) +\\partial_\\theta\\log g_{t,\\theta}(y_t\\mid x_t^i) +(\\dot x_t^i)^{\\mathsf T}\\nabla_x\\log g_{t,\\theta}(y_t\\mid x_t^i)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.
- Serialization position `2`: `branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_09316758dcc84dd69be0cb48c9171014d418514f4de4f0bfc38a0cdf059f783d', 'ledger_217961c995177929764ef7c92c8b6a1820568054f15ae41f9a1a3bf86ce63368', 'ledger_388f12794f91cee650a4c073297ffa1de9266acb579749c63d0fb5dae501dae2', 'ledger_46768a9265097da2cd96d916fb1e819f0120b1133421a34f1936909fcbd50697', 'ledger_7a65469d0f424968d6bfebb161167f7c1b83960491d6cdab07cb16a63dbd1c21', 'ledger_8095f604dcd59521aadfcb0cfabc039a2ad724d07cf3cb8d86df92a5336fb34e', 'ledger_896cd7dc01d6c30382ace6a613952fdcc255c6e9c18152ca97b3ddf1632199c2', 'ledger_bd41954f78245b002e1c0f1e7832f7d78e62d4b2874f2c1a853ce1a19912ef40'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'A conditional kernel or probability law is fixed for the random object.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'All random terms inside the expectation are measurable under that law.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Those terms are dominated by an integrable envelope or have finite conditional first moments.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9']`
  - Typed unresolved constructs: `['derivative', 'conditional', 'conditional_law', 'derivative_expectation_interchange', 'differentiability']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization', 'lean'], 'blocked_by_assumption_ids': ['requirement_choice_independent_transition_law', 'requirement_conditional_law_defined', 'requirement_expectation_derivative_interchange', 'route_assumption_target_function_is_differentiable_on_the_stated_domain'], 'unsupported_constructs': ['conditional', 'conditional_law', 'derivative_expectation_interchange'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['The conditioned shock or path has finite support.', 'Every payoff/value term inside the expectation is finite at each support point.', 'The conditioning state or information set is explicitly defined.', 'The conditioning object `x_{t-1}^a) +(\\dot x_{t-1}^a)^{\\mathsf T} \\nabla_x\\log r_{t,\\theta}(u_t^i\\mid x_{t-1}^a) +\\partial_\\theta\\log g_{t,\\theta}(y_t\\mid x_t^i) +(\\dot x_t^i)^{\\mathsf T}\\nabla_x\\log g_{t,\\theta}(y_t\\mid x_t^i)` is defined as a sigma-field, information set, state, or conditioning variable for this equality.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 4 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_derivative_expectation_interchange_required` (derivative_expectation_interchange_required): The derivative-under-expectation step is not justified as an encodable theorem instance.
      Why: The backend needs a finite-state sum route or a dominated/Leibniz interchange condition before this derivation step can be checked.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_choice_independent_transition_law', 'requirement_conditional_law_defined', 'requirement_expectation_derivative_interchange', 'route_assumption_target_function_is_differentiable_on_the_stated_domain']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dot', '\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`
- `branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9']`
  - Typed unresolved constructs: `['derivative', 'conditional', 'conditional_law', 'derivative_expectation_interchange', 'differentiability']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization', 'lean'], 'blocked_by_assumption_ids': ['requirement_choice_independent_transition_law', 'requirement_conditional_law_defined', 'requirement_expectation_derivative_interchange', 'route_assumption_target_function_is_differentiable_on_the_stated_domain'], 'unsupported_constructs': ['conditional', 'conditional_law', 'derivative_expectation_interchange'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['A conditional kernel or probability law is fixed for the random object.', 'All random terms inside the expectation are measurable under that law.', 'Those terms are dominated by an integrable envelope or have finite conditional first moments.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_derivative_expectation_interchange_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_derivative_expectation_interchange_required` (derivative_expectation_interchange_required): The derivative-under-expectation step is not justified as an encodable theorem instance.
      Why: The backend needs a finite-state sum route or a dominated/Leibniz interchange condition before this derivation step can be checked.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_choice_independent_transition_law', 'requirement_conditional_law_defined', 'requirement_expectation_derivative_interchange', 'route_assumption_target_function_is_differentiable_on_the_stated_domain']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dot', '\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['derivative/interchange step requires differentiability and domain formalization', 'LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Define conditional law`: Specify the kernel or conditional distribution used by the expectation.
- `Check integrability`: Verify each random payoff, value, or derivative term has a finite conditional expectation.
- `Use expectation as scalar`: Only after those checks should the equality be treated as a scalar derivation step.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `missing_assumptions`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-analytical-score-recursion` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-analytical-score-recursion > line 1720, add an assumptions paragraph: "For this displayed equality, assume: The conditioned shock or path has finite support. Every payoff/value term inside the expectation is finite at each support point. The conditioning state or information set is explicitly defined. The conditioning object `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)` is defined as a sigma-field, information set, state, or conditioning variable for this equality. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
- `patch_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-analytical-score-recursion` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-analytical-score-recursion > line 1720, add an assumptions paragraph: "For this displayed equality, assume: A conditional kernel or probability law is fixed for the random object. All random terms inside the expectation are measurable under that law. Those terms are dominated by an integrable envelope or have finite conditional first moments. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `x_{t-1}^a) +(\dot x_{t-1}^a)^{\mathsf T} \nabla_x\log r_{t,\theta}(u_t^i\mid x_{t-1}^a) +\partial_\theta\log g_{t,\theta}(y_t\mid x_t^i) +(\dot x_t^i)^{\mathsf T}\nabla_x\log g_{t,\theta}(y_t\mid x_t^i)`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
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
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_derivative_expectation_interchange_required` (derivative_expectation_interchange_required)
  Problem: The derivative-under-expectation step is not justified as an encodable theorem instance.
  Why: The backend needs a finite-state sum route or a dominated/Leibniz interchange condition before this derivation step can be checked.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_choice_independent_transition_law', 'requirement_conditional_law_defined', 'requirement_expectation_derivative_interchange', 'route_assumption_target_function_is_differentiable_on_the_stated_domain']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dot', '\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_finite_state_conditional_expectation_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_derivative_expectation_interchange_required` (derivative_expectation_interchange_required)
  Problem: The derivative-under-expectation step is not justified as an encodable theorem instance.
  Why: The backend needs a finite-state sum route or a dominated/Leibniz interchange condition before this derivation step can be checked.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_choice_independent_transition_law', 'requirement_conditional_law_defined', 'requirement_expectation_derivative_interchange', 'route_assumption_target_function_is_differentiable_on_the_stated_domain']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dot', '\\log', '\\mathsf', '\\nabla', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: derivative/interchange step requires differentiability and domain formalization; LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_kernel_integrability_condition_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_conditional_law_defined` (probability_condition)
  Problem: A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Required next evidence: Makes the expectation operator well defined.
- `blocker_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_measurable_integrable_payoff_terms` (integrability_condition)
  Problem: Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Required next evidence: Turns the displayed expression into a finite scalar equality.
- `blocker_semantic_packet_eq_disturbance_analytical_score_recursion_fbdb3266296546c9_conditioning_information_defined` (information_condition)
  Problem: A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Required next evidence: Fixes the scope of the conditional expectation used in the derivation.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

### 11. `eq:control-known-integral`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:control-known-integral > line 1788`
- Claim type: `definition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1788, 'line_end': 1789, 'label': 'eq:control-known-integral', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 88485, 'end_byte': 88575, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_563a189f1ff7ab591ecb37cbedc41c27ea0cd57ee4f4c759c1b4226f8ad1e4b9', 'obligation_digest': '563a189f1ff7ab591ecb37cbedc41c27ea0cd57ee4f4c759c1b4226f8ad1e4b9', 'labels': ['eq:control-known-integral'], 'environment': 'equation'}`
- Operators: `['equality', 'integral']`
- Symbols: `{'latex_commands': ['\\dd', '\\lambda', '\\pi', '\\theta'], 'bare_identifiers': ['C', 'c', 'y', 'z']}`
- Context graph statuses: `{'nearby_stated': 1, 'inferred_candidate': 3}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_control_known_integral_563a189f1ff7ab59`
- Typed obligation status: `ready_for_backend`
- Typed unresolved constructs: `[]`

Source row target:

```tex
C(\theta)=\int\pi_\theta(z;y)c_\theta(z)\,\dd\lambda(z)
 \label{eq:control-known-integral}
```

Full display target:

```tex
C(\theta)=\int\pi_\theta(z;y)c_\theta(z)\,\dd\lambda(z)
 \label{eq:control-known-integral}
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
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1770-1779', 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1807-1830']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_control_known_integral_563a189f1ff7ab59`
  Diagnostic status: `ready_for_backend`
  Encodability: `{'status': 'candidate', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
  Unresolved constructs: `[]`
  Route hints: `[{'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}]`
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
  - `document_gap_report_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_control_known_integral_563a189f1ff7ab59', 'branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_control_known_integral_563a189f1ff7ab59']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:control-known-integral > line 1788`
  - Context branch: `branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target remains diagnostic because no branch has enough evidence for certification.
  - Why this is a derivation problem: No missing typed assumptions were detected by the bounded typed IR builder. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. The source span contains macros `['\\dd', '\\lambda', '\\pi', '\\theta']` whose mathematical types and backend names are not fixed.
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
- Nondominated branches: `['branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_macro_translation_required', 'target_ids': ['branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first'], 'ledger_entry_ids': ['ledger_2102b64031de1a51b06ac98efac5ff3d346c176c11d4cc10d3d52f122cba295e'], 'prerequisites': ['scope_bound:ledger_2102b64031de1a51b06ac98efac5ff3d346c176c11d4cc10d3d52f122cba295e'], 'launch_vetoes': ['ledger_dfb9afb768523fd21ea3672e3500743f56a1791c35b377591e0b243def2385d5', 'ledger_ea7bdf1afcca5c3d2194e62658f25ebafb6901828a28df06d1ff372c45b21bff'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'macro_translation_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_5cdfc0b503096f026a26255bc6806b70069995464006534527cc1e15cd5cc5fd'}`
- Serialization position `1`: `branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_2102b64031de1a51b06ac98efac5ff3d346c176c11d4cc10d3d52f122cba295e', 'ledger_dfb9afb768523fd21ea3672e3500743f56a1791c35b377591e0b243def2385d5', 'ledger_ea7bdf1afcca5c3d2194e62658f25ebafb6901828a28df06d1ff372c45b21bff'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_control_known_integral_563a189f1ff7ab59']`
  - Typed unresolved constructs: `[]`
  - Typed encodability: `{'status': 'candidate', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
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
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\dd', '\\lambda', '\\pi', '\\theta']` whose mathematical types and backend names are not fixed.
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
- `patch_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:control-known-integral` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:control-known-integral > line 1788, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
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
- `blocker_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\dd', '\\lambda', '\\pi', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_control_known_integral_563a189f1ff7ab59_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 12. `eq:control-variate-coefficient`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:control-variate-coefficient > line 1801`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1801, 'line_end': 1803, 'label': 'eq:control-variate-coefficient', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 89047, 'end_byte': 89173, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_a64a9033c276902e7a6b30517a9ee45f5393127271cbce446f16aa3418cfc16a', 'obligation_digest': 'a64a9033c276902e7a6b30517a9ee45f5393127271cbce446f16aa3418cfc16a', 'labels': ['eq:control-variate-coefficient'], 'environment': 'equation'}`
- Operators: `['equality']`
- Symbols: `{'latex_commands': ['\\beta'], 'bare_identifiers': ['A', 'B', 'Cov', 'N', 'Var']}`
- Context graph statuses: `{'nearby_stated': 1, 'inferred_candidate': 3, 'missing': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['route_assumption_denominator_is_nonzero']`

Source row target:

```tex
\beta^*=\frac{\operatorname{Cov}(A_N,B_N)}
                 {\operatorname{Var}(B_N)}.
 \label{eq:control-variate-coefficient}
```

Full display target:

```tex
\beta^*=\frac{\operatorname{Cov}(A_N,B_N)}
                 {\operatorname{Var}(B_N)}.
 \label{eq:control-variate-coefficient}
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
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1770-1779', 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1807-1830']`
- `route_assumption_denominator_is_nonzero` status `missing`
  Role: route-required assumption from assumption_discovery
  What: denominator is nonzero
  Why status: The low-level route detector marked this assumption as missing.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1801-1803']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e`
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
  - `document_gap_report_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_control_variate_coefficient_a64a9033c276902e', 'branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:control-variate-coefficient > line 1801`
  - Context branch: `branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['route_assumption_denominator_is_nonzero'] block constructs ['route_assumption_denominator_is_nonzero'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`. The source span contains macros `['\\beta']` whose mathematical types and backend names are not fixed.
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
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:control-variate-coefficient > line 1801-1803']`
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
- Nondominated branches: `['branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_formalization_required', 'target_ids': ['branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first'], 'ledger_entry_ids': ['ledger_1ce795a4e4c5d2e2faa28d2b1accbf61c41873ba57937f6c9a871681d3f89661'], 'prerequisites': ['scope_bound:ledger_1ce795a4e4c5d2e2faa28d2b1accbf61c41873ba57937f6c9a871681d3f89661'], 'launch_vetoes': ['ledger_253c5a12eafa91d321377d74453f22634b25c3d875b2eeec75a5d28d9c861cc9', 'ledger_7c0a8fe4d6d04c29f8c6c74e9076a05e42eb13dd8fbaa08a2c1bbeebbfcff741', 'ledger_dff40f83307a02aec853d399fafb104e5a3cba77ebba275a2027f1fbdb469af8'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_45fce2989640a8e88e1750cce3bcf23b67c2becd77f2f730ddad0fdd971d0a08'}`
- Serialization position `1`: `branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_1ce795a4e4c5d2e2faa28d2b1accbf61c41873ba57937f6c9a871681d3f89661', 'ledger_253c5a12eafa91d321377d74453f22634b25c3d875b2eeec75a5d28d9c861cc9', 'ledger_7c0a8fe4d6d04c29f8c6c74e9076a05e42eb13dd8fbaa08a2c1bbeebbfcff741', 'ledger_dff40f83307a02aec853d399fafb104e5a3cba77ebba275a2027f1fbdb469af8'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e']`
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
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\beta']` whose mathematical types and backend names are not fixed.
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
- `patch_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:control-variate-coefficient` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:control-variate-coefficient > line 1801, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
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
- `blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\beta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_control_variate_coefficient_a64a9033c276902e_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 13. `eq:control-variate-variance`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:control-variate-variance > line 1818`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1818, 'line_end': 1821, 'label': 'eq:control-variate-variance', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 89602, 'end_byte': 89784, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_70c545c55a329e563c4ea2a758a921b81cf80447f057cfcea354385559851161', 'obligation_digest': '70c545c55a329e563c4ea2a758a921b81cf80447f057cfcea354385559851161', 'labels': ['eq:control-variate-variance'], 'environment': 'equation'}`
- Operators: `['equality']`
- Symbols: `{'latex_commands': ['\\beta', '\\rm'], 'bare_identifiers': ['A', 'B', 'Cov', 'D', 'N', 'Var', 'cv', 'v']}`
- Context graph statuses: `{'nearby_stated': 1, 'inferred_candidate': 3}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_control_variate_variance_70c545c55a329e56`
- Typed obligation status: `ready_for_backend`
- Typed unresolved constructs: `[]`

Source row target:

```tex
\operatorname{Var}(\widehat D_{\rm cv}^{\,v})
 =\operatorname{Var}(A_N)-2\beta\operatorname{Cov}(A_N,B_N)
       +\beta^2\operatorname{Var}(B_N).
 \label{eq:control-variate-variance}
```

Full display target:

```tex
\operatorname{Var}(\widehat D_{\rm cv}^{\,v})
 =\operatorname{Var}(A_N)-2\beta\operatorname{Cov}(A_N,B_N)
       +\beta^2\operatorname{Var}(B_N).
 \label{eq:control-variate-variance}
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
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1807-1830']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_control_variate_variance_70c545c55a329e56`
  Diagnostic status: `ready_for_backend`
  Encodability: `{'status': 'candidate', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
  Unresolved constructs: `[]`
  Route hints: `[{'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}]`
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
  - `document_gap_report_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_control_variate_variance_70c545c55a329e56', 'branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_control_variate_variance_70c545c55a329e56']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:control-variate-variance > line 1818`
  - Context branch: `branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target remains diagnostic because no branch has enough evidence for certification.
  - Why this is a derivation problem: No missing typed assumptions were detected by the bounded typed IR builder. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. The source span contains macros `['\\beta', '\\rm']` whose mathematical types and backend names are not fixed.
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
- Nondominated branches: `['branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_formalization_required', 'target_ids': ['branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first'], 'ledger_entry_ids': ['ledger_2e186387c191e8c20f46988c7cdc655bfb96f28f3f24b20be718341cc5affa1a'], 'prerequisites': ['scope_bound:ledger_2e186387c191e8c20f46988c7cdc655bfb96f28f3f24b20be718341cc5affa1a'], 'launch_vetoes': ['ledger_356e2fd02c0be5e698704e4df070c75f653c682d4cb852eb0ed185b3398bdfa1', 'ledger_3875ed760c36415208ca0ef9e1b536c61f54aac452186edd23947e60e005ada3'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_91537a5399853f676a68a07d41041cfe648d839daed28effbd1a23b1115b5725'}`
- Serialization position `1`: `branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_2e186387c191e8c20f46988c7cdc655bfb96f28f3f24b20be718341cc5affa1a', 'ledger_356e2fd02c0be5e698704e4df070c75f653c682d4cb852eb0ed185b3398bdfa1', 'ledger_3875ed760c36415208ca0ef9e1b536c61f54aac452186edd23947e60e005ada3'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_control_variate_variance_70c545c55a329e56']`
  - Typed unresolved constructs: `[]`
  - Typed encodability: `{'status': 'candidate', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': [], 'unsupported_constructs': [], 'why': 'No missing typed assumptions were detected by the bounded typed IR builder.'}`
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
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\beta', '\\rm']` whose mathematical types and backend names are not fixed.
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
- `patch_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:control-variate-variance` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:control-variate-variance > line 1818, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
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
- `blocker_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\beta', '\\rm']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_control_variate_variance_70c545c55a329e56_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_control_variate_variance_70c545c55a329e56_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 14. `eq:disturbance-reference-residual`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-reference-residual > line 1877`
- Claim type: `statistical_estimator`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1877, 'line_end': 1884, 'label': 'eq:disturbance-reference-residual', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 92754, 'end_byte': 93080, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_547be8b0ca24a57ee4f338a2def0a58d40c72f9df19d079e6b06ac206b5f679b', 'obligation_digest': '547be8b0ca24a57ee4f338a2def0a58d40c72f9df19d079e6b06ac206b5f679b', 'labels': ['eq:disturbance-reference-residual'], 'environment': 'equation'}`
- Operators: `['equality', 'summation']`
- Symbols: `{'latex_commands': ['\\beta', '\\mathsf', '\\nabla', '\\pi', '\\rm', '\\theta'], 'bare_identifiers': ['D', 'N', 'Q', 'T', 'Z', 'i', 'ref', 'v', 'y']}`
- Context graph statuses: `{'nearby_stated': 1, 'inferred_candidate': 4, 'missing': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['derivative', 'route_assumption_denominator_is_nonzero']`

Source row target:

```tex
\widehat D_{\rm ref}^{\,v}
 =\beta v^{\mathsf T}\nabla_\theta\widetilde Z+
 \frac1N\sum_{i=1}^N
 \frac{v^{\mathsf T}\nabla_\theta\pi_\theta(\mathsf Z_i;y)
       -\beta v^{\mathsf T}\nabla_\theta
                    \widetilde\pi_\theta(\mathsf Z_i;y)}
      {Q_\theta(\mathsf Z_i)}.
 \label{eq:disturbance-reference-residual}
```

Full display target:

```tex
\widehat D_{\rm ref}^{\,v}
 =\beta v^{\mathsf T}\nabla_\theta\widetilde Z+
 \frac1N\sum_{i=1}^N
 \frac{v^{\mathsf T}\nabla_\theta\pi_\theta(\mathsf Z_i;y)
       -\beta v^{\mathsf T}\nabla_\theta
                    \widetilde\pi_\theta(\mathsf Z_i;y)}
      {Q_\theta(\mathsf Z_i)}.
 \label{eq:disturbance-reference-residual}
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
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1856-1889']`
- `route_assumption_denominator_is_nonzero` status `missing`
  Role: route-required assumption from assumption_discovery
  What: denominator is nonzero
  Why status: The low-level route detector marked this assumption as missing.
  Required next evidence: Resolve this assumption in typed IR before backend proof attempts.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1877-1884']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review'], 'blocked_by_assumption_ids': ['route_assumption_denominator_is_nonzero'], 'unsupported_constructs': [], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['derivative', 'route_assumption_denominator_is_nonzero']`
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e', 'branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_domain_constraints', 'typed_repair_obligation_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_domain_constraints', 'blocker_formalization_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-reference-residual > line 1877`
  - Context branch: `branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['route_assumption_denominator_is_nonzero'] block constructs ['derivative', 'route_assumption_denominator_is_nonzero'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`. The source span contains macros `['\\beta', '\\mathsf', '\\nabla', '\\pi', '\\rm', '\\theta']` whose mathematical types and backend names are not fixed. Derivative or gradient notation requires differentiability assumptions.
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
    - `missing_domain_or_shape_required`: The backend translation lacks required domain, dimension, or conformability constraints. Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: lean stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `branch_bound_backend_execution_required`: This assumption branch has no branch-bound backend request/result evidence. Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:disturbance-reference-residual > line 1877-1884']`
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_branch_bound_backend_execution_required', 'target_ids': ['branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first'], 'ledger_entry_ids': ['ledger_1fdabd2efd2076b36e22661669f374649b1008eba496312dd2cb065b6f003655'], 'prerequisites': ['scope_bound:ledger_1fdabd2efd2076b36e22661669f374649b1008eba496312dd2cb065b6f003655'], 'launch_vetoes': ['ledger_2c5c23fd8008a313376855cc88c83dd10d9f8c67d50b62a302acb6e65a2ac5de', 'ledger_7f17ce475d0eb708e27602a6da7f649959cbfc625ad3b90874d62ae2250197c9', 'ledger_8bb170c6d03f6159e7ede40a1bcec2495632ff1af0dc7d08988017907f97eb90', 'ledger_c07813d99760aef97ed828075c12be44ba22ccb74781cce862d48d7f5e36e56d'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'branch_bound_backend_execution_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_f9d1c1c1b47f64749de8d6f87232ae4892ddd9eceba6b1b4668f99dc291f3f16'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_1fdabd2efd2076b36e22661669f374649b1008eba496312dd2cb065b6f003655', 'ledger_2c5c23fd8008a313376855cc88c83dd10d9f8c67d50b62a302acb6e65a2ac5de', 'ledger_7f17ce475d0eb708e27602a6da7f649959cbfc625ad3b90874d62ae2250197c9', 'ledger_8bb170c6d03f6159e7ede40a1bcec2495632ff1af0dc7d08988017907f97eb90', 'ledger_c07813d99760aef97ed828075c12be44ba22ccb74781cce862d48d7f5e36e56d'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e']`
  - Typed unresolved constructs: `['derivative', 'route_assumption_denominator_is_nonzero']`
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
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Add explicit scalar, vector, matrix, and conformability declarations before backend translation.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\beta', '\\mathsf', '\\nabla', '\\pi', '\\rm', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
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
- `patch_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-reference-residual` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-reference-residual > line 1877, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
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
- `blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\beta', '\\mathsf', '\\nabla', '\\pi', '\\rm', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_reference_residual_547be8b0ca24a57e_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

### 15. `eq:disturbance-conditional-score`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-conditional-score > line 1935`
- Claim type: `definition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['conditional_expectation']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1935, 'line_end': 1937, 'label': 'eq:disturbance-conditional-score', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 95442, 'end_byte': 95568, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_98f188333eff8b32d04bd71d693eb407401738e8b5898bdc17a845a8f41a93d6', 'obligation_digest': '98f188333eff8b32d04bd71d693eb407401738e8b5898bdc17a845a8f41a93d6', 'labels': ['eq:disturbance-conditional-score'], 'environment': 'equation'}`
- Operators: `['equality', 'conditional_bar']`
- Symbols: `{'latex_commands': ['\\log', '\\nabla', '\\pi', '\\theta'], 'bare_identifiers': ['B', 'E', 'a', 'h', 'y']}`
- Context graph statuses: `{'nearby_stated': 1, 'inferred_candidate': 4, 'missing': 2}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['derivative', 'conditional', 'conditional_law', 'integrability']`

Source row target:

```tex
\nabla_\theta\log\overline\pi_\theta(a)
 =\mathbb E_\theta[h_\theta(a,B;y)\mid a,y].
 \label{eq:disturbance-conditional-score}
```

Full display target:

```tex
\nabla_\theta\log\overline\pi_\theta(a)
 =\mathbb E_\theta[h_\theta(a,B;y)\mid a,y].
 \label{eq:disturbance-conditional-score}
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
- `assumption_relevant_functions_differentiable` status `nearby_stated`
  Role: supports local derivative notation in the FOC route
  What: The relevant functions are differentiable.
  Why status: The condition is stated in the local paragraph/proposition context.
  Required next evidence: Use this as differentiability evidence, but do not treat it as integrability or derivative-expectation interchange evidence.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1927-1951', 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1953-1969']`
- `requirement_conditional_law_defined` status `missing`
  Role: well-definedness condition for conditional expectation
  What: A conditional law for the expectation is defined.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: Cite or add the transition kernel/probability law used by the conditional expectation.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1935-1937']`
- `requirement_conditional_integrability` status `missing`
  Role: finite-scalar condition for expectation-valued equations
  What: Random terms inside the conditional expectation are measurable and integrable.
  Why status: No matching local source evidence states the required condition.
  Required next evidence: Cite or add measurability and finite conditional first-moment/dominated-envelope conditions.
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1935-1937']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32`
  Diagnostic status: `blocked_on_missing_typed_assumptions`
  Encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_integrability', 'requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law', 'integrability'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  Unresolved constructs: `['derivative', 'conditional', 'conditional_law', 'integrability']`
  Route hints: `[{'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing assumptions or unsupported notation prevent verified backend routing.'}, {'backend': 'manual_formalization', 'suitability': 'required_before_cas', 'reason': 'Conditional expectation requires a typed probability kernel and integrability assumptions before CAS or Lean encoding.'}, {'backend': 'human_review', 'suitability': 'required', 'reason': 'Missing or unresolved typed assumptions block certifying backend attempts.'}]`
  Assumption statuses:
  - `requirement_conditional_integrability` status `missing`: Random terms inside the conditional expectation are measurable and integrable.
  - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32', 'branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation', 'actionable_abstention:conditional_expectation', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_domain_constraints', 'typed_repair_obligation_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_domain_constraints', 'blocker_formalization_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_lean', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-conditional-score > line 1935`
  - Context branch: `branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation`
  - Context selection authority: `serialization_only_nondominated_context`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['requirement_conditional_integrability', 'requirement_conditional_law_defined'] block constructs ['derivative', 'conditional', 'conditional_law', 'integrability'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification. The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument. Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed. Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
  - Missing or unresolved assumptions:
    - `requirement_conditional_integrability` status `missing`: Random terms inside the conditional expectation are measurable and integrable.
    - `requirement_conditional_law_defined` status `missing`: A conditional law for the expectation is defined.
  - Candidate assumption set that remains blocked:
    - The conditioned shock or path has finite support.
    - Every payoff/value term inside the expectation is finite at each support point.
    - The conditioning state or information set is explicitly defined.
    - The conditioning object `a,y` is defined as a sigma-field, information set, state, or conditioning variable for this equality.
  - Candidate derivation route that remains blocked:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `a,y`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `a,y`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Exact blockers before this can become a repair proposal:
    - `conditioning_scope_translation_required`: The conditional bar has no backend-level conditioning object yet. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `conditional_law_translation_required`: The conditional law required by the expectation is not stated as an encodable object. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `integrability_translation_required`: Integrability of the random payoff/value terms is not established. Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `missing_domain_or_assumption_required`: The branch still has missing or unresolved typed assumptions. Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `macro_translation_required`: LaTeX macros must be translated into backend symbols before execution. Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `missing_domain_or_shape_required`: The backend translation lacks required domain, dimension, or conformability constraints. Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
    - `formalization_required`: sympy stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
    - `formalization_required`: sage stub is not yet a certifying formalization. Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:disturbance-conditional-score > line 1935-1937']`
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition']`
- Unique top branch, only if one exists: `None`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'blocked_for_human_or_formalization_choice', 'target_ids': ['branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation', 'branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition'], 'ledger_entry_ids': ['unresolved_branch_choice'], 'prerequisites': ['resolve_nondominated_branch_choice'], 'launch_vetoes': ['ledger_1304356fa9b3e49b8d1636d048fc635fb1a16c29c484e66d3f544b02ed2a4fac', 'ledger_1cf87495bd35a3015f48b28f9d5e2160668434ea494643336612ff5bf61829af', 'ledger_223a2774e0e4ca27c721fc753f196d5c247404d068b9082c1672c6d182db2f1b', 'ledger_3382d4c802aac9d541f3e7ff445c130863883e4af99652ff24de04748a976d5c', 'ledger_4a2e2c4155d1e716920c3306bf675d915dc2fb6caaf023f9a6a6fa47d5e7cb2c', 'ledger_663ec540a89409d2fa4d126e35f07f99d61f6e493b958e82b3c4f1c07d306fe7', 'ledger_70617da75286b4cc1fe9a7d498db730ae223c15ba24d63f9162bb0bb65eab282', 'ledger_88d9e3027d5db4afceaab408c11a87a47c7a64a8a5224c1f5ef46da0dae2d089', 'ledger_b8e2f24193c701045db4cddcb8d7e7f8664593658c10d78e71347318f78328b9', 'ledger_b9d77ff51b43ed9ec8e09cc34a93451b8ace4b82fc2af04b43ee2e95df499ae9', 'ledger_c3fae1fe1034dbe75cf5a535881ec36b9e08a7ee032b8c69fd2936afe051b050', 'ledger_ce4d5d06673eae433ce7f979d3699f312cc54cdc8797fcf75b5e1d55acde3749', 'ledger_d4c48dcdc875a3eae834b6da77f10a5dac2ca40096ec99c18e23d2941608158f', 'ledger_e6441c00a478e4ae7c39018f4d2c054483f250f7a38683f086999e69e8b6edb5', 'ledger_e9531dfa5228a05bb0cf548311068b7cb618e4a5ca608a3ff60f328e3922138c', 'ledger_fb5c970f2797faa2480678016b2b72af01b1ff1fd14b6c436514581d90a14734'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_choice_record', 'schema_version': 'p06_formalization_choice@1', 'binding_fields': ['branch_ids', 'target_id'], 'path_role': 'decision_blocker'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_e826be110c256491a65f7fbe1a7bd690e929ed51cbbb6455adf51ba7047c9614'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_1304356fa9b3e49b8d1636d048fc635fb1a16c29c484e66d3f544b02ed2a4fac', 'ledger_223a2774e0e4ca27c721fc753f196d5c247404d068b9082c1672c6d182db2f1b', 'ledger_3382d4c802aac9d541f3e7ff445c130863883e4af99652ff24de04748a976d5c', 'ledger_4a2e2c4155d1e716920c3306bf675d915dc2fb6caaf023f9a6a6fa47d5e7cb2c', 'ledger_70617da75286b4cc1fe9a7d498db730ae223c15ba24d63f9162bb0bb65eab282', 'ledger_88d9e3027d5db4afceaab408c11a87a47c7a64a8a5224c1f5ef46da0dae2d089', 'ledger_b8e2f24193c701045db4cddcb8d7e7f8664593658c10d78e71347318f78328b9', 'ledger_fb5c970f2797faa2480678016b2b72af01b1ff1fd14b6c436514581d90a14734'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'The conditioned shock or path has finite support.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Every payoff/value term inside the expectation is finite at each support point.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'The conditioning state or information set is explicitly defined.', 'status': 'candidate'}, {'id': 'legacy_assumption_4', 'statement': 'The conditioning object `a,y` is defined as a sigma-field, information set, state, or conditioning variable for this equality.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.
- Serialization position `2`: `branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_1cf87495bd35a3015f48b28f9d5e2160668434ea494643336612ff5bf61829af', 'ledger_663ec540a89409d2fa4d126e35f07f99d61f6e493b958e82b3c4f1c07d306fe7', 'ledger_b9d77ff51b43ed9ec8e09cc34a93451b8ace4b82fc2af04b43ee2e95df499ae9', 'ledger_c3fae1fe1034dbe75cf5a535881ec36b9e08a7ee032b8c69fd2936afe051b050', 'ledger_ce4d5d06673eae433ce7f979d3699f312cc54cdc8797fcf75b5e1d55acde3749', 'ledger_d4c48dcdc875a3eae834b6da77f10a5dac2ca40096ec99c18e23d2941608158f', 'ledger_e6441c00a478e4ae7c39018f4d2c054483f250f7a38683f086999e69e8b6edb5', 'ledger_e9531dfa5228a05bb0cf548311068b7cb618e4a5ca608a3ff60f328e3922138c'], 'covered_obligation_ids': ['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'A conditional kernel or probability law is fixed for the random object.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'All random terms inside the expectation are measurable under that law.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Those terms are dominated by an integrable envelope or have finite conditional first moments.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32']`
  - Typed unresolved constructs: `['derivative', 'conditional', 'conditional_law', 'integrability']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_integrability', 'requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law', 'integrability'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['The conditioned shock or path has finite support.', 'Every payoff/value term inside the expectation is finite at each support point.', 'The conditioning state or information set is explicitly defined.', 'The conditioning object `a,y` is defined as a sigma-field, information set, state, or conditioning variable for this equality.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `a,y`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `a,y`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 4 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `a,y`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `a,y`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_integrability_translation_required` (integrability_translation_required): Integrability of the random payoff/value terms is not established.
      Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_integrability', 'requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\log', '\\nabla', '\\pi', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`
- `branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition` status `blocked_before_backend_certification`
  - Closes obligations: `['conditional_law_defined', 'measurable_integrable_payoff_terms', 'conditioning_information_defined']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32']`
  - Typed unresolved constructs: `['derivative', 'conditional', 'conditional_law', 'integrability']`
  - Typed encodability: `{'status': 'blocked_pending_typed_assumptions', 'candidate_backends': ['human_review', 'manual_formalization'], 'blocked_by_assumption_ids': ['requirement_conditional_integrability', 'requirement_conditional_law_defined'], 'unsupported_constructs': ['conditional', 'conditional_law', 'integrability'], 'why': 'Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing.'}`
  - Backend evidence status: `typed_translation_blocked`
  - Raw backend promotion history (diagnostic only): `{'can_promote': False, 'supported_status': None, 'reason': 'No certifying proof/refutation evidence supports promotion.', 'errors': [], 'evidence_refs': [], 'boundary': 'A branch can be promoted to proved or refuted only from scoped certifying backend evidence or a concrete counterexample. Route plans, retrieval hits, static extraction, proof-state traces, and backend unavailability are diagnostic evidence only.'}`
  - Effective document promotion: `{'can_promote': False, 'supported_status': None, 'reason': 'Document promotion is disabled because current backend evidence has no exact Phase 01 binding.', 'errors': ['legacy_unbound_document_evidence', 'document_repair_publication_quarantined'], 'evidence_refs': [], 'boundary': 'Raw lower-level evidence is diagnostic history only. Exact source, target, assumptions, branch, native input, result, tool/version, and edit binding are required before document promotion.'}`
  - Evidence binding: `no_branch_evidence`
  - Failure classification: `branch_execution_pending`
  - Why: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.
  - Proposed assumptions: ['A conditional kernel or probability law is fixed for the random object.', 'All random terms inside the expectation are measurable under that law.', 'Those terms are dominated by an integrable envelope or have finite conditional first moments.']
  - Route under assumptions:
    - Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `a,y`.
    - Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `a,y`.
    - Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step.
  - Expansion records:
    - `assumption_addition` status `proposed`: Propose 3 assumption(s) for this branch.
    - `derivation_split` status `diagnostic_route`: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `a,y`.
    - `derivation_split` status `diagnostic_route`: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `a,y`.
    - `derivation_split` status `diagnostic_route`: Only after those checks should the equality be treated as a scalar derivation step.
    - `formalization_route` status `blocked_before_execution`: sympy translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: sage translation is blocked by typed/source translation blockers.
    - `formalization_route` status `blocked_before_execution`: lean translation is blocked by typed/source translation blockers.
    - `rule_hypothesis_candidate` status `candidate_pending_tree_verification`: Declare the conditioning object as a sigma-field or information set and rewrite the conditional expectation relative to that object.
  - External-tool ledger: `['sympy:requires_formalization', 'sage:requires_formalization', 'lean:requires_formalization', 'leansearchv2:available', 'lean_explore:available', 'jixia:requires_formalization', 'pantograph:requires_formalization', 'lean_dojo:requires_formalization']`
  - Translation attempts:
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_missing_domain_constraints']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_missing_domain_constraints']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_conditioning_scope_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_conditional_law_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_integrability_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_latex_macro_translation_required', 'blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_missing_domain_constraints']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required): The conditional bar has no backend-level conditioning object yet.
      Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required): The conditional law required by the expectation is not stated as an encodable object.
      Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_integrability_translation_required` (integrability_translation_required): Integrability of the random payoff/value terms is not established.
      Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
      Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['requirement_conditional_integrability', 'requirement_conditional_law_defined']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\log', '\\nabla', '\\pi', '\\theta']` whose mathematical types and backend names are not fixed.
      Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
    - `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_missing_domain_constraints` (missing_domain_or_shape_required): The backend translation lacks required domain, dimension, or conformability constraints.
      Why: Derivative or gradient notation requires differentiability assumptions.
      Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
  - Formalization stubs:
    - `sympy` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `sage` status `requires_manual_translation`; unsupported: `['LaTeX macros require translation to backend symbols']`
    - `lean` status `skeleton_contains_sorry_not_certifying`; unsupported: `['LaTeX macros require translation to backend symbols', 'Lean theorem statement for the LaTeX equality has not been generated']`

How the derivation can work:
- `Define conditional law`: Specify the kernel or conditional distribution used by the expectation.
- `Check integrability`: Verify each random payoff, value, or derivative term has a finite conditional expectation.
- `Use expectation as scalar`: Only after those checks should the equality be treated as a scalar derivation step.

Backend attempts:
- `sympy_algebra_attempt` with `sympy`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`
- `bounded_counterexample_attempt` with `sympy_finite_domain`: status `not_encodable`, evidence `diagnostic`, certification `diagnostic`

Blocked patch candidates (non-applicable):
- `patch_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-conditional-score` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-conditional-score > line 1935, add an assumptions paragraph: "For this displayed equality, assume: The conditioned shock or path has finite support. Every payoff/value term inside the expectation is finite at each support point. The conditioning state or information set is explicitly defined. The conditioning object `a,y` is defined as a sigma-field, information set, state, or conditioning variable for this equality. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `a,y`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `a,y`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation becomes a finite weighted sum.` by making the operators and objects in the displayed equality well-defined before backend certification.
- `patch_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-conditional-score` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-conditional-score > line 1935, add an assumptions paragraph: "For this displayed equality, assume: A conditional kernel or probability law is fixed for the random object. All random terms inside the expectation are measurable under that law. Those terms are dominated by an integrable envelope or have finite conditional first moments. Under these assumptions, the derivation route is: Define conditional law: Specify the kernel or conditional distribution used by the expectation. In this source span, the conditioning object is `a,y`. Check integrability: Verify each random payoff, value, or derivative term has a finite conditional expectation. In this source span, the conditioning object is `a,y`. Use expectation as scalar: Only after those checks should the equality be treated as a scalar derivation step."
  Blocked reason: Document repair publication is disabled and the branch evidence is legacy/unbound.
  Rationale: This branch closes `The expectation is a well-defined finite conditional integral.` by making the operators and objects in the displayed equality well-defined before backend certification.

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
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_integrability_translation_required` (integrability_translation_required)
  Problem: Integrability of the random payoff/value terms is not established.
  Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_integrability', 'requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\log', '\\nabla', '\\pi', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_finite_state_conditional_expectation_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_conditioning_scope_translation_required` (conditioning_scope_translation_required)
  Problem: The conditional bar has no backend-level conditioning object yet.
  Why: The expression must identify whether the conditioning object is a state, sigma-field, information set, or kernel argument.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_conditional_law_translation_required` (conditional_law_translation_required)
  Problem: The conditional law required by the expectation is not stated as an encodable object.
  Why: Conditional expectation notation is only meaningful after the transition kernel or conditional distribution is fixed.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_integrability_translation_required` (integrability_translation_required)
  Problem: Integrability of the random payoff/value terms is not established.
  Why: Without finite conditional first moments or a dominated envelope, the backend cannot treat the expectation as finite.
  Required next evidence: State or verify the required typed assumption, update the branch assumptions, and rerun the backend translator.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['requirement_conditional_integrability', 'requirement_conditional_law_defined']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\log', '\\nabla', '\\pi', '\\theta']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_missing_domain_constraints` (missing_domain_or_shape_required)
  Problem: The backend translation lacks required domain, dimension, or conformability constraints.
  Why: Derivative or gradient notation requires differentiability assumptions.
  Required next evidence: Declare the missing domain/shape constraints or split the obligation before backend execution.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_kernel_integrability_condition_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_conditional_law_defined` (probability_condition)
  Problem: A conditional probability law for the random variables inside the expectation.
  Why: A conditional expectation is not a real-valued operator until the conditioning law or kernel is specified.
  Required next evidence: Makes the expectation operator well defined.
- `blocker_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_measurable_integrable_payoff_terms` (integrability_condition)
  Problem: Measurability and finite conditional first moments for every payoff, value, or derivative term inside the expectation.
  Why: Without measurability and integrability, the expectation may be undefined or infinite.
  Required next evidence: Turns the displayed expression into a finite scalar equality.
- `blocker_semantic_packet_eq_disturbance_conditional_score_98f188333eff8b32_conditioning_information_defined` (information_condition)
  Problem: A definition of the conditioning information set, state, or sigma-field.
  Why: The notation after the conditional bar determines what information the expectation conditions on.
  Required next evidence: Fixes the scope of the conditional expectation used in the derivation.

Smallest next audit: `audit_and_propose_assumptions` - Generate explicit assumption proposals for the missing route conditions.

### 16. `eq:disturbance-moving-line-likelihood`

- Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-moving-line-likelihood > line 1998`
- Claim type: `theorem_proposition`
- Tree status: `partial`
- Promotion guard: `can_promote=False`
- Semantic domains: `['generic_formalization']`
- Extraction uncertainty: `[]`
- Full display span: `{'file': 'docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex', 'line_start': 1998, 'line_end': 2000, 'label': 'eq:disturbance-moving-line-likelihood', 'section_path': ['A model score that averages over ancestors', 'A disturbance-coordinate proposal and its score'], 'start_byte': 97877, 'end_byte': 98004, 'source_digest': '071506a3f90c3ff32d89b6a0c88e161f4c638420e30b6b7defd0df9fbca4022a', 'obligation_id': 'obl_69d4ced2e5ee82e324f563e6a7175ab0c5cff1f19b34c948c0093127dbed9ef0', 'obligation_digest': '69d4ced2e5ee82e324f563e6a7175ab0c5cff1f19b34c948c0093127dbed9ef0', 'labels': ['eq:disturbance-moving-line-likelihood'], 'environment': 'equation'}`
- Operators: `['equality']`
- Symbols: `{'latex_commands': ['\\log', '\\pi'], 'bare_identifiers': ['Z', 'b', 'd', 'y']}`
- Context graph statuses: `{'inferred_candidate': 4, 'missing': 1}`
- Typed repair obligation: `typed_repair_obligation_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3`
- Typed obligation status: `blocked_on_missing_typed_assumptions`
- Typed unresolved constructs: `['route_assumption_denominator_is_nonzero']`

Source row target:

```tex
\log Z=-\log(2\pi)-\tfrac12\log d
        -\tfrac12(y_1^2+y_2^2)+\frac{b^2}{2d}.
 \label{eq:disturbance-moving-line-likelihood}
```

Full display target:

```tex
\log Z=-\log(2\pi)-\tfrac12\log d
        -\tfrac12(y_1^2+y_2^2)+\frac{b^2}{2d}.
 \label{eq:disturbance-moving-line-likelihood}
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
  Source refs: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex:1998-2000']`

Typed repair obligation:
- ID: `typed_repair_obligation_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3`
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
  - `document_gap_report_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first` type `gap_report`, closure `blocked_at_exact_node`, publishable_as_repair=`False`, publishable_as_gap_report=`True`, reportable_as_partial_evidence=`False`
    - Failure classifications: `['mathematical_blocked', 'branch_execution_pending', 'formalization_blocked']`
    - Veto ids: `['document_repair_publication_quarantined']`
    - Evidence refs: `['semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3', 'branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first', 'actionable_abstention:generic_formalization', 'blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_latex_macro_translation_required', 'typed_repair_obligation_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3']`
    - Remaining blocker ids: `['blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_latex_macro_translation_required', 'blocker_formalization_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_sympy', 'blocker_formalization_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_sage', 'blocker_formalization_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_lean', 'blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_branch_bound_execution_required']`

Document-ready repair proposals:
- None generated from the ranked branch evidence.

Document partial-evidence reports (non-repair):
- None generated from the ranked branch evidence.

Document gap reports:
- `document_gap_report_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first`
  - Contract: `document_gap_report`
  - Closure status: `blocked_at_exact_node`
  - Location: `ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-moving-line-likelihood > line 1998`
  - Context branch: `branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first`
  - Context selection authority: `unique_nondominated`
  - Nondominated branches: `['branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first']`
  - Context branch outcome: `blocked_with_specific_next_evidence`
  - Problem: The target is not yet a certifiable derivation because missing or unresolved assumptions ['route_assumption_denominator_is_nonzero'] block constructs ['route_assumption_denominator_is_nonzero'].
  - Why this is a derivation problem: Missing/unresolved assumptions or stochastic/interchange constructs must be resolved before certifying backend routing. This branch closes `Makes the abstention inspectable by deterministic tooling.` by making the operators and objects in the displayed equality well-defined before backend certification. Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`. The source span contains macros `['\\log', '\\pi']` whose mathematical types and backend names are not fixed.
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
  - Source refs for missing/unresolved evidence: `['docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex > eq:disturbance-moving-line-likelihood > line 1998-2000']`
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
- Nondominated branches: `['branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first']`
- Unique top branch, only if one exists: `branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first`
- Selected discriminating action: `{'schema_version': 'p06_discriminating_action@1', 'action_kind': 'resolve_formalization_required', 'target_ids': ['branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first'], 'branch_ids': ['branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first'], 'ledger_entry_ids': ['ledger_03ad206191e470453da29e09b50b752db6f2e802ef9d0f57fbfbd51f477293b2'], 'prerequisites': ['scope_bound:ledger_03ad206191e470453da29e09b50b752db6f2e802ef9d0f57fbfbd51f477293b2'], 'launch_vetoes': ['ledger_2c0e67ccb353a6f4521db1088d5d77c27001d54c4e08405a0be00672af365bf6', 'ledger_6e4b25a1aa5e27a5fb2a7831580f7a67765125c53953551fd821fb006989fc0c', 'ledger_a0452188c17ec4b76409a6014a46626591e749a5ba45e288003f8483982acd7c'], 'tool_route': {'tool': None, 'role': 'local_function', 'route': 'mathdevmcp.failure_ledgers.select_next_discriminating_action', 'availability_state': 'available'}, 'budget': {'profile': 'synthetic_local', 'max_attempts': 1, 'timeout_ms': None, 'max_output_bytes': None, 'provenance': 'Phase 06 smallest-discriminator default; unknown fields remain null'}, 'expected_artifact': {'kind': 'formalization_required_resolution', 'schema_version': 'p06_legacy_discriminator@1', 'binding_fields': ['branch_id', 'origin_id', 'target_id'], 'path_role': 'diagnostic_decision_evidence'}, 'outcomes': {'unavailable': {'stop': True, 'means': 'The action ended with scoped outcome unavailable; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unsupported': {'stop': True, 'means': 'The action ended with scoped outcome unsupported; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'timeout': {'stop': True, 'means': 'The action ended with scoped outcome timeout; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'execution_error': {'stop': True, 'means': 'The action ended with scoped outcome execution_error; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'malformed': {'stop': True, 'means': 'The action ended with scoped outcome malformed; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'certified': {'stop': True, 'means': 'The action ended with scoped outcome certified; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'refuted': {'stop': True, 'means': 'The action ended with scoped outcome refuted; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}, 'unknown': {'stop': True, 'means': 'The action ended with scoped outcome unknown; record it in the appropriate ledger.', 'does_not_mean': 'It does not by itself establish document proof, repair correctness, publication authority, or scientific optimality.'}}, 'non_claims': ['the action does not authorize publication or source editing', 'the selected action is a discriminator, not a predicted success'], 'action_id': 'action_2a95c347a5dae305013d95ccfb838955a596479fe5b8acfd733a4c4e9f627b25'}`
- Serialization position `1`: `branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first` outcome `blocked_with_specific_next_evidence`, nondominated `True`
  - Decision dimensions: `{'exact_verified_evidence': False, 'veto_entry_ids': ['ledger_03ad206191e470453da29e09b50b752db6f2e802ef9d0f57fbfbd51f477293b2', 'ledger_2c0e67ccb353a6f4521db1088d5d77c27001d54c4e08405a0be00672af365bf6', 'ledger_6e4b25a1aa5e27a5fb2a7831580f7a67765125c53953551fd821fb006989fc0c', 'ledger_a0452188c17ec4b76409a6014a46626591e749a5ba45e288003f8483982acd7c'], 'covered_obligation_ids': ['formalized_local_obligation'], 'typed_assumptions': [{'id': 'legacy_assumption_1', 'statement': 'Define every symbol, domain, and operator in the cited source line.', 'status': 'candidate'}, {'id': 'legacy_assumption_2', 'statement': 'Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic.', 'status': 'candidate'}, {'id': 'legacy_assumption_3', 'statement': 'Rerun the relevant assumption/proof audit after the typed obligation exists.', 'status': 'candidate'}], 'execution_cost': None}`
  - Explanation: blocked_with_specific_next_evidence; relation membership is determined by validity gates and set/comparable-cost relations, never attempt or blocker volume.

Candidate assumption branches:
- `branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first` status `blocked_before_backend_certification`
  - Closes obligations: `['formalized_local_obligation']`
  - Typed obligation ids: `['typed_repair_obligation_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3']`
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
    - `sympy` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_latex_macro_translation_required']`
    - `sage` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_latex_macro_translation_required']`
    - `lean` status `blocked_before_execution`; attempt ids `[]`; blockers `['blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_missing_typed_assumptions', 'blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_latex_macro_translation_required']`
  - Translation blockers:
    - `blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_missing_typed_assumptions` (missing_domain_or_assumption_required): The branch still has missing or unresolved typed assumptions.
      Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
      Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
    - `blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_latex_macro_translation_required` (macro_translation_required): LaTeX macros must be translated into backend symbols before execution.
      Why: The source span contains macros `['\\log', '\\pi']` whose mathematical types and backend names are not fixed.
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
- `patch_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first` status `blocked_non_applicable`
  Applicable: `False`
  Blocked candidate text: Near `eq:disturbance-moving-line-likelihood` at ledh_younis_kdm_score.tex > A model score that averages over ancestors > A disturbance-coordinate proposal and its score > eq:disturbance-moving-line-likelihood > line 1998, add an assumptions paragraph: "For this displayed equality, assume: Define every symbol, domain, and operator in the cited source line. Choose whether the line is a definition, identity, optimization condition, estimator, or diagnostic. Rerun the relevant assumption/proof audit after the typed obligation exists. Under these assumptions, the derivation route is: Formalize local obligation: Convert the cited line into a typed obligation before proposing a document edit."
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
- `blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_missing_typed_assumptions` (missing_domain_or_assumption_required)
  Problem: The branch still has missing or unresolved typed assumptions.
  Why: Typed encodability is blocked by `['route_assumption_denominator_is_nonzero']`.
  Required next evidence: Close each listed typed assumption with a source citation or an explicit proposed assumption before certifying backend translation.
- `blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_latex_macro_translation_required` (macro_translation_required)
  Problem: LaTeX macros must be translated into backend symbols before execution.
  Why: The source span contains macros `['\\log', '\\pi']` whose mathematical types and backend names are not fixed.
  Required next evidence: Map each macro used by the target to a typed backend symbol or definition and rerun the translator.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_sympy` (formalization_required)
  Problem: sympy stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_sage` (formalization_required)
  Problem: sage stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_formalization_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_lean` (formalization_required)
  Problem: lean stub is not yet a certifying formalization.
  Why: LaTeX macros require translation to backend symbols; Lean theorem statement for the LaTeX equality has not been generated
  Required next evidence: Translate the source-local equality and branch assumptions into the backend language, then run the backend check.
- `blocker_branch_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_typed_obligation_first_branch_bound_execution_required` (branch_bound_backend_execution_required)
  Problem: This assumption branch has no branch-bound backend request/result evidence.
  Why: Root attempts bind only the original root target. They do not bind this branch's assumptions, formalization request, or result and therefore cannot close the branch.
  Required next evidence: Create and execute an exact branch request that binds the target, typed assumptions, branch id, native input, tool/version, and result.
- `blocker_semantic_packet_eq_disturbance_moving_line_likelihood_69d4ced2e5ee82e3_formalized_local_obligation` (formalization_condition)
  Problem: A typed local obligation with defined symbols, domains, and operator meanings.
  Why: The diagnostic source does not yet expose enough structure for a mathematical repair.
  Required next evidence: Creates the next deterministic target for assumption discovery or proof audit.

Smallest next audit: `audit_and_propose_fix` - Regenerate concrete proposals after adding the listed obligations.

## Non-Claims

- `document_tree_audit_not_document_proof`: This workflow is a semantic gap and tree-evidence report; it does not prove the whole document.
- `semantic_packets_not_certificates`: Missing obligations, assumption sets, and derivation routes are deterministic guidance, not proof certificates.
- `proof_search_not_final_certificate`: LeanDojo, Pantograph, retrieval, route plans, and static extraction are diagnostic until direct Lean or another certifying backend checks the scoped target.
- `document_repair_publication_quarantined`: No returned candidate is an applicable document edit while publication mode is disabled.
