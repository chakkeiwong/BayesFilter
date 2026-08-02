# Read-only audit of `docs/main.tex` and included chapters

Date: 2026-08-01
Scope: the current monograph rooted at `docs/main.tex`
Mode: read-only review, plus this audit note

## What this audit covered

This audit reviewed the current document included by `docs/main.tex` chapter by chapter and section by section, with emphasis on:

- readability and teaching flow;
- compliance with the BayesFilter governance and scholarly-literature policies;
- mathematical correctness claims;
- derivation completeness;
- provenance and citation discipline;
- stale or contradictory draft content; and
- document-structure and LaTeX-integrity issues that affect reviewability.

The review also examined `~/research-assistant`, `~/MathDevMCP`, and `~/DynareMCP` to identify tooling that can help with future monograph audits.

## Executive summary

The document has substantial mathematical value, but it is not currently review-ready as a monograph.

The most important global problems are:

1. **Internal project provenance leaks into monograph prose.** Many chapters still rely on phrases such as “MacroFinance note”, “current MathDevMCP audit”, “source-project audit”, or unnamed local notes where the monograph should instead provide either a local derivation or an exact source anchor.
2. **Several chapters are too large to teach linearly.** The worst cases are `ch18b_structural_deterministic_dynamics.tex`, `ch28a_neural_network_state_space_model_applications.tex`, `ch32c_entropic_ot_sinkhorn.tex`, and `ch32c2_ledh_pfpf_ot_custom_gradient.tex`.
3. **There are real LaTeX integrity defects.** At least four duplicate labels exist in included chapters, and one duplicate subsection title/label in `ch32c` is an obvious structural bug.
4. **Some chapters still read like project governance memos rather than durable exposition.** This is especially visible where validation policy, source-project history, or current campaign status are embedded directly into theory chapters.
5. **A smaller number of local mathematical or notation defects remain.** The clearest examples are the under-specified auxiliary block in `ch02`, the `e_i`/`e_j` mismatch in `ch37`, and several places where a target or derivative claim needs sharper qualification.
6. **The repository contains chapter-scale shadow files not included by `main.tex`.** This increases drift risk and makes it harder to know which exposition is authoritative.

## Most useful helper repositories for future audits

### Best primary helper: `~/MathDevMCP`

Most useful capabilities for this monograph audit:

- label-aware and corpus-level LaTeX search with provenance;
- equation localization with byte offsets and masking of irrelevant command-definition lines;
- derivation and proof-audit workflows with explicit certification boundaries;
- document-level rigor audit and derivation-tree audit;
- notation, assumption, and literature-local audits; and
- report paging and artifact-resolution tools for large audit outputs.

Useful entry points:

- `src/mathdevmcp/mcp_server.py`
- `src/mathdevmcp/latex_index.py`
- `src/mathdevmcp/equation_locator.py`
- `src/mathdevmcp/proof_audit_v2.py`
- `src/mathdevmcp/document_exposition.py`
- `src/mathdevmcp/literature_local_audit.py`
- `src/mathdevmcp/math_document_rigor.py`
- `src/mathdevmcp/document_derivation_tree.py`
- `src/mathdevmcp/cli.py`

Useful command patterns:

- `PYTHONPATH=src python -m mathdevmcp.cli search-latex "<query>" --root <doc-root> --limit 10`
- `PYTHONPATH=src python -m mathdevmcp.cli plan-math-document-rigor-audit <tex_path> --max-labels 30`
- `PYTHONPATH=src python -m mathdevmcp.cli audit-math-document-rigor <tex_path> --report-profile actionable`
- `PYTHONPATH=src python -m mathdevmcp.cli audit-document-derivation-tree <tex_path> --response-mode compact --artifact-root <dir>`
- `PYTHONPATH=src python -m mathdevmcp.cli audit-derivation-v2-label <label> --root <doc-root> --summary-only`

Main limitation: many workflows are still experimental/diagnostic. They help locate and classify obligations, but they do not automatically certify truth unless the scoped obligation is actually verified.

### Best supporting source-extraction helper: `~/research-assistant`

Most useful capabilities:

- detect the main TeX file in a source tree;
- flatten `\input` and `\include` structure;
- extract sections, equations, theorem-like blocks, labels, references, citations, bibliography, and macros from LaTeX source;
- expose a read-only MCP surface for source inspection.

Useful entry points:

- `src/research_assistant/source/latex_bundle.py`
- `src/research_assistant/source/latex_flatten.py`
- `src/research_assistant/source/latex_extract.py`
- `src/research_assistant/adapters/mcp_server.py`
- `src/research_assistant/cli.py`
- `docs/mcp.md`

Useful command patterns:

- `scripts/ra-dev parser-tool-matrix`
- `scripts/ra-dev parser-preflight`
- `scripts/ra-dev parse-pdf --pdf /path/to/file.pdf`
- `scripts/ra-dev source-sections --paper-id <id>`
- `scripts/ra-dev source-equations --paper-id <id>`
- `scripts/ra-dev source-citations --paper-id <id>`
- MCP launch: `scripts/ra-mcp-dev --root /home/chakwong/research-assistant`

Main limitation: this is a strong intake/extraction tool, not a proof checker. Math remains raw LaTeX and macro expansion is intentionally conservative.

### Best structural cross-check helper: `~/DynareMCP`

Most useful capabilities:

- TeX structure indexing over entry documents and included files;
- structured contradiction/terminology/open-promise diagnostics when given a request JSON;
- document-substance checks for “thin chapter” or checklist-theater problems.

Useful entry points:

- `src/dynaremcp/scholarly_harness/tex_structure.py`
- `src/dynaremcp/scholarly_harness/contradiction_scan.py`
- `src/dynaremcp/scholarly_harness/document_utility.py`
- `src/dynaremcp/document_substance.py`
- `src/dynaremcp/cli.py`

Main limitation: this is structural and request-driven. It does not by itself check mathematical correctness or source-faithfulness.

## Structural observations about the current document

### Included structure

`docs/main.tex` currently organizes the monograph into:

- 11 parts;
- 55 included chapter files under `docs/chapters/`;
- 7 included appendices under `docs/appendices/`; and
- bibliography `docs/references.bib` with `plainnat` style.

### Shadow and drift-prone chapter files not included by `main.tex`

The repository also contains chapter-scale `.tex` files in `docs/chapters/` that are **not** included by the current `main.tex`:

- `docs/chapters/ch34_highdim_gaussian_and_sparse_quadrature.tex`
- `docs/chapters/ch35_highdim_particle_transport_tensor_filters.tex`
- `docs/chapters/ch36_nonlinear_ssm_hmc_research_program.tex`
- `docs/chapters/ch37_highdim_filtering_candidate_synthesis.tex`

It also contains restart-staging chapter files under `docs/chapters_restart_staging/` and `docs/main_highdim_restart_staging.tex`.

These files are not necessarily wrong, but they materially increase ambiguity about which exposition is canonical.

### Duplicate labels in included files

This audit found at least these duplicate labels inside the current included document:

- `sec:bf-eot-teacher-versus-engineering`
  - `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:126`
  - `docs/chapters/ch32c_entropic_ot_sinkhorn.tex:198`
- `sec:bf-neural-ot-direct-scalable-boundary`
  - `docs/chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex:506`
  - `docs/chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex:552`
- `sec:bf-neural-ot-dynamic-same-scalar`
  - `docs/chapters/ch32f_dynamic_geodesic_operator_learning_target_contract.tex:166`
  - `docs/chapters/ch32f_dynamic_geodesic_operator_learning_target_contract.tex:504`
- `sec:bf-neural-ot-dynamic-sliced-localized`
  - `docs/chapters/ch32f_dynamic_geodesic_operator_learning_target_contract.tex:356`
  - `docs/chapters/ch32f_dynamic_geodesic_operator_learning_target_contract.tex:432`

These are real LaTeX/reference defects, not merely stylistic issues.

---

# Part-by-part findings

## Part I. Scope and Interfaces

### `docs/chapters/ch01_introduction.tex`

- **Anchor:** `\section{Reader Map}`
- **Issue class:** stale-context / flow
- **Finding:** the reader map no longer matches the actual current part ordering in `docs/main.tex`. It describes a shorter and older pipeline and omits the current particle/transport and high-dimensional parts.
- **Suggested correction:** rewrite the reader map against the current `main.tex` structure, explicitly naming the particle, transport, learned-transport, high-dimensional, and later HMC-geometry parts.

### `docs/chapters/ch02_state_space_contracts.tex`

- **Anchor:** `\section{Structural Nonlinear Transition Contract}`
- **Issue class:** derivation-gap / readability
- **Finding:** the chapter defines dynamics for `s_t` and `d_t`, then writes `x_t = pack(s_t, d_t, a_t)` without giving a law or completion rule for `a_t`. That leaves the integration object under-specified.
- **Suggested correction:** either add an explicit auxiliary update/completion map for `a_t`, or state plainly that `a_t` is omitted unless a separate auxiliary-completion rule is declared.

### `docs/chapters/ch03_hmc_target_requirements.tex`

- **Anchor:** `\section{Finite Failure Policy}`
- **Issue class:** derivation-gap / policy
- **Finding:** the piecewise nature of fallback branches is not stated sharply enough. The text risks sounding globally smooth where the guarantee is only branchwise.
- **Suggested correction:** add one explicit sentence saying that the guarantee is piecewise, away from branch-switch boundaries, unless a separate support/boundary construction is given.

### `docs/chapters/ch04_bayesfilter_api.tex`

- **Anchor:** `\section{Fixed-Center Curvature API}`
- **Issue class:** flow / readability / stale-context
- **Finding:** this section abruptly switches from monograph-level API description to lane-specific implementation names and statuses.
- **Suggested correction:** either move this material to the later curvature chapter or rewrite it first as an abstract curvature-initialization contract, with concrete function names relegated to an appendix or source-map note.

## Part II. Linear Gaussian Filtering

### `docs/chapters/ch05_prediction_error_decomposition.tex`

- **Status:** no material issue found in this audit.
- **Strengths:** clear value-side derivation, appropriate qualification around singular `Q`, and a clean separation from later derivative chapters.

### `docs/chapters/ch06_stable_linear_filtering.tex`

- **Anchor:** `\section{Diagnostics}`
- **Issue class:** stale-context / unsupported-claim / readability
- **Finding:** the discussion uses internal trigger terminology such as “no trigger, candidate trigger, confirmed trigger” and invokes MacroFinance-backed telemetry without local definition or durable artifact anchoring.
- **Suggested correction:** either define these categories and cite the supporting artifacts, or rewrite them as generic backend-telemetry states.

### `docs/chapters/ch07_missing_data_mixed_frequency.tex`

- **Anchor:** `\section{Implementation Policy}`
- **Issue class:** stale-context / unsupported-claim
- **Finding:** the chapter says MacroFinance tests explicitly check the policy, but gives no local artifact anchor.
- **Suggested correction:** cite the exact supporting artifact or restate the policy generically without historical source-project framing.

### `docs/chapters/ch08_large_scale_lgssm.tex`

- **Anchor:** `\section{Scale Targets}`
- **Issue class:** readability / stale-context
- **Finding:** literal fixture names such as `baseline_10x3x5` are meaningful to the harness but opaque to a monograph reader.
- **Suggested correction:** replace raw fixture names with prose scenario descriptions, or move raw names to an appendix table.

- **Anchor:** `\section{Validation Ladder}`
- **Issue class:** readability / unsupported-claim
- **Finding:** terms like “medium and strict HMC recovery” are not defined locally.
- **Suggested correction:** replace them with explicit criteria or cross-reference the exact place where those tiers are defined.

- **Anchor:** `\section{Derivative Consolidation Hypotheses}`
- **Issue class:** stale-context / flow / policy
- **Finding:** this section still reads like a live project to-do list rather than stable exposition.
- **Suggested correction:** move the planning language into a reset memo or rewrite as a forward-looking chapter handoff rather than a current action list.

## Part III. Analytic Derivatives

### `docs/chapters/ch09_kalman_score.tex`

- **Anchor:** `\section{Innovation Derivatives}`
- **Issue class:** unsupported provenance / derivation anchor gap
- **Finding:** “The source note derives the innovation derivatives” does not identify the note or equations.
- **Suggested correction:** replace that phrase with an exact citation or add a local derivation sentence from the observation equation to `v_t` and `S_t`.

- **Anchor:** `\section{Gain and Update Derivatives}`
- **Issue class:** derivation completeness
- **Finding:** the filtered covariance derivative is presented for the simplified update without warning that this is not the Joseph form and only matches on the same algebraic branch.
- **Suggested correction:** add a sentence qualifying the branch and requiring Joseph/square-root implementations to validate reconstruction to the same propagated covariance.

- **Anchor:** `\section{Evidence Status}`
- **Issue class:** stale/project-internal evidence framing
- **Finding:** the section depends on transient MathDevMCP and MacroFinance status language.
- **Suggested correction:** recast as durable project evidence status or move tool-specific details to an appendix/result note.

### `docs/chapters/ch10_kalman_hessian.tex`

- **Anchor:** chapter intro and `\section{Second-Order Prediction}`
- **Issue class:** unsupported provenance / teaching-flow gap
- **Finding:** the Hessian recursions are attributed to unnamed MacroFinance notes instead of exact source anchors or a short in-text derivation roadmap.
- **Suggested correction:** add source anchors or a compact product-rule roadmap.

- **Anchor:** `\section{Solve-Form Hessian}`
- **Issue class:** readability density
- **Finding:** the notation arrives too abruptly for the reader to keep track of what each term means.
- **Suggested correction:** add a brief mnemonic list for the log-det term, innovation-linear term, and quadratic solve term.

- **Anchor:** `\section{Evidence Status}`
- **Issue class:** stale content / over-specific audit artifact
- **Finding:** the text currently embeds a transient MathDevMCP mismatch report directly into the theory chapter.
- **Suggested correction:** move it to an audit appendix or summarize the evidence state more generically.

### `docs/chapters/ch11_structural_derivatives.tex`

- **Anchor:** `\section{Provider Map}`
- **Issue class:** teaching flow
- **Finding:** the chapter reads like provider-interface documentation before it teaches a small concrete example.
- **Suggested correction:** begin with a miniature worked provider record, then generalize.

- **Anchor:** `\section{Parameter Transforms}`
- **Issue class:** unsupported provenance
- **Finding:** the chain-rule formulas are justified via MacroFinance provenance instead of exact anchors or a direct derivation.
- **Suggested correction:** cite exact source equations or present the chain rule directly in BayesFilter notation.

- **Anchor:** `\section{Initial Conditions}`
- **Issue class:** derivation completeness
- **Finding:** first derivatives of stationary conditions are given, but second derivatives are only gestured at.
- **Suggested correction:** either add the second-derivative formulas or explicitly defer them to a cited appendix/source note.

### `docs/chapters/ch12_factor_derivatives.tex`

- **Anchor:** `\section{Cholesky Factor Derivatives}`
- **Issue class:** derivation/citation gap
- **Finding:** the formulas are stated cleanly but without a citation or brief derivation of the projection operator.
- **Suggested correction:** add a short derivation or a standard citation.

- **Anchor:** `\section{QR Factor Derivatives}`
- **Issue class:** readability density
- **Finding:** the second-derivative QR block is too compressed for first-pass reading.
- **Suggested correction:** add a bridge sentence about deriving `\Gamma^{(ij)}` from second derivatives of `Q^\top Q = I` plus the triangular constraint.

- **Anchor:** `\section{Square-Root Kalman Stacks}`
- **Issue class:** possible notation confusion
- **Finding:** `Q_t` is overloaded between process covariance and orthogonal factors.
- **Suggested correction:** rename the orthogonal factor symbol.

- **Anchor:** `\section{Backend Parity Gates}`
- **Issue class:** unsupported project-evidence phrasing
- **Finding:** readiness language depends on MacroFinance tests without local artifact anchoring.
- **Suggested correction:** either cite concrete tests/artifacts or state the gates as BayesFilter-local requirements.

### `docs/chapters/ch13_custom_gradient_wrappers.tex`

- **Anchor:** `\section{Same-Scalar Contract}`
- **Issue class:** target-versus-computation clarity
- **Finding:** the chapter should state even more explicitly that a gradient of a different surrogate target is wrong relative to the claimed HMC target unless a corrected algorithm is defined.
- **Suggested correction:** add a plain statement distinguishing same-scalar gradients from different-target gradients and state the verdict directly.

- **Anchor:** `\section{Hessian and Observed-Information Path}`
- **Issue class:** notation inconsistency
- **Finding:** the subsection shifts from `u` coordinates to `\psi` coordinates without a clear map.
- **Suggested correction:** keep one coordinate system per subsection or explicitly map the symbols.

### `docs/chapters/ch14_derivative_validation.tex`

- **Anchor:** `\section{Validation Ladder}`
- **Issue class:** stale/project-specific framing
- **Finding:** the text says “MacroFinance currently supplies examples,” which is too transient for the monograph.
- **Suggested correction:** rewrite as “existing project tests include…” or move donor-project language into a provenance note.

- **Anchor:** `\section{Finite-Difference Tolerances}`
- **Issue class:** derivation completeness / operational incompleteness
- **Finding:** the chapter recommends stable plateaus and multiple step sizes, but gives no minimal operational rule.
- **Suggested correction:** add a concrete acceptance heuristic.

- **Anchor:** `\section{Audit Record}`
- **Issue class:** meta-level repetition
- **Finding:** the section repeats surrounding audit context without adding a crisp conclusion.
- **Suggested correction:** condense it into a summary table or move it to an appendix.

## Part IV. Nonlinear Filtering

### `docs/chapters/ch15_ekf.tex`

- **Anchor:** `\section{Local Linearization}`
- **Issue class:** derivation completeness
- **Finding:** the covariance-update alternatives are only mentioned in prose.
- **Suggested correction:** give both forms explicitly or cite a canonical EKF source section.

- **Anchor:** `\section{Likelihood Status}`
- **Issue class:** emphasis / teaching clarity
- **Finding:** the chapter correctly draws the approximate-target boundary, but the main point could be sharper.
- **Suggested correction:** open the section by saying plainly that HMC with EKF targets the EKF posterior, not the exact nonlinear posterior.

### `docs/chapters/ch16_sigma_point_filters.tex`

- **Anchor:** `\section{Conjugate Unscented Transform Rules}`
- **Issue class:** stale absolute-path content / portability
- **Finding:** the chapter cites `/home/chakwong/python/src/dsge_hmc/filters/CUTSRUKF.py`, which is machine-specific and not monograph-stable.
- **Suggested correction:** replace it with a repo-relative or descriptive reference.

- **Anchor:** `\section{Conjugate Unscented Transform Rules}`
- **Issue class:** citation/derivation scope mismatch
- **Finding:** the degree-five Gaussian-moment exactness result is not explicitly separated from nonlinear filtering accuracy.
- **Suggested correction:** add a sentence stating that quadrature exactness does not by itself prove posterior or filter accuracy.

- **Anchor:** `\section{Evidence Ladder}`
- **Issue class:** stale source-project wording
- **Finding:** the evidence ladder is described as a consolidation of source-project validation plans.
- **Suggested correction:** restate it as BayesFilter’s adopted evidence ladder and footnote the historical provenance if needed.

### `docs/chapters/ch17_square_root_sigma_point.tex`

- **Anchor:** `\section{Backend Contract}`
- **Issue class:** readability
- **Finding:** the chapter would teach faster if it explicitly contrasted factor-propagating SR-UKF with covariance-reconstructing “square-root” implementations.
- **Suggested correction:** add a one-line contrast or small mini-table.

- **Anchor:** `\section{Factor-Propagating SR-UKF Score Contract}`
- **Issue class:** density / teaching flow
- **Finding:** too many concepts are introduced in one continuous block.
- **Suggested correction:** add a roadmap paragraph and possibly an end-of-section validation recap.

- **Anchor:** `\section{CUT With Square-Root Backends}`
- **Issue class:** stale absolute-path content
- **Finding:** it repeats the same machine-specific path problem as `ch16`.
- **Suggested correction:** replace the absolute path with a stable reference.

### `docs/chapters/ch18_svd_sigma_point.tex`

- **Anchor:** whole chapter
- **Issue class:** density / policy framing / source-project bleed
- **Finding:** the chapter is mathematically substantive, but it repeatedly leans on source-project audit language and is long enough that its main lesson can blur.
- **Suggested correction:** keep the central lesson—value-side robustness versus derivative fragility—but replace transient provenance phrases with exact anchors or clearly labeled project evidence.

### `docs/chapters/ch18b_structural_deterministic_dynamics.tex`

This is one of the strongest chapters in substance and one of the weakest in editorial discipline.

- **Anchor:** chapter intro
- **Issue class:** overclaim wording
- **Finding:** “this chapter proves the law-level reason” is too strong for a chapter that mixes proofs, examples, and project lessons.
- **Suggested correction:** narrow this to “derives and illustrates the law-level reason,” or enumerate which claims are formally proved.

- **Anchor:** `\section{Standard UKF, Structural UKF, and the Predictive Law}`
- **Issue class:** chapter organization
- **Finding:** the main contrast is buried under long subsections.
- **Suggested correction:** surface the main theorem roadmap immediately, then treat algorithms and examples as support.

- **Anchor:** `\subsection{The Original Unscented-Transform Pattern}`
- **Issue class:** claim precision
- **Finding:** the statement about improving on first-order linearization should be more narrowly qualified.
- **Suggested correction:** bound it to local moment-approximation improvement under stated smoothness assumptions.

- **Anchor:** `\subsection{UKF in Sequential-Monte-Carlo Kernel Notation}`
- **Issue class:** readability density
- **Finding:** this subsection is effectively a mini-chapter.
- **Suggested correction:** split it into exact structural kernel versus projected Gaussian-kernel subsections, and move side cautions into boxed remarks or cross-references.

- **Anchor:** lines around the “actual stochastic-volatility rows” paragraph
- **Issue class:** contextual intrusion / stale cross-topic content
- **Finding:** an abrupt actual-SV digression interrupts the chapter’s structural DSGE narrative.
- **Suggested correction:** move it to a footnote, remark, or cross-reference.

- **Anchor:** `\subsection{Why the Sigma-Point Variable Uses Pre-Transition Uncertainty}`
- **Issue class:** proof-boundary clarity
- **Finding:** the exact pushforward-equivalence condition is central and should be highlighted more strongly.
- **Suggested correction:** restate the decisive equivalence condition and forward-reference it explicitly.

- **Anchor:** proposition on UT accuracy
- **Issue class:** claim-scope clarity
- **Finding:** readers may overread a local small-uncertainty moment statement as a general UKF endorsement.
- **Suggested correction:** add a bold remark that this is a local moment statement, not a global filtering-accuracy theorem.

- **Anchor:** `\section{Worked Example A: Nonlinear Structural Transition}` and `\subsection{Numerical Illustration}`
- **Issue class:** repetition / unsupported numeric provenance
- **Finding:** the failure modes are repeated across prose and example, and the numerical values are not explicitly sourced as hand-computed or script-generated.
- **Suggested correction:** compress repetition and add provenance for the numeric witness values.

- **Anchor:** `\section{Worked Example B: Degenerate Linear Transition with Nonlinear Measurement}`
- **Issue class:** length / repeated failure analysis
- **Finding:** the same failure structure is re-explained at several layers.
- **Suggested correction:** abstract the repeated logic into a reusable lemma and refer back to it.

- **Anchor:** `\section{Structural Correctness Boundary}`
- **Issue class:** formatting/readability
- **Finding:** this is important but visually too dense.
- **Suggested correction:** convert the “does not guarantee” items into a boxed checklist.

- **Anchor:** `\section{Structural Degeneracy, Numerical Degeneracy, and SVD}`
- **Issue class:** chapter bloat / density
- **Finding:** this section is large enough to be its own chapter.
- **Suggested correction:** split it into formal PSD-degenerate accuracy, collapsed-law diagnostics, and UKF/HMC implications.

- **Anchor:** `\section{Source-Project Lesson}`
- **Issue class:** stale/project-specific content in core exposition
- **Finding:** it reads like a project case-study memo embedded in a theory chapter.
- **Suggested correction:** move it to a case-study appendix or mark it explicitly as project evidence.

- **Anchor:** `\section{Validation Gates and Final Policy Rule}`
- **Issue class:** policy/document-role mixing
- **Finding:** governance language about what future BayesFilter reports may claim sits inside the theory chapter.
- **Suggested correction:** either label it explicitly as BayesFilter validation policy or move it into a checklist/policy appendix.

## Part V. Particle Foundations and Proposal Transport

### `docs/chapters/ch19_particle_filters.tex`

- **Anchor:** `\section{Nonlinear state-space model and filtering recursion}`
- **Issue class:** readability / duplicate derivation
- **Finding:** the marginal-likelihood factorization is stated twice in close succession.
- **Suggested correction:** keep one displayed derivation and compress the other to a short sentence.

- **Anchor:** `\section{The bootstrap particle-filter likelihood estimator}`
- **Issue class:** density / teaching flow
- **Finding:** the proof is careful but would teach better with an explicit proof roadmap.
- **Suggested correction:** add a three-step roadmap before the proof.

- **Anchor:** `\section{Degeneracy and effective sample size}`
- **Issue class:** source-discipline nuance
- **Finding:** the cited collapse results could be read too strongly as impossibility statements.
- **Suggested correction:** add one sentence distinguishing severe scaling pressure from impossibility of all proposal/transport improvements.

### `docs/chapters/ch19b_dpf_literature_survey.tex`

- **Anchor:** opening framing
- **Issue class:** naming / teaching flow
- **Finding:** the file name suggests a survey, but the chapter is really a derivation-led foundations chapter.
- **Suggested correction:** clarify that in the opening or retitle it internally as a foundations-from-the-literature chapter.

- **Anchor:** `\section{EDH under Gaussian closure}`
- **Issue class:** density
- **Finding:** the algebra is strong but reaches a long derivation before a small witness example.
- **Suggested correction:** add a 1D linear-Gaussian micro-example.

- **Anchor:** `\section{Stiffness and discretization}`
- **Issue class:** unsupported literature pointer
- **Finding:** the numerical-stiffness discussion is plausible but not locally cited.
- **Suggested correction:** add citations or explicitly mark the section as BayesFilter’s implementation-facing consequence of the earlier ODE structure.

### `docs/chapters/ch19c_dpf_implementation_literature.tex`

- **Anchor:** `\section{Li--Coates Algorithm 1 as an implementation contract}`
- **Issue class:** density / section length
- **Finding:** too many moving parts are introduced in one uninterrupted block.
- **Suggested correction:** split into covariance lifecycle, auxiliary versus actual paths, determinant accumulation, and resampling payload.

- **Anchor:** `\subsection{What the later OT chapters inherit from PF-PF}`
- **Issue class:** repetition
- **Finding:** the transport-object doctrine is repeated too closely to `ch32b`.
- **Suggested correction:** shorten it and frame it as a handoff lemma.

## Part VI. Differentiable Resampling and Transport

### `docs/chapters/ch32a_soft_differentiable_resampling.tex`

- **Status:** one of the clearest chapters in the set.

- **Anchor:** `\section{Soft resampling rule}`
- **Issue class:** source discipline
- **Finding:** the pedagogical interpolation rule is not clearly marked as either a direct simplification of a cited source or a BayesFilter toy surrogate.
- **Suggested correction:** state the provenance of the specific rule explicitly.

- **Anchor:** `\section{Verification contract}`
- **Issue class:** minor omission
- **Finding:** the verification list does not explicitly call for comparison against hard resampling under the same fixed randomness.
- **Suggested correction:** add that check.

### `docs/chapters/ch32b_deterministic_ot_equalweighting.tex`

- **Anchor:** running three-particle cell
- **Issue class:** mathematical completeness
- **Finding:** the coupling used in the worked cell is feasible, but the chapter does not state whether it is also optimal.
- **Suggested correction:** either mark it explicitly as merely feasible or add a short optimality argument.

- **Anchor:** unregularized OT baseline section
n- **Issue class:** teaching flow
- **Finding:** the chapter should more explicitly tell the reader that the cell illustrates the transport object before the optimization problem is formally stated.
- **Suggested correction:** add a bridging sentence back to the running cell.

### `docs/chapters/ch32c_entropic_ot_sinkhorn.tex`

This is one of the most serious editorial problems in the current document.

- **Anchor:** `\subsection{Exact teacher versus exact engineering route}`
- **Issue class:** LaTeX correctness / duplicate subsection
- **Finding:** the subsection appears twice with the same label `sec:bf-eot-teacher-versus-engineering`.
- **Suggested correction:** remove the duplicate and keep one authoritative copy.

- **Anchor:** whole chapter
- **Issue class:** overlong dense section / scope drift
- **Finding:** the chapter mixes at least four chapter-scale ideas: entropic OT/Sinkhorn foundations; barycentric covariance loss; Contract E reset families; and higher-moment / TT-teacher extensions.
- **Suggested correction:** split it into multiple chapters or major files.

- **Anchor:** `\section{Why the current reset can accumulate bias}`
- **Issue class:** wording precision
- **Finding:** some “bias” language is stronger than what is locally proved.
- **Suggested correction:** prefer “finite-`N` approximation error” unless the bias direction is actually proved.

- **Anchor:** `\section{A complete higher-moment Contract E candidate}`
- **Issue class:** chapter scope drift
- **Finding:** this section becomes a full algorithm-design chapter inside the Sinkhorn chapter.
- **Suggested correction:** move it to its own chapter.

### `docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`

- **Anchor:** chapter opening
- **Issue class:** source-faithfulness wording
- **Finding:** “local exact Daum–Huang particle flow” is stronger than the surrounding exactness discipline allows.
- **Suggested correction:** rename it to a Daum–Huang-type or local-linearized flow unless exactness is explicitly scoped.

- **Anchor:** `\section{Canonical row quotient and Contract E composition}`
- **Issue class:** density
- **Finding:** the row quotient, pullback, normalization, and residual stage are introduced too compactly.
- **Suggested correction:** add a short bullet roadmap before the equations.

- **Anchor:** `\section{A proposed score-aware teacher-projection extension}`
- **Issue class:** chapter scope / proposal mixing
- **Finding:** the chapter starts as a custom-VJP chapter and then turns into a long research-proposal dossier.
- **Suggested correction:** split the Contract E–TP proposal and certification material into a separate chapter or appendix.

## Part VII. Learned Transport Extensions

### `docs/chapters/ch32d_retained_teacher_neural_ot.tex`

- **Anchor:** broader retained-teacher survey block
- **Issue class:** teaching flow / scope creep
- **Finding:** the warm-start chapter becomes a second survey chapter midway through.
- **Suggested correction:** keep the warm-start core in the main chapter and reduce the broader family survey to a classification table plus forward references.

### `docs/chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex`

- **Anchor:** labels near lines 506 and 552
- **Issue class:** LaTeX correctness / duplicate label
- **Finding:** `sec:bf-neural-ot-direct-scalable-boundary` is defined twice.
- **Suggested correction:** rename one label or merge the duplicated cross-reference target.

### `docs/chapters/ch32f_dynamic_geodesic_operator_learning_target_contract.tex`

- **Anchor:** labels near lines 166 and 504
- **Issue class:** LaTeX correctness / duplicate label
- **Finding:** `sec:bf-neural-ot-dynamic-same-scalar` is defined twice.
- **Suggested correction:** rename the later “revisited” section label.

- **Anchor:** labels near lines 356 and 432
- **Issue class:** LaTeX correctness / duplicated content / readability
- **Finding:** `sec:bf-neural-ot-dynamic-sliced-localized` is also defined twice, and adjacent sections repeat sliced/subspace/localization material.
- **Suggested correction:** merge the overlapping sections and keep one label.

## Part VIII. HMC Target Interpretation, Verification, and Filter Choice

### `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`

- **Status:** no material issue found in this audit.
- **Strengths:** largely aligned with current target-discipline and non-overclaim requirements.

### `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`

- **Anchor:** `\section{Proposal log-probability autodiff topology}`
- **Issue class:** readability / traceability
- **Finding:** the key topology lesson arrives only after a long numerical trace.
- **Suggested correction:** state the topology rule first as a boxed principle, then present the numerical trace as a worked example.

### `docs/chapters/ch20_filter_choice.tex`

- **Anchor:** `\section{Recommended Order}`
- **Issue class:** stale methodology / policy conflict
- **Finding:** the chapter still recommends attempting DSGE-scale SVD sigma-point HMC in a way that conflicts with the current principal-square-root promotion policy.
- **Suggested correction:** replace the SVD-forward recommendation with the promoted square-root route and keep SVD as diagnostic/comparison only.

## Part IX. High-Dimensional Nonlinear Filtering

Overall assessment: this block is much stronger on target-discipline than on first-pass readability.

### `docs/chapters/ch33_highdim_nonlinear_filtering_foundations.tex`

- **Anchor:** `\subsection{Actual-SV SR-UKF Augmented-Noise Adapter}`
- **Issue class:** pedagogical flow
- **Finding:** too implementation-heavy and model-specific for a foundations chapter.
- **Suggested correction:** keep the target-mismatch verdict here but move the full adapter derivation to a sigma-point appendix/chapter.

- **Anchor:** `\section{Compact Sensitivity Score For Finite LEDH--PFPF--OT}`
- **Issue class:** scope drift / stale organization
- **Finding:** technically strong, but it belongs more naturally to the transport lane than to high-dimensional foundations.
- **Suggested correction:** relocate it or reduce it to an exported-contract summary.

- **Anchor:** uses of “current” and “previous” lane language
- **Issue class:** stale content
- **Suggested correction:** replace transient sequencing words with stable route names.

### `docs/chapters/ch34_highdim_gaussian_projection_and_point_rule_foundations.tex`

- **Anchor:** `\section{Exact Filtering And Gaussian Projection}`
- **Issue class:** readability / abstraction density
- **Finding:** the chapter promises a running cell, but most of the derivation stays abstract until later chapters.
- **Suggested correction:** insert a tiny one-step 1D example earlier.

- **Anchor:** `\subsubsection{How This Matches The Jia--Xin--Cheng Gaussian Approximation Block}`
- **Issue class:** source anchoring
- **Finding:** the section names Jia/Xin/Cheng but does not give the exact source anchor.
- **Suggested correction:** cite the specific paper section/equation used.

### `docs/chapters/ch35_highdim_sparse_grid_quadrature_and_fixed_cloud_scalar.tex`

- **Anchor:** old internal labels such as `p31`, `p32`, `p34`, `p38`
- **Issue class:** stale content / maintenance hazard
- **Finding:** old internal label names remain throughout the chapter.
- **Suggested correction:** rename them to stable chapter-local labels.

- **Anchor:** 3D sparse-grid walkthrough
- **Issue class:** pedagogical completeness
- **Finding:** the chapter now gives the 3D reduction story, but still lacks a concrete explicit test function carried through the example.
- **Suggested correction:** carry one explicit `F(\xi_1,\xi_2,\xi_3)` through the dense-versus-sparse illustration.

- **Anchor:** `\subsection{Exactness, Point Count, And The UKF Relation}`
- **Issue class:** unsupported claim / policy compliance
- **Finding:** the text says it “uses the source exactness and UKF-relation results” without naming the exact theorem/equation or restating hypotheses.
- **Suggested correction:** add exact anchors or derive the low-level equivalence directly in BayesFilter notation.

### `docs/chapters/ch35b_highdim_fixed_cloud_filtering_and_sgqf_validation.tex`

- **Anchor:** overall middle-to-late structure
- **Issue class:** chapter altitude / pacing
- **Finding:** oracle, derivative derivation, branch contract, validation ladder, I/O contract, defaults, and full algorithm are all mixed into one chapter.
- **Suggested correction:** keep oracle + derivative + same-branch contract in the main chapter and move lower-level implementation/default details to an appendix or technical note.

- **Anchor:** `\section{Lane-Specific Validation Burden}`
- **Issue class:** stale content
- **Finding:** old internal planning phrases remain.
- **Suggested correction:** rewrite them in timeless explanatory language.

### `docs/chapters/ch36_highdim_low_rank_density_filters_and_kr_maps.tex`

- **Anchor:** `\section{Running Example And Notation}`
- **Issue class:** pedagogical flow
- **Finding:** the chapter promises a running example but stays mostly symbolic.
- **Suggested correction:** add a tiny retained-object example, even in two coordinates.

- **Anchor:** `\section{TT Sequential Learning And Conditional KR Transports}`
- **Issue class:** source anchoring
- **Finding:** Zhao/Cui is named but exact section/algorithm anchors are missing.
- **Suggested correction:** cite the exact source locations used.

### `docs/chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`

- **Anchor:** stale line names such as “operational middle of the p50 lane”
- **Issue class:** stale content
- **Suggested correction:** replace old workflow markers with chapter-local names.

- **Anchor:** `\section{Consolidated Fixed Least-Squares Formulation}`
- **Issue class:** readability / derivation density
- **Finding:** the chapter reaches deep tensor notation before a concrete toy branch arrives.
- **Suggested correction:** introduce a very small toy branch first.

- **Anchor:** initialization sections versus later stored-object walkthrough
- **Issue class:** internal inconsistency
- **Finding:** one section prefers scout-derived warm starts when available, while a later flow still hard-codes deterministic constant-channel initialization.
- **Suggested correction:** unify the initialization policy or make the distinction explicit.

### `docs/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`

- **Anchor:** `\section{Finite-Difference Contract And Branch-Validity Logic}`
- **Issue class:** mathematical correctness / notation consistency
- **Finding:** the direction index switches from `e_i` to `e_j`, making the formal contract ambiguous.
- **Suggested correction:** use one index consistently, almost certainly `e_i`.

- **Anchor:** same section
- **Issue class:** pedagogical completeness
- **Finding:** the I/K/W classification would teach better with a tiny worked example.
- **Suggested correction:** add a three-row toy table for branch-identity failure, copied-core failure, and decreasing-window success.

### `docs/chapters/ch38_highdim_validation_defect_calculus_and_promotion.tex`

- **Anchor:** benchmark/promotion sections
- **Issue class:** policy completeness
- **Finding:** the chapter is good on veto-first logic, but it does not explicitly state that runtime/ESS/tail metrics are descriptive only unless uncertainty analysis supports ranking.
- **Suggested correction:** add an explicit inference-status rule or table.

- **Anchor:** opening and restart language
- **Issue class:** stale content
- **Suggested correction:** remove “restarted” migration language and present the architecture as the live one.

## Part X. HMC, Geometry, and Diagnostics

### `docs/chapters/ch21_hmc_for_state_space.tex`

- **Status:** no material issue found in this audit.

### `docs/chapters/ch22_mass_matrices.tex`

- **Status:** no material issue found in this audit.

### `docs/chapters/ch23_boundary_gradients.tex`

- **Anchor:** `\section{Finite Invalid Returns}`
- **Issue class:** policy/methodology ambiguity
- **Finding:** the section blurs mathematically declared out-of-support states with implementation or numerical failures.
- **Suggested correction:** split them into separate cases, with numerical failures treated as diagnostics rather than ordinary rejection semantics.

### `docs/chapters/ch24_xla_jit.tex`

- **Anchor:** `\section{Explicit CPU Multiprocess Cloud Evaluation}`
- **Issue class:** policy clarity / readability
- **Finding:** GPU-hiding requirements and GPU memory-growth rules are conflated in a way that can read as if CPU-only workers need GPU memory growth.
- **Suggested correction:** separate CPU-only import discipline from the GPU memory-growth rule.

### `docs/chapters/ch25_diagnostics.tex`

- **Status:** no material issue found in this audit.

### `docs/chapters/ch26_transport_surrogates.tex`

- **Anchor:** `\section{NeuTra Position}`
- **Issue class:** stale governance/process wording
- **Finding:** “The Phase 2B literature gate accepted NeuTra…” is internal campaign jargon rather than self-contained exposition.
- **Suggested correction:** replace it with a direct technical status statement and optionally footnote the historical decision note.

### `docs/chapters/ch26b_neutra_transport_hmc.tex`

- **Anchor:** `\section{Additional historical model tests}`
- **Issue class:** teaching flow / readability
- **Finding:** the historical catalog interrupts the cleaner theory-to-evidence arc.
- **Suggested correction:** compress it into a summary table and move details to an appendix or result note.

- **Anchor:** `\section{Interpretation, failure modes, and nonclaims}`
- **Issue class:** unsupported external-strength wording
- **Finding:** the chapter’s summary overstates the breadth of the supporting evidence.
- **Suggested correction:** separate common-protocol evidence from heterogeneous historical context and weaken the cross-geometry summary language.

### `docs/chapters/ch26c_hnn_surrogate_hmc.tex`

- **Anchor:** corrected-kernel experiment sections
- **Issue class:** policy clarity / methodology presentation gap
- **Finding:** warm-up handling is not stated sharply enough, and the result presentation lacks the explicit inference-status table used elsewhere in governance.
- **Suggested correction:** say directly that warm-up was archived but excluded from posterior estimates, and add a decision/inference-status table.

## Part XI. Industrial Applications and Case Studies

### `docs/chapters/ch27_lgssm_validation.tex`

- **Anchor:** `\section{Evidence Already Available}`
- **Issue class:** provenance/support gap
- **Finding:** “strongest local evidence stream” is not tied to exact source-map or result-note anchors.
- **Suggested correction:** cite the concrete supporting artifacts.

### `docs/chapters/ch28_nonlinear_ssm_validation.tex`

- **Anchor:** Model B and Model C setup sections
- **Issue class:** terminology / mathematical precision
- **Finding:** “reference oracle” overstates what is really a dense Gaussian-moment comparator, and `x_1 ~ N(0, 0.2)` is ambiguous between variance and standard deviation.
- **Suggested correction:** rename the comparator and disambiguate the distribution parameterization.

- **Anchor:** `\section{Current V1 Filter Evidence}`
- **Issue class:** provenance/support gap
- **Finding:** strong summary phrases like “score parity is certified” lack local artifact anchors.
- **Suggested correction:** cite the exact tests/result notes and branch conditions.

### `docs/chapters/ch28a_neural_network_state_space_model_applications.tex`

This is another major editorial problem chapter.

- **Anchor:** overall chapter structure
- **Issue class:** teaching-flow / scope overload
- **Finding:** the chapter tries to do four jobs at once: model definition, predictive-equivalence theory, experiment chronology/results, and matrix-free derivative design.
- **Suggested correction:** split it into at least two units, or add a strong roadmap and clearer section transitions.

- **Anchor:** predictive-feature and dependence-aware sections
- **Issue class:** readability
- **Finding:** the exposition moves too quickly from predictive law to features to HAC covariance to practical-equivalence decisions.
- **Suggested correction:** add a “test construction roadmap” paragraph.

- **Anchor:** direct SSL-LSTM tangent recursion
- **Issue class:** notational correctness
- **Finding:** the observation tangent appears to use `C \dot z` where the preceding notation suggests `C \dot z^+`.
- **Suggested correction:** fix the symbol or explicitly redefine it there.

- **Anchor:** experiment-history and result sections
- **Issue class:** provenance/support gap
- **Finding:** many quantitative claims are not tied to explicit local result-note anchors.
- **Suggested correction:** add direct artifact references throughout the result summaries.

### `docs/chapters/ch29_nk_svd_case_study.tex`

- **Status:** no material issue found in this audit.
- **Strengths:** concise, readable, and disciplined about separating passed, blocked, and lesson layers.

### `docs/chapters/ch30_cip_afns_case_study.tex`

- **Anchor:** `\section{DZ5 Fixed-Center Curvature Evidence}`
- **Issue class:** provenance/support gap
- **Finding:** numerical claims and interpretations are not tied to local result-note anchors.
- **Suggested correction:** add explicit local result-note/source-map references.

### `docs/chapters/ch31_nawm_design_target.tex`

- **Status:** no material issue found in this audit.
- **Strengths:** it correctly presents NAWM as a design target and hypothesis rather than completed evidence.

### `docs/chapters/ch32_production_checklist.tex`

- **Anchor:** whole chapter
- **Issue class:** policy completeness gap
- **Finding:** the checklist omits several required manifest/evidence-contract items from current governance: git commit, command, environment, CPU/GPU status, seeds, artifact paths, plan/result paths, baseline/comparator, and explicit nonclaims.
- **Suggested correction:** add a manifest-and-evidence-contract subsection or integrate those items into the checklist itself.

## Appendices

### `docs/appendices/app_a_notation.tex`

- **Anchor:** notation for backend execution
- **Issue class:** policy-wording mismatch
- **Finding:** the appendix says chapters discussing “TensorFlow or JAX execution” may store tensors differently, but JAX is an exception-only backend in current policy.
- **Suggested correction:** replace “TensorFlow or JAX” with “TensorFlow execution or another explicitly approved backend.”

### `docs/appendices/app_b_matrix_calculus.tex`

- **Status:** no material issue found in this audit.

### `docs/appendices/app_c_factor_derivative_proofs.tex`

- **Issue class:** derivation completeness
- **Finding:** this appendix is explicitly a placeholder and is not publication-ready if the proof spine is supposed to live in the monograph.
- **Suggested correction:** either relabel it as an outline/roadmap appendix or fill the promised proofs before promotion.

### `docs/appendices/app_d_mathdevmcp_workflows.tex`

- **Status:** no material issue found in this audit.

### `docs/appendices/app_e_researchassistant_workflows.tex`

- **Anchor:** workspace-policy wording
- **Issue class:** minor wording/formatting
- **Finding:** the phrase about durable decisions belonging in `docs/plans` result notes is awkwardly formatted.
- **Suggested correction:** use consistent TeX path formatting such as `\path{docs/plans}`.

### `docs/appendices/app_f_source_map.tex`

- **Status:** no material issue found in this audit.

### `docs/appendices/app_g_experiment_templates.tex`

- **Issue class:** policy compliance / template under-specification
- **Finding:** the templates are too light relative to current evidence-governance requirements.
- **Suggested correction:** extend each template with required fields for question/comparator, promotion criterion, vetoes, nonclaims, budget, artifact root, command/env/seeds/hardware, plan/result paths, and next-step decision table.

---

# High-priority fix queue

## Tier 1: real correctness and integrity defects

1. Remove duplicate labels and duplicate subsection content in:
   - `docs/chapters/ch32c_entropic_ot_sinkhorn.tex`
   - `docs/chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex`
   - `docs/chapters/ch32f_dynamic_geodesic_operator_learning_target_contract.tex`
2. Fix the `e_i`/`e_j` notation mismatch in `docs/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`.
3. Clarify the under-specified auxiliary block `a_t` in `docs/chapters/ch02_state_space_contracts.tex`.
4. Remove absolute filesystem path references from `docs/chapters/ch16_sigma_point_filters.tex` and `docs/chapters/ch17_square_root_sigma_point.tex`.

## Tier 2: monograph-discipline repairs

5. Replace unnamed or transient internal provenance in `ch09` through `ch14` with exact source anchors or local derivations.
6. Rewrite `ch01`’s reader map to match the actual current document.
7. Separate theory exposition from project-governance or source-project-history content, especially in `ch18b`, `ch26*`, and `ch32_production_checklist`.
8. Add explicit local artifact anchors to quantitative case-study and validation claims in `ch27`, `ch28`, `ch28a`, and `ch30`.

## Tier 3: chapter-level restructuring

9. Split or heavily reorganize:
   - `docs/chapters/ch18b_structural_deterministic_dynamics.tex`
   - `docs/chapters/ch28a_neural_network_state_space_model_applications.tex`
   - `docs/chapters/ch32c_entropic_ot_sinkhorn.tex`
   - `docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`
10. Reduce drift by deciding what to do with non-included shadow chapters and restart-staging files.

# Strongest parts of the current document

To avoid losing what already works well, these were among the strongest audited pieces:

- `docs/chapters/ch05_prediction_error_decomposition.tex`
- `docs/chapters/ch19_particle_filters.tex`
- `docs/chapters/ch32a_soft_differentiable_resampling.tex`
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`
- `docs/chapters/ch29_nk_svd_case_study.tex`
- `docs/chapters/ch31_nawm_design_target.tex`

These chapters are not uniformly perfect, but they are comparatively strong in scope control, claim discipline, and teaching clarity.

# Bottom line

The document already contains enough substantive material for a strong monograph, but it currently mixes:

- durable mathematics,
- project-history evidence,
- live governance language,
- donor/source-project provenance,
- and chapter drafts of very different editorial maturity.

The next serious pass should prioritize:

1. cleaning real LaTeX and notation defects;
2. replacing transient provenance with exact anchors or local derivations;
3. splitting the most overloaded chapters; and
4. re-establishing a linear teaching flow in the chapters that now read like merged research memos.
