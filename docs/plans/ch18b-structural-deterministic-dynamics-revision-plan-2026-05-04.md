# Plan: strengthen Chapter 18b structural deterministic dynamics exposition

## Date

2026-05-04

## Scope

This is a bounded documentation plan for revising
`docs/chapters/ch18b_structural_deterministic_dynamics.tex`.

The goal is not to change the chapter’s core policy conclusion. The goal is to
make the derivation more explicit, remove common conceptual misunderstandings,
and distinguish carefully between:

- exact structural filtering claims;
- exact linear-Gaussian degenerate-transition claims;
- approximate nonlinear filtering choices;
- artificial transition modifications that change the model.

## Motivation

The current chapter is broadly correct, but it is still too easy for a careful
reader to misread key points.

In particular, the current exposition does not yet make explicit enough that:

1. a deterministic endogenous coordinate means no independent innovation in that
   coordinate, not zero predictive variance conditional on the past;
2. the nonlinear filtering target is the pushforward law induced by the
   structural transition map from lagged state and current shock variables;
3. exact linear Kalman filtering can tolerate singular or rank-deficient
   transition covariance because it works with the induced conditional Gaussian
   law, not because structural determinism disappears;
4. adding artificial noise to deterministic coordinates defines a different
   state-space model unless it is declared as an approximation or proposal with
   correction semantics.

This plan turns those points into a concrete rewrite ladder.

## Inputs

Primary chapter under revision:

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex`

Related BayesFilter plans and framing documents:

- `docs/plans/bayesfilter-structural-ssm-monograph-consolidation-pass-plan-2026-05-04.md`
- `docs/plans/bayesfilter-six-step-structural-filtering-closure-plan-2026-05-04.md`
- `docs/plans/dsge-structural-filtering-refactor-plan-2026-05-03.md`
- `docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`
- `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`
- `docs/source_map.yml`

Source literature already cited in the chapter and to be used conservatively:

- `herbst2015bayesian`
- `an2007bayesian`
- `kim2008calculating`
- `andreasen2018pruned`
- `durbin2012time`
- `gordon1993novel`
- `doucet2001sequential`
- `andrieu2010particle`
- `julier1997new`

Local tools that may help in bounded ways:

- MathDevMCP for small equation-label and proof-obligation audits;
- research-assistant for paper-source retrieval and section-level citation
  support if any claim expansion requires re-checking cited sources.

## Non-goals

This pass should not:

- change the chapter’s main policy conclusion unless a source-backed derivation
  requires it;
- add unsupported claims about full DSGE classes, HMC correctness, or empirical
  convergence;
- introduce new filtering algorithms or package APIs;
- add long literature surveys when a short source-backed clarification is
  enough;
- rewrite unrelated chapters except for minimal cross-references if needed.

## Core diagnosis to address

The rewrite should explicitly resolve the following exposition weaknesses.

### D1. No-independent-innovation versus no-predictive-variance

The chapter currently risks letting readers confuse:

- “this coordinate has no independent shock at time `t`”, and
- “this coordinate has zero conditional variance given `x_{t-1}`”.

For mixed structural transitions, the first can hold while the second is false
because the coordinate inherits randomness through current exogenous shocks.

### D2. Missing formal statement of the predictive law

The chapter says the filter must respect the structural map, but it does not yet
state the predictive distribution as the pushforward of the joint law over
lagged state and current shocks. That should be made explicit.

### D3. Linear exactness versus nonlinear misuse

The chapter correctly warns that linear Kalman success can hide a structural
mistake, but it should separate more sharply:

- exact linear-Gaussian degenerate-transition filtering;
- assumed-density approximation to the same target law; and
- artificial transition modification by adding independent noise to
  deterministic coordinates.

### D4. UKF source-of-randomness explanation

The worked UKF example is numerically correct, but it should extract the general
rule that sigma points belong in the pre-transition uncertainty variables,
usually `(x_{t-1}, \varepsilon_t)` or a declared stochastic block, not in an
artificially padded full post-transition state.

### D5. “Off-manifold” wording

The chapter’s geometric language is intuitive, but it should avoid overstating a
smooth-manifold claim when “model-implied support”, “structurally admissible
image”, or “constraint set induced by the transition map” is more precise.

## Execution cycle

Use the standard documentation research cycle:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

Each phase should end with an explicit go/no-go judgment on whether the next
phase remains justified. The reset-memo update is part of the pass contract,
not an optional afterthought.

## Additional execution constraints

1. Treat Chapter~\ref{ch:bf-state-space-contracts} and
   Chapter~\ref{ch:bf-api-design} as terminology and metadata source-of-truth
   chapters for this pass.
2. Audit nearby consistency with the sigma-point, particle, filter-choice, and
   production-checklist chapters before promoting any revised Chapter 18b policy
   language.
3. Classify every strengthened claim as one of:
   - local derivation in BayesFilter notation;
   - restatement of an existing BayesFilter contract chapter;
   - source-backed literature claim;
   - BayesFilter implementation or release-gating policy.
4. Treat MathDevMCP or research-assistant abstention as an evidence boundary,
   not as silent support.
5. Keep the final chapter compact; prefer short clarifying paragraphs and
   cross-references over a long FAQ.

## Phase C18b-0: baseline and evidence check

### Objective

Freeze the exact chapter revision target, record what is already justified
versus what still needs source re-checking, and incorporate the independent
plan-audit findings before prose edits begin.

### Actions

1. Re-read the full chapter and mark every place where the exposition moves from
   structural statement to mathematical claim.
2. Re-read the governing contract and metadata chapters:
   - `docs/chapters/ch02_state_space_contracts.tex`;
   - `docs/chapters/ch04_bayesfilter_api.tex`.
3. Re-read nearby policy chapters for consistency:
   - `docs/chapters/ch16_sigma_point_filters.tex`;
   - `docs/chapters/ch18_svd_sigma_point.tex`;
   - `docs/chapters/ch19_particle_filters.tex`;
   - `docs/chapters/ch20_filter_choice.tex`;
   - `docs/chapters/ch32_production_checklist.tex`.
4. Classify each Chapter 18b claim as one of:
   - exact derivation from the chapter’s own equations;
   - standard state-space fact already supported by cited sources;
   - implementation policy statement;
   - empirical or case-specific statement needing softer wording.
5. Record whether MathDevMCP and research-assistant are materially useful for
   this pass and where their evidence boundaries lie.
6. Tighten this plan if the audit reveals missing provenance, consistency, or
   validation steps.

### Pass gate

- A bounded list of claims to tighten is identified before any prose expansion.
- The plan explicitly covers contract reconciliation, nearby-chapter
  consistency, provenance classification, semantic claim audit, and reset-memo
  requirements.
- A go/no-go judgment is recorded for Phase C18b-1.

## Phase C18b-1: contract and notation reconciliation

### Objective

Align Chapter 18b vocabulary with the structural state-partition and metadata
contracts before substantive prose expansion.

### Actions

1. Map the chapter’s DSGE-facing notation `(m_t,k_t)` onto the structural
   contract language `(s_t,d_t,a_t)` without sacrificing readability.
2. Ensure the chapter uses or explicitly references:
   - structural state partition;
   - stochastic block;
   - deterministic-completion block;
   - integration space;
   - approximation label.
3. Add or tighten cross-references to
   Chapter~\ref{ch:bf-state-space-contracts} and
   Chapter~\ref{ch:bf-api-design} where needed.
4. Check whether the existing AR(2)/lag-stack example in Chapter 2 should be
   cross-referenced instead of re-explained in generic terms.

### Pass gate

- Chapter 18b uses the same contract language as the rest of the monograph.
- A go/no-go judgment is recorded for Phase C18b-2.

## Phase C18b-2: strengthen the structural target statement

### Objective

Make the chapter explicit about what law the nonlinear filter is approximating.

### Actions

1. Add a short formal statement after the structural split section that the
   transition is a measurable map from lagged state and current shock variables
   to current state.
2. Write the predictive law as the pushforward/integral representation over
   `(x_{t-1}, \varepsilon_t)`.
3. State explicitly that deterministic endogenous coordinates are computed
   pointwise from the structural map, not assigned independent transition noise.
4. Keep the exposition readable; one displayed formula plus short explanatory
   prose is enough.
5. Record whether the new statement is being presented as local BayesFilter
   derivation, as a restatement of an existing contract chapter, or as a
   literature-backed claim.

### Pass gate

- A reader can see exactly what object the nonlinear filter is meant to
  approximate.
- A go/no-go judgment is recorded for Phase C18b-3.

## Phase C18b-3: add the critical conditional-variance clarification

### Objective

Remove the most common misunderstanding about deterministic endogenous blocks.

### Actions

1. Add a dedicated paragraph or subsection titled along the lines of:
   - “No independent innovation does not mean zero predictive variance”, or
   - “Deterministic completion versus inherited randomness”.
2. Use the existing toy model or a minimal linear variant to show that
   `k_t = T_k(\cdot)` may still satisfy
   `\operatorname{Var}(k_t \mid x_{t-1}) > 0`.
3. State the distinction in words and notation.
4. Ensure later uses of “deterministic coordinate” are qualified consistently.
5. Prefer a cross-reference to the Chapter 2 lag-stack example if it avoids
   redundant generic exposition.

### Pass gate

- The chapter cannot be fairly read as claiming that deterministic completion
  implies zero one-step predictive variance.
- A go/no-go judgment is recorded for Phase C18b-4.

## Phase C18b-4: refine the linear Kalman section

### Objective

Make precise why the linear exact case can tolerate degenerate transitions while
still supporting the chapter’s warning about nonlinear misuse.

### Actions

1. Rewrite the beginning of the linear Kalman section to say that exact linear
   filtering is valid when the induced conditional law is the correct linear
   Gaussian law, even if its covariance is singular or rank-deficient.
2. Separate clearly:
   - representation of a degenerate conditional Gaussian law;
   - numerical regularization for stable computation; and
   - structural alteration of the model.
3. Add one sentence explaining that the success of moment-based linear filtering
   does not justify adding fictitious independent shocks in a nonlinear
   transition approximation.
4. Keep source claims conservative and avoid over-general numerical claims not
   backed by `durbin2012time` or standard state-space facts.
5. Audit any use of words such as `exact`, `correct`, `singular`, and
   `regularization` for unintended overclaim.

### Pass gate

- The section no longer sounds like “collapse is always harmless”; it instead
  says “collapse is acceptable when it preserves the same conditional law in the
  exact linear-Gaussian setting”.
- A go/no-go judgment is recorded for Phase C18b-5.

## Phase C18b-5: strengthen the nonlinear sigma-point and particle sections

### Objective

Turn the current policy warning into a mathematically cleaner algorithmic rule.

### Actions

1. In the nonlinear-filter section, state the integration variables explicitly.
2. Explain that sigma points or particles should be generated in the declared
   stochastic integration space, then lifted through the structural transition.
3. Add one compact particle-filter formula to parallel the UKF discussion.
4. Distinguish between:
   - exact structural propagation;
   - assumed-density or quadrature approximation to that same target; and
   - altered transition laws with artificial noise.
5. Replace or qualify “off-manifold” language where a more precise support-based
   description is better.
6. Audit consistency with:
   - `docs/chapters/ch16_sigma_point_filters.tex`;
   - `docs/chapters/ch18_svd_sigma_point.tex`;
   - `docs/chapters/ch19_particle_filters.tex`;
   - `docs/chapters/ch20_filter_choice.tex`.

### Pass gate

- The chapter distinguishes target-law approximation from model-changing noise
  injection.
- A go/no-go judgment is recorded for Phase C18b-6.

## Phase C18b-6: upgrade the UKF worked example into a general lesson

### Objective

Make the worked example teach the reusable rule, not just the arithmetic.

### Actions

1. Add a short preface before the example explaining why the augmented UKF
   variable is `(x_{t-1}, \varepsilon_t)`.
2. After the numerical example, add a short summary paragraph extracting the
   general rule for structural UKF/CKF design.
3. Keep the existing numeric table and values unless a real correction is found.
4. If useful, add one sentence clarifying that `k_t` has no independent new
   innovation but does inherit uncertainty through `m_t`.
5. Frame the extracted lesson as a BayesFilter structural-design rule for mixed
   structural models, not as a universal theorem about all UKF formulations.

### Pass gate

- A reader can generalize the example to DSGE adapters without guessing the
  principle.
- A go/no-go judgment is recorded for Phase C18b-7.

## Phase C18b-7: audit the required-tests section and policy language

### Objective

Make the chapter’s policy outputs precise, non-duplicative, and aligned with
BayesFilter promotion/claim discipline.

### Actions

1. Audit the “Required tests before any DSGE nonlinear filter claim” section
   against:
   - `docs/chapters/ch16_sigma_point_filters.tex`;
   - `docs/chapters/ch19_particle_filters.tex`;
   - `docs/chapters/ch32_production_checklist.tex`.
2. Tighten wording so policy requirements are not misread as already completed
   implementation evidence.
3. Harmonize the strongest claims with BayesFilter labels such as `value-valid`,
   `gradient-valid`, `sampler-usable`, `converged`, and `production-ready`
   where relevant.
4. Add a compact common-misunderstandings subsection only if it clarifies the
   chapter without duplicating Chapter 2 or Chapter 32.
5. Run a semantic claim audit on words such as `production-ready`, `correct`,
   `exact`, `guarantee`, `must always`, `converged`, `manifold`, and `support`.

### Pass gate

- The chapter’s policy language is consistent with the production-checklist
  discipline.
- A go/no-go judgment is recorded for Phase C18b-8.

## Phase C18b-8: provenance and reset-memo closure

### Objective

Make the chapter revision durable and reboot-safe.

### Actions

1. Update `docs/source_map.yml` if the revised chapter rationale, plan
   dependencies, or provenance classification changed.
2. Record whether strengthened Chapter 18b claims are local derivation,
   contract restatement, literature-backed, or BayesFilter policy.
3. Append a reset-memo entry for every completed phase and a final completion
   entry that records:
   - results;
   - interpretation;
   - whether the next phase remained justified;
   - what remains unresolved or intentionally approximate;
   - explicit next hypotheses to test.
4. Audit overlap with Chapters 2, 16, 19, and 32 so that the rewrite does not
   create avoidable duplication.

### Pass gate

- Provenance is explicit enough for another agent to continue after a reboot.
- A final go/no-go judgment is recorded for the completion/commit phase.

## Validation ladder

After the rewrite, run the smallest checks first.

1. `git diff --check`
2. Parse `docs/source_map.yml` if touched.
3. Build the monograph or the smallest relevant LaTeX target:
   - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
4. If labels or derivation blocks materially change, use MathDevMCP on a small
   bounded set of labels to surface unsupported derivation jumps.
5. If citations are expanded materially, use research-assistant only to fetch
   section-level support for those added claims.
6. Run semantic searches in the touched chapter for terms such as:
   - `deterministic`
   - `innovation`
   - `variance`
   - `noise`
   - `exact`
   - `approximation`
   - `production-ready`
   - `manifold`
   - `support`
7. Confirm any new Chapter 18b policy language is consistent with the nearby
   sigma-point, particle, filter-choice, and production-checklist chapters.

## Deliverables

1. Revised `docs/chapters/ch18b_structural_deterministic_dynamics.tex`
2. Updated `docs/source_map.yml` if provenance pointers change
3. Phase-by-phase and final reset-memo entries in
   `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`
4. Tightened execution plan at
   `docs/plans/ch18b-structural-deterministic-dynamics-revision-plan-2026-05-04.md`
5. Optional follow-up note only if the rewrite uncovers a genuine unresolved
   mathematical point requiring a separate audit

## Stop rules

Stop and ask for direction if any of the following happen:

- a key claim in the chapter turns out not to be supported either by direct
  derivation or by the cited sources;
- the rewrite would require introducing a stronger theorem-style claim than the
  sources justify;
- the existing policy conclusion appears to depend on a distinction that cannot
  be stated cleanly in the chapter’s notation;
- the rewrite cannot be reconciled cleanly with Chapters 2, 4, 16, 19, 20, or
  32 without a broader restructuring pass;
- provenance changes in `docs/source_map.yml` become ambiguous rather than
  clarifying;
- the “required tests” section would need to promise implementation evidence
  that does not exist and cannot be reframed as policy;
- the LaTeX rewrite requires broader monograph restructuring rather than a
  bounded chapter revision.

## Expected result

After this pass, Chapter 18b should read as a careful structural-filtering
argument rather than a correct but somewhat compressed warning note. A reader
should be able to see:

- what the exact target law is;
- why deterministic completion is compatible with inherited randomness;
- why singular linear-Gaussian transitions are not the same issue as nonlinear
  full-state noise injection; and
- what approximation labels are required when the backend intentionally departs
  from the structural transition.
