# Plan: comprehensive restructuring and literature-strengthening pass for Chapter 18b

## Date

2026-05-06

## Scope

This plan governs a full rewrite pass for
`docs/chapters/ch18b_structural_deterministic_dynamics.tex`.

It is not a local cleanup plan. It is a chapter-level restructuring pass aimed
at turning the current Chapter 18b from a correct but accreted argument into a
coherent, teachable, literature-grounded doctrine chapter.

The intended result is a chapter that another reader -- or another coding agent
-- can follow without already being expert in UKF, sigma-point filtering, or
DSGE timing structure.

## Motivation

The current chapter now contains several good ingredients:

- the structural split between exogenous shock-driven and endogenous
  deterministic-completion states;
- the distinction between exact linear-Gaussian collapse and nonlinear
  structural propagation;
- a structural UKF example;
- a degenerate linear-transition + nonlinear-measurement counterpart example;
- a distinction between structural-model failure and observation-side quadrature
  error;
- an SVD-related caution that structural degeneracy is not the same thing as a
  covariance-factorization problem.

However, the chapter still has the following weaknesses:

1. it reads like a sequence of accumulated patches rather than a clean argument;
2. the algorithm sections are still not explicit enough for reliable
   code-generation by another agent;
3. the literature support is too weak and too implicit;
4. the distinction between structural degeneracy, numerical degeneracy, and
   derivative-path risk is present but not cleanly organized;
5. the formal statements mix assumptions, exact derivations, and interpretation
   too freely;
6. the worked examples do not appear in the most pedagogically useful order;
7. the chapter still over-relies on prose where exact formulas or explicit
   algorithm steps would be clearer.

The reviewers' criticism that the chapter is abstract, hard to read, and light
on literature support is therefore substantially correct. This plan addresses
that directly.

## Primary objective

Rewrite Chapter 18b so that it:

- defines clearly what problem is being solved;
- reconstructs the original additive-noise UKF / unscented-transform logic from
  the literature;
- presents structural UKF as an explicit modification of the sigma-point
  variable and propagation rule, not as a mysterious alternative filter;
- distinguishes law-specification errors from moment-approximation errors and
  from covariance-factorization errors;
- connects every major claim to the appropriate literature class;
- gives another agent enough detail to implement the correct structural UKF path
  without guessing.

## Inputs and reference materials

### Primary chapter to rewrite

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex`

### BayesFilter contract and nearby policy chapters

- `docs/chapters/ch02_state_space_contracts.tex`
- `docs/chapters/ch16_sigma_point_filters.tex`
- `docs/chapters/ch17_square_root_sigma_point.tex`
- `docs/chapters/ch18_svd_sigma_point.tex`
- `docs/chapters/ch20_filter_choice.tex`
- `docs/chapters/ch28_nonlinear_ssm_validation.tex`
- `docs/chapters/ch32_production_checklist.tex`

### Literature/source materials already present in repo

- `docs/A general method for approximating non-linear transformations of probability distributions Julier(96).pdf`
- `docs/references.bib`

### Existing in-chapter reference set to preserve or refine

- `herbst2015bayesian`
- `an2007bayesian`
- `kim2008calculating`
- `andreasen2018pruned`
- `durbin2012time`
- `gordon1993novel`
- `doucet2001sequential`
- `andrieu2010particle`
- `julier1997new`

### Supporting process/provenance file to update

- `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`

## Non-goals

This pass should not:

- attempt full monograph-wide theorem normalization;
- settle every possible UKF variant in the literature;
- claim full machine certification of the chapter's proofs by MathDevMCP;
- introduce new package dependencies or global LaTeX preamble redesign unless
  absolutely necessary;
- rewrite unrelated chapters except for minimal cross-reference or terminology
  alignment if required.

## Core rewrite principles

1. **One chapter, one spine.**
   The reader should feel one argument developing, not multiple repair notes.

2. **Exact identities first, interpretation second.**
   Pushforward formulas and structural identities should be stated before policy
   conclusions derived from them.

3. **Structural law vs approximation quality vs numerical representation.**
   These must be kept distinct at all times.

4. **Algorithmic explicitness over rhetorical warning.**
   When a coding agent could misunderstand, prefer an explicit numbered step.

5. **Literature should be mapped by function.**
   Each paper/book cited should be used for a specific kind of claim.

6. **Prune before expanding.**
   Do not carry forward duplicated arguments, repeated theorem-like statements,
   or parallel algorithm explanations that no longer add new content.

## Mandatory Phase 0: inventory, pruning, and claim classification

Before rewriting prose, perform a pruning/classification pass on the current
chapter.

Required tasks:

1. inventory every current section, subsection, theorem-like block, algorithm,
   comparison table, and worked example;
2. classify each item as:
   - retain as-is with light edits,
   - merge into another section,
   - downgrade from proposition to remark/definition/example,
   - remove entirely;
3. classify each major claim as one of:
   - exact derivation,
   - accepted assumption,
   - source-backed literature claim,
   - BayesFilter implementation policy,
   - toy numerical illustration,
   - project-specific audit evidence;
4. identify duplicated content that must be removed before new material is added.

Pass gate:
- another agent can point to every current chapter block and say why it remains,
  moves, or is cut.

## Required chapter structure after rewrite

The chapter should be reorganized into the following sections in this general
order.

### C18b-R1. Problem statement and scope

Goal:
- open with the production warning and state exactly what this chapter is about.

Required content:
- mixed structural transitions;
- why sigma-point/particle filters are at risk if the wrong law is propagated;
- what this chapter will and will not prove.

### C18b-R2. Structural split and contract language

Goal:
- define the stochastic block and deterministic-completion block once and align
  with Chapter 2.

Required content:
- the model equations using `(m_t,k_t)` for this chapter;
- explicit connection to `State-Space Model Contracts` terminology;
- clear notation for what is stochastic and what is completed.

### C18b-R3. Exact linear-Gaussian collapse versus nonlinear structural propagation

Goal:
- explain why exact linear Kalman can sometimes collapse the transition without
  error, while nonlinear approximate backends cannot do so blindly.

Required content:
- singular or rank-deficient conditional Gaussian law;
- structural determinism versus numerical regularization;
- short, not repetitive.

### C18b-R4. Reusable structural filtering algorithm

Goal:
- provide one canonical backend-neutral algorithm block early in the chapter.

Required content:
- inputs from the model;
- integration variable;
- stochastic propagation;
- deterministic completion;
- observation evaluation;
- update backend;
- metadata / approximation label / diagnostics.

This section should be explicit enough that another agent could implement the
logic skeleton.

### C18b-R5. The original unscented-transform / additive-noise UKF pattern

Goal:
- reconstruct the original sigma-point logic from Julier (1996) and the UKF
  tradition in BayesFilter notation.

Required content:
- sigma-point construction from mean/covariance;
- transformed mean and covariance formulas;
- augmented-state idea for additive-noise models;
- what the original literature actually establishes.

This section should cite Julier (1996) explicitly and, where appropriate,
`julier1997new`.

### C18b-R6. Standard additive-noise UKF algorithm

Goal:
- write the standard additive-noise UKF in explicit numbered algorithm form.

Required content:
- augmented state/noise variable;
- sigma-point generation;
- state propagation;
- observation propagation;
- predicted moments;
- cross covariance;
- gain and posterior update.

This section should be detailed enough for code generation.

### C18b-R7. Structural UKF algorithm

Goal:
- write the structural UKF at the same algorithmic granularity as the standard
  UKF.

Required content:
- choice of sigma-point variable `(x_{t-1}, \varepsilon_t)` or equivalent
  structural pre-transition uncertainty;
- stochastic-block propagation;
- deterministic-completion propagation;
- observation evaluation;
- predicted moments;
- gain and posterior update.

This section must make clear that the update equations are the same and the
sigma-point variable / predictive-law construction is what differs.

### C18b-R8. Exact comparison of standard UKF versus structural UKF

Goal:
- prevent reader confusion by comparing the two algorithms line by line.

Required content:
- table or explicit bullets for:
  - filtered state;
  - integration variable;
  - augmented variable;
  - what gets perturbed;
  - what support is preserved;
  - when the method is structurally valid;
  - when it is only an approximation.

### C18b-R9. Core formal statement for the sigma-point variable

Goal:
- keep one main proposition about the predictive-law pushforward and the correct
  sigma-point variable.

Required content:
- accepted assumptions clearly separated from proved identities;
- exact pushforward identity for the predictive law;
- no unnecessary invertibility assumptions for the core result;
- separate remark/assumption about when one may reparameterize from
  `\varepsilon_t` to `m_t`.

Use the notation already adopted in the chapter. Keep formal claims narrow.

### C18b-R10. Worked example A: degenerate linear transition + nonlinear measurement

Goal:
- make the practically important case appear first.

Required content:
- linear but degenerate latent transition;
- nonlinear exponential-affine measurement map;
- structural and naive numerical updates side by side;
- explicit statement of which failures still apply and why;
- explicit distinction between structural-model failure and observation-side
  quadrature error.

### C18b-R11. Worked example B: nonlinear structural transition

Goal:
- show that the same structural UKF principle persists when the latent map is
  itself nonlinear.

Required content:
- reuse only what adds new value relative to Example A;
- avoid re-explaining the entire UKF doctrine again.

### C18b-R12. What structural correctness does and does not guarantee

Goal:
- add a short section clarifying the limit of the chapter's main doctrine.

Required content:
- structural correctness prevents targeting the wrong latent law;
- it does not make the Gaussian closure exact;
- it does not remove nonlinear observation quadrature error;
- it does not solve derivative/HMC safety problems;
- it does not solve SVD spectral autodiff issues.

### C18b-R13. Distinction table: structural degeneracy versus numerical degeneracy

Goal:
- answer the reviewer concern about SVD sigma-point filters directly.

Required content:
- structural degeneracy of latent law;
- numerical singularity / factorization issue;
- square-root/SVD representation issue;
- derivative-path / spectral-gap issue;
- approximation-label issue.

This section should state clearly that an SVD sigma-point backend may solve a
covariance-factorization problem without solving a structural-law problem.

### C18b-R14. Pruned DSGE and adapter implications

Goal:
- connect the doctrine back to DSGE application structure.

Required content:
- perturbation-order decomposition versus structural role decomposition;
- why `(x^f,x^s)` is not enough by itself;
- what a robust adapter must expose.

### C18b-R15. Source-project lesson / concrete failure mode

Goal:
- preserve the motivation from the source implementation audit without letting
  the chapter turn into a local postmortem.

Required content:
- short bounded statement of what the source-project path did;
- why it matters structurally;
- no overemphasis on project-local details.

### C18b-R16. Validation gates and final policy rule

Goal:
- end with a concise validation and release-gate section.

Required content:
- metadata / support / linear recovery / degenerate-row / toy reference /
  case-study documentation requirements;
- compact misunderstandings section if still needed;
- short final policy box.

## Literature mapping requirements

The rewrite must explicitly map claims to literature classes.

### L1. Unscented-transform / sigma-point mechanics
Use:
- Julier (1996) PDF under `docs/`
- `julier1997new`

Support these claims only:
- sigma-point mechanics;
- transformed moment approximation;
- augmented-state/additive-noise UKF pattern;
- accuracy framing of the unscented transform.

Do **not** use these citations as if they alone justify structural treatment of
mixed deterministic-completion models.

### L2. State-space filtering foundations
Use chapter-local or book citations already in the monograph to support:
- predictive law language;
- update-law language;
- conditional Gaussian reference path.

### L3. DSGE timing / structural split
Use:
- `herbst2015bayesian`
- `an2007bayesian`
- `kim2008calculating`
- `andreasen2018pruned`

Support these claims only:
- exogenous/endogenous timing split;
- perturbation/pruning structure;
- state-space representation after solution.

### L4. Particle/filtering references
Use:
- `gordon1993novel`
- `doucet2001sequential`
- `andrieu2010particle`

Support these claims only:
- particle propagation/weighting semantics;
- reference discussion of transition law versus proposal behavior.

### L5. Numerical/SVD/square-root distinction
Use nearby BayesFilter chapters and source-audit references to support:
- stable covariance factorization;
- what SVD/square-root methods buy numerically;
- what they do **not** solve structurally.

## UKF-variant discipline

The rewrite must explicitly fix which additive-noise UKF / unscented-transform
formulation is being reconstructed for exposition.  Do not write as if every
later UKF variant is identical.

Required discipline:
- state clearly which construction is reconstructed from Julier (1996);
- state which parts of BayesFilter notation abstract over later UKF variant
  choices;
- avoid claiming that every additive-noise UKF in the literature uses exactly
  the same augmentation, weighting, or covariance-update conventions.

## Theorem/assumption hygiene rules

1. Keep theorem budget tight.
   - One main proposition in the general structural-UKF part.
   - Example-specific formal statements should be downgraded to remarks,
     identities, or worked derivations unless indispensable.

2. Use explicit assumptions for:
   - innovation independence/factorization;
   - measurability / finite moments;
   - optional `m_t`-instead-of-`\varepsilon_t` reparameterization.

3. Use remarks for interpretive claims, not propositions, whenever possible.

4. Every formal statement should be classified in the rewrite notes as:
   - exact derivation,
   - accepted assumption,
   - source-backed claim,
   - implementation policy,
   - empirical/testable hypothesis.

## Algorithmic explicitness requirements

Every algorithm section must answer, explicitly:
- What is the filtered object?
- What is the sigma-point / integration variable?
- What is the augmented Gaussian object?
- What points are propagated through the latent transition?
- Which coordinates are completed deterministically?
- What observation points are evaluated?
- Which moments are computed?
- Which update equations are applied?
- What metadata and approximation labels are returned?

If a coding agent could not implement the correct logic from the prose without
guessing, the section is not detailed enough.

## Degeneracy and SVD distinction requirements

The rewrite must contain an explicit distinction that says:

- structural degeneracy = latent law has deterministic-completion coordinates;
- numerical degeneracy = covariance factorization is singular or ill-conditioned;
- SVD sigma-point filtering may address the second without automatically fixing
  the first;
- derivative-path / spectral-gap issues are yet another layer.

This distinction should be explicit enough to answer a skeptical reviewer who
believes that an SVD sigma-point filter already solves the problem.

## Example requirements

### Example A: degenerate linear transition + nonlinear measurement
Must include:
- structural update values;
- naive full-state update values;
- explicit connection to prediction-law, cross-covariance, and update-law
  failure;
- explicit identification of the observation-side-only error regime when the
  latent law is preserved.

### Example B: nonlinear structural transition
Must include:
- only material that is genuinely new relative to Example A;
- explicit statement of what additional complexity comes from nonlinear latent
  propagation rather than from the structural split itself.

## Execution instructions for another agent

Follow this order:

1. Re-read the current chapter and mark every duplicated claim.
2. Re-read the Julier (1996) PDF carefully enough to reconstruct the original
   additive-noise algorithm and the transform-accuracy discussion fairly.
3. Re-read Chapters 2, 16, 17, 18, 20, and 28 for contract, approximation,
   SVD, and validation language.
4. Produce a pruning/classification inventory before rewriting prose.
5. Rewrite the chapter section by section in the target order above.
6. Remove duplicated UKF explanations once the new algorithm and comparison
   sections are in place.
7. Add or refine citations in `docs/references.bib` only if needed and only
   after checking for key conflicts and claim support.
8. Run a full LaTeX build.
9. Audit the rewritten chapter for:
   - theorem/assumption hygiene,
   - algorithmic explicitness,
   - literature support,
   - stale labels/references,
   - repeated claims that survived the rewrite.
10. Update the reset memo with:
    - restructuring rationale,
    - literature additions and what they support,
    - any remaining evidence boundaries,
    - which claims were narrowed, cut, or downgraded.

## Verification

After the restructuring pass:
1. `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
2. Search the rewritten chapter for duplicated heavy phrases and remove obvious
   repeats.
3. Check that every major formal claim has either:
   - a proof,
   - an explicit accepted assumption,
   - or a literature citation.
4. Check that every major literature-backed claim is tied to the correct source
   class.
5. Check that the chapter explicitly distinguishes:
   - structural-law correctness,
   - observation quadrature error,
   - covariance-factor numerical issues,
   - derivative/HMC stability issues.
6. Confirm the algorithm sections are explicit enough for code-generation use.
7. Confirm example-order and cross-reference consistency after reordering.

## Expected outcome

After execution of this plan, Chapter 18b should become:
- easier to read;
- better connected to the literature;
- clearer about what exactly the problem is;
- explicit enough for a coding agent to implement the intended structural UKF
  logic;
- and much less vulnerable to the reviewer criticisms of abstraction, weak
  citation support, poor flow, and mixed evidence classes.
