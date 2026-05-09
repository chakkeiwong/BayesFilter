# Program: BayesFilter differentiable particle filter monograph rebuild

## Date

2026-05-09

## Status

This program supersedes the earlier lightweight DPF chapter-writing pass as the
controlling plan for the documentation work.  The previous pass produced prose
that was too governance-heavy, insufficiently mathematical, insufficiently
self-contained, and not rigorous enough for the intended R\&D monograph.

The corrected goal is to produce a mathematically serious DPF monograph block
that can stand beside the stronger chapters of the CIP monograph, while being
rewritten into BayesFilter notation and aligned with BayesFilter's eventual
implementation path.

## Purpose

Build a self-contained, mathematically rigorous documentation program for
BayesFilter's differentiable particle filter work.  The resulting chapters must:

1. survey the differentiable particle-filter literature thoroughly and
   accurately;
2. derive the main equations in a self-contained way or tie them precisely to
   source equations with explicit assumptions;
3. distinguish exact, unbiased, approximate, relaxed, and surrogate objects
   rigorously;
4. explain the implementation implications of each mathematical choice;
5. assess, in a mathematically disciplined way, whether the chosen path is
   appropriate for HMC estimation of nonlinear DSGE and MacroFinance models;
6. avoid reader-facing repo commentary, file-path talk, casual governance
   language, and informal verbal comparison in place of mathematics.

## Governing correction to the previous pass

The earlier three-chapter DPF block was not acceptable because it behaved like a
program memo rather than a monograph treatment.  It must be treated as a draft
artifact to be replaced or rewritten substantially.  The replacement work should
preserve only reusable source provenance and high-level scope decisions, not the
reader-facing exposition style.

## Reader and document standard

The intended reader is a mathematically mature R\&D reader who wants:

- exact state-space and filtering definitions;
- careful measure/probability statements where needed;
- explicit derivations or tightly sourced formulas;
- rigorous comparison among particle filters, particle-flow filters,
  differentiable resampling schemes, OT schemes, and HMC-related likelihood
  targets;
- a clear understanding of what can be implemented and what would still be only
  a research surrogate.

The reader is not primarily interested in:

- which local repo file contains what;
- internal project-management commentary;
- casual verbal notes about student work;
- implementation governance in place of mathematics.

## Main output target

The final deliverable is not assumed to be only three chapters.  The current
working assumption is a larger Part IV / Part V literature-and-theory block,
likely requiring more than three chapters to treat the topic adequately.

The final chapter architecture should emerge from the literature audit, but it
must at minimum cover:

- particle-filter mathematical foundations and likelihood estimators;
- particle-flow foundations (EDH, LEDH, stochastic / kernel / related flow
  variants);
- PF-PF importance correction and Jacobian/log-determinant machinery;
- differentiable resampling, including soft resampling and OT resampling;
- neural/amortized OT or learned resampling operators as approximations;
- implementation-oriented mathematical constraints;
- HMC target correctness and suitability for nonlinear DSGE and MacroFinance
  estimation.

## Primary source hierarchy

When sources disagree or differ in emphasis, use this order:

1. primary literature and source papers;
2. mathematically strong CIP monograph chapters and their cited source trail;
3. BayesFilter's existing mathematically disciplined chapters;
4. student reports and code, treated as comparison, critique, and coverage
   checks rather than production doctrine.

The student work is still useful, but in the monograph it should appear only
where it sharpens mathematical or implementation comparison.  The monograph
should not read like commentary on student work.

## Required source lanes

### Lane A: core literature survey

Use ResearchAssistant and primary-source review to build the literature basis
for:

- bootstrap / auxiliary / classical particle filtering;
- pseudo-marginal and particle-MCMC references relevant to likelihood status;
- EDH and related Daum--Huang particle-flow literature;
- LEDH / PF-PF / invertible particle-flow literature;
- kernel and stochastic particle-flow literature where relevant;
- differentiable particle filters for end-to-end learning;
- soft resampling literature;
- entropy-regularized OT resampling literature;
- learned / amortized transport or resampling operators where relevant;
- HMC / pseudo-marginal / approximate-target implications.

### Lane B: CIP monograph extraction and rewrite

Use the CIP monograph as a mathematically rich local source base, especially:

- `~/python/latex-papers/CIP_monograph/chapters/ch17_nonlinear_filtering.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch20_hmc.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch21_advanced_hmc.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch26_differentiable_pf.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch27_ledh_pfpf_neural_ot.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch32_diff_resampling_neural_ot.tex`

The task is not to copy those chapters.  It is to extract the mathematical
spine, verify it against the literature, and rewrite it into BayesFilter
notation and purpose.

### Lane C: student critique and coverage lane

Use the student report and repos to answer these questions:

- what mathematical topics do they cover that the current BayesFilter/CIP draft
  does not cover;
- what conclusions do they draw that differ from our intended exposition;
- which of those differences are justified by the literature and which are not;
- which implementation concerns surface in their experiments that should appear
  in the monograph's implementation or HMC chapters.

This lane is not for treating student work as authority.  It is for identifying
coverage gaps, overclaims, missed caveats, and useful implementation stress
points.

### Lane D: equation and derivation audit lane

Use MathDevMCP for bounded audits of load-bearing formulas, source labels, and
local derivation checks.  It is an audit assistant, not an oracle.  Every
Mathematical claim should be tagged, at least internally during drafting, as one
of:

- exact derivation in current notation;
- sourced formula with verified assumptions;
- approximation with explicit source and approximation point;
- project hypothesis or implementation conjecture;
- human review required.

## Hard requirements for the final monograph treatment

The rebuilt DPF chapters must satisfy all of the following.

### 1. Mathematical self-containment

The exposition should define its objects carefully enough that a reader can
follow the development without consulting internal repo plans.

### 2. Explicit target-status discipline

For each method, the text must make mathematically clear whether the object in
question is:

- an exact likelihood in a special case;
- an unbiased estimator of the likelihood;
- an approximate or relaxed likelihood surrogate;
- a transport proposal requiring correction;
- a learning objective not directly identical to the final HMC target.

### 3. Rigorous comparison

Comparisons to existing literature should be mathematical when possible:

- compare target definitions;
- compare proposal constructions;
- compare Jacobian or weight corrections;
- compare where bias enters;
- compare what HMC would be targeting.

### 4. Implementation relevance

The chapters must still connect to implementation.  But implementation
relevance should arise from the mathematics, for example:

- what quantity must be computed;
- what Jacobian/log-determinant quantity is needed;
- what numerical stiffness implies for discretization;
- what kind of autodiff path is required;
- what kind of approximation label would be necessary.

### 5. HMC suitability analysis

The final assessment must be stronger than generic prose.  It should analyze the
HMC question in terms of:

- value/gradient consistency;
- target preservation or lack thereof;
- pseudo-marginal versus surrogate-target interpretations;
- structural-model difficulties for DSGE and MacroFinance;
- practical implications for compiled differentiable implementations.

## Program phases

This program will proceed in distinct phases, each with its own plan note under
`docs/plans`.

1. Phase M0: source inventory and previous-pass failure audit
2. Phase M1: literature survey and source-grounding audit
3. Phase M2: mathematical architecture and chapter-map design
4. Phase M3: particle-filter and particle-flow theory chapter plan
5. Phase M4: differentiable resampling and OT theory chapter plan
6. Phase M5: HMC-target and structural-model assessment chapter plan
7. Phase M6: drafting and equation-audit execution plan
8. Phase M7: monograph integration, bibliography, and audit plan

## Global audit protocol

Every phase must record:

- sources inspected;
- exact questions asked of the literature;
- what was established mathematically;
- what remains approximate or unresolved;
- what would justify the next phase.

No phase should promote a method to HMC suitability merely because it appears in
student code or because it yields smooth gradients in experiments.

## Success criteria for the program

This program succeeds only if it produces:

1. a literature-grounded chapter architecture stronger than the current three
   DPF draft chapters;
2. a mathematically disciplined exposition that could plausibly stand beside the
   stronger CIP monograph chapters;
3. explicit derivation or source support for all load-bearing equations;
4. a rigorous HMC-suitability assessment for nonlinear DSGE and MacroFinance
   models;
5. a clean separation between mathematical exposition, implementation
   consequences, and experimental comparison.

## Failure modes to avoid

- verbal governance prose in place of mathematics;
- repo/file-path commentary in the reader-facing monograph text;
- casual comparison to student work without mathematical analysis;
- source claims that are not tied to specific papers or equations;
- treating differentiability as equivalent to HMC correctness;
- presenting frontier research heuristics as production-ready doctrine.

## Immediate next step

The immediate next step is Phase M1: a literature survey and source-grounding
audit that is broad enough to determine the correct final chapter architecture
before any further reader-facing DPF chapter rewriting proceeds.
