# Phase M1 plan: DPF literature survey and source-grounding audit

## Date

2026-05-09

## Purpose

This is the first substantive phase of the monograph rebuild.  Its purpose is to
survey the literature broadly and rigorously enough that the final chapter
architecture and mathematical exposition can be designed from source knowledge
rather than from the shape of the earlier draft.

## Central goal

Build a source-grounded map of the differentiable particle-filter literature,
with enough mathematical clarity to determine:

- what the major method families are;
- which equations and target definitions are load-bearing;
- which approximations matter for implementation and HMC;
- which topics the student work covers that CIP or BayesFilter does not yet
  cover sufficiently;
- and whether the current intended BayesFilter path is justified or needs
  adjustment.

## Required source lanes

### A. Primary literature lane

Survey at minimum:

- classical particle filtering and SMC;
- pseudo-marginal / PMCMC references relevant to target correctness;
- Daum--Huang particle-flow literature;
- LEDH / PF-PF / invertible particle-flow literature;
- kernel and stochastic particle-flow literature where relevant;
- differentiable particle filtering for end-to-end learning;
- differentiable or soft resampling literature;
- entropy-regularized OT resampling literature;
- learned / amortized OT or learned transport-operator literature where
  relevant;
- HMC and approximate-target literature needed to interpret the DPF path.

### B. CIP source lane

Audit the mathematical content of:

- `~/python/latex-papers/CIP_monograph/chapters/ch17_nonlinear_filtering.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch20_hmc.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch21_advanced_hmc.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch26_differentiable_pf.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch27_ledh_pfpf_neural_ot.tex`
- `~/python/latex-papers/CIP_monograph/chapters/ch32_diff_resampling_neural_ot.tex`

### C. Student critique lane

Audit the student report/repo materials for:

- additional literature coverage;
- stronger or weaker conclusions than the primary literature supports;
- mathematical topics or implementation issues covered there but not in CIP;
- experiments that reveal target/bias/gradient issues useful for the monograph.

### D. Tool lane

Use ResearchAssistant and MathDevMCP to record:

- paper identities and citation neighborhoods;
- exact source sections/equations for key claims;
- bounded audits of load-bearing formulas;
- unresolved claims marked for human review.

## Required output

Create a structured literature audit note with at least these tables:

### Table 0: citation-key and source-identity audit

| Claim family | Primary paper | Alternate versions | Preferred citation key | Notes |
| --- | --- | --- | --- | --- |

### Table 1: method-family map

| Method family | Core sources | Mathematical object | Exact / unbiased / approximate / surrogate | Implementation implications | HMC relevance |
| --- | --- | --- | --- | --- | --- |

### Table 2: student-coverage comparison

| Topic | Primary literature coverage | CIP coverage | Student coverage | Gap / disagreement | Action |
| --- | --- | --- | --- | --- | --- |

### Table 3: unresolved or load-bearing claims

| Claim | Source support | Needed audit | Status |
| --- | --- | --- | --- |

### Table 4: theorem / equation obligation register

| Theorem or equation family | Derive fully | Cite and adapt | Compare variants | Human review required |
| --- | --- | --- | --- | --- |

## Hard requirements

- Do not draft final chapter prose in this phase.
- Do not settle mathematical questions by paraphrasing README files.
- Do not let student code or notebooks substitute for literature support.
- Do record where student experiments reveal implementation issues worth
  elevating into the monograph.

## Exit gate

Proceed only when the literature map is broad enough to justify the final chapter
architecture and when the main method-family distinctions are mathematically
clear.
