# Contract E Score-Aware Teacher Projection: 2D LGSSM Witness Plan

Date: 2026-07-15

Status: `ACTIVE_REFERENCE_WITNESS`

## Research intent ledger

| Item | Declaration |
|---|---|
| Main question | Can a proposed score-aware extension of Contract E compress a finite parent-by-innovation teacher into a smaller positive weighted cloud while preserving selected primal feature expectations and, by differentiating the same projection, their parameter tangents? |
| Candidate | A fixed-anchor, equality-constrained quadratic projection whose constraints contain mass, first moments, second moments, and one next-observation predictive-potential feature. |
| Exact comparator | The finite 6561-point teacher defined by fixed 81-point parent and 81-point innovation Gauss--Hermite designs, transition, current observation update, and next-observation predictive potential. |
| Promotion criterion | The 2D LGSSM witness has strictly positive student weights; matches every declared teacher feature to `1e-10` in float64; matches the derivative of every feature to `1e-8`; and its predictive log increment and score match the teacher to `1e-10` and `1e-8`, respectively. |
| Promotion veto | Rank deficiency, nonpositive student weights, nonfinite values, feature residual above tolerance, tangent residual above tolerance, or autodiff/central-FD disagreement above `1e-7`. |
| Continuation veto | The stated fixed witness is infeasible after bounded deterministic anchor/subset search, or the emitted artifact cannot reproduce the displayed numbers. |
| Repair trigger | A nonpositive KKT solution triggers only a documented change to the fixed student anchor subset or reference weights within the same bounded witness; it does not authorize clipping or dropping constraints. |
| Explanatory diagnostics | Singular values/condition number, selected anchors, student compression ratio, raw weights, and per-feature residuals. |
| Must not be concluded | Universal nonlinear validity, exactness outside the selected feature span, NAWM feasibility, variance reduction, HMC readiness, production readiness, or canonical status for `contract_e_chol_v1`. |

## Evidence contract

The artifact must preserve the model matrices, observations, parameter value,
fixed parents and innovations, teacher weights, selected student anchors,
projection system, feature values and tangents, predictive increment and score,
finite-difference comparison, tolerances, and pass/fail decision.  The planned
artifact is
`docs/benchmarks/artifacts/contract_e_score_aware_teacher_projection_2d_lgssm_2026_07_15.json`.

## Default and assumption audit

| Choice | Provenance and role | Failure mode | Early diagnostic |
|---|---|---|---|
| 2D LGSSM | Exact differentiable reference model; diagnostic baseline | Linear-Gaussian success may not transfer to nonlinear models | Explicit nonclaim and later nonlinear fixture requirement |
| 81 parents by 81 innovations | Higher-order `N^2` teacher formed from tensor order-9 Gauss--Hermite rules; diagnostic design | Teacher quadrature error relative to Kalman remains | Claims are relative to the finite teacher; report the separately computed Kalman gap |
| Feature basis `1,x,vech(xx^T),r` | Hypothesis motivated by moment restoration and the next likelihood increment | Does not determine the full tangent filtering measure | Exact selected-span theorem and explicit limitation |
| Fixed anchor subset | Required for a differentiable fixed-branch finite program | Rank switches or negative weights | Freeze indices before the reported evaluation; report rank and minimum weight |
| Euclidean KKT projection | Semi-analytical baseline | Positivity is not guaranteed | Hard positivity veto; no clipping |
| Float64 CPU | Independent small reference exception | Does not evidence GPU/XLA production behavior | Record `CUDA_VISIBLE_DEVICES=-1`; make no GPU claim |

## Skeptical pre-execution audit

The baseline is the same finite teacher, not the exact Kalman filter.  The
predictive feature is a promotion criterion only for its one declared next
observation, not a proxy silently promoted to full filtering correctness.
Positivity, fixed-branch differentiability, feature rank, and finite-difference
agreement are hard vetoes.  The run has a bounded deterministic search over at
most all student subsets of one declared size and stops if none is feasible.
The emitted artifact answers the stated finite feature/tangent question but
cannot answer broad nonlinear, structural, or HMC questions.  On that basis the
plan is adequate for the bounded reference witness.

## Commands and budget

CPU-only reference command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-contract-e-score-aware python docs/benchmarks/contract_e_score_aware_teacher_projection_2d_lgssm.py
```

Document checks:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Budget: one reference run plus at most two localized repairs, less than five
minutes total CPU time.  Preserve any failed attempt rather than overwriting it
if its scientific configuration changes.
