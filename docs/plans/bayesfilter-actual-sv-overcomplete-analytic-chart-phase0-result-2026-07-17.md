# Actual-SV Overcomplete Analytical Chart Phase 0 Result

Date: 2026-07-17

Status: `PASS_PHASE_0_DOCUMENTATION`

Plan:
`docs/plans/bayesfilter-actual-sv-overcomplete-analytic-chart-repair-plan-2026-07-17.md`

## Objective And Scope

Document the Actual-SV overcomplete analytical chart amendment before changing
the implementation.  This result covers the mathematics and document build
only.  It does not establish chart feasibility, implementation correctness,
score accuracy, GPU/XLA feasibility, HMC readiness, canonical Contract E--Chol
correctness, or leaderboard readiness.

## Artifact

The amendment is in
`docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`, subsection
`sec:bf-ledh-actual-sv-overcomplete-chart`.

It now states:

- the zero-based time-748 square-chart counterexample and its two negative FD
  endpoints;
- why adding anchors differs from adding features;
- deterministic weighted-quantile indices and stable duplicate fill;
- positive Voronoi mass as a hard preparation condition;
- the analytical Pearson projection of Voronoi mass;
- a proposition and proof of the center reference minimizer;
- the frozen Pearson metric and analytical runtime solve;
- a proposition and proof of the frozen-reference total JVP;
- the distinction between frozen indices and moving gathered locations;
- binding finite TensorFlow rank, condition-roundoff, residual, finiteness, and
  computed-positivity gates;
- weakest-case increasing-precision recomputation as corroboration rather than
  interval proof;
- a preparation/runtime algorithm; and
- time `O(K q^2 + q^3)` and temporary storage `O(K q + q^2)`, with Actual-SV
  `q=4`.

## Mathematical Review

The documented reference is

\[
 r=v+VA_0^\top(A_0VA_0^\top)^{-1}(b_0-A_0v).
\]

It is the unique equality-constrained minimizer under `v_i>0` and full row
rank.  The document does not infer `r_i>0`; it makes strict positivity a
separate hard gate.

For frozen `r`, `R=diag(r)`, row scale, and indices, the runtime JVP is derived
from `S lambda = c` and includes both `dot A` and `dot b`.  The proof explicitly
includes teacher motion, feature parameter dependence, and motion of gathered
anchor locations.  This matches the intended TensorFlow diagonal-P algebra and
does not differentiate candidate selection.

Review verdict: `MATHEMATICALLY_CONSISTENT_FOR_IMPLEMENTATION` under the stated
positive-`v`, positive-`r`, frozen nonsingular-row-scale, and full-row-rank
conditions.

## Checks

Command:

```text
cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Result:

- exit code: `0`;
- output: `docs/main.pdf`;
- output size: `1614516` bytes;
- pages: `408`;
- no new undefined reference to an amendment label;
- no `Undefined control sequence`, `LaTeX Error`, or package error; and
- `git diff --check` passed for the plan and chapter.

The full monograph log retains unrelated pre-existing warnings: four duplicate
section labels and eleven unresolved citations outside this amendment, plus
layout warnings.  They are not evidence against this phase, and this result
does not claim a globally warning-free monograph.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Pass Phase 0 | Amendment is self-contained, mathematically consistent, and builds | No Phase 0 stop condition fired | Empirical feasibility and implementation remain untested | Freeze the Phase 1 machine-readable design and command contract | No numerical, scientific, GPU, HMC, canonical, or leaderboard claim |

## Handoff

Phase 1 may begin.  No candidate capacity has been evaluated and no held-out
point has been inspected.
