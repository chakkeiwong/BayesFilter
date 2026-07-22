# Contract E Score-Aware Teacher Projection: 2D LGSSM Result

Date: 2026-07-15

Status: `SCOPED_MECHANISM_WITNESS_PASS`

## Result

The proposed Contract E--teacher projection mechanism is mathematically
specified and numerically demonstrated for one two-observation 2D LGSSM
fixture. A 6561-point parent-by-innovation Gauss--Hermite teacher was compressed
to a frozen positive seven-point student by enforcing mass, two first moments,
three second moments, and the exact next-observation predictive density.

At `theta=(0.72,-1.05,-0.70)`, the seven student weights had minimum
`2.961221186e-4`; the scaled feature matrix condition number was `84.2606`.
Student and teacher two-observation values were both
`-1.6375421323997048`. Their score difference had maximum absolute value
`1.11e-16`; the maximum selected-feature tangent difference was `4.05e-15`;
and the largest autodiff/centered-FD difference was `1.83e-10`.

The exact Kalman value was `-1.637545116868383`, and its score was
`(-0.843840883733,-0.405394763108,-0.667711074507)`. The finite teacher score
was `(-0.843638383670,-0.405408322446,-0.667677325884)`. Thus compression error
is at roundoff for the retained span, while teacher quadrature error remains
separate and nonzero.

## Decision table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain Contract E--TP as a research proposal and proceed to a multi-step LGSSM implementation plan | Pass for the declared finite teacher and selected feature span | No nonpositive weight, rank failure, nonfinite result, residual failure, or AD/FD failure | Whether a fixed positive chart covers a useful parameter region and whether repeated local compression preserves full-rollout value/score | Multi-step LGSSM with per-time Kalman value, score, positivity, and chart-margin audits | Full LEDH correctness, nonlinear/NAWM validity, HMC readiness, canonical/default status |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | Passed for this deterministic float64 fixture |
| Statistically supported ranking | Not applicable; this is a deterministic identity/numerical witness, not a stochastic method ranking |
| Descriptive-only differences | Teacher-minus-Kalman value and score gaps describe finite quadrature error at one parameter point |
| Default readiness | Failed/not tested; the route remains proposed and noncanonical |
| Next evidence needed | Multi-step LGSSM, parameter-region positivity, structural singular fixture, then small pruned DSGE |

## Evidence and checks

- Plan: `docs/plans/bayesfilter-contract-e-score-aware-teacher-projection-lgssm-plan-2026-07-15.md`
- LaTeX: `docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`
- Witness: `docs/benchmarks/contract_e_score_aware_teacher_projection_2d_lgssm.py`
- Structured result: `docs/benchmarks/artifacts/contract_e_score_aware_teacher_projection_2d_lgssm_2026_07_15.json`
- Literature ledger: `docs/plans/bayesfilter-contract-e-score-aware-teacher-projection-literature-ledger-2026-07-15.md`
- CPU reference run: passed.
- Python compile: passed.
- `git diff --check` on scoped text/code files: passed.
- Full `latexmk` book build: passed, 394 pages.

The full book retains eleven undefined citations and four multiply defined
labels from unrelated pre-existing chapters. The added Del Moral citation and
all added labels resolve. The two files named as cached Poyiadjis PDFs are HTML
block pages and were explicitly excluded as technical evidence.

## Post-run red team

Strongest alternative explanation: exact student--teacher agreement is partly
constructed because the predictive density is explicitly one of seven equality
constraints. That is the intended mechanism, but it does not show that a small
fixed feature set works across many future observations or nonlinear models.
The conclusion would be overturned as a useful algorithmic direction if
multi-step LGSSM charts frequently become nonpositive/singular, or if
unconstrained future increments recover large value/score errors immediately.
The weakest evidence is global chart coverage: only one center has been tested.

