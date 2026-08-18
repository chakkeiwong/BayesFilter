# GenUT Austria-SIR Root-Cause Hypothesis Plan

Date: 2026-08-17

## Research question

Trace the large Austria-SIR `T=20`, `j0` discrepancy and distinguish:

1. an SIR callback equation/tangent defect;
2. a score-recursion/JVP defect in the finite GenUT program;
3. variance amplification in transport/reset/shape correction;
4. a finite-program bias that remains after the JVP is internally consistent.

The target is the derivative of the exact finite GenUT value program at the
fixed Austria-SIR observation path. This is not an exact nonlinear likelihood
claim.

## Evidence contract

- Baseline: current scalar `finite_value_score` route, Austria-SIR `T=20`,
  `N=1008`, fixed observations and fixed noise per seed.
- Primary hard criteria: finite outputs, source-law/event-order hash match,
  adapter transition/observation tangent finite-difference agreement, scalar
  score-increment additivity, and reset-JVP finite-difference agreement.
- Secondary diagnostics: full-score finite-difference ladder, per-time score
  increments, seed SD/MCSE, transport residuals, covariance-gap eigenvalues,
  and shape-control ablations.
- Upstream vetoes retained: classifier Gaussian exact-oracle failure and GenUT
  LGSSM Kalman-oracle failure. They cannot be weakened by this diagnostic.
- Nonclaims: no exact SIR score, no classifier oracle admission, no algorithm
  ranking, no default readiness, and no NeuTra/HMC readiness.

## Hypotheses and discriminating tests

| ID | Hypothesis | Test | Falsifier / interpretation |
|---|---|---|---|
| H1 | SIR adapter equation or tangent is wrong | Transition and observation callback central-FD tests on source-scale random states | Failure is an implementation blocker; pass weakens this hypothesis |
| H2 | Full score JVP/recursion is wrong | Same-program central-FD ladder at `T=2,5,20` with fixed noise and branch checks | Failure with valid endpoints localizes score recursion/JVP; pass leaves finite-program bias/variance |
| H3 | A few time steps dominate `j0` variance | Persist per-time `score_increments[:,0]`, compare energy concentration and seed SD | Concentration supports variance amplification; no concentration weakens it |
| H4 | Transport/reset/shape controls amplify variance | Ablate higher moments, pairwise correction, caps; record residuals and seed SD | Large SD reduction with stable value suggests variance amplifier; unchanged SD points elsewhere |
| H5 | Contract-E reset JVP is wrong | Synthetic cloud JVP versus centered finite differences of the reset map | Failure is a reset implementation blocker |
| H6 | Model law mismatch | Reuse source observation hash and CPU/GPU replay plus clipping audit | Hash/event mismatch blocks score comparison; prior audit passed this |

## Default and assumption audit

- `N=1008`, controls, and seeds come from the existing Austria-SIR claim scope;
  they are a baseline, not promoted defaults.
- `T=2,5,20` is a localization ladder, not evidence for horizon scaling.
- FP32/XLA is the existing GenUT execution route; callback checks also record
  FP64 reference values where practical.
- Three fixed claim seeds are sufficient for a bounded diagnostic only; all
  seed summaries are descriptive.

## Skeptical plan review

- The exact Kalman score is not used as an SIR baseline.
- Classifier values are descriptive only because its Gaussian oracle failed.
- Finite-difference agreement is checked on the same executed value program;
  it cannot establish observed-data score correctness.
- Shape-control ablations are explanatory, not promotion screens.
- The run stops if source hashes, branch validity, or finite checks fail.

## Execution and artifacts

Run `docs/benchmarks/run_genut_sir_root_cause_hypotheses_20260817.py` in
`tftwogpu` on the requested physical GPU1. Write a fresh unique artifact root:

`docs/benchmarks/artifacts/genut-sir-root-cause-hypotheses-20260817/attempt01/`

The JSON must contain the manifest, hypothesis rows, raw score increments,
finite-difference ladders, ablation summaries, and a decision table. A result
and reset memo follows the run.
