# BayesFilter V1 Structural Sigma-point Score Plan Audit

## Date

2026-05-13

## Scope

This audit reviews:

```text
docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-plan-2026-05-13.md
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

The audit is deliberately limited to the BayesFilter V1 lane.  It does not
open MacroFinance, DSGE, structural SGU, or the shared monograph reset-memo
lanes.

## Independent Audit Findings

### A1. Main-goal alignment

The plan is aligned with the current phase goal: resolve the default Model C
score blocker by implementing the Chapter 18b structural sigma-point score
path.  The plan correctly rejects a separate fixed-null SVD theory as the
controlling mathematical story.

### A2. Drift checks

No HMC, GPU/XLA, nonlinear Hessian, MacroFinance switch-over, DSGE switch-over,
or SGU claim belongs in this phase.  These are correctly marked as non-goals.

### A3. Necessary clarification

The initial plan could be misread as resizing every sigma rule to the active
rank of the pre-transition covariance.  That would be wrong for the current
BayesFilter backend comparison, especially for SVD-CUT4: default Model C has a
rank-two stochastic support inside the declared three-dimensional
\((x_{t-1},\varepsilon_t)\) variable, while the documented CUT4-G rule is
defined for the declared dimension.  The tightened plan now requires preserving
the backend rule dimension and representing the deterministic phase direction
by a zero factor column with a zero derivative.

### A4. Required implementation gate

The implementation must check that the null direction is structurally fixed.
It may not silently accept a parameter-dependent zero eigenvalue, a floor
branch, or a hidden nugget.  The score path should therefore block if the
null-null block of the covariance derivative is nonzero or if a positive
placement floor changes the structural law.

### A5. Testing gate

The finite-difference target must be the same structural sigma-point
likelihood used by the value path.  Passing against a different collapsed
likelihood would not close R1.

## Audit Conclusion

The plan is execution-ready after the clarification in A3.  Continue with S0
only if MathDevMCP confirms that the derivation follows the Chapter 18b
pushforward and UKF moment equations without changing the structural law.
