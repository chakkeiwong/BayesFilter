# BayesFilter V1 P7 Exact Reference Strengthening Result

## Date

2026-05-14

## Governing Plan

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-p7-exact-reference-strengthening-plan-2026-05-14.md
```

## Phase Scope

P7 decides whether V1 needs a stronger exact or high-accuracy reference for
nonlinear Models B-C now.  It must not promote dense one-step Gaussian
projection to exact full nonlinear likelihood evidence, and it must not add a
production SMC dependency.

## Plan Tightening

The P7 plan had one ambiguity: P3 records exact full nonlinear likelihood for
Models B-C as a blocked claim, but the current V1 public claims do not require
that blocked evidence.  The plan was tightened inside the V1 lane to allow an
explicit deferral branch:

```text
If no current V1 claim requires stronger nonlinear reference evidence, P7
closes by deferral and does not add reference code.
```

## Independent Audit

Audit question:

```text
Would a dense quadrature or seeded high-particle SMC reference close a current
V1 claim, or would it create scope drift?
```

Audit findings:

- Model A already has exact linear-Gaussian Kalman reference evidence.
- Models B-C are documented as dense one-step Gaussian projection diagnostics,
  not exact full nonlinear likelihood references.
- P3, P4, and P6 all preserve this reference boundary.
- Current nonlinear value, score, HMC-smoke, and GPU/XLA diagnostic claims are
  branch-scoped and shape-scoped; none require exact full nonlinear likelihood
  certification.
- Adding SMC or full dense multi-step quadrature now would create a new
  approximation policy and runtime tier without a named downstream consumer.

Veto diagnostics checked:

| Veto | Result |
| --- | --- |
| Stochastic reference lacks seed/reproducibility metadata | clear; no stochastic reference added |
| Dense projection mislabeled as exact full likelihood | clear; existing wording preserves the distinction |
| Reference dependency leaks into production imports | clear; no dependency added |
| Reference artifact used to justify HMC/GPU claims | clear; no such claim made |

## Decision

Exact nonlinear reference strengthening remains deferred for V1.

No dense multi-step quadrature, high-particle SMC, or production reference
dependency is added in P7.  The current claim language remains:

- Model A: exact linear-Gaussian Kalman reference;
- Models B-C: dense one-step Gaussian projection diagnostics only;
- full exact nonlinear likelihood for Models B-C: blocked/deferred.

## Validation

Documentation-only checks:

```bash
git diff --check
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml')); print('source_map ok')"
```

Result:

- whitespace check passes;
- source-map YAML parses;
- no production code or tests are changed by P7.

## Gate Result

P7 passes by the explicit-deferral branch:

```text
No current V1 claim requires a stronger nonlinear reference than the existing
exact affine and dense one-step projection evidence.
```

## Next Phase Justification

P8 is justified as an integration-planning phase because P1-P7 have current
result artifacts, optional claims are labeled at their true scope, and P8 does
not require MacroFinance or DSGE source edits.
