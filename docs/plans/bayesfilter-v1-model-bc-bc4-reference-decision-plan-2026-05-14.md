# BayesFilter V1 Model B/C BC4 Approximation-Quality Reference Decision Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase BC4 in:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

## Purpose

Decide whether Models B-C need stronger approximation-quality references than
dense one-step projection for any current V1 claim.

## Entry Gate

BC4 may start after BC1-BC3 identify stable boxes and robustness envelopes or
explicit blockers.

## Evidence Contract

Question:
- Does any current Model B/C claim require a stronger reference artifact, and
  if so, which bounded reference is appropriate?

Baseline:
- P7 exact-reference deferral, BC1-BC3 artifacts, and current claim vocabulary.

Primary criterion:
- Either a stronger reference artifact is justified and specified, or exact
  nonlinear references remain explicitly deferred because no current claim
  needs them.

Veto diagnostics:
- Monte Carlo reference lacks seed/particle metadata;
- dense one-step projection is called exact full likelihood;
- reference dependencies enter production imports.

What will not be concluded:
- That a new reference is required merely because a diagnostic would be
  interesting.

Artifact:
- BC4 decision/result file and optional follow-on reference subplan.

## Required Decision Table

The BC4 result must include one row for every `certified` or candidate
promotion claim from BC1-BC3:

| Claim | Current comparator/reference basis | Comparator type | Why sufficient | Out of scope |
| --- | --- | --- | --- | --- |

Allowed comparator types:
- `exact`;
- `deterministic_approximation`;
- `monte_carlo`;
- `diagnostic_only`.

If any claim cannot justify its comparator basis, BC4 must either specify a
bounded stronger reference subplan or downgrade the claim.

## Execution Steps

1. Review which BC1-BC3 claims are intended as `certified` rather than
   `diagnostic`.
2. Identify whether any claim depends on exact/reference likelihood evidence
   beyond the implemented filter value and score checks.
3. If a stronger reference is needed, choose one bounded reference:
   - dense low-dimensional multi-step quadrature; or
   - seeded high-particle SMC.
4. Specify seeds, quadrature nodes or particle counts, runtime budget, and
   acceptance criteria before implementation.
5. If no current claim needs the reference, keep it deferred with rationale.
6. Write the BC4 result artifact and update the V1 reset memo.

## Primary Gate

BC4 passes if the reference status is explicit and non-promotional.

## Veto Diagnostics

Stop and ask for direction if:
- reference work begins without a named current claim;
- Monte Carlo uncertainty is omitted from the evidence contract;
- exact/deterministic approximation/Monte Carlo labels are conflated;
- production code imports reference-only dependencies.

## Expected Artifacts

Use the execution date in result filenames.  The plan date remains
2026-05-14, but future result artifacts should use `YYYY-MM-DD`.

```text
docs/plans/bayesfilter-v1-model-bc-bc4-reference-decision-result-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Optional only if justified:

```text
docs/plans/bayesfilter-v1-model-bc-reference-artifact-subplan-2026-05-14.md
```

## Continuation Rule

Continue to BC5 only when the HMC target cells do not depend on unresolved
reference evidence.  If a stronger reference is required for a named claim,
stop and execute that reference subplan first.
