# Claude Handoff: C2 Transformed-Guide TT-DMIS Review

Date: 2026-08-29

Purpose: obtain a rigorous read-only review of the proposition-proof
mathematics and its implementation/test plan. Review one exact path at a time.
Do not edit files, run commands, launch agents, or start implementation. If a
stage needs more context, request the next single exact path or line range.

## Review Order

Complete Stage 1 before Stage 2. A `REVISE` verdict should identify exact
blocking lines and a concrete correction; it does not authorize an edit.
Stages 3 and 4 are cross-checks after the two primary reviews.

## Stage 1: Proposition-Proof Manuscript

Use this prompt verbatim:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/benchmarks/artifacts/c2_completion_20260824/attempt05/ukf_guided_defensive_tt_dmis_analytical_gradient.tex
Do not edit, run commands, launch agents, or review the whole repo. Question:
Is every material mathematical claim correct for the declared frozen-proposal
target? Audit, rather than summarize, (1) the raw-C2 zero-cross-covariance and
zero-gain result, including its integrability condition; (2) the log-square
transformation and log-chi-square moments; (3) the transformed Kalman guide,
positive-definite covariance, and covariance-matched Student scale; (4) support,
normalization, and the mixture second-moment bound; (5) the distinction between
random-mixture J(alpha) and exact fixed-bank DMIS variance; (6) the joint
ancestor-state pilot and the squared W/a factor; (7) fixed-bank and sequential
DMIS base masses and conditional correctness; (8) the same-finite-scalar
analytical score recursion; and (9) the pseudo-marginal and source-faithfulness
boundaries. Check dimensions, conditioning, measures, denominators, derivative
scope, and all assumptions needed by each proof. Report findings first in
severity order with exact line references. For every objection state the
claimed target, the quantity actually derived, and why they differ. Do not
treat MathDevMCP status as proof or refutation. End with exactly VERDICT: AGREE
or VERDICT: REVISE.
```

Expected source identity at the original review:
`3d6f8e5312f02f72150aa556f8e8cd91aef5d7bd9ed0a6ca3a1e219921fe12a7`

The repaired manuscript source identity is now
`1c9232244a7cf4adaabbd4fe7d07527df757e277b31cc980319f8cc77af858dd`.

Stage 1 must explicitly answer these discriminator questions:

- Does a raw-observation C2 UKF really have zero population gain?
- Is the transformed guide a proposal approximation while the exact raw
  likelihood remains in the importance numerator?
- Does changing `nu` preserve covariance because the Student scale is
  `((nu-2)/nu) P_D`?
- Is `J(alpha)` correctly prevented from masquerading as the exact equal-bank
  variance?
- Does the joint second-moment pilot require `(W_j/a_j)^2`, not one copy?
- Does the analytical recursion differentiate the identical finite scalar
  that the value path computes?

## Stage 2: Implementation and Test Plan

After completing Stage 1, use this prompt verbatim:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-implementation-test-plan-2026-08-29.md
Do not edit, run commands, launch agents, or review the whole repo. Question:
Does this plan faithfully and completely implement and test the reviewed
manuscript without changing its target? Audit the shared-APF call chain,
nonuniform base masses, complete mixture denominator, transformed-guide
compiler, frozen analytical score, joint alpha/tail calibration, pilot/final
independence, baseline ladder, heuristic adversaries, statistical criterion,
continuation vetoes, default assumptions, GPU/XLA/memory-growth contract,
artifact sufficiency, compute budget, and exactness nonclaims. Look
specifically for wrong or duplicate baselines, proxy metrics promoted to
criteria, hidden tuning, unfair particle budgets, stale attempt05 snapshot
reuse, horizon off-by-one errors, invalid pairing/common-random-number logic,
silent ridge/log-offset policy, and commands that could succeed without
answering the research question. Compare every mathematical-to-code crosswalk
row against the plan phases. Report findings first in severity order with exact
line references and distinguish candidate failure from experiment invalidity.
End with exactly VERDICT: AGREE or VERDICT: REVISE.
```

Expected plan identity at the original review:
`0b7e609cdfaf67ceb595f5839afe3984f6941ff92978efbd870fc580ddad9495`

The executed plan, including its execution addendum, is now tracked by the
current hash recorded in its run manifest.

Stage 2 must explicitly answer:

- Does Phase 1 generalize the one shared APF evaluator rather than create a
  C2-specific capability fork?
- Are the first implementation's TT samples drawn from the current complete
  `q_floor`, not an unavailable pure `q_H` sampler?
- Is one timewise alpha used consistently across ancestors?
- Are calibration and final banks independent, with exact fixed-bank variance
  used for selection rather than `J(alpha)` alone?
- Is the old categorical TT path only a regression authority, while the
  generalized TT route is the scientific baseline?
- Is paired replicate inference valid even where common random numbers are not
  used?
- Do the result labels avoid HMC, posterior, exact-pseudo-marginal, default, or
  universal-performance claims?

## Stage 3: Skeptical-Review Cross-Check

Only after Stages 1 and 2, use:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-plan-review-2026-08-29.md
Do not edit, run commands, launch agents, or review the whole repo. Question:
Does this skeptical review accurately identify and close every material flaw
in the plan you just reviewed, without overstating what has been proved or
tested? List any missed, falsely closed, or newly introduced issue with exact
line references. End with exactly VERDICT: AGREE or VERDICT: REVISE.
```

## Stage 4: MathDevMCP Boundary Cross-Check

Only after the mathematical review, use:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/benchmarks/artifacts/c2_completion_20260824/attempt05/ukf_guided_defensive_tt_dmis_math_audit_summary.md
Do not edit, run commands, launch agents, or review the whole repo. Question:
Does this summary report the MathDevMCP evidence and its limitations honestly,
and does it agree with your independent mathematical verdict? Treat
needs_formalization, typed_abstention, and budget_exhausted as diagnostic
statuses, not counterexamples and not proof. Identify any overclaim or omitted
mathematical blocker with exact line references. End with exactly VERDICT:
AGREE or VERDICT: REVISE.
```

The original digest-bound native reports named by that summary were:

- `ukf_guided_defensive_tt_dmis_mathdev_rigor_audit_v6.md` and `.json`;
- `ukf_guided_defensive_tt_dmis_derivation_tree_audit_v3.md` and `.json`.

Earlier numbered reports are superseded diagnostics and must not be used as
the current source binding. The current focused report is
`ukf_guided_defensive_tt_dmis_mathdev_rigor_audit_v8.md` and `.json`, bound to
the repaired manuscript digest above.

## Facts the Review Must Not Blur

- The raw-observation UKF idea fails mathematically for C2 because its gain is
  zero. The proposed replacement is the log-square moment guide.
- The replacement may still fail empirically because Gaussian log-chi-square
  closure can miss shape and tails.
- Zhao and Cui support squared-TT proposal construction and exact
  target/proposal correction. They do not supply the transformed guide, outer
  DMIS mixture, generalized base masses, or frozen score developed here.
- The manuscript proves identities for the declared finite program. It does
  not prove finite-sample efficiency, posterior correctness, or exact HMC.
- The original memo predates implementation. The fixed-half implementation and
  trusted GPU/XLA diagnostic have since completed. Their terminal evidence is
  the post-run paired audit under
  `docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/attempt01/`.
  The alpha/nu calibration pilot was subsequently executed as a separate
  bounded follow-up under
  `docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/pilot-attempt02/`.
  Its point minimum failed the predeclared bootstrap-stability gate, so it
  returned the fixed-half fallback and did not produce selected evidence.

## Execution Follow-Up for Claude (2026-08-30)

The implementation and bounded serious diagnostic have now completed. Please
review these exact paths read-only if an independent terminal check is desired:

1. `docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-execution-result-2026-08-30.md`
2. `docs/benchmarks/artifacts/c2_ukf_guided_defensive_tt_dmis_20260829/attempt01/paired_audit.md`
3. `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py`
4. `bayesfilter/highdim/c2_transformed_observation_student_proposal_tf.py`
5. `bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py`

Specific questions for the follow-up review are:

- Does the generalized base-mass APF core still compute exactly the manuscript
  scalar and recursive analytical score, with the uniform route as a special
  case?
- Is the TensorFlow chi-square sampler's `beta=0.5` rate convention correct,
  and does the empirical covariance test adequately protect it?
- Does the equal-bank DMIS compiler evaluate the complete outer mixture density
  for both banks and use the correct `omega_s/N_s` base masses?
- Does the paired audit compare the DMIS candidate against the constructed
  heuristic adversaries rather than against retained TT, and does it preserve
  the distinction between a mechanism nomination and promotion?
- Are the recorded gaps (pilot fallback due to unstable bootstrap minimizers,
  and no per-time maximum weight in the preserved pre-observability run)
  stated accurately?

The serious result is descriptive and diagnostic only. The pilot follow-up is
also calibration evidence only, not a selected final arm. A valid follow-up review
must not turn the positive ESS contrast into a default, posterior, HMC,
pseudo-marginal, or source-faithfulness claim.

## Return Format

For each stage, return:

1. findings in severity order with exact path and line references;
2. open questions only where the inspected path cannot resolve them;
3. a short consistency summary; and
4. exactly one terminal verdict line.

Do not return `AGREE` merely because the document is self-consistent. A valid
`AGREE` requires checking the mathematics, target identity, assumptions,
call-chain obligations, and nonclaims requested in that stage.
