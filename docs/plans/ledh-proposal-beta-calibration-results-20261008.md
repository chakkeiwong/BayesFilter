# Proposal mixture probabilities: completed documentation repair

The shared chapter previously took beta as a frozen input without specifying
its selection. Section 27.5 of the monograph (printed pages 252–255; PDF pages
272–275), also Section 5 of the standalone note (pages 8–11), now supplies the
missing algorithm. The current PDFs contain 647 and 23 pages respectively.

The proposal is a mixture of densities. Each draw selects one transformation
with its branch probability. Applying the weighted average of transformations
would define a different distribution and would not justify the stated mixture
denominator.

The added procedure fixes calibration clouds and maps, uses independent pilot
draws from an equal mixture, minimizes an estimated importance-weight second
moment on a simplex with a compulsory transition component, and validates and
freezes the selected probabilities. It gives explicit derivatives, projection,
backtracking and a convex optimization gap certificate. Algorithm 0 records
finite solver budgets and failure conditions. Its calibration uses no oracle
likelihood or score.

The safety parameter has a declared mathematical meaning: beta_0 >= epsilon
bounds the predictive-to-proposal density ratio by 1/epsilon. Neither epsilon
nor pilot counts are silently adopted as universal defaults. Frozen contexts,
context weights, scaling and independent pilot requirements are explicit.
The variance identity assumes independent final draws and finite second moments.
For stratified/SQMC designs the same second moment is a proposal-quality
criterion, not their complete sampling variance formula.

## Checks and their limits

- Both documents built successfully: five LaTeX passes and one BibTeX invocation
  per document, including two repair passes after a displayed-equation layout
  correction. The four build-result files preserve exact commands and timings.
- All 49 previous labelled equations and all three previous algorithms are
  unchanged. Eight labelled equations and Algorithm 0 were added. All shared
  labels and the new citation resolve; compiler input records show that both
  PDFs consume the same body.
- Direct SymPy checks confirm the scalar row gradient and directional Hessian
  identities on their nonzero-denominator domains. The first verification
  script incorrectly counted equation environments only; adding align
  environments repaired that diagnostic, and the rerun passed.
- MathDevMCP's focused four-label audit returned needs_evidence: extraction
  quarantined all four targets. Initial symbolic requests were routed as
  matrices and returned unknown. Revised scalar requests proved pilot-density
  cancellation; its two derivative requests were not_encodable. Those are
  tool-coverage limits, not proof or refutation of the whole derivation.
  Exact responses and the independent derivative checks are preserved.
- Rendered monograph pages 272–275 and standalone pages 8–11 were inspected.
  The new chapter has no overfull boxes. The standalone has no warnings or
  overfull boxes. The monograph still has three existing hyperref warnings and
  220 overfull boxes elsewhere, versus 215 in the protected baseline; changed
  pagination/references affect other chapters. It is not a warning-free book.

## Decision

| Decision | Primary criterion | Veto checks | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete the documentation repair | Selection algorithm, derivation, sampling semantics and both builds present | No unresolved labels, changed old equations, or clipping in new material | Finite pilot quality and full recursive value/score performance remain unevaluated | Explain the mechanism; implement and calibrate only under a separate bounded plan if requested | No fitted numerical beta, runtime implementation, score improvement or default promotion |

This is not a stochastic comparison, so no empirical ranking is available.
Minimizing the fixed-cloud local likelihood second moment does not optimize the
analytical score or the complete recursive filter. Changing beta changes later
clouds; independent recursive validation remains necessary. Oracle-free weight
and replication diagnostics can reject an unstable candidate but cannot certify
its accuracy. The branch probabilities and base branch-label uniforms remain
fixed throughout the admitted parameter-evaluation scope.

Post-run red team: the main remaining risk is mistaking an exactly solved finite
pilot objective for a globally optimal filter. The chapter explicitly prevents
that inference. Held-out recursive failures would reject a fitted candidate;
they would not by themselves refute the fixed-context variance derivation.

Sources and source boundaries are in source-audit.md under the artifact root.
The preservation and symbolic checks are reproducible with
verify_calibration_document.py. No filtering experiment, framework/GPU run,
runtime edit, commit or push was performed.
