# Minimal shared importance correction

The owner requests a direct local patch and matched numerical verification,
superseding the integration-worktree/dependency-extraction steps in the earlier
merge plan. No broad merge or default-policy change is part of this step.

Question: can one component-selection implementation reproduce the existing
LGSSM/KSC finite likelihood and analytical score with each particle's own
component, and the other branch's range-bearing calculation with all components?
Keep the incoming outer particle weight in both cases. One component means the
generating ancestor of each particle, not a globally one-hot particle cloud.

Implementation: retain the existing flow, UKF, reset and analytical recursion.
Expose the affine map and its derivative from that same flow loop; evaluate
Gaussian transition/proposal densities through one diagonal/full-mixture engine.
Preserve the ancestor default and unsupported-model compatibility; reject an
unsupported mixture request. Forward the option through both batch wrappers.

Evidence: save current-checkout outputs before editing, compare fixed LGSSM and
KSC inputs afterwards, and compare range-bearing outputs against source commit
f5e69d716818ff9ecf28b732a8952f5ec7f87739. Use identical inputs, FP64 CPU/XLA
as an explicit reference exception, complete score directions, finite outputs,
and recursive Contract-E fixtures as well as one-step checks. Include density
and derivative checks, nonuniform incoming weights, and consumer wiring checks.
Initial parity tolerances: absolute value 1e-8 and score 1e-7, plus relative
1e-8. A mismatch triggers localization; do not relax it silently. Report actual
values and differences, not only pass/fail. Mechanics parity is the criterion;
no scientific ranking, oracle accuracy, GPU default-readiness or PP/SIR result
follows from it. These are engineering checks, not a research comparison of
methods, so a heuristic accuracy ladder is outside this contract.

Budget: at most 30 CPU job-minutes for bounded engineering checks; stop and
localize nonfinite results, mismatched inputs, changed source files or unrelated
branch differences. No long campaign, tuning or HMC. Preserve complete logs and
manifests under docs/plans/artifacts/ledh-marginal-minimal-patch-20261006-01.

Assumptions: full-rank Gaussian transition, flow coefficients independent of the
current innovation, one child per ancestor, one annealing stage. These are
eligibility conditions, not new model defaults. Existing correction controls
are frozen comparison inputs, not newly tuned settings. Their failure can veto
this fixture but cannot disprove the correction. The diagonal path must avoid
quadratic work; the full path uses the repository chunk selector.

Skeptical audit: the pre-edit checkout is the compatibility baseline; the pinned
other branch is an implementation comparator, not an oracle. Compare the same
finite program and parameterization; inspect existing branch differences if
parity fails. Validate total score terms (weights, means, covariance and flow),
not just likelihood. This bounded plan answers the owner's request without
importing unrelated changes. Audit passed for execution.

Completed: all 16 matched compatibility comparisons pass, and 35 distinct tests
pass after fixture repairs. Actual values, differences, limitations and the
decision are recorded in
`docs/benchmarks/ledh-marginal-minimal-patch-results-2026-10-06.md`.
