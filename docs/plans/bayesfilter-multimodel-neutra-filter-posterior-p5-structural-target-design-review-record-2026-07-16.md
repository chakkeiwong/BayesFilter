# P5 Structural Target-Design Review Record

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Reviewed path:
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p5-structural-target-design-subplan-2026-07-16.md`

## Local Skeptical Review

Decision: `PASS_FOR_TARGET_DESIGN_EXECUTION`.

The first local review found four material defects and repaired the same plan
before execution:

1. The original information check could have included the source prior and
   chart Jacobian, which would manufacture full rank even when the data
   likelihood did not identify a parameter. The repaired gate is
   likelihood-only and uses predictive-innovation mean and variance
   sensitivities in source coordinates.
2. A raw physical-parameter Hessian condition number was scale-dependent. The
   repaired source-coordinate Fisher surrogate has explicit rank, minimum
   eigenvalue, and condition-number criteria at truth and fixed neighbors.
3. T=100 was described as a convenience but lacked an exact decision. The
   repaired plan uses three disjoint T=200 design trajectories, exact prefixes,
   and a prospective `TARGET_DESIGN_HORIZON_REPLAN_REQUIRED` branch.
4. Adding 0.04 to a covariance after structural propagation need not produce
   pointwise deterministic residuals. The repaired negative control has an
   explicit independent `eta_k` innovation, a two-dimensional innovation
   contract, off-manifold residuals, and a 0.04 covariance increment.

The second local review checked wrong baseline, proxy promotion, silent
defaults, stop/repair rules, target/HMC boundary, environment, and artifact
coverage. The chapter constants remain synthetic truth rather than defaults;
support boxes remain prospective convenience hypotheses; prior-predictive and
information screens cannot prove posterior correctness or global
identifiability; and no HMC, target signature, or NeuTra is allowed in this
rung. No remaining material defect was found.

## Claude Review Attempt

A one-path, read-only Opus/max-effort review was requested with Claude limited
to `Read`. The managed platform rejected the call before process creation
because sending the private workspace plan to an external Claude service was
classified as unacceptable data-exfiltration risk. Claude produced no review
content or verdict. No workaround or indirect disclosure was attempted.

Under the current BayesFilter review-proportionality policy, unavailable
advisory review does not block trusted local research when a material local
review, evidence contract, bounded compute, and scientific gates are adequate.
This record documents that limitation; it is not represented as Claude
agreement.

## Verdict

`LOCAL_VERDICT: AGREE`

`CLAUDE_VERDICT: UNAVAILABLE_PLATFORM_DENIED_EXTERNAL_DISCLOSURE`
