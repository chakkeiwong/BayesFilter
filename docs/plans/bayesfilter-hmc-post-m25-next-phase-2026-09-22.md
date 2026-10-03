# HMC follow-up after the audited M25 repair

M26 has completed the first three steps below. Its
[result](bayesfilter-hmc-m26-lifetime-and-policy-result-2026-09-22.md) and
[post-M26 plan](bayesfilter-hmc-post-m26-next-phase-2026-09-22.md) supersede this
execution agenda and opening budget; retain this document as the reviewed
predecessor.

This is the refreshed continuation design, not a claim that every remaining
scientific gap has a known successful fix. The preceding
[result](bayesfilter-hmc-gap-closure-result-2026-09-22.md) closes the implemented
option/accounting repairs, the specified exact-AR(1) allocation cell and the
supplied-map GPU engineering matrix. General posterior sufficiency requires
further evidence. No finite diagnostic can certify arbitrary burn-in or discover
an unknown missed mode with certainty.

## Order and decision rules

1. Diagnose repeated-fit process lifetime before another large batch. The
   beta-binomial worker retained roughly 90 GiB after writing its final output.
   Measure resident memory, live graph/runner counts and shutdown time at fit
   boundaries on a short repeated Gaussian/beta-binomial sequence. Compare the
   existing persistent worker with ordinary process isolation between complete
   fits. Preserve dataset IDs, seeds, candidate receipts and complete-fit
   independence. Prefer bounded worker recycling if it fixes resource growth;
   cache or refactor numerical code only when profiles identify the cause.
   Require unchanged numerical results, restart behavior and failure retention.
   Do not use a fast exit to hide missing output or an invalid fit.

2. Run a posterior-policy development study on Gaussian, beta-binomial and
   rotated-Gaussian targets using the newly wired options. Start by reading the
   saved quantity-level failures and estimating dependence and scale separately.
   Compare the original lugsail/count policy with the existing autocorrelation
   option and justified longer readiness windows/counts. Keep the original
   model-specific precision requests (.05 for Gaussian/rotated Gaussian, .005
   for beta-binomial) and health/R-hat thresholds. Do not transplant the AR(1) 20000/80000
   allocation as a universal default. Use pilot estimates and independent
   analytic moments to nominate each model's allocation; fresh replications
   must assess delivery and all declared mean/median intervals jointly in the
   sense that every quantity meets its predeclared screen. Preserve the
   existing pointwise .90 lower-bound screen unless a new inferential objective
   is explicitly chosen. Missing fits and caps stay in the full denominator.

3. Keep tuning membership separate from posterior member selection. A fixed
   L=3 assessment rule missed two beta-binomial fits that had verified siblings.
   A new experiment can predeclare `first_verified`, or a finite identity-based
   list of siblings, before looking at their new posterior streams. Freeze
   that rule and count before confirmation. Assess each requested member
   separately; never choose a posterior winner from the same evidence or count
   siblings as independent complete fits. A sibling failure does not remove
   other tuning members.

4. For the funnel, use exact whitening as the positive control and improve a
   supplied partial map's tail behavior under a new scope. The current saturating
   maps leave `exp(2*s(v)-v)` unbounded. An analytically supplied map with a
   bounded residual is a useful new control, but it must be implemented through
   a supported frozen codec and independently checked for target, Jacobian,
   score and inverse starts. It is not evidence of learned-map quality. As an
   alternative, predeclare a bounded sibling assessment of the existing maps.
   Retain model-coordinate precision and numerical health requirements; do not
   launch a third unmotivated epsilon grid. For the exact map, derive precision
   costs in the actual funnel quantities rather than assuming latent Gaussian
   precision transfers unchanged.

5. Reprice complete-fit defect power after any measured maintenance repair.
   Keep full ordinary preparation/tuning/sampling inside each fit, baseline and
   no-op arms, and the mathematically defined location-shift defect. The old
   design costs roughly 20 CPU hours per 384-fit experiment per arm. At least
   17 independent experiments with no misses would be needed for a two-sided
   95% binomial lower bound above .8, because `.025**(1/17) > .8`.
   That optimistic cost is over 340 CPU hours per arm at the measured rate.
   This is an affordability diagnosis for that design, not a theorem that all
   possible power designs are expensive. A redesigned statistic/inventory needs
   an inspected statistical justification and fresh null calibration; it cannot
   inherit the old power claim. Without a justified affordable design, report
   power as unresolved and do not substitute a small nonrejection.

6. Execute exact MacroFinance integration only on a newly qualified matching
   target/source, data, prior and coordinate bundle. The consumed bootstrap run
   must not be rerun. Require a same-target uncertainty-bearing independent
   reference for the separate full-joint MIDAS claim. The missing-input inventory
   is explicit; broader integration tests cannot manufacture these inputs.
   Learned-map training remains an independent upstream work package requiring
   its own batched GPU protocol and downstream posterior validation.

## Execution boundaries and skeptical review

Steps 1–3 were executed under the bounded
[M26 plan](bayesfilter-hmc-m26-lifetime-and-policy-plan-2026-09-22.md).

The terminal ledger leaves 84512.57 CPU and 84803.52 GPU worker-seconds under
the existing authorization. These are available resources, not a commitment
that every scientifically adequate inventory fits. Use a fresh versioned root
and a source snapshot for each new scope. First reserve a bounded cost/mechanism
pilot inside the remainder; only then freeze a numerical confirmation inventory
with its measured cost, seeds, precision calculation, comparator and stop rules.
No new compute authorization is needed for local repairs within this ceiling.

Primary engineering criteria are complete independent fits, bounded process
lifetime and unchanged numerical/restart behavior. Posterior delivery and
interval calibration are separate statistical criteria. Nonfinite transitions
veto the affected posterior; source corruption, invalid target/Jacobian, broken
reference, essential missing input or exhausted budget stop the affected
experiment. Failed candidates, caps and missed modes trigger the next declared
repair; they are not direction-wide continuation vetoes. Runtime, acceptance
and local curvature remain explanatory. No superiority or default promotion
follows from a pilot or a passing tuning screen.

Skeptical review: the old window/count allocation was inadequate in a known
exact law, but that does not determine HMC allocations. The old fixed-L
assessment can have no selected member even when tuning succeeds; its missing
outcomes must not be silently replaced. Changing a map or estimator changes
the validation scope. A process-lifetime repair must not corrupt checkpoint or
failure evidence. Power affordability must count complete independent fits,
not correlated draws or sibling candidates. These distinctions prevent the
next experiments from answering a cheaper but different question.

After each bounded phase: classify the outcome, reconcile all worker time,
repair confirmed implementation failures, record the result, then refresh the
next experiment from that evidence. An observed failure is not repaired by
relaxing its criterion. This design deliberately leaves target-specific counts
and the confirmation inventory to measured pilots; those values are not yet
established by the M25 result.
