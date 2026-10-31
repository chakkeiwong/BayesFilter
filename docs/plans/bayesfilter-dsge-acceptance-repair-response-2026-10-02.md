# DSGE review: acceptance qualification still needs statistical repair

The DSGE agent's verdict is correct. The October 1 work delivered experimental
arithmetic, diagnostics and falsification evidence; it did not complete a
calibrated replacement for tuning admission. The October 2 progress-queue
repair solves a separate execution problem. More completed runs cannot repair
the acceptance rule's statistical assumptions.

This review inspected the [DSGE review](../../../dsge_hmc/docs/experiments/bgs/2026-09-28-longer-evidence/review-bayesfilter-acceptance-uncertainty-20261002.md),
the current implementation, the public capability registry, the October 1
calibration/replay JSON, and the existing master next steps. It used saved
evidence and source inspection only. It did not rerun calibration or change
the frozen state-space or DSGE campaigns.

## Checked findings and ownership

| Finding | Checked evidence | Consequence |
| --- | --- | --- |
| Strong-dependence calibration fails | At rho=.995 and 8,192 draws, experimental lugsail reports 37/256 conflicts, Wilson interval 10.67%--19.29%; the declared ceiling was 5% | This tested interval rule cannot be promoted. It is not evidence that every use of lugsail is wrong |
| v6 averages away opposing changes | `_temporal_contrast_function` averages signed block differences across chains; its comparator detects 0/256 opposing-drift fixtures | Its output cannot establish within-chain stability; the existing default's raw screen is also not calibrated merely because it was preserved |
| BGS's 512-draw allocation is structurally insufficient for the experimental diagnostic | Four 128-draw windows; batch sizes 11 and 22 yield 11 and 5 complete batches, below the required eight at the second scale | The eleven insufficient-information reports are honest abstentions, not eleven statistical failures or candidate rejections |
| The particular BGS candidate remains unresolved | L=3, epsilon=0.00044697315517561615, unchanged historical `inconclusive_evidence`; 11/11 saved replay decisions match | Neither acceptance nor rejection of this candidate follows from the diagnostic development |

The counts come from
`artifacts/acceptance-uncertainty-validation-20261001/run-03/result.json`.
The mathematical cancellation is visible in
`bayesfilter/inference/hmc_verification.py::_temporal_contrast_function`;
the experimental window construction and sufficiency counts are in
`bayesfilter/inference/hmc_acceptance_uncertainty.py`.

BayesFilter owns the reusable estimand, uncertainty calculation, decision
semantics, repeated-look/search accounting, preflight feasibility checks and
validation infrastructure. BGS owns target-specific evidence about starts,
preparation, residual geometry and affordable trial lengths, and supplies
representative integration traces. Synthetic stationary failures establish a
library-level statistical gap independently of BGS. They do not establish a
numerical HMC defect or invalidate previously completed posterior assessments.

## Recommended direction and necessary qualifications

Independent repetitions of a fixed prepared trial are a sound **reference
experiment to develop**, not an already validated replacement policy. They
make the sampling unit and conditioning explicit.

Let G denote the frozen preparation: target, coordinates/map, metric, kernel
pair, numerical policy and start bank. For start s, repetition r and a declared
T-transition horizon, define

\[
 A_{rs}(T)=T^{-1}\sum_{t=1}^{T}\alpha(X_{rst},P_{rst}),\qquad
 \mu_s(T\mid G)=\mathbb E[A_{rs}(T)\mid G].
\]

Each repetition restarts from the same declared initial state for that start
using a new random stream; it does not continue the previous endpoint. A trial
also freezes any discarded prefix and the exact measured range. Metropolis
probabilities retain the existing support-rejection semantics and remain
distinct from realized accept/reject frequencies. Conditioning on G excludes
uncertainty from fitting the geometry or selecting the start bank. A change to
G or T defines a different finite-start target and requires fresh qualification.

If repetitions are independent with the same conditional law, then

\[
 \operatorname{Var}\left(R^{-1}\sum_{r=1}^{R} A_{rs}\mid G\right)
 =R^{-2}\sum_{r=1}^{R}\operatorname{Var}(A_{rs}\mid G)
 =R^{-1}\operatorname{Var}(A_{1s}\mid G).
\]

The first equality uses zero covariance between repetitions; it does not
assume independent transitions within a trial. This variance identity alone
does not justify a Student-t interval at small R. The interval method needs
its own finite-sample argument or clearly limited calibration evidence. If
the starts in a trial share randomness, treat that trial's whole start vector
as the independent unit and retain its covariance rather than assuming
independence between start columns. Four different starts are not four
exchangeable repetitions of one start.

The exploratory baseline should keep each \(\mu_s\) and within-start block
contrast. A weighted start-bank mean \(\sum_s w_s\mu_s\) needs declared weights;
it must not hide opposing changes. The product decision about qualifying the
weighted mean versus requiring every start to satisfy a robustness condition
must be stated before validation. No unconditional all-start requirement or
new weighting is silently introduced here. This finite-start quantity need
not equal acceptance averaged under the target's stationary distribution.

Temporal contrasts estimate expected changes over the declared trial. A real
change is not, by itself, proof that the geometry is defective, burn-in is
insufficient, the kernel is unusable, or the posterior is unconverged. Treat it
as a preparation investigation trigger unless an explicit, validated
qualification requirement makes it a veto. Failure to detect a change does
not prove equivalence or stationarity. Likewise, reproducible differences
between deliberately different starts can be an expected finite-start effect.
R-hat remains reporting-only during tuning; posterior convergence stays separate.

## Changes to the remaining program

The master already calls for estimand clarification, fresh validation and
versioned integration. Retain that order, with these more concrete phases:

1. **Specify decisions and preflight information needs.** Describe the
   finite-start reference and stationary comparator separately; declare starts,
   horizon, discarded prefix, independent unit and the role of temporal/chain
   contrasts. Separate supported epsilon repair, a measured preparation
   discrepancy that warrants investigation, and insufficient evidence. Preserve
   divergence, invalid-state, movement, recurrence and fresh-verification checks.
   Count every planned trial, including invalid/aborted trials; no qualification
   from a favorable subset. Add a framework-free information-feasibility report
   before launching expensive evidence work.
2. **Build the replicated-trial reference on cheap controlled examples.**
   Reset all starts between repetitions, use explicit seed identities, preserve
   each start and temporal contrast, and retain any cross-start covariance.
   Compare v5, v6 and the failed batch-means/lugsail procedure as recorded
   baselines. Do not add repetitions inside an already frozen campaign or
   reinterpret historical decisions. The expensive BGS case is an integration
   target after this phase, not the calibration tuning set.
3. **Calibrate errors, informative delivery and cost together.** Use fresh
   held-out seeds, exact controlled expectations where available, and original
   denominators. Assess false directional/conflict decisions, false compatible
   decisions, material-change detection, unresolved frequency and runtime at
   declared allocations. Budget candidates, comparisons and adaptive looks,
   and use disjoint fresh verification. A rule that abstains on everything
   fails useful-delivery requirements even if its false-positive rate is low.
   Derive precision requirements and assess cost before fixing repetition
   counts; neither small-R t intervals nor a large arbitrary replication count
   are justified defaults. Predeclare error/delivery criteria and a bounded
   compute allocation before new claim-bearing calibration.
4. **Integrate only a supported scope.** If the reference is reliable but
   unaffordable for BGS, retain it as a comparator and investigate a cheaper
   procedure against it. Do not replace admission merely to unblock L=3.
   Any changed public policy needs explicit estimand/seed/conditioning metadata,
   both applicable public entry points, checkpoint/receipt versioning, fresh
   evidence and updates to the official tuning chapter and agent reference.

For the existing experimental design, the deterministic feasibility condition
is

\[
 \left\lfloor\frac{\lfloor T/4\rfloor}{2b}\right\rfloor\ge m,
 \quad\text{equivalently}\quad T\ge 8bm,
\]

where b is the frozen base batch size and m the minimum complete-batch count.
For b=11 and m=8 this gives 704 draws, a derived structural floor, **not** a
recommendation that 704 draws suffice statistically. If b is recomputed from T,
the feasibility test must be recomputed too. The requirement to resolve
dependence remains separate even after this arithmetic floor passes.

| Tricky case | Required check |
| --- | --- |
| Stationary rho=.995, with long persistent stretches | False decisions and interval coverage across independent trials; availability alone is not a pass |
| Equal opposing or one-start-only temporal changes | Preserve per-start contrasts; signed averaging must not erase a material change |
| Different fixed starts and a correct kernel with transient behavior | Estimate conditional finite-start behavior; do not mislabel all differences as a stationarity failure |
| 512 draws under the existing batch allocation | Declare design infeasibility before expensive sampling; distinguish this from a stochastic variance failure |
| Few repetitions, near-zero/one acceptance or rare events | Verify interval behavior and delivery; do not interpret zero observed variance as certainty |
| Many adaptive epsilon candidates and repeated looks | Assess the whole selection/verification procedure, including fresh streams and all non-deliveries |
| Invalid or aborted trial and a favorable surviving subset | Preserve health vetoes and planned denominators; no survivor-only qualification |

## Review decision

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept DSGE's principal critique | Current code and saved counts agree with the review | Experimental calibration fails promotion | General admission rule remains unresolved | Refine the master with the phases above | The BGS candidate is bad, or the synthetic failure is BGS-specific |
| Develop independent repetitions as a reference | Conditional estimand and variance identity are explicit | No new policy may be promoted without validation | Interval validity, required repetitions and affordable cost | Preflight/cheap reference development, then held-out calibration | Small-R Student-t validity or stationary acceptance |
| Preserve active campaign semantics | Frozen source and original evidence remain the comparator | No new numerical veto established by this review | Existing heuristic admission limits remain | Continue the separately authorized resource recovery | That recovery closes the statistical gap |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Prior synthetic calibration vetoes promotion; invalid numerical trials remain unqualified |
| Statistically supported ranking | No general ranking among estimators, candidates or samplers is established |
| Descriptive-only differences | Saved BGS block changes, runtime and non-calibrated comparison metrics |
| Default-readiness | Statistical admission repair remains open; compatibility is not validation |
| Next evidence needed | Error, power, useful-delivery and cost evidence for an explicitly declared estimand |

Skeptical review: wrong estimands, heterogeneous-start pseudo-replication,
opposing-sign cancellation, under-budgeted information, survivor selection,
uncounted adaptive looks and promoting a noise test into a convergence gate
are the principal risks. The amended phase order exposes them before a new
large campaign. This response updates the plan; it does not implement or
promote the proposed statistical replacement.
