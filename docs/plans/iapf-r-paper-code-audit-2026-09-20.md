# R iAPF code and paper audit

2026-09-20. Owner request: trace the implementation against the paper, identify
remaining gaps, review a repair plan, and execute it. This continues the
authorized independent R reference for the first linear-Gaussian study.

## Question and source audit

Does the executed R reference implement the stated mathematical algorithm, and
does the small d20 fitting residual describe a useful approximation away from
the particles used to fit it? The source is the locally stored Guarniero,
Johansen and Lee accepted manuscript, Sections 2--3, 5.1--5.2, equations
(5)--(6), (13), (15)--(16), and Algorithms 3--5. Public comparator
`.localresources/code/sempreteamo-iapf-a8811439/iapf.R` is inspected as a
third-party comparator; its authorship is unverified and its objective differs
from equation (15).

Checked call chain: Python driver -> R replication runner -> `iapf_iterate`
-> `iapf_apf` and `iapf_fit_backward` -> Gaussian fit, two-component proposal,
sampler and weights. The Gaussian floor enters both proposal components and
the importance correction. Backward targets use the newly fitted next-time
twist. Adaptive resampling retains weights and normalizing factors. The final
likelihood is a fresh run. These operations agree with the cited algorithms.

Confirmed gaps before execution:

1. `paper_eq15` accepts a nonzero optimizer convergence code. Reporting the
   code later does not establish the numerical minimization in equation (15).
2. Fit failures save points and targets, but omit the next-time twist and the
   outer iteration. Their target cannot be evaluated independently from the
   saved failure alone. This prevents the discriminating coverage check.
3. The driver snapshots sources but executes live workspace paths. An edit
   during a run can invalidate the claimed source provenance.
4. Equation (15) has the already derived amplitude escape. `relative_l2` fixes
   that escape by changing the objective. Neither it nor the alternative
   `nlminb` solver closes the strict paper-replication gap.
5. Tiny training residual and a converged solver do not establish accuracy
   beyond the training cloud. The d20 failures need independent evaluations.
6. The paper does not specify its optimizer/initialization, positive floor,
   early iteration doubling convention or original data seeds. Published
   five-dimension, 1000-repeat replication remains incomplete.

## Plan and evidence contract

1. Reject unconverged fits in every fitting mode, preserving their inputs.
   Save the frozen next-time twist, model, iteration and particle history with
   a failed fit. Add meaningful regressions through the actual controller.
2. Run numerical consumers from their captured source tree. Test that changing
   the live source after capture cannot change the invoked computation.
3. Replay the two saved d20 failures with exactly their prior seeds/settings.
   Require matching training inputs and parameters before interpreting them.
   Evaluate the frozen failed fit, its original initialization, a constant
   twist, and a diagonal Gaussian matching the moments of the normalized
   frozen backward target on independent APF clouds and exact smoothing draws.
   This target is an analytically tractable two-Gaussian mixture in the linear
   model. The moment baseline is diagnostic only, not an iAPF repair.
   Record both Gaussian-only and actual Gaussian-plus-floor residuals, squared
   target effective sample size, normalized log-ratio spread, and per-time
   context. Repeat each independent support check over six seeds, N=1000.
4. Record the mathematical interpretation, remaining gaps and next justified
   repair in the master program and concise checkpoint. Preserve old results.

Primary engineering criteria: regression tests pass, source execution is
frozen, failed-input replay matches, and all required diagnostics are finite.
Mismatch or invalid probability computation is a continuation veto. Training
residual, independent residual, effective point count and log-ratio spread are
explanatory diagnostics, not promotion criteria. A solver failure remains a
promotion veto for that configuration; it is not rejection of iAPF. No new
method or numerical default will be selected using these diagnostic datasets.
There is no stochastic ranking claim from six support draws. Published
replication, runtime superiority, TensorFlow agreement, LEDH or HMC readiness
cannot be concluded.

Output: `docs/plans/artifacts/iapf-r-paper-code-audit-20260920-01/`.
The existing budget-enforcing driver stores the experimental launch at
`docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01/attempt10-code-audit/`;
the audit result and checkpoint link to it.
One diagnostic launch, at most 150 worker seconds, charged against the existing
258.803021 seconds and final remaining experimental slot. Focused tests and
mechanics use at most the remaining 80 seconds of the prior mechanics budget.
CPU-only base R 4.1.2, CUDA hidden, one BLAS/OpenMP thread. Preserve command,
Git HEAD, source/paper hashes, seeds, R version, duration and all failure RDS
files. No packages, GPU runs or external services are needed.

## Default and assumption audit

| Choice | Provenance and reason | Failure risk and early check | Status |
| --- | --- | --- | --- |
| Existing d20 seeds and maxit5000 | Exact replay of the two failures | A changed stream invalidates comparison; match saved inputs/parameters | Frozen diagnostic |
| Relative objective/solvers | Prior labeled repair, not eq15 | Overfitting rare points; independent support checks | Hypotheses |
| Floor power2 | Prior reconstruction; paper only requires positive c | Dominant floor can hide poor Gaussian; report both | Unverified reconstruction |
| Independent APF clouds | Algorithm 3's actual support distribution | Finite support misses relevant region; add exact smoothing draws | Diagnostic |
| Exact smoothing draws | Linear model and full covariance Gaussian oracle | Wrong time/conditioning; test against a joint Gaussian calculation | Reference only |
| Six seeds, N1000 | Small discriminating diagnosis within remaining budget | Not enough for likelihood ranking; report every support draw | Convenience, no promotion |

## Skeptical review before execution

Self-review: the earlier idea to loosen convergence tolerances or increase the
iteration cap is insufficient because a tiny residual can depend on one or
two points. This plan first distinguishes that problem from termination.
Equation (15) is retained explicitly; the alternative is never relabeled as
paper replication. The comparator ladder includes cheap fits and a tractable
oracle. Independent clouds and smoothing draws are separate situations, not
pooled. A positive diagnostic cannot authorize 1000 repeats. Failed replay,
source mismatch and budget exhaustion have explicit stops; a poor candidate
instead determines the next fitting investigation. No GPU/XLA evidence is
claimed from R. Verdict: proceed. No independent reviewer was launched.

Pre-execution review correction: the smoothing marginal is a useful validation
distribution but the wrong density to approximate the backward function.
The diagonal moment baseline therefore uses the actual frozen backward target,
not the smoothing density. Reject equation-(15) objective underflow as well as
nonconvergence; a rounded zero is not an attained mathematical optimum.

Post-replay extension within the mechanics allowance: all frozen functions in
these two cases are Gaussian mixtures. Compute the continuous relative-L2
error under the exact smoothing Gaussian by analytic Gaussian product
integrals. Check the formula against independent one-dimensional quadrature
before evaluating the captured fits. This removes Monte Carlo uncertainty for
this fixed-function diagnostic; it still says nothing about full-filter
likelihood variance or future runs. At most 10 seconds, no optimizer or pilot.
