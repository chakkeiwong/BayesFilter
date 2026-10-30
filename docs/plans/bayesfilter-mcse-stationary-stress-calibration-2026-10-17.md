# Strong-dependence and skewness checks for posterior stopping

17 October 2026. Status: 65 focused tests and the six-condition smoke pass;
the serious study launched at 18:44 China time through the bounded CPU service. This is a
bounded CPU reference phase of the October 7 modern-HMC master. The existing
SIR GPU services and their frozen numerical policies continue independently.

## Question and evidence contract

Does the existing posterior assessment report usable uncertainty, unavailable
estimates and nonstopping honestly under stronger serial dependence and skewed
marginals? Compare the original lugsail policy with the explicit optional
`lugsail_max_batch_means`, ordinary batch means and autocorrelation estimates
on identical stationary reference paths. The latter two are cheap classical
adversaries. The question concerns precision assessment, not HMC transitions,
initialization or warm-up adequacy.

Use four independent stationary Gaussian AR(1) chains at rho = 0, .99, -.99.
Assess both X=Z and X=exp(Z), using the same latent paths. Monitor X, X squared
and the binary event |X|>2. For Gaussian margins the truths are 0, 1 and
2 Phi(-2); for lognormal margins they are exp(1/2), exp(2) and
1-Phi(log(2)). All lognormal moments exist, although their magnitudes make
finite-sample estimation difficult. Marginal laws sharing a latent path are
paired conditions, not additional independent replications.

The actual frozen TensorFlow assessment runs at looks 1000, 2000, ..., 10000
per chain. Record every method's first passing look and the fixed 10000-draw
endpoint, including invalid estimates and nonstoppers. Keep the inherited
R-hat <=1.01, bulk/tail ESS >=400, MCSE/SD <=.02 and lugsail r=3, c=.5,
minimum 20 batches. These are the comparator's operational criteria, not newly
validated universal defaults. Reference error must never enter stopping.

The engineering pass criterion is reproducible complete-prefix records, exact
source/seed provenance, correct analytic comparators and preserved denominators.
Fixed and stopped 95% normal-interval misses are a falsification screen for
uncertainty calibration. Report exact binomial intervals, separately from
availability and stopping probabilities. For the 6 conditions x 4 estimators x
3 quantities x 2 endpoints = 144 predeclared marginal screens, also report
Clopper-Pearson intervals at alpha=.05/144. A lower endpoint above .05 is
evidence against nominal miss probability for that specific reported estimand.
Stopped coverage is conditional on producing a valid stopped interval;
nonstopping remains in its own denominator. Passing this low-power screen
does not establish nominal, simultaneous or anytime coverage.

Pairwise original/maximum differences use the existing conservative discordance
intervals and retain unmatched endpoints. They are pointwise diagnostics;
no estimator ranking or new default will be selected. Continuous variance
ratios, runtime and stopping counts are explanatory only. Source changes,
wrong numerical imports, nonfinite reference generation, invalid carried
records or corrupted artifacts veto continuation until localized repair.
Exhausting the allocation is a resource stop. A nonstopping or undercovered
condition is evidence about the current assessment, not a veto on completing
the other conditions or on the HMC research direction.

## Independent finite-length comparators

For a stationary quantity Y with covariance gamma(h), n observations in each
of C independent chains give

    Var(mean) = [gamma(0) + 2 sum_{h=1}^{n-1}(1-h/n) gamma(h)]/(C*n).

This follows by counting n-h pairs at lag h in each chain and pooling the
independent chain means. For Gaussian X=Z, gamma_X(h)=rho^h and
gamma_(X squared)(h)=2*rho^(2h), by the fourth moment of a centered bivariate
normal. For lognormal X=exp(Z), the Gaussian moment-generating function gives

    E[exp(k*Z_t+k*Z_(t+h))] = exp(k*k*(1+rho^h)),
    gamma_(X^k)(h) = exp(k*k)*expm1(k*k*rho^h), k=1,2.

At h=0 these yield the known marginal variances. At rho=0 only that term
remains. Test these limiting cases and direct small covariance sums before
execution. At selected stopping lengths these formulas remain fixed-length
comparators, not conditional stopped variances. No analytic variance claim is
made for the binary event in this phase.

## Defaults, assumptions and pre-mortem

| Choice | Provenance and purpose | Failure mode and early check | Status |
| --- | --- | --- | --- |
| rho 0, +/-.99 | New stress hypotheses; exact covariance is available | Positive persistence makes .02 unattainable at this cap; compute exact coordinate/square scales before interpreting stops | Diagnostic hypotheses |
| exp(Z), log-scale SD 1 | Convenience skewness control with known finite moments | Large moments make normal intervals poor despite correct arithmetic; compare exact moments and interval misses | Diagnostic hypothesis |
| 128 units per rho, shared across two laws | Inherited completed study size; miss-rate SE near .05 is about .019 | Weak power and dependent law pairs; retain per-cell intervals and latent-unit counts | Bounded design, not certification |
| Four chains, ten looks, assessment thresholds | Existing public-posterior comparator | Acceptance or stationary starts could be confused with burn-in evidence; explicitly record stationary initialization and no HMC transitions | Baseline only |
| Original plus optional maximum, BM, autocorrelation | Existing implementation and previous calibration | A larger variance could falsely be called conservative; inspect misses and availability, retain every arm | No default selection |
| CPU/XLA with GPUs hidden, two threads | Independent reference exception and existing study environment | Wrong package/snapshot or CPU competition; check imports/hashes and measured occupied time | Explicit diagnostic exception |
| Seeds [74117,rho_group,replication] | Fresh convenience namespace, declared before results | Reused/misassigned streams; test group identity, prefix validation and unchanged continuation bytes | Reproducibility choice |

The strongest misleading success would be zero stops under persistence being
reported as correct coverage, or high lognormal square error being hidden by
validity conditioning. Always report all denominators, first-pass counts and
fixed-endpoint errors. A failure could instead be an oracle, seed-layout or
import defect; small exact-reference checks and a one-replication smoke precede
the serious study. Stationary paths cannot reveal missing modes, common bias,
burn-in failure, sampler invariance or state-space target correctness.

## Budget, environment and executable sequence

Allocate **4200 CPU-worker seconds** from the reconciled SGQF SNAPER release
137294.5091979513, leaving **133094.5091979513** unallocated there. The release
receipt is
`artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/snaper-feasibility-01/released-reservation-01.json`.
This does not alter the independent SIR pool or caps. The cap comprises two
attempt ceilings of 1800 seconds (inherited from the prior calibration, whose
total work was approximately 2019 seconds) plus a convenience 600-second
engineering/smoke/audit allowance. These are spending limits, not forecasts.
If overhead uses more of the total, reduce the remaining attempt allowance.
Charge measured supervisor-inclusive occupation once; never add child time
again. Observer active time is separate; idle waiting is excluded.

Artifacts:
`artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/stationary-stress-20261017-01/`.
Preserve every attempt in a fresh directory. Freeze the tested diagnostic
driver before launch. Reuse the reviewed precision overlay at
`priced-comparison-01/precision-repair-01/source`, based on numerical commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`; capture its dependency hashes.
Extend the existing calibration CLI with a separately named stress profile,
preserving historical condition/seed layouts. Generalize continuation group
validation explicitly; reject a changed driver, design, seed, method, source,
versions or unfinished parent. Focused tests cover analytic oracles, skewed
moments, historical layout, group changes, mixed-law pairing and resume.

Environment: `CUDA_VISIBLE_DEVICES=-1`, `TF_FORCE_GPU_ALLOW_GROWTH=true`,
`TF_NUM_INTEROP_THREADS=2`, `TF_NUM_INTRAOP_THREADS=2`, `OMP_NUM_THREADS=2`.
Python: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. Use the calibration
CLI with `--profile stationary_stress --replications 128 --seed 74117`, the
above `--source-root`, a fresh `--output` and `--max-wall-seconds` <=1800.
A single bounded continuation may carry every completed original record from
a terminal budget-exhausted first attempt, using the same frozen driver and
seed design. Localized harness repairs consume this allocation, with focused
checks and fresh outputs. Preserve any incomplete denominator at the final cap.

The separate one-replication smoke uses seed **74116**, chosen only to keep
its streams disjoint from the serious seed 74117. Its cooperative wall cap is
180 seconds and its supervisor cap is 210 seconds, convenience spending limits
within the 600-second engineering allowance, not completion forecasts. A smoke
pass checks all six conditions, four methods, endpoint structure and source
provenance; it supplies no calibration or estimator-ranking evidence.

For the durable study, each numerical attempt has a 1740-second cooperative
cap inside its 1800-second reservation, with a 1790-second outer watchdog.
The 60-second cooperative headroom and 10-second watchdog headroom are
convenience shutdown allowances, not evidence about assessment runtime. At
most two attempts run; only a returned, terminal `budget_exhausted` result
that passes the independent source/prefix audit may resume. A watchdog kill,
source mismatch or exception requires inspection. The same supervisor runs
the terminal reader and writes measured settlement within the total 4200
seconds. It cannot change the SIR campaign or any numerical policy.
The service hard cap is 3900 seconds: 3600 for the two attempt reservations,
120 for two reader watchdogs and 180 administrative headroom, all spending
ceilings inside the 4200-second allocation. The measured preflight charge is
36.21836149925366 seconds. Static supervisor review confirms disjoint smoke
and study seeds, fresh outputs, CPU-only child environments, at most two
attempts, audited prefix continuation and no policy changes. No substantive
plan flaw remains after this review.

The run manifest records checkout commit, numerical snapshot and dependency
hashes, frozen driver hash, actual command/environment, package versions,
deliberately hidden GPUs, synthetic data identity, all seeds, occupied time,
this plan and the result note. A terminal audit independently reaggregates
records and miss intervals, validates every stream and carried prefix, and
settles measured costs. The result note includes decision and inference-status
tables and a post-run red-team discussion.

## Exact finite-batch explanation at closeout

Before interpreting the stress results, compute an independent deterministic
expectation for the unmasked batch-means and raw lugsail formulas. This does
not generate new chains, tune any setting or select an estimator. It explains
whether the inherited square-root batch length can estimate the fixed-length
mean variance in these known covariance models. Charge its measured runtime
inside the existing engineering allowance.

For a batch length b and a=floor(n/b) complete batches, write V(m) for the
exact variance of one chain's m-draw mean. The unbiased sample variance of
the correlated batch means has expectation a/(a-1)[V(b)-V(ab)]: expand
the sum of squared batch means about their average and take expectations.
Thus E[BM(b)]=ba/(a-1)[V(b)-V(ab)]. The raw lugsail expectation is
[E[BM(b)]-c E[BM(floor(b/r))]]/(1-c). Divide by Cn for the four-chain
pooled mean-variance estimator, matching the implementation's complete-batch
truncation and full-n divisor. Compare with the exact C-chain V(n).

Use n=10000, b=100, floor(b/r)=33, r=3 and c=.5 from the frozen policy.
Check the derivation against a direct small covariance-matrix trace, including
positive/negative correlation and lognormal coordinate/square quantities.
These are expectations before the per-chain positivity screen, not expectations
conditional on available intervals, observed Monte Carlo ratios, a formula for
the optional maximum, or variances conditional on a stopping time. A negative
raw expectation is a design limitation, not proof of a coding error.
Skeptical review: the known covariance and exact trace avoid a fitted reference;
batch truncation, pooling and the nonlinear validity/max operations are explicit.
No threshold, numerical source, method or campaign budget changes.

## Skeptical pre-execution review

### Closeout reader repair, 17 October, 19:06 China time

An exhaustive check over all binomial counts for denominators 1..128 found
73 reader comparison failures caused by root-inversion versus beta-quantile
roundoff, with largest absolute discrepancy 4.998224056862455e-13. The former
absolute tolerance 1e-13 is too strict near zero. Use 1e-12 only for comparing
these two interval computations: twice the measured worst discrepancy. Keep
analytic-oracle tolerances, coverage thresholds, numerical driver and seeds
unchanged. Receipt: `stationary-stress-20261017-01/binomial-interval-check-01.json`.
It costs 25.42711169505492 post-launch engineering seconds and is charged once.

The live frozen auditor is preserved. If its first terminal audit rejects this
roundoff, allow the bounded numerical attempt to finish, inspect the exact
failure, and audit that same prefix using a tested fresh auditor snapshot.
A corrected read-only audit may establish eligibility for the already-planned
second attempt under the remaining original 4200-second allocation. It cannot
erase the failed engineering attempt, change the driver or redo completed
replications. A genuine provenance, record, source or policy failure remains
a stop. Skeptical repair review: this affects reader arithmetic only; no
stochastic outcome selects the threshold or expands the scientific claim.

Passed after making the following limits explicit. The comparator is the
actual public assessment and original estimator, rather than a new surrogate
stop rule. Slow mixing is expected to miss the precision cap; that is not a
harness defect. The normal intervals are a falsification target, not guaranteed
intervals. Stationary initialization, skewed large moments, shared law pairs,
conditional stopping, multiple testing and limited power remain explicit.
Analytic covariance checks precede stochastic evidence. CPU diagnostics use
the frozen precision overlay, not the mutable checkout or running GPU source.
The 4200-second allocation is an explicit use of settled existing funds, with
stop conditions and complete denominators. No material scientific/default
change or additional compute grant is implied.
