# Fixed-length and stopped-output MCSE diagnostic

Date: 12 October 2026. This bounded CPU reference study runs while the reviewed
GPU cohort prices execute. It uses at most 1800 seconds of the priced
comparison's existing 3600-second local allocation, not an additional budget.
Output: `N5/generic-adaptation/priced-comparison-01/mcse-calibration-01`.

The question is whether the existing lugsail MCSE, plain batch means and
positive-pair autocorrelation estimates provide usable uncertainty on stationary
antithetic and persistent chains, and how their errors change after the actual
retained posterior assessment first passes. These are controlled diagnostic
chains, not HMC runs or evidence about a sampler's stationary distribution.
The exact finite-length Gaussian AR(1) variance is the independent comparator;
Student-t(5) marginals add a finite-fourth-moment heavy-tail condition. No
fallback or estimator default will be selected from this study.

Use the same immutable `full-model-source-e68cdf1c4` numerical snapshot as the
priced comparison. Generate four independent stationary Gaussian AR(1) chains
with rho=0,.9,-.9, and transform the same latent paths by the exact normal-CDF/
t5-inverse-CDF map for a paired heavy-tail diagnostic. Rho values are inherited
stress controls, not representative frequencies of real HMC problems. Monitor
the coordinate, its square, and the event |x|>2. The last is explicitly typed
binary so the library's event-probability diagnostic applies. Means are known
exactly: Gaussian 0,1,2 Phi(-2); t5 0,5/3,2 P(t5>2). All are evaluated by the
same TensorFlow precision and posterior assessment functions used by consumers.
Reference generation and exact binomial intervals use diagnostic NumPy/SciPy.

Run 128 independent replications per rho, each with 10000 draws/chain. This is
four times the prior 32-replication diagnostic, a bounded precision compromise:
at a .05 rate the binomial standard error is about .019. It is insufficient
to certify 95% coverage. Seeds are `[72110,rho_index,replication]` using NumPy's
SeedSequence; the same data are paired across methods and laws, never treated
as independent method observations. Save every completed replicate and each
unstarted denominator if the wall allocation stops execution.

At cumulative looks 1000,2000,...,10000, call the actual
`assess_posterior(stage="retained")` with rank/folded R-hat <=1.01, bulk/tail
ESS >=400 and MCSE/SD <=.02. This matches the retained assessment criteria of
the SGQF study, but conditions on stationary initialization and does not test
warm-up. For each estimator record its first passing look without using true
reference error in stopping. Also always evaluate the fixed 10000-draw endpoint
on the same pre-generated path. Common random numbers enable paired descriptive
differences; selection of successful paths or estimators is forbidden.

Compare inherited lugsail (sqrt(n) batches, r=3,c=.5,min_batches=20), plain
batch means at sqrt(n), lugsail with an explicitly even batch length
`2*floor(sqrt(n)/2)`, and the existing autocorrelation method. The even-length
arm is a sensitivity hypothesis motivated by alternating autocovariances,
not an approved repair. Batch remainder handling stays as in the library.
Nonpositive or missing MCSE is unavailable, never zero and never a pass.

At the fixed endpoint and at each method's actual stopping time, report valid
uncertainty, stopping frequency, squared error, MCSE squared, and per-quantity
95% normal-interval miss indicators. The multiplier 1.959964 is the standard
normal .975 quantile, a CLT diagnostic rather than a guaranteed finite-sample
or anytime interval. Report exact pointwise binomial intervals for independent
replicate miss frequencies; also report joint misses and nonstopping counts.
For Gaussian coordinate means, compare estimated variance to the exact
finite-length variance

`Var(mean over four chains) = [1+2 sum_(k=1)^(n-1) (1-k/n) rho^k]/(4*n)`.

This follows by summing covariances of the n observations in each chain and
dividing by four independent chains. At stopped lengths that expression is
only the fixed-length comparator evaluated at the selected n; it is not the
conditional variance after stopping. Do not make that substitution silently.

| Diagnostic | Role | Action |
| --- | --- | --- |
| Exact marginal/reference and covariance identities; recorded source unchanged | Engineering criterion and continuation veto | Repair a mismatch before interpreting outputs |
| Nonpositive/invalid MCSE, failure to stop | Estimator availability/operational outcome | Count all failures; do not drop replications |
| Interval misses and MCSE-to-exact-variance ratios | Calibration evidence with binomial uncertainty | Diagnose where the current policy lacks support |
| Comparative point estimates, stopping lengths and compute | Explanatory | No superiority/default promotion without a later predeclared confirmation |
| Resource budget | Continuation bound | Preserve every completed result and unfinished denominator |

Pre-mortem and skeptical audit: Gaussian negative correlation makes coordinate
means easy but squares persistent; monitor both. Exact margins alone do not
make adjacent observations independent; each whole four-chain replicate is the
statistical unit. A favorable stopped interval can be selection bias; compare
with the fixed endpoint on the same paths and retain nonstoppers. These controls
cannot establish global posterior exploration, warm-up sufficiency, tail
calibration for an arbitrary state-space model, or a repaired default. The
chosen sample count, replication count and even-batch alternative are explicit
hypotheses. The plan passes for bounded diagnosis with these limits.

Command: `env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true
TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=2 OMP_NUM_THREADS=2
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/calibrate_hmc_stopped_mcse.py
--source-root <absolute full-model-source-e68cdf1c4> --output <fresh output>
--replications 128 --seed 72110 --max-wall-seconds 1800`.
The driver records actual command, source hashes, seed list, Python/TF/TFP and
reference-library versions, CPU-only role and elapsed time. This CPU/XLA
reference exception supplies no GPU performance or default-readiness evidence.

Implementation check: ten independent covariance/denominator regression tests
pass. A one-replication-per-condition CPU/XLA smoke (seed 72109) completed all
six law/correlation conditions in 25.331465 seconds in `mcse-smoke-01`. It
exercised the actual posterior assessment, all four estimator policies and
both endpoint reports; it is harness evidence, not coverage evidence. The
first calls compile a bounded set of ten lengths and three batch kernels;
the TensorFlow retracing notice reflects these predeclared shapes. The serious
run keeps the fixed 1800-second allocation and reports partial replication
counts if shared CPU load prevents all 128. The smoke and tests are charged
to the same local reserve.
