# Prospective calibration of binary-function diagnostics

The question is whether a narrowly repaired uncertainty diagnostic can support
the surrogate-HMC tests. The candidate remains the joint batch-randomization
guard, with one explicit treatment of an uninformative folded statistic.
The previous candidate failed its calibration; its
[result](surrogate-hmc-mcse-randomization-result-2026-09-20.md) remains unchanged.
This stage has no authority to rescore closed LG1/C01 chains or promote a
general MCMC diagnostic.

## Evidence contract and repair

Comparator: the current equivalence implementation, including its fixed lag
screen and full modern R-hat, evaluated on the same fresh simulations as the
candidate. Preserve batch-MCSE estimation, sensitivity threshold 2, parameter
R-hat 1.01, bulk/tail ESS 400, Student-t36 intervals, 15 functions, two looks,
and all finite/positive-variance checks. Replace only the lag rule and the
specific binary folded degeneracy described below. The randomization rule,
1,023 permutations and per-look threshold 0.01 remain as previously derived.

For an observable taking exactly zero and one, with both values present in
equal pooled counts, the midpoint median is 0.5. Every folded value is then
0.5. Its within- and between-chain variances are both zero, so folded R-hat is
undefined. The folded component contains no dispersion information in this
case; the rank-normalized bulk component still measures differences in event
frequencies. The proposed diagnostic policy preserves the raw folded result,
records an explicit uninformative-fold flag, and uses max(1, bulk R-hat) only when the
observable is exactly binary, balanced, and bulk R-hat is finite. It does not
replace undefined diagnostics generally. Constant observables, constant
chains with different values, nonbinary observables and nonfinite inputs keep
their vetoes. This is a scoped diagnostic policy, not a claim that 0/0 equals 1.
Production R-hat and equivalence code remain unchanged during this experiment.

The initial focused test found that roundoff in the rank transform and its
variance can produce an apparently finite folded R-hat for this constant fold.
Detect the degeneracy from the exact binary frequencies, not a NaN test.
Also reject nonfinite or constant raw observables explicitly, independently of
their numerically computed rank variance. The first batch passed ten checks
and failed one overly narrow NaN assertion; it consumed 16.275051 CPU seconds.
The corrected assertion checks the mathematical constant-fold condition and
the rejection behavior, without prescribing a floating-point NaN.

## Fresh calibration and decision

Use the same normal/AR laws, 15 C01 functions and 2,048 replication counts:
512 IID; 256 each for rho 0.5, -0.5, 0.9, 0.99, chain offsets and drift.
Fresh data keys are (20260923, base + replication), with bases 500000 through
560000 in that order. Randomization uses (20260924, base + 2*replication + look).
No generated data from the first calibration are reused as validation.
The previous failure motivated the repair and the longer declared scope below.

Use nested 1,000/2,000 draws per chain except rho 0.9, which uses 4,000/8,000.
For rho 0.9 the exact AR mean variance gives effective size approximately
4*n/19; these longer looks provide about 842/1,684 effective draws. This puts
the experiment above the unchanged ESS 400 requirement, so admitted coverage
can be assessed. It does not authorize longer HMC chains or additional looks
in any closed inference attempt. For this longer diagnostic only, use the
same 40/20/10-batch formula at the corresponding batch lengths, with parity
tests against the actual implementation at 1,000 and 2,000 draws.

Primary criterion: the Wilson 95% upper bound for any-look full-diagnostic
false veto in IID replications is at most 5%. In each of IID, rho 0.5, -0.5 and
0.9, both unconditional and admitted-conditional interval-failure upper bounds
must be at most 5%, with at least 64 admitted replications. Intervals cover all
15 functions and both looks simultaneously. Bounds on replication rates are
per regime, not simultaneous across regimes. For rho 0.99, offsets and drift,
the Wilson lower bound for veto at both looks must be at least 95%. As before,
if at least 64 rho 0.99 replications are admitted, their coverage must also
pass. Otherwise report that coverage as unassessed, with rejection protection
decided separately. Do not select the candidate on interval width or timing.

Constructed diagnostic adversaries: naive IID standard errors ignore serial
dependence; the existing fixed-lag rule is the runtime comparator; exact AR
finite-length mean variance is the oracle. Report these separately by regime.
Acceptance of the candidate requires all calibration criteria, not merely
improvement relative to the old rule. A failure rejects this candidate and
prompts a documented repair; corrupted data/source drift, a failed invariant,
or exhausted budget stops execution. Gaussian calibration cannot certify
heavy tails, multimodality, another function family, or all-model inference.

## Defaults, audit and execution

The binary exception follows the derivation above and is tested on balanced,
unbalanced, stuck-chain and constant inputs, including compiled calls. It is
not a numerical perturbation or clipping of a diagnostic. All other thresholds
are inherited comparison hypotheses whose false veto, coverage and dependence
protection are measured here. The longer AR scope follows its known effective
sample size; it is not a threshold relaxation. The batch count and permutation
count retain their previous rationale and limitations. Normal tails remain
the main limitation; no runtime default is changed even if this stage passes.

Skeptical audit: identical laws and draws are used for comparator and candidate.
The policy change is visible, the short correlated scope is replaced
prospectively, and the same hard criteria apply. A false pass could result from
unrecorded undefined diagnostics or a changed estimator; explicit field checks
and estimator parity tests address these risks. A failed coverage screen with
few admitted samples remains inconclusive, not proof of undercoverage.

Allocate at most 900 CPU-worker seconds and 900 wall seconds across an initial
run and one infrastructure retry, plus 60 CPU seconds for focused tests, from
the existing shared repair reserve. This is within the unchanged 1,800-second
reserve and whole-campaign 54,000 CPU / 226,800 GPU-second caps. Previous MCSE
work used about 305 CPU seconds. No GPU work is needed. After eight replications
in each regime, forecast remaining work with a 1.5 margin and account for the
longer scope; stop before exceeding the remaining allocation. Watchdogs bound
actual execution. Source drift, missing diagnostics or nonfinite generated data
stop the experiment. Preserve every failed attempt in a fresh output directory.

Run the tftwogpu interpreter on
`scripts/calibrate_surrogate_hmc_mcse_randomization.py --protocol binary-fold-v2 --output docs/plans/artifacts/surrogate-hmc-all-models-20260920-mcse-binary-calibration-01`
with CUDA_VISIBLE_DEVICES=-1, TF threads 2/1 and GPU intentionally hidden.
Use stable TensorFlow signatures and XLA for hot generation/batch/randomization
kernels. Tied-rank diagnostics retain their recorded non-XLA exception.
Store per-replication JSONL, manifest, source snapshots, accounting, result note
and active checkpoint. The candidate code stays in the diagnostic package.
