# Antithetic precision diagnosis and next repair sequence

**Status: completed 18 October 2026.** The 24-case source-parity diagnosis,
four preserved-case regression tests, focused documentation checks and fresh
nine-page LaTeX build passed. The diagnosed negative denominators are inherited
from the installed TFP positive-pair estimator; no numerical runtime repair or
default change is made. Terminal evidence is in
`bayesfilter-mcse-antithetic-ess-result-2026-10-18.md`.

18 October 2026. This phase continues the October 7 modern-HMC master after
the [stationary stress result](bayesfilter-mcse-stationary-stress-result-2026-10-18.md).
It leaves the running SIR cohort, all numerical defaults and tuning admission
unchanged. No NUTS route is involved.

## Question and evidence contract

Is the unavailable autocorrelation MCSE on strongly antithetic Gaussian paths
caused by a BayesFilter translation error or inherited from the installed
TFP positive-pair estimator? The actual baseline is TFP 0.25.0
`mcmc.effective_sample_size` with `filter_beyond_positive_pairs=True` and
`cross_chain_dims=1`. Compare the shared BayesFilter `_cross_chain_ess`,
an independent NumPy diagnostic implementation of the inspected equations,
and the public `mean_precision(method='autocorrelation')` availability flag.
The existing, separately labeled Stan-style monotone ESS is an explanatory
alternative only. Numerical positivity does not promote it for MCSE.

Reconstruct preserved replications 0 and 1 in each correlation group, both
Gaussian/lognormal margins, at n=1000 and 10000. This is 24 paired diagnostic
cases, not new statistical evidence. Replication 1 includes a confirmed
unavailable antithetic coordinate. For each, retain the pair-sum truncation
index, weighted denominator, ESS values, raw mean-variance estimate and
public availability. Check the independent autocovariance at the first 16
lags by direct dot products. Record raw coordinates, square and event.

Pass means source/seed identity, near-floating-point parity for every finite
reference value, and unavailable MCSE when the underlying variance estimate
is nonpositive/nonfinite. Inspect a sign mismatch even when values are near
zero; do not hide it in a tolerance. Parity tolerance 1e-8 relative/1e-9
absolute is a diagnostic hypothesis allowing FFT cancellation; observed
errors and denominator values must accompany any pass. Direct covariance
tolerance is 1e-10 relative/1e-12 absolute, a float64 diagnostic choice.
No finite-sample coverage, accuracy ranking, convergence or default claim
follows. A mismatch or source change requires localization; an inherited
failure motivates a separate estimator design rather than rewriting history.

## Source and assumptions

Use the unchanged reviewed precision snapshot from the stress phase. Inspect
the installed TFP `python/mcmc/diagnostic.py` implementation, especially
lines 203–336. It forms a lag-weighted sum after positive-pair truncation and
returns Cn/(-1+2 sum(weighted pairs)), without a positive denominator floor.
BayesFilter's `hmc_diagnostic_math.py:77–140` intentionally implements that
same estimator with real FFTs. `hmc_precision.py` turns a nonpositive variance
into unavailable MCSE. `hmc_ess.py` has a separately identified monotone
estimator with its own bound. Do not conflate these algorithms or silently
replace one with the other. Capture installed and repository source hashes.

The preserved seeds are reproduction choices, not new independent confirmation.
Both laws sharing a stream remain paired. CPU execution deliberately hides
GPUs; NumPy/SciPy are independent diagnostic references only. The inherited
two-thread environment, float64 and frozen source prevent a backend change
from masquerading as a repair. The actual public call-chain test covers
`mean_precision` to `_cross_chain_ess`; tuning remains outside this path.

## Budget and execution

Reserve 300 worker-seconds from the reconciled 134872.4638433829-second SGQF
balance, leaving 134572.4638433829 unallocated during this phase. This is
existing compute, not a new grant or an addition to SIR caps. The diagnostic
has a 180-second outer limit; reserve 120 seconds for focused regression and
closeout. These are convenience cost bounds, not speed estimates. One
diagnostic attempt; a repair can use only the remaining original allocation.

Fresh output root:
`artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/antithetic-ess-20261018-01/`.
Store the exact command, env, source hashes, preserved seeds, complete case
inventory, logs, result and measured inclusive costs. Python is
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. Set `CUDA_VISIBLE_DEVICES=-1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, TF inter/intra-op and OMP threads to 2,
and disable bytecode writes. TFP comparison graphs may use non-XLA CPU
reference execution; the existing monotone routine retains its compiled path.

## Skeptical pre-execution review

Pass for diagnosis only. The official installed implementation is the actual
parity target, and its stated formula is inspected rather than inferred from
a function name. A negative finite denominator is evidence about the estimator,
not an environment or HMC failure. Independent FFT plus direct lag sums can
distinguish translation error from truncation behavior. Neither the old
positive-pair implementation nor the monotone alternative is assumed correct
as a coverage procedure. Bounds, complete cases, source drift checks and
nonclaims are explicit. No material promotion decision occurs in this phase.

## Subsequent repair, refreshed after this diagnosis

1. Preserve the diagnosed counterexamples as regression tests of arithmetic
   and honest unavailable status. If translation is wrong, repair the shared
   implementation and audit every caller. If parity holds, label the
   estimator limitation explicitly instead of calling it a translation fix.
2. Design optional dependence-aware variance and batch-length policies.
   Compare exact finite-batch expectations, availability and uncertainty of
   variance estimates. Include independent, persistent, antithetic, tied-event
   and skewed-square cases. Do not pick a policy using failed HMC members.
3. Calibrate the full interval and stopping procedure on fresh seeds with
   predeclared denominators and multiplicity control. Distinguish variance
   bias from non-normal sampling distributions and optional stopping. An
   unavailable estimate must not become an automatic pass.
4. Evaluate viable optional policies on preserved full-model HMC quantities
   and then fresh matched nonlinear/state-space runs. Retuning membership,
   posterior assessment and research-method ranking remain separate.
5. Update the official LaTeX tuning discussion and operational API reference
   with checked limitations or new optional policies; no competing guide.

Steps 2–5 require a concrete candidate design, reviewed numerical provenance,
fresh artifact root and bounded allocation after the diagnostic result. They
are not authority for an unpriced sweep or automatic default replacement.
