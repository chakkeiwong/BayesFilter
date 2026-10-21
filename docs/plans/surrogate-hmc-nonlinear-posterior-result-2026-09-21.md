# M10 independent posterior reference result — 2026-09-21

The independent model posterior reference R passes all six predeclared numerical
checks for the frozen nominal M10 T2 dataset. This completes reference
preparation for testing surrogate HMC with an imperfect canonical DCTR force.
No M10 HMC tuning or retained inference ran. B/R, C/D, D/R and C/R remain
NOT_RUN; a failure of D/R would not by itself invalidate the C/D question.

The target is the actual sine-transition, quadratic-observation law with
independent Normal(0,log(2)^2) log-noise priors. The computation integrates the
same likelihood using both observation roots, then integrates over the
parameter square plus analytical bounds for the omitted prior and latent tails.
It approximates the stated full-prior posterior within the declared empirical
quadrature allowance and analytical omission bounds. It is neither an exact
posterior oracle nor evidence that the DCTR force is an exact gradient.

[Plan](surrogate-hmc-nonlinear-posterior-2026-09-21.md),
[qualified report](artifacts/surrogate-hmc-all-models-20260921-m10-posterior-05/report.json),
[manifest](artifacts/surrogate-hmc-all-models-20260921-m10-posterior-05/run-manifest.json).
The manifest preserves the actual command, frozen data hash/seed, source snapshot,
environment, Git commit, GPU UUID, memory-growth checks, dtype and XLA settings.

## Numerical evidence

Parameter orders40/64/96, state orders20/32, and Gaussian convolution orders64/128
separate three refinement components. Final64/96 relative normalization change
is7.356e-9; state refinement changes8.482e-16 and convolution refinement1.154e-12.
The largest final parameter-function change is8.317e-6, latent-state change
3.331e-16, and convolution change1.276e-11. All thirteen total allowances
fit one tenth of their eventual comparison margins. Allowances include twice
the largest of the last two parameter changes, plus twice the state and
convolution changes, plus analytical omission allowances.

The positive analytical evidence lower bound is3.251838322365109e-6; it is
constructed from a finite positive-prior state/parameter rectangle, not from
the estimated normalization. It yields a posterior omitted-mass bound of
5.6569971092097004e-9. The largest moment omission allowance is7.529e-7.
The numerical evidence estimate is0.049087116943537984. Descriptive posterior
means are(0.004058775,0.106290358) and standard deviations(0.643293277,0.559785238).

| Function | Total reference allowance | Maximum allowed |
|---|---:|---:|
| standardized_log_q | 6.459532205e-06 | 0.01 |
| standardized_log_r | 0.0001031816394 | 0.01 |
| centered_log_q_second | 4.275616479e-05 | 0.01 |
| centered_log_r_second | 0.0003292932941 | 0.01 |
| standardized_cross | 0.0001271547839 | 0.01 |
| log_q_cdf_q10 | 0.0004386670165 | 0.002 |
| log_q_cdf_q50 | 0.0001411320557 | 0.002 |
| log_q_cdf_q90 | 0.0003604356956 | 0.002 |
| log_r_cdf_q10 | 0.0001357844765 | 0.002 |
| log_r_cdf_q50 | 0.001420363094 | 0.002 |
| log_r_cdf_q90 | 0.001742862503 | 0.002 |
| replicated_y2_le_0.5 | 2.206232078e-05 | 0.003 |
| replicated_y2_le_2 | 3.647489641e-05 | 0.003 |

The fine96-node marginal interpolants pass the prospective tolerance screen;
the minimum observed CDF increment is about-2.03e-10 and minimum density about
-1.07e-6. These tiny oscillations are recorded rather than clipped. The screen
and refinement agreement do not prove positive interpolants or rigorous global
quadrature error bounds. The40/64 interpolants remain failed coarse diagnostics.
Predictive functions concern an independent replicate of y2 conditional on x2,
not a future observation y3. Zero likelihoods from FP64 underflow remain counted
in each grid; this value-only route emits no HMC force or endpoint authority.

## Failures, repair and cost

| Attempt | Outcome | CPU seconds | GPU seconds |
|---|---|---:|---:|
| 01 | DEFERRED_BY_COST | 7.492621479 | 0.000000000 |
| 02 | POSTERIOR_REFERENCE_REPAIR_REQUIRED | 66.172461720 | 59.288297284 |
| 03 | GRID_NUMERICAL_FAILURE | 55.556120143 | 51.994043099 |
| 04 | DEFERRED_BY_COST | 15.627339705 | 12.431436516 |
| 05 | QUALIFIED_MODEL_POSTERIOR_REFERENCE | 241.861501052 | 232.203673213 |

Total integration-stage cost is386.710044099 CPU, 355.917450112 GPU and361.560297497
wall seconds, within the reallocated850 CPU/800 GPU/800 wall stage ceilings
and unchanged M10 family1800 CPU/7200 GPU ceilings. GPU1 ran FP64/XLA with
TF32 disabled and verified memory growth; peak allocator use was34,791,168
bytes, below4 GiB. Nine CPU/GPU parity checks passed.

Attempt01 was a CPU cost stop. Attempt02 exposed coarse parameter integration
and interpolation; latent-state refinement and analytical tails already passed.
Attempt03 stopped on a validity guard at likelihood6.014e-320: independent
predictive summation differed by one FP64 subnormal increment. The repaired
guard uses1e-12*max(likelihood,minimum-normal-FP64) for rounding tolerance and
changes no computed output. Two focused regression tests passed in
15.566281225 CPU seconds; the exact GPU failure point also passes. Attempt04
was a conservative cost stop. Reallocating existing family resources permitted
attempt05 without relaxing numerical criteria. Failed evidence is preserved.
Earlier analytical/quadrature regression tests passed; their accounting remains
separate from the integration stage. No stochastic result has been rescored.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Qualify nominal M10 R | All13 reference allowances pass | All6 checks,9 parity comparisons and GPU underflow regression pass | Empirical quadrature error allowance | Master reuse, scoped DCTR controls, inference endpoint qualification and cost forecast | HMC convergence, exact force, DCTR posterior fidelity, stress-scope validity |

| Inference question | Status |
|---|---|
| Hard veto screen | No remaining reference veto at the declared tolerances; coarse candidates remain failed |
| Statistically supported ranking | None; no M10 stochastic comparison ran |
| Descriptive-only differences | Reference moments and timings |
| Default readiness | Not established |
| Next evidence needed | Public scoped HMC tuning, retained comparisons and conditional heuristic checks |

Heuristic dominance is NOT_RUN. The planned direct-reference sampling, prior
independence Metropolis, frozen Gaussian independence proposal and zero-force
HMC adversaries remain to be evaluated centrally and in marginal tails. They
were not used to select these numerical grids or DCTR controls.

Post-run red team: the weakest reference evidence is empirical quadrature error,
particularly the CDF allowance0.001743 against0.002. A more highly refined or
independently integrated posterior outside the allowance would overturn this
qualification. Local mechanics of the old513-node endpoint do not establish
its full-prior accuracy; this new reference must not silently promote that
endpoint. Preparation failures say nothing adverse about the surrogate-HMC
mechanism until the appropriate downstream comparison is performed.

## Master reuse verification

The fresh focused orchestration suite passed57 tests in28.74 seconds
(30.463686 charged CPU seconds including the wrapper). The preceding incomplete
watchdog-limited run remains charged at19.677892667 CPU seconds. The public
`--scope M10 --stage P3-M10-posterior` invocation reused attempt05 after its
numerical source, data, grid and precision checks; no integration or GPU child
ran. It charged9.082378 CPU seconds and correctly returned
INCOMPLETE_IMPLEMENTATION, with B/R, C/D, D/R and C/R all NOT_RUN.
Evidence roots end in `20260921-m10-master-tests-02` and
`20260921-m10-posterior-master-01`. Total master validation/reuse expenditure
is59.223956667 CPU seconds within the revised60-second allocation.

After this execution, global spending is9089.573534514 GPU and10581.035121981
CPU seconds. M10 retains6793.015465505 GPU and1061.166598019 CPU seconds.
The next stage must select M10-specific DCTR controls on calibration and
validation data and assess complete inference affordability. The qualified
reference supplies no authority to promote the original local endpoint.
