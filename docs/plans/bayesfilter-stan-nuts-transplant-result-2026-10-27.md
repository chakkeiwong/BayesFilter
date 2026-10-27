# Stan warm-up transplant: implementation and native comparison

Subsequent code audit: the
[adaptation audit and repairs](bayesfilter-stan-adaptation-code-audit-result-2026-10-27.md)
expanded source parity tests and fixed edge-case/validation defects after this
comparison. The results below retain their original source snapshots and
meaning; they are not timings of the repaired implementation.

The standalone TensorFlow/XLA warm-up transplant is implemented and checked
against installed CmdStan 2.37.0. Both samplers passed the declared posterior
screens on five of six targets for all three seeds. Both failed the centered
funnel. Native Stan had lower measured tuning-inclusive elapsed time on all
five passing targets; reproducing its adaptation did not eliminate TFP's
execution overhead.

The user-facing [NUTS tuning guide](../reference/nuts-tuning.md) explains the
algorithm, metric convention, parameters and command. The
[plan](bayesfilter-stan-nuts-transplant-2026-10-27.md) and
[thorough review](bayesfilter-stan-nuts-transplant-review-2026-10-27.md)
record the evidence contract, assumptions, findings and repairs. Review was
performed by the implementing agent, not an independent reviewer.

## Compared implementations and validity

TF 2.20 / TFP 0.25 uses stock multinomial NUTS inside complete, stable-signature
XLA warm-up and retained-sampling functions. The transplanted adaptation has
per-chain reasonable-epsilon searches, Stan window boundaries, Welford
regularized diagonal/dense covariance, boundary dual-averaging restart and
frozen smoothed epsilon. Constant metric adaptation is represented by affine
coordinates with the checked inverse-metric convention.

The comparator is an actual locally built CmdStan 2.37.0 executable, with
matching targets, metric family, starts, warm-up count, retained count,
acceptance target, depth limit and jitter. Stan is an external native C++
reference, not an XLA path. Its RNG and tree builder differ from TFP's.
The guide identifies the unsupported short-budget fallback and bounded
step-search safeguard; full Stan sampler equivalence is not claimed.

The claimed adaptation recurrences and their computed TF versions agree
with the actual unmodified Stan headers under deterministic tests. Native
and TF target log densities and gradients agree on all six target fixtures
within the predeclared 1e-10/1e-9 relative tolerances. Dense momentum/metric
orientation, physical-position handoff, chain-specific metrics, reproducible
draws and enclosing XLA compilation pass. Twelve focused tests passed,
including the repository guard against reopening other NUTS routes.

Four independent chains ran in parallel, with 1000 warm-up and 2000 retained
draws per chain. Three complete repetitions used seeds 20261027–20261029.
All tuning was repeated for each seed; warm-up and discarded compilation
executions were excluded from posterior diagnostics. Starts came from the
known reference law: these results do not assess arbitrary initialization.
The rotated Gaussian used dense adaptation; the other targets used diagonal
adaptation. Parameters were fixed before evaluation: delta 0.8, depth 10,
initial epsilon 1, zero jitter, buffers 75/25/50 and standard Stan DA settings.

## Results

Times below are median elapsed seconds for one four-chain warm-up plus
retained-sampling run, after compilation. TF time includes all numerical
initialization and metric-boundary searches. Native time is blocking process
wall time, including startup, initialization and CSV writing. TF serialization
adds 0.005–0.023 seconds for the passing targets. Common posterior diagnostics
are excluded from both columns and take roughly 0.18–0.74 seconds after their
first compilation. These boundaries favor TF, so omitted TF output cost does
not explain the observed native advantage.

| Target | TF/XLA seconds | Native Stan seconds | TF / Stan paired time ratio | Posterior screens, TF / Stan |
| --- | ---: | ---: | ---: | --- |
| Gaussian, d32 | 0.656 | 0.121 | 5.41 [5.01, 5.83] | 3/3 / 3/3 |
| Rotated Gaussian, d32, condition 1000 | 1.356 | 0.335 | 4.05 [4.02, 4.08] | 3/3 / 3/3 |
| Product banana, d8 | 0.661 | 0.101 | 6.43 [5.42, 7.62] | 3/3 / 3/3 |
| Centered funnel, d10 | 1.211 | 0.242 | No valid efficiency ranking | 0/3 / 0/3 |
| Noncentered funnel, d10 | 0.554 | 0.075 | 6.98 [5.14, 9.47] | 3/3 / 3/3 |
| Gaussian regression, 512x32 | 0.979 | 0.359 | 2.71 [2.57, 2.86] | 3/3 / 3/3 |

Ratios are geometric means of paired per-seed ratios, not ratios of medians.
Brackets are the predeclared 95% Student-t intervals for log ratios with
two degrees of freedom. All passing-target intervals exclude one under this
model. Three same-machine repetitions provide limited uncertainty evidence:
the log-normal assumption, shared-host load and unadjusted multiple target
comparisons preclude a broad superiority claim. This is a fixed-budget elapsed
comparison conditional on diagnostic screens, not optimal time to precision.

Among passing cases, maximum R-hat across seeds/coordinates is at most 1.00433.
The smallest bulk ESS is 1351.7 and smallest tail ESS is 1152.3; maximum
standardized MCSE is 0.03224. Known reference moments pass the declared screen.
Native energy flags are reporting-only under current policy: the banana had
up to 0.001/0.003 flagged fractions for TF/Stan, respectively. These rates were
not used to reject or retune a candidate.

The centered funnel fails more than a speed screen. Maximum R-hat reaches
1.2095 in TF and 1.2709 in Stan; minimum tail ESS falls to 17.1 and 6.3;
standardized MCSE reaches approximately 0.24. Neither retained estimate is
qualified. Its noncentered control passes for both engines. This supports
investigating parameterization and longer targeted adaptation; it does not
justify abandoning NUTS or claiming a valid funnel speed advantage.

## Cost and execution interpretation

The full CPU/GPU calibration used two discarded targets, two seeds and
200/256 warm-up/retained allocations. Sum of target median times:

| Layout | Sum of calibration medians, seconds |
| --- | ---: |
| CPU scalar, four workers, intra-op1 | 0.379 |
| CPU batched, four intra-op threads | 0.833 |
| GPU batched | 2.780 |
| GPU scalar | 8.042 |

CPU scalar was the pre-evaluation choice for this benchmark. The CPU is an
AMD EPYC 7742; GPU calibration used GPU0, an RTX 4080 SUPER. GPU memory growth
was verified before initialization; recorded allocator peaks were about
1.6–2.1 MB for completed calibration workers. The CPU choice is local to these
small targets and four chains; it does not change the repository GPU default
or establish the optimal device for larger matrix targets/batch counts.

For the Gaussian and regression, both engines averaged essentially seven
retained leapfrogs per transition. Nevertheless, TF retained sampling was
roughly 0.325 versus native 0.061 seconds for the Gaussian, and 0.496 versus
native 0.208 seconds for regression (native values are maximum printed
per-chain sampling times). Extra leapfrogs cannot explain those gaps.
Scalar chains also remove the earlier cross-chain masking explanation.
Remaining candidates are compiled tree bookkeeping/RNG/control costs,
target and gradient lowering, and host/executor scheduling. Their individual
contributions have not been profiled in this campaign.

On the noncentered control, TF additionally used median 11.95 leapfrogs versus
Stan 6.51. Matching adaptation does not force matching trajectory lengths:
finite warm-up, random streams and transition details differ. On the rotated
Gaussian, both use about seven retained leapfrogs and pass the posterior
screen after dense adaptation, unlike the earlier identity-metric HMC pilot.
That does not establish a comparison with the public tuned fixed-HMC procedure.

Each target's two TF first calls took 4.50–5.48 seconds in total, including
compilation and discarded executions. They are not pure compile-time estimates.
Framework startup is recorded separately. The isolated CmdStan library/tool
build took approximately 103 seconds by install-manifest/build-log timestamps;
native model compilation and shared library build are one-time installation
costs, not charged as if repeated for each seed. Both compiled-reuse and
cold-start cost components are preserved; no cold-start winner is asserted.

## Evidence, repairs and reproducibility

Evidence root:
docs/benchmarks/artifacts/stan-transplant-20261027/.
The authoritative corrected run is comparison-03/manifest.json; compact
results are comparison-summary.json. layout-decision.json records the
pre-evaluation execution choice. campaign-manifest.json records all attempts,
failures and accumulated worker wall time. Every final run keeps native CSV,
TF tensors, warm-up telemetry, exact commands, seeds, source snapshots, hashes,
HLO and environment/device settings. Native caches are not exported; finite
native samples and successful execution were checked rather than claiming
direct verification of Stan's internal gradient cache.

The release tarball is retained at
.localresources/cmdstan/cmdstan-2.37.0.tar.gz with SHA-256
635e2e1cf9c4774c9019001325354d3610bae2adc4ed5af2fed87872dfc671fc.
The installed root is .localresources/cmdstan/cmdstan-2.37.0; benchmark binary
docs/benchmarks/stan_targets. Actual release adaptation sources and executable
hashes are archived in install-01/source-manifest.json. Shared Python
environments were not modified.

There were nine benchmark-worker attempts consuming 488.98 worker seconds,
including failures and the timing rerun, plus three focused test launches.
The complete benchmark campaign finished within the declared build and
post-build wall caps. The corrected six-target worker took 120.66 seconds and
peaked at approximately1.79 GiB host RSS. The machine was shared, with load
average about 42 across 256 logical CPUs at the recorded hardware check.

Two infrastructure failures and one timing flaw were repaired within budget:
CPU placement for GPU tensor serialization; single-point native log-prob
inputs to avoid CmdStan's multi-point reshape mismatch; and blocking waitpid
to remove Python timeout-poll timing quantization. All earlier artifacts are
preserved. timing-repair-validation.json verifies identical TF retained
tensors and native numeric CSV rows for every target/seed across the final
timing repair. No settings were selected from those evaluation results.

Validation command:

    CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
      BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
      /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest \
      tests/test_stan_nuts_warmup.py tests/test_stan_native_targets.py \
      tests/test_modern_hmc_nuts_comparison.py -q

Authoritative run:

    timeout 600 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
      docs/benchmarks/benchmark_stan_transplant.py --mode compare \
      --device cpu --layout scalar --warmup 1000 --draws 2000 \
      --depth 10 --seeds 20261027,20261028,20261029 \
      --output docs/benchmarks/artifacts/stan-transplant-20261027/comparison-03

## Decisions and remaining uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep standalone Stan-style adaptation | Native-header arithmetic and XLA checks pass | No target/retained-state failure on the five passing targets | Complete NUTS transition is still TFP | Use this route for future diagnostic comparisons | Full Stan sampler identity or public default readiness |
| Record native elapsed advantage on five targets | Paired intervals exclude one under stated model | All three posterior screens pass | Only three seeds, favorable starts, shared hardware, timing boundaries | Profile matched fixed kernels and complete time-to-precision protocol | Universal Stan superiority |
| Reject centered-funnel estimates | Precision/convergence criteria fail | Both engines fail every seed | Neck exploration and finite adaptation | Targeted parameterization/adaptation study | NUTS research direction rejected |
| Close bounded campaign | Planned comparison and guide delivered | No remaining execution blocker | Public fixed-HMC comparison not performed here | Integrate the public HMC candidate-set tuner in a separate comparison | Old grid results are canonical HMC evidence |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Native target parity and implementation checks pass; centered funnel fails posterior qualification for both. |
| Statistically supported ranking | Lower native fixed-budget elapsed time on five passing targets under the declared three-repetition log-ratio model; no general efficiency ranking. |
| Descriptive-only differences | Individual ESS, tail ESS, divergence rates, epsilon/metric variability, leapfrogs and CPU/GPU layout timings. |
| Default readiness | Not established; standalone implementation only. |
| Next evidence needed | Larger independent repetitions, less favorable starts, more expensive models, controlled profiles, and a full tuning-inclusive time-to-precision comparison with public fixed HMC. |

Post-run red team: favorable exact-reference starts and fixed terminal checks
are the weakest validity evidence. A slower method could reach a chosen
precision earlier despite a higher fixed-draw cost; this experiment does not
test that. More expensive gradients may change the runtime gap and device
choice. A matched-profile result attributing costs differently would overturn
the tentative overhead explanation, while recurrence/target failures would
invalidate implementation conclusions rather than merely reject a candidate.
