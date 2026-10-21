# M13: ordinary FP32 is viable in the measured cases

Ordinary FP32, with TF32 matrix operations disabled, completed all 144 filter
evaluations. All 18 case-by-particle-count comparisons passed the predeclared
arithmetic-materiality screen. Its observed median runtime was 3.74 to 7.53
times faster than FP64. The previous FP64 particle ladder matched its earlier
baseline; it did not establish a need for FP64 in this filter.

TF32-enabled FP32 is a separate result: it failed the existing covariance
restoration guard at the first observation for both preflight parameter points
at every particle count. No complete-filter timing or score-materiality claim
is available for that mode. This is a guard failure, not evidence that ordinary
FP32 is unusable or that TF32 necessarily causes materially wrong posteriors.

## What was measured

The current covariance-guided defensive-mixture filter was evaluated on the
wrapped range-bearing M13 model: T20, four state dimensions, two observation
dimensions and two parameter dimensions. Three datasets, each at truth and
the existing difficult parameter point, used eight fixed particle designs.
Each precision pair consumed identical FP32-representable inputs; source,
data, controls and serialized-input hashes were checked. Parameters and
mixture weights were also rounded identically. Compared with the earlier
FP64 runs, this rounding changed a score by at most 6.17e-7 and a log-likelihood
by at most 3.19e-7.

Each N retained its own previously selected controls and mixture weights.
These are frozen FP64-tuned comparators, not new FP32 tuning artifacts. N1008
and N2016 use 16 flow and 80 reset steps; N4032 uses 8 and 40. Consequently,
cross-N changes concern those tuned configurations, rather than particle count
alone. Seeds with the same labels across N do not provide nested particle clouds.

The claim is about sensitivity of the same finite algorithm to arithmetic
precision. FP64 is its numerical comparator, not an exact filtering oracle.
The analytical score path was used in both precisions. This experiment did
not run surrogate HMC or assess posterior accuracy.

## Runtime

Seconds below are medians of 48 complete, synchronized evaluations after a
cold call and two warm-up calls. Each evaluation computes one log-likelihood
and both analytical score components. Data preparation, compilation, tuning
and serialization are excluded. All workers used the same RTX 4080 SUPER,
TensorFlow/XLA, one graph trace per worker and verified memory growth.

| Particles | FP64 seconds | FP32, TF32 off, seconds | Observed speed ratio | Valid full evaluations per precision |
|---:|---:|---:|---:|---:|
| 1008 | 2.106 | 0.563 | 3.74x | 48/48 |
| 2016 | 6.257 | 0.852 | 7.34x | 48/48 |
| 4032 | 16.206 | 2.152 | 7.53x | 48/48 |

Thus the measured N4032 FP32 evaluation costs about as much as N1008 FP64.
The first calls, including compilation and execution, took FP64/FP32
10.79/7.96, 14.44/8.25 and 24.97/10.22 seconds respectively. Timing ranges and
worker allocator peaks are in the manifests/report. These are descriptive
measurements from one serial campaign, not an independently replicated timing
study. Another process used the other GPU; host contention was not eliminated.

## Precision error relative to particle-count changes

For each case, define the score discrepancy as the RMS over eight designs of
the Euclidean difference between FP32 and FP64 scores. Its denominator is the
RMS difference between the N1008 and N2016 FP64 scores in that same case.
The log-likelihood calculation uses scalar differences in the same way. This
answers whether arithmetic changes are small compared with changing the
particle configuration; it does not establish which configuration is more
accurate. The predeclared screen requires both ratios below 10% in every case.

| Particles | Score RMS discrepancy | Score discrepancy / N1008-to-N2016 change | Log-likelihood RMS discrepancy | Log-likelihood discrepancy / N1008-to-N2016 change |
|---:|---:|---:|---:|---:|
| 1008 | 0.00319–0.12479 | 0.270–2.179% | 0.000373–0.004088 | 0.101–0.855% |
| 2016 | 0.01242–0.15429 | 0.663–5.381% | 0.001483–0.014122 | 0.313–1.131% |
| 4032 | 0.00721–0.13763 | 0.610–2.798% | 0.002150–0.020090 | 0.585–1.697% |

Ranges span the six cases, not confidence intervals. Relative to each N's
own design variation, score discrepancy is at most 5.46% of RMS score spread,
and log-likelihood discrepancy is at most 3.64% of log-likelihood SD. The
largest individual score and log-likelihood differences across all paired
evaluations are 0.36083 and 0.054893 respectively. RMS screening should not
be mistaken for a bound on every individual difference or on an HMC trajectory.

There is another reasonable meaning of the owner's word "gap": the earlier
change in *mean score error against the saved reference*. The following
explanatory calculation answers that interpretation too. Positive numbers
mean N2016 has lower mean error; negative numbers mean it has higher error.

| Dataset and point | Mean error reduction, FP64 | Mean error reduction, FP32 | Precision changes the observed gap by |
|---|---:|---:|---:|
| 1704 truth | 0.646313 | 0.647492 | 0.18% |
| 1704 difficult | 1.741975 | 1.659074 | 4.76% |
| 18020 truth | 0.599967 | 0.603529 | 0.59% |
| 18020 difficult | -0.201261 | -0.180299 | 10.42% |
| 18021 truth | 0.324113 | 0.340657 | 5.10% |
| 18021 difficult | 0.168169 | 0.157660 | 6.25% |

The sign is unchanged in all six cases. The percentage can be unstable when
the observed accuracy gap is small. The saved particle-filter score reference
also has Monte Carlo and genealogy uncertainty, so these are descriptive
changes, not new evidence of a statistically supported particle-count ranking.
For any fixed reference r, the reverse triangle inequality gives
`abs(norm(s32-r)-norm(s64-r)) <= norm(s32-s64)`; this connects arithmetic
discrepancy to possible changes in measured reference error without treating
that reference as exact.

## Why TF32 stopped

At the first observation, the six distinct TF32 preflight cases have reset
covariance relative errors between 0.00037237 and 0.00059915, or
0.0372–0.0599%. The existing FP32 guard is `100 * eps32 * d = 0.000047684`,
or 0.00477%. These errors are 7.8–12.6 times the guard. Both directional
evaluations stop at step one. The guard and algorithm were left unchanged;
failed prefixes were not reported as fast full-filter runs. Further TF32 work
would need its own numerical evaluation, for example testing which operations
require ordinary FP32. This completed diagnostic does not authorize or perform
that algorithm change.

## Decisions and remaining uncertainty

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Ordinary FP32 is viable in these filter cases | All 18 arithmetic screens pass | 144/144 full FP32 evaluations valid | Broader parameter region, retuning and long trajectories | Use this evidence to nominate FP32 without TF32 for a future scope-tuned HMC campaign | HMC/posterior/default readiness |
| TF32 enabled fails this implementation's numerical screen | No complete comparison | First-step covariance guard fails at all N | Whether a precision repair would preserve material accuracy | Separate bounded precision repair if pursued | Fundamental rejection of TF32 or FP32 |
| Precision/timing request completed | All nine workers terminal; paired evidence complete | Audit passes | Single-machine descriptive timings | Report the results; no remaining phase in this diagnostic | Permission to launch another HMC campaign |

| Inference status | Result |
|---|---|
| Hard veto screen | FP64 and ordinary FP32 pass all full evaluations; TF32 fails preflight |
| Statistically supported ranking | None for particle-count accuracy or posterior quality |
| Descriptive-only differences | Runtime, finite-design RMS/maxima, reference-error changes |
| Default-readiness | No new default promotion; frozen FP64 controls are not FP32 tuning admission |
| Next evidence needed | Scope-specific FP32 tuning and actual HMC diagnostics for a later posterior claim |

Engineering checks passed: source/data/control equality, exact input hashes,
seed completeness, all-step validity, repeatability, one trace, growth policy,
independent recomputation of ratios and correct exclusion of early-stop timing.
The terminal review is a standard-library evidence audit, not an extra sampling
or convergence-validation phase. No numerical implementation or guard was changed.

The strongest alternative explanation for a favorable result is that these six
fixed cases miss a more sensitive parameter region. A failure or a material
precision discrepancy in new cases would overturn broad extrapolation. Frozen
FP64 controls and the small number of datasets are the weakest parts of that
extrapolation. The earlier heuristic-comparison weaknesses remain: this
arithmetic diagnostic does not show that either precision improves the filter
against prior-only, Gaussian-guide or bootstrap alternatives.

## Reproduction and evidence

Plan: `docs/plans/ledh-range-bearing-precision-ladder-2026-10-10.md`.
Entry point: `scripts/range_bearing_precision_diagnostic.py`.
Git HEAD: `10884528ca8b1a7463895256d30b9655d69c0955`; dirty numerical sources
and the complete source dependency set are hashed in every worker manifest.
Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.
GPU UUID: `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`.
The exact launch is recorded in `execution.json`; worker commands, data seeds,
design seeds, data/control/source hashes, memory policy and wall times are in
each manifest. The nine-worker campaign completed in 1574.37 seconds (26.24
minutes) within the 5400-second budget, with no infrastructure retry.

Output root: `docs/plans/artifacts/ledh-range-bearing-precision-20261010-01`.
`report.json` preserves all conditional metrics; `materiality-detail.json`
preserves reference-error gap calculations; `terminal-review.json` records
the passed evidence audit. All raw measurement/preflight JSONL and worker
logs remain under that root. Previous ladder artifacts were preserved.
