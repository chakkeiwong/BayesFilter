# q20 factor reuse: current-source result

The conservative safe-factor cache passed the complete current-source GPU/XLA
comparison. It reduced warm HMC execution time by about 35–37% in the paired
measurements while retaining the declared value, analytic-score, trajectory and
health tolerances. Fresh repository GPU/XLA qualification passed at both
temperatures. The terminal master status is `MASTER_CACHE_GRAPH_QUALIFIED`;
complete preparation and posterior validity remain open.

Plan: [performance continuation](bayesfilter-ssl-lstm-q20-performance-continuation-2026-09-19.md).
Profile source: `/tmp/BayesFilter-q20-performance-20260919-r3`.
Profile result:
`artifacts/ssl-lstm-q20-performance-2026-09-19/retry-02/campaign/attempts/00000-factor-profile/worker/result.json`.
The exact command, source inventory, environment, input-bank hashes, seeds,
memory policy, target identities, per-call timing and checks are in that result
and its campaign ledger. The machine selected GPU 2, an RTX 4080 SUPER, with
memory growth verified before initialization. The data and target remain the
fixed q20/T30, four-parameter, float64 UKF-approximate posterior.

## What changed and what passed

The optional backend is `tensorflow_eigh_strict_factor_cached`. It reuses the
eigenbasis of the same safe covariance used to construct the principal square
root when solving the derivative's Sylvester equation. It removes one of three
refined eigensolves. It does not reuse information across parameter values or
observations, change the sigma-point geometry, reduce the eight-sweep cap,
relax residual checks, change precision or omit target-status checks.

Both backends passed the saved Phase 9B preflight and endpoint references.
Independent fixtures covered eigenpairs, repeated/near-zero spectra, indefinite
and nonfinite inputs, roundoff repair, and failure to converge. Complete target
calls passed at both positive betas and B=1/B=4. The largest observed analytic
score difference on those banks was 7.11e-15. All twelve paired HMC comparisons
passed, including proposed states and momenta, rejection decisions and health
telemetry; the largest floating trace difference was 5.68e-13. These are checks
at the declared inputs, not a proof of uniform error over the posterior.

The sixteen inspected target/HMC graphs each traced once, required XLA, and
contained no Python callbacks or HostCompute operations. Graph inventories
show three eigensolver nodes for strict and two for the cache. Repeated matrix
work inside the compiled graph is a real cost; graph operation counts alone do
not measure its share of device time.

## Measured costs

Each HMC call used L=25, epsilon .011048543456039808, two transitions and the
existing prior-scale coordinates. The four-chain calls used independent momenta
at the same scalar bootstrap origin. Two warm calls followed a separate cold
call; arm order alternated. Every timing synchronized samples and all trace
fields. The table reports the mean of those two warm calls, in seconds.

| Beta | Chains in one batch | Strict | Safe-factor cache | Observed time reduction |
| --- | ---: | ---: | ---: | ---: |
| .5 | 1 | 67.182 | 43.320 | 35.52% |
| .5 | 4 | 82.287 | 53.892 | 34.51% |
| 1 | 1 | 65.532 | 41.393 | 36.84% |
| 1 | 4 | 77.492 | 50.262 | 35.14% |

Warm complete-target calls took 1.27–1.33 seconds for strict scalar rows and
.79–.85 seconds for cached scalar rows. Four-row calls took 1.52–1.63 seconds
and .97–1.08 seconds respectively. The isolated initial covariance factor with
four identity derivative directions took approximately .027 and .019 seconds.
That initial-matrix probe cannot attribute a percentage of full-filter cost.
All timings are descriptive observations on this device and these inputs.

The unchanged 1,000-transition mass minimum extrapolates as follows, using the
maximum warm scalar cost and the inherited factor-two scheduling margin:

| Beta | Cached seconds/transition | Unmargined minimum | With scheduling margin |
| --- | ---: | ---: | ---: |
| .5 | 21.8660 | 6.074 hours | 12.148 hours |
| 1 | 20.8316 | 5.786 hours | 11.573 hours |

The combined scheduling forecast is 85,395.333131 seconds, or 23.721 hours,
before subsequent tuning, training, reference and posterior work. This exceeds
the approximately 22.36 campaign hours available before fresh qualification.
The unmargined sum is 11.86 hours; the larger figure is a scheduling reservation,
not a measured runtime or statistical confidence bound. Actual changing metrics,
epsilon searches and compilation may change the cost. Complete mass preparation
has not been timed, and a smaller smoke count would not answer that question.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep conservative cache as an optional candidate | Full declared parity screen and fresh GPU/XLA qualification passed | No numerical or resource veto in the corrected run | Unchecked parameter regions and long adaptation paths | Plan affordable full preparation under the new backend identity | Default readiness, whitening or posterior convergence |
| Preserve full mass requirement | Both cached forecasts still require substantial time | No target invalidity established; complete preparation remains unfunded under the scheduling forecast | Actual mass-stage and later campaign costs | Further runtime repair and complete cost planning before a long launch | Research direction rejected or remaining allowance exhausted |
| Reject the earlier harness disposition | Equal +infinity eigen-gap sentinels were incorrectly compared with a finite near-equality assertion | Harness defect, not candidate evidence | None about the identified scalar-observation sentinel | Preserve the failed attempt and use the corrected named-field comparison | Permission to accept nonfinite targets, gradients or HMC states |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Current engineering/reference/paired-transition checks pass; earlier import and comparator failures are preserved infrastructure evidence |
| Statistically supported ranking | No stochastic-method or posterior-efficiency ranking |
| Descriptive-only differences | Warm runtime reductions, observed absolute errors, primitive timing and extrapolated preparation cost |
| Default-readiness | Strict remains the template default; the cache is an explicit optional protocol with a distinct identity; full tuning and posterior gates remain open |
| Next evidence needed | Complete affordable mass/tuning work, target-specific training assessment and declared posterior/reference checks |

The corrected profile's source passed 18 focused CPU mechanics checks; the
qualification overlay passed 15 focused checks. These suites overlap. No trained
map, kernel-tuning receipt or posterior sample is promoted by these tests.

Post-run red-team: the comparison uses a fixed origin and short trajectories,
and only two warm timings per cell. Long trajectories could encounter different
conditioning, and mass changes could alter the forecast. The strongest numerical
counterevidence would be a reproducible score or status discrepancy at a valid
new state, or failure of fresh qualification. The weakest performance evidence
is extrapolation from fixed-metric calls to complete adaptation. The current
result supports the local repeated-work repair and leaves those checks open.

## Terminal qualification, accounting and continuation

The master now supports `profile` and `cache-qualify`. The latter requires the
completed parity result, unchanged numerical dependencies, a separate candidate
protocol and fresh qualification; it cannot transfer strict-route tuning. Both
qualification workers selected host GPU 2, verified memory growth, completed the
real four-chain public runner with one trace, and returned healthy GPU endpoints.
Their supervised times were 145.540433 and 140.040791 seconds. The final source is
`/tmp/BayesFilter-q20-cache-qualified-20260919-r1`.

The concrete candidate protocol and qualification are:

- `artifacts/ssl-lstm-q20-performance-2026-09-19/qualification/campaign/candidate-protocol.json`
- `artifacts/ssl-lstm-q20-performance-2026-09-19/qualification/campaign/qualification.json`

The current authoritative campaign ledger is
`artifacts/ssl-lstm-q20-performance-2026-09-19/qualification/campaign/campaign.json`,
SHA-256 `c8709ef624eb83d7db012a84f5a5204fb47a974a1ce5232eba5c7771a265b271`.
All attempts are settled, every recorded stage-artifact hash was checked, and
both numerical services are inactive with a successful exit. Terminal
verification is saved in `artifacts/ssl-lstm-q20-performance-2026-09-19/terminal-verification.json`.

| Charge or balance | Seconds |
| --- | ---: |
| All profile attempts and fresh qualification workers | 2,104.077664 |
| Measured CPU verification, including prior attempts | 19.806927 |
| Predeclared setup/accounting | 180.000000 |
| Total phase charge | 2,303.884591 |
| Unused portion of the 3,000-second worker allocation, released | 895.922336 |
| Remaining campaign | 80,218.927012 (22.2830 hours) |
| Remaining diagnostics, included in campaign | 4,459.359231 (1.23871 hours) |

No budget was renewed. The current 23.721-hour combined mass scheduling forecast
exceeds the entire 22.283-hour remainder before later campaign work. The master
therefore does not launch full preparation on this forecast. This completes the
bounded performance repair and qualification; it does not end the research
direction or imply that the remaining budget is exhausted. A further runtime
repair and complete cost plan are needed before the full campaign can proceed.
The smallest next useful investigation concerns the remaining eigensolver work
inside each target call. Any change to refinement must retain the accuracy checks
and pass independent and downstream parity before it can affect the active route.
