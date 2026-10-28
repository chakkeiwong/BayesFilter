# Native TFP NUTS versus Stan: execution result

The rewrite and seven-target comparison are complete under the
[reviewed plan](bayesfilter-stan-tfp-native-2026-10-28.md). The new
`stan_nuts_tfp.py` uses stock TFP NUTS and `sample_chain`, with TFP
fixed-metric transformations and the existing Stan-style adaptation.
It no longer calls the direct Stan tree. The actual adapter, not the prior
manual sampling loop, is the measured TFP route.

Native TFP and Stan passed the declared posterior screens on six targets
for all three seeds. Both failed on the centered funnel. The direct controls
passed all seeds on Gaussian32 and banana8. Native TFP showed lower elapsed
times than the direct port on these controls, but native Stan had lower
elapsed times than TFP on all passing targets. These are descriptive
observations; three seeds do not establish statistical efficiency superiority.

## Comparison and timing boundaries

Median elapsed seconds over seeds 20261028, 20261029 and 20261030. Four
parallel scalar CPU chains, FP64, 1000 warm-up plus 2000 retained draws per
chain, target acceptance .8, max depth10, zero jitter. Dense inverse mass
for rotated Gaussian; diagonal for all other targets. Matched starting
positions per fit and common target/density checks.

| Target | Native TFP total | Stan process wall | Ratio of medians | Posterior screen per engine |
| --- | ---: | ---: | ---: | --- |
| Gaussian32 | .610 | .121 | 5.06 | 3/3 |
| Rotated Gaussian32 | 1.355 | .330 | 4.10 | 3/3 |
| Banana8 | .629 | .095 | 6.63 | 3/3 |
| Centered funnel10 | 1.082 | .168 | Not interpreted | 0/3 |
| Noncentered funnel10 | .506 | .074 | 6.84 | 3/3 |
| Regression32 | .972 | .353 | 2.75 | 3/3 |
| Regression64 | 9.414 | 3.810 | 2.47 | 3/3 |

TF totals include numerical warm-up (step search, adaptation and final
freezing) plus retained sampling, with synchronization. Native Stan process
wall includes process startup, initialization, warm-up, retention and CSV
writing. Compilation and common posterior diagnostics are excluded.
TF output serialization is separately measured: adding it gives workflow
medians .624, 1.361, .634, 1.087, .511, .982 and 9.433 seconds respectively.
These boundaries are explicit but not identical; the table is not a
microbenchmark of identical operations. Native endpoint caches are not
exported, so its checks use successful native execution and finite draws.

| Target | TFP warm-up | TFP retained | Stan max-chain warm-up | Stan max-chain retained |
| --- | ---: | ---: | ---: | ---: |
| Gaussian32 | .318 | .292 | .042 | .060 |
| Rotated Gaussian32 | 1.015 | .339 | .214 | .097 |
| Banana8 | .218 | .404 | .031 | .051 |
| Centered funnel10 | .382 | .688 | .072 | .081 |
| Noncentered funnel10 | .182 | .321 | .026 | .032 |
| Regression32 | .489 | .483 | .120 | .206 |
| Regression64 | 5.769 | 3.646 | 1.407 | 2.266 |

Each column is a separate median, so components need not sum to the median
total. Native columns are the maximum of CmdStan's printed chain timings
within each fit, then the median across fits. They have a different boundary
from measured process wall and are secondary explanatory diagnostics.

Native-TFP first calls, including compilation and a discarded execution,
sum to 6.19–8.88 seconds per target for warm-up and sampling. They are not
pure compiler time and are excluded from reusable-run totals. Native model
builds were reused. Common diagnostic costs are separate in each fit result.

## What changed relative to the direct port

The new class delegates tree building and selection to TFP, keeps native
target/gradient caches, and uses elementwise diagonal or triangular dense
bijections. The same audited native-TFP Stan-style warm-up remains in
`stan_nuts_warmup.py`. There is no copied adaptation or custom-tree path in
the rewritten class.

| Control | Direct total | Native TFP total | Direct retained | Native TFP retained |
| --- | ---: | ---: | ---: | ---: |
| Gaussian32 | .911 | .610 | .617 | .292 |
| Banana8 | 1.028 | .629 | .669 | .404 |

The observed total medians are 33% and 39% lower for native TFP, respectively.
This supports the practical rewrite for these tested controls but does not
establish general statistical superiority. On Gaussian32 all engines take
about seven leapfrogs per retained transition, so integration count alone
cannot explain the remaining runtime gap. No profile was run, and the
fractions attributable to RNG, TFP control flow, autodiff and tree memory
remain unmeasured. Regression64 still spends most TFP time in warm-up.

The rewrite changes algorithmic authority: it is native TFP with Stan-style
adaptation, not an exact transfer of Stan's transition or random generator.
The separate direct port retains its existing source-fidelity tests and
remains unchanged. The former adapter/tests/guide are preserved as text in
`previous-interface/`; historical results retain their original meaning.

## Validity and engineering evidence

All passing fits have worst R-hat below 1.005, minimum bulk/tail ESS above
1400, and maximum standardized MCSE below .029. They pass the predefined
reference-coordinate first/second moment checks and chain-movement checks.
These screens cover the declared observables, not arbitrary nonlinear
physical-coordinate estimands. Banana has occasional native divergence
flags (median fractions .00075 TFP and .000375 Stan); those are reporting
diagnostics under the declared policy, not convergence evidence.

The centered funnel fails all three seeds in both engines. Worst R-hat is
1.318 for TFP and 1.271 for Stan; minimum tail ESS is 5.65 and 6.28. Those
failures veto posterior and efficiency promotion for this configuration.
Reference-moment checks also fail some seeds. The planned noncentered
control passes all seeds, so this is a failure of the sampled centered
configuration under its budget, not evidence against the NUTS research
direction. No outcome was used to change tuning during this comparison.

October 28 follow-up: [independent saved-chain audit](bayesfilter-centered-funnel-audit-2026-10-28.md)
confirms these failures with native CmdStan `diagnose` and `stansummary`.
The target is the standard centered Neal funnel. Two Stan seeds have no
retained v < -3 although that region has 15.87% marginal probability. This
used native Stan's standard automatic adaptation with its default settings,
including delta=.8 and 1000 warm-up iterations. Adaptation completed normally;
the retained chains failed convergence checks. There was no additional
target-specific search over acceptance targets or run lengths. The result does not establish
that a tuned centered Stan run fails, and the noncentered control does not
replace that comparison.

Nineteen focused tests passed without skips: native TFP common-seed
equivalence, unit/diagonal/dense coordinate and gradient-cache checks,
exact continuation, reproducible explicit seeds, dense/diagonal adaptation
handoff, cumulative invalid-input flags, correctly rejected nonfinite
proposals, API migration/copy, and retired-route enforcement. Optimized
HLO is preserved for every benchmark warm-up and retained function. The
native-TFP functions each recorded one trace and no Python callbacks/pfor.
The log warning about repeated score-function tracing comes from creating
separate density-check functions for separate targets; measured functions
did not retrace.
Native/TF target-value and gradient relative discrepancies at the checked
pre-run points were at most 1.25e-15. This supports matching targets at those
points, without claiming an exhaustive proof over the entire parameter space.

## Provenance and budget

Artifacts: `docs/benchmarks/artifacts/stan-tfp-native-20261028/`.
`comparison-summary.json` contains medians, ranges, phase costs, diagnostics
and source-manifest references. Each fit preserves warm-up/retained output,
native command/CSV/logs and diagnostics. The benchmark manifest records
actual source hashes (including dirty worktree sources), TFP dependency
hashes, executable hash, seed/data identity, runtime, compilation and memory.
Final numerical sources match their recorded hashes.

Environment: Python3.13.13, TensorFlow2.20.0, TFP0.25.0, CmdStan2.37.0,
AMD EPYC7742. CPU scalar layout follows the earlier measured calibration
for these same fixtures; GPUs were intentionally hidden before import.
This run does not establish the new adapter's GPU performance or select a
general CPU default. All TF sampling/adaptation numerics were XLA compiled;
native Stan was C++, and host diagnostics/I/O were outside the timed kernels.

Attempt01: 19 tests passed, 149.693 seconds. Attempt02: 48 engine fits
(21 native TFP, 21 Stan, six direct controls), complete, 243.359 seconds.
Total charged subprocess wall: **393.052 seconds**, two attempts, within
the 1200-second/eight-attempt budget. Benchmark worker wall was 241.969
seconds; maximum host RSS was 2,992,564 KiB. No failed scientific launch,
package installation, GPU workload or default-policy promotion occurred.

Exact run command (bounded by the archived attempt launcher):

```text
/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/benchmark_native_tfp_stan.py
```

## Decision and final review

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain native TFP standalone implementation | Native API/parity/XLA tests pass; six-target screen passes | Centered configuration fails | Unmeasured GPU layout and other targets | Use documented native API; validate new target scopes | Exact Stan transition fidelity or public default promotion |
| Keep native Stan as timing reference | Lower observed elapsed times on passing cells | No passing-cell reference/retained-health failure | Three seeds, timing boundaries, machine variation | Profile costs if closing the native gap remains the goal | Statistical time-to-precision superiority |
| Reject centered-funnel posterior result | Convergence/precision criteria fail | R-hat/ESS and some reference checks | Reparameterization and budget adequacy | Use tested noncentered control; plan fresh tuning if needed | Failure of the target definition or NUTS generally |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Six targets remain viable in both engines; centered funnel fails |
| Statistically supported ranking | None |
| Descriptive differences | Native TFP lower times than direct controls; Stan lower than TFP |
| Default-readiness | Not assessed; standalone exception only |
| Next evidence needed | Profiling, more paired replications with uncertainty, and target-specific time-to-precision comparison |

Post-run review: the strongest alternative explanation for timing changes
is a mixture of different trajectories, compilation decisions and CPU
scheduling, not the API alone. The Gaussian work counts and fresh direct
controls narrow that explanation but do not isolate a causal component.
A repeated controlled profile or uncertainty-qualified replication could
overturn the descriptive ordering. The weakest evidence is generalization
to GPU, expensive non-Gaussian targets and nonlinear tail functionals.

The centered failure invalidates that candidate's posterior result, not
the harness, target, mathematical density, archive or implementation. The
planned geometry repair was completed and passed; no continuation veto
requires abandoning the research direction. Engineering and sampler
validity evidence remain separate from scientific ranking.

Reset note: use `stan_nuts_tfp.make_functions` for the complete native-TFP
lifecycle. Seeds are now one stateless pair, results wrap native TFP caches,
and `from_warmup` accepts the native-TFP adaptation result, not the direct
port's schema. These API changes are documented in `nuts-stan-tfp.md`.
The next speed task is profiling; further wrapper changes alone are not
evidence that the native-Stan gap will close.
