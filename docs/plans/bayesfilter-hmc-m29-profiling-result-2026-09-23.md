# M29 numerical-child profiling result

M29 closes the isolated-child profiling gap. Profiling off/on produced exactly
the same numerical experiment in the public ordinary Gaussian and beta-binomial
pipelines. Both child profiles contain the actual tuning, retained posterior,
and fixed-comparator calls. The terminal audit passed source, design, runtime,
receipt, candidate, observation, tensor and posterior-decision checks.

The question was whether trustworthy host cost attribution could be obtained
without changing the fit. The comparator was the identical unprofiled design,
not an earlier seed or a coordinator wait profile. One paired fit per target
answers this engineering question; it does not establish coverage, statistical
superiority, a tuning ranking, or a new numerical default.

## Repair and checked execution

The execution-only `run --profile-execution` flag preserves the numerical
design and seed identity. With isolated fits, the profile starts in the
numerical child after framework and pipeline imports; setup time is recorded
separately. Profiles are saved in a `finally` path on success and failure.
Setup, disable, write and warning failures cannot replace the numerical
outcome. Completed resume inspects availability and checksums without replaying
a fit to manufacture missing historical instrumentation.

Starting a default profiler in the child still omitted outer numerical calls.
A small TensorFlow reproduction motivated `enable(builtins=False)`, which
restored those calls in the checked Python 3.13/TensorFlow environment. The
failed test attempts are preserved as instrumentation failures. The exact
C-extension event mechanism is not established. The resulting Python-frame
profile includes synchronization and compilation within TensorFlow calls and
cannot separate GPU kernel time from XLA compilation.

The numerical source identity is
`71ef22422903ca463337c99801ca632b91c7cba8675cb7d437bcfe43f70cce53`,
from committed base `a5aa1fa94` plus M29-owned changes. Unrelated Q20 and
NeuTra work was excluded. The paired designs preserve the complete M27
model-specific numerical settings, with new convenience seeds 2026092391 and
2026092392. A checked written count table includes the Gaussian 60000 warmup
ceiling. All four attempts ran serially on GPU 1 with XLA, verified memory
growth, and normal process exits.

| Target | Candidates | Verified retained | Observations / numerical evidence | Tensor records | Off seconds | On seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 92 | 22 | 155 | 84 | 617.993 | 625.242 |
| Beta-binomial | 100 | 18 | 171 | 94 | 613.070 | 669.296 |

All 40 distinct verified members remain retained. The two predeclared selected
members passed their posterior checks: Gaussian at 30000 warmup and 4000
retained draws per chain, beta-binomial at 2000 and 5000. Their fixed-count
comparators were also recorded. Other members remain explicitly unassessed.
R-hat, ESS and MCSE did not alter tuning membership or rank candidates.

The audit verifies full integrity hashes separately before replacing
clock-dependent references with numerical digests for comparison. Apart from
declared clocks and output paths, every evidence item, tuning lifecycle,
numerical execution binding, posterior selection and independent assessment
matched. Profiling arms are replays and add no independent-fit denominator.

## Measured costs and interpretation

Unprofiled tuning invocations consumed 462.321 seconds for Gaussian and 517.266
seconds for beta-binomial. Their separately reported preparation times,
37.905 and 42.664 seconds, are contained within those invocations. They must
not be added to them. Retained-controller costs were 50.568 and 35.605
seconds; fixed-comparator costs were 60.321 and 30.924 seconds.

In the profiles, TensorFlow `quick_execute` accounts for 455.85 and 477.24
seconds. Almost all of this is through graph calls. The existing runner cache
uses (L, count), already reusing epsilon changes. Four scalar-chain runners
per key and multiple counts create many graphs; the unprofiled terminal
snapshots contain 132 and 148 reusable runners and 1139 and 1280 registered
TensorFlow functions. This motivates distinguishing first-call compilation
from warmed execution before changing graph reuse.

The subsequent saved-chunk audit confirms 29 and 33 cache keys in the
observation stage. First calls consume 313.707 of 354.947 seconds and 369.978 of
399.670 seconds respectively. The first version of that diagnostic confused
work-creation order with execution order; its assertion was repaired using the
saved observation sequence. No sampler needed rerunning. The checked ledger is
`m29-r1/cost-attribution-r2.json`.

Checkpoint writing is a separate measured cost: 708 calls / 61.911 cumulative
seconds and 794 calls / 73.458 seconds in the respective profiles. JSON encoding
and repeated integrity hashing contribute. Nested cumulative times overlap.
Skipping integrity checks or changing checkpoint cadence is not justified.
Even eliminating this entire roughly ten-percent cost would not fund the
unchanged confirmation inventory.

The observed profiling overheads are 7.249 and 56.227 seconds. One pair per
target cannot separate instrumentation overhead from run variability and does
not support a speed ranking. Unprofiled prices imply 65.92 GPU hours for 384
Gaussian fits and 65.39 hours for 384 beta-binomial fits, compared with about
21.47 GPU hours remaining. These are descriptive prices, not upper bounds.
The adequate confirmation inventory remains under-budgeted.

Resource snapshots record Gaussian allocator peaks of 39.21 / 40.97 MB and
beta-binomial peaks of 8.56 / 8.80 MB for off/on. Host high-water RSS was
approximately 5.7--6.0 GiB. Allocation and graph-object counts are descriptive;
garbage-collection timing can change live-object counts between paired runs.

## Tests, guide and accounting

The final focused suite passed 43 tests covering success/failure profiling,
setup/disable/write failures, warnings-as-errors, missing/corrupt profiles,
resume, actual two-model public pipeline parity and design/count drift.
All 15 documentation-contract tests passed after completing missing
non-numerical snapshot inputs. Early failures remain charged and recorded.
The official chapter and agent reference describe the same procedure.
The official book was rebuilt with BibTeX; the changed page was inspected,
including a repair to preserve the space in the CLI example. Build evidence
and hashes are recorded in `m29-r1/guide-validation.json`.

The terminal ledger is
`artifacts/hmc-repair-master-2026-09-16/m29-r1/reconciliation-terminal.json`.
It charges every outer attempt once, including failed diagnostics, document
builds and the mistaken self-metered reconciliation attempts. A declared
60-second conservative CPU allowance covers short bookkeeping and inspection.
The final totals in that ledger supersede intermediate checkpoint balances.
Raw fits, profiles and tensors remain local; the compact committed terminal
summary binds the full audit by checksum.

M29 charged **668.153 CPU / 2525.602 GPU seconds**, including 32 outer
attempts and the CPU allowance, within its 1800/4800 ceiling. The campaign
remainder is **74979.679 CPU / 77294.489 GPU seconds**. No numerical workers
or M29 reservations remain.

## Decisions and remaining evidence

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close M29 engineering cell | Two exact numerical pairs and complete child profiles | No unresolved source, design, runtime or artifact veto | Only two model/seed cells | Measure first-call versus warmed graph cost | No speed or broad reliability guarantee |
| Preserve posterior results | Both selected members passed operational health and precision | No declared health veto; native divergence telemetry remains unavailable | Finite-window diagnostics | Keep unassessed members and independent-fit accounting explicit | No global convergence or coverage proof |
| Continue the repair program | Remaining campaign allowance is positive | Confirmation affordability remains a limit for that inventory | Cost attribution cannot yet price a repair | Execute reviewed M30 within its own ceiling | No campaign completion or default promotion |

| Inference status | Evidence and limit |
| --- | --- |
| Hard veto screen | Exact parity and artifact/runtime checks passed; failed instrumentation attempts are preserved |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime, profiler overhead, graph counts, RSS, allocator peaks and posterior diagnostics |
| Default readiness | Profiling remains optional; no numerical/default-policy change |
| Next evidence needed | First-call/warmed attribution, behavior-preserving repair if justified, then independently powered confirmation |

The strongest alternative explanation is that most graph-call time is useful
transition execution rather than avoidable compilation. That would invalidate
graph-cache optimization as the next major affordability repair. The next
small diagnostic is designed to distinguish those explanations. General
stopping calibration, global exploration, subtle full-fit null/power, exact
MacroFinance inputs, and upstream learned-map quality remain separate gaps.
The next executable phase is
[M30](bayesfilter-hmc-post-m29-next-phase-2026-09-23.md).
