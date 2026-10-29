# Author centered-funnel comparison: implementation and results

The authors' experiment is copied, adapted and executed. Both the author R2P
sampler and our TensorFlow/XLA R2P sampler explore the narrow funnel neck.
All compared arms nevertheless fail the combined posterior screen at 8192
retained draws per chain. This reproduces the paper's qualitative distinction
between successful local refinement and slow global mixing; it does not
reproduce its million-iteration posterior experiment or prove equality of the
stationary distributions.

## What was implemented and reviewed

Plan: [reviewed plan](bayesfilter-walnuts-author-funnel-2026-10-29.md).
Nine upstream files, including experiments and MIT license, are copied unchanged
under ../benchmarks/walnutpie_gpu_tf/author_reference/. The source pin is
bob-carpenter/walnuts 895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a; hashes are
checked before loading. This is the author WALNUTS repository, not a funnel
test shipped in the separate walnutpie C++ package.

New funnel_reference.py adapts the original experiment as an independent NumPy
reference. The runtime funnel_xla.py uses only TensorFlow numerical kernels,
stable signatures and XLA, reusing the existing R2P implementation. The
bounded funnel_master.py records commands, hashes, seeds, environment, time,
device settings and fresh output directories. funnel_worker.py saves draws
and computes post-run modern diagnostics with the repository TF authority.
The author source is never imported into the TF sampling kernel.

The skeptical review identified mismatched dimensions, different adaptation,
initialization semantics, source weighting, run length, and an infeasible
extreme-start assumption before execution. The authors' funnel10 has 11 total
dimensions; our older funnel10 has 10. Both are now tested explicitly. The
literal normal(1) start is preserved: it draws from N(1,1), producing v=8.292
for seed zero. Sampling restarts at the original q0 as in mainFunnel.py.

Tuning uses the original 1000-iteration Appendix C quantile procedure on the
CPU reference, with the last *reported* parameters transferred to all four
chains, identity metric and no subsequent adaptation. This matches the script's
parameter extraction, not its internal post-final-iteration update. Tuning is
not yet native TF/XLA. It is not replaced by Walnutpie's Adam/metric warmup.

| Total dimension | Tuned macro h | Tuned local delta | Tuning seconds |
|---|---:|---:|---:|
| 11, author target | 0.3228754012 | 0.1544232653 | 8.22 |
| 10, prior local target | 0.3335953197 | 0.1531784720 | 10.46 |

The 10D timing above is rounded from the saved tuning result; these settings
are the results for the pinned source and seed, not a claim to exactly recover
the paper's rounded h=.36/delta=.21. Direct normalized analytic log density
replaces repeated scipy distribution calls after value/gradient checks against
the original target. Floating-point ordering can change an adaptive chain.

The headless adapter drops an unused matplotlib import in memory because that
package is absent. No package was installed. The author-original arm preserves
all sampler arithmetic. The author-corrected arm adds one missing
`lwtSumb += intOut.lwt` before the second backward leaf's weight (upstream
WALNUTS.py, around line 462). This follows the cumulative correction in paper
equations (13)--(14). Both arms are retained; no copied file is changed. The
local TF implementation already includes the correction and uses exact
negative infinity for zero probability rather than the author's -700 sentinel.

## Comparison

Four chains per arm, 512 fixed-kernel settling draws archived and excluded,
8192 retained per chain. Author sampling seeds are 0--3, with seed zero's
post-tuning random state restored; TF Philox keys are 104729--104732. Starts,
target, h/delta, identity metric, M=12, jitter=.2, probability=2/3 and cap=10
are matched. Random streams and tree implementations are not bitwise matched.

Primary screen: finite retained states, movement in all coordinates, maximum
rank-normalized split/folded R-hat <=1.01 and minimum bulk/tail ESS >=400,
including analytically whitened coordinates. Tail frequencies and batch
standard errors explain discrepancies but cannot override a failed screen.

| Dimension / implementation | Max R-hat | Min bulk ESS | Min tail ESS | P(v<-3) | P(v<-6) | Screen |
|---|---:|---:|---:|---:|---:|---|
| 11 author original | 1.0192 | 308.6 | 501.5 | .1766 | .02194 | Fail |
| 11 author corrected | 1.0119 | 287.6 | 261.9 | .1735 | .02545 | Fail |
| 11 TF GPU XLA | 1.0080 | 368.8 | 540.2 | .1383 | .01343 | Fail |
| 11 TF CPU XLA | 1.0080 | 368.8 | 540.2 | .1383 | .01343 | Fail |
| 10 author corrected | 1.0164 | 329.7 | 503.5 | .1972 | .03796 | Fail |
| 10 TF GPU XLA | 1.0033 | 319.6 | 449.8 | .1767 | .02689 | Fail |
| Analytic marginal N(0,9) | — | — | — | .15866 | .02275 | Reference |

All retained samples are finite and every coordinate moves. TF reports zero
nonfinite-trajectory flags in these runs. GPU sample ranges reach v=-9.13 in
11D and v=-10.87 in 10D. Author corrected chains reach -10.08 and -11.74.
Author critical refinement maxima are 8 and 9, below cap 10; a few trajectories
reach the tree-depth cap. TF run telemetry does not expose critical cap-hit
counts, so author counts cannot establish their absence on TF.

The ladder shows why the earlier short tests were misleading as a judgment of
the algorithm. For the 11D original author code, max R-hat is 1.310 at 512
draws, 1.033 at 2048 and 1.019 at 8192. For 11D GPU it is 1.198, 1.041 and
1.008. For 10D GPU it is 1.230, 1.077 and 1.003. More draws substantially
change the diagnostics, but the final minimum bulk ESS remains below 400.

Tail agreement is not established statistically. GPU minus corrected-reference
P(v<-3) differences are -.03519 in 11D (combined batch SE .02198) and -.02054
in 10D (SE .02566). Descriptive 95% intervals include zero in both cases.
For P(v<-6), differences are -.01202 (SE .00706) and -.01108 (SE .00973).
These intervals assume stationarity and adequate blocks, which the failed
screens do not establish. They neither prove equivalence nor demonstrate a
bias or a ranking. In particular, 11D GPU underrepresents v<-6 in this run;
longer independent replications are needed to distinguish noise from bias.

The historical 10D deterministic benchmark used different starts, Walnutpie
adaptation, depth limit and only 500 warmup +500 retained draws. Its GPU
R-hat=1.3972/min bulk ESS=8.83 and native C++ R-hat=1.1425/ESS=21.58 both
failed. The current result does not attribute the difference to R2P alone: the
method, tuning and amount of sampling all changed. It demonstrates that the
old experiment was not a reproduction of the author's R2P experiment.

## Execution and speed

Python 3.13.13, TF 2.20.0, FP64, TF32 off. GPU runs use trusted execution on
physical GPU 1 (RTX 4080 SUPER), with growth verified before initialization.
CPU references deliberately hide GPUs. Both 10D and 11D GPU graphs have a
single trace, in-loop model gradients, nine while operations and no custom
calls. No Python sampler callback is involved.

| Implementation | Dimension | Sampling seconds, four chains | First XLA call seconds |
|---|---:|---:|---:|
| Author original, four CPU processes | 11 | 63.86 | N/A |
| Author corrected, four CPU processes | 11 | 74.63 | N/A |
| TF CPU XLA | 11 | 16.74 | 7.07 |
| TF GPU XLA | 11 | 148.82 | 20.20 |
| Author corrected, four CPU processes | 10 | 70.89 | N/A |
| TF GPU XLA | 10 | 142.17 | 32.72 |

Sampling includes the 512 settling +8192 retained transitions. TF times include
materializing the resulting state tensor; first-call compilation and preview
execution are separate. Reference time includes process startup and result
collection, while TF time excludes JSON/reporting. Different random streams
produce different work totals. These single-run numbers are descriptive, not
ESS/sec rankings or a comparison with optimized native C++. CPU XLA is the
measured faster execution for this small four-chain reference workload; the
repository's GPU default and larger-model behavior are unchanged.

Attempt 05's inherited async_wait helper measured GPU dispatch before subsequent
tensor reads blocked on completion. Its timing is excluded; attempts 07 onward
use the explicit result-tensor barrier. Prior historical timing numbers are
not relied on for a speed claim. GPU allocator-reported peaks were 194304
bytes (11D) and 186624 (10D); these are TF allocator measurements, not total
CUDA/compiler memory. Host peak RSS was approximately 1.2 GiB for these GPU
workers.

CPU/GPU 11D runs have identical depths, macro counts, reverse-zero flags and
tail frequencies; evaluation totals differ by one (4420072 vs 4420073).
Maximum absolute draw difference is .00402 over the full long trajectories.
They are not bitwise identical. Local matched-input checks, not long chaotic
path equality, supply the numerical implementation evidence.

## Neck-start test and incomplete extreme case

The bounded 512-transition, four-chain v=-6/x=0 test stays finite and escapes
the neck on both corrected author and GPU arms. Final v values are
[.81,2.90,.61,6.23] for the author and [3.10,2.26,1.99,2.31] for GPU. This is
mechanics evidence, not convergence.

The paper's Figure 16 starts at v=-30 with ten x coordinates zero; the script
uses maxC=30 and runs 1000 iterations (the plotted portion is 300). Our exact
one-macro author cost probe at that state exceeded 60 seconds and was stopped
as planned. It supplies no completed transition or failure-of-method evidence.
The TF port currently validates maxC only through 19, so the complete extreme
selector is also not yet representable by its supported configuration. The
full v=-30 experiment remains incomplete; the v=-6 test is not a substitute
for that claim. A future exact reproduction needs an audited large-exponent
implementation and a separately budgeted cost ladder.

## Checks, decisions and terminal review

Five focused tests pass: copied source identity/patch, normalized target and
gradient parity in both dimensions (including v=-30), initialization semantics,
fixed-kernel continuation, original/corrected author smoke, and 16 matched-input
whole-tree cases through the author macro plus independent paper-equation tree.
The latter compare values, gradients, selected states, depth, work, macros and
random-tape consumption at rtol/atol 2e-11. Existing core/api/r2p source hashes
are unchanged from the sampling runs. The prior 504-case per-backend and
528-record native regression remain earlier engineering evidence, not new
posterior tests. The prebuilt C++ oracle SHA-256 remains
ea5d4965854c950c3a8d91f9b4ff5452a6283422edfa0b2e3d51ec15a326f45b.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep copied experiment and adapter | Source/target/continuation checks pass | No engineering veto in completed runs | Whole-author tree differs by known correction | Reuse bounded harness | Universal source identity |
| Do not promote posterior output | All six arms fail combined screen | R-hat and/or ESS veto | Slow mixing and finite runs | Longer independent runs with predeclared precision | R2P research direction rejected |
| Treat timings descriptively | Explicit synchronized timings captured | Posterior veto forbids ESS/sec ranking | One run, different work, CPU reference overhead | Replicated timing on valid posterior workloads | General GPU inferiority or native-C++ comparison |
| Leave extreme-start reproduction open | Author macro incomplete at 60s | Cost cap, not mathematical veto | maxC support and potentially large microstep work | Audit int/work bounds, then budget exact test | Paper's cold-start claim disproved |

| Inference status | Finding |
|---|---|
| Hard veto screen | All retained states finite; full posterior screen fails for every arm |
| Statistically supported ranking | None |
| Descriptive-only differences | R-hat/ESS, tail frequencies, work and elapsed times |
| Default-readiness | No production/default promotion; standalone scope only |
| Next evidence needed | More independent tuning/sampling replications, adequate ESS and analytic-tail precision; exact extreme-start implementation/cost audit |

Terminal self-review: the strongest alternative explanation for observed tail
differences is long autocorrelation under different streams, with the known
author tree bug another reason not to merge reference arms. Larger independently
tuned chains with precise persistent tail discrepancies would overturn the
current descriptive agreement. The weakest evidence is the far-left-tail
comparison and incomplete v=-30 probe. No implementation or target failure
was found by the completed checks. The failed posterior screens reject these
finite-run estimates, not the algorithm or research direction.

This was a skeptical self-review, not an independent-agent review. The final
post-run diagnostics intentionally inspect several bounded sample/parameter
shapes, triggering TF retracing warnings; repeated sampling kernels remain
single-trace. No sampler core or public default was modified in this task.

## Artifacts and reset

Machine-readable summary:
../benchmarks/walnutpie_gpu_tf/author-funnel-evidence-20261029.json.
Full evidence root:
../benchmarks/artifacts/walnuts-author-funnel-20261029/. Final diagnostics:
attempt-18-final-analysis/result.json. All 18 launch manifests include commands,
source snapshots and timing. Attempt 01 failed on unused matplotlib and was
repaired; attempt 17 hit its planned cost cap. Used 943.22/3600 group worker
wall seconds and 18/24 launches. No processes remain running. No commit was
requested. Unrelated dirty worktree changes and the saved binary are preserved.

The bounded plan is complete with the explicit extreme-start and native-tuning
limitations above. Do not restart the old gap-closure campaign or treat the
unspent budget as permission to silently expand this into the million-draw
paper experiment.
