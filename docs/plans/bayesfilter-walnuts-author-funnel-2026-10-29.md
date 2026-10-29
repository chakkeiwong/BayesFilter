# Reproduce the authors' centered-funnel experiment

## Question and scope

Does the standalone TensorFlow/XLA R2P sampler reproduce the authors'
centered-funnel behavior when the target, identity metric, macro step, local
tolerance, jitter and tree limit match? The owner requests implementation,
review and execution. This remains the standalone benchmark exception; no
public sampler or production default changes. Previous deterministic C++ and
TF results and the prebuilt binary remain preserved.

The author experiment is in bob-carpenter/walnuts, commit
895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a, not a walnutpie package test. Copy
the required source and experiment files unchanged into a clearly independent
reference directory, retaining MIT attribution and hashes. The paper is the
locally preserved JMLR 27 (2026), 25-1452, sections 3.1--3.4, 4.3 and Appendix C.
Section 4.3 reports one million draws, h=.36, delta=.21 and explicitly slow
mixing. The scripts instead tune from h=delta=.3 for 1000 iterations.

## Evidence contract and research intent

| Role | Definition |
|---|---|
| Question | Transfer the author experiment, then compare posterior diagnostics with our R2P and historical deterministic tests |
| Mechanism | Author quantile tuning on CPU; freeze its h/delta and transfer to fully compiled TF sampling with identity mass |
| Exact comparators | Unmodified author Python sampler; separately labeled one-line backward-weight correction; current TF/XLA R2P |
| Engineering pass | Copied source hashes, target/gradient parity including normalizers, continuation equivalence and XLA graph checks pass |
| Posterior screen | Finite retained states, all coordinates move, maximum rank/folded R-hat <=1.01 and minimum bulk/tail ESS >=400; analytic transformed means/variances and tail rates assessed with MC uncertainty |
| Promotion veto | Failed posterior screen, wrong target, incorrect weights/caches, callback in TF sampling; no speed or equivalence promotion on failure |
| Continuation veto | Invalid launch/memory policy, unresolved target mismatch, corrupted evidence, exhausted budget |
| Repair trigger | Local source-loader, diagnostics, compilation or serialization defect; repair and retry within unchanged budget |
| Explanatory only | Runtime/work, chain traces, neck coverage, cap/depth counts, marginal summaries and historical short-run differences |
| Nonconclusions | Million-draw paper reproduction, identical random streams, entire-chain bitwise identity, native-XLA Appendix C tuning, superiority or universal convergence |

Each result separates implementation validity from posterior adequacy. A short
chain failing convergence motivates the next budgeted rung, not rejection of
the method. Paired chain-mean/tail differences and standard errors are
descriptive when convergence fails; they cannot establish equivalence.

## Sequence

1. Preserve author files and add a bounded independent-reference adapter.
   Validate target/gradient values at ordinary and neck states. Retain the
   source's normal(1) initialization (mean one, not size one), restart at q0,
   and last reported tuning parameters. Record these peculiarities explicitly.
2. Smoke/test the source loader, corrected-source patch and fixed-kernel chunk
   continuation. Run the unmodified 1000-iteration author quantile tuning for
   11 total dimensions, then the explicitly generalized 10-dimensional target.
   Freeze seed-zero tuned h/delta across the four comparison chains, so the
   transition comparison does not confound different tuning outcomes.
3. Compare author-original, author-weight-corrected and TF/XLA on 11 dimensions;
   compare corrected reference and TF on 10 dimensions. Four chains use seeds
   0,1,2,3 and identical source-generated starting positions. Use source-style
   restart; archive initial 512 fixed-kernel settling draws separately. Retain
   a ladder of 512, 2048, and at most 8192 draws per chain if measured cost and
   remaining budget permit. Budget allocation favors complete matched arms
   over an unmatched long run. CPU-XLA is an explicit reference, GPU-XLA the
   target. Record first compilation separately from synchronized sampling.
4. Adapt the transient test at v=-6 as a bounded mechanics check. Audit the
   exact v=-30/maxC=30 author's test separately: our current cap <=19 cannot
   represent its complete selector. Permit a <=60-second author macro cost
   probe; a timeout is cost evidence only. Do not silently lower its cap or
   claim that the -6 test reproduces the -30 experiment.
5. Write comparative diagnostics, preserved draws, manifests and terminal
   review. Compare historical 10-dimensional deterministic results only as
   a separately labeled context row, never an algorithm- or effort-matched arm.

## Assumptions, provenance and pre-mortem

| Choice | Provenance/status; risk and early diagnostic |
|---|---|
| 11 versus 10 total dimensions | Author target slices x[1:11]; prior local fixture D=10. Explicit labels and gradient parity prevent silent mismatch |
| 1000 tuning, M=12, jitter=.2, probability=2/3, cap=10 | Author mainFunnel.py and integrator defaults. Cap and nonfinite telemetry cannot be inferred from convergence alone |
| Shared seed-zero tuning | Comparison design, not an author default. Preserves original seed-zero protocol and isolates transition behavior; not per-chain tuning superiority |
| Seeds 0..3 | Convenience replication of author's seed-zero experiment. Four chains are not four independent tuning replications |
| 512 settling and 512/2048/8192 retained | Bounded diagnostic ladder, not adequate-convergence assertion. R-hat/ESS and analytic checks determine adequacy |
| R-hat 1.01; ESS 400 | Existing modern diagnostic screening convention, not a proof. Include v and whitened coordinates, tail counts and uncertainty |
| FP64, identity mass, frozen source tuning | Source numerical baseline. Avoids substituting the unrelated Walnutpie Adam/metric warmup |
| Analytic generalized log density | Independent reference expression; test against original normalized scipy target at D=11 before use for cheaper reference evaluations |
| Corrected backward weight | Paper eqs 13--14 and local prior audit: source second backward leaf omits lwtSumb += intOut.lwt. Preserve unmodified arm and patch exact unique context |
| v=-6 transient | Convenience affordable mechanics case, not paper's v=-30. Exact extreme case has distinct cost and exponent limitations |

The run could look successful simply because a chain visits the neck while
mixing poorly; require full diagnostics. It could fail due to a different
dimension, warmup, start, source bug or short run; isolate these before judging
the algorithm. Python-reference versus GPU timing is descriptive and does not
compare the GPU to optimized native C++. Author tuning remains independent
reference code; no NumPy enters the TF execution path.

## Budget, commands and artifacts

Convenience bounded budget: 3600 cumulative launched-worker wall seconds,
24 launches, at most 4 CPU workers per launch, 600 seconds per ordinary
launch and 60 for the extreme-neck probe. Sum worker wall times across
sequential launches; a parallel four-chain launch counts elapsed group time
and records worker count (not a claim to CPU-seconds). Stop before launching
a rung whose measured projected cost exceeds its allocation. No new packages.

Environment: /home/ubuntu/anaconda3/envs/tfgpu/bin/python. CPU reference hides
GPUs with CUDA_VISIBLE_DEVICES=-1. GPU uses trusted execution, selected idle
device, TF_FORCE_GPU_ALLOW_GROWTH=true before imports, verified growth,
TF32=false, stable signatures and jit_compile=True.

Driver: python -m docs.benchmarks.walnutpie_gpu_tf.funnel_master
with --phase tests|tune|sample|transient|extreme|analyze, explicit --engine,
--dimension, --draws and --output as appropriate. Exact commands are captured
in each manifest, with git revision, source hashes, environment, seeds,
device/allocator, wall time, plan/result paths and failure classification.
Output root: docs/benchmarks/artifacts/walnuts-author-funnel-20261029/.
Result: docs/plans/bayesfilter-walnuts-author-funnel-result-2026-10-29.md.

## Skeptical pre-execution review

Reviewed before implementation and experiments. A naive copy would confound
dimensions, substitute Adam tuning, interpret normal(1) incorrectly, ignore
the known source weighting discrepancy, and call a short run a reproduction
of one million draws. Those flaws are corrected above. The exact deep-neck
selector exceeds our current supported cap; it is a bounded cost audit, not
silently altered parity evidence. No proxy (tail visitation or speed) can
override a posterior-screen failure. The amended bounded plan passes this
self-review; statistical equivalence and a full Appendix C XLA port remain
outside its claims.

## Execution repairs and measured ladder decisions

Attempt 01 exposed an unused matplotlib import in the author sampler. The
environment has no matplotlib; the headless adapter omits that unused import
in memory and records the diff instead of installing a plotting dependency.
Copied files remain byte-identical. Attempt 02 passed all four focused tests.

Attempt 05 exposed a timing defect in the inherited async_wait helper: elapsed
dispatch time excluded GPU execution that subsequently blocked tensor reads.
Its draws and graph remain useful, but its steady timings are disqualified.
The new harness times through materializing the returned state tensor, an
explicit device completion barrier. Subsequent runs record that barrier.
This also means historical timing rows are not used for a speed conclusion.

The 11-dimensional 2048-draw author run completed in 21.35 group wall seconds;
the corrected 8192-draw run completed in 74.85 seconds. These measurements
justify the planned 8192-draw rung within the existing budget. Frozen author
tuning is h=0.3228754012018425, delta=0.15442326531457057. These are this
source/seed/environment's results, not the paper's rounded .36/.21 constants.

## Completion and reset

Completed the bounded comparison on 2026-10-29: copied source/experiments,
headless reference adapter, original and corrected author arms, author tuning
for D=10/11, matched frozen-parameter CPU/GPU XLA sampling, five tests with
16 dimension-specific matched-input trees, the 512/2048/8192 ladder, neck-start
mechanics, extreme-start cost probe and final diagnostics/review. All six
8192-draw arms fail the combined posterior screen; none supports a superiority
or posterior-equivalence claim. GPU graphs are single-trace and callback-free.
The v=-30 author macro timed out at the predeclared 60-second cap; the full
experiment is explicitly incomplete and requires a separate exponent/cost
audit. Author quantile tuning is a CPU reference input, not native TF tuning.

Used 943.22/3600 worker-group wall seconds and 18/24 launches. Result and
decision tables: bayesfilter-walnuts-author-funnel-result-2026-10-29.md.
Final source, analysis-input and prebuilt hashes verified; no live launch
remains. Numerical sampler core unchanged during this task.
