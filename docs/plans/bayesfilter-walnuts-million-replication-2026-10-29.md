# Four-method million-iteration funnel replication

## Active amendment: analytic evaluator restart, October 29

The owner explicitly stopped the literal-SciPy campaign and requested the faster
equivalent analytic evaluator for this replication. This amendment supersedes
the literal-evaluator requirements below; those paragraphs preserve the first
attempt's provenance. The target, four methods, starts/seeds, million transitions
per chain, four chains, Appendix C tuning algorithm, posterior screens and
uncertainty analysis remain unchanged. Both author Python arms use
`funnel_reference.target(11)` for warmup and sampling. TF CPU/GPU already use
the same analytic formula, omitting only its state-independent normalizing
constant. Their numerical kernels remain TensorFlow/XLA; there is no SciPy or
NumPy callback inside them. This is algorithm/protocol replication with an
equivalent evaluator, not literal floating-point reproduction of the script.

Evidence: expanding the eleven Gaussian log densities gives the normalized
formula in `target(11)`; the 8,277-point diagnostic in
`docs/benchmarks/artifacts/walnuts-analytic-scipy-parity-20261029-01.json`
records maximum scaled relative density discrepancy 2.574e-15. Numerical
agreement supports evaluator substitution, not pathwise RNG/adaptation equality,
posterior convergence or speed ranking. Fresh analytic warmup is required; its
final h/delta and post-warmup RNG are shared across all new arms. Do not pool old
and new draws or silently reuse the SciPy-tuned parameters.

Execution: add an explicit evaluator selector, reject mismatched tuning records,
test both selectors and chunk/RNG continuation, then run a 64-draw four-process
mechanics smoke. If it passes, launch all four million-draw processes together
into `docs/benchmarks/artifacts/walnuts-million-analytic-20261029-01/`. The smoke
and fresh tuning use `walnuts-million-analytic-preflight-20261029-01/`. The
smoke has no posterior-validity interpretation. Source snapshots include this
amendment and the numerical code; the run records evaluator identity.

Attempt 01 is preserved under `walnuts-million-20261029-01/`, marked stopped by
owner, with progress/checkpoints and a shutdown record. Both Python arms were
terminated after about 2,223 seconds; TF CPU completed in 1,480 seconds; TF GPU
had already failed after 555 seconds with CUDA_ERROR_LAUNCH_FAILED. The kernel
log records NVIDIA Xid 13 for that process on physical GPU 2. This is execution
failure evidence, not evidence against the analytic density or sampler. Its
root cause is not established. Physical GPU 1 is now idle and the same hardware
class; use it with verified growth for the restart, without changing the XLA
kernel or claiming the failure is repaired. Another GPU execution failure is a
continuation veto for that arm and must preserve its diagnostics.

Budget: count the stopped attempt against the previously approved 16-hour
per-method / 64 aggregate method-hour ceilings. The smallest recorded remaining
allowance is 55,377 seconds. Round the new common per-arm cap down to 54,000
seconds (15 hours), reserving at least 1,377 seconds for fresh tuning/smoke.
At most one new long launch per method remains in the two-launch campaign.
Analysis retains its original one-hour cap. Stop conditions and scientific
vetoes below continue unchanged; lower evaluator overhead does not guarantee
faster TF execution, which was analytic already.

Skeptical audit: passes with the preceding conditions. The main risks are mixing
tuning/evaluator identities, mislabeling literal replication, treating density
parity as posterior evidence, reusing completed CPU results under changed tuning,
and overlooking the independent GPU failure. Explicit identity checks, fresh
tuning, separate versioned outputs, focused tests and the GPU smoke address the
engineering risks. The million-draw diagnostics, not the smoke or microtiming,
remain the evidence for the scientific question. No criteria are relaxed.

Restart verification: nine focused tests passed in 3.31 seconds (both evaluators'
chunk/state/RNG continuation, evaluator mismatch rejection, no SciPy call on the
analytic path, binary integrity and TF/XLA density/gradient parity). The fresh
64-draw concurrent preflight completed all four methods in 56.43 seconds on
physical GPU 1. GPU growth was verified before initialization, XLA had one trace
and no custom calls, and checkpoint serialization succeeded. Fresh 1,000-step
analytic warmup took 8.37 seconds and produced h=0.3228754012018425 and
delta=0.15442326531457057. These differ from the SciPy-warmup parameters, as
floating-point adaptation can follow different trajectories. Both warmups use
the same mathematical target. This smoke does not establish that the earlier
GPU launch failure cannot recur. Proceed with the bounded second long attempt.

## Research intent and authorization

The October 29 owner request authorizes this standalone study and four concurrent
method processes. It extends the existing standalone benchmark exception only;
public sampler defaults and unrelated work are untouched. The question is whether
the original author Python, explicitly corrected author Python, TF CPU/XLA, and
TF GPU/XLA explore the same 11-dimensional centered funnel at the author's
million-iteration scale. This is replication of the pinned `mainFunnel.py`
protocol, with four seed replications, not reproduction of every published figure
or a claim that its newly tuned h/delta equal the paper's rounded .36/.21.

Source: bob-carpenter/walnuts commit 895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a,
vendored immutable `author_reference`; paper §4.3 and Appendix C, local PDF
`.localresources/funnel-local-adaptation-20261028/walnuts-jmlr-2026.pdf`.

## Protocol and assumptions

- Eleven coordinates (v plus ten x), v~N(0,9), x|v~N(0,exp(v) I).
- Literal upstream scipy density/analytic gradient in both Python arms. The
  previous analytic substitution is NOT used for the replication baseline.
- Seed-zero original-author warmup: 1000 iterations, h=delta=.3, Appendix C
  quantile tuning, original initialization including `normal(1)` meaning mean 1.
  Restart at q0 with final reported h/delta; restore its post-tuning RNG.
- All four methods use the same frozen h/delta, identity mass, depth 12, jitter
  .2, R2P probability 2/3 and cap 10. Python seeds 0..3; TF keys 104729..104732.
  These seeds are inherited reproducibility choices, not independent tuning
  replications. Other than seed zero, starts/RNG are an explicitly added extension.
- One million transitions per chain, four chains per method; retain all draws.
  The primary script comparison uses all million; a separately labeled
  sensitivity excludes the first 512, inherited from the preceding experiment.
- Author-corrected applies only the recorded backward second-leaf weight fix.
  The original is an observation baseline, not assumed mathematically correct.
- CPU reference processes hide GPU; TF graphs use FP64, stable signatures, XLA,
  TF32 off; GPU growth verified before initialization. Different RNG engines
  preclude pathwise equivalence. Concurrent timings include contention.

## Implementation and evidence contract

1. Implement bounded-memory chunk writers, full-state/RNG checkpoints, live
   progress, source snapshots/hashes and an automatically completing supervisor.
2. Before long sampling, test chunked-vs-monolithic author state/RNG equality,
   density/gradient parity, TF continuation, artifact integrity, and four-arm
   concurrent smoke. Reference tuning uses literal upstream target.
3. Launch four method processes together. Each Python method owns four CPU
   chain workers; TF methods batch four chains in their own process.
4. Analyze archived draws on CPU; save modern rank split/folded R-hat, bulk/tail
   ESS, standardized z=(v/3,x exp(-v/2)), means, variance, quantiles, v-tail
   frequencies, and batch-mean uncertainty. Preserve plots and source settings.

Engineering pass criteria: tests pass, exactly one million finite states per
chain, caches valid, no callback/host sampler in TF HLO, complete files/checksums.
Posterior screen (inherited reviewed screening thresholds, not a proof): all
coordinate and transformed R-hat <=1.01, bulk/tail ESS >=400, every chain moves.
Compatibility is assessed only after these screens, using simultaneous
Bonferroni 95% batch-mean intervals for z means and v<-3/v<-6 frequencies
against analytic truth and corrected-author comparisons. An interval including
zero is lack of detected difference, not proof of equivalence; report its width.
Original-vs-corrected differences cannot veto TF's agreement with the corrected
mathematical target. No speed/ESS ranking from these single concurrent runs.

Repair triggers: serialization/checkpoint/harness errors; localized retries
within budget. Promotion veto: failed posterior/reference screen. Continuation
veto: source corruption, invalid retained state/cache, lost required draws,
resource deadline, unsupported GPU memory policy. Slow mixing alone does not
abort planned million draws. The extreme v=-30 experiment is separate and is
not being claimed here. A million draws does not guarantee convergence.

## Compute, commands and artifacts

Environment: /home/ubuntu/anaconda3/envs/tfgpu/bin/python, TF 2.20.0.
Output: docs/benchmarks/artifacts/walnuts-million-20261029-<UTC timestamp>.
Preflight cap: 20 minutes total; each arm cap: 16 hours wall time; overall cap:
17 hours wall time, 64 aggregate method-hours; at most two long launches per
method, charged cumulatively. These are convenience safety ceilings, not
performance predictions. Preflight reports literal-scipy runtime before launch;
if projected reference cost exceeds the cap, keep the run bounded and report
incompleteness rather than silently substitute a faster target or extend budget.
No environment installs, no public defaults, no unrelated processes killed.

Supervisor command (written during implementation):
`python -m docs.benchmarks.walnutpie_gpu_tf.replication --output ROOT --draws 1000000`.
Fresh output directories preserve previous results. Source and runtime manifests
record actual commands, Git commit, settings, seeds, threads, timings and device.
Terminal result: ROOT/comparison.json and ROOT/REPORT.md, linked from a result note.

## Skeptical pre-execution review

Reviewed before implementation: previous timing used an analytic substitute,
million-draw JSON would create excessive host memory, old launcher forbids
concurrency and draws >8192, warmup resets would change the experiment, and
concurrent timings cannot establish isolated speed. The new plan repairs those
issues with literal source evaluation, binary chunks, explicit state/RNG
continuation tests, a separate supervisor and no ranking claim. Frozen identity
replication is separated from optional mass adaptation work. Source cap10
cannot reproduce the separate cap30 extreme-start experiment. All remaining
convenience choices are labeled; none changes the author transition kernel.
Audit passes subject to the specified preflight tests and runtime feasibility.

Preflight review update: the installed GPU probe only supports GPUs 0/1 and
both have other research processes. Trusted NVIDIA inventory also finds GPU 2
(same RTX 4080 SUPER hardware class), with only the remote-desktop daemon and
over 31 GiB free. Use GPU 2 with verified TF growth and record sharing; do not
kill other processes. Eight focused source/continuation/artifact tests pass
(17.75 seconds). A separate four-method 64-draw concurrent mechanics smoke
precedes the million-draw launch. It has no convergence interpretation.

Attempt 01: four 64-draw methods launched together. The three CPU arms passed;
GPU numerical sampling completed but checkpoint serialization attempted a
CPU-only op under explicit GPU placement. Repair: explicitly place TensorFlow
serialization on CPU outside the numerical graph. No scientific method changed.
Attempt 02: four concurrent 1024-draw methods all completed after that repair.
Literal warmup produced h=.3666358310669994, delta=.19520158705471408. The
slowest original-author chain took 47.61 seconds; proportional extrapolation
is 12.92 hours/million, versus corrected-author 8.05h, TF GPU 6.78h, TF CPU
0.69h. These startup-sensitive short-run estimates are descriptive. The prior
two-hour reference estimate used an analytic substitute and was not applicable
to literal scipy execution. The owner explicitly approved a 16-hour per-method
limit on October 29 after seeing this cost correction. That approval replaces
the initial 8-hour ceiling. Long methods execute concurrently, not sequentially.

Before launch, the 34 CPU metric/source regressions, two additional dense
algebra/runtime-data tests and four reference harness tests passed. Store all
eleven coordinates rather than the script's generated (v,x1) pair; the focused
projection test confirms this reporting change leaves the chain/RNG/diagnostics
unchanged. Full-source snapshots include the numerical diagnostic dependencies.
No exact published-figure, extreme-start, statistical equivalence, or isolated
speed claim follows merely from launching or completing the campaign.
