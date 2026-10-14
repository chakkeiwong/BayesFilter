# Replication continuation: mathematical checks and cost measurement

## Active comparator-cost amendment, 2026-09-24

Continue B within the existing 12-launch, 1800 CPU-second, 900 GPU-second
tranche. At entry, seven launches, 25.488099 CPU seconds and 118.317269 GPU
seconds are charged; five launches remain. No full campaign is authorized.

Question: what source stages and provisional costs are required for the
prespecified TG-MDPF, IRG-MDPF and DIS-PF bearings comparators? First resolve
their actual author configuration includes and checkpoint dependencies without
importing a numerical framework. Then run at most one short GPU cost probe per
arm, with one repair launch reserved. Preserve the original model factory,
resampling implementation, optimizer, loss, batch sizes, N=25 and T=17. Use the
existing data/pretraining/full-training seeds 924011/924012/924013; any DIS
bandwidth-only stage uses 924014. These are reproducible cost-probe seeds, not
the final independent-trial seed policy. Each child remains capped at 300s.

The paper Section 6 explicitly trains discrete comparators on MSE before a
separate bandwidth-only likelihood fit. Therefore DIS must use its source MSE
configuration and its dependent bandwidth-fitting configuration, while TG/IRG
use their source NLL configurations. Short probes use one pretraining epoch
and two epochs per subsequent source stage. Source early stopping and scheduler
settings remain recorded. One loader worker, short duration and sparse timing
are diagnostic overrides; checkpoint lineage and loss/frozen-parameter masks
must be verified at the actual consumer.

Pass criterion: resolved dependencies and executed source endpoints agree with
the named method; source stages produce finite changed parameters, preserve
their intended frozen parameters, and emit timing/checkpoint records. A method
nonfinite failure is candidate evidence and must be distinguished from a
harness failure. A timeout yields a partial cost/lower bound, not a fabricated
epoch extrapolation. Missing/wrong source identity, corrupted data/checkpoints,
or exhausted budget veto further dependent execution. Timing is explanatory
only; no ranking, training-quality, paper-replication or TF/GPU/XLA admission
claim follows. Artifacts remain under the -02 continuation root in fresh
attempt directories.

Skeptical/default audit: reusing the MDPF NLL loss for DIS would be wrong for
the paper comparison. A directory name is not enough to identify an estimator;
check the model class and invoked resampler/gradient method. Shared seeds make
short cost probes reproducible but do not replace independent final repeats.
Short early epochs cannot bound late plotting, convergence, retries or optional
MSE fine-tuning. Preserve these unmeasured costs in the inventory. The existing
reference PyTorch/NumPy exception is unchanged, and no runtime default changes.

The owner's 2026-09-24 instruction, “if not, continue”, authorizes this bounded
continuation of the master. The completed 12-launch ledger is preserved. This
plan does not authorize the full paper experiments or transfer another
campaign's budget.

## Question and budget

Can the original MDPF/IWSG implementation satisfy matched-sample derivative
checks, what causes the unresolved iAPF local fit failure, and what does
unprofiled source training cost? Neither a short training run nor a solver
control can establish either paper replication.

Use the existing execution checkout and reference environments, with a new
output root `docs/plans/artifacts/iapf-kdm-replication-20260924-02` in that
checkout. Caps: 1,800 CPU worker seconds, 900 GPU worker seconds, 12 launches,
300 seconds per child, including failed attempts and repairs. Retain unused
budget; do not run redundant checks to consume it. CPU reference checks hide
GPUs. GPU cost probes run with escalated access. No package changes.

## Evidence contract and sequence

1. **K1, reference mechanics.** Follow the author resampling and subsequent
   weight normalization into an explicitly diagnostic loss. Compare values and
   derivatives against an independently expressed Gaussian/von-Mises full
   mixture with frozen samples and denominator. Cover overlapping/separated
   components, angle wrap, tiny weights, and the two distinct source epsilons.
   Use float64 analytical/autodiff and finite-difference checks, then float32
   numerical comparison. Match the actual source target before comparing it
   with the paper's unsmoothed target. A mismatch is a hard mechanics failure;
   a documented source/paper epsilon difference limits interpretation. No
   unbiasedness or full loss/training correctness follows from pointwise checks.
2. **I2a.1, local-fit diagnosis.** Audit the paper equation and saved inputs.
   Separate ten distinct fit inputs from duplicated records. Derive and inspect
   amplitude escape, initialization and local step failures. Any changed fit
   policy is a reconstruction hypothesis, must retain Equation (15), and cannot
   be promoted from small raw loss. Known Gaussian controls and fresh shape
   checks are required before a subsequent adaptive pilot. No full pilot is
   authorized if the fit/floor policy remains unresolved.
3. **B1, costs.** Add a source-training mode without pervasive Python profiling.
   Use sparse synchronized timing of setup, training, validation and persistence;
   preserve source batch/trajectory/particle sizes and architecture. Measure
   representative batches within the child timeout, including startup cost.
   Record components that remain unmeasured instead of extrapolating them as
   zero. A cost projection is descriptive, never a performance ranking or a
   full-run authorization.

## Assumptions, vetoes and review

The pinned author source and existing isolated compatibility patch are the
reference baseline, not an eligible TensorFlow runtime. Numerical tolerances
must be stated in each diagnostic before execution and checked on exact/simple
controls. Fixed fixtures expose mechanics, not Monte Carlo unbiasedness.
Short-run cost scaling is a hypothesis: compilation, loading, plotting,
validation, evaluation and early stopping can invalidate a linear projection.

Heuristic adversaries for these mechanics checks are the single-component
Gaussian score, an explicitly summed full mixture, and constant downstream
weights/losses. Conditional situations are overlap/separation, ordinary/tiny
weights, and angle-wrap/interior states. These are mathematical checks; no
learned-method ranking is attempted. The full experiments still owe their
conditional heuristic comparisons.

Stop a dependent phase on an invalid target, unexpected numerical failure,
corrupt input, unresolved fit policy or exhausted resource/attempt cap. A failed
candidate triggers the next declared diagnosis; it does not reject the whole
research direction. Do not change objective, source method or acceptance
criteria to make a check pass. Preserve commands, source/environment hashes,
seeds, elapsed time, results and failures in the new ledger and terminal note.

Skeptical review before implementation: the old training check was insufficient
for mathematical validity and its pervasive profiling made full-cost estimates
unreliable. This continuation addresses those exact gaps. It preserves the
original comparator and explicitly leaves full-paper statistics, fitted-policy
promotion, TF/GPU/XLA conformance and joint replication open. Recheck each
implemented diagnostic against this contract before launching it.

Implementation review before launch: source resampling (KDEParticleFilter:675–810),
weight normalization (:928–934), source KDE density (:230–268), KDE_NLL (:35–80),
and KDEComputer (:27–104) were inspected. The executable check uses the actual
source NLL, not a substituted objective, with independent mixture calculations.
The paper anchors are Younis–Sudderth Equations (14)–(15) and Guarniero et al.
Equations (15)–(16). Exact Gaussian and constant-loss controls are included.
Twelve existing dispatcher tests and Python/R syntax checks passed. B1 retains
5,000/1,000 trajectories, T=17, N=25 and source batch sizes; its epochs are
shortened to 1 pretraining/2 full-training. No unmeasured component is treated as
zero. Review passes for bounded diagnostics only.

## B1 evaluation cost amendment, before launch

Run the actual author `setup_evaluation` call chain for bearings-only, N=25,
T=150, all 5,000 source evaluation trajectories, batch 1,024 and six loader
workers. Preserve source NLL/RMSE, raw-output persistence and plots. Use the
short-trained checkpoint from this tranche's attempt03 solely for workload
measurement; generated data are diagnostic calibration data, never the final
scientific holdout. Source and checkpoint hashes are required. NumPy/PyTorch
remain explicitly independent author references.

Seed dataset generation with 924131. Source EvaluationBase computes a configured
random_seed but seeds RNGs with its default argument zero (lines 22–39); preserve
and report this behavior instead of silently repairing it. Initial model creation
uses 924132 before checkpoint loading. No training update occurs.

Pass requires finite source metric values, five full-sequence calls (745
filter transitions), recorded memory/timing and persisted source outputs. This
checks evaluation wiring and cost only; accuracy and full-training cost remain
unproved. Inclusive boundary timings must not be added to nested timings. A
timeout or resource failure is a repair trigger within the unchanged 300s child,
900s GPU and 12-attempt caps; no full campaign is launched.

Skeptical audit: source sequence length, particles and dataset/batch sizes match
the resolved evaluation configuration. The selected checkpoint is deliberately
undertrained and therefore cannot answer accuracy or method ranking. The call
chain, seed defect and persistence overhead are explicitly measured. This
amendment passes for bounded cost evidence only.

## K1 sequence-gradient amendment, before implementation

The fixed-sample endpoint checks passed but did not cover network-parameter
gradients through a sequence. Add a CPU-only independent-reference check of
the actual author FullSequenceTrainer.do_forward_pass at B=2,T=17,N=25, with
its sparse labels and truncated BPTT modulo four. Use three fixed sampling
seeds (924141–924143), one random initialization and the short-trained checkpoint.
This is gradient parity of the source's truncated estimator, not the total
gradient of an untruncated sequence or an unbiasedness claim.

Compare each source run with an identical model/RNG replay in which only KDE
log density and NLL are independently expressed using explicit Gaussian/von-Mises
mixtures and torch.special.i0e. Retain source epsilons and angular conventions.
No optimizer update. Preserve sampled locations, loss, used/unused parameter
names and per-parameter gradient errors. Require exactly matched sampled
locations, matching absent-gradient masks, finite nonzero gradient norm, scalar
loss tolerance 3e-5 + 3e-4 relative, per-coordinate gradient tolerance
3e-5 + 3e-4 relative and aggregate relative L2 gradient error <=3e-4.
These FP32 bounds extend the passed endpoint bounds; they are engineering
checks, not statistical promotion thresholds. Three seeds are diagnostic only.

Skeptical audit: using ordinary finite differences while redrawing/detaching the
denominator would test a different derivative. Replay compares the declared
source derivative directly; pointwise finite differences already covered fixed
denominators. The fixture uses the source forward method with explicitly
constructed trainer fields; constructor/optimizer wiring is supported separately
by training attempt03. No claim of independently verifying the whole model
architecture follows. A mismatch blocks this check and triggers diagnosis.
All work remains within the existing CPU/attempt budget.

K1 localized harness repair: attempt05 expected 16 density calls, but source
initialization emits do_resample=False (kde_particle_filter.py:671), and the
resampler bypasses the KDE at :708. There are 16 transitions and 15 density
calls. The repaired diagnostic asserts the complete [False, True x 15] schedule.
This changes an incorrect observer count, not the source estimator, tolerance,
scientific target or campaign budget. Attempt05 remains charged.

## K1b expectation amendment, before implementation

Check the expectation claimed by paper Equations (14)–(15) separately from the
finite-particle normalized source consumer. Use a diagnostic linear statistic,
f(z)=z_x, and the actual source resampler and weight-normalization methods with
constant measurement weights. This statistic has a known mixture-mean derivative;
it does not replace the training NLL or establish its expected gradient.

Let q be the actual unsmoothed sampler, s_e the epsilon-smoothed source density
score, c=1+1e-8 the source injected weight at the reference parameter, and N the
number of draws. The unnormalized statistic has mean gradient E_q[f s_e]. The
normalized consumer has mean gradient (N-1)/(N c) Cov_q(f,s_e). For a matched
proposal and exact score, the latter is (N-1)/(N c) times the mixture-mean
derivative. Thus source normalization is not covered by a blanket unbiasedness
claim for the paper's unnormalized estimator. Derive this explicitly in the
result note; do not classify a predicted difference as implementation failure.

Use CPU FP64 independent-reference arithmetic, one thread, eight seeds
924151–924158, 1,024 independent clouds per seed, and N=2 and N=25. Construct
two-component Gaussian mixtures by repeating identical left/right components
in the actual N-component source representation. Cover overlapping and
separated means, with identical independent second-coordinate/angular kernels
so they cancel in the mixture-weight score. Compare the actual source Monte
Carlo gradient with independent 64/128-point Gaussian quadrature predictions.
Controls are the exact mixture mean derivative, zero gradient for a normalized
constant statistic, and the explicit full-mixture score on the sampled draws.
Quadrature agreement must be within 1e-13 + 1e-7 relative; sampled algebra within
1e-10 + 1e-8 relative; expectation checks within six Monte Carlo standard errors
plus quadrature error and 1e-12 rounding allowance. Preserve the per-seed means,
cloud-level variances, approximate 95% intervals, and all declared tolerances.

A rare-weight separated case (right mass 1e-10) uses quadrature only: the current
budget cannot resolve its rare events with Monte Carlo. Record that limitation
explicitly. This is an epsilon sensitivity check, not a new sampling default.
No optimizer updates, learned-method ranking, whole-loss unbiasedness claim,
algorithm alteration, full-paper inference or TensorFlow admission is allowed.

Skeptical/default audit: source particle count determines draw count, so a
two-component fixture alone cannot claim N=25 behavior; repeated identical
components preserve the target while exposing its actual N-dependent epsilon.
Gaussian quadrature supplies an independent expectation comparator and exact
Gaussian moments check its scaling. Fixed-fixture parity is insufficient for
this question. Source BPTT is held outside this one-step expectation experiment.
Any unexpected algebra/quadrature/nonfinite failure vetoes interpretation and
triggers repair under the unchanged remaining CPU/attempt budget.

## Comparator loader repair, before the reserved retry

Attempt08 resolved all four source configurations (PASS, 0.666274 CPU s).
Attempts09-11 failed during import before training: the new job names were
omitted from the existing reference-library allow list, so Matplotlib loaded
system libstdc++ without CXXABI_1.3.15. This is a local dispatcher defect, not
an author-method rejection. All three failures remain charged (9.278338 GPU s).
The loader policy now covers every actual author-interpreter dispatch, with an
executable wiring regression. A classified shared library failure stops the
remaining dependent reference jobs; ordinary candidate failures still allow
independent diagnostics to continue.

Use the single remaining launch for DIS-PF because its two different objectives,
checkpoint transition and frozen-parameter mask remain unexercised. TG/IRG timing
remains unmeasured. First run the focused CPU harness and CPU-only import check
under the actual reference environment. No method, tolerance, dataset, hardware
or campaign cap changes. Attempt12 retains the 300s child cap.
