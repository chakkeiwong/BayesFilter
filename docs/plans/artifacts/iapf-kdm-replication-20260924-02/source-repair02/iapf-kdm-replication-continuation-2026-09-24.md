# Replication continuation: mathematical checks and cost measurement

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
