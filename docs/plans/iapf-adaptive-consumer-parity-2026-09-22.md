# Actual adaptive iAPF consumer: R/TensorFlow comparison

Status: COMPLETE, 2026-09-22; terminal skeptical review passed for the limited
conformance claim. Result: `artifacts/iapf-adaptive-consumer-20260922-01/result.md`.
This is the next dependency of
`younis-kdm-score-master-program-2026-09-14.md`, within the owner's renewed
48 CPU-hour / 48 GPU-hour allocation. It is an implementation/reference
comparison, not a new paper-replication or default-promotion campaign.

## Question and evidence contract

Does `score_study.iapf_adapter.execute_iapf` implement its declared adaptive
diagonal-density-fitting program correctly for multivariate linear models?
The kernels accept general dimensions, but the adapter currently rejects
everything except d=o=1. Component agreement alone cannot justify removing
this restriction.

The exact comparator is a new independent R reconstruction of the **actual
local program**: the six-parameter model in `gaussian_tf.parameterized_model`,
particle initialization at X0, propagation before the first observation,
resampling at every step including initial lookahead, diagonal Gaussian plus
constant guide, profiled density least squares, the same explicitly bounded
local projected solver, sample-CV stopping and particle doubling, and an
independent final stream. The existing frozen paper reference supplies only
checked Gaussian primitives. It and the author-choice files are not edited.
R reconstructs each filter cloud and fits its own guides; simply evaluating
TF-fitted coefficients in R is insufficient for the full comparison.

Primary pass: the actual adapter reaches the same actions/counts/stopping
iteration, fitted coefficients and final finite value as R on the declared
fixtures. Before comparing discrete decisions, record continuous errors and
choice margins. FP64 tolerances: filter values/clouds 1e-7 absolute plus 1e-8
relative; fitted coefficients/floors 2e-5 absolute plus 1e-6 relative (local
optimizer stopping sensitivity); final frozen-guide fixed-label score agrees
with two-step R finite differences to 2e-5 absolute plus 2e-5 relative wherever
labels are stable. The latter is a finite-program derivative, not a marginal
model score. Matching cap/convergence failures are valid failure-path evidence
but do not establish a successful multivariate adaptive consumer.

Hard implementation vetoes: differing mathematical inputs, invalid covariance,
nonfinite accepted quantities, objective/gradient mismatch, unaccounted model
or resampling differences, different actions unexplained by numerical boundary
proximity, scalar/row-mapped fallback, or missing actual-consumer evidence.
Solver failure is a repair trigger; do not remove convergence checks to pass.
An unstable finite-difference label is an explanatory diagnostic and blocks
that derivative check; it does not establish a wrong filter value.

This plan does not establish original-author solver identity, paper-scale
replication, likelihood unbiasedness empirically, finite-N score unbiasedness,
statistical superiority, default readiness, KDM validity, canonical LEDH
validity, or HMC readiness. FP64 GPU evidence cannot clear the remaining strict
TF32 veto from the preceding campaign.

## Source and mathematical boundaries

The preceding campaign's `source-and-math-audit.md` inspected GJL (2017)
Propositions 1–3, Algorithms 3–5, Sections 3.3–3.4 and 5.1–5.2, Equations
15–16 and the variance appendix. Local copy:
`.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf`.
The paper's study has X1 initialization, its specified coupled linear model,
adaptive resampling and retained weights. This comparison deliberately matches
the different local consumer and cannot close those paper-replication gaps.
The author's original numerical solver and floor remain unrecovered.

For fixed target samples b, Equation 15 profiles lambda as (p'b)/(b'b).
Its gradient is 2 mean[(p-lambda b) dp/deta]; the scale derivative cancels
because b'(p-lambda b)=0. The new R calculation uses this direct expression,
checked against finite differences, independently of TF's log-scaled algebra.
The projected optimizer is matched only to identify the implemented program;
its residual and active bounds are reported, not treated as statistical quality.
R computes proposal moments in precision form, independently of the repaired
TF gain expression. Float literals first rounded by existing TF `tf.cast`
calls are matched explicitly; no unnoticed literal differences may masquerade
as algorithm errors.

## Execution and repair sequence

1. Save exact pre-edit sources and source hashes. Run small CPU/FP64 reference
   fixtures for the multivariate density fit and its analytical gradient at
   d=2 and 5, including an exactly representable interior diagonal Gaussian.
   Check the actual recursive fitting factory's shape and convergence behavior.
2. If those pass, provisionally remove the adapter's scalar restriction in the
   working tree, keeping nonlinear scalar restrictions and claim/tuning checks.
   Retain the change only after actual multivariate adapter tests pass. Instrument
   the existing factories externally in the diagnostic harness to preserve every
   input, output, random stream, fit diagnostic and call identity. No forked
   implementation is allowed inside the adapter.
3. Execute paired CPU/FP64/XLA and independent R full reconstructions at
   (d,o)=(1,1),(2,1),(2,2),(5,3),(5,5), with two fixed seeds (41 and 42).
   Use T=4, initial N=64, max N=512, k=2, tau=.5, max 12 iterations;
   fitting mean bound 4, SD bounds [.2,4], max 2000 steps/30 backtracks,
   tolerance 1e-7, floor ratio .01, density_l2 objective. Final theta equals
   the declared nominal six-vector [.55, log(.6), log(.7), 1, .2, log(.8)].
   Data come from this actual model, with separate stateless data seeds.
   Preserve failures before any focused repair. At most two localized repair
   attempts per distinct failure; if this is insufficient, diagnose rather
   than silently widen the experiment or thresholds.
4. If CPU/R passes, repeat the actual consumer on the selected GPU with FP64,
   XLA and verified memory growth. Save actual device/build provenance and
   actual draws; GPU and CPU need not produce identical normal draws from
   an integer seed. Each is compared with R using its own realized inputs.
   Add focused regression tests for the actual consumer/call chain and boundary
   behavior. Close with a result/decision table, inference-status table, source
   hashes, commands and charged times; refresh the master and checkpoint.

Any discovered objective, stopping or casting defect gets a minimal documented
repair and a focused regression before continuation. Changes to density_l2,
the guide family or frozen scientific settings require a separately declared
comparison arm; do not silently swap in log regression or relative_shape.

## Assumptions, defaults and adversaries

| Choice | Provenance/status | Failure risk | Early diagnostic |
|---|---|---|---|
| Six-parameter model and X0 timing | Existing local consumer, comparator definition | Wrong paper object | R model matrices and initial lookahead |
| Every-step resampling and diagonal guide | Existing local consumer | Incorrect retained-weight or covariance semantics | Full clouds/labels and precision-form moments |
| Box, step/backtracking rules, tolerance, floor | Existing bounded adapter/test controls; hypotheses for this check | Local minima, density shrinkage, false convergence | Objective finite differences, residuals, active bounds, gradient norms |
| d/o fixtures, T4/N64, two seeds | Engineering coverage convenience | No high-dimensional/statistical power | Explicit restriction to conformance; includes rectangular observations |
| CPU FP64/XLA | Independent-reference exception | Missed production-device error | Actual GPU FP64 repeat |
| Sample CV and count rule | Existing adapter, not recovered author solver | Off-by-one/doubling mismatch | Independent R decisions from complete histories |
| Matching draws and own R guides | Deterministic comparison design | Replay that hides fitting mistakes | Compare R-propagated clouds and independently refitted coefficients |

Constructed cheap adversaries are the constant-guide/BPF identity (remove all
twisting), the observation-only Gaussian guide (omit future recursion), and
the exact full-Gaussian future guide/Kalman value (where analytic). They were
implemented in the preceding reference work. Here constant-guide equality and
exact known Gaussian fitting are structural veto checks; they are not tuning
targets. Likelihood-accuracy rankings against these methods are deliberately
outside this implementation-only comparison. Conditional situations are
scalar versus multivariate, full versus rectangular observation, interior
Gaussian fit versus poor/boundary fit, and CPU versus GPU. No method promotion
is allowed from pooled average error or from passing these fixtures.

## Budget, artifacts and commands

Start balance from `artifacts/iapf-renewed-mechanism-20260922-01/budget.json`:
CPU 165341.31453889224 seconds, GPU 172503.16047407698 seconds remaining.
Suballocation cap: 4 CPU worker-hours and 2 GPU process-hours; at most 12
launches including repairs, with one process/one BLAS thread for CPU checks.
All failures and verification time are charged. Stop on exhausted suballocation,
unexplained invalid evidence, or a target/permission boundary; a rejected
candidate alone triggers the specified diagnosis rather than stopping research.

New evidence root `docs/plans/artifacts/iapf-adaptive-consumer-20260922-01/`;
fresh attempt directories, never overwritten. The preceding campaign manifest
preserves a master-plan snapshot; this master refresh does not rewrite its
closed evidence. Initial/final source snapshots bind current dirty work without
claiming that HEAD alone describes it.

Python: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.
CPU: `CUDA_VISIBLE_DEVICES=-1`, single-thread BLAS/TF; R `--vanilla`.
GPU: escalated command, `CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, repository memory-policy helper before tensors.
Exact harness commands and elapsed times go in each attempt manifest.

## Skeptical pre-run review

PASS for the bounded conformance question. The wrong-baseline risk is addressed
by explicitly reconstructing the local program, retaining the separate paper
reference and listing the semantic differences. A projected-gradient pass is
not promoted to a good filter. Full consumer wiring and independent recursive
fits are required, preventing component-only evidence. CPU success cannot
substitute for GPU verification. Discrete-choice boundaries and optimizer
stopping can amplify rounding, so both continuous differences and labels are
saved before assigning blame. No threshold, optimizer or cap may be tuned to
obtain parity without a recorded amendment and preserved original result.
The result may expose new defects even after earlier component tests passed.

## Execution record

Attempts 01 and 02: CPU FP64/XLA preflight passed 22/22 and 32/32 independent
R checks respectively. The second adds the required backward recursive factory
check before changing the guard. Known diagonal Gaussian centers/variances
recover to within 9.15e-7 at d2 and d5. Objective/gradient finite differences,
R/TF fitted guides and recursion all pass. Both processes and R time are
charged. No numerical correction was needed. The scalar adapter restriction
is provisionally replaced by positive integer dimension validation; nonlinear
scalar restrictions and claim/tuning verification remain in their existing
call chain. Next: the full consumer on the ten prespecified CPU fixtures.

Attempt 03: all ten actual CPU consumers completed. R independently matched
every cloud, fitted coefficient, count, action and stopping iteration; maximum
cloud error 1.88e-11 and fitted-center error 3.69e-10. The (d,o,seed)=(5,5,41)
case crosses discrete labels under the original finite-difference perturbations
1e-5 and 5e-6, so seven derivative checks fail their precondition/comparison.
This is a diagnostic failure, not evidence of a wrong analytical derivative.
Preserve the full 960-check table and its failures.

Reviewed local diagnostic repair: select the first pair of label-stable finite
differences from h=(1e-5,1e-6,1e-7,1e-8) and h/2. Selection depends only on
unchanged ancestor and mixture labels for all six directions, never score
agreement. Retain every attempted step and its numerical differences. Require
the same original derivative tolerances and two-step consistency after label
stability; if none is stable, the derivative remains unchecked. Re-run only
the independent R comparison against preserved TF inputs/outputs, in a new
attempt directory, then continue to the GPU comparison. This changes neither
the scientific program nor its settings or pass thresholds.

Attempt 04 independently rechecked the saved ten TF cases in R; all pass.
Only d5/o5/seed41 needs the next pair h=1e-6 and h/2. Both keep every label,
and the score discrepancy is 7.04e-9 versus the original jump-contaminated
0.111. The base minimum choice margin is 4.56e-7. The original failed table
remains untouched. GPU availability checked with escalated nvidia-smi: selected
4080 SUPER idle with 11 MiB in use; the 5080 belongs to another workload.
Proceed with the same ten cases on the selected GPU in FP64/XLA.

Attempt 05: all ten GPU FP64/XLA cases passed, 850/850 independent checks.
Together with attempt04 and the recursive preflight, final reference checks
are 1732/1732. Twenty-two focused and dependent regression tests pass, including
the actual public consumer at d2/o1 and d5/o3. Retain the positive-dimension
validation and removal of the scalar-only linear restriction. Result note
records limits, preserved failed diagnostic, decisions, inference status and
terminal skeptical review. No paper/default/TF32/model-score promotion follows.
