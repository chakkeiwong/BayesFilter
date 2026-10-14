# Controlled-SMC random-input sensitivity experiment, 6 October 2026

Question: can 1e-8 changes to the native random inputs explain the observed
author/TF likelihood differences, including recursive fitting? The owner has
authorized this experiment. Replication only: retain author settings and failures;
do not repair the author or add precision protections. KDM is outside this test.

## Frozen design and evidence contract

Use the six saved Lorenz cases (F=2.5,4.8801,8.5; seeds 20260929/30), d=8,
N=512, T=100, I<=4, quadratic fitting, adaptive stop after a refinement when
minimum ESS/N>0.9. Fix the original dataset and APF initial policy. The unchanged
author cSMC_Resample.m is the comparator, using its pinned Octave dependencies.
Source anchors: cSMC_VaryF.m:1-79; cSMC_Resample.m:185-193 (normal equations),
222-231 (backward recursion), 239-264 (adaptive control); the published controlled
SMC paper, sections 4.2 and 6.2, supplies the method context. This is a test of the
actual author implementation, not an accuracy claim about an exact likelihood.

For each case run an unperturbed control and four independent Rademacher
directions (seeds 2026100601 through 2026100604), each with both signs, for
normal-only, uniform-only, and combined perturbations of absolute size 1e-8.
For direction zero also run both signs of combined 1e-10 and 1e-6. Total: 29
variants per case, 174 source runs and 348 port runs (two existing solvers).
Each perturbed run evolves its own clouds and fits and applies its own native
adaptive stop. Preserve a five-stage random tape so different stop counts do
not misalign subsequent draws. Perturb the base systematic uniform U, not U/N.
Reflect uniforms into [0,1], with endpoint protection; count boundary reflections.
Normal and uniform direction streams are generated independently of the author
streams. The zero control must exactly reproduce saved author outputs and RNG.

Compare both existing TF float64/GPU/XLA fits: source_normal_equations and
centered_qr. No solver changes, damping, clipping of policies, or ridge. Fit and
forward use the real shared endpoints and stable signatures; only experiment
orchestration and native adaptive stopping are host loops. Record trace counts.

Primary diagnostic: S_A=abs(ell_A(u+delta)-ell_A(u)), corresponding S_T,
and D=abs(ell_T(u)-ell_A(u)) at each implementation's final adaptive stage.
Report signed changes and 100*expm1(delta ell), separately by case, arm, and
epsilon. The predeclared typical scale is the median of the eight absolute
responses at 1e-8; ranges and fraction exceeding D are explanatory. Never use
the largest perturbation as an acceptance tolerance. Four independent direction
pairs are insufficient to estimate tails or establish statistical non-inferiority.

Qualification here means only 'compatible with the measured typical perturbation
scale' if D <= median S_A for that case/arm. Additional TF failures on valid
source runs veto current-port acceptance regardless of that scale. Also report
S_T/S_A, stopping differences, ancestry changes, and paired discrepancies.
This experiment cannot establish accuracy relative to the unknown exact
likelihood, absence of systematic bias, full replication, or solver superiority.
Previously reported conditional-refit errors are secondary context; autonomous
port errors are the matched primary comparison.

## Intent, assumptions, and skeptical audit

Mechanism: ill-conditioned fitting may amplify tiny input changes, while
resampling is discontinuous. Sensitivity is explanatory; additional invalid
port results are a promotion veto and motivate the existing solver repair.
Zero-control disagreement, altered data/source, tape mismatch, or invalid
artifacts invalidate the experiment and stop that phase. A candidate failure
does not stop other prespecified valid runs. Time limits stop execution and
retain partial evidence; incomplete comparisons cannot pass.

The author settings are source baselines. 1e-8 comes from the owner and is a
calibration hypothesis, not floating-point machine epsilon. Rademacher directions
are a convenience design with constant coordinatewise perturbation magnitude;
four directions are a bounded pilot, not a tail sample. Larger/smaller epsilon
checks expose threshold dependence. The six existing cases are calibration
cases and provide no untouched validation. All uniforms/draw counts are checked.
The unprotected TF fits are comparison baselines, not numerical safety defaults.

Audit: using a final-forward-only perturbation would miss the fitting problem;
the full recursion corrects that flaw. Pooling refinement stages or comparing
percent changes in logZ would misstate the likelihood effect; retain stage
labels and use expm1 of log differences. Cheap adversaries are unchanged replay,
the APF initial stage, and normal-only/uniform-only arms, evaluated within each
forcing/seed. No performance or scientific-improvement claim is being tested,
so a separate heuristic-method leaderboard would not answer this question.
Audit disposition: suitable for a bounded sensitivity diagnosis, not automatic
relaxation of the final replication criterion.

## Execution and budget

Dispatcher phases p0_perturb_source and p0_perturb_tf, via the existing approved
Python interpreter and scripts/run_controlled_smc_kdm.py run --phase NAME.
Source: CPU only, CUDA hidden, one BLAS thread, 1200 seconds per attempt,
1500 total. TF: trusted GPU, memory growth verified before initialization,
float64, TF32 off, XLA on, 1800 seconds per attempt, initially 2100 total (2750 after the recorded runtime revision). Two attempts per
phase allow a localized harness retry. Total GPU allocation remains the existing
2812.436806963349 seconds; no new compute is added. Artifacts use new numbered
attempt directories under artifacts/controlled-smc-kdm-replication-20260928.
Preserve source snapshots, hashes, tapes/directions, per-run summaries, logs,
manifest and resource charges. Result: controlled-smc-perturbation-result-20261006.md.

## Runtime and compatibility repair

Attempt152 stopped before filtering because the pinned Octave lacks JSON support.
Attempt153 uses MAT bookkeeping; all174 author runs and six exact controls pass.
Observed TF full-recursion costs exceed the initial estimate. Allocate at most
2750 GPU seconds from the unchanged2812.4368-second balance, retaining the
1800-second first-attempt cap and two-attempt limit. A continuation may reuse
verified completed rows with identical core/source hashes; replay the interrupted
case zero control to reconstruct ancestry diagnostics. This changes scheduling
only, not the design, seeds, methods, tolerances or sample count.

Completed: attempts152-155, result controlled-smc-perturbation-result-20261006.md.
174 source and348 unique TF trials; no additional TF failures. Typical source-scale
compatibility holds for three of six cases per solver; full acceptance remains open.
CPU299.169712 seconds, GPU1938.990620 seconds charged; no workers remain active.
