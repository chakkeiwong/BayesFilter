# Short q20 repair fitting canary

Authorization: the owner approved the proposed short fit test with “I agree.
do the test.” This authorizes the bounded experiment below. The inherited
campaign has 143975.28597232018 seconds remaining. Its old diagnostic allocation
has 7.711412891243526 seconds left; this new requested test receives a dedicated
1800-second diagnostic allowance **inside**, not in addition to, the remaining
campaign time. Record actual worker time in the inherited campaign ledger.
The 1800-second cap is a conservative operational ceiling, not a runtime forecast
or a statistical threshold. At most two infrastructure attempts share it.

## Question and evidence contract

Can small corrections to the frozen depth-four map learn useful shape on the
same four-parameter q20/T30 UKF approximate posterior in 64 Adam updates?
Separately, does the exact path gradient show less minibatch variability at
the same unchanged checkpoint? This is a fitting canary, not complete training,
hyperparameter optimization, an HMC experiment, or a production qualification.

Baseline: the actual depth-four endpoint, SHA-256
`63dcff1c3da92fb81095889d557fd78b63b106bd7373c6a4f066a88471cdcfa5`, candidate
`direct-w16-lr0.0005-r0`, in the earlier training-repair campaign's
`attempts/00005-continue-depth/worker/data/cohort-00000.json`.
Restore against `/tmp/BayesFilter-q20-training-repair-20260923-r1`, whose target
and checkpoint identities must still match. New optional code is loaded by
exact path and separately checksummed. Do not mix today's changed target code
with the saved baseline.

Arms use `F(C(z))`, where F is frozen:

1. Identity-initialized free diagonal affine correction (8 parameters).
2. A three-component scalar sigmoid mixture in zero-based coordinate 1,
   followed by the same free diagonal affine correction (17 parameters).
   Slopes start at one, offsets at (-1, 0, 1), and weights at 1/3. This
   symmetric, distinct-component initialization exposes nonlinear directions;
   it is not identity. Evaluate it separately before training so an initial
   shape change cannot masquerade as learning.

Train both with the standard exact reverse-KL estimator, fresh batch-normal
draws, and fresh Adam slots for the new parameters. F and its old optimizer
state are never updated. The path-gradient test uses the scalar arm's initial
checkpoint, paired on the same latent points and target evaluations. It does
not replace the training estimator in this canary.

| Role | Predeclared rule |
| --- | --- |
| Primary fit screen | Final expected RKL integrand is lower than both the arm's own initial map and the frozen baseline on a fresh 128-point heldout bank, with paired Monte Carlo intervals below zero |
| Uncertainty | Four planned loss contrasts (two arms x own initial/baseline), simultaneous approximate 95% normal intervals by Bonferroni; conditional on these fitted checkpoints, not repeated-training uncertainty |
| Promotion veto | Any invalid target/score, nonfinite state, wrong Jacobian/gradient, failed reconstruction, mutated base, or missing required heldout rows |
| Repair trigger | Loss inconclusive/worse, score error stays high, clipping activates, parameter motion too small, or useful nonlinear direction remains weak |
| Explanatory | Score norm/RMS and coordinate-2 RMS, centered log-density variation, update sizes, gradient variability and timing; they cannot alone establish convergence |
| Continuation veto | Invalid target/source/checkpoint, corrupted artifact, resource or platform boundary, or exhausted 1800-second allowance |

The loss is `mean(-log pi(F(C(z))) - log|J_(F o C)(z)|)`.
The omitted Gaussian base term is identical in paired differences. Measure
the score residual with the full change-of-variables derivative, including
the log-Jacobian score. No reuse of old target scores at moved points.
Inconclusive 64-step learning is not evidence against the architecture or
NeuTra. Neither this bank nor the eventual 1000-point procedure proves mode
coverage. There is no across-training-run ranking or default promotion.

The cheap adversaries are the unmodified saved map, the initialized but
untrained correction, and the trained affine correction for the scalar arm.
These construct the do-nothing, initialization-only, and linear-only alternatives.
Report descriptive conditional losses/score errors for z2 < -1, |z2| <= 1,
and z2 > 1 (one base-normal standard deviation; sparse tails are uncertain).
A detected regression against these controls blocks a quality promotion; the
canary itself cannot award that promotion. Do not tune on these heldout slices.

## Numerical choices, calibration, and implementation

| Choice | Provenance, rationale, risk and early check |
| --- | --- |
| 64 updates, B=32 | Owner-approved short tranche and inherited native batch size; may be too short; save every update, interpret failure as local evidence only |
| 128 heldout rows | Four complete batches of 32, low-cost screening choice; intervals may be wide; preserve paired row data and report inconclusive outcomes |
| Adam beta1=.9, beta2=.999, epsilon=1e-7 | Existing trainer values, warm-start hypotheses for new corrections; fresh slots avoid inherited bias; analytic Adam equivalence and state-rejection tests |
| Initial LR=.01, fallbacks .001/.0001 | Canary hypotheses allowing visible motion within 64 steps, not tuned defaults; use one separate 32-row calibration batch and accept the first finite, loss-decreasing trial step, restoring state after each trial |
| Clip norm = 10 x largest calibration norm | Local measured scale from 16 fresh B32 batches; factor ten is an explicit emergency-envelope hypothesis; record all norms and clipping events, never infer calibration from silence alone |
| 16 gradient batches | 512 rows, low-cost variance screen, no precision guarantee; paired standard/path gradients at one fixed scalar checkpoint, variability estimates descriptive |
| K=3, offsets -1/0/1 | Small symmetric nonlinear hypothesis on base-normal units; distinct components avoid affine-only first-order degeneracy; compare to own initialization and affine control |
| Inverse atol/rtol=1e-10, cap=100 | Float64 reconstruction check only; bisection halves a finite derived bracket; fail on unmet input/output tolerance, never used by RKL training |
| Float64, XLA, beta=1, native batch target | Inherited q20 target scope; use original strict factor-cached eigensolver and verify signatures; no scalar target fallback or pfor |
| Random streams | Deterministic domain-separated SHA-256 seeds from this plan name and role; calibration, training, gradient probe and heldout streams disjoint; in-graph fresh training noise |

Implement a bounded Adam adapter around the already-tested exact gradient
engine. It must reject invalid updates without changing parameters, slots, or
iteration, and preserve serializable endpoint parameters and optimizer state.
This is sufficient for frozen-base correction fitting; the full production
checkpoint/HMC codec integration is not a prerequisite for this test and is
not silently claimed complete. Validate it with focused analytic CPU/XLA tests
(GPUs deliberately hidden), including Adam formula and rollback/continuation.

GPU memory growth is required before import and verified before initialization.
Select an idle GPU using the existing inventory helper, record co-resident load,
and run trusted/escalated. Two small arms run sequentially to reuse target setup
and the heldout baseline; this avoids two redundant target compilations. The
numerical batches remain parallel TensorFlow/XLA operations. Report setup and
steady-state times separately; historical 40–41 min/1024 original updates are
only a rough comparator because frozen-base correction work differs.

## Skeptical pre-run review

The initial plan would be misleading if it used the tiny SGD correctness smoke
as the q20 optimizer, reused target scores after moving points, compared scalar
initialization changes with trained improvements, selected on heldout data, or
used decreased RKL as proof of Gaussian geometry. The above design removes
these faults. It also binds the executed target source, retains the frozen map,
and limits claims to paired fit changes for specific checkpoints. Remaining
risks are short training, one stream per arm, inherited Adam moments/epsilon,
an unoptimized small K, untested mode coverage, and a small heldout bank. These
are explicit hypotheses with diagnostics, not unexamined production defaults.
Review verdict: proceed with the focused adapter tests and bounded GPU canary.

## Commands and artifacts

Runner: `docs/benchmarks/diagnose_q20_mechanism_fit_2026_09_23.py`.
Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, with memory growth set before
framework import. Supervisor loads a frozen request, enforces the shared cap,
records each attempt, and charges actual worker wall time to both the dedicated
diagnostic allowance and the existing total campaign. GPU worker commands are
recorded verbatim in the manifest. Preserve all outputs under
`docs/plans/artifacts/q20-short-fit-canary-2026-09-23/run-01/`.
The result note is `docs/plans/bayesfilter-q20-short-fit-canary-results-2026-09-23.md`.
Record source and data identity, seeds, checkpoint and optimizer states,
per-update history, heldout rows, exact commands, environment, numerical/device
settings, wall time, accounting, and the interpretation/decision tables.

## Implementation check and launch

The focused suite passed 23 tests (seven Adam adapter cases plus the sixteen
existing mechanism cases). The first run's analytic Adam expectation omitted
the installed Keras implementation's float32 conversion of Python beta
constants in bias-correction powers. Inspecting `keras/src/optimizers/adam.py`
resolved the approximately 6.7e-8 first-step discrepancy; the final test checks
the installed formula at 1e-12. No numerical production default was changed.
The remaining controls are inherited optimizer hypotheses, not proof of optimality.

The cached-gradient probe must agree with the actual trainer at 1e-10 absolute/
relative tolerance, and cached physical points at 1e-12 scaled tolerance.
These float64 engineering checks guard reuse at unchanged points. Endpoint
parameter reconstruction must reproduce forward/logdet and inverse inputs within
1e-8; the inverse's own tighter 1e-10 absolute/relative stopping rule also applies.
The reconstructed map is diagnostic only and cannot enter HMC through this file.

Final code review checked both optimizer-slot rollback and candidate rejection
without scientific-direction rejection. The original target bridge is the sole
target callback; batch size 32 is static and all target calls remain native
batches. Noise is generated in a fixed-signature XLA graph. The old target tree,
new numerical module, runner, checkpoint, and request are separately recorded.
The installed trusted NVIDIA probe found GPU1 idle and GPUs0/2 occupied.
The two arms therefore share GPU1 sequentially, avoiding other workloads.

## One fresh confirmation after the 128-row canary

The completed pilot used 543.1404912448488 supervised worker seconds. Scalar
training reduced loss from its own initialization, but its paired difference
against the original map was -0.3100 with an interval spanning zero. Thus the
question of improvement over the actual baseline remains unresolved. The
scalar residual RMS fell from 4.7493 to 3.2693, making a fixed-checkpoint
confirmation more discriminating than another training run.

Before any new target evaluation, fix this follow-up: evaluate the original
map and the frozen 64-step scalar endpoint on **one new 512-row base-normal
bank**, in 16 batches of 32, with the separate `confirmation` seed namespace.
Do not train, alter hyperparameters, select a checkpoint, reuse the old bank,
or pool the two banks. The sole primary contrast is the mean scalar-minus-
baseline RKL integrand; use an approximate 95% paired normal interval (1.96
standard errors). Selection of this fixed candidate on the pilot is independent
of these new points. Score RMS and the same left/central/right latent slices
remain explanatory and can flag concerns, not establish posterior validity.
Stop after this one bank whether the interval passes or remains inconclusive.

512 is derived from the pilot SE=0.17146 at n=128: four times as many rows
halves the expected SE to about 0.0857, assuming the pilot variance transfers.
The observed effect is only a planning estimate, not a promised confirmation.
At the measured 2.4 seconds per batch, the two maps need roughly 77 seconds of
target/geometry work plus startup/compilation. A 360-second operational cap
uses the **same** 1800-second total canary allowance; no budget increase.
Output root: `docs/plans/artifacts/q20-short-fit-canary-2026-09-23/run-02/`.

Skeptical audit: the pilot's scalar initialization was not identity, so the
confirmation uses the original map directly. Fresh independent points prevent
optional expansion of the already-inspected bank from being treated as a fixed
sample test. The left latent slice's loss increased in the pilot; preserve this
concern even if the overall primary contrast passes. Its physical points differ
between maps, so it is not itself a posterior left-tail probability error.
Review verdict: proceed with this single bounded confirmation; no more sampling
after its result and no training-method or HMC promotion.

Completed: both arms and the independent confirmation finished, using 645.17
supervised worker seconds in total. See the
[result and terminal review](bayesfilter-q20-short-fit-canary-results-2026-09-23.md).
The confirmation supports lower RKL for this scalar endpoint while leaving a
conditional whitening concern. No more training or validation is running.
