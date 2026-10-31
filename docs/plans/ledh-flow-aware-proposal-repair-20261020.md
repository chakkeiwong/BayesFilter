# Flow-aware proposal repair: 20 October 2026

## Active question and scope

Does expressing cSMC/KDM guidance in the physical, post-flow state remove the
verified P-versus-Q proposal mismatch, and does simulated-label training then
reduce terminal filtering-moment risk on fresh SIR observations?

Worktree: `.localresources/worktrees/ledh-iapf-execution-20260922`;
branch `ledh-refactor-with-policy-fix`; base `4d6915cc6`.
This is a diagnostic research candidate, not canonical LEDH admission or a
default change. Historical LEDH results remain invalid for current claims.
The prior failed fixture is a regression only, never tuning/efficacy data.

## Mechanism and evidence contract

- Keep the shared UKF lifecycle, affine LEDH solver, GenUT correction and
  Contract E--Chol reset. Derive output guide moments from the Gaussian
  current-observation update, then apply the existing future-only quadratic
  cSMC or KDM mixture-product moment conditioner in those coordinates.
- The affine preimage of output moments has covariance B^-1 V B^-T. Its
  additional information need not be PSD. Use the actual SPD output Gaussian,
  not a PSD projection of this preimage. Differentiate every moving quantity.
- A flow-independent output target pulled back and flowed cancels LEDH.
  Therefore beta=1 is an exact-conditional engineering comparator (for linear
  Gaussian observations), not evidence that LEDH was augmented.
- Genuine augmentation uses the density mixture
  (1-beta) q_LEDH + beta q_output with beta=0.5 fixed offline. Independent
  uniform labels choose one component. Evaluate both densities at the chosen
  point; never average particle positions. Labels and beta are theta-independent.
- Primary engineering criterion: density/change-of-variable and total-tangent
  identities, shared call-chain wiring, beta=0 legacy parity, beta=1 cancellation,
  and an integrated formerly failing fixture with all required validity flags.
- Primary scientific criterion: paired terminal first/raw-second-moment
  simulated-label risk on untouched data, with dataset-cluster bootstrap 95%
  intervals. Score labels, ESS, guide-fit error and runtime explain only; they
  neither train nor select candidates. The simulator loss identity needs
  independent particle/fit randomness conditional on the observations.
- Promotion vetoes: any required validity failure; conditional underperformance
  against a cheap heuristic; no heldout uncertainty supporting improvement.
  A candidate failure triggers the next planned repair, not direction rejection.
- Continuation vetoes: invalid baseline/comparator for a claimed comparison,
  broken target/density identity, corrupted data/artifact, or exhausted budget.
- No posterior, HMC, canonical, cross-model, default-readiness or score-accuracy
  claim follows from passing these checks. A small training search remains a
  scoped pilot, not proof of adequate neural capacity or global optimization.

## Comparator and sanity-check construction

Evaluate low/high terminal-observation regimes using a threshold frozen from
calibration, as well as all datasets. Paired particle seeds are shared by arms.

1. Plain shared LEDH: isolates any proposal repair from other algorithm changes.
2. Exact conditional proposal, no future guide, beta=1: strong cheap per-parent
   comparator, also the cancellation control; differing parents retain differing
   predictive-likelihood weights.
3. Half-mixture, no future guide: isolates geometry from cSMC/KDM learning.
4. Learned cSMC and learned KDM half-mixtures, plus their untrained adapters:
   isolate training from representation and from simple conditional guidance.
5. Prediction-only Gaussian moments: ignores data and exposes gross failures.
6. Observation-only correction of the prediction: the simplest data-using rule.
7. Recursive Gaussian assumed-density moments: a cheap classical filtering
   adversary. These last three are moment references, not LEDH variants.

The heuristic set is a falsification instrument, never an optimization target.
Missing heuristic evidence prevents promotion; it does not invalidate a unit
test or prevent a localized repair. Report a machine-readable verdict.

## Defaults and assumptions audited before execution

| Choice | Provenance/status | Risk and earliest check |
|---|---|---|
| SIR d18, T5, N64, FP64 GPU/XLA, TF32 off | failed-scope diagnostic comparator | no transfer to production FP32; record exact scope |
| OT epsilon1, Sinkhorn5, balance32, K=N64 | owner-frozen diagnostic scope | no retuning or claim-bearing admission; verify unchanged |
| UKF P and actual transition Q | canonical architecture | never substitute Q for P; call-chain regression |
| GenUT/reset/damping/ridge controls | frozen comparison arm, not safety endorsement | emit every validity flag; no silent numerical alteration |
| Linearized current-observation Gaussian | exact only for linear Gaussian observations | exact SIR identity and nonlinear nonclaim in document |
| Future quadratic/KDM messages | inspected future-only SIRGuideFitter export | current observation must not be counted twice |
| beta0.5, independent labels | new mechanistic hypothesis | full-mixture density/FD test; compare no-guide half-mixture |
| Pooled/time-specific 3-coefficient adapters | existing capacity hypotheses | target-specific capacity comparison, two optimization seeds |
| Paired random-direction descent | existing optimizer hypothesis | two step sizes, budget ladder, compare initial validation risk |
| Simulated terminal-state labels | population squared-risk identity | independent U, fixed calibration, disjoint datasets |

Fresh seeds: calibration 263020001; training 263021001--263021032;
validation 263022001--263022008; holdout 263023001--263023032.
Fit seeds add 1,000,000; particle seeds use separate 275/276 million ranges.
Never use scores to choose an adapter. Freeze selection before opening holdout.

## Phases, commands and bounded resources

Output root: `docs/plans/artifacts/ledh-flow-aware-proposal-repair-20261020/`.
Preserve baseline TeX/PDF there. All launches use new numbered subdirectories.
Python: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.

1. Extend the manuscript with propositions/proofs, cancellation, analytical
   tangents, full-mixture weights and end-to-end pseudocode. Targeted MathDevMCP
   checks are advisory evidence, not mathematical certification.
2. Implement an optional physical-proposal interface in the existing executor,
   not a duplicate flow or covariance lifecycle. CPU reference tests explicitly
   set CUDA_VISIBLE_DEVICES=-1. Check all claim-bearing call chains executably.
3. Escalated GPU/XLA integration smoke: old failure only as regression, healthy
   FD checks and bitwise legacy parity. Maximum 900 seconds.
4. Fresh training/validation/holdout pilot. Per family: pooled/time-specific,
   two optimizer seeds, step sizes 0.05/0.15, eight updates; validation nominates
   one configuration for a 24-update extension. Minibatches contain two fresh
   simulated datasets; validation uses two particle replications. Freeze before
   32-dataset/two-particle-replication holdout. Maximum 5,100 seconds; checkpoint
   on timeout without opening holdout if training protocol is incomplete.
5. Reserve one localized-repair launch, at most 800 seconds, without changing
   the scientific contract. Total maximum 6,800 GPU-process seconds and three
   launches, below the inherited remaining 6,800.396892893 seconds/three launches.
   Campaign remaining starts at 26,862.354012593 seconds. Charge failed attempts.
6. Build PDF twice, inspect new rendered pages and retained equations. Write
   result/manifest with actual commands, source hashes, git commit, environment,
   GPU memory growth, seeds, runtime, decision and inference-status tables.
   Update the active checkpoint and master-plan pointer.

Commands are recorded verbatim in each launch manifest. Required environment:
TF_FORCE_GPU_ALLOW_GROWTH=true; visible GPU0; no pfor, no NumPy runtime; XLA on.
GPU probes and jobs require escalated permissions. A timing preflight may reduce
only the pilot's declared claim strength, not silently truncate its protocol.

## Skeptical review before implementation: PASS with explicit limits

The previous idea was incomplete because a pre-flow PSD guide was fitted to a
physical-state target. The repair now tests the actual consumer, not just an
available helper. Exact-conditional cancellation is an explicit control, full
mixture denominators and theta-independent labels protect the derivative claim,
and no heldout score is an optimization target. New partitions avoid reuse of
the failed holdout. Equal data/particle streams and identical reset/correction
controls isolate the proposal. Small adapter searches cannot establish general
training adequacy: if they do not finish, report under-budgeted, not promoted.
Expected candidate failure is a repair trigger. Only the stated validity and
budget conditions stop execution. Independent review will inspect the amended
math/implementation; unavailable review does not create an execution gate.

## Execution record

Documentation and shared-executor implementation complete. CPU reference suite:
37 passed in62.99 seconds. Initial two failures were test-contract errors:
graph versus eager bitwise comparison differed by5.55e-17, and a traced loop
body was incorrectly expected to call Python twice. Corrected comparisons use
the same execution lane and verify the shared traced body plus both time steps.
The first attempted rerun used a wrong test path and ran no tests; preserved.

GPU integration-01:115.110395934 seconds,46 endpoint calls. The known failed
plain-LEDH fixture remains invalid, while conditional, half-mixture, cSMC and
KDM physical proposals all pass on that fixture and on the healthy fixture.
All12 healthy analytical/FD checks pass; maximum relative error8.45871e-8.
These fixtures are engineering regression evidence, not scientific ranking.
Remaining stage budget6685.286496959 seconds/two launches; training cap5100
and repair reserve800 remain unchanged. MathDevMCP requested explicit SPD
conditions around the conditional update; added Q,R positive definite.

Pre-training skeptical re-review: PASS. Fresh partitions and selection-before-
holdout are explicit. The inherited exposed failure is expected invalid-baseline
evidence, not a new efficacy observation. On fresh data an invalid comparator
vetoes every numerical ranking against that comparator; it does not erase a
valid independently specified comparison against the conditional or half-mixture
control. No invalid loss is averaged into an apparent improvement. The three
Gaussian heuristic estimators use spherical-radial cubature only as explicitly
distinct diagnostic references, never as alternative LEDH implementations.
Bootstrap intervals are pointwise, not simultaneous. Conditional heuristic
underperformance is a conservative promotion veto, not a statistically proven
ranking or an optimization objective. Holdout starts only after selection is
frozen.

During training, the scoped math/code audit verified the shared consumer chain
and identified no additional density or derivative mismatch. MathDevMCP's
bounded algebra engine returned inconclusive results; its code comparison is
lexical and reported missing LaTeX tokens, not a semantic counterexample.
These limitations and the algebra-to-code mapping are preserved in
`artifacts/ledh-flow-aware-proposal-repair-20261020/math-code-audit.md`.
The document now spells out conditional-moment derivatives and changing
mixture-product weights. A second focused CPU run passes14 tests, including
six new dynamic-mixture, invalid-uniform and reporting checks (43 distinct
tests across the two successful runs). Synthetic reporting checks verify that
invalid baselines cannot produce rankings and conditional heuristic losses
cannot be concealed by a favorable pooled average. No runtime code changed
during the active GPU training process.

Terminal execution: training and all32 independent datasets/two replications
completed in2501.845760758 GPU-process seconds,2474 endpoint calls. All544
holdout records are valid, including plain LEDH. The selections were frozen at
2079.719240449 seconds: cSMC pooled, seed263025002, step.15,8+24 cumulative
updates; KDM time-specific, same seed/step,8 updates. All eight recorded runtime
source hashes still match. Peak TensorFlow allocator use122725632 bytes.

Terminal skeptical review: engineering repair PASS on the stated fixtures;
learned-candidate promotion FAIL. Both trained guides lose to the recursive
Gaussian estimator in both16-dataset strata, with positive pointwise paired
95% intervals. Pooled excess risk versus conditional-only is.153186
[.066461,.251363] for cSMC and.132792 [.041363,.247269] for KDM. Neither
has an established gain over its own untrained adapter or the no-future
half-mixture. Mean ordering is not a global ranking. The label-risk identity
is unchanged; this is a negative result for these trained candidates. There
is no remaining infrastructure failure, so no localized retry is needed.

The [result note](artifacts/ledh-flow-aware-proposal-repair-20261020/result.md)
contains conditional heuristic comparisons, decision/inference-status tables,
commands, uncertainty and the post-run red team. All planned experimental
phases are complete. Total stage charge2616.956156692 seconds/two launches;
remaining4183.440736201 stage seconds and24245.397855901 parent-campaign
seconds. The reserved repair allowance remains800 seconds/one launch; unused
time is not an unplanned tuning sweep. Final manuscript build/render passes:
98 pages, no LaTeX warnings, new mathematics and result pages visually checked.
The protected supervision text and154 main-document math blocks are preserved;
all30 result-table cells match the structured output. See
`artifacts/ledh-flow-aware-proposal-repair-20261020/document-validation.json`.
All six phases are complete. The next research diagnostic is component allocation
and pre/post-reset moment decomposition, with new partitions for any tuning.
