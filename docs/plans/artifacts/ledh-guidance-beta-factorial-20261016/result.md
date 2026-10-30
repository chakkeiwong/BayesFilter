# Guidance and mixture selection: completed Gaussian diagnostic

No future-guided arm clears the descriptive heuristic promotion screen in
every regime. All20 fixed-beta guidance contrasts for log-process-SD have
paired95% intervals containing zero. This616-evaluation experiment therefore
does not establish score improvement or a ranking. It does establish that the
earlier result cannot be explained simply by saying beta calibration damaged
an otherwise beneficial guide: fixed-global guidance itself has a large,
uncertain stress-case loss, while calibrating beta for that guide reduces the
observed loss. These differences remain descriptive.

All616 numerical evaluations are valid. The initial calibration interruption
was repaired within budget; no numerical continuation blocker remains. This
is the optional covariance-guided density mixture, not canonical
Contract-E/UKF LEDH admission. Epsilon1 and Sinkhorn5 remain frozen development
settings, not tuned or promoted defaults.

## Primary comparison

The table holds beta=(0.1,0.9,0) and evaluation randomness fixed, changing only
the frozen future message. Errors concern the exact Kalman model score in
log-process-SD coordinates. Each regime has12 final independent datasets;
the10000 paired bootstrap resamples preserve dataset/randomness pairing.

| AR coefficient / regime | Current-only MSE | Future MSE | Future minus current | Paired95% interval |
|---|---:|---:|---:|---|
|0.2 / ordinary|0.001953|0.001717|-0.000236|[-0.001757,0.001161]|
|0.2 / impulse|0.013982|0.012935|-0.001048|[-0.006356,0.004399]|
|0.9 / ordinary|3.324929|1.413726|-1.911203|[-5.076960,0.039174]|
|0.9 / impulse|9.076029|21.606853|12.530825|[-10.380248,50.978611]|

All four primary intervals include zero. The same is true of the sixteen
guidance contrasts at local, equal, current-calibrated and future-calibrated
beta. Numerical validity makes these candidates eligible for further
investigation; it does not establish their accuracy or superiority.

## What the crossed beta arms explain

In the persistent impulse regime, independent calibration selects
current beta=(0.1,0.848017,0.051983) and future
beta=(0.283202,0.307134,0.409664). Applying both policies to both guidance
states gives the following log-process-SD MSEs.

| Frozen beta source | Current-only guidance | Future guidance |
|---|---:|---:|
| Current-only calibration |6.265445|33.099794|
| Future calibration |7.882634|5.767819|

The future-minus-current calibration effect within future guidance is
-27.331975, interval[-70.858285,0.980373]. The interaction contrast is
-28.949164, interval[-72.561217,-0.229480]. This single pointwise explanatory
interval suggests that guidance and mixture selection interact in this
regime; it is not a multiplicity-adjusted ranking or proof of improved scores.
The cheap current-only local heuristic has MSE5.650380, compared with5.767819
for future guidance calibrated on future contexts; their difference0.117439
has interval[-3.306582,4.401507]. This descriptive loss still vetoes nomination.

The full conditional heuristic tables, all four score coordinates, all
prespecified contrasts and candidate-specific losses are in `decision.json`.
The primary future-global arm loses descriptively to all four heuristics in
the persistent impulse regime. Every other future arm also loses to at least
one constructed heuristic somewhere in the grid. No candidate is promoted.

## Concentration and sample sensitivity

Saved diagnostics give a concrete next question. In the persistent impulse
regime, the mean per-sequence minimum ESS is14.241 for current-global and4.588
for future-global; mean per-sequence maximum weight is0.165 versus0.519.
Future-specific calibration changes these observed diagnostics to7.635
and0.304. These are explanatory summaries, not score criteria.

Final seed220205 contributes a squared-error difference212.810020 to the
fixed-global comparison: current error16.457570 versus future229.267590.
Its minimum ESS changes from14.746749 to1.148137. Leaving one dataset out
changes the stress-case mean difference over the range[-5.676375,16.788388].
No dataset is excluded from the reported comparison. This sensitivity and
the wide intervals prevent attributing the observed loss to a general
guidance defect. Likewise, low ESS alone does not prove the score-error cause.
One particle realization per dataset cannot separate particle Monte Carlo
variation, dataset variation and finite-N bias.

The next discriminating experiment is therefore repeated independent particle
clouds at fixed fresh datasets, followed by a small particle-count ladder,
with current-only/exact-future guidance crossed at fixed beta and with
scope-specific independent calibration. Preserve the existing holdouts; do
not tune guide strength or beta on their observed score errors. Keep KDM
capacity expansion and nonlinear transfer pending this localization.

## Engineering, numerical and mathematical checks

- 21 focused pairing, corruption, shared-consumer and numerical-validity tests
  pass; two additional beta simplex/certificate and convexity tests pass.
- Attempt01 stops after154 valid rows at condition1 current-only calibration.
  The original2000-step solver cap leaves gap4.400315e-6; CPU reproduction
  reaches the unchanged certificate at4022 iterations, gap1.017546e-7 below
  required1.018624e-7. Only the diagnostic runner's iteration cap changes to
  20000. The shared solver, objective, tolerance, update and default are unchanged.
- Attempt02 confirms4022 iterations on GPU and completes616 rows. All154
  previously accepted score vectors and likelihood errors are bitwise unchanged.
- Every paired arm has the same data/randomness fingerprint and exact reference;
  beta is identical within each guidance pair and frozen across evaluation
  datasets. No rows are missing or dropped. Loaded-source hashes match.
- One old-roundoff warning is retained; maximum covariance relative residual
  is9.411910e-15. There is no material moment, density, cap or finiteness failure.
- The manuscript's `prop:bg-mixture-guided-score` fixes guide coefficients and
  beta, matching the analytical finite-program derivative computed here.
  Comparison to the Kalman score measures model-score error separately.
  Neither the derivative proposition nor the conditional likelihood-variance
  identity proves that this model-score error is smaller.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not nominate any guided default | All primary intervals include zero | Descriptive heuristic promotion veto; no remaining numerical continuation veto |12 datasets; strong stress-tail sensitivity; guide/beta interaction | Repeat particle randomness at fresh fixed datasets, then particle-count localization | Overall inferiority, rejection of backward information, canonical admission or KDM failure |

| Inference item | Status |
|---|---|
| Hard veto screen | Calibration cap failed on attempt01; localized repair passes. All final evaluations valid |
| Statistically supported ranking | No overall or fixed-beta guidance ranking; all20 guidance intervals include zero |
| Descriptive-only differences | MSE table, beta effects, ESS/weights and runtime; one pointwise interaction interval is exploratory |
| Default readiness | No; heuristic veto and missing target-scope tuning/canonical evidence |
| Next evidence needed | Conditional repeated-cloud uncertainty and finite-N behavior, preserving independent calibration and untouched evaluation |

Post-run skeptical review: this answers the fixed-beta intervention question
on the declared Gaussian consumer, not canonical LEDH performance. The
strongest alternative explanation is a small number of particle/dataset
tails, reinforced by the leave-one-out sensitivity. Stable conditional
replication showing concentration unrelated to score error would overturn
the proposed concentration explanation. The weakest evidence is only12
dataset clusters, particularly for the impulse tail. No scientific direction
is rejected. The next phase repairs this evidence limitation rather than
abandoning the program or increasing guide capacity without justification.

## Reproduction and budget

Plan: `docs/plans/ledh-guidance-beta-factorial-20261016.md`.
Checkout: `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`,
branch `ledh-refactor-with-policy-fix`, commit `ea1018de1` plus the source hashes
in `attempt02/manifest.json`. Environment is `tftwogpu`, Python3.11,
trusted RTX4080 SUPER, FP64 GPU/XLA, TF32 off, verified memory growth; the
Kalman oracle is an explicitly CPU reference. CPU tests and analysis hide GPU.
Shared-device timings do not support speed comparisons.

The executed retry uses `scripts/run_ledh_guided_mixture_score.py
--guidance-factorial --beta-max-steps 20000 --output
docs/plans/artifacts/ledh-guidance-beta-factorial-20261016/attempt02
--budget-seconds 700`, under external timeout740. Analysis uses
`scripts/analyze_ledh_guidance_factorial.py --input
docs/plans/artifacts/ledh-guidance-beta-factorial-20261016/attempt02 --output
docs/plans/artifacts/ledh-guidance-beta-factorial-20261016/decision.json`.

Preserved evidence: both attempt directories, both GPU logs, both CPU test
logs, CPU calibration diagnostic JSON/log, analysis log and `decision.json`.
The runner records exact command/environment, seeds, controls, source hashes,
calibration results, per-row diagnostics and GPU allocation policy.

Conservative accounting uses the larger internal/supervisor clock for each
attempt:151.485+577.514=728.999 of900 GPU seconds. Supervisor times are139.79
and532.09 and are retained rather than substituted. CPU checks/analysis take
52.58 seconds; reserve54 of180 including short inspections. Both planned GPU
attempts are finished. There are no live experiment processes. This completes
the bounded factorial stage; the next research stage requires its own concise
design and budget, not a new approval token.
