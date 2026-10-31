# Whole repair plan: iAPF and MDPF/IWSG replication

R4d closure: source-semantic repair and bounded regressions are complete.
The repaired small CPU/GPU-XLA consumers pass; strict paper-horizon coefficient
parity remains FAIL. [Latest result](artifacts/iapf-mdpf-whole-repair-20260928-01/r4d-result.md)
and [review](artifacts/iapf-mdpf-whole-repair-20260928-01/r4d-review.md)
supersede stale R4c next-step statements. Both full replications remain incomplete.

Updated: 28 September 2026, Hong Kong.
Authority: [master program](iapf-kdm-replication-master-program-2026-09-24.md).
Both scientific replications remain INCOMPLETE. The R4d result and review above
are current. R4c results remain historical evidence for their own source
snapshots; the [previous plan](iapf-mdpf-whole-repair-plan-history-before-r4c-close-20260928.md)
and failed attempts remain intact.

## Research intent and evidence contract

The main question is whether we can reproduce the declared iAPF and MDPF/IWSG
filtering/learning experiments and implement their declared procedures in
TensorFlow/TFP on GPU. KDM names the mixture representation, not a third method.
Score/HMC, learned extensions and LEDH integration come later.

For iAPF, the source is Guarniero–Johansen–Lee, JASA 2017,
DOI 10.1080/01621459.2016.1222291: Algorithms 1–4, Section 5.1/Eq. (15) and
Section 5.2. The local publisher paper and Guarniero thesis Section 7.3 must
remain distinct: the article's regularized fit description does not supply a
complete executable penalty/solver recipe, while the thesis's Part III finite
BFGS procedure supports an explicitly labeled reconstruction. The original
initializer, coordinates, controls, floor, regularizer and seeds are not
recovered. Frozen recipes are in
[execution-spec.md](artifacts/iapf-mdpf-whole-repair-20260928-01/execution-spec.md).

For MDPF, use Younis–Sudderth NeurIPS 2023 technical sections/Appendix B
Eqs. (20)–(22) and author commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.
The scalar linear dynamics and bimodal Gaussian observations, learned
coefficients/mixture weights, length 5 training sequences, final posterior
NLL and IWSG/IRG/truncated/mixture-Kalman comparisons define Appendix B.
The source nonlinear Cappe toy problem is not this experiment.
Bearings source settings/metrics and initialization are anchored in
[mdpf-source-and-screen-spec.md](artifacts/iapf-mdpf-whole-repair-20260928-01/mdpf-source-and-screen-spec.md).

| Research-intent field | Binding interpretation |
| --- | --- |
| Candidate/mechanism | Complete finite adaptive iAPF; kernel-mixture filtering and IWSG learning |
| Exact comparators | iAPF: independently reconstructed R, exact Kalman, BPF, fully adapted APF; MDPF: pinned author code, exact mixture-Kalman where tractable, matched classical densities |
| Primary correctness criterion | Checked probability/gradient identities and actual-consumer agreement under the phase-specific contract |
| Primary empirical criterion | Frozen source metrics, all declared cells/repetitions, failures retained, appropriate uncertainty; no claim of equality from an unspecified margin |
| Expected failure | Finite-fit scale/curvature sensitivity, amplitude escape, optimizer branch/resampling divergence, source-choice mismatch, or training-protocol failure |
| Promotion veto | Failed parameter parity for that claim, invalid fit/density/gradient, heuristic underperformance, missing retuning/source scope or required repetitions |
| Continuation veto | Wrong target/data/comparator, corrupted evidence, unusable required diagnostics, invalid accepted probability computation, exhausted applicable budget |
| Repair trigger | Numerical mismatch, invalid candidate, infrastructure failure or conditional heuristic loss; preserve it and take the discriminating next phase |
| Explanatory diagnostics | Fit loss/gradient, iteration cap, ESS, compilation/runtime, one-seed likelihood closeness, local sensitivity |
| Excluded conclusions | Full replication from a smoke, recovered author settings from reconstruction, superiority from descriptive means, HMC/default readiness from filtering success |

A failed candidate is not rejection of the research direction. Stationarity
is not a universal validity requirement for finite BFGS. A failed strict
parameter check remains failed even if a later downstream diagnostic succeeds.
Exact path agreement and statistical filtering agreement answer different
questions; new statistical criteria must be declared before fresh runs.

## Current evidence and immediate correction

Attempts 01–16 established 20 independent R controls, a 16-run finite pilot,
source metric/conditional point checks for one MDPF checkpoint, ten paired
frozen-twist value fixtures across three backends, and a localized legacy
TensorFlow graph-optimizer problem. A_raw completed 8/8 pilot runs; B_scaled
completed 5/8 with three rejected nonpositive-amplitude fits. No ranking follows.

Attempts 17–28 implemented the shared finite fitter, backward recursion and
adaptive controller. Small CPU/GPU graph/GPU XLA checks pass 129/129 each.
The paper-horizon exact-fixture comparison fails in backward call 2.
The actual GPU/XLA consumer nevertheless completes d=5, T=100 with eight filter
calls, six backward passes, stopping iteration 6 and fresh final N=2000.
Final logZ is -894.0119586545842, versus R -894.014159908217 and
Kalman -894.071542920976. This is one diagnostic case, not accuracy inference.

A proven CSV rounding defect was repaired with 17-significant-digit export
and 3,142 exact matrix roundtrip assertions. It was not the sole cause:
the repaired fixture still fails strict parity. Same-backend log-target
perturbations around 1e-14 move fitted coefficients by about 0.003. Earlier
analytic-gradient replay errors were at most 4.1e-17. CPU/GPU arithmetic and
finite optimizer trajectories need investigation; XLA-only blame is unsupported.
The new actual route has one trace per signature, native TensorFlow loops,
no NumPy numerical path and no per-particle Python loop.

## R4d — Numerical closure after source-semantic repair

The finite fitter exists and now uses one shared TF optimizer loop. R4d
attempts29–37 repaired nonfinite-trial backtracking and lower-triangle Hessian
semantics. CPU/GPU graph/GPU-XLA controls pass; final43-check XLA diagnostics and
129-check CPU/XLA actual small consumers pass. A loop-state initialization
regression and a rejected-trial counter bug were caught and repaired. The exact
contract and all attempts are preserved in
[r4d-finite-semantics-spec.md](artifacts/iapf-mdpf-whole-repair-20260928-01/r4d-finite-semantics-spec.md).

The longer parameter criterion remains failed, with three failures among95
checks. Exact-initializer replay still diverges late in100-step fitting across
all three backends. The original CSV precision defect, initialization alone,
and XLA alone do not explain the whole issue. General source equivalence and
the first differing arithmetic primitive are not proved.

The next question is whether the remaining finite arithmetic sensitivity
materially changes the filtering/likelihood-estimation distribution. Before
further research runs:

1. Write a separate paired downstream protocol. Freeze the finite recipe and
   collect calibration data to set a numerical error budget relative to Monte
   Carlo variability and the intended claim. Preserve the previous parameter
   gate as FAIL; do not select a threshold from its observed failures.
2. Freeze all margins, seeds/partitions, sample counts and stop conditions
   before untouched evaluation. Use actual adaptive consumers, exact Kalman
   likelihoods and BPF/fully adapted comparators; include adaptation and
   compilation/steady-state costs separately. Compare likelihood estimates on
   the scale justified by the statistical target, not merely one logZ value.
3. Use paired uncertainty intervals and conditional diagnostics across declared
   dimensions/regimes. A short smoke or one similar likelihood can explain or
   veto; it cannot establish equivalence or the paper's empirical result.
4. If a newly identified finite-source mismatch is repairable, repair it and
   rerun the original unchanged gate. If a damping/regularization candidate is
   tested, define the changed target and source basis, retain the unmodified
   comparator, and evaluate non-harm with a dedicated calibration. Do not
   silently adopt a new recipe because a descriptive loss decreases.
5. Retain original-author uncertainty. The article says regularized fitting;
   the executable controls we inspected do not recover its exact numerical
   choices. A labeled reconstruction can be evaluated, but cannot be called
   recovery of undocumented settings.

Proposition 1 concerns likelihood-target invariance for bounded continuous
positive twists, not equality of fitted coefficients. That distinction explains
why the next downstream question is meaningful; it does not erase the strict
finite-port failure or prove numerical harmlessness. No validated remedy or
scientifically justified downstream equivalence margin is claimed yet.
The next phase remains explicitly unimplemented until that contract and actual
consumer commands exist.

## R3b — Complete normalized MDPF density comparisons

This work is independent of iAPF. Preserve the saved NLL checkpoint and test
data as evaluation evidence; do not tune a repair to those results.

Construct four information-matched density baselines: persistence with
training-calibrated dispersion; constant-motion density using training-only
motion estimates; robust EKF/UKF with circular observation treatment; and
a fitted generative BPF with the same observations/training information.
Use the identical saved initial cloud because the author supplies noisy
truth-centered initialization. An oracle with true generative parameters
must be labeled information-advantaged, not a fair learned competitor.
Bearings have one sensor at (-5,0), concentration 50 and 85/15 contamination;
do not invent two-sensor triangulation or use the ignored source bias setting.

All baselines must produce normalized densities in position and circular
heading. Check normalization on tractable controls before comparing NLL.
Tune only on training/calibration data and freeze all controls. Retain the
source mean-over-trajectories of time-root-mean-square mixed-state metric
and separately report position error with units.

Use the predeclared early 1–17, middle 18–50 and late 51–150 windows and
training-threshold observation-residual strata from R3a. Conditional truth-based
strata are post-run diagnostics, not predictor information. An observed
conditional loss to any heuristic vetoes promotion. The checkpoint remains
one training replicate despite 5,000 test trajectories.

Required output: normalization controls, frozen baseline specification,
per-trajectory/stratum NLL and error tables, a machine-readable
heuristic-dominance verdict, paired trajectory uncertainty conditional on
this checkpoint, and explicit absence of between-training-seed inference.

## R3c — Implement the actual Appendix B learning experiment

First finish source discovery for the numerical true coefficients, parameter
initialization/perturbations, schedule and seeds. Do not assert that no code
exists merely because none was identified in the inspected tree.
If unavailable, freeze an explicit reconstruction with its rationale and
sensitivity questions. Exact-author numerical reproduction stays unavailable.

Build the scalar linear-Gaussian/bimodal model from Eqs. (20)–(22). Implement an
independent exact mixture-Kalman reference for the short horizon. Validate
normalization, initial-state convention and mixture weights. Trace the actual
IWSG/IRG/truncated consumers: sampling, stop-gradients, denominator conditioning,
finite-sample normalization and posterior/resampling bandwidths. Gradient
checks precede, but cannot replace, the learning experiment.

Freeze a target-specific protocol: objective/scale, architecture if applicable,
optimizer and learning-rate search, training budget ladder, seed policy,
calibration/validation/audit split and downstream posterior criterion. The
paper's 1,000 length 5 sequences, batch 64 and final posterior NLL are source
anchors; missing numerical choices remain hypotheses. Required output is
actual learning and posterior evidence against the mixture oracle and source
comparators, not a substitute nonlinear toy or one-step gradient demonstration.

## R4e — Implement eligible MDPF TF/TFP consumers

The author PyTorch implementation remains an independent reference, not the
BayesFilter runtime. Use batch-native TensorFlow/TFP for data, sampling,
density evaluation, IWSG gradients and optimizer updates. No NumPy runtime,
scalar-row fallback or pfor. GPU memory growth must be configured and verified
before device initialization.

Check the actual consumer call chain against the author reference with
identical primitive inputs/entropy, stable signatures and explicit precision.
Then test full downstream filtering/training under frozen source/reconstruction
settings. GPU/XLA is the default; CPU/graph modes are labeled diagnostics.
Record compatibility, finite validity, gradient agreement, peak memory,
compilation/steady-state cost and bounded trace variants.
Source quirks and intentional mathematical corrections must be separately
labeled. Passing a reference test alone does not establish empirical replication.

## R5 — Complete the empirical replications

R5a: freeze the surviving iAPF recipe without ranking it on the small pilot.
A_raw is the source-ordered representative, not a statistically established
winner. Measure complete adaptive-run steady-state cost before a large cell.
Use new untouched data/streams for d5, then10,20,40,80 with T=100 and the
source's 1,000 complete repetitions per fixed dataset/dimension.
Retain the source particle allocations separately from any equal-cost study.
Record incomplete runs, fitting failures and adaptation cost.
Kalman is the exact likelihood comparator. Construct BPF, fully adapted APF
and a simple observation-twist filter as cheap adversaries; condition checks on
dimension, horizon and training-calibrated ordinary/extreme observation regimes.
No heuristic is a tuning target.

R5b: bearings must include the stated objective/refinement stages, declared
comparators and independent training repetitions. Check the effective random
seeds actually used, not just configuration values. Preserve source-faithful
and corrected-RNG branches distinctly. Apply the same conditional density gate.
Report uncertainty across independent training runs; trajectories are nested
evaluation units.

Before either confirmatory phase, freeze primary metrics, equality/equivalence
or replication criterion, uncertainty method and effect margins using source
anchors or a reviewed calibration argument. Do not silently invent thresholds.
No full empirical cell starts unless its worst-case allowance fits the balance.
A smaller pilot retains its smaller claim.

## R6 — Terminal result review

Keep separate source, engineering, numerical and scientific conclusions.
A negative replication may be complete evidence without being method promotion.
Review source anchors, actual call chains, all failures, uncertainty, conditional
heuristics and the resource ledger. Use a decision and inference-status table.
State the strongest alternative explanation and what evidence would overturn
the conclusion. There is no terminal acceptance until the declared scoped
replications and GPU consumers have their required evidence.

## Defaults and assumptions

| Choice | Provenance/status | Failure risk and earliest discriminating check |
| --- | --- | --- |
| A_raw/B_scaled finite 100-step BFGS | Explicit thesis/source-motivated reconstruction hypotheses; not recovered original code | Scale/density escape and sensitive finite trajectory; saved optimizer replay and actual-filter check |
| Unpenalized Eq. (15) | Chosen to reconstruct thesis Part III; article's exact regularizer unknown | Degenerate density/amplitude direction; record norms/validity and separately evaluate any justified regularized recipe |
| Full-rank nonpivoted QR initializer | Engineering restriction checked against R on healthy cases | Rank-boundary mismatch; reject relative rank margin<=1e-7, no claim there |
| Floor delta=1/N and peak formula | Frozen reconstruction supporting positive mixture/floor probability control | Wrong convolution/floor scale; independent mixture identities and exact-twist controls |
| FP64, TF32 off | Deliberate reference-parity baseline, not an imported LEDH default | Cost and cross-dtype fragility; record hardware, no FP32/TF32 transfer claim |
| R fixture arrays/seeds | Full-precision paired diagnostic inputs | Serialization/reuse bias; exact roundtrip, hashes, fresh confirmatory streams |
| MDPF checkpoint/initialization/metric | Pinned author source; one NLL training run | Information advantage, mixed units, pseudoreplication; matched clouds, normalized density controls, independent training seeds |
| Missing Appendix B numbers | Unresolved, cannot be defaulted silently | Replicating another problem; recover source or label/review reconstruction before learning |

## Budget, commands and execution review

After attempt 37: CPU remaining 753.166936 seconds; GPU remaining
130,951.300351 seconds (36.375361 hours), within the existing 1,800 CPU and
173,700 GPU-second ceilings. Failed starts, compilation and numerical
diagnostics consume the same campaign balance.

Provisional phase ceilings from that balance, not new grants:

| Phase | GPU hours | CPU seconds |
| --- | ---: | ---: |
| R4d remaining numerical closure | 0.439443 | 210.821790 |
| R3b normalized comparators | 1 | 60 |
| R3c Appendix B | 1 | 120 |
| R4e MDPF port | 6 | 120 |
| R5a iAPF experiments | 8 | 120 |
| R5b MDPF experiments | 18 | 60 |
| Terminal/repair reserve | 1.935918 | 62.345146 |

These ceilings do not make unimplemented phases executable or promise full
completion. Tighten each when an actual command and source specification exist.
A bounded tranche may be added without renewed approval when target, hardware
and total budget remain unchanged. Do not repeatedly retry the same fault
without a new discriminating question.

A full previous MDPF trial took 8.589h. Ten additional trials would require
about 85.89h at that rate, before comparators/refinement. The full source suite
is under-budgeted. Measure eligible runtime before revising the forecast;
do not reduce the completion claim to fit available time.

The dispatcher is scripts/run_replication_master.py:
status reports live gaps/charges; validate checks structure; run-next launches
one ready phase; run-ready stops on failed children; recover preserves
uncertainty about lost exits. It currently has no ready numerical phase.
Future phase entries are explicitly NOT_IMPLEMENTED until real commands exist.
Use approved direct interpreter invocations, escalated GPU access, verified
memory growth and CPU hiding for CPU-only references. No new approval tokens,
broad allow-list permissions, external messages or environment changes.

Skeptical review: PASS for the executed R4d bounded repair/diagnostic scope; REVISE/OPEN
for full source equivalence, statistical replication and default promotion.
Wrong legacy targets, unsuitable toy data, stale missing-fitter statements,
proxy-to-promotion errors and hidden failure statuses have been identified
and corrected in the active program. Tested source-semantic edges are repaired; general equivalence, numerical
sensitivity, missing source choices, conditional density tests and full-suite
resources remain explicit. This is self-review; no independent agent was used.

Premortem: successful commands can reproduce the wrong model, overfit test
data, confuse trajectories with training seeds or hide failures in averages.
A failed fit can reflect finite arithmetic/source choices rather than invalid
filter mathematics. Exact source mapping, local semantic controls, fresh
partitions, conditional adversaries and actual consumers distinguish these.
The R4c post-run sensitivity/impact diagnostic followed this rule and did not
erase its failed parameter-parity result.
