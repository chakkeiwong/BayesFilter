# M10 first B/R surrogate-HMC test

## Research question and evidence contract

Test whether endpoint-corrected HMC using the full canonical analytical DCTR
position-only proposal field recovers the independently computed M10 model
posterior. M10 is the sine-transition, quadratic-observation model with T=2,
N=K=1024, the original nominal claim dataset, and independent
Normal(0,log(2)^2) priors on the two log-noise multipliers. Identity coordinates
are used. The B endpoint is the qualified 32/128 observation-root quadrature
in log arithmetic; R is the independently refined full-prior posterior05.
Both approximate the same model likelihood. Neither is an exact oracle.

Primary criterion: B/R equivalence of all 13 predeclared functions, under the
existing simultaneous intervals, reference allowances and frozen margins.
Compare standardized means, centered second moments and cross moment (0.1),
CDFs at each reference marginal's 10/50/90 percentiles (0.02), and replicated
second-observation probabilities below0.5 and2 (0.03). Use only cumulative
1,000 and2,000 retained draws per chain, four chains, with the shared
assess_equivalence procedure: modern parameter/function R-hat, bulk/tail ESS,
batch-MCSE sensitivity and batch correlation checks. Allocate error across
13 functions, two looks and four comparison families even though only B/R is
run here. Stop after a valid EQUIVALENT or DISCREPANCY result. An invalid
uncertainty screen is INCONCLUSIVE, not a discrepancy or equivalence.

Public tune_hmc_kernel with the typed independent endpoint/force binding is
the only tuning authority. Nomination, discarded warmup and fresh verification
are excluded from retained inference. A failed public tuning/health screen
rejects this attempt; it does not reject surrogate HMC. Source/data drift,
invalid archives, an accepted nonfinite state, a required reference failure,
or exhausted compute are continuation vetoes. Force-versus-reference gradient
disagreement, acceptance and runtime are explanatory only. Failed mixing or
uncertainty at1,000 permits the predeclared2,000 look; no third look or post hoc
MCSE rule change is allowed.

C/D, D/R and C/R remain NOT_RUN for M10: its named finite D endpoint and
independent D reference have not been qualified. This run cannot establish
an exact DCTR score, statistical superiority, heuristic usefulness, default
readiness or all-model success. The constructed heuristic set is prior-center
prediction, prior-predictive integration, and a linearized Kalman approximation,
evaluated conditionally on nominal and small-noise situations before any later
usefulness claim. The prior-center predictor ignores parameter uncertainty;
prior-predictive integration retains it without learning from observations;
the linearized Kalman approximation tests whether inexpensive Gaussian
inference suffices. Those comparisons remain NOT_RUN here. R supplies the
comparator for the 13 declared posterior functions under its stated allowances;
agreement of those functions does not prove equality of full distributions.

## Defaults, assumptions and skeptical audit

| Choice | Provenance and justification | Risk and earliest check | Status |
|---|---|---|---|
| Full canonical FP32 TF32 DCTR force, selected multiplier0.1 | Disjoint scoped calibration/validation, controls01; canonical call parity in endpoint01 | New claim states can fail despite the control screen; public tuning and every retained chunk preserve original status | Frozen proposal hypothesis |
| FP64 log-reference endpoint, 32/128 quadrature | Posterior05 refinement and endpoint01's16 passing checks | Sparse probes cannot prove global accuracy; reference allowances, full-prior tail bounds and per-call validity remain necessary | Qualified numerical reference, not exact likelihood |
| Four starts at(-2,-2),(-.7,.7),(.7,-.7),(2,2) times prior SD | Covers both log-noise directions using this target's actual prior scale | Starts miss separated posterior modes; independent R and functional checks veto | Target-specific initial-state hypothesis |
| Initial metric prior variance, epsilon0.1; two mass windows32/64, adaptation128 and verification64 for each L1/L2 candidate | Prior supplies dimensional scale; short windows only nominate a candidate, followed by fresh250/1,000 warmup/verification | Poor geometry or force gives false tuning failure; preserve candidate diagnostics | Warm-start hypotheses, independently adapted |
| Target acceptance0.7; covariance shrinkage0.1, jitter/eigenvalue floor1e-9 | Existing public tuner; ridge relative to this prior variance is tiny, all realized eigenvalues/condition checks preserved | Distorted metric or an unrepresentative short window; fresh verification and retained posterior tests | Tuning hypotheses, not transferred promotion evidence |
|1,000/2,000 retained looks, four chains; unchanged13 functions and margins | Campaign minimum schedule and qualified reference | Low power, especially CDF tails and batch correlation; INCONCLUSIVE remains possible | Frozen inference design |
| Full GPU/XLA execution with memory growth | Repository default; verified endpoint01 | Compilation/device drift; source archives, early GPU policy and per-chunk guards | Qualified execution mode |

The expected failure is poor movement or invalid uncertainty despite finite
endpoint correction. That is a repair trigger for a future separately planned
kernel candidate, not permission to select from failed claim draws. Frozen
thresholds must not become tuning targets. The pre-mortem includes passing
sparse endpoint probes while missing modes, and mistaking a short-chain screen
for posterior equivalence. Independent R and all13 retained functions are the
discriminating evidence.

Skeptical audit before launch: the old513-point local endpoint was wrong for
this test and has been replaced. Log arithmetic passes old/new and derivative
parity, underflow and refinement checks, direct canonical-force wiring and
reversal (1.11e-16). All10 complete L1/L2 transitions were healthy. The original
cost report counted one nomination candidate; recompute its preserved costs
for both candidates and explicitly include all8,000 predictive-functional
evaluations. The report is preserved; its old forecast cannot authorize launch.

## Prospective compute allocation and execution

Transfer2,800 CPU-worker seconds from unused P5 replication to P3. P3 rises
from15,500 to18,300, P5 falls from6,100 to3,300, and M10's cumulative CPU ceiling
rises from1,800 to4,600. Global54,000 CPU /226,800 GPU ceilings and M10's7,200
GPU-second ceiling stay fixed. LG2, SV20 and M10 have all had their initial
preparation turn. The measured M10 route is affordable; the larger LG2/SV20
cost screens are still deferred. This allocation funds one initial M10 B/R
attempt; it is not replication or a change of model, criteria or hardware.

At endpoint closeout cumulative global use is15,260.974448 CPU /13,593.589082
GPU seconds; M10 has548.778079 CPU seconds left before this transfer.
The corrected forecast uses3,778 worst-case L2 transition equivalents,
240 seconds compilation,90 seconds analysis,8,000 times the slowest measured
steady fine-reference value/score/predictive call, and a1.5 safety factor.
Reserve3,200 CPU /3,600 GPU seconds for the worker,60 CPU for focused tests
and30 CPU for orchestration; all fit after the transfer. Record actual charges,
including any failed attempt. No retry can silently reuse a closed look.

Use /home/chakwong/anaconda3/envs/tftwogpu/bin/python, escalated GPU1
GPU-68251639-fe82-8f81-3ccc-2953c32e805b, verified growth, two TensorFlow
intra-op threads, one inter-op thread, OMP2 and at most four CPU cores.
Master stage P3-M10-inference dispatches scripts/run_surrogate_hmc_nonlinear.py.

Prelaunch engineering review: 58 focused checks pass across the inference
reader, public tuning configuration, posterior-functional units, endpoint cost
forecast, M10 master integration and shared orchestration. One initial test
expected a list where the public payload returns a tuple; corrected the test
and reran it successfully. No scientific rule changed. Total test charge is
13.754641 CPU seconds (tests01 and tests02), within the60-second reservation.
The preserved-result reader checks source/data/input hashes, public tuning,
raw samples and traces, seeds, continuation and the two closed-look rules.

Launch01 stopped before the worker created an inference directory: the reader
mistook the outer master manifest for a prior inference attempt. This is an
orchestration defect, not a tuning or sampler failure. Restrict discovery to
actual inference-worker commands and the named child directory; add a
regression for the master manifest existing before child launch. Preserve
launch01 and charge its7.65603 CPU/.25093 GPU seconds. Retry the same first
statistical attempt in versioned master output20260922-m10-inference-02 under
the unchanged contract and budget; no tuning or retained look was consumed.
All20 inference-reader/configuration checks pass after this repair (tests03,
7.364928 CPU seconds); total focused-test charge21.119569 remains below60.
Root: docs/plans/artifacts/surrogate-hmc-all-models-20260922-m10-inference-01.
Preserve exact commands, source snapshots, inputs, independent tuning, raw
retained archives, both looks, per-chunk statuses, accounting and report.json.
Write a terminal result with decision/inference tables and strongest alternative
explanation. No reviewer unavailability or procedural formatting issue creates
a new launch gate; a material numerical or scientific finding does.
