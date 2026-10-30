# SIR proposal-mixture repair — conditional next stage

## Question and dependency

The first SIR experiment holds the proposal probabilities at equal thirds to
isolate guidance. That is not Algorithm 0's independently fitted mixture. If
the completed comparison finds no robust gain, test whether its poor scores
come partly from assigning too much probability to unstable observation maps.
This stage starts only after the first result is recorded. OT remains (1,5);
the model, data, frozen guides, N256, T50, flow16 and analytical score remain
unchanged. No score or Zhao–Cui reference enters calibration.

## Research intent and evidence contract

Candidate: the existing convex beta solver applied to the actual three proposal
densities for each frozen guide. Main question: does independently calibrated
beta make guided score estimates more accurate than unguided proposals on
fresh paired particle seeds? The conditioning is still on two fixed data sets
and one frozen guide each; this is not original KDM training or canonical LEDH.

For fixed incoming clouds/maps, let a=g p-minus and h=(q0+qG+qL)/3. Pilot draws
come from h. Estimate J(beta)=integral a^2/q_beta with averages of
(a/h)^2/(q_beta/h). Normalize a/h within each context as the existing solver
does, which produces a sum of positive rescaled context objectives. This is
convex in beta but is not the variance of the full filter or its derivative.
The gap certificate checks optimization of this finite pilot objective only.

Use 50 contexts from independent prior-proposal filter histories. The same
context clouds are used by all three guide families to isolate proposal-density
differences. Each context has 256 independently generated mixture draws. Fit
with beta0 >= .1 using the repository solver and relative gap tolerance 1e-7.
The .1 floor is the existing defensive hypothesis: q_beta >= .1 q0, so p-minus
/q_beta <= 10. It does not bound g, prevent covariance rank loss, or certify
score accuracy. Its use is not a promoted default.

Use separate calibration and validation particle histories and noise streams,
then freeze beta before score evaluation. A valid, converged fit with finite
independent validation objective remains evaluable even if validation J is
worse than equal thirds: record this rather than silently selecting a fallback.
An invalid pilot/solver stops that candidate for a bounded diagnosis; do not
substitute old-scope beta. Validation J is explanatory, never a score-promotion
criterion.

Seven arms on eight fresh paired seeds per data set (112 rows): unguided with
its calibrated beta; quadratic with its calibrated beta; kernel with its
calibrated beta; unguided with quadratic beta; unguided with kernel beta;
physical prior; defensive local (.1,0,.9). The last two and calibrated unguided
form three cheap adversaries. The raw defensive global comparator's failed
first-stage seed remains an explicit numerical failure; it is not silently
reintroduced as a healthy benchmark.

Primary contrasts: quadratic versus unguided at quadratic beta, kernel versus
unguided at kernel beta, and each guided arm versus independently calibrated
unguided; evaluated separately on both data sets. Eight contrasts use paired
99.375% bootstrap intervals (20,000 draws, seed 26101998). A guidance family
needs negative upper endpoints in both data sets for its matched-beta contrast
and for the independently calibrated comparison to support a conditional gain.
Component MSE losses to any valid heuristic veto promotion; heuristic tables
are descriptive, not statistical rankings. Any invalid row blocks that arm's
complete comparison and stops the attempt for diagnosis.

## Defaults, failure modes and skeptical audit

The fixed first-stage guide is deliberately reused: fitting and guide-capacity
changes would confound this mechanism check. Fresh evaluation seeds prevent
particle-seed selection; reusing observed data means no fresh-data confirmation.
Prior histories avoid selecting a guide's own favorable cloud, but may poorly
cover guided evaluation states. The independent validation objective is the
earliest check of this mismatch, not proof against it. Heavy-tailed importance
weights can make a finite pilot objective misleading; retain pilot max weights,
effective sample sizes, gap and per-context objectives.

GPU FP64/XLA, TF32 off and memory growth retain the first-stage reference
exception. The beta floor, solver tolerance, 256 pilot draws and 50 contexts
are inherited diagnostic hypotheses; no default readiness follows.
Solver iterations are capped at 20,000 rather than the 2,000-step library
convenience limit: the relative duality gap, not an iteration count, defines
convergence. This larger ceiling changes only the allowed optimization effort;
the 1,800-second phase cap remains binding and an unconverged fit is invalid.
All
functions must use existing shared maps/densities/solver, explicit signatures,
TensorFlow loops and analytical score paths. No pfor or runtime retuning.

Skeptical audit: the principal threat is confusing lower J with a better score.
Matched-beta unguided arms and held-out score comparisons prevent that inference.
The failed global seed is preserved, not used to relax numerical checks.
This is a bounded repair of mixture selection already in the master, not an
OT sweep or a new research direction. If betas stay near equal thirds or the
paired score errors do not improve, beta calibration has not rescued this
candidate. Do not infer rejection of controlled SMC/KDM as a whole.

## Budget and artifacts

Within the owner's twelve-hour campaign ceiling, allocate 1,800 GPU-process
seconds to calibration/derivative checks and 7,200 to 112 evaluation rows, with
localized infrastructure retries drawing from the same 9,000-second allocation.
Fresh output root `artifacts/ledh-sir-beta-repair-20261019/`; record exact source
hashes, seeds, data/guide identities, device/memory policy, J diagnostics, beta,
scores, full traces, failure dispositions, wall time and remaining campaign
budget. Calibration seeds 261019501/502 and 261019601/602; evaluation seeds
261019701–708 and 261019801–808. Freeze controls before any evaluation.

After the stage, distinguish implementation validity, numerical failure and
statistical evidence; update both masters, manuscript and active checkpoint.
No public release, environment changes, commit or push.

## Pre-run implementation audit

The new pilot calls the shared build_maps, draw_from_maps, component_densities
and observation_density functions. A CPU parity test checks actual call-chain
outputs. Fourteen additional synthetic calibration/reporting checks pass; the
focused beta objective, derivative, gap and wiring selection passes seven
checks in 6.44 seconds (28 unrelated cases deselected). Calibrated evaluation
calls the same make_filter, with probabilities frozen as explicit inputs.
GPU derivative checks cover all seven arms on both data sets before any score
evaluation. No optimizer or reference-score lookup occurs inside the filter.

The manuscript derives empirical-objective convexity and the gap bound.
MathDevMCP's two bounded label audits are inconclusive because the parser
did not preserve/extract the proposition label. The complete responses are
in ../plans/artifacts/ledh-sir-guidance-20261019/beta-mathdev-audit.json.
This is an audit-tool limitation, not proof certification or a scientific
veto. The displayed derivatives and convexity argument were checked directly;
solver derivative/convexity tests are executable supporting evidence.

## Localized calibration repair, 19 October

Calibration-01 failed after 57.481171 seconds. Its first plain pilot remained
finite, but the projected-gradient line search exhausted 40 backtracks near
beta_G = 8.36e-13 with a nonzero .0805857 gap. Independent replay preserved
the objective at 1.69054325 and exposed a rapidly changing directional
derivative near zero probability; this is optimizer failure, not evidence
against the statistical objective or a failed covariance reset.

Use an optional pairwise simplex solver on the identical objective. Transfer
mass from the largest-gradient positive-excess coordinate to the smallest-
gradient coordinate; minimize that convex one-dimensional restriction by
derivative bisection or its endpoint. Stop only at the unchanged global gap
tolerance. Log-domain sums are algebraically equivalent evaluations, not
clipping, damping, a changed floor or a changed objective. The old solver
remains unchanged for existing consumers; this campaign calls the optional
repair explicitly. No score reference enters this repair.

Four CPU checks cover the analytical boundary optimum, symmetric interior
optimum, direct/log objective and gradient parity, and small densities. The
exact failed pilot now converges in 19 iterations to beta approximately
(.1, 1.24e-9, .9), objective 1.66381703, gap 3.92e-10. This is optimization
evidence only, not score improvement. CPU replay took 1.51 seconds. The first
test attempt caught an endpoint representation issue; explicit lower-plus-
excess coordinates fixed it without relaxing the feasibility check.

Skeptical re-audit: the target, guides, pilot seeds, floor, relative gap,
comparison arms, OT and budgets are unchanged. A fresh calibration-02 has
at most 1,742 GPU-process seconds, preserving the original 1,800-second
calibration allocation after the failed attempt. Its independent validation
and held-out evaluation must still run; a successful optimizer is not a
promotion result.

Final focused solver regressions pass: six tests, 18 unrelated cases
deselected, 5.18 seconds. The tiny-density fixture checks an exact finite
gradient (-2,-8,-2) where direct squared-denominator arithmetic underflows.
The repaired objective retains those terms through log sums rather than
discarding observations. Eighteen calibrated-runner/analysis/solver tests
also pass in 5.18 seconds. The manuscript includes the first-stage result
table; two successful LaTeX passes produce 80 pages and the rendered
result page has been inspected.

Calibration-02 spends 575.421810 seconds: plain and quadratic fits converge,
but greedy pair selection cycles around the kernel's tiny global coordinate
for 20,000 iterations with gap .0553699. This is an optimization failure,
not a failed scientific candidate. Cycle deterministically through all three
coordinate pairs instead, choosing the downhill orientation within each pair.
This prevents a tiny coordinate from starving the substantial prior/local
pair. The objective, exact line minimizations and global gap are unchanged.
CPU replay of all three saved pilots converges in 6, 2 and 3 iterations
(1.05 seconds total); kernel objective 1.80350817, gap 9.85e-10.

Calibration-03 has a 1,167-second cap: prior attempts consume 632.902981
seconds of the original 1,800-second calibration allocation. The GPU activity
probe is diagnostic overhead within the total campaign allowance. Progress
now identifies pilot versus optimization phases and preserves pilots before
solver execution. No reference score, OT control, model or seed changed.

Calibration-03 completes in 168.703374 seconds. All six fits pass the same
gap criterion and have finite independent validation objectives, each lower
than its equal-mixture objective. Total calibration spending is 801.606355
seconds, below the 1,800-second cap. The global probabilities are nearly zero
in all fits; this is a descriptive allocation result, not score evidence.
The frozen checksum is recorded in calibration-03/manifest.json and consumed
by evaluation-01, launched with its original 7,200-second cap and 112-row
design. No settings are selected using those held-out scores.

## Completed evaluation and decision

Evaluation-01 completes all 112 rows without a numerical failure in
4,173.834361 GPU-process seconds, including 162.076230 seconds of derivative
checks. All 84 finite-difference and 42 Richardson checks pass. Source and
calibration checksums, full keys, three 50-step directions and RMSE arithmetic
pass the independent terminal check. See
[the result](artifacts/ledh-sir-beta-repair-20261019/result.md) and
`artifacts/ledh-sir-beta-repair-20261019/terminal-audit-02.json`.

Both matched-beta contrasts on data 26100832 have negative upper interval
endpoints; the other six contrasts include zero. Neither guide meets the
predeclared gain requirement, and both fail the componentwise heuristic
screen. The supported comparisons are conditional on two fixed datasets,
one fitted guide each and approximate references; eight-seed bootstrap
coverage is not exact and reference uncertainty is not propagated.
No numerical continuation veto fires. Do not promote either candidate,
reject the research direction, tune on final scores, or launch an OT sweep.

The beta stage spends 4,975.440715/9,000 GPU-process seconds; total campaign
spending is 9,325.730915/43,200 including the first-stage preflight reserve.
No campaign GPU job remains. Next prepare SIR proposal/reset versus message
attribution and verify canonical call-chain conformance before an LEDH claim.
