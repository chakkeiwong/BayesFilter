# Tuned range-bearing comparison of the new covariance proposal

The owner asks whether the merged method helps range bearing and requests
scope-specific tuning and at least 1,008 particles. This is the next preparation
stage within the surrogate-HMC campaign, not a posterior-sampling result.

## Question and evidence contract

Test M13's existing constant-velocity state4/observation2 model, T=20, N=1008,
parameters log process and observation scales. The candidate is the shared
covariance-guided defensive-mixture filter. The comparator is the existing
canonical analytical filter with marginal mixture importance weights, which
was the previous repair candidate. Both get identical initial/process draws
and observations; extra fixed categorical uniforms belong to the new method.
Use the shared wrapped-Gaussian observation law and model callbacks. Do not
replace range bearing with an unwrapped or linearized observation density.

Tune on disjoint calibration/validation observations and fixed particle designs.
Freeze numerical controls and mixture beta before held-out evaluation. Test
truth (0,0) and the previously studied local parameter point
(-0.73272085,-0.01465508) separately on data1704 and two fresh datasets.
Start with four paired designs per cell; extend to eight only if the budget
permits and uncertainty needs it. Held-out errors never select controls.

The primary accuracy evidence is per-cell paired new-minus-old score-L2 error
against the same independent FP64 bootstrap/Fisher reference. Report likelihood
errors, every raw score, reference MCSE, replicate scores and a particle-count
refinement. Negative paired intervals support conditional improvement; intervals
including zero support no ranking. Reference bias/uncertainty can prevent a
ranking even when an interval is negative. Dataset-level uncertainty remains
limited; a single averaged number must not replace conditional comparisons.

Finite-program score finite differences are engineering checks, not an oracle
for the model score. Invalid values, factorization, reset moments, derivative
parity, target/input mismatch or missing artifacts invalidate the affected
candidate/comparison. A valid but inaccurate arm is a candidate rejection or
repair trigger, not rejection of surrogate HMC. Acceptance, HMC convergence,
posterior recovery, speed superiority and default readiness are not concluded.

Constructed cheap adversaries: (1) the prior/bootstrap proposal beta=(1,0,0),
which tests whether guidance helps; (2) the global Gaussian guide with the same
defensive floor, which tests whether the local flow earns its complexity; and
(3) the same-N independent bootstrap/Fisher filter, which checks adequacy
against plain filtering. Evaluate them separately at each held-out regime.
Heuristic losses veto broad promotion, not completion of this diagnostic.

## Defaults, tuning and skeptical audit

FP32/TF32 GPU/XLA is the owner default. Enable and verify memory growth before
device initialization. FP64/TF32-off is an explicit derivative/reference arm.
All N=1008 transport calls use the repository exact-divisor selector (K=N).
The new method uses the shared single-cloud adapter; no numerical model fork.

Candidate calibration compares the inherited 16/40/epsilon1 hypothesis with
8/40/epsilon1 and 16/40/epsilon0.5. Beta minimizes the existing independent-pilot
weight second-moment objective at early/middle/late observations. Defensive
beta0>=0.1 implies the pointwise prior/proposal ratio is at most10; it is a
bounded defensive hypothesis, not a globally optimal setting. Flow and entropy
controls are tuning hypotheses, not defaults transferred from another model.
Validation score/value reference error can select a viable complete candidate;
all references and uncertainties must be retained. Old controls start from the
two existing M13 hypotheses and remain scope-specific candidates. A repeated
rank/moment failure triggers a localized numerical investigation before any
larger run. No silent clipping, damping or replacement likelihood is allowed.

Pre-mortem: successful differentiation can still differentiate a biased filter;
FP32 validity can differ from the saved FP64 evidence; wrapped-angle guidance
can be locally poor; particle genealogies can bias the score reference. Check
model callback parity and fixed-design finite differences first. Keep reference
replications and refinement, freeze tuning, and report all cells and failures.
The audit passes for a bounded diagnostic, not for HMC or default promotion.

## Budget, execution and artifacts

At most four hours aggregate local worker wall time, including probes, focused
tests, tuning, reference refinement and held-out comparisons; at most24 worker
attempts. Bound children by the remaining budget and write completed rows
incrementally. Infrastructure repairs and expected candidate failures stay
within this allocation. No external services, environment mutation or paid
compute. A stop preserves evidence and states what remains incomplete.

Master: `scripts/run_range_bearing_covariance_comparison.py`, commands `check`,
`worker`, `run`, `report`. GPU commands use trusted/escalated execution. The
master records git/source hashes, exact commands, environment, data/design
seeds, tuning scopes, timing, GPU and allocator policy. Versioned root:
`docs/plans/artifacts/ledh-range-bearing-covariance-20261009-01`.
Result note: `docs/plans/ledh-range-bearing-covariance-results-2026-10-09.md`.
The active checkpoint tracks stage, verified findings, budget and next action.

Next within the existing master objective: if this test supports a viable
force, bind that force to the unchanged declared HMC endpoint, retune the HMC
kernel and measure its cost. This diagnostic does not launch a long HMC run.

## Executable protocol and pre-run review

Calibration observation seeds18010/18011, validation18012/18013 and held-out
1704/18020/18021 are disjoint. Observations and particle designs are rounded
once to FP32; FP64 references use those identical numbers. Calibration designs
are20001/20002; independent pilot seeds22001/22002,23001/23002. Validation
uses21001/21002,21101/21102; held-out31001--31004,31101--31104,31201--31204.
Validation selects minimum mean score-L2 error across eight rows; mean absolute
log-likelihood error breaks exact ties. These are bounded tuning hypotheses,
not exhaustive optima or canonical HMC tuning artifacts.

Held-out references use16 replicated bootstrap/Fisher runs at N16384 and65536.
Validation uses N16384 with16 replications. Same-N bootstrap comparisons use
every individual replicate, not the aggregate as a single N1008 competitor.
Five angular aliases suffice for the narrow angular noises of these two points.
Per-cell paired95% Student intervals use four designs (df3). The reference
sensitivity envelope expands each endpoint by twice the larger of coarse/fine
score displacement and three times the fine score MCSE norm. The triangle
inequality justifies this sensitivity factor; it does not bound particle bias.
Conditional ranking requires a stable sign. No across-dataset superiority claim.

Skeptical review before launch: shared callback binding and theta-independent
initialization, paired inputs and wrapped likelihood, disjoint seeds, scope
check before held-out execution, validity-aware selection, source hashes,
timeout and incremental writes checked. Syntax check passed. Runtime checks
start with N1008,T20 FP32 validity and N1008,T2 FP64 finite differences. User's
existing dirty source and core algorithm defaults are preserved.

Exact launch in trusted GPU environment tftwogpu:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_range_bearing_covariance_comparison.py run --particles 1008 --budget-seconds 14400 --gpu 0
```
Fresh worker processes save complete logs and manifests. If FP32 candidates
fail, local diagnosis and a newly tuned, explicitly labelled FP64 comparison
are allowed within the remaining budget; FP64 cannot support FP32 promotion.


Attempt02 exposed invalid covariance restoration in all three new FP32/TF32
calibration candidates (relative error 3.47e-4 versus guard 4.77e-5 in the
first candidate). No guard is relaxed. TF32 rounding is an unproved
explanation. The next precision diagnostic disables TF32, retunes both
methods, and repeats the held-out comparison in a fresh scope. If that fails,
the planned FP64 fallback is independently retuned. Neither precision
exception promotes the default FP32/TF32 route. Complete, unique, valid
held-out pairs and valid references are required for intervals; partial output
cannot earn a pass.

Budget after attempts01/02: 269.083 worker seconds and 5 worker attempts used;
14130.917 seconds and 19 attempts remain. Exact next launch:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_range_bearing_covariance_comparison.py run --output docs/plans/artifacts/ledh-range-bearing-covariance-20261009-03 --particles 1008 --budget-seconds 14100 --gpu 0 --no-tf32
```

Attempt03 completed in 246.013s (4 workers). FP32 without TF32 passed the
initial T20 check. All three new candidates nevertheless failed calibration
or validation. Candidate0 first failed validation with reset mean error
5.586e-5; candidate2 failed with mean error5.526e-5. Covariance errors at
those failures were1.661e-6 and1.093e-5. No held-out ranking is available.
Continue the predeclared FP64 fallback with new tuning for both arms.
Used515.096s and9 workers; remaining13884.904s and15 workers. Launch:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_range_bearing_covariance_comparison.py run --output docs/plans/artifacts/ledh-range-bearing-covariance-20261009-04 --particles 1008 --budget-seconds 13800 --gpu 0 --dtype float64 --no-tf32
```
This remains a precision diagnostic; numerical guards and promotion criteria
are unchanged. The altered precision is separately tuned, not transferred.


Attempt04 (FP64, TF32 off) completed in 241.274s. The comparator remained
valid, but every new candidate failed validation. The default and epsilon0.5
candidates reached validation seed18012/point1 and failed at step4 with reset
mean errors 1.819e-13 and 1.328e-13, above the FP64 guard 8.882e-14; flow8
failed calibration. This is a numerical validity failure, not evidence that
the method is worse. The guard remains unchanged. A bounded repair expands
reset iterations to 80 and 120 as fresh tuning hypotheses.

Attempt05: 80/120 reset iterations also failed calibration; total time over
attempts01--05 is1039.313s,17 workers. This rules out that tuning repair in this
scope. Seven workers and13360.687s remain. Next is a shared arithmetic repair,
not a change to the observation model or validity threshold.

For the documented reset, A=L_P L_S^{-1}, e_i=y_i-mean(y), and
z_i=mu+A e_i. In exact arithmetic mean(Ae)=A mean(e)=0. Explicitly forming
v_i=Ae_i and replacing v_i by v_i-mean(v) is therefore the same mathematical
map; its directional derivative is dv_i-mean(dv). Finite arithmetic can leave
mean(e) nonzero and A can amplify it. We will re-centre transformed deviations
and their analytical tangents before adding mu and dmu. The guard stays fixed.
This is a numerics-altering repair with a dedicated non-harm criterion: existing
healthy reset/value/score finite-difference tests must still pass (including
LGSSM and KSC), and the N1008 M13 diagnostic must satisfy the original guards.
No score-accuracy tuning is used to accept the arithmetic repair. The finite
program changes at roundoff scale, so all prior candidates stay archived and
both methods get a fresh tuning run before any held-out comparison. The method
remains optional. No change to canonical default or HMC consumer is made.
Skeptical audit: only the centring identity is changed; both direct and total
derivative terms are retained; failed source rank remains a hard validity
failure; the remedy is general across models. Explicitly inspect derivative
parity and moment validity before interpreting score-reference errors.


## Terminal execution

Attempt06 passed fresh FP64 tuning after the shared centring repair. Primary
pairs all valid; reporting separation repaired an incorrect dependence on the
failed global-guide screen, without altering samples or criteria. Attempt07
executed the predeclared eight-design extension with unchanged frozen tuning
and verified numerical dependency hashes. No heldout retuning, data/method
change or budget expansion. Its audit checks identical baseline/scope/reference,
separate primary validity and no posterior inference from filtering metrics.
It produced192rows and six complete primary cells; first-four replication is
exact. Five global-guide failures remain explicit. The planned eight-design
stopping point is reached. Final scientific interpretation:
ledh-range-bearing-covariance-results-2026-10-09.md.
