# KSC SV four-route comparison — 2026-09-28

## Authority, scope and budget

The owner requested retention of all four SQMC routes, a Git commit, thorough
review of the KSC SV testing plan, and execution. This is a new model scope;
the completed Kalman experiments will not be rerun. No environment change,
HMC, publication, push, default or scientific promotion is requested.

Use the unspent original aggregate cap: prior charge 21,377.329993 seconds,
remaining 21,822.670007 seconds (6.061853 GPU hours) before a provisional
300-second commit-hook reserve. The launch balance is 21,522.670007 seconds
(5.978519 GPU hours), inclusive of every new GPU check and failed attempt.
The reserve is conservative accounting, not measured GPU use. The owner
renewed elapsed execution for 48 hours on 2026-09-29 Hong Kong time.
Renewal start: 2026-09-28T16:15:40.010888+00:00.
New elapsed deadline: 2026-09-30T16:15:40.010888+00:00. CPU checks also stop there. At most
two infrastructure retries per unit; numerical candidate rejection is recorded
separately. A supervisor enforces both time limits, records subprocess wall
time conservatively as GPU time, and preserves partial evidence on timeout.
The elapsed renewal does not increase the aggregate GPU cap or retry limit.

Versioned outputs: `docs/plans/artifacts/sqmc-ksc-sv-20260928/attempt-NN/`.
Interpreter: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.
GPU execution requires escalation, verified memory growth, FP64, TF32 off,
and XLA with stable signatures. This is an explicit precision comparison,
not the production FP32/TF32 program. N=1008, exact chunk K=N, all two score
coordinates. Source commit, hashes, command, seeds, observations, random-input
hashes, tuning artifacts, device identity, allocator memory and timings are
mandatory. The bounded first stage covers T=10,20,50,120 from the older plan;
N=2016 and a larger parameter-regime study are deferred beyond this budget.

## Skeptical audit of the existing plan

The older `sqmc-oracle-comparison-master-program-2026-09-09.md` and
`sv-oracle-resolution-record-2026-09-14.md` incorrectly call the moment-matched
Kalman filter an exact reference for the current KSC mixture target. In
`ledh_canonical_models_tf.py::ksc_sv_canonical_model`, the proposal uses the
mixture moments, while the likelihood uses the seven-component log-sum-exp.
Replacing that mixture by one Gaussian changes the marginal likelihood and
its score. A parameter-independent transformation of observations preserves
scores only when the density family is unchanged; it cannot remove this gap.
The archived transform test does not establish Kalman exactness.

The same factory omits the direct parameter term 2*d(log_beta) from its
observation-value tangent. It must be added and checked against finite
differences before score comparisons. Replace NumPy scalar constants on this
touched runtime path with standard-library constants. Reuse the shared
canonical executor; do not create a reduced KSC filtering lane.

Disposition: old plan REVISE. Execute the corrected bounded plan only after
the reference and complete callback/consumer checks below pass. No expensive
comparison may use the old incorrect oracle assertion.

## Target and independent reference

The target is exactly the repository's two-parameter, one-state KSC mixture:
theta=(gamma_raw, log_beta), gamma=Phi(gamma_raw), Q=1; h0~N(0,1),
h_t=gamma*h_(t-1)+eta_t; transition BEFORE every observation;
z_t=h_t+2*log_beta+e_t, where e has the seven fixed weights, means and variances
in the existing factory. The evaluation point is theta=(1.5,0), giving
persistent volatility without an almost-unit transition. This is a declared
single-regime diagnostic, not a full empirical SV model or exact log-chi-square
target. Synthetic data come from this mixture, using TensorFlow stateless RNG.

Use one-dimensional deterministic grid integration of that mixture as the
accuracy reference. Preserve every quadrature likelihood and both scores.
For normalized filtering density f and derivative df, prediction is
g(x)=integral K_theta(x,u)f(u)du, dg=integral(dK*f+K*df)du. At fixed grid nodes,
dK/dgamma_raw=K*(x-gamma*u)*u*phi(gamma_raw). For observation density l,
d l/dlog_beta=2*sum_k w_k N_k(z-2log_beta-x)*(z-2log_beta-x-m_k)/v_k.
The update has a=g*l, da=dg*l+g*dl, Z=integral a, dlogZ=integral da/Z,
f=a/Z, df=da/Z-f*dlogZ. At t=1 prediction is N(0,1+gamma^2).
These recursions compute the derivative of the discretized mixture likelihood.

Convergence ladder: uniform trapezoidal grid 401,801,1201 on [-40,40], plus
1201 on [-48,48]. The last two reference differences must be below 1e-6
absolute log likelihood and 1e-5 per score coordinate on each dataset; retain
the finest accepted reference and its observed discrepancy. These are
numerical convergence checks, not a mathematical bound. Validate T=1,2 against
explicit Gaussian-mixture enumeration and both coordinates against centered
finite differences. Kalman using mixture moments is an explanatory comparator
only. No claim of exactness for the long-horizon grid approximation.

## Research intent and evidence contract

Question: how accurately do the four retained routes estimate this KSC mixture
likelihood and its two-parameter score at each horizon? Candidate mechanisms:
IID vs inverse-CDF vs permutation ancestry and the 0.97 coordinate-cap ablation.
Expected failures: nonlinear observation-density mismatch with the Gaussian
proposal, finite flow/transport bias, invalid moment corrections, incomplete
parameter tangents, and apparent wins caused by particular random seeds.

Primary descriptive outcomes: actual likelihood and both score coordinates,
per-coordinate absolute errors, score-vector L2 error, mean absolute likelihood
error. Accuracy comparator is the converged mixture reference. No method is
promoted or eliminated from descriptive rankings. A paired 95% t interval for
mean error differences is exploratory evidence, conditional on the model,
theta and selected controls; familywise ranking is not claimed. For e_i=||score_i-reference_i||_2, report
mean(e), sample SD(e), and SE(mean(e))=SD(e)/sqrt(8), with the replication
count explicit. Also report signed coordinate means and their SEs. The norm
of the mean error vector and the mean of error norms are different quantities;
do not label one as the other. These SEs describe variation over independent
dataset/design pairs, not conditional Monte Carlo variance for one fixed
dataset. Fixed-dataset uncertainty requires further design replications.

Constructed heuristic set: zero score (uninformative force); first-observation
score (ignores later data); moment-matched Kalman score (cheap Gaussian
approximation); IID route (standard stochastic-design comparator). Evaluate
each separately for every horizon and dataset/design pair. Any observed loss
to a heuristic vetoes promotion in that situation; it does not invalidate a
finite candidate or stop the other routes.

Validity/promotion vetoes: nonfinite values/scores, inconsistent directional
values, failed moment/reset guard, missing coordinates or provenance, tuning
scope mismatch. Reference nonconvergence, wrong target/timing, broken shared
derivatives, unsafe resources or exhausted budget are continuation vetoes.
Harness/serialization/XLA faults are repair triggers under the retry cap.
Finite large errors are candidate evidence, not infrastructure failures.
Runtime, allocations and tuning-selection frequency are explanatory only.

## Tuning, assumptions and smallest checks

Each route/horizon gets its own calibration and validation. Calibration seeds
211001,211002; validation 212001; final datasets 213001..213008 with distinct
paired filter seeds 214001..214008. The eight dataset/design pairs are
independent; routes share each pair for paired comparison. Calibration,
validation and final seeds never overlap. Full 16-seed replication from the
older plan is deferred; eight pairs are an explicitly bounded first stage.

Use the existing control family as warm-start hypotheses: flow steps 2/8,
epsilon .4/1.6/6.4/25.6/102.4, Sinkhorn 24, balance 12, ridge 1e-5, moment
correction strength .12 and pairwise strength .03, LM .01, trust radius .5,
pairwise RMS cap 2, coordinate cap .98 (.97 ablation). No control is a reviewed
KSC default. Existing dual-cap safeguards stay on; their local rationale is
to bound moment-correction displacement while preserving a finite proposal
whose realized density is corrected by the PF-PF weights. The smallest
calibration checks must expose nonfinite behavior, reset mass error and
score changes under flow/transport refinement. Do not choose settings on final
data or tune safety parameters against final accuracy. If controls cannot be
validated for a scope, record that scope as a failed candidate, not tuned.

Select the first epsilon passing both calibration mass checks (row error<=1e-8,
column TV<=1e-6); select flow resolution by calibration score L2, independently
per route/horizon. Validate the frozen choice on the disjoint validation seed.
The touched reusable tuning issuer must record the KSC data regime and exact
target scope, not the inherited LGSSM description.

After nominal tuning/validation, run a dedicated one-at-a-time sensitivity
check on calibration seed 211001, using the same random design and frozen
other controls: ridge and LM damping factors 0.1 and 10, and trust-radius
factors 0.5 and 2. Save every variant, invalidity, value and score change.
This diagnostic does not select safeguards by reference error. It evaluates
whether the numerical protection materially changes the computed program;
a sensitivity or invalidity finding blocks default/non-harm claims and
motivates a separately scoped repair. No protection is silently disabled.
The six checks per scope consume the same aggregate budget.

| Material assumption/control | Provenance and rationale | Risk, earliest diagnostic, status |
| --- | --- | --- |
| Q=1, h0~N(0,1), predict first, theta=(1.5,0) | Existing KSC factory; a declared persistent, tractable single-regime diagnostic | Not stationary initialization or a near-unit-root study; exact T1/T2 and saved simulation; scope assumption |
| Seven-component density | Existing canonical observation callback, copied independently into the reference | Different from native log-chi-square SV; callback density parity; target definition |
| Grid extent/resolution | Wide range relative to this model's innovation/prior scales, then explicit refinement | Joint truncation/discretization error; per-dataset convergence plus enumeration; approximate reference |
| N=1008, FP64, TF32 off | Prior bounded comparison count divisible by 2D; precision isolates algorithmic error | No particle-convergence or production-precision claim; N1008 GPU smoke; comparison choice |
| Flow 2/8, Sinkhorn24, balance12, epsilon ladder | Shared executor's finite-program control family; mass errors nominate transport resolution | Resolution bias; both flow candidates and mass diagnostics on calibration; hypotheses |
| Ridge1e-5, LM .01 and floor1e-4 | Positive regularization keeps factorization/LM systems away from singularity; magnitude uncalibrated for KSC | Alters finite program; one-at-a-time scope sensitivity above; hypothesis, not reviewed default |
| Trust .5, pairwise cap2, coordinate .98/.97, power8 | Shared bounded-displacement mechanisms; .97 is the retained ablation | Active caps can change proposal quality and derivatives; trust sensitivity, full-program FD, both cap routes; no optimum/non-harm claim |
| Correction strengths .12/.03, fixed iterations | Same declared moment-correction baseline in all applicable arms | Limited search may hide a viable repair; validity/trace and flow checks; hypothesis |
| Adaptive state map, Hilbert12, residual design | Shared deterministic route and Contract-E design, identical across comparable SQMC arms | Ordering changes and finite-design bias; fixed-input FD/graph parity; frozen mechanism |
| Eight final independent pairs | Bounded initial replication, separate from all calibration/validation | Small-sample and dataset/design variation; SD/SE and paired intervals; exploratory evidence |


Premortem: a successful command could compare different observation densities,
different timing, partial derivatives or an unconverged oracle. T1/T2 exact
mixture checks, callback finite differences, all-direction GPU graph/XLA
parity, finite-program differences at a tiny admissible N, and saved actual
inputs expose these failures before the final campaign. Cheap checks do not
establish score accuracy, nonlinear superiority, HMC readiness or defaults.

## Execution and terminal review

1. Preserve and commit completed Kalman work and the four-route retention note.
2. Implement the narrow target adapter/reference and repair the confirmed
   observation tangent; run CPU-hidden-GPU reference and derivative checks.
3. Run escalated NVIDIA and TensorFlow device checks, then bounded GPU/XLA
   call-chain parity at N16 and a bounded N1008 check for every route. Record
   elapsed/budget use before any larger launch.
4. Run T=10,20,50,120 sequentially with all four routes; new versioned attempt
   directories, frozen per-scope controls and independent final pairs. Stop
   children at either deadline; preserve every rejected/partial cell.
5. Assemble actual coordinates, absolute errors, likelihoods, heuristic verdicts,
   paired uncertainty, material changes, decision/inference tables and a
   terminal review. Review oracle convergence, target/callback wiring, data and
   tuning separation, accounting, hidden defaults and unsupported claims.

The review is Codex with executable checks unless a separately authorized
reviewer is available. Unavailability of an independent reviewer is recorded
as a limitation; a material scientific finding is repaired before continuing.

## Reviewed disposition and exact next commands

The five CPU reference/callback tests passed before the deadline (pytest 4.10s;
launcher 6.486s). No KSC GPU check or campaign has run. The original elapsed
window expired at **2026-09-28 07:25:28.470267 UTC**. That stop is historical: the owner has now granted 48 more elapsed hours.
The active renewed deadline is recorded above and in the shared budget ledger;
the aggregate GPU cap remains unchanged.

Static review and remaining executable checks are recorded in
`docs/benchmarks/sqmc-ksc-sv-plan-review-20260928.md`. Syntax-only inspection
passed after runner repairs. GPU call-chain equivalence, final-data reference
convergence, safety diagnostics and full comparison remain NOT CHECKED.

After renewal, record its exact start/deadline, update the runner DEADLINE and
shared `docs/plans/artifacts/sqmc-ksc-sv-20260928/budget.json` consistently, and
run the focused CPU tests again to cover the final source version. Then run
these commands with escalated GPU permissions from the repository root:

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_sqmc_ksc_comparison.py --mode check --output docs/plans/artifacts/sqmc-ksc-sv-20260928/attempt-02
/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_sqmc_ksc_comparison.py --mode campaign --output docs/plans/artifacts/sqmc-ksc-sv-20260928/attempt-02
```

Inspect the first command's complete checks before the second. The supervisor
requires matching-source passing GPU checks, shares one ledger across output
roots, records running attempts before launch, archives code, and leaves time
for shutdown inside the budgets. The internal worker modes are supervisor
implementation details; launch through check/campaign modes for accounting.
On an interrupted attempt, reconcile its process and charge before retrying.
After execution, assemble the IID cross-route heuristic comparison alongside
the three per-cell heuristics, full actual-score table and uncertainty table.
A completed cell or tuning failure is not authority to rerun exposed final
seeds after a method change; new data partitions are required for that repair.

Renewal execution note: checkout/branch/dirty KSC changes reverified; workspace
and /tmp write probes passed. Skeptical re-audit retains the corrected mixture
target, separate tuning/final seeds, reference convergence veto, explicit
FP64 comparison status, and bounded checks before campaign launch. No new
scientific or cost scope was inferred from the elapsed renewal.
