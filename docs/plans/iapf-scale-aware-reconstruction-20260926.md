# iAPF scale-aware reconstruction: bounded independent reference campaign

Date: 2026-09-26. Owner requested a plan, review, and execution. Status: COMPLETE negative diagnostic;
see the terminal result and retained pre-execution review below. This supplements the authoritative
`iapf-kdm-replication-master-program-2026-09-24.md`; KDM work is unchanged.

## Question and evidence contract

Can a scale-aware optimizer obtain useful finite local solutions of the paper's
unchanged Equation (15) on Algorithm 3 backward inputs, and can those solutions
support a bounded Algorithm 4 diagnostic? The author's optimizer and positive
floor formula remain unknown. This is a disclosed reconstruction, not recovery
of those missing choices or completion of the scientific replication.

For density vector p and positive target y, profile lambda=(p'y)/(y'y).
The objective remains L=||p-lambda*y||^2. A positive residual scale, fixed during
each local step and every trial evaluation for that step, changes neither its
minimizers nor its derivatives after undoing the scale. Recompute that scale
only between accepted steps. Do not minimize normalized shape error instead.
Record original log(L), gradient divided by mean(p^2), shape error, parameter
movement, density-norm change, damping, and rejection counts. Absolute small
loss or gradient alone cannot certify convergence.

The primary numerical criterion is recovery from perturbed exact-Gaussian
starts with parameter error <=1e-4 and relative gradient <=1e-6, plus analytic
versus finite-difference gradient agreement <=1e-5 and original-objective
descent on every accepted step (roundoff tolerance 1e-10 in log loss).
An exact starting solution tests preservation only. The relative-gradient
tolerance is a hypothesis to be checked against recovery and derivative noise;
it does not become valid merely by producing more passing fits.

Nonfinite values, invalid covariance, failed derivative/descent controls, wrong
filter/Kalman wiring, or missing provenance invalidate that experiment and are
continuation vetoes until repaired. A fit reaching its iteration cap, a poor
shape approximation, a floor-heavy proposal, or a candidate failing to finish
Algorithm 4 rejects that candidate and triggers diagnosis; it does not reject
iAPF or prevent another predeclared arm. Shape error, target ESS, density-norm
change and floor-mixture quantiles explain results and never substitute for
downstream likelihood accuracy. A stationary fit must also have no negative
curvature detected by a finite-difference Hessian check in the d=5 lane;
near-flat directions remain explicitly uncertain.

No statistical superiority, default readiness, full paper replication, or
TF/XLA readiness can be concluded. Result artifacts preserve all failures and
distinguish finite numerical qualification from downstream performance.

## Phases and defaults under test

1. Implement a separate R/FP64 diagnostic solver using profiled lambda,
   diagonal variance in log coordinates, fixed start-based parameter scales,
   and Levenberg-Marquardt steps. Its damping starts at 1e-3, changes only by
   actual/predicted loss agreement, and uses a relative Hessian-diagonal floor.
   These are disclosed numerical hypotheses, not paper settings. Controls use
   exact Gaussian inputs in d=1,5,20,80, two perturbation sizes (0.05,0.20),
   and fresh deterministic seeds. Recovery clouds deliberately sample near the
   known Gaussian center (relative SD 0.5, or 0.35 at d=80) to keep this
   identifiability control separate from particle-cloud coverage. This is a
   convenience fixture, not evidence on arbitrary filter clouds.
   Compare fixed residual scalings and verify
   the known exact-twist filter against Kalman truth. Baseline: old absolute
   stopping is retained as historical evidence, not an admissible solver.
2. With numerics frozen, collect genuine first-sweep backward inputs from the
   paper's d=5,T=100,N=1000 model on two calibration data seeds. Fit each
   backward recursion for two floor hypotheses: minimum base-component mass
   delta=1/N^2 and delta=1/N. Set c=delta/(1-delta)*max_mu N(mu;m,I+V).
   This formula supplies a known lower bound for the base mixture probability;
   distant parents can have much larger probabilities. Both are sensitivity
   arms, not ranked winners. Save every input and exact failure context.
3. For each arm completing phase 2, run one independent full adaptive pilot
   on a fresh dataset: N0=1000,k=5,tau=kappa=0.5, stopping window 6, sample SD,
   doubling after k, maximum 20 iterations and 16000 particles. These are paper
   settings plus disclosed resource caps and indexing reconstruction. Use the
   shared reference controller and its independent final filter draw. No oracle
   twists or silent fallback in fitted runs. A failed fit may receive one local
   retry from the same start with 800 rather than 400 steps; no altered loss,
   data, floor, tolerance, or objective. Record both attempts.
4. Whether or not a pilot completes, validate and report the cheap comparator
   set on the fresh dataset. If a pilot completes, freeze its learned twists
   and evaluate 8 independent draws at its final particle count against all
   comparators at that same count. Report fit/adaptation overhead separately.
   BPF (plain transition), observation-only twisted filtering (uses only current
   observation), and fully adapted APF (exact one-step conditioning) are the
   constructed heuristic adversaries. Exact Kalman likelihood is the oracle.
   Evaluate incremental log-likelihood errors separately for ordinary and
   large Kalman innovations (chi-square 95% threshold), and total errors. Treat
   these few-seed differences as descriptive; do not rank or tune on the
   heuristic comparison. A clear conditional loss to a cheap comparator vetoes
   promotion; passing this small diagnostic cannot authorize promotion either.

Floor protection is assessed separately from whole-proposal performance. At the
parent attaining the Gaussian convolution peak, the mixture mass equals delta.
In that exact healthy case the second-moment inflation from mixing delta of the base
proposal is bounded by 1/(1-delta), since q_mix >= (1-delta)*q_exact. This bound
is a local non-harm check, not a global performance guarantee. Away from that
parent the bound uses the actual mixture probability, not delta. Preserve the
earlier underdispersed floor quadrature evidence and its limitations. No floor
is selected by optimizing a heuristic comparison.

## Execution, budget and provenance

Independent R reference only; CPU deliberately selected and GPU hidden. The
repository default remains TensorFlow/TFP GPU/XLA. Python supervision uses only
the standard library. No NumPy or package/environment mutation. Freeze and hash
all executed sources and inputs, record both checkout commits/branches, R
version, seeds, command, CPU/GPU status, wall time and result paths per attempt.

Data seeds: calibration 9262001 and 9262002; pilot 9263001. Numerical controls
9261000 plus case index. Filter seed families 9264000 and 9265000. Use unique
attempt directories under
`docs/plans/artifacts/iapf-scale-aware-reconstruction-20260926-01/`.
Runner: `scripts/run_iapf_scale_aware_reconstruction.py --phase controls`
then `--phase backward`, then `--phase pilot`, each with an explicit timeout.
Use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python` from the planning root.

Campaign cap: 1500 CPU wall seconds, GPU 0, at most 6 launches including
localized repairs. Existing combined CPU charge 89.936265 seconds leaves
1710.063735 seconds of the 1800-second overall CPU cap. Charge every failed
attempt; the new campaign leaves at least 210 seconds reserve. Stop on budget,
unrepaired invalidity, or exhausted attempt cap. Candidate failure is not a
continuation veto. Update result, master and checkpoint at close.

## Skeptical review before execution

Self-review (no independent reviewer claimed): the prior universal shape gate
was a wrong baseline for numerical qualification, and exact-start tests did not
test recovery. Both are corrected prospectively; prior results stay unchanged.
The raw objective has a zero-density infimum, so a converged numerical run does
not establish a global useful minimum. Relative stopping, perturbed recovery,
curvature checks, actual backward inputs and full-filter measurements separate
these questions. An oracle target cannot replace Algorithm 3's evolving target.
Three cheap comparators and innovation-conditioned reporting prevent an
internally consistent but ineffective method from being promoted. Equal N is
not equal cost: record adaptation and filter runtimes; do not claim efficiency.
One pilot and 8 draws cannot reproduce the paper's 1000-replicate experiment.
The floor and solver are unresolved author choices, R is a reference exception,
and d=5 cannot establish d=80 validity. Explicit caps and no fallback prevent
unbounded execution or false success. Decision: PASS for this limited diagnostic.

Implementation review before launch: checked the consumer chain from the driver
through sa_backward to sa_fit, and from the adaptive callback to the shared
iapf_iterate controller. Numerical controls execute that same solver and the
same shared filter against exact-twist/Kalman authority; real backward and
pilot evidence remain pending. No alternate objective or fallback is wired.
Trial acceptance holds residual scale fixed; damping uses a relative Hessian
scale. Fixed rescaling should leave steps unchanged. Curvature checks can detect
negative curvature but cannot prove a unique/global minimum. Corrected an
initially overbroad floor bound: delta is the actual mass only at the maximizing
parent. Review: PASS for this reference campaign, subject to executable controls.

Pre-mortem: the solver could descend into diffuse densities, pass only easy
exact data, or find positive-curvature local fits that make poor proposals.
Actual sweeps and Kalman/heuristic diagnostics distinguish those outcomes.
Failure can reflect the objective, local basin, initialization or damping;
save inputs and trajectories before attributing it to the scientific method.


## Attempt 01 result and scope clarification before further execution

Attempt 01 charged 10.491983 CPU seconds. All derivative, fixed-scale invariance,
profiled/joint identity, floor and exact-twist/Kalman controls passed. Exact
parameter recovery passed both perturbations in d=1,5 and the smaller
perturbation in d=20,80. Larger d=20,80 perturbations escaped to diffuse fits,
hit 400 iterations and retained large relative gradients. Original loss descent
passed. These are failed local-basin candidates, not derivative/harness failures.
The all-dimension numerical qualification is FAILED and stays failed.

The initial executable gate incorrectly conflated that broad qualification
with permission to perform the already-planned d=5 diagnostic. The d=5 numerical
criterion passed unchanged. Prospectively emit separate d=5 eligibility and
all-dimension qualification; require the former for d=5 sweeps/pilots. No
tolerance, data, fit, floor, or objective change. Re-execute controls with the
revised reporting to verify the executable scope gate. This is an explicit plan
correction, not a relabeling of attempt 01 or a broad solver promotion.

Skeptical re-review: high-dimensional failures are material for future d=20/80
replication and must headline those scopes. They do not show invalid math or
numerics in the checked d=5 route. The next experiment uses actual d=5 inputs
to test precisely the missing applicability; a candidate failing there remains
ineligible for the adaptive pilot. PASS to continue the d=5 diagnostic only.


## Actual-input result and bounded damping repair, before execution

Attempt 03: both arms on both seeds failed at backward time 98, after two
stationary fits each. All four 400-step fits and same-start 800-step retries
followed the original objective into diffuse densities. Charge 7.182504 seconds.
No arm is eligible for a pilot. This rejects the current reconstruction in d=5
under these starts; it does not invalidate the tested derivatives or iAPF.

Use one remaining launch for a localized optimizer-mechanism test on the four
saved failure inputs, before the comparator phase. Hypothesis: initial damping
1e-3 takes the method out of the useful local basin; compare initial damping
1, 1000 and 1000000, each with at most 800 steps and the same raw objective,
parameterization, QR start, relative stopping and curvature checks. This changes
the path of numerical optimization and is explicitly a candidate change.
Damping thereafter follows the same agreement rule. No bounds or regularization
of the objective. For each candidate rerun both frozen d=5 perturbed controls
and finite-difference gradient checks on all four real inputs.

A candidate may nominate only if all four saved inputs are stationary with
relative gradient <=1e-6, valid curvature and original loss descent, both d=5
recovery controls retain parameter error <=1e-4, and real-input derivative
errors <=1e-5. If more than one passes, use the least changed damping (ascending
order), not shape or downstream performance, as a disclosed representative.
Then rerun both complete calibration sweeps before any pilot. If none passes,
perform only the planned independent comparator evaluation and preserve the
unresolved local-basin question. Maximum remains six launches and 1500 CPU
seconds; three launches have used 27.969601 seconds.

Skeptical review: increasing iteration budget already failed, so another long
identical retry would not distinguish a mechanism. The damping experiment can
test path sensitivity without changing Eq15; failure cannot prove absence of a
finite local minimum. Inputs are calibration data and the pilot dataset is
still untouched. PASS for the bounded repair. This is the final repair in this
campaign; unresolved source choices or objective restrictions require a new
explicit scientific plan, not silent constraint insertion.


## Terminal execution record

Six launches complete, including a read-only terminal replay/hash audit. Total
CPU wall 41.725506s, GPU0. All four actual backward sweeps failed at time98 and
all12 stronger-damping fits failed; no adaptive candidate was admitted. The
baseline-only branch completed24 runs. Numerical controls and terminal audit
passed within their recorded scope. See
[full result](artifacts/iapf-scale-aware-reconstruction-20260926-01/result.md)
and [decision](artifacts/iapf-scale-aware-reconstruction-20260926-01/decision.json).

Reporting clarification: code records the mean Eq15 residual square, L/N;
the paper uses the sum. Fixed-N scaling preserves the optimization problem.
The observation-only and fully adapted comparators are equivalent for this
model, so their labels are not independent heuristic evidence. The new fitting
candidate is rejected; the iAPF research direction is not rejected. Further
finite-branch/source investigation needs its own bounded plan. Do not recycle
this campaign's exhausted six-launch allowance.
