# Phase 0E: separate fitting, twist approximation, and score error

Owner request, 2026-09-18: create, review, and execute the proposed diagnosis.
This is a new bounded campaign after the completed public-code comparison and
closed September 17 campaign. Main checkout: `/home/chakwong/BayesFilter`,
branch `surrogate-hmc`. Preserve unrelated edits. No subagents or external
review is needed; the skeptical review below is the requested plan review.

## Question and evidence contract

Why does the curved scalar iAPF return an inaccurate model score: incomplete
optimization, Gaussian/bound/floor approximation, too few particles, or the
derivative's treatment of discrete sampling? The target is the six-parameter
score of the physical likelihood, checked against the existing refined-grid
forward recursion. A derivative of the frozen finite program is a separate
target. KDM kernels and runtime defaults remain fixed.

Comparators: the original bounded density objective, relative-shape objective
with original controls, a bounded fitting-control candidate, and the actual
repository EKF, UKF, bootstrap, and local-linear consumers. An additional
bootstrap importance sampler with resampling disabled isolates discrete
resampling. It is useful for the short horizon; it is not a proposed general
replacement for resampling.

Primary diagnostic outcomes are (a) reproducible convergence changes on the
same fitting cloud, (b) approximation residual against a backward grid
reference, and (c) likelihood and score errors as particle count grows with
coefficients held fixed. These nominate the next repair; they cannot promote
an algorithm. Fresh validation and untouched confirmation test the nominated
controls downstream. A promotion would require no fit/validity/heuristic veto,
replicated model-score agreement, and an eligible scope-specific tuning study;
this diagnostic campaign issues no tuning/admission artifact.

| Diagnostic | Role and response |
|---|---|
| Forward grid mesh/domain relative error > 1e-7 or tail mass > 1e-9 | Continuation veto for that dataset; repair reference before interpreting it |
| Backward/forward likelihood disagreement > 1e-7 relative | Harness continuation veto |
| Nonfinite output, invalid covariance, failed coefficient cast | Reject affected candidate; preserve diagnostics; independent arms continue |
| Fit projected gradient > 1e-7 at cap or line-search failure | Candidate veto and solver repair trigger, not direction rejection |
| Active fit bound | Promotion veto, explanatory diagnostic and bounds repair trigger |
| Gaussian shape residual, floor contribution, fit iterations | Explanatory only; never substitute for downstream score accuracy |
| Same-stream finite differences fail to stabilize | Investigate categorical branch changes before alleging wrong analytical calculus |
| Larger observed score error than any heuristic | Promotion veto on that dataset; conditional uncertainty also reported |
| Mean score excludes reference as N grows while values improve | Evidence motivating score-construction repair; not proof of a universal asymptotic limit |

No claim of canonical LEDH conformance, HMC readiness, default readiness,
unbiased expected gradients, broad method superiority, or globally optimal
Gaussian fitting follows from this campaign.

## Mathematical discrimination

The model is x_t = a*x_(t-1) + c*sin(x_(t-1)) + sqrt(Q)*e_t and
y_t = h*x_t + b*x_t^2 + sqrt(R)*v_t. Existing six-parameter mappings and
initial law are used without alteration. The backward function obeys
psi_T(x)=g_T(x), psi_t(x)=g_t(x)*integral f(z|x)*psi_(t+1)(z) dz.
At the final time, log(g_T(x)) has quartic coefficient -b^2/(2R).
Thus for b nonzero it is not exactly a single Gaussian log density on an
interval. This algebra does not show that a Gaussian approximation is poor
where the predictive distribution puts mass. Report predictive-mass-weighted
residuals and the actual downstream errors.

On a fixed cloud, the relative-shape objective is
Rshape = 1 - (sum p_i*y_i)^2/(sum p_i^2 * sum y_i^2).
The existing projected-gradient fitter and its analytical derivative are used.
A diagnostic two-dimensional parameter-box search supplies an independent
feasible Gaussian competitor. It can expose a poor local fit but cannot prove
global optimality or an irreducible Gaussian-family error. Evaluate both pure
Gaussian and Gaussian-plus-floor shapes against the backward reference.

For a discrete draw K with probabilities q_k(theta),
d/dtheta E[F_K(theta)] = sum q_k*dF_k/dtheta + sum F_k*dq_k/dtheta.
The existing particle kernels differentiate fixed realized ancestors and
mixture labels. The second term is absent from that pathwise derivative and
need not vanish. The code makes this limitation explicit. Finite differences
of a locally unchanged program check the first term, not the derivative of
the expectation. The no-resampling control, repeated particle ladder, and
separate value/score errors test whether this matters in the present model.
Finite-N normalized importance scores still have ratio-estimation error.

Source anchors: Guarniero--Johansen--Lee, local
`.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.txt`,
Propositions 1 and 4, Algorithm 3, and Section 5.1 equations (15)--(16).
Public source reconciliation and checked differences are in
`younis-iapf-kdm-reference-comparison-2026-09-18.md` and its result.
Repository call chain: `iapf_adapter.execute_iapf` ->
`make_density_recursive_fit_kernel` / `make_fitted_twist_kernel`;
`make_particle_filter(..., resampling=False)` is the existing control.
New backward quadrature is explicitly a diagnostic reference, not a filter
runtime fork. It must reproduce the existing forward likelihood.

## Stages and frozen choices

Use T=2, d=o=1, theta=(.62,-.8,-.6,.9,.25,-.3), weak (c,b)=(.12,.04),
curved (.35,.12). Master seed 9182026. Fresh dataset IDs: calibration
1300/1301, validation 1310/1311, confirmation 1320/1321 (weak/curved).
Do not inspect confirmation observations or outputs before freezing controls.
Never reuse previous claim IDs 1120--1125 or 1220--1221.

1. Reference and fitting localization. Check 201/401 grid meshes at radius 9
   and 601 points at radius 13.5, including forward/backward agreement. On
   identical bootstrap clouds, compare relative-shape caps 2000, 5000, 10000
   at unchanged projected-gradient tolerance 1e-7. Compare the original
   density fit and widened bounds on those same clouds. Separately fit the
   exact backward function at 257 deterministic predictive quantiles; compare
   with an 81x81 bounded parameter search followed by two local refinements.
2. Actual consumer fitting. Run the existing adaptive iAPF adapter with all
   controls explicit. Candidate controls are original relative-shape, wider
   bounds, and floor ratios .001 and .1 versus .01, all with a 10000 cap.
   Retain density/2000 and relative/2000 as historical-control comparators
   executed afresh. Choose among converged, interior relative-shape candidates
   by smallest worst-regime predictive-grid shape residual on calibration.
   This is a nomination criterion only. If none is interior, record no
   admissible nomination, retain the original relative-shape/10000 arm as a
   diagnostic representative, and continue score localization.
3. Freeze all control selection. On calibration, run N=16,64,256,1024,4096
   with 32 paired final seeds and fixed fitted coefficients. Include original
   density, plain relative-shape/10000, and nominated/representative controls
   where fits are valid. Record exact finite-program FD checks on a small
   FP64 case at h=1e-4,1e-5,1e-6. Evaluate the five heuristic controls at
   every N; EKF/UKF are deterministic. The no-resampling control isolates
   label dependence. No particle-size result becomes a tuned scope artifact.
4. Fresh downstream verification. Fit with the frozen controls on validation,
   then untouched confirmation, and evaluate N=4096 with 32 independent
   paired final seeds. Keep baseline arms and all heuristic controls visible.
   Failure triggers the identified next repair; it is not authority to tune
   on confirmation. Only if score bias remains ambiguous after N=4096,
   allow a predeclared N=16384 diagnostic at the same 32 seeds and controls
   within remaining budget; it cannot become a selected setting.

Report Monte Carlo means, standard errors, coordinatewise reference errors,
value and score squared errors, and paired differences with 99% bootstrap
intervals (2000 fixed-seed resamples). Conditional inference is restricted to
these fixed datasets and fits. One dataset per regime/partition cannot support
population-wide rankings. Heuristic losses veto promotion even when the
paired interval is inconclusive. Do not select controls on heuristic scores.

## Default and assumption audit

| Choice / provenance | Justification, failure mode, early diagnostic, status |
|---|---|
| Existing scalar model, T=2, theta and two regimes | Tractable reference and previously failing mechanism; cannot establish long-horizon behavior; grid checks first; mechanism fixture |
| Nfit=16, adaptive cap128, k1, tau100, max iterations4 | Reproduce prior fitting regime; small clouds can mislead; compare predictive-grid fit and N ladder; baseline hypothesis |
| Bounds mean +/-4, sd [.2,4]; widened +/-8, [.1,8] | Explicit factor-two boundary sensitivity, not established defaults; watch active constraints and residuals; diagnostic hypotheses |
| Floor .01 with .001/.1 arms | One decade each side tests regularization contribution; floor changes the proposal, so preserve exact corrections; diagnostics only |
| 2000/5000/10000, 30 backtracks, tolerance1e-7 | 4201-step prior finding motivates cap ladder; relative-shape standardized parameter gradient has fixed scale; record gradient and termination; bounded hypotheses |
| FP64 fitting and reference | Separate optimization/model issues from FP32 arithmetic; no production precision claim; exact/FD/reference checks |
| GPU/XLA FP32/TF32 final consumer | Existing target execution policy; coefficient cast and value/score cross-checks; declared target, not promoted default readiness |
| Fixed coefficients across N | Isolate final particle error; not fully retuned iAPF at each N; record scope limitation |
| 32 seeds, two regimes | Bounded conditional Monte Carlo evidence; rare tails and data variation remain uncertain; no broad ranking |

Heuristic construction: EKF is the cheapest first-order Gaussian approximation;
UKF tests whether sigma-point curvature is already sufficient; bootstrap uses
the physical transition; local-linear uses the current observation; bootstrap
without resampling preserves continuous path derivatives at this short horizon.
Evaluate each regime separately. Refined-grid distance remains the main
downstream accuracy measure.

## Skeptical review before execution

Reviewed 2026-09-18: revised to avoid three misleading designs. Merely raising
the optimizer cap cannot diagnose a Gaussian-family limitation; add backward
reference fits and an independent feasible box-search competitor. A generic
finite-difference pass cannot establish a model-score claim; add the existing
no-resampling control and separate likelihood/score ladders. A best-looking
calibration score cannot justify promotion; select fitting controls using only
calibration shape/validity and freeze them before fresh downstream verification.
Boundary failure is a promotion veto, not a continuation veto. The plan retains
failed candidates, tests the actual consumer call chain, checks the numerical
reference, and explicitly limits its inference. Review verdict: proceed with
this bounded diagnostic campaign; no material unexamined default is promoted.

## Execution, budget, and artifacts

Driver: `docs/benchmarks/diagnose_younis_iapf_curved.py`.
Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.
GPU commands use escalation; first record `nvidia-smi` and an escalated TF
device/memory-growth probe. Select RTX 5080 by UUID, never guessed ordinal.
All GPU processes set `TF_FORCE_GPU_ALLOW_GROWTH=true` before import and call
the repository runtime policy helper. CPU reference/tests set
`CUDA_VISIBLE_DEVICES=-1` when run in a separate process. In the GPU process,
FP64 grid/data work explicitly uses CPU; reference placement is recorded.

Budget: at most four numerical driver attempts, 2400 total driver wall seconds,
600 CPU test/probe seconds, 40 adaptive consumer fits, 320 fixed-cloud fits,
and 3500 final filter evaluations. Check cumulative elapsed time and counts
between blocks. Infrastructure repair/retry under this contract is authorized;
new output directory per attempt. Stop for exhausted budget, invalid harness,
or unavailable required reference. No package install or external publication.

Output root:
`docs/plans/artifacts/younis-iapf-curved-diagnosis-20260918-01/`.
Preserve manifest (commit, source hashes, command, environment, seeds, device,
TF32/XLA/memory policy, data identity, wall time), incremental results,
selection, decision/inference tables, complete logs, and concise checkpoint.
The driver records reference and candidate rejection separately and saves after
each substantive block. Update the master/checkpoint at completion.
