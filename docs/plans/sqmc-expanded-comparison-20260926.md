# SQMC horizon and full-matrix comparison — 2026-09-26

## Request and scientific question

Execute the owner's sequence: the existing P44 d=3 model at T=10,120;
then full-matrix d=3 at T=2,10,120; then full-matrix d=10 at T=2,10,120.
Compare likelihood and every analytical score component with the exact Kalman
likelihood and its derivative on identical observations. The question is how
finite-particle/numerical error changes with horizon, dimension, and the larger
parameter vector. This is an FP64 GPU/XLA reference comparison, not evidence
for the FP32/TF32 production setting, nonlinear models, HMC, or a new default.

Worktree: /home/chakwong/BayesFilter-SQMC, branch sqmc-development.
Existing uncommitted repairs remain intact. No merge or push is in this task.

## Models and parameter coordinates

All models predict before every observation, including the first:
x0~N(m0,P0), xt=A xt-1+et, yt=xt+vt, et~N(0,Q), vt~N(0,R).
The P44 family and its four existing coordinates are unchanged; N=1008 keeps
the preceding T=2 comparison's particle count.

The dense family uses all d*d entries of A, the d*(d+1)/2 entries of a lower
Cholesky factor L of Q, log(R scale), and an initial-mean scale. Diagonal L
coordinates are logarithms; off-diagonal L coordinates are unconstrained.
Q=L L^T, so every finite parameter vector has positive definite Q provided
the positive diagonal remains representable. Counts: 17 at d=3, 157 at d=10.
Scores are reported in these named coordinates, not mislabeled as derivatives
with respect to Q entries. H=I, P0=diag(linspace(.6,1,d)); R=exp(rho)I;
m0=mu*[(-.5)^i]. Baseline rho=log(.12), mu=.04.

Baseline A has diagonal linspace(.65,.45,d) and off-diagonal
.12*(-1)^(i+j)/(d-1). Its absolute row sums are <=.77, giving spectral radius
<1. Baseline L has diagonal sqrt(linspace(.16,.24,d)) and lower entries
.08*(-1)^(i+j)/sqrt(d-1). These fixed, data-independent values exercise dense
transition/covariance derivatives with moderate persistence. They are test
hypotheses, not estimates or MLEs. Use N=1020 for both dense dimensions:
N>=1000 and divisible by 2d for Contract E. Exact chunk extent is K=N.

## Evidence contract and intent ledger

All four existing routes are included: IID dual cap, inverse-CDF Hilbert,
permutation Hilbert, and permutation with coordinate cap .97 instead of .98.
That final route changes one cap; it does not remove the permutation.

Primary measurements: signed likelihood error, raw exact/estimated scores,
absolute component errors, per-component RMSE across final cases, whole-vector
L2 error, and block RMS for A, Cholesky-Q, R, m0. Relative score error is not
a scientific criterion. Report dimensions separately; an L2 norm grows with
the number of coordinates. No predeclared accuracy threshold exists, so the
run measures error and cannot certify scientific accuracy by finiteness.

Hard validity vetoes: wrong timing/parameter coordinates; mismatched data;
failed callback or Kalman derivative check; nonfinite result; inconsistent
primal values across analytical directions; missing component; mismatched
tuning scope; failed required replay. These invalidate the affected evidence.
A valid but inaccurate candidate is a repair trigger, not a reason to cancel
later requested scopes. Budget exhaustion, unsafe resource use, or a broken
oracle/shared implementation is a continuation veto. Do not tune on final data.

Explanatory diagnostics: runtime, absolute score-block errors, worst coordinate,
and numerical-settings sensitivity. Plain IID is the practitioner particle
baseline; exact Kalman is the certifying reference for this linear problem.
Also construct zero-score and first-observation-only Kalman score baselines,
which expose whether noisy estimates help beyond ignoring parameter evidence
or ignoring temporal accumulation. Evaluate them separately for each (d,T)
and data seed. Losing to these cheap baselines vetoes promotion; exact Kalman
dominates by construction. No generic SQMC superiority claim follows here.

Four final cases per route/scope: data seeds 197001,197002 crossed with filter
seeds 198001,198002. They permit descriptive error/variation comparisons only.
No statistically supported ranking is planned with just two independent data
sets. Comparisons share observations; randomized routes share frozen inputs
where their designs agree. Preserve all raw components in CSV and JSON.

## Calibration, assumptions, and numerical protections

For each exact model/dimension/horizon/N/route scope, use the existing shared
repository tuning workflow, two calibration seeds 195001,195002, and validation
seed 196001. Reserve final seeds before calibration. Select between flow
substeps 2 and 8 using calibration mean absolute score L2 error, validate the
selected candidate, then freeze all controls before final evaluation. Final
filter repetitions use an issued artifact with exact scope verification.
This is a small scope-specific calibration, not exhaustive tuning or admission.

Other controls start from the repaired diagnostic settings: epsilon=.4,
Sinkhorn=24, balance=12, correction steps=1/strength=.12, pairwise steps=1/
strength=.03; Contract E ridge=1e-5, LM damping=.01, scale floor=1e-4,
trust radius=.5, pairwise RMS cap=2, cap power=8, adaptive empirical chart,
Hilbert bits=12. Provenance: preceding repaired diagnostic, NOT a transferable
optimum. They are explicit baseline hypotheses. Ridge versus covariance scale,
flow-grid sensitivity, finite directional parity, and held-out error expose
possible non-negligible distortion. No setting is promoted to a default.
The radius/caps bound correction displacement; positive LM and reset ridge
protect solves. Their quantitative adequacy in these new scopes is unproved.
Preserve that limitation even if the two-setting calibration succeeds.

Q Cholesky parameters guarantee SPD but not a well-conditioned covariance;
record eigenvalue bounds/condition number. A stability follows from the stated
row-sum bound; record its norm. Identity H/full observations and Gaussian data
make Kalman exact but omit partial observation and nonlinear failure modes.
FP64 isolates numerical accuracy at higher cost; it does not validate TF32.
The two final datasets are independent of tuning but too few for ranking.

## Implementation and checks

1. Add a full-matrix LGSSM specification using the existing shared canonical
   value/analytical-score executor. Never replace its analytical score by AD.
   Implement dQ=dL L^T+L dL^T and total Gaussian density tangents, including
   determinant and inverse-covariance terms. Name all coordinates explicitly.
2. CPU-only focused reference checks (CUDA_VISIBLE_DEVICES=-1): callback finite
   differences for A, diagonal/off-diagonal L, R, and initial mean; exact Kalman
   score against central differences; known diagonal specialization; simulator
   shape/timing. Check all coordinate callbacks at d=3 and d=10. Use fixed
   directions for finite-program parity to keep the test bounded.
3. Trusted GPU probe and memory-growth verification before initialization.
   GPU UUID selects the idle RTX4080SUPER explicitly. FP64, TF32 off, XLA on.
   Stable TensorFlow signatures; no pfor or NumPy runtime. A bounded T=2
   full-parameter pilot checks executable call-chain/parity and measured cost.
4. Execute the eight requested scopes in order, in fresh subprocesses with
   per-scope/route incremental results. Small fixtures are mechanics only;
   every reported accuracy comparison uses >=1000 particles.
5. Audit completeness, source/input hashes, score dimensions and exact scope;
   assemble actual-score tables, error summaries and an honest result note.

## Budget, output, recovery

Total budget: at most 12 GPU-hours / 14 elapsed hours, including calibration,
compilation, pilot and retries; at most two infrastructure retries per failed
unit within that budget. Each child has an external timeout of 90 minutes;
the supervisor enforces the total remaining budget. Reuse frozen completed
units after a localized harness repair; preserve failed attempts. The timed
pilot projects the T=120 cost before expensive launches. If the projection
cannot fit, stop before committing the remaining compute and report the
actual cost; do not silently drop parameters, routes, data, or particle count.

Versioned output root: docs/plans/artifacts/sqmc-expanded-20260926/attempt-01/.
Save every child stdout/stderr log, command, return code, wall time, source
closure hashes, Git SHA/dirty status, TF/Python/GPU identity, memory policy,
data/filter seeds and observations, controls, tuning artifacts and all scores.
Runner: docs/benchmarks/run_sqmc_expanded_comparison.py.
Result: docs/benchmarks/sqmc-expanded-results-20260926.md.
Update the active reset checkpoint after implementation and each scope.

## Skeptical review before implementation — PASS with explicit limitations

Reviewed against wrong baselines, proxy promotion, fairness, stale context,
environment mismatch, hidden defaults, runtime and invalid stop conditions.
The old diagonal parameterization cannot answer the full-A/Q question: add an
explicit full specification. Keep P44 unchanged, identify score coordinates,
use the matching predict-first Kalman reference and require callback/parity
checks. Inherited controls are not tuned for a new scope: calibrate each scope
on separate data and state the limited search. Larger parameter count changes
error-norm scale: include component/block RMS. Two datasets cannot establish
ranking: prohibit it. GPU ordinal ambiguity is resolved with UUID selection.
Preserve exact logs and an external timeout, correcting the previous harness
limitations. A candidate's poor accuracy does not invalidate later phases.
The design answers the requested descriptive comparison without promoting
unreviewed defaults or promising affordable runtime before measuring it.
