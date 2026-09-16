# Phase 0E: explicit relative-shape fitting repair

Owner instruction, 2026-09-17: continue execution of the updated master.
This continues the existing 280-charge, three-launch, 3,000 GPU-wall-second,
7,200 CPU-second campaign. Prior consumption is 171 charges, one launch,
71.30 GPU wall seconds and 669.73 conservatively charged CPU seconds.
Source starts from clean commit 418e5388 in the isolated
`younis-score-iapf-density-repair-20260917` worktree. Completed source and
artifacts remain preserved. No external reviewer or additional agent is needed.

## Question and source reconciliation

Does explicitly fitting relative shape avoid amplitude-collapse failure and
produce a numerically viable iAPF candidate on fresh nonlinear scalar data?
The comparator is the previously declared bounded Eq. (15) density fit, with
its original controls. This is a fitting-objective comparison inside Algorithm
3, not a claim that the repaired objective equals Eq. (15).

Primary source inspected: Guarniero, Johansen and Lee (2017), local PDF/text
`.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter`.
Proposition 1 and its telescoping proof (text lines 260–312) preserve the
likelihood for bounded, continuous, positive twists with the stated correction.
Algorithm 3 and the immediately following discussion (lines 651–673) permit
different approximation procedures. Section 5.1, Eqs. (15)–(16), lines 911–954,
chooses density least squares and a positive Gaussian floor. Proposition 4's
backward recursion remains unchanged. Author code has not been located; author
code parity is not checked. An attempted fresh web search returned HTTP 503;
the technical conclusions here use the inspected local primary paper.

For finite cloud values p_i > 0 and y_i > 0, let a=sum(p_i^2), b=sum(p_i y_i),
c=sum(y_i^2). Profiling scale gives lambda=b/c and L=a-b^2/c.
The previous result derives how L can approach zero by shrinking Gaussian
amplitude without matching shape. The proposed objective is explicitly

    R = L/a = sum((p-lambda*y)^2)/sum(p^2).

It is invariant to positive amplitude rescaling. With r=p-lambda*y and
s_i=gradient(log p_i), the analytical gradient is

    gradient(R) = 2/a * sum((r_i*p_i - R*p_i^2)*s_i).

The profiled-scale derivative cancels because sum(r_i*y_i)=0. For Gaussian
mean m and log standard deviation l, s_m=(z-m)*exp(-2*l) and
s_l=((z-m)*exp(-l))^2-1. Evaluate p relative to its maximum log density to
avoid amplitude underflow; the common scaling cancels from R and its gradient.
The original density loss remains an emitted diagnostic. A zero original loss
does not invalidate this different optimization if its actual objective and
gradient remain valid; the existing density-objective guard remains unchanged
for the comparator. No clipping, damping, pfor, or autodiff score is introduced.

The Gaussian-plus-positive-floor twist remains bounded, continuous and positive
for accepted finite SPD fits. The transition/weight correction and frozen-fit
analytical final score are unchanged. Proposition 1 concerns the value target;
it does not make the finite-program score unbiased for the model score.

## Evidence contract and diagnostic roles

Numerical admission requires finite coefficients/objective/gradient, converged
projected optimization, valid cast to FP32, positive covariance, correct
independent final streams, one trace per signature, and matching refined-grid
reference/data provenance. A fit failure rejects the candidate row and blocks
its selected claims; independent baseline and heuristic work continues.

Boundary activity and an observed loss to any constructed heuristic on a claim
dataset veto promotion. They trigger repair, not abandonment of iAPF. Shape
residual, density amplitude, realized particle count and timing explain the
result and are not score-promotion criteria. No default promotion is possible
from this small pilot even if every screen passes. No statistical ranking,
canonical LEDH completion, HMC readiness, or broad nonlinear validity is claimed.

Continuation vetoes are a failed numerical unit/call-chain check, invalid
reference, source or data mismatch, missing diagnostics, infrastructure failure
that cannot be repaired in budget, or budget exhaustion. A failed objective
candidate alone is not a continuation veto.

## Defaults, fairness and smallest checks

| Choice | Provenance and role | Failure mode and early diagnostic |
|---|---|---|
| Relative-shape objective | Derived optional Algorithm-3 adaptation | Shape optimum can still be poor downstream; finite differences, exact-Gaussian recovery, then untouched score comparisons |
| Original density objective | Frozen comparator, not a recommended default | Amplitude collapse; preserve its original underflow rejection and report residual/bounds |
| mean bound 4, sd [0.2,4], floor 0.01 | Same controlled baseline values, not tuned defaults | Bound activity or floor domination; report coefficients, boundaries, shape and score errors; no tuning on heuristic data |
| k=1, tau=100, at most 4 iterations, N=16 to 128 | Same diagnostic pilot controller | Early stopping and varying work; record complete adaptive counts and actual fitting work; no fixed-cost superiority claim |
| 2,000 fit steps, 30 backtracks, projected tolerance 1e-7 | Same finite solver budget; tolerance has objective-specific units | Inadequate convergence or numerical stagnation; analytical-gradient and end-to-end checks; fail closed, no tolerance relaxation |
| FP64 offline fit, FP32/TF32 filter, XLA | Existing tested precision split | Cast can damage covariance; preserve cast guards and independent finite-difference tests |
| T=2, d=o=1, weak (.12,.04), curved (.35,.12) | Existing mechanism fixtures, fresh data | Short/small regime cannot support generality; retain explicit pilot scope |

Heuristic adversaries are constructed for each regime: EKF uses cheap local
Gaussian propagation, UKF captures curvature with sigma points, bootstrap PF
uses the unmodified stochastic transition, and local-linear proposal uses the
current observation without iterative twisting. Compute all four separately
on each claim dataset. Retain refined-grid score agreement as the primary
error measure. Heuristic checks are falsification instruments, never tuning
targets. The one repaired control arm is a mechanism test, not a completed
hyperparameter search. Selection artifacts bind its complete scope.

Skeptical audit: the earlier global density-argmin interpretation is rejected;
the new objective is labeled as different and optional. Comparator settings,
data and random streams match where appropriate. Reference validity precedes
score interpretation. Source-fit convergence and residuals cannot promote a
model-score claim. Fresh partitions prevent reuse of previous claim data.
The plan is adequate for this bounded repair screen, not comprehensive tuning.

## Execution and budget

Use `scripts/run_younis_iapf_fresh_calibration.py --protocol relative_shape`.
New prepared-data identity `nonlinear_iapf_shape_repair_20260917_<regime>`;
weak calibration 1200/1202, validation 1210/1212, claim 1220; curved identities
are the corresponding odd numbers. Global seed stays 1000. Each claim uses
two independent final replications with the same offline fit per arm. The
repair and baseline each have four source rows and two claim rows per regime;
four heuristic rows per regime yield 32 total rows, at most 104 charges.
This fits the 109 remaining charges. It leaves five charges under worst-case
consumption; no additional sweep or repeated rejected claim is preauthorized.

Run CPU reference/unit tests with `CUDA_VISIBLE_DEVICES=-1` before framework
import, explicitly labeled diagnostic. Tests cover analytical gradients,
amplitude/target-scale invariance, exact Gaussian recovery, the prior underflow
witness, real adapter wiring and rejection of a mislabeled objective. Re-run
the affected 19-test suite and driver checks. Record timed command logs.

GPU execution uses the existing tftwogpu Python environment, trusted permissions,
RTX 5080 selected by UUID (CUDA ordinal previously differed from nvidia-smi),
memory growth configured and verified before device initialization. Query and
record UUID/model before launch; record framework physical-device details.
Commit the isolated source before the run; preserve source fingerprint and
the actual command in the manifest. No environment/package mutation.

Versioned output root:
`docs/plans/artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-relative-shape-repair-20260917-01/`.
Use `launch02` for the campaign run, unique directories for any localized
infrastructure retry. Charge prior consumption and all new work. Candidate
failures remain saved. Total launch limit stays three; total wall/CPU/charge
limits remain unchanged. Final result must include decision and inference
status tables, the conditional heuristic table, remaining budget, and the
smallest justified next action. Update the master and active checkpoint.
