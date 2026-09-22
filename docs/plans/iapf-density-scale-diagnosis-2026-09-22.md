# iAPF density scale, initialization and stopping

Status: COMPLETE on CPU and GPU FP64/XLA. Eight inaccurate zero-step fits per
device (all shifted targets at d40/d80); fixed scaling alone does not repair
recovery. The independent initialized R reference recovers all 30 targets.
Result: `artifacts/iapf-density-scale-20260922-01/result.md`.
Parent: `younis-kdm-score-master-program-2026-09-14.md`.
Previous completed phase: `iapf-adaptive-consumer-parity-2026-09-22.md`.

## Question and contract

Can the local bounded Equation-15 fitter satisfy its absolute-gradient tolerance
without learning a known, exactly representable diagonal Gaussian as dimension
grows? Does a fixed rescaling of the same objective distinguish loss of numerical
scale from the density objective's separate vanishing-density escape?

Baseline: the actual `iapf_fit_tf.bounded_density_fit`, density_l2, initialization
at the cloud mean and marginal variance, unchanged box [.2,4] in standardized
SD and +/-4 in standardized mean, 2000 iterations, 30 backtracks, tolerance
1e-7, floor .01. Comparator B executes the same fitter but multiplies its loss
and analytical gradient by the fixed positive constant exp(-2 a0), where a0
is the maximum initial log density on the cloud. This is a diagnostic experiment
with objective units, not a runtime/default change or a relative-shape objective.
The multiplier is fixed throughout a fit. Comparator C is the frozen independent
R paper-reference `iapf_fit_gaussian(..., fit_mode='paper_eq15')`, which combines
QR initialization, fixed density scaling and its own local optimizer. C can
test representability but does not isolate those choices from each other and
is not identified with original author code.

Use d=5,10,20,40,80; N=1000; two independent clouds with seeds 71/72. For each
cloud, construct three targets: healthy (exactly the baseline initial Gaussian),
shifted/narrow (mean .4*sin(j), variance linearly .6 to .9), and shifted/broad
(mean .4*cos(j), variance linearly 1.4 to 1.8). All are diagonal Gaussian
functions on the same points; their exact mean, covariance and profile loss
zero are known. No observation simulation or filtering randomness is needed.

Primary diagnosis: at any d/seed, a valid/converged baseline with zero optimizer
steps and known-target Gaussian KL > .01 or relative shape residual > .01
demonstrates an uninformative convergence signal. These are diagnostic effect
thresholds, not new scientific/default admission thresholds. The conclusion
must also report exact errors and a target-independent analytical gradient
bound, so it does not depend only on the threshold. Good gradient arithmetic
with a tiny objective is distinguished from true underflow and from a wrong
gradient. No failed-candidate result rejects the iAPF research direction.

Explanatory diagnostics: objective, initial/terminal density amplitude,
projected gradient, normalized shape residual, iterations, active bounds,
KL to exact Gaussian, parameter errors and independent R QR/fit recovery.
Continuation veto: invalid source/draw identity, nonfinite accepted results,
wrong objective or gradient, or exhausted budget. A candidate's poor recovery
is evidence, not a continuation veto. A scaling candidate that changes a
healthy exact target materially is rejected; the healthy comparison tolerance
is 1e-10 for coefficients. No primary-metric selection of a new default occurs.

Heuristic adversaries are explicitly constructed: the initial cloud-moment
Gaussian (what zero updates return), direct diagonal log-quadratic least
squares (can recover the target exactly here), and the known exact target.
Evaluate separately by dimension, target regime and seed. Losing to exact QR
is a promotion veto for a claim to learn these targets, not a fair likelihood-
runtime ranking. This fixture does not claim general non-Gaussian robustness.

## Mathematics and source audit

Let b be target values, p the fitted Gaussian values, lambda=(p'b)/(b'b), and
r=p-lambda*b. Since r is an orthogonal projection residual, ||r||<=||p||.
If s_ij=partial log p_i / partial eta_j, then

    |partial L / partial eta_j|
      = |2 r' (p * s_j) / N|
      <= 2 max_i |s_ij| mean_i(p_i^2).

At initial standardized parameters eta=0, p_i=exp(-||z_i||^2/2), and the
score columns are z_ij and z_ij^2-1. The bound can therefore be extremely
small independently of b. If it falls below the absolute tolerance, the
implemented stopping rule can accept its initialization for every target.
This is mathematical consistency with that stopping rule, not evidence of a
useful guide. It does not require zero-valued floating-point underflow.

GJL Equation 15 defines the density loss, but the original numerical solver,
initialization and stopping units are unrecovered. The fixed-scale R reference
and TF consumer differ here. Paper technical sections and variance appendix
are grounded in the preceding campaign source audit and local paper copy.
The existing 2026-09-16 underflow note already distinguishes tiny gradients
from underflow; this experiment measures the high-dimensional consequence.

## Defaults and skeptical review

| Choice | Origin/status | Misleading failure risk | Early check |
|---|---|---|---|
| Actual TF controls | Existing tested adapter; baseline only | Small absolute loss masquerades as learning | Zero-step flag, bound and exact-target KL |
| Fixed objective scale | Algebraically equivalent objective; diagnostic hypothesis | Density escape persists after initial scaling | Relative residual and terminal amplitude, no silent promotion |
| R initialized local fit | Frozen independent reference | Multiple choices change together | Interpret as recovery comparator, not isolated causal effect |
| Gaussian fixtures | Constructed oracle, not the paper study | Too easy/generalization error | Condition results on regime and dimension, no filtering claim |
| Seeds 71/72 and N1000 | Bounded engineering coverage | No stochastic method ranking | Per-case reporting and deterministic inequality |
| CPU FP64/XLA then GPU FP64/XLA | Explicit reference precision | CPU success hides device errors | Same saved inputs on selected trusted GPU |

Skeptical review PASS: unlike a repeat likelihood sweep, these runs identify
whether the stopping signal depends on the target. The exact target and
gradient inequality make the question identifiable. The rescaling arm is
kept separate from normalized shape optimization and does not modify runtime
defaults. QR's information advantage and Gaussian fixture limitations are
explicit. A successful R reconstruction remains non-canonical. No proxy is
treated as filter accuracy or model-score evidence.

## Execution, budget and preservation

Execute CPU FP64/XLA first; preserve actual clouds, target values and all
fit outputs. Run the frozen R reference on those same points. If arithmetic
checks pass, repeat both TF arms on the selected GPU using identical saved
inputs, FP64/XLA and verified growth. Repair at most two localized harness
errors, preserving attempts. At most six launches, 1 CPU hour and 15 GPU
minutes. Start balance: CPU 165192.3481590662 seconds and GPU
172409.30359378795 seconds (see prior budget.json for authoritative precision).
All launched work and verification are charged, including failures.

Evidence root: `docs/plans/artifacts/iapf-density-scale-20260922-01/`.
Use the existing tftwogpu Python, single-thread CPU with CUDA_VISIBLE_DEVICES=-1,
R --vanilla; GPU UUID GPU-68251639-fe82-8f81-3ccc-2953c32e805b with escalation
and TF_FORCE_GPU_ALLOW_GROWTH=true. Save exact commands, source snapshots,
environment, seeds, times, result and machine-readable decisions.
The frozen R files and production fitter are not modified in this diagnostic.
