# GenUT Feasible Trust-Region Repair Plan

Date: 2026-08-15

Status: executed with bounded follow-up evidence; candidate remains opt-in

## Research intent ledger

### Main question

Can the finite higher-moment Contract E correction be made finite and
differentiable on collapsed-weight GenUT states, while retaining useful
third/fourth-moment correction and computing the total derivative of the exact
finite value program that is executed?

### Demonstrated failure to repair

The preserved LGSSM NeuTra failure has `N=1008`, ESS about `1.0037`, maximum
weight about `0.9982`, and target marginal kurtoses as large as `3006.8`.
An equal-weight standardized cloud of size `N` satisfies the directly proved
necessary bound

```
kurtosis <= N - 1,
```

which is `1007` here.  The observed target near `3006.8` is still decisively
infeasible.  The current iteration nevertheless sends the
infeasible residual through unscaled normal equations.  At iteration 3 its
local Jacobian has scale about `1.25e5`, the normal matrix has condition
estimate about `1.9e8`, the absolute `1e-5` ridge is ineffective, and
`tf.linalg.solve` emits NaN.

### Candidate mechanism under test

The selected candidate is a finite composition of established components:

1. retain the empirical weighted skewness and kurtosis as the declared
   least-squares target, while recording necessary Pearson and finite-`N`
   equal-weight realizability diagnostics;
2. replace the unscaled absolute-ridge normal equations by a column-scaled
   Levenberg--Marquardt local least-squares solve; and
3. apply a smooth trust-region cap to every diagonal cloud displacement before
   restandardization.

The target is deliberately not clipped or tempered.  When it is unattainable,
the finite algorithm returns a bounded residual instead of claiming exact
moment matching.  This follows the constrained-GenUT principle that enforcing
point constraints can require loss of exact kurtosis matching, without changing
the weighted source moments that define this BayesFilter candidate.

### Expected failure mode

The candidate could remain finite but become ineffective because feasibility
continuation suppresses most higher-moment information.  It could also compute
the wrong score if the value and JVP routes use different continuation,
damping, cap, or restandardization operations.

### Promotion criterion

This repair is promoted only as a new opt-in candidate if all of the following
hold:

- the preserved failure replay is finite in scalar, batch value, and batch JVP
  routes under GPU/XLA;
- analytic JVPs agree with finite differences at ordinary and near-collapse
  points under declared tolerances;
- no correction-step update exceeds its declared smooth trust radius;
- requested feasible targets remain unchanged up to numerical tolerance;
- infeasible targets are identified, the executed displacement remains bounded,
  and no exact-matching claim is made;
- existing healthy LGSSM/GenUT regression cases do not materially lose value
  or score accuracy under the predeclared comparison; and
- a broader fresh NeuTra-support screen has no non-finite target values or
  scores.

### Promotion vetoes

- any non-finite value, score, tangent, factorization, or correction state;
- scalar/batch or value/JVP program mismatch;
- finite-difference derivative failure not explained by a declared
  measure-zero switching boundary;
- silent target change or caller-overridable route identity;
- literature/source claim without an inspected primary technical anchor; or
- a checked established method that dominates the provisional local
  composition for this exact setting and has not been fairly evaluated.

### Continuation vetoes

- the preserved replay cannot be reconstructed from immutable artifacts;
- implementation requires changing the GenUT likelihood, observation model,
  data, particle count, or NeuTra objective outside the higher-moment reset;
- the complete executed map cannot be differentiated consistently; or
- the serious GPU/XLA environment cannot be made source-bound and finite
  within the attempt budget.

Candidate rejection alone is not a continuation veto.  It triggers the next
smallest repair allowed by the remaining budget.

### Explanatory diagnostics

ESS, maximum weight, raw and continued moments, continuation amount, Jacobian
scales, damping, singular/eigenvalue estimates, pre/post-cap displacement,
moment residuals, runtime, and TF32 sensitivity explain behavior.  They are not
by themselves promotion criteria unless stated above.

### What must not be concluded

Passing this plan does not establish exact moment matching, exact filtering,
posterior correctness, NeuTra convergence, HMC readiness, superiority over all
filters, or readiness to replace the current default.  A repaired candidate
needs fresh model-specific admission and later sampler evidence.

## Evidence contract

- **Question:** does a source-grounded feasible trust-region correction repair
  the demonstrated non-finite GenUT value/score path without discarding useful
  higher-moment information?
- **Baseline:** the current empirical-target, unscaled normal-equation diagonal
  correction with the same Contract E/OT controls and preserved inputs.
- **Comparator ladder:** no diagonal correction; current diagonal correction;
  stable-solve-only diagnostic; final feasibility plus LM plus trust-region
  candidate.  Existing pairwise/dual-cap stages remain fixed downstream.
- **Primary criterion:** finite, value/JVP-consistent execution plus derivative
  parity on the preserved failure and fresh support screen.
- **Hard vetoes:** the promotion vetoes above.
- **Explanatory only:** moment residual magnitude, ESS, maximum weight, runtime,
  and descriptive value/score differences unless uncertainty is supplied.
- **Artifact:** versioned JSON/log/result files under
  `docs/plans/artifacts/genut-feasible-trust-region-repair-20260815/` and a
  terminal Markdown result note.

## Literature and source audit

The literature work must maintain six ledgers: source support, citation/venue
metadata, backward snowballing, forward snowballing, claim support, and omitted
paper risk.  The seed sources are Ebeigbe et al. on GenUT, primary
Levenberg--Marquardt/trust-region sources, particle/ensemble transform sources,
and direct work on moment-constrained equal-weight quadrature or resampling.

Each algorithmic statement must be classified as one of:

- explicitly stated by a checked primary source;
- derived in BayesFilter notation from checked assumptions; or
- a new local finite composition requiring empirical validation.

Implementation is blocked until the audit answers whether an established
method directly covers the requested equal-weight differentiable reset.

## Default and assumption audit

| Choice | Provenance/status | Why used | Misleading failure mode | Early diagnostic |
|---|---|---|---|---|
| Preserve original weighted mean/covariance | Current Contract E target; reviewed default | Required target identity | Shape repair alters lower moments | post-restoration residual test |
| Necessary marginal moment feasibility | Pearson moment inequality plus finite equal-weight bound; project derivation | Identify impossible exact requests | Necessary bounds mistaken for sufficient joint realizability | explicit nonclaim and adversarial moments |
| Unchanged empirical target | Current finite GenUT program; selected design | Avoid a hidden teacher/objective change | huge residual dominates an unbounded update | trust-cap and residual diagnostics |
| Relative LM damping | Established nonlinear least-squares mechanism; exact scaling still hypothesis | Avoid ineffective absolute ridge | damping masks rank loss or changes steps excessively | singular-scale and stable-solve diagnostics |
| Smooth trust radius | Established trust-region principle; exact radius is a tunable hypothesis | Bound each cloud move | inherited radius is under/over restrictive | small radius ladder on calibration cases |
| Existing correction count/strength | Warm start only | Fair baseline comparison | transferred settings hide repair quality | small bounded retuning on calibration only |
| TF32 setting | Fixed by compared scope | Preserve production arithmetic | precision blamed for structural bug | identical-input TF32 on/off diagnostic already shows same failure |

No threshold or damping constant is a default until target-specific calibration
and untouched validation support it.

## Pre-mortem

- **Passes while misleading:** the preserved row becomes finite only because
  the correction is effectively disabled.  Detect with continuation amount and
  attainable moment-residual improvement.
- **Fails for implementation reasons:** scalar and batch paths implement
  different algebra.  Detect with shared primitive tests before GPU replay.
- **Fails for tuning reasons:** an inherited trust radius is too small.  Detect
  with a bounded calibration ladder; do not change untouched validation data.
- **Wrong scientific inference:** a finite target is interpreted as a correct
  posterior or converged sampler.  The result note must keep these as explicit
  nonclaims.

## Phased execution

1. Complete the source survey and literature ledgers.  Prefer an established
   direct method if one exists; otherwise label the exact local composition.
2. Derive the feasibility, LM, trust-cap, and full JVP equations in the GenUT
   section of `docs/chapters/ch32c_entropic_ot_sinkhorn.tex`, which is included
   by `docs/main.tex`.
3. Audit the derivation with MathDevMCP and a bounded Claude Code read-only
   review if available.  Repair material findings.
4. Implement shared TensorFlow primitives first, preserving the historical
   route as comparator and issuing a new non-overridable candidate identity.
5. Add focused CPU-hidden reference/mechanics tests: bounds, feasible identity,
   finite ill-conditioned solve, trust cap, scalar/batch parity, and finite
   differences.
6. Run the preserved failure replay and fresh bounded support screen with
   escalated GPU access, TensorFlow memory growth, GPU/XLA provenance, and fresh
   versioned outputs.
7. If the validity gates pass, run only the bounded healthy-case comparison.
   A NeuTra training run is a later phase and is not automatically authorized
   by numerical repair.
8. Write the result note with separate engineering, numerical/derivative, and
   scientific ledgers plus decision and inference-status tables.

## Compute and attempt budget

- Literature/derivation audits: local work plus at most two bounded external
  review attempts.
- CPU-hidden focused tests: at most six repair cycles.
- GPU/XLA preserved replay: at most four fresh attempts.
- Fresh support screen: at most two seeds and one bounded control ladder after
  the preserved replay passes.
- Stop before NeuTra optimization or HMC sampling in this plan.

## Execution Addendum (2026-08-15)

The public batch status now carries feasibility margins, scaled-system
condition, and pre/post-cap RMS diagnostics. The positive-damping candidate
identity is issued from the repository-owned `GENUT_SHAPE_SOLVER_ID`; legacy
zero-damping payloads retain their historical signature shape.

The old step-169 checkpoint under `/tmp/BayesFilter-dual-cap-beta` is not an
exact current-route replay authority. Its target signature and source route
differ from this checkout, so reusing its row with the current candidate would
be a route-mismatched comparison. MathDevMCP substantive calls and the bounded
Claude probe were retried, but the permission-review layer timed out; they are
unavailable reviews, not agreements or mathematical counterevidence.

The trusted RTX 5080 replay was attempted twice after the code change and was
blocked before process launch by the same permission-review timeout. The
earlier GPU artifact is historical evidence for the pre-addendum checkout and
is not treated as current-code GPU evidence.

## Skeptical pre-execution audit

Revised verdict: **PASS FOR DOCUMENTATION AND FOCUSED IMPLEMENTATION**.

The initial smooth target continuation was removed because it would silently
change the moment teacher.  Checked direct methods do not provide a drop-in
replacement: Ebeigbe et al.'s constrained GenUT preserves lower moments while
explicitly accepting loss of exact kurtosis, and Easley--Berry HOUT uses a
variable signed-weight rule whose condition number can grow as the requested
moment tolerance shrinks.  Neither is a positive equal-weight differentiable
reset of the current particle cloud.  The selected bounded least-squares map
therefore has the correct baseline, preserves the declared target, has explicit
stop conditions, and produces artifacts that answer the stated numerical and
derivative question.  Literature completeness and default promotion remain
outside this pass.
