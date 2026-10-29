# R4d finite solver source-semantics repair

Continuation under the existing whole-repair campaign; no new compute grant.
The R4c failed paper-horizon parameter comparison remains evidence. A completed
diagnostic does not erase it. This tranche repairs demonstrated finite-program
differences before asking whether sensitivity persists.

## Question and source

Does the shared TensorFlow fitter implement the finite R 4.1.2 `vmmin`
backtracking, restart and symmetric inverse-Hessian semantics on its supported
domain, and do those repairs resolve the unchanged paper-horizon comparison?
Reference: `.localresources/code/r-4-1-vmmin-reference/optim.c`, lines 113–280;
in particular lines 180–245. R evaluates a finite objective at rejected trials,
backtracks from nonfinite objectives, and computes gradients only at accepted
progress points. Its stored lower triangle defines a symmetric inverse Hessian.
The current TF early abort on a nonfinite trial is wrong relative to that
finite procedure. Full-matrix roundoff asymmetry is a separate hypothesis.

## Evidence contract and research intent

- Comparator: installed R `optim(method="BFGS")`, explicit analytic gradients,
  maxit=100 except the declared four-step negative-curvature control; reltol
  sqrt(double epsilon), no additional damping/regularization. Record R version.
- Primary semantic checks: termination code, function/gradient counts and finite
  endpoints against deterministic R controls (1e-8 absolute+relative endpoint
  tolerance). Cases: quadratic, nonfinite rejected exponential trial,
  Rosenbrock, zero-gradient flat objective, floating-point no-change, and
  negative curvature/restarts. An invalid initial objective must be rejected.
- The nonfinite-trial case must show a rejection followed by a finite result;
  accepted invalid gradients, nonfinite accepted states, lost budgets/source
  provenance, missing outputs and diagnostic trace corruption veto the run.
- Paper comparison remains the existing full-precision fixture and unchanged
  1e-5 absolute+relative parameter criterion. Small controls must pass on the
  repaired actual shared fitter/backward/controller call chain before this run.
- Explanatory only: rejected-trial counts, resets, curvature, Hessian symmetry,
  first divergent accepted iterate, exact-initializer replay, loss/gradient and
  timing. These cannot establish empirical replication or original-author
  fitting identity. No heuristic-quality claim is made by solver parity.
- Constructed controls serve as exact, cheap adversaries to solver behavior:
  analytic quadratic optimum; exact stationary constant; exact no-change
  floating-point input; analytic exponential optimum. Conditional situations
  are regular, overflow trial, stationary, no-change and negative curvature.
- Repair trigger: a semantic mismatch or regression. A sensitive healthy
  endpoint rejects strict numerical parity, not the filter mathematics. Preserve
  it, and specify separate downstream statistical validation if necessary.
- Continuation veto: wrong source/objective, corrupt data, invalid accepted
  computation, exhausted tranche/parent budgets. No convergence or statistical
  ranking is inferred from a finite solver return code.

## Defaults, implementation, and pre-mortem

The unchanged finite recipe is a reconstruction, not a recovered author recipe.
FP64 and fixed fixtures isolate arithmetic; they do not validate FP32 transfer.
Extract the existing optimizer loop into one shared TF function so deterministic
controls and `_finite_fit` exercise the same loop. Keep explicit stable graph
signatures and native while loops; no NumPy, pfor, scalar training path, added
ridge or optimizer retuning. Reconstruct the dense Hessian from the updated lower
triangle, matching the reference storage meaning without averaging or damping.
Record numerical diagnostics rather than discarding them. Initializer override
is an explicitly named diagnostic factory and cannot silently affect the filter.

The numerical target is unchanged. Mirroring the stored triangle changes finite
roundoff ordering; small controls and full old gates test its effect. It cannot
be promoted merely because a primary loss decreases. More general vmmin inputs
outside the frozen maxit>=2, full-mask, fixed-control scope are not claimed.

A misleading pass could come from a disconnected test solver, loose post-hoc
tolerances or rounded reference inputs. All tests call the shared routine, save
exact 17-digit arrays, and preserve prior criteria. A failure could be harmless
trajectory sensitivity or an implementation fault; fixed-initializer replay and
accepted-step comparisons distinguish those explanations without asserting a
successful repair in advance.

## Execution, budget, and review

At opening: 782.345146 CPU seconds and 131169.305299 GPU seconds remain.
R4d ceilings: 240 CPU seconds and 1800 GPU seconds including retries. Increase
the attempt-count ceiling from 28 to 36 within these unchanged time limits.
Implement bounded source fixture, semantic CPU/GPU graph/GPU XLA controls, small
shared-consumer regression, then unchanged full-precision paper comparison.
Each phase runs through `scripts/run_replication_master.py run-next` with its
recorded interpreter, unique attempt directory, timeout and source snapshot.
GPU commands require escalated device access and verified memory growth. CPU
reference work hides GPU devices. No package/environment mutation is planned.

Skeptical pre-execution audit: PASS for this diagnostic tranche. Baseline is the
actual reference solver; metrics measure the stated finite object, not a proxy
for replication. Earlier failed endpoints stay failed; later phases are not
skipped merely because the candidate remains sensitive. Scope, assumptions,
source lines, stop conditions, resource ceilings and required outputs are above.
This is a source-based self-review, not an independent-agent review.

Results will be saved as `r4d-result.md` plus per-attempt full JSON/CSV traces,
decisions, manifests and logs. Both full replications remain INCOMPLETE.

Actual supervisor command for one phase: the approved environment interpreter
imports scripts.run_replication_master, calls validate(), selects snapshot()["next_phase"]
and calls execute(phase). This is the same run-next dispatcher operation with a
compact console result; every child command and source snapshot remains in its
manifest. The first R fixture used the run-next CLI directly.


Localized retry amendment after attempt33:
The newly exposed optimizer diagnostics added keys to the backward loop state;
the empty-state initializer had not been updated, so tf.while_loop rejected the
structure before execution. Update that initializer and rerun the actual consumer.
The source controls also exposed a reporting error: a no-change line-search exit
could count a stale accepted value as a function call, emitting -1 rejected trials.
Exclude no-change from that counter; this does not alter solver states. Add a
nonnegative counter regression in every control, rerun GPU/XLA semantic checks
before consumer retries. Attempt ceiling38, same240 CPU/1800 GPU seconds and
unchanged parent budget. Old artifacts remain preserved. This is a local
implementation/observability repair, not evidence against either paper.
