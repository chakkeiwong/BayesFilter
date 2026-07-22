# Cubature/GenUT Gap-Closure Program

Date: 2026-07-21

Status: `PHASE3_DIAGNOSTIC_PASSED_PHASE4_MODEL_PILOTS_PENDING`

## Objective

Close the engineering and evidence gaps that currently prevent the
experimental Cubature/GenUT candidate from being considered for nonlinear
leaderboard cells. Contract E-Chol remains the canonical/default/HMC-facing
route throughout this program.

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | Can the candidate be made a TensorFlow/XLA-native, source-bound, dimension-scalable finite value/total-score route whose model-specific claims can later be compared fairly with Contract E? |
| Candidate | Non-fused positive Cubature and feasible-positive GenUT residual route. |
| Comparator | Same-target Contract E-Chol finite value/total score, using identical fixed streams and scope controls when a model adapter exists. |
| Primary promotion criterion | Every claimed cell has a repository-issued identity, no Python loop/NumPy in the XLA computation, target-matched value and recursive total score, scope-specific tuning, full-horizon evidence, and replicated uncertainty. |
| Promotion veto | Python loop in the traced core, NumPy/host conversion in the traced core, partial score, caller-forged identity, target substitution, negative/nonrepresentable OT masses, stale tuning, nonfinite branch, failed individual FD direction, or missing comparator. |
| Continuation veto | Missing target law/adapter ownership, mathematical contradiction, unavailable trusted GPU, or exhausted bounded campaign budget. |
| Diagnostics | Graph op/source audit, CPU/GPU parity, FP64/reference differences, compile/warm timings, allocator telemetry, FD spot checks, and per-time score decomposition. |
| Nonclaims | No exact nonlinear filtering theorem, unbiasedness, method superiority, HMC readiness, NAWM result, or default promotion from this repair program alone. |

## Evidence Contract

The XLA candidate function must be pure TensorFlow: `tf.while_loop`,
`tf.TensorArray`, TensorFlow linear algebra, and adapter TensorFlow callbacks.
Python is allowed only for configuration, artifact serialization, and test
orchestration outside the traced computation. NumPy is diagnostic-only and may
not be imported, called, or reached by any candidate function traced by XLA.
Every serious artifact records commit, exact command, environment, seeds,
dtype/TF32/XLA, memory policy, wall time, and output paths.

## Skeptical Plan Audit

1. **False loop-closure risk:** replacing the outer time loop while retaining
   Python Sinkhorn or parameter loops would leave graph specialization and
   violate the requirement. The implementation gate therefore scans the full
   traced call closure and tests dynamic horizons.
2. **Host-conversion risk:** `.numpy()` used after a compiled call is artifact
   serialization, not XLA computation, but any conversion inside adapter/core
   code is a veto. The source audit distinguishes runtime modules from test and
   reporting code.
3. **Dynamic-shape risk:** `tf.while_loop` must preserve static particle/state/
   parameter shapes needed by XLA. Shape invariants and TensorArray element
   shapes are explicit, with short dynamic-horizon replay tests.
4. **Derivative risk:** a batched parameter tangent must remain the total JVP
   of the same finite scalar. The existing central-FD tests remain mandatory
   after loop conversion.
5. **Baseline drift risk:** no Contract E or leaderboard route is modified;
   all new evidence remains candidate-only and target-bound.
6. **Numerical-risk proxy:** compile time and finite output are feasibility
   diagnostics, not accuracy or superiority evidence. FP64/reference and
   same-target comparisons remain separate gates.

Audit decision: `PASS_WITH_BOUNDED_REPAIR`. The loop-native core and source
audit are safe to implement now; model admission and default changes remain
out of scope until later phases pass.

## Phases

### Phase 1: Loop-Native TensorFlow Core

Replace every Python loop in the traced candidate computation with
`tf.while_loop` and TensorArray accumulation: Sinkhorn iterations, reset JVP
parameter directions, and time recursion. Preserve output schema and score
increments. Add source checks and dynamic-horizon CPU FD parity tests.

Pass: no Python `for`/`while` in the traced core, finite outputs, same-scalar
FD agreement, and unchanged reset diagnostics.

### Phase 2: Runtime Purity And Identity Boundary

Audit the full candidate call closure for NumPy, `.numpy()`, host branching,
and caller-stamped route identity. Move all reporting conversions outside the
compiled functions. Bind candidate identities to repository-owned callable
symbols and a source-closure digest; reject unregistered adapters.

Pass: runtime purity test, identity forgery rejection, and manifest-bearing
smoke artifacts. No production policy changes.

### Phase 3: Dimension And Precision Ladder

Add a genuinely multidimensional toy adapter (`d=2` and `d=4`) and run a
bounded `N` divisible by `2d` ladder under float32/TF32 XLA. Run a separate
float64 CPU/reference arm on identical fixed inputs. Record compile/warm time,
allocator telemetry, value/score differences, and residuals.

Pass: finite dimensions and declared precision budget; failure remains a
diagnostic scaling blocker, not evidence against the method.

### Phase 4: Target-Bound Model Pilots

Wire exact transformed-SV first, then predator-prey, then KSC-SV only after
target measure and score owners are checked. Generalized SV and Austria SIR
remain blocked by their existing feature/measure gaps. Each pilot must have a
same-target Contract E comparator before any claim.

### Phase 5: Scope Tuning And Variance

Tune each model/route/horizon/dtype scope on disjoint calibration/validation
data. Evaluate antithetic/common-random-number options as declared finite
estimators, with replicated seeds and uncertainty. Freeze controls before
untouched claims.

### Phase 6: Leaderboard Admission And Default Decision

Assemble only rows satisfying the existing leaderboard contract, including
full horizon, individual FD directions, score/value pairing, identities,
tuning artifacts, and comparator completeness. Conduct a terminal audit before
any owner policy request; otherwise retain experimental status.

## Compute And Stop Rules

Phase 1/2 are CPU-hidden focused tests. Phase 3 uses a bounded trusted GPU
ladder with a fresh artifact root and a hard stop on nonfinite output, device
fallback, memory-policy failure, or compile failure. Each failed attempt is
preserved and classified. No historical artifact is overwritten or upgraded.
