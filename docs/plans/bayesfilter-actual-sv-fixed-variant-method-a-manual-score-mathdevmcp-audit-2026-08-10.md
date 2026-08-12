# Actual-SV fixed-variant Method A manual score MathDevMCP audit

Date: 2026-08-10
Status: `DERIVATION_AUDIT_PARTIAL_PROGRESS_BLOCKED_ON_IMPLEMENTATION`

## Question
Can the fixed-variant actual-SV frozen-core batch TT scalar be given a same-program manual score backend, and have we now frozen the right scalar and derivative target before implementation?

## Inputs audited
- Derivation lock note: `docs/plans/bayesfilter-actual-sv-fixed-variant-method-a-manual-score-derivation-lock-2026-08-10.md`
- Implementation target: `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`
- Reusable derivative helpers:
  - `bayesfilter/highdim/derivatives.py`
  - `bayesfilter/highdim/zhao_cui_moment_teacher_als.py`
  - `bayesfilter/highdim/zhao_cui_moment_teacher_xla.py`
  - `bayesfilter/highdim/filtering.py`

## Commands run
```bash
PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli assumptions-for \
  "actual-SV fixed-variant frozen-core TT scalar derivative" \
  --provided-assumption "fixed TT cores are frozen after comparator fit at theta_0" \
  --provided-assumption "score must match same frozen-core scalar"

PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli debug-derivation \
  --step "A dot c = dot b - dot A c" \
  --assumption "same frozen-core scalar" \
  --assumption "fixed TT cores frozen at theta_0"

PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli audit-math-to-code \
  "A dot c = dot b - dot A c" \
  "/home/chakwong/BayesFilter/bayesfilter/highdim/derivatives.py"
```

## Findings

### 1. Assumption discovery is still conservative / inconclusive
`assumptions-for` did not certify a route-complete assumption set. It returned `human_review_required` / `unknown`, which is consistent with prior SVX audits: the target still needs a typed obligation and human review rather than blind promotion.

Interpretation:
- this is **not** a proof failure,
- it means MathDevMCP did not certify a complete assumption set for the whole target automatically,
- so the derivation lock note must remain the authority on what is frozen and what is allowed to vary.

### 2. The old single-equation debug query is too underspecified
`debug-derivation` on only the LSQ equation reported a gap because it needs a step chain, not a single isolated equation.

Interpretation:
- the tool did not refute the LSQ derivative,
- the query shape was not rich enough to certify the full chain.

### 3. Math-to-code audit confirms the current implementation still does not implement the intended manual chain
`audit-math-to-code` returned `structural_mismatch` for the manual derivative equation against `bayesfilter/highdim/derivatives.py` as a whole-file target.

Interpretation:
- this does **not** mean the helper equation is wrong,
- it means the actual active code surface still does not expose the required same-program manual derivative backend,
- which matches the current state of the repo: `batched_fixed_tt_likelihood_value_score_status(...)` still uses autodiff.

## Additional audit findings from code tracing

The deeper code audit uncovered one issue that was easy to miss earlier:

### The active runtime scalar is not a fully frozen fitted-TT scalar
What is frozen after the one-time comparator fit is:
- the seed one-axis core `initial_core`,
- the seed adjacent-state cores `adjacent_core0`, `adjacent_core1`,
- the basis / quadrature / coordinate-map tensors.

But inside `batched_fixed_tt_likelihood_value_trace(...)`, each time step still runs:
- a one-axis or two-axis deterministic ALS replay,
- starting from those frozen seed cores (or the previous step’s fitted cores).

So the exact scalar for Method A is:
- a **frozen-seed, same-program batch TT scalar**,
- not a scalar with fully theta-independent fitted TT coefficients.

This was the key semantic ambiguity that kept causing confusion. The correct manual backend therefore must differentiate:
- the same deterministic ALS replay owned by `batched_fixed_tt_likelihood_value_trace(...)`,
- not a different helper route,
- not a “fully frozen fitted coefficient” shortcut.

### The repo already contains the right algebraic building blocks
The following components appear mathematically reusable:
- fixed-branch square-root target derivative:
  - `square_root_target_jvp(...)`
- design-aware ALS tangent replay:
  - `fixed_als_value_jvp(...)`
  - `padded_fixed_als_value_jvp_xla(...)`
- squared-TT normalizer derivative:
  - `squared_tt_log_normalizer_derivative(...)`
- retained marginal quotient-rule derivative:
  - `squared_tt_normalized_marginal_jvp(...)`
  - `_normalized_retained_log_density_derivatives_chunked(...)`

This means the remaining blocker is **implementation integration**, not lack of the needed derivative algebra.

## Correct implementation plan implied by the audit
1. Keep `batched_fixed_tt_likelihood_value_trace(...)` as the sole value authority.
2. Replace `batched_fixed_tt_likelihood_value_score_status(...)` with a same-program forward derivative replay.
3. For each parameter direction:
   - build explicit local target derivatives,
   - propagate them through the one-axis fit at `t=0`,
   - propagate them through the two-axis ALS replay for `t>=1`,
   - propagate retained marginal tangents to the next time step,
   - accumulate `dot(log_shift) + dot(log_normalizer)` per step.
4. Validate first against the current autodiff backend on the same scalar.
5. Then validate against centered finite differences.

## What was wrong in earlier procedures
- We mixed frozen-seed semantics with fully frozen fitted-coefficient semantics.
- We let nearby transformed-SV helper routes stand in for the actual runtime scalar.
- We used end-to-end FD to discover route mismatch instead of freezing the runtime scalar contract first.
- We did not explicitly record that the active route still replays the deterministic ALS fit at each time step.

## Current blocker
A real blocker remains:
- the manual backend is still **not implemented** in `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`.
- The math audit now supports the implementation direction, but the code work itself is still outstanding.

## Non-claims
- No proof certificate was obtained from MathDevMCP.
- No claim is made that the transformed-SV independent-panel route is equivalent.
- No claim is made that actual SV is already Method-A admitted.
- No claim is made that the implementation is finished.
