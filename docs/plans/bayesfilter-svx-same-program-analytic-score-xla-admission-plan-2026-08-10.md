# SVX same-program analytic score XLA admission plan

Date: 2026-08-10  
Status: `ACTIVE_XLA_ADMISSION_PLAN`

## Question

Can the newly implemented same-program SVX-ZC analytic score backend be made
`jit_compile=True` / XLA-correct without changing the target semantics?

## Mechanism being tested

We will separate two claims that are currently bundled together:

1. the eager same-program analytic score is correct for the active finite
   program;
2. the same score can be expressed in a graph-native XLA-compatible form.

The present backend already supports claim (1) locally. This plan addresses
claim (2).

## Current baseline

Current baseline facts:

- The active same-program analytic backend now lives in
  `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`.
- The active adapter wires that backend in
  `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`.
- Focused eager-mode SVX tests pass, and the XLA smoke/parity test now passes on
  the reviewed CPU fixture as well.
- The adapter metadata has been admitted to `xla_hmc_ready=True` and
  `full_chain_xla_diagnostic_ready=True` after parity checks.

## Why XLA is not already admitted

The current analytic backend is Python-loop driven and uses host-side structure
that has not been demonstrated to survive `tf.function(jit_compile=True)` with
correct outputs.

The main technical risks are:

1. Python/dataclass trace structures in the score replay path.
2. Dynamic-shape or dynamic-index operations in the active score replay.
3. The scaled-ridge derivative replay not yet packaged into a graph-native,
   XLA-safe kernel.
4. Potential mismatch between eager analytic outputs and XLA outputs even if the
   graph compiles.

## Promotion criterion

Promote `xla_hmc_ready=True` only if all of the following are true:

1. an XLA-targeted score path exists for the same active finite program;
2. that path compiles under `tf.function(jit_compile=True)` on the reviewed
   CPU fixture;
3. XLA value and score agree with the eager analytic backend within declared
   tolerances on the probe batch;
4. finite-difference checks for the same active finite program still pass on the
   reviewed probe point;
5. focused SVX tests pass;
6. adapter metadata truthfully reflects the admitted XLA status.

## Promotion vetoes

Any one of these blocks XLA admission:

- XLA compile failure;
- eager/XLA value mismatch for the same finite program;
- eager/XLA score mismatch for the same finite program;
- failed finite-difference agreement after the XLA refactor;
- any target-semantic drift;
- any attempt to admit XLA readiness by metadata only.

## Continuation vetoes

Stop only if one of these happens:

- we discover the current same-program backend cannot be expressed in TensorFlow
  graph form without changing the target;
- the XLA-capable route would require a different score target than the active
  finite program;
- a required dependency or permission boundary makes the necessary implementation
  impossible in this session.

A failed first compile or failed first parity attempt is **not** a continuation
veto. It is a repair trigger.

## Repair trigger order

If XLA admission fails, repair the first failing layer in this order:

1. graph packaging / Python object leakage,
2. static-shape schedule encoding,
3. scaled-solve derivative kernel,
4. retained-marginal XLA form,
5. eager/XLA parity checks,
6. adapter metadata and binding tests.

## Scope

Primary files allowed:

- `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`
- `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`
- `tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py`
- `tests/test_zhao_cui_actual_sv_neutra_target.py`

Reference helper files allowed for reuse/audit:

- `bayesfilter/highdim/zhao_cui_moment_teacher_xla.py`
- `bayesfilter/highdim/zhao_cui_moment_teacher_als.py`
- `bayesfilter/highdim/derivatives.py`
- `bayesfilter/highdim/filtering.py`
- `bayesfilter/highdim/fitting.py`

## Phases

### Phase X0 — freeze the XLA question

Goal:

- keep the current same-program eager backend fixed;
- treat XLA admission as a separate correctness claim.

Required checks:

- confirm focused eager tests pass before XLA work;
- confirm adapter metadata still says XLA is not admitted.

Exit gate:

- eager backend is the frozen baseline for all XLA parity checks.

### Phase X1 — isolate the XLA-critical kernel surface

Goal:

- identify the smallest kernel set that must be graph-native.

Required work:

- isolate the scaled-solve derivative replay,
- isolate the retained-density derivative replay,
- encode the fixed sweep schedule as static TensorFlow tensors,
- remove reliance on Python-side trace objects inside the XLA path.

Expected artifact:

- an explicit list of tensors that define one XLA score step.

Exit gate:

- the candidate XLA path is expressible as tensor-only helper calls.

### Phase X2 — build the XLA score kernel

Goal:

- implement a graph-native same-program score path.

Required work:

- reuse or adapt the padded fixed-ALS derivative kernel style from
  `zhao_cui_moment_teacher_xla.py`;
- encode the active `(0, 1, 1, 0)` sweep schedule as a tensor;
- keep the active scaled-solve derivative semantics;
- produce value/score tensors without Python-side trace traversal.

Not allowed:

- replacing the same-program backend with the transformed-SV route;
- dropping the scaled-solve derivative terms;
- admitting a compile-only route without parity evidence.

Exit gate:

- a candidate XLA score function runs eagerly and under XLA on the reviewed
  probe batch.

### Phase X3 — eager/XLA parity checks

Goal:

- prove that XLA computes the same quantity as the eager analytic backend.

Required checks:

1. eager value vs XLA value;
2. eager score vs XLA score;
3. same active probe batch status keys remain finite;
4. same-program finite-difference agreement still holds.

Exit gate:

- parity passes within declared tolerances.

### Phase X4 — adapter admission

Goal:

- update metadata only after XLA proof exists.

Required work:

- set `xla_hmc_ready=True` only if X0-X3 pass;
- set `full_chain_xla_diagnostic_ready=True` only if the full compiled route is
  actually tested;
- update focused tests to require the truthful metadata.

Exit gate:

- adapter metadata and focused tests agree with reality.

## Skeptical audit of this plan

- **Wrong baseline risk:** controlled. The eager same-program backend is frozen
  as the comparison baseline.
- **Proxy promotion risk:** controlled. Successful compilation alone does not
  admit XLA readiness.
- **Hidden target change risk:** controlled. All parity checks compare against
  the same active finite program.
- **Missing stop condition risk:** controlled. Each phase has an explicit exit
  gate and veto set.
- **Implementation drift risk:** controlled. Only the XLA surface and metadata
  are in scope; tuning and route redesign remain out of scope.

Audit verdict: `PASS_FOR_TARGETED_SVX_XLA_ADMISSION_WORK`

## Immediate next action

1. keep the current eager same-program backend as the baseline;
2. isolate the smallest graph-native kernel surface for the active score;
3. implement the first XLA-safe score kernel step;
4. do not flip XLA metadata until parity checks pass.
