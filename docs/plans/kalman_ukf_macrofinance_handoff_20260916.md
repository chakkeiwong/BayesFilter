# MacroFinance integration handoff: compiled rectangular SRUKF

BayesFilter's reusable runtime repair is complete for the declared engineering
fixtures. Consume `TFRectangularSRUKFModel`,
`TFRectangularSRUKFDerivatives`, `TFRectangularSRUKFFixedBranch` and
`tf_rectangular_srukf_value_and_score` from
`bayesfilter.nonlinear.rectangular_srukf_tf`. The default direct-factor API still
requires square factors. The rectangular API is the explicit contract requested
for DZ5; full-rank square factors are one supported specialization.

[Results and evidence](kalman_ukf_compiled_runtime_result_20260916.md) distinguish
the tested generic runtime from the remaining integrated DZ5 work. No
MacroFinance files were edited, and CD scientific/retained-sampling status is
not changed by this handoff.

## Factor and branch contract

| Quantity | Tensor shape | Documented CD full-rank specialization |
| --- | --- | --- |
| Parameters / observations | `[B,P]` / `[B,T,M]` | `[4,18]` / `[4,96,26]` |
| Initial mean / factor | `[B,N]` / `[B,N,Rf]` | `[4,9]` / `[4,9,9]` |
| Process factor | `[B,Q,Rq]` | `[4,10,10]` |
| Observation factor | `[B,M,Ro]` | `[4,26,26]` |
| Predicted / filtered factor | `[B,N,Rp]` / `[B,N,Rf]` | `[4,9,9]` / `[4,9,9]` |
| Innovation rank | fixed `Ry` | `26` when full rank |
| Derivatives | Insert `P` immediately after `B` | e.g. initial factor `[4,18,9,9]` |

These CD dimensions come from the supplied R5 handoff, not a fresh integration
run. Choose the actual ranks from the intended model. Freeze positive `Rp`,
`Ry`, `Rf` and row permutations before differentiation/HMC; the initial factor
width must equal `Rf`. A candidate full-rank branch is:

```python
branch = TFRectangularSRUKFFixedBranch(
    predicted_rank=9, predicted_permutation=tuple(range(9)),
    innovation_rank=26, innovation_permutation=tuple(range(26)),
    filtered_rank=9, filtered_permutation=tuple(range(9)),
)
```

Identity permutations are valid only while the returned chart/pivot/support
checks accept them. The defaults retain the existing tolerances (pivot 1e-12,
chart/support 1e-10); this repair does not tune them for DZ5. Ranks and
permutations must not be rediscovered at different HMC positions. The separate
SVD discovery API is value-only and is not a score fallback.

The point rule is `tf_factor_srukf_dz5_rule(Rf + Rq)` placed using the supplied
factor columns. With ranks 9+10 it gives 39 points. Removing zero columns
changes latent rank and the point rule even when the represented covariance
is unchanged. Preserve factor orientation for parity. Do not add noise or
jitter to deterministic directions. If genuinely singular factors are needed,
select their fixed support explicitly and start new approximate-target lineage.

The update factors direct residual/loading stacks by fixed-pivot QR with
positive diagonal signs. The likelihood is Gaussian density on the declared
affine innovation support (`affine_support_gaussian_fixed_qr`); full innovation
rank recovers ordinary observation-space density. Off-support observations or
invalid charts make the analytical score invalid. The matched nonlinear fixture
tests parity of values, scores, terminal means/factors and their derivatives
between full-rank rectangular and square routes. Covariance equality alone is
insufficient to extend that result to a different factor orientation or rank.

## Exact consumer changes

In `two_currency_double_zlb_credit_target.py`, migrate the model/derivative
construction and the call currently made through
`tf_default_srukf_value_and_score`. The new rectangular dataclasses take the
same mean/factor/callback fields, but do not accept the square model's `name`
arguments. Supply the frozen `branch` to the explicit rectangular function.

Pass the existing transition helper through the new optional
`transition_value_and_derivatives_fn(x, q)` field on the derivatives object.
It returns this tuple in order:

| Output | Shape |
| --- | --- |
| Transition value | `[B,R,N]` |
| State Jacobian | `[B,R,N,N]` |
| Process Jacobian | `[B,R,N,Q]` |
| Direct parameter derivative, holding x/q fixed | `[B,P,R,N]` |

BayesFilter invokes this joint callback once per date and combines it with the
incoming point derivatives. When omitted, the original separate callback
contract still works. MacroFinance must assemble all four results from one CIR
evaluation to obtain reuse; wrapping three independent helper calls in a new
function does not obtain that reuse. Parameter dependence through state/process
points remains BayesFilter's recursive derivative responsibility.

For per-target CIR preparation, prove which innovation points are date-invariant
after the rectangular rank/orientation decision. Precompute quantiles and their
required derivatives within that parameter evaluation when valid, and close
over those tensors in the joint callback. Preserve both tails, the CIR law,
parameter derivatives and invalid-row behavior. Do not cache these tensors
across different HMC parameter states. MacroFinance owns this model-specific
change and its numerical qualification.

Cache one outer function for each static batch/parameter shape, as the existing
adapter already does. Return only numerical tensors from it:

```python
@tf.function(
    input_signature=[tf.TensorSpec([batch_size, parameter_dim], tf.float64)],
    jit_compile=True,
    autograph=False,
)
def compiled_target(theta):
    model, derivatives = build_rectangular_model_and_derivatives(theta)
    result = tf_rectangular_srukf_value_and_score(
        observations, model, derivatives, branch=branch,
    )
    valid = result.diagnostics["score_valid"]
    valid &= tf.math.is_finite(result.log_likelihood)
    valid &= tf.reduce_all(tf.math.is_finite(result.score), axis=-1)
    return (
        result.log_likelihood, result.score, valid,
        result.diagnostics["branch_status_code"],
        result.diagnostics["minimum_chart_pivot"],
        result.diagnostics["maximum_chart_residual"],
        result.diagnostics["maximum_support_residual"],
    )
```

`build_rectangular_model_and_derivatives` in this example is the consumer's
migrated builder. Keep prior value/score and existing input-admissibility checks
inside the outer function as well. The filter's convenience API creates an
inner function when called eagerly; enclosing it once prevents repeated
reconstruction/tracing on hot parameter evaluations. Tensor state is never
cached across different theta values.

Update `two_currency_double_zlb_credit_estimation.py`'s numerical-status mapping:
rectangular outputs use `score_valid`, `branch_status_code` (0 valid, 1 invalid),
`minimum_chart_pivot`, `maximum_chart_residual` and
`maximum_support_residual`. They do not supply the square route's
`relative_qr_pivot`, `row_class_code`, `roundoff_repair_count` or
`valid_pre_regularized_score` fields. Do not invent a condition estimate or
silently inherit those diagnostics. Preserve the consumer's explicit rejection
value/score policy and per-chain invalid status. Branch strings/identity and
other reporting metadata stay outside the compiled return values.

## Native HMC and remaining qualification

Use the native `HMCTransitionArchiveConfig` with
`chain_execution_mode="tf_function"` and `use_xla=True`. The exact
`HMCTransitionArchiveRunner._runner` has compiled and executed on CPU and GPU
with a nonlinear analytical-score filter, frozen affine/nonlinear coordinate
map and mass whitening. The optimized HLO, reachable graph counts, stable trace
counts and saved-momentum numerical replay are in `cpu-v5` and `gpu-v4` under
the result artifact root. This covers `kernel.one_step` and the archive loop,
not just a `sample_chain` surrogate.

The existing MacroFinance adapter advertises
`full_chain_xla_diagnostic_ready=False`. Retain that limitation until its exact
updated target and native runner successfully execute with saved outer HLO,
finite/invalid-row status checks, deterministic numerical replay, and stable
tracing. Do not promote the flag solely because the generic fixture passed.
No compilation failure may silently select `use_xla=False`.

MacroFinance next needs to verify:

1. Exact CD ranks, factor orientation, support and point rule, including
   deterministic state directions; value/score/moment and validity comparisons
   on matched full-rank fixtures and independent score differences.
2. The complete revised import/callback closure, CIR joint-call/hoisting behavior,
   prior and rejection status mapping, and the real target/native archive XLA
   block on CPU and GPU. Pin actual dirty-source hashes as well as Git HEAD.
3. The owner-requested GPU versus 1/2/4/8-core CPU comparison, including remaining
   host-controlled special-function loops. The generic fixture timings do not
   decide this comparison.

The saved R5 profile attributes 97.5% of host predicate transfers to gamma-family
CIR work. QR batching and whole-block compilation alone do not establish a
repair of that bottleneck. No new synchronization profile was collected here.
For L6, accepted value/score reuse means six new evaluations per transition plus
one bootstrap per archive block; the 16-transition block has 97 evaluations.

If the selected rectangular representation changes the approximate target,
requalify old scores, mass and transports under new lineage. This handoff issues
no tuning artifact. For later ordinary tuning use `tune_hmc_kernel`; for a
supported frozen transport use `tune_fixed_transport_hmc_kernel`, following
`docs/reference/hmc-tuning-interface.md` and
`HMC_TUNING_INTERFACE_CAPABILITIES`. A native archive runner is not a tuner.
R-hat is reporting-only during kernel tuning under the current owner directive;
posterior convergence assessment remains a separate evidence stage.
