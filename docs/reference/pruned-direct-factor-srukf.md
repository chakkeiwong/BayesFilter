# Pruned second-order direct-factor SRUKF

Use `TFPrunedSRUKFModel` and `tf_pruned_srukf_filter` from
`bayesfilter.nonlinear` when the model has separate first- and second-order
states. Use `make_pruned_srukf_value_and_score` to compile a parameterized
model and its total likelihood gradient. All three APIs use TensorFlow;
compiled entry points default to XLA and have explicit tensor signatures.

Pruning is part of the model definition. If `first` and `second` each have N
components, the filter carries their **joint** 2N-dimensional distribution:

```text
first_next  = A first + B innovation
second_next = A second + c + 0.5 H[(first,innovation),(first,innovation)]
observation = d + C(first_next+second_next) + 0.5 J[first_next,first_next] + noise
```

The tensors H and J are ordinary Hessians. Supply both symmetric mixed
state/innovation entries; together their two half terms give the full mixed
term. Supply any second-order volatility constant explicitly as c. The
library does not derive model coefficients or infer the order of an intercept.
Second-order states do not feed into the quadratic terms. The filter updates
both components from observations; a correction computed only at the filtered
mean would discard that joint uncertainty.

| Model field | Shape |
| --- | --- |
| `initial_mean` | `[B,2N]` |
| `initial_factor` | `[B,2N,R]`, `0 <= R <= 2N` |
| `transition_matrix` | `[B,N,N]` |
| `innovation_loading` | `[B,N,Q]` |
| `transition_hessian` | `[B,N,N+Q,N+Q]` |
| `second_order_constant` | `[B,N]` |
| `observation_constant` | `[B,M]` |
| `observation_matrix` | `[B,M,N]` |
| `observation_hessian` | `[B,M,N,N]` |
| `innovation_factor` | `[B,Q,Rq]`, `Rq <= Q` |
| `observation_factor` | `[B,M,Rm]`, `Rm <= M` |

All dimensions are static and all tensors use float64. Factors multiply their
transpose to give covariances. A deterministic second-order initial state
uses an initial factor with zero lower rows, without an added ridge. A
parameter-dependent initial law and its derivatives belong to `model_fn`.

```python
from bayesfilter.nonlinear import make_pruned_srukf_value_and_score

value_score = make_pruned_srukf_value_and_score(
    model_fn,
    tf.TensorSpec([batch_size, parameter_dim], tf.float64),
    tf.TensorSpec([batch_size, time_steps, observation_dim], tf.float64),
)
value, score, diagnostics = value_score(parameters, observations)
```

`model_fn(parameters)` returns a `TFPrunedSRUKFModel` and must preserve row
locality: row b depends only on parameter row b. The score includes the
dependence of every coefficient, initial factor and noise factor. See
`tests/test_pruned_srukf_tf.py` for a complete small model and independent
covariance and finite-difference references.

The unscented rule is fixed at alpha=1, beta=2, kappa=0 in dimension 2N+Q.
Rectangular factors are zero-padded to keep this rule and its dimensions
fixed. Transition and observation use the same propagated cloud. This rule
does not integrate every fourth-order polynomial exactly, so the resulting
Gaussian likelihood is an approximation even for a quadratic model.

For every observation the filter builds a joint residual stack and applies
scaled Householder QR directly. The innovation factor occupies its upper-left
block, the conditional state factor its lower-right block. The mean update
uses a triangular solve with the innovation factor. There is no recurrent
covariance construction, covariance downdate, eigenvalue floor, Cholesky
refactorization, or principal square root. The strict full-rank direct-factor
SRUKF remains a separate API.

The QR implementation permits exact zero state pivots and differentiates that
fixed structural branch. It does not certify derivatives at parameter-induced
rank changes or identify an arbitrary numerical null space. Innovation pivots
must be strictly positive and all inputs and outputs finite. Failed rows
return `valid=False`, `invalid_count=1` and a negative-infinite likelihood;
the score wrapper emits NaN for an invalid score. Check this numerical status
under XLA; Python/TensorFlow assertion operations alone are insufficient.

For GPU use, select one device and enable memory growth before importing
TensorFlow. HMC requires a separately qualified enclosing XLA chain and the
appropriate BayesFilter tuner. Replacing an unpruned target invalidates its
old target-specific tuning and validation; reusing trained transports requires
fresh checks on the new target.
