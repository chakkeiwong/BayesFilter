# Filtering and gradient policy audit

Scanned `613` Python files; selected `596` algorithm modules.

## Static findings

- Numerical-loop candidates: `5120`
- NumPy import findings: `55`
- `tf.function` declarations: `308`
- Explicit non-XLA declarations: `31`

Loop classifications: `host_orchestration`=2296, `numerical_filter_candidate`=5109, `numerical_gradient_or_transport_candidate`=11, `other_algorithm_dependency`=1, `reference_or_diagnostic`=1555, `schema_or_validation`=1137, `simulation_or_predictive_boundary`=49

The JSON report contains exact paths, functions and source hashes. Findings require runtime-role review; reference, diagnostic, schema and host loops are not automatically policy violations.
