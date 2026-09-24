# iAPF underflow-guard repair result

The repair addresses the diagnosed condition in which the profiled density
objective is represented as zero while its normalized shape residual remains
positive. That condition is floating-point underflow of the amplitude-scaled
objective, not an exact fit. Before this repair the diagnostic was retained but
the fit was still admitted as valid and converged.

The implementation now defines `objective_underflow` once in
`bayesfilter/score_study/iapf_fit_tf.py` and requires its negation in the
existing `valid` flag. The recursive kernel already propagates that flag, and
the nonlinear iAPF adapter already fails closed when a fit is invalid or
unconverged. Therefore the actual consumer call chain is guarded without
changing the objective, optimizer, fitted coefficients, score formula, or
healthy-path arithmetic. The finite underflow diagnostics remain available for
the result report.

The source was patched first in an isolated checkout based on
`f5a4d411a4a0197fe9c1d25c24573337a633c96a`, then ported as the same minimal
patch to the main checkout. The negative fixture now asserts `valid == false`
and `objective_underflow == true`; the positive exact-fit and analytical
gradient tests remain unchanged.

Both checkouts pass the focused CPU/XLA reference and consumer suite:

```
19 passed, 2 warnings in 41.62s  (isolated checkout)
19 passed, 2 warnings in 43.17s  (main checkout)
```

The logs are preserved in
`isolated-cpu-focused-tests.log` and `main-cpu-focused-tests.log`. Bytecode
compilation and `git diff --check` also pass. The run intentionally hides CUDA;
the guard is a Class B admission check and does not alter device arithmetic, so
an additional GPU smoke would not answer a new numerical question. The prior
calibration allocation is already closed at 64 charged attempts and two
launches, so no GPU rerun is charged to that phase.

## Decision

| Question | Result | Interpretation |
|---|---|---|
| Healthy fits retain their accepted behavior | Pass | Existing positive-path and gradient regressions pass. |
| Reproduced underflow is admitted | Fail as intended | The fixture remains finite and diagnostic, but `valid` is false and the consumer rejects it. |
| Calibration candidate is promoted | No | The prior selected rows are now invalid under the guard and remain holdout evidence only. |
| iAPF/twisting direction is rejected | No | This repair tests admission validity, not score quality or proposal usefulness. |

The calibration result therefore remains `partial_budget_exhausted`, with no
ranking, default change, or scientific promotion. Its missing curved heuristic
table and claim-stream iAPF baseline must be collected in a new phase rather
than inferred or backfilled from old rows. The next phase must use fresh data
and a new bounded budget, reject invalid recursive fits before score comparison,
evaluate the frozen-control claim baseline, complete both conditional heuristic
tables, and obtain independent replications. Full numerical-control
calibration and fitted-moment integration into LEDH remain open master work.
