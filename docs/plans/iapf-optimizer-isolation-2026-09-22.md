# Independent bounded optimizer on the captured iAPF fitting problems

Status: COMPLETE. All 64 independent optimizer problems completed their
diagnostic checks; 57 met the unchanged fit-convergence criterion.
[Result](artifacts/iapf-optimizer-isolation-20260922-01/result.md). Prior:
[actual fitting inputs](artifacts/iapf-fit-input-isolation-20260922-01/result.md).

Question: do the remaining relative-shape nonconvergences arise from the local
projected-gradient solver, or do the same sampled targets and diagonal family
remain problematic under an independent bounded quasi-Newton solver? This is
optimizer diagnosis on fixed inputs, not a new filter or default policy.

Evidence contract: use exactly the preceding 64 problems (four cases, four
times, actual/exact targets, density/relative-shape objectives). Baseline is the
saved actual TensorFlow bounded fit. Compare R 4.1.2 `stats::optim` L-BFGS-B
with the same initial parameters, box, target normalization, objective scale,
and 2000-iteration cap. Respect the actual float32-to-float64 scalar casts in
the TF bounds and tolerance. R counts function/gradient evaluations separately;
equal iteration caps do not mean equal computational work.

The existing independent R analytic objective/gradient formulas are used,
with pre-solve agreement against the saved initial loss/gradients <=1e-8.
Re-evaluate final loss, energy, shape, KL, projected gradient and bound activity.
An accepted R fit requires finite parameters/metrics, solver status zero and
the original TF projected-gradient criterion <= its realized 1e-7 tolerance.
R status zero alone is insufficient. The two solvers need not have equal paths
or minima; a difference is the experiment, not a harness failure.

Research intent: convergence on the identical relative-shape problem can
nominate a solver repair; low empirical shape but high exact-message KL points
instead to sampled-cloud geometry or target error. A lower density loss can
still be objective escape, already proved in the preceding phase. Decompose
loss=energy*shape before interpretation. No numerical improvement promotes a
filter or clears a downstream heuristic veto.

Constructed comparators: unchanged cloud Gaussian, strict QR initialization,
exact KL-diagonal Gaussian and full exact Gaussian, all saved on these exact
clouds. Evaluate conditionally by case, time, target and objective. The previous
bootstrap/constant-guide/one-step/Kalman filtering vetoes remain unresolved.
No stochastic ranking or likelihood-accuracy inference is allowed.

Default audit: L-BFGS-B is chosen for an independent bounded quasi-Newton
comparison; it is not identified as the author's solver. `lmm=5` is R's
documented memory default, recorded as a diagnostic hypothesis rather than
tuned. `factr=0` disables the usual positive relative function-change threshold
to avoid substituting a small density objective for gradient convergence;
floating stagnation can still end a solve, so the original gradient gate is
rechecked. `pgtol` uses the realized original tolerance, `maxit=2000`, and
`parscale=1`. R's line search differs from the TF backtracking search and is
part of the optimizer comparison. The locally installed official `optim`
documentation was inspected and will be preserved. No hyperparameter search.

Pre-mortem: R may report success on function stagnation, or optimize a slightly
different box/scale. Explicit scalar-cast exports, start-point agreement and
the shared projected-gradient test catch these. A successful solve with poor
KL is explanatory evidence against that fit, not a successful guide repair.
No author-code, original-paper replication, universal optimizer, default,
TF32, marginal model-score, canonical LEDH or HMC claims.

Continuation vetoes: changed inputs, wrong initial objective/gradient, invalid
accepted result, missing evidence, or exhausted budget. Nonconvergent optimizer
cells and poor KL are preserved candidate failures, not reasons to abandon
the investigation. Local harness repairs may retry under the same contract.

Skeptical review PASS: the comparator changes only optimizer mechanics, keeps
the real fixed problems and numerical acceptance criterion, and explicitly
forbids interpreting proxy improvement as filtering quality. No unfair runtime
ranking, tolerance relaxation, particle expansion or production edit.

Execution: root `artifacts/iapf-optimizer-isolation-20260922-01/`; transfer
45.851 CPU / 47.803 GPU hours from the preceding exact ledger. Phase cap one
CPU hour, no GPU spending, four launches <=300 seconds including repairs.
CPU-only independent R with `CUDA_VISIBLE_DEVICES=-1` and one BLAS/OMP thread.
Command: `python3 docs/benchmarks/diagnose_iapf_optimizer_reference.py --attempt
<unique>`. Preserve inputs/source hashes, command/environment, seeds (no new
randomness), wall time, conditional results, decision/inference tables and
terminal review. Continue within existing owner authorization.
