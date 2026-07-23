# Higher-Moment Contract E/GenUT Campaign Plan

Date: 2026-07-23

Status: `EXECUTED_OPT_IN_CANDIDATE_NOT_PROMOTED`

## Research Intent Ledger

| Field | Contract |
|---|---|
| Main question | Can an opt-in, finite higher-moment correction applied after the existing OT + Contract E reset reduce distributional moment drift without regressing the value or recursive score of the tested model suite? |
| Candidate | `contract_e_higher_moment_projection_candidate_v1`, a TensorFlow/XLA FP32/TF32 finite map. The canonical `contract_e_chol_v1` route is unchanged. |
| Existing baseline | The current non-fused OT barycentric quotient, Contract E residual injection, and Cholesky affine restoration in `finite_value_score`. |
| Higher-moment mechanism | Estimate weighted standardized diagonal skewness/kurtosis, apply a bounded finite Hermite correction in the whitened cloud, and re-center/re-whiten after each correction. The candidate reports moment residuals; it does not claim a globally solved nonconvex projection. |
| Primary promotion criterion | No hard validity failure, same-scalar JVP/finite-difference parity on representative points, and no regression in baseline value/score gates on LGSSM, genuine actual-SV, predator-prey, and canonical Austria-SIR value parity. Higher-moment residual reduction is a nomination diagnostic, not a standalone correctness criterion. |
| Promotion veto | Nonfinite tensors, invalid Cholesky branch, failed covariance/mean identity, failed score accounting, stale tuning scope, NumPy/Python numeric path in XLA, changed canonical route, or any required model regression failure. |
| Continuation veto | If the candidate cannot pass scalar moment/JVP tests, stop candidate integration and retain documentation plus baseline regressions only. A failure to improve moments rejects the candidate, not the research direction. |
| Explanatory diagnostics | Pre/post skewness and kurtosis residuals, row-coupling cross moments, ridge and OT residuals, per-time value/score increments, runtime, allocator peak, and paired seed deltas. |
| Nonclaims | No exact nonlinear likelihood theorem, unbiasedness, exact posterior score, global optimizer/uniqueness theorem for the projection, default/leaderboard/HMC promotion, or NAWM result. |

## Skeptical Plan Audit

1. **Target mismatch:** the existing label “GenUT” is a residual design, not a full unscented transform. The candidate therefore keeps the transformation/likelihood path explicit and measures the carried-cloud moments after Contract E.
2. **Projection existence:** exact mean/covariance/skewness/kurtosis equality is a nonconvex feasibility problem and is not assumed to have a solution. The implementation uses a finite bounded correction and reports residuals; it must not claim exact higher-moment restoration unless a separate feasibility certificate passes.
3. **Derivative boundary:** the score is the total derivative of the executed finite map on a fixed SPD and active-branch path. Every correction, rewhitening, and control must be differentiated; a frozen adaptive design would be a partial derivative and is vetoed.
4. **Canonical-policy risk:** Contract E is canonical and remains untouched. The candidate has a distinct route ID and cannot silently replace or fall back to the canonical route.
5. **Wrong SIR comparator:** actual Austria SIR is fixed-parameter and value-only for this regression. The demoted reduced-SIR fixture is excluded; no fake SIR score is reported.
6. **Statistical risk:** one-seed results are diagnostic. The claim matrix uses the existing scope-specific tuning/regression harness with `N>1000`, common seeds where available, and uncertainty summaries.
7. **Resource risk:** full dense moment tensors are forbidden. Only diagonal moments and fixed-size state tensors are used, so the candidate remains compatible with high-dimensional intent.

Audit decision: `PASS_WITH_STRICT_NONCLAIMS`. Implementation may proceed as an opt-in candidate after the LaTeX/math audit and focused tests pass.

## Phases

### Phase 1: Mathematical Documentation

Extend `docs/chapters/ch32c_entropic_ot_sinkhorn.tex` with:

- the full time recursion;
- the OT coupling, barycentric quotient, and marginal conventions;
- all Contract E stages and the ridged mean/covariance proposition;
- weighted standardized diagonal third/fourth moments;
- the finite bounded higher-moment correction and exact re-centering/re-whitening proposition;
- the fixed-branch total-derivative proposition;
- positivity, feasibility, and high-dimensional diagonal-moment limitations.

### Phase 2: MathDevMCP Audit

Run focused audits on the new labels with the local MathDevMCP installation. Resolve unsupported claims, missing assumptions, and notation/code mismatches before implementation. Record the audit artifact under `docs/plans/`.

### Phase 3: Candidate Implementation

Add an opt-in correction module under `bayesfilter/highdim/` using TensorFlow only:

- weighted diagonal standardized moments;
- bounded Hermite/cubic correction;
- exact mean/covariance re-centering and Cholesky re-whitening;
- manual JVP for every operation;
- finite validity and moment-residual diagnostics;
- route identity binding and no canonical fallback.

The correction iteration count defaults to zero in the existing route and is enabled only by a separate candidate entry point.

### Phase 4: Focused Tests

Run CPU-hidden tests for moment identities, correction finiteness, Cholesky/rewhitening identities, JVP parity, zero-iteration baseline parity, static XLA shapes, and source checks forbidding NumPy, autodiff, and runtime finite differences.

### Phase 5: Scope Tuning

Tune correction strength, correction iterations, ridge, OT epsilon, balance count, row permutation policy, and moment clipping on disjoint calibration/validation partitions. The score contribution is evaluated only through the recursive candidate score; finite differences are used at representative tuning points only.

### Phase 6: Model Regression Campaign

Run untouched GPU/XLA/TF32 regressions with `N>1000`:

| Model | Scope | Reference/role |
|---|---|---|
| LGSSM | `N=1008`, `T=2,10,50`, 16 seeds | dense Kalman value/score oracle |
| Actual transformed SV | fresh DGP, `N=1998`, `T=50`, 16 seeds | refined dense transformed-SV value/score |
| Predator-prey | `N=1002`, `T=20`, 16 seeds | value/score descriptive, no exact score oracle |
| Austria SIR | canonical fixed source-order SGQF value route | value-only CPU/GPU parity and prior value |

The candidate is compared to the prior route with paired seeds where the scope is identical. No ranking is inferred from descriptive one-seed differences.

### Phase 7: Result And Decision Note

Write JSON/Markdown artifacts containing manifests, selected controls, pre/post moment residuals, value/score means, standard deviations, confidence intervals, hard veto status, and decision/inference tables. State separately whether the candidate reduced moment drift, preserved baseline behavior, or merely remained viable.

## Execution Commands

```bash
PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli audit-math-document-rigor docs/chapters/ch32c_entropic_ot_sinkhorn.tex --output-md docs/plans/bayesfilter-higher-moment-contract-e-genut-mathdevmcp-audit-2026-07-23.md --output-json docs/plans/bayesfilter-higher-moment-contract-e-genut-mathdevmcp-audit-2026-07-23.json --report-profile actionable
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/test_higher_moment_contract_e.py tests/highdim/test_cubature_genut_filter.py -q
python docs/benchmarks/run_genut_transport_repair_regressions.py --output-root docs/benchmarks/artifacts/higher_moment_contract_e_regression_20260723/baseline
```

The final GPU command is run only after the candidate tests and MathDevMCP audit pass, with a fresh versioned output root and the repository memory-growth policy enabled.

## Stop Conditions And Budget

- Candidate implementation/test budget: 20 minutes CPU and 30 minutes GPU.
- Stop immediately on nonfinite state, invalid factor, score/JVP mismatch, or baseline regression veto.
- Preserve every failed attempt in a fresh artifact directory; never overwrite prior evidence.

## Evidence Interpretation

| Evidence | Role |
|---|---|
| Mean/covariance identity | hard engineering validity |
| JVP versus representative FD | diagnostic same-scalar score check |
| Dense value/score agreement | model-specific diagnostic evidence |
| Higher-moment residual reduction | explanatory/nomination diagnostic |
| Cross-model no-regression | continuation/promotion veto |
| Runtime/memory | feasibility diagnostic only |

## Execution Record

- Focused implementation and regression tests: `26 passed`.
- MathDevMCP parsed the chapter. Its proposition-label selector selected zero
  display-equation targets; focused derivation-label calls were `inconclusive`
  with no semantic counterexample because the configured scalar backend cannot
  encode the matrix/finite-program obligations. This is recorded in
  `docs/plans/bayesfilter-higher-moment-contract-e-genut-mathdevmcp-audit-2026-07-23.md`
  and is not a proof certificate.
- The full monograph build reached the new chapter and then stopped on the
  pre-existing missing SSL-LSTM figure
  `plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/direct-visual-validation/ssl-lstm-launch-traces-z.pdf`.
- GPU attempt 01 did not start the scientific run because TensorFlow was
  initialized before memory-growth configuration; the failure is preserved in
  `docs/benchmarks/artifacts/higher_moment_contract_e_regression_20260723/attempt01/failure.json`.
- GPU attempt 02 completed with `hard_valid=true`, FP32/TF32/XLA, memory growth,
  `N>1000`, and 16 untouched seeds per oracle-backed scope.
- Candidate result:
  `docs/benchmarks/artifacts/higher_moment_contract_e_regression_20260723/attempt02/result.json`.
- Prior-versus-candidate comparison:
  `docs/benchmarks/artifacts/higher_moment_contract_e_regression_20260723/comparison_attempt01/comparison.md`.

## Terminal Decision

| Decision | Primary criterion | Veto status | Result | Next action |
|---|---|---|---|---|
| Candidate engineering viability | finite GPU/XLA execution, route identity, recursive score sum | nonfinite/invalid branch | `PASS` | retain as opt-in diagnostic |
| Value/score no-regression | paired 95% CI of candidate-minus-prior oracle absolute error | CI entirely above zero | `PASS; no supported regression` | do not claim improvement |
| Higher-moment correction | residual reduction/exact matching | exact-feasibility claim | `FAILS promotion claim` | treat residuals as diagnostic only |
| Default/leaderboard promotion | model-specific accuracy and broad evidence | unresolved LGSSM score bias | `NOT READY` | investigate a different distributional correction |

The candidate remains separate from canonical `contract_e_chol_v1`. The
campaign does not establish exact likelihood, exact posterior score, HMC
readiness, default readiness, leaderboard promotion, or any NAWM conclusion.
