# Zhao-Cui Predator-Prey Fixed-Variant Result

Date: 2026-07-23  
Status: `PASS_SEALED_CLAIM_IMPLEMENTATION_GATES_NOT_ADMITTED`  
Plan: `docs/plans/bayesfilter-zhao-cui-predator-prey-fixed-variant-active-plan-2026-07-23.md`  
Source audit: `docs/plans/bayesfilter-zhao-cui-predator-prey-fixed-variant-source-audit-2026-07-23.md`

## Decision

The target-specific Zhao-Cui-derived predator-prey route is implemented and
passes its sealed implementation gates. It is not admitted to the Zhao-Cui
leaderboard or production route. The assembled finite APF value/score program
is explicitly `extension_or_invention`; only its squared-TT defensive density,
paired-core conditional, and sequential weighting operations are source-backed
Zhao-Cui primitives.

The result answers the engineering question "can this route be implemented for
the sealed predator-prey target?" with **yes, for a fixed finite program**. It
does not establish an exact observed-data likelihood, adaptive author-route
parity, posterior correctness, HMC readiness, scalability beyond this target,
or superiority over another method.

## Evidence Contract

| Item | Declared contract |
| --- | --- |
| Question | Can a parameter-independent squared-TT/TTSIRT branch produce a finite source-order T20 value and the analytical score of that same program without a retained tensor grid? |
| Target | Sealed seed-81104 predator-prey data, `x0 -> 20 transitions -> y1:y20`, `N=1002`, physical order `(r,K,a,s,u,v)`. |
| Primary criterion | Finite deterministic value and six-coordinate score, increment identities, same-program central-FD agreement, valid fixed branch, and GPU/XLA tie-out. |
| Hard vetoes | Target/hash/order mismatch, invalid proposal or auxiliary law, collapsed claim proposal, non-finite output, runtime autodiff/FD, retained-grid fallback, failed score audit, or missing memory-growth/GPU/XLA evidence. |
| Explanatory diagnostics | Fit residuals, ESS away from the reference point, weight spread, runtime, allocator bytes, and SGQF/GenUT gaps. |
| Nonclaims | Exact likelihood, unbiased pseudo-marginal estimator, source-faithful assembled route, posterior/HMC validity, statistical ranking, default readiness, or broad high-dimensional scalability. |
| Artifact | Final GPU result: `docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/attempt_gpu_claim_final_20260723_1145/result.json`; CPU replay: `docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/attempt_cpu_claim_repaired_20260723_1100/result.json`. |

## Target And Identity

| Field | Value |
| --- | --- |
| Target ID | `zhao_cui_predator_prey_tf_seed81104_x0_then_y1_y20_v1` |
| Event order | `x0 -> transition_1..20 -> y1..y20` |
| State/observation | `(P,Q)`, dimension 2 / dimension 2 |
| Parameter order | `(r,K,a,s,u,v)` |
| Truth | `(0.6,114.0,25.0,0.3,0.5,0.5)` |
| Particle count | `1002` (`N > 1000` claim scope) |
| State SHA-256 | `63cc7d7e8e3a251f76ebb607b152b58b59cd8ceda4489057e60070b44ab1d2ec` |
| Observation SHA-256 | `fea0681d43a4bd502d1f5a90e04f58da435c6e891e72d9da4d54f4cf0584f00a` |
| FP64 branch ID | `dcef32896b89aba0d46553260d8f1c488e5722e733e33f5752395d8848660fb4` |
| FP32 branch ID | `cc19d91ab49c5d0ca01e7569e95d05858a65f7517e3c876349435e4e3ab01aef` |
| FP64 program ID | `5558d91952ba792e682e340e7ed630e587c453881005c6134c11140c43d22ec5` |
| FP32 program ID | `5559dea9110cdd23626395cad8152f048d421cc3ba47b13a106cb3e9796c9f00` |

The CPU replay and final GPU run have byte-identical frozen branch hashes,
including states, ancestors, auxiliary log probabilities, and proposal log
densities. This device-invariance check was added after an earlier GPU-fitted
branch differed from a CPU-fitted branch.

## Source Classification

| Operation | Classification | Technical anchors |
| --- | --- | --- |
| Squared-TT defensive density | `source_faithful` operation only | Paper Eq. 13, `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:539-573`; `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/SIRT.m:51-85` |
| Paired-core prefix marginal/conditional | `source_faithful` operation only | Paper Proposition 2, lines 592-670; `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m`; `eval_cirt_reference.m:43-100` |
| Frozen settings, uniforms, genealogy | `fixed_hmc_adaptation` | Paper Algorithm 3, lines 890-924; `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-43` |
| Reordered finite-grid inverse and source-order APF value/score | `extension_or_invention` | Project derivation and implementation tests; not present as one assembled route in the author source |
| Assembled route | `extension_or_invention` | Must not be called source-faithful or used for production admission |

The paper and author-code audit therefore supports implementation of the
derived route, but not a claim that it reproduces the adaptive MATLAB filter.

## Implementation

- `bayesfilter/highdim/zhao_cui_predator_prey_fixed_variant_tf.py` stores states as `[T+1,N,2]` and observations as `[T,2]`, with no observation assimilated at `x0`.
- The online value is `c0 + sum(c_t)` with fixed proposal and auxiliary corrections; the score is the analytical normalized-weight recursion of that same scalar.
- Time iteration is graph-native `tf.while_loop`; runtime finite differences and autodiff are absent.
- `bayesfilter/highdim/zhao_cui_predator_prey_proposal_tf.py` performs scope-specific offline fitting and validates the repository-issued tuning artifact.
- Offline fitting, branch draws, auxiliary-law construction, and both branch compilations are explicitly pinned to `/CPU:0`; only the claim-bearing FP32/XLA evaluator uses `/GPU:0`.
- The generic retained-grid evaluator is not called and the branch manifest records `retained_tensor_product_grid: false`.
- The harness preserves claim-specific fit/branch seeds and records repository-issued branch/program IDs in the outer result manifest.

## Tuning Evidence

Authoritative tuning artifact:
`docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/attempt_cpu_tuning_20260723_0531/tuning.json`

Artifact ID: `91ea1fe2918b7c13d137bc2460e179c392e0a1ae94a3cdf41d4d8b37afe3d7eb`  
Tuning scope ID: `3de3127272570ddfa7d7d4d02b60c74c59bdd7fbaa42e3293e4cea1daeb57f5f`

| Control | Selected value |
| --- | --- |
| Degree / rank | `4 / 8` |
| Coordinate map / scale | `gaussian_quantile / 2.0` |
| Defensive tau / ridge | `1e-6 / 1e-8` |
| Prefit / train steps | `8 / 8` |
| CDF grid / bisection | `33 / 12` |
| L1 | `0.0`, selected only after an explicit positive-L1 arm and validation margin rule |

The audit residual gate passed (`max RMS=0.31096`, `max absolute=30.30988`).
At the reference theta, validation ESS was `466.59/1002` and audit ESS was
`371.96/1002`. The interior alternative diagnostic had ESS `23.32/1002` and
is explanatory evidence of poor off-reference robustness, not a promotion
criterion for this fixed reference-point claim.

## Value And Score Comparison

Scores use physical order `(r,K,a,s,u,v)`. The Zhao-Cui row is the FP64 value
of the fixed finite program; SGQF and GenUT are same-target descriptive
comparators only.

| Method | Value | Score `(r,K,a,s,u,v)` | Value minus Zhao-Cui |
| --- | ---: | --- | ---: |
| Zhao-Cui fixed variant | `-102.41967599576537` | `(-22.67643304, 0.13827965, -0.08341680, 0.24588744, 17.60534898, -22.81586181)` | `0` |
| Fixed SGQF | `-102.62270352134469` | `(-27.64114285, 0.08410678, -0.08414332, 0.85569906, 17.52559777, -22.63497837)` | `-0.20302753` |
| GenUT | `-102.61174011230469` | `(-26.34052658, -0.03717786, -0.09041615, 1.12744200, 19.01264381, -24.49935532)` | `-0.19206412` |

These are descriptive one-seed differences. No method ranking is statistically
supported. In particular, the table does not show that Zhao-Cui is better,
more accurate, or superior.

## Claim-Gate Results

| Diagnostic | CPU FP64 reference | GPU FP32/XLA online | Status |
| --- | ---: | ---: | --- |
| Value | `-102.41967599576537` | `-102.41967010498047` | Tie-out pass; absolute gap `5.89e-6` |
| Maximum score-coordinate gap vs CPU | N/A | `8.15e-5` | Tie-out pass; threshold `2.0` |
| Minimum ESS | `448.3811/1002` | `448.3809/1002` | Noncollapsed claim screen pass; threshold `100.2` |
| Maximum log-weight spread | `21.9080` | `21.9080` | Diagnostic only |
| Increment-sum residual | `0` | `0` | Pass |
| Score-sum residual | `7.1e-15` | `3.8e-6` | Pass under dtype-specific execution |
| Same-program FD max absolute error | `5.94e-9` | N/A (audit uses FP64) | Pass; diagnostic only |
| Same-program FD max relative error | `8.64e-9` | N/A | Pass; diagnostic only |
| Branch/program identity | Exact match with GPU | Exact match with CPU replay | Pass |
| Retained grid | `false` | `false` | Pass |

GPU manifest evidence records TensorFlow 2.19.1, `/GPU:0`, `tf32_enabled=true`,
`jit_compile=true`, and memory policy
`bayesfilter.tensorflow.gpu_memory_policy.v1` with growth verified on every
physical device before logical-device initialization. TensorFlow allocator
current/peak bytes were `437,760 / 10,752,768` for the final online process.
The shared-device probe after completion showed the claim had released its
allocation.

## Decision And Inference Status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep fixed route as an implementation candidate | Pass: finite same-program value/score, exact branch identity, ESS, FD, and GPU/XLA tie-out | No hard implementation veto fired | One sealed seed and fixed `N=1002`; off-reference ESS is weak | Run a predeclared multi-seed/larger-N study after fresh scope-specific tuning | No exact likelihood, adaptive source parity, posterior/HMC, or superiority claim |
| Do not admit leaderboard/production route | Not eligible: assembled route is `extension_or_invention` | Admission boundary remains closed | Policy requires a production-admissible fixed-variant route and broader evidence | Obtain explicit route-governance review and target-specific uncertainty evidence | Passing implementation gates do not imply route admission |

### Inference-status table

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the sealed implementation claim: finite outputs, identity, increments, score audit, ESS, memory growth, GPU/XLA, and CPU/online tie-out. |
| Statistically supported ranking | None. One seed and no uncertainty interval cannot rank Zhao-Cui, SGQF, or GenUT. |
| Descriptive-only differences | All value/score gaps in the comparison table, weight spreads, ESS, runtime, and off-reference diagnostics. |
| Default-readiness | Not assessed and not eligible for promotion from this artifact. |
| Next evidence needed | Fresh scope-specific tuning, multiple seeds and/or larger `N`, uncertainty analysis, and a reviewed route-admission decision. |

## Attempts And Repairs

The initial CPU/GPU claim artifacts were retained as historical attempts. They
fit the frozen proposal in the process's default device, so CPU and GPU runs
could produce different branch hashes despite equal seeds. That exposed a real
reproducibility defect. The repair pins all branch preparation to CPU and was
verified by the CPU replay and final GPU run having identical branch/program
IDs and state/proposal hashes.

Two later GPU launches ended before artifact creation (one wrapper timeout and
one background-process cleanup). They are infrastructure failures, not
numerical failures. The final direct trusted run completed and is the
authoritative GPU evidence.

## Post-Run Red Team

| Item | Assessment |
| --- | --- |
| Strongest alternative explanation | The finite APF can be internally self-consistent while still being biased by the fitted proposal, finite-grid inverse, frozen auxiliary law, or reordered extension. |
| Result that would overturn this decision | A target/hash/order mismatch, branch mismatch, failed exact/reference likelihood check, multi-seed collapse, or a source-route governance finding would block continuation or admission. |
| Weakest evidence | One target seed, `N=1002`, no uncertainty interval, and poor ESS at an off-reference parameter point. |
| Repair trigger | Fresh calibration/validation/audit tuning and a predeclared larger-`N`/multi-seed study; do not tune on this sealed claim data. |

## Reproduction

Focused CPU-only regression command (GPU intentionally hidden):

```text
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q tests/highdim/test_zhao_cui_predator_prey_fixed_variant_tf.py tests/highdim/test_zhao_cui_predator_prey_proposal_tf.py
```

Result: `13 passed`.

Final claim command:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/run_zhao_cui_predator_prey_fixed_variant.py --claim --tuning-artifact docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/attempt_cpu_tuning_20260723_0531/tuning.json --output-root docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/attempt_gpu_claim_final_20260723_1145
```

Run manifest: commit `4281adf3c6067b706d83841bfc7a8fba022a65dd`, dirty
worktree preserved, environment `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`,
wall time `333.49 s`, offline preparation `/CPU:0`, online `/GPU:0`, and
descriptive comparators `/CPU:0`.
