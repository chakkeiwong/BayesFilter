# Zhao-Cui Integrated Audit: Stage 01 Source Inventory

Date: 2026-09-11

Status: complete for the bounded inventory stage; overall audit verdict remains
`REVISE`.

## Incident boundary

The attempted integrated child did not produce a review result. It remained in
initial reasoning/spinner state and was interrupted after the saved-session
recovery had already demonstrated context overflow. No child conclusion is
used here. This file is the compact restart point for the direct audit.

## Question and evidence boundary

The question is whether the active Zhao-Cui Algorithm 2 preparation and
Algorithm 3 value/score path implements the source route described by Zhao and
Cui, and whether its mathematical assertions and claim-bearing call chain are
complete enough for further audit.

This stage is an inventory, not a scientific result. It does not establish
posterior correctness, source-scale accuracy, rank convergence, HMC readiness,
GPU readiness, default readiness, or superiority over another method. No GPU,
training, HMC, leaderboard, or long numerical run was performed.

## Checkout identity

The inspected research checkout is
`/home/chakwong/BayesFilterZhaoCui`, branch
`zhao-cui-tt-regression-20260908`, at commit
`47176bce7cdaa91ddd4466c39f90a11ad005c800`. The checkout has seven tracked
modified files and eight untracked files before this inventory; those changes
are preserved. The notes workspace is
`/home/chakwong/BayesFilter`.

## Authoritative source anchors

The paper PDF is
`/home/chakwong/BayesFilterZhaoCui/.local_sources/highdim_nonlinear_filtering/zhao_cui_tt_sequential_learning_jmlr_23-0743.pdf`.
PDF page numbers below are authoritative; the temporary `pdftotext` extraction
used for locating passages is not itself an evidence artifact.

| Claim | Paper anchor | Author source anchor | Inventory finding |
|---|---|---|---|
| Adjacent target | Algorithm 2(a), Eq. (15), PDF p. 13: `q_t(x_t,theta,x_{t-1}) = pi_hat(x_{t-1},theta|y_{1:t-1}) f(x_t|x_{t-1},theta) g(y_t|x_t,theta)` | `source/models/full_sol.m:21-43`, with `solve` calling `push_samples`, `reapprox`, then drawing from the fitted SIRT | The paper target contains both transition and observation factors. A local target omitting either factor is wrong relative to Eq. (15). |
| Squared-TT construction | Algorithm 2(b-c), Eqs. (16), Proposition 2, PDF pp. 13 and 10-11 | `source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25-51` (right-to-left mass contraction, QR core propagation) | The author route contracts the squared TT through one-dimensional mass operations and carries the resulting factors into the marginal. |
| Upper conditional | Eq. (20), PDF p. 14: current state coordinates are integrated from left to right to define the upper conditional KR map | `source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:1-12,102-153` | The conditional sampler is an inverse-CDF construction with the conditioning coordinates supplied as fixed input. |
| Per-particle draw | Eq. (21), Algorithm 3(b-c), PDF pp. 14-15 | `source/models/full_sol.m:139-173` (`eval_cirt` receives fixed retained coordinates and independent uniforms) | Each existing particle is continued directly. The inspected source path has no auxiliary categorical ancestor draw. |
| Proposal joint law and weight | Eqs. (22)-(23), Algorithm 3(c-d), PDF pp. 14-15 | `source/models/full_sol.m:185-200` computes model density against proposal density and normalizes weights | The model-to-proposal ratio is the required finite-program ratio; an auxiliary ancestor probability belongs to a different APF program. |
| Author preparation route | Algorithm 2 and author driver | `source/eg3_sir/mainscript.m:39-65`; `source/models/full_sol.m:21-130` | The author driver uses `full_sol.solve`, whose squared route constructs `TTSIRT` from `logfun_post`, after affine sample preparation and reapproximation. |

The inspected source hashes are:

```text
c547b9af2e407c7a0d28bf49ca594fed65d9794d4f37ca605edebd91f9755e35  .local_sources/highdim_nonlinear_filtering/zhao_cui_tt_sequential_learning_jmlr_23-0743.pdf
486b14cbaef914cf37d3910bdb24af598845649028161271cb8ac3d0a5361330  third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m
871354ca450524cd6656bfe318a2ca25729c501fdf6d29f55417626b56ada41a  third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m
4e20961f4330d536debb1fab24f14484a1df502e32a7e69cba2cbaf3c981d011  third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m
8a4baa02f819032145bca1be596de0cd6a4c0a97f77c7c63ef97956aebcf3b69  docs/benchmarks/artifacts/c2_completion_20260824/attempt05/attempt05_n4_failure_analysis.tex
447a4240c4cf9e5d690555eeaad047fdc58e0d55e7a08ed88f17a150dee55740  bayesfilter/highdim/zhao_cui_algorithm2_preparation_tf.py
ba08772c3ebcbb208a303f470d95747e6e3aaec993fcdc5b7e134c003d14f387  bayesfilter/highdim/zhao_cui_algorithm3_tf.py
1d9aef4e01327bac331069ec54289c2e6feb9199d37f243be89f9617bbf27aad  bayesfilter/highdim/squared_tt_engine_v0_tf.py
021bce04a5d47d1496350ab576040a26908bfed8c6aff7196e58aee5ceb174c2  tests/highdim/test_zhao_cui_algorithm3_tf.py
```

## Local route inventory and classification

The classification follows the repository Zhao-Cui source-anchor rule:
`source_faithful` requires a cited paper operation and author-source anchor;
`fixed_hmc_adaptation` freezes an author route for a differentiable finite
program; `extension_or_invention` denotes a route not present in the cited
paper/source.

| Local path | Lines | Classification | Current boundary |
|---|---:|---|---|
| `zhao_cui_algorithm2_preparation_tf.py` adjacent scalar target | 70-123 | `source_faithful` in target structure; `fixed_hmc_adaptation` in implementation details | Includes transition and observation terms and fits a square-root target, but uses a local scattered-row ALS fit, fixed reference parameter, affine chart, and explicit defensive mass. Those numerical choices are not the author MATLAB implementation and need their own evidence. |
| `zhao_cui_algorithm2_preparation_tf.py` sigma-point chart | 27-67 | `extension_or_invention` | The positive likelihood-weighted sigma-point chart and its covariance projection are local additions. They must not be described as Zhao-Cui source behavior. |
| `zhao_cui_algorithm2_preparation_tf.py` proposal construction | 124-145 | `fixed_hmc_adaptation` with local extension components | Gram mass and adjacent conditioning axes are retained; `GaussianHermiteTTTransport` and affine chart interfaces are local TensorFlow machinery. |
| `zhao_cui_algorithm3_tf.py` conditional proposal | 42-111 | `fixed_hmc_adaptation` of the upper conditional route | It requires a complete conditional TT transport, maps fixed physical conditioning coordinates into guide coordinates, samples by inverse CDF, and subtracts the affine Jacobian in the physical density. Numerical bounded-grid CDF evaluation is a distinct numerical proposal unless its law is explicitly defined. |
| `zhao_cui_algorithm3_tf.py` frozen value/score | 114-209 | `fixed_hmc_adaptation` | States, proposal log densities, uniforms, fitted coefficients, charts, and compiled paths are frozen. The score differentiates the resulting finite scalar through model terms only; it is not the total derivative of an adaptively rebuilt TT proposal. |
| `zhao_cui_algorithm3_tf.py` compiler | 212-247 | `fixed_hmc_adaptation` | Continuation is identity (`states[-1]`), with no ancestor index or lookahead probability. Metadata explicitly says `hmc_admitted: False`. |
| manuscript numerical CDF paragraph | `attempt05_n4_failure_analysis.tex:2788-2790` | `extension_or_invention` unless corrected and bounded | The text calls grid/bisection a local fixed-variant approximation and says it is “not a new probability law.” The earlier diagnostic found a smooth-density gap of `3.68e-3`; the artifact must distinguish the smooth TT density from the piecewise numerical proposal law. |
| manuscript guide and score propositions | `attempt05_n4_failure_analysis.tex:2827-2913,3055-3129` | Mathematical claims with explicit fixed-program boundary | The affine Jacobian, no-ancestor weight identity, and frozen-score recurrence are coherent conditional claims; adaptive rebuild derivatives are explicitly outside the proposition. |
| consumer wiring | repository name/AST inventory | `not checked` for dynamic dispatch; no non-test consumer found by bounded name search | The local preparation/compiler modules are imported by tests and by each other, but no claim-bearing production endpoint was found. Existence of primitives is not call-chain closure. |

The rank initializer is a separate active concern. In
`squared_tt_engine_v0_tf.py:136-148`, every internal core is initialized with
nonzero basis index zero only. The bounded diagnostic already reproduced an
effective rank-one start and RMS `0.234375` for
`h(x,y)=1+0.25xy`, while an exact rank-two warm start reached RMS about
`1.03e-10`. This is an optimization/capacity defect to repair or explicitly
bound; it is not evidence about the Zhao-Cui paper route.

## Proof and executable obligations

The next stage must check these obligations in this order:

1. **Adjacent target.** Verify that the finite local target contains the prior
   approximation, transition density, and observation density exactly in the
   roles of Eq. (15), including coordinate order.
2. **Squared-TT marginal.** Verify the Gram/mass contraction and defensive term
   against Proposition 2. Record whether the local defense is the paper’s
   `tau lambda` construction or a declared extension.
3. **Upper conditional law.** Prove the exact continuous inverse-CDF law under
   positive conditional mass. Separately define the implemented bounded-grid
   CDF/slope law and quantify its relation to the smooth TT density.
4. **Physical proposal density.** Verify the affine change of variables and
   one, and only one, current-state Jacobian determinant.
5. **Algorithm 3 weights.** Verify fixed-particle identity continuation and the
   model-to-proposal ratio. Reject any hidden ancestor or lookahead term.
6. **Frozen score.** Verify the centered recursive derivative for the exact
   frozen finite scalar. Keep the total derivative of an adaptively rebuilt
   proposal as a separate, currently unsupported claim.
7. **Chart closure.** Check whether eliminating a previous state in the TT
   marginal leaves a current-state chart independent of that eliminated state.
   If not, require physical marginalization and re-projection before reuse.
8. **Fail-closed aggregates.** Require finite log-sum-exp increments, finite
   normalized weights, positive finite weight sums, and finite ESS. The current
   evaluator writes `c` and `inc` without including them in its validity mask;
   the earlier extreme-input diagnostic accepted `valid=true` with
   `log_likelihood=-inf` and weight sums `2.0`.
9. **Rank activation.** Add a no-fire healthy-fit check after repairing the
   initializer; a rank setting must not silently collapse to rank one.
10. **Consumer closure.** Trace one claim-bearing endpoint from preparation
    through conditional draw, physical density, weights, and score. If no such
    endpoint exists, downgrade the current implementation to primitives and
    proposals rather than treating tests as production wiring.

## Current decision and next exact action

Decision: `REVISE`. The source anchors support the structural Algorithm 2/3
description, but the local route still contains unclassified extensions and
four blocking implementation/evidence gaps: rank activation, aggregate
fail-closed checks, numerical-CDF law wording, and absent claim-bearing
consumer wiring. The frozen score is valid only as a derivative of its declared
frozen finite program until a broader total-derivative route is implemented and
checked.

Next exact action: perform a bounded mathematical verification of obligations
2-6, using MathDevMCP if available, and save only compact proof/verdict fields.
Do not run a long experiment or modify runtime code before those obligations
and the source-boundary classifications are recorded. After that, run at most
the smallest deterministic CPU checks needed to discriminate a proof failure
from an implementation failure.

