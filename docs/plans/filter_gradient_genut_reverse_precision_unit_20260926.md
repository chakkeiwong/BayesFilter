# GenUT reverse precision localization and repair

GPU FP32 test 04178 fails the unchanged 2e-5 value/gradient bounds after the
loop-storage repair; FP64 CPU/GPU and FP32 CPU pass. Both GPU graph and XLA
gradients differ from CPU records on identical seeded FP32 inputs. This is a
newly executable derivative failure, not proof that the loop bounds changed
the mathematical derivative. Do not benchmark rejected derivatives as valid
work or promote the highest-dot candidate through this discrepancy.

First compare the current FP32 graph/XLA outputs and derivatives with TF32 on
and off against FP64 evaluation of the exact rounded FP32 inputs. Freeze the
smooth scalar functional's coefficient array in FP32 too, then cast that same
array into both precisions, so the reference does not change the cotangent.
Save every output field and derivative before evaluating comparisons. Record
covariance conditioning and a centered FP64 finite-difference check. TF32-off
is an explicit localization arm only; default GPU TF32 remains enabled.

The installed TensorFlow 2.19 `tensorflow/python/ops/linalg_grad.py` supplies
candidate source anchors: `_CholeskyGrad` lines 464--487 contains three matmul
operations, and `_MatrixTriangularSolveGrad` lines 685--700 contains a matmul
for the factor gradient. These may reintroduce TF32 rounding even when primal
covariances use FP32 reductions. Inspect and preserve the actual source hash;
do not infer causation from these possible sites alone.

If the toggle supports this cause, test two separate local substitutions and
their combination, initially as diagnostic source transforms. Keep the
existing Cholesky/triangular-solve forward call and replace only pullback matrix
products with TensorFlow broadcast products and reductions. Use exactly the
installed derivative equations: for `Y=L^-1 B`, `G_B=L^-T G_Y` and
`G_L=tril(-G_B Y^T)`; for Cholesky use the installed symmetrized
`L^-T Phi(L^T G_L) L^-1`, with Phi retaining the lower triangle and halving the
diagonal. Preserve all dependence terms and precision; no stopped derivative,
pfor, global TF32 change, algorithmic ridge, tolerance increase, or canonical
LEDH analytical-score claim is allowed.

Primary repair criteria: primitive FP32/FP64 VJPs against independent FP64
matrix formulas and centered directional differences; complete forward values
unchanged in the same mode; full-program gradients pass independent FP64 at
2e-5 absolute/relative on the rounded FP32 operands, and FP64 derivatives pass
2e-10 with finite differences at 1e-6. Isolate Cholesky and solve contributions
before installing a combined repair. Record rejected individual substitutions.
Differentiate smooth outputs only; the full value record retains every discrete
report. Any report discrepancy remains in its existing separate gate.

Register bounded `genut_reverse_precision_*` groups. Reserve at most eight CPU
and eight GPU workers, 300 seconds each, within the unchanged 56/52-hour caps.
Use the campaign runner, explicit devices, trusted GPU selection and growth,
and fresh versioned artifacts. This is separate from the completed pullback
primitive unit and the deferred cost cohort. Source/input drift, an invalid
reference or exhausted allocation stops this unit; a failed trial triggers
local diagnosis at unchanged tolerances. Runtime installation requires passing
primitive and full-program checks plus renewed consumers and memory/cost
qualification. No new scientific target or budget is introduced.

Skeptical review: both GPU arms may be inaccurate, and CPU agreement alone is
not an independent truth. Exact rounded inputs, frozen scalar coefficients,
FP64 derivatives and finite differences distinguish comparator failure from a
changed objective. A TF32 toggle implicates precision globally but does not
locate a particular product; separate source substitutions test that claim.
This is a first-derivative execution repair, not canonical analytical recursion
or a claim about full LEDH filtering or higher-order derivatives.

04181/04182 confirm the fixture covariance condition is only 1.3561. All
smooth FP32 CPU fields pass independent FP64; both GPU modes fail source,
weight and reset gradients with TF32 on and pass with TF32 off. The discrete
cap report remains a separate failure. 04183 passes eight primitive checks.
04184 identifies triangular-solve factor pullbacks as a cause in GPU XLA:
replacing them alone makes all smooth fields pass, whereas Cholesky alone does
not. Neither substitution repairs graph gradients; all forward outputs remain
bitwise identical. The combined trial is correctly rejected for graph mode.

Trace the remaining graph route through the diagonal correction's batched
two-by-two normal solve, Gram product and transposed matvec. The installed
`_MatrixSolveGrad` at linalg_grad.py lines 558--570 uses another matrix product.
Test that solve pullback alone, paired with the triangular-solve pullback, and
then those plus Gram/matvec and Cholesky pullbacks. The Gram derivative is
`G_J=J(G+G^T)`; for `y=J^T r`, `G_J=r g^T` and `G_r=J g`.
Forward calls remain exactly the existing TensorFlow calls. This bounded
additional-site trial uses the same reference, controls, tolerances and
remaining 8/8-worker allocation. An all-site success does not justify retaining
unnecessary substitutions; identify the smallest sufficient set before adoption.

04185 rejects the additional pullback-only candidates in graph mode. Saved
04182 records explain why: TF32 changes graph forward particles by up to
6.62e-6 and the scalar loss by 3.05e-5, whereas XLA forward outputs are bitwise
unchanged by the toggle. The graph discrepancy cannot be assigned solely to
reverse products. The remaining forward matmul/matvec calls are the diagonal
normal matrix and right-hand side. Test their exact contractions with native
FP32 reductions, alone and with the already-qualified triangular-solve
pullback. This preserves equations, dtype and controls but changes graph
rounding; require independent FP64 gradients and retain complete original
same-mode comparison results, rather than falsely requiring bitwise unchanged
forward output for a primal precision intervention. No tolerance is relaxed
and any thresholded-report difference remains explicit. This new diagnostic
still does not install a runtime repair or accept changed report semantics.

04186 passes the combined trial in both modes. 04188 isolates the graph cause
to the Gram product `J^T J`; changing the right-hand-side contraction alone
does not fix it. The smallest demonstrated repair is therefore the Gram
primal reduction and triangular-solve factor pullback. Freeze the pre-repair
numerical source at `5c9aa438e` for subsequent diagnostic substitutions and
FP64 references. No Cholesky, general-solve or right-hand-side substitution
is selected.

Qualify that candidate at dimensions 1/3/18, zero/default-four iterations,
FP32/FP64, graph/XLA, changed inputs and exact replay. Use frozen-coefficient
FP64 derivatives/finite differences and complete prior same-mode forward
records; retain the known cross-precision cap report separately. Four extent
groups (two CPU, two GPU) and one GPU primitive renewal complete this unit's
remaining allocation. The larger state dimension is a correctness fixture,
not full target capacity. The subsequent fresh-process cost cohort remains
necessary before runtime precision adoption.

04189 passes the first ten FP32 CPU extent/mode cases and then fails a weight
gradient at dimension 18/four iterations/graph. Coordinate 18 is -0.061888
versus FP64 -0.061923; its 3.52e-5 error exceeds the unchanged 2.12e-5 bound.
The original FP32 graph also fails there (3.13e-5 error). Preserve the failure;
do not attribute it to the precision candidate, call it ill-conditioned without
evidence, or relax its tolerance. Complete the remaining extent captures before
asserting the recorded numerical criteria so the first discrepancy does not
hide other modes or changed-input failures. Finite/validity and execution
failures still stop immediately. One CPU retry is reserved under the original
eight-worker CPU allocation; cost/adoption remain deferred.
