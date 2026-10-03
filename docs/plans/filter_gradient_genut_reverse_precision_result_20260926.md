# GenUT reverse compilation and GPU precision findings

The current reduced GenUT value program had two independent execution defects.
Its reverse pass could not compile under XLA because the iteration loops did
not bound reverse storage. Once that compiled, FP32 GPU gradients failed the
existing tolerance on a well-conditioned fixture. A diagnostic candidate now
repairs the tested precision failure while keeping GPU TF32 enabled. It has
not been installed as a runtime precision repair.

Runs 04170/04171 preserve the unbounded-TensorList compilation failures.
04172 shows why a zero-step loop still needs one allocated reverse slot: XLA
validates a traced one-element update before pruning the zero-trip body.
The current repair branch uses `maximum_iterations=max(configured_steps, 1)`
and leaves the original condition and body unchanged. Zero still executes
zero iterations. CPU FP64/FP32 04173/04174 and GPU FP64 04177 pass complete
original forward records, zero/two/four iterations, changed inputs and
independent finite differences. GPU FP32 04178 fails gradient comparisons.

The precision reference in 04181/04182 evaluates the exact rounded FP32 inputs
and frozen scalar coefficients in FP64, with a finite-difference check. The
source covariance condition is 1.3561. Both GPU modes fail source, weight and
reset gradients with TF32 enabled; both pass those fields with TF32 disabled.
This is not an ill-conditioning dismissal or evidence that changing loop
bounds changes the mathematical derivative. CPU FP32 passes the FP64 screen.

| Intervention | Graph result | XLA result | Interpretation |
| --- | --- | --- | --- |
| Current TF32-on derivative (04182) | Gradient fails | Gradient fails | Both GPU comparators need repair |
| Cholesky pullback reductions (04184) | Fails | Fails | Insufficient; do not retain without need |
| Triangular-solve pullback reductions (04184) | Fails | Passes | Localizes the tested XLA error |
| Additional solve/Gram/matvec pullbacks (04185) | Fails | Passes only when triangular solve is repaired | Graph error is not solely in the reverse products |
| Diagonal primal products as reductions (04186) | Passes | Gradient still fails | Localizes graph precision to the diagonal contractions as a set |
| Diagonal primal reductions plus triangular-solve pullback (04186) | Passes | Passes | Viable uninstalled precision candidate |

Every forward output is bitwise preserved by the pullback-only interventions.
With the combined primal/triangular candidate, XLA forward outputs remain
bitwise identical and all graph forward fields pass the original 2e-5 bounds.
The repaired gradients intentionally differ from the erroneous FP32 GPU
reference; independent FP64 establishes the correct comparison. No criterion
was widened and no global TF32 flag was changed in the candidate.

The thresholded coordinate-cap report still disagrees with FP64 and across
execution modes as previously documented. Every raw record retains it.
Passing smooth numerical and derivative checks does not close that reporting
gate, admit canonical LEDH scores, or establish full reset/filter capacity.

Separately, a highest-precision raw-dot performance candidate now has a
TensorFlow pullback. Recursive custom raw-dot derivatives failed graph capture
in 04168; native reduction pullbacks pass nine CPU and nine GPU primitive
checks, including all contraction orientations, mixed second derivatives,
graph parents, exact replay, one trace and collection (04169/04179). Full
GenUT comparisons pass 04175/04176/04180, but agreement with the current FP32
GPU derivative is not independent correctness. The dot remains uninstalled;
its earlier value-only timings cannot qualify its new derivative or memory.

| Decision | Criterion status | Veto | Next action | Not established |
| --- | --- | --- | --- | --- |
| Retain loop-bound repair on repair branch | Reverse compilation restored in tested modes | FP32 GPU precision still open | Qualify the precision candidate | Complete numerical admission |
| Continue smallest sufficient precision candidate | Tested smooth values/gradients pass FP64 | Wider extents, consumers and costs unchecked | Separate the two diagonal products; qualify sizes, then isolated costs | Runtime adoption or general performance ranking |
| Keep highest-dot candidate diagnostic | Primitive derivatives pass | Full precision/cost gates incomplete | Revisit after underlying precision repair | Canonical analytical LEDH score |

Skeptical result review: TF32 on/off alone did not locate a product. The source
substitutions distinguish the triangular pullback from diagonal primal work,
but the two diagonal products still require separate attribution before calling
either individually necessary. All numerical results above use a 72-by-3
reduced-primal fixture. Larger dimensions, default iteration counts and actual
consumers can overturn the current candidate's viability. The prepared cost
cohort remains deferred; there is no new supported speed ranking.

## Extent qualification through 04207

Individual attribution 04188 selects two changes: the diagonal Gram primal
reduction and triangular-solve factor pullback. The right-hand-side product,
Cholesky, general-solve and other pullback substitutions are unnecessary for
the tested TF32 failure. Frozen numerical references now use commit `5c9aa438e`.

The extent sweep covers dimensions 1/3/18, zero/four iterations, both precisions,
both execution modes, changed operands, exact replay, fixed scalar coefficients
and FP64 finite differences. Each backend/precision records twelve cells.

| Precision/backend | Result | Remaining discrepancy |
| --- | --- | --- |
| FP64 CPU 04191 | All 12 cells pass | None in this fixture scope |
| FP64 GPU 04193 | All 12 cells pass | None in this fixture scope |
| FP32 CPU 04190 | Dimensions 1/3 and zero-step cases pass | Dimension 18/four-step weight gradient in graph and XLA, including changed input |
| FP32 GPU 04192 | All six XLA cells pass smooth values/gradients | Graph dimension 18: changed zero-step weight gradient and four-step weight gradients |

The FP32 CPU residual already occurs in the original implementation. At weight
coordinate 18, the FP64 reference is -0.0619231299. Candidate graph gives
-0.0618879795 and original graph -0.0618917942; both exceed the unchanged
2.12385e-5 bound. Other weight-gradient coordinates reach magnitude 111.64,
but that contrast alone does not prove ill-conditioning or identify the
rounding source. The full CPU capture 04190 preserves all twelve cells after
the earlier first-failure capture 04189. No tolerance was widened.

At dimension 18/four steps, the GPU graph candidate also differs from the old
graph in particles, pairwise-cap statistics and scalar loss beyond the old
comparison bounds. Those candidate values pass independent FP64; the original
graph is an inaccurate comparator in this scope. Every original comparison is
preserved. The candidate's remaining weight-gradient failures still prevent
general adoption; a rejected comparator does not make the candidate correct.

The selected triangular primitive passes GPU renewal 04207 (eight cases in the
primitive group). Current-runtime primal/consumer renewals 04194--04199 pass
on CPU/GPU, and 04200 passes all 129 policy checks without new allowances.
The unchanged cross-mode complete-record gates fail again in 04201/04202;
04203--04206 reproduce the already-established cap operand/lowering mechanism.
These renewals do not close the cap-report gate.

The precision candidate remains uninstalled. Only the bounded-loop compilation
repair is active on the repair branch. The next unit must localize the residual
weight-gradient error and determine whether it is repairable rounding or a
precision limitation requiring an explicit error disposition. Do not call it
ill-conditioned before obtaining evidence. The prepared fresh-process cost
cohort remains deferred, so no new memory or speed claim is made for this
candidate. Main promotion and the broader master-program gaps remain open.
