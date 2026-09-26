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
