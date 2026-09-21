# Singular design-condition diagnostic decision

The uniform-cloud XLA fitter is blocked by one field on exact duplicate-column
fixtures: `design_condition`. The original and XLA raw precision, design rank
and rejection agree at the existing 1e-10 tolerance, but the condition ratios do
not. The current contract in `filter_gradient_quadratic_numerics_20260921.md`
requires every original field unchanged, so the following correction is a
proposal, not an installed change or waived test.

Evidence is preserved in runs 01881 CPU and 01882 GPU. For D3/D5, the original
ratios are 2.435e16/1.153e16 CPU and 2.684e16/4.795e15 GPU; XLA reports
2.594e18/1.411e17 CPU and 5.459e17/4.286e17 GPU. All report ranks D-1 and reject
the geometry. The fixture explicitly sets the last column equal to the first.
Thus the real matrix has a nonzero null vector e_first - e_last; its smallest
singular value is zero and its mathematical 2-norm condition is infinite.
Finite ratios here measure SVD roundoff. Matching those ratios is not evidence
of accurate singular conditioning, although it remains required by the current
all-field execution contract.

Proposed correction: preserve the existing rank threshold, raw least-squares
coefficient, precision, optimizer controls and all decisions. Define the
reported design condition as infinite when the already-existing numerical rank
is below the dimension. For full numerical rank, retain the same largest/smallest
singular-value ratio. Document that this field describes conditioning under the
existing numerical rank policy. A nearly singular full mathematical-rank matrix
can be numerically rank deficient under that unchanged threshold; its reported
infinity would therefore refer to the rank policy, not assert exact algebraic
singularity.

Approval would authorize this diagnostic definition change only. It would not
authorize dropping the field, loosening tolerances, changing the rank threshold,
changing fitted precision, tuning, or bypassing a rejection. Tests must retain
the original raw ratio as historical evidence, require all other original
fields unchanged, assert infinite condition independently for rejected-rank
cases, and check finite/full-rank conditions at the original tolerance. Cover
exact duplicates, zero design, near-threshold designs on both sides, nonsingular
rotated designs and original consumer decisions. Inspect the actual consumer
`posterior_curvature_refinement.py` condition and rank checks and compare its
complete result, plus uniform quadratic refinement, on CPU/GPU before enabling
XLA. Numerical and cost gates remain mandatory.

The alternative is to retain the raw SVD ratio definition and continue trying
to reproduce each backend's rounding with an XLA implementation. Uniform XLA
stays blocked in that case. No runtime or test has been changed for this proposal.
