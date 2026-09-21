# Rejected dense precision comparison proposal

Status: proposed, not installed. The mandatory1e-10 comparison remains failing.
This is a comparison-only decision; no runtime threshold, solver, score,
accepted geometry or sampling algorithm would change.

The new D3 `ill_conditioned` controller fixture uses a frozen33x3 offset matrix
whose retained two-direction condition is about1.17e8. Every arm rejects it:
design rank2, infinite policy design condition, non-SPD precision, no usable
geometry, same row counts, same failure partition and selection residual.
Only five entries of the rejected diagnostic precision matrix fail the ordinary
1e-10 comparison. Preserve all original full records in02378 and same-input
source/rank diagnostics02376--02377.

Independent100/160-digit truncated SVD in02389 agrees across both precisions.
It uses the unchanged Eigen epsilon*3 cutoff and retains exactly two directions.
The original solver itself differs from that reference by5.73e-10 in Frobenius
relative precision. Current graph/XLA differences are1.81e-9/7.40e-10. Their
relative response residuals are3.02e-16/3.44e-16. Perturbing only the original
input by one ULP fails2--5precision entries at the unchanged1e-10 comparison,
with entry differences up to3.29e-9. These observations explain sensitivity;
they do not automatically authorize a comparison change or prove correctness
of arbitrary ill-conditioned fits.

Proposed exact scope: only
`tests/test_filter_repair_posterior_curvature_extras.py::test_deficient_design_preserves_full_rejection[ill_conditioned]`,
only `diagnostics.replicates[0].precision_z`, only this frozen D3 fixture. Require
Frobenius relative error <=1e-8 against both the original and the independently
computed100/160-digit precision. Require the fixture's design checksum, exactly
one rejected replicate of rank2, unchanged rejected status, null top-level
geometry, matching complete schema/shapes and exact decisions/accounting.
All other numerical fields retain1e-10. Fail closed if the fixture is accepted,
the design/reference changes or another field is offered for normalization.
Add mutation checks for those boundaries before using the comparator.

The comparison must preserve the original strict failures alongside its proposed
result. It cannot apply to accepted fit precision, score calculations, derivative
checks, rank thresholds, runtime decisions, other designs, or other modules.
Require renewed complete controller CPU/GPU tests and original consumer/cost
evidence before public wiring. Existing approvals for iterative factor outputs,
infinite deficient-rank condition and huge-scale raw precision do not cover this
new comparison; explicit owner agreement is needed before installation.

Primary-agent review: a small residual alone can hide a wrong minimum-norm
solution, as the separate exact rank-one bug demonstrated. That bug is repaired
at the original threshold and strict criterion. The proposed ill-conditioned
exception therefore also requires independent truncated-SVD precision agreement,
exact rejection and full-field comparison. The weakest evidence is GPU coverage,
which is pending contention. This proposal does not admit the public controller,
establish posterior validity, relax filters/gradients, or close the master task.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Seek this one fixture-specific comparison allowance | Independent precision and original self-sensitivity explain the failed rejected diagnostic | Existing strict comparison still fails; no public switch | GPU and full renewed consumer results | If approved, implement fail-closed comparator and mutation tests, then qualify | No general tolerance change or algorithm/default admission |

GPU follow-up before any decision is installed:02415preserves the same
ill-conditioned precision mismatch onGPU3, but its procedurally generated
`sin` design differs in48binary64 entries from CPU02378. Standard-library
SHA-256 of compact JSON offsets is
`678c51ebbcd6d99a83eab7f3052d7d135faa26556e3d81eae59cbd66eb76d024`
for CPU and
`bcf4d5e40a900766b85b4154d2609beb4c0a511d7b3d78a34ab0a4f56020ddab`
for GPU. Each original/current comparison within its run uses identical offsets;
these are not yet one cross-device frozen byte sequence. The existing100/160-digit
reference supports the captured CPU inputs only. Do not silently apply that
reference or proposed checksum allowance to the GPU design. GPU qualification
requires its own captured-input reference or a reviewed, explicitly identical
frozen-design comparison, preserving all current failures. No allowance is
installed and the public endpoint stays unchanged.
