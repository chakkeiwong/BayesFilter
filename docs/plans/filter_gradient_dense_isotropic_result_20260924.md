# Isotropic factor initialization discrepancy

The seeded discrepancy originates in the original standalone factor fitter.
It is not caused by enclosing the initializer or by the new random generator.
Runs 03537 and 03548 reproduce the mechanism with original sources pinned to
3582b4ac and identical archived clouds. All evidence is under
`artifacts/filter-gradient-repair-20260917` in those numbered directories.

The original `_weighted_dense_precision` gives an almost identity correlation
matrix. Its eigenvalue spread is at most 4.22e-15 in the D23 cases. Original
`_initial_factor_state` (factor_correlation_geometry.py, lines 429--478 at
3582b4ac) multiplies a selected eigenvector by
`sqrt(max(eigenvalue - 1, 1e-6))`. At an exactly repeated eigenvalue of one,
any orthonormal basis spans the same eigenspace. Roundoff can therefore select
different vectors without violating the eigensystem equations. The inherited
floor nevertheless gives these arbitrary directions nonzero loadings.

For one factor and unit marginal deviations, the decoded correlation is
`diag(1 - l*l) + l*l^T`. A loading with one nonzero coordinate gives identity;
a spread loading gives off-diagonal entries `l_i*l_j`. This is a direct
initialization dependence, not a new factor model. The original D23 selected
precision differs from identity by up to 1.632e-7; the current fitter returns
identity for the same clouds. Original/current ranks and anchors also differ,
as recorded in 03537. Better agreement with this exact target does not prove
full original-program equivalence.

03548 re-evaluates the original weighted fit, initialization, encoding and loss
on the preserved original clouds. Its initial precision equals the original
reported final precision exactly in all four D3/D23 replicates. All optimizers
take zero iterations. D23 maximum absolute gradients are 6.05e-12 and 6.22e-12,
already below the inherited 1e-9 stopping tolerance. The D3 gradients are zero
or below 2.77e-18. Thus the loading-floor perturbation survives the original
stopping rule. The diagnostic changes no optimizer, clipping, tolerance or
acceptance criterion and uses 4.98 charged CPU seconds.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | What is not established |
| --- | --- | --- | --- | --- | --- |
| Attribute the discrepancy to original initialization and stopping | Initial and final precision agree exactly; zero iterations and sub-tolerance gradients | No contradictory recorded case in this fixture | Other tied or nearly tied spectra and factor counts | Preserve the complete discrepancy and evaluate a separately specified deterministic initialization treatment | General equivalence or scientific admission |
| Keep isotropic original-record qualification open | Rank, anchors and precision do not satisfy the unchanged comparator | Complete-record mismatch remains | A basis convention alone may still alter the initialized covariance | A bounded repair must define the accepted numerical target before changing initialization or comparison | Permission to ignore raw diagnostic or accepted-geometry differences |

The repair path belongs to E5/F19: first retain scale-aware eigengaps, raw
loadings, initial loss/gradient and optimizer termination as observability;
then evaluate a deterministic treatment of an unresolved eigenspace with the
same factor covariance model. An initialization change must be checked on
healthy separated spectra, exact isotropy, nearly repeated modes and both
factor counts, including original decisions and actual callback counts. Its
evaluation must distinguish an execution repair from a change to the inherited
initialization/stopping method. No such method change or comparison exception
is installed by this result. Unrelated controller, cost and consumer work can
continue under the existing plan.

Review: the strongest alternative explanation was a RNG or enclosure error.
03537 reproduces the discrepancy on both original and generated clouds, whose
maximum differences are only 1.39e-17 (D3) and 2.78e-17 (D23); 03548 reproduces
the original final precision before optimization. This identifies the mechanism
for these fixtures, but does not justify a new tolerance or claim that all
isotropic inputs share the same rank. This is a primary-agent review; no
independent reviewer was used.
