# Exact-input CDF fitter diagnosis

Four fresh workers consume bit-identical CPU04591 operand arrays. Before is
5d398a45b; after is the shared clipped-anchor repair from23f681a14. The
diagnostic artifacts preserve all raw fields before interpreting numerical
status, and161 readback/policy checks pass in04610.

| Source/backend | Run | Numerical result | Worker seconds |
|---|---:|---|---:|
| Before GPU | 04606 | Rejected, stability error1 (left precision) | 69.964 |
| After GPU | 04607 | Rejected, stability error2 (right precision) | 68.579 |
| Before CPU | 04608 | Usable selected geometry | 52.680 |
| After CPU | 04609 | Usable selected geometry, exact same raw/public record | 49.754 |

The four input hashes match in every worker. CPU records reproduce04591 apart
from four explicitly diagnostic `jit_compile` fields that04591's normalization
omitted. There are no CPU numerical changes from the anchor repair on these
inputs. GPU rejection occurs before and after; the repair is not its origin.

The matrix diagnosis finds finite, positive-definite fitted matrices with
condition numbers about11,175--12,246. The GPU Cholesky inverse is asymmetric
by at most7.63e-12 in absolute entries and5.47e-16 relative matrix norm. A small
off-diagonal pair0.02116426252630558/0.02116426251867667 exceeds the existing
elementwise symmetry bound by2.448x. Its inverse backward error is9.73e-18;
the independent symmetric inverse differs by at most1.39e-9 across all entries.
This is not evidence of a singular covariance or a reason to widen thresholds.

Source tracing identifies the final precision construction in
`factor_correlation_geometry._make_factor_program`: `cholesky_solve(chol,I)`
returns independent floating-point columns without an explicit symmetric
representation. `fixed_center_stability_tf.validate` correctly retains its
finite/elementwise-symmetry check. In before-GPU factor2 replicate0 fails that
check; after-GPU it passes, but replicate1 fails in both. The nonaccepted
factor1 replicate1 also has tiny asymmetry. CPU matrices pass the same check.

| Decision | Primary criterion | Veto status | Next action | Not concluded |
|---|---|---|---|---|
| Keep both GPU fits rejected | Identical-input comparison is valid | Existing symmetry checks fail | Evaluate final symmetric inverse representation with raw-error observability and residual tests | GPU full-fit equivalence |
| Preserve anchor repair | CPU before/after raw and public records are exactly equal; GPU failure predates it | Independent initialization and GPU regressions pass | Keep the failure attribution separate | Same optimizer trajectory across all backends |
| Do not change conditioning or symmetry tolerances | Matrices are finite/SPD with small normwise inverse errors | Elementwise failure remains active | Repair the representation rather than waive the gate | A general ill-conditioning waiver |

Local skeptical review: symmetrization can conceal a bad solve unless the raw
skew and inverse residual are checked. The follow-up must preserve those
diagnostics, compare an independent inverse and analytic fixtures, and leave
optimizer arithmetic, covariance, condition cap and thresholds unchanged.
Single-run timings/memory are descriptive; they cannot rank a rejected fit.
Actual-consumer source renewal/lifetime and terminal F01--F20 work remain open.

Archive `artifacts/filter-gradient-repair-20260917/exact-fit-inputs-04610-evidence.tar.gz`
contains39 reopened/hash-verified members,1,798,496bytes, SHA-256
`dc44b1ea16c490b4f4f43f92464823af68b544cd373b66ece69ec6d5741af05c`.
It retains both rejected GPU numerical results, successful CPU results and
the independent matrix diagnosis. Its prerequisite archives are listed in the
verification JSON; no evidence is overwritten.
