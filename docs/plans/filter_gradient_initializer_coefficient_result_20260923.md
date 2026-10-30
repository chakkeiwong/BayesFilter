# Initializer coefficient rounding diagnosis

The remaining D3 clipping-count disagreement occurs in a well-conditioned
least-squares problem (condition2.4994), at a coefficient of order1e-16. It is
not the ill-conditioned posterior precision failure. On identical saved arrays,
the original NumPy solver itself often places the coefficient on the opposite
side of zero from the independent100-digit answer. Reproducing its clipping
count therefore would not demonstrate a more accurate least-squares solve.
The existing exact discrete gate remains open; this diagnosis does not waive it.

Runs03248--03251 pass scalar/batch CPU/GPU diagnostics at the four centers from
preserved02775. Every36x2 design is full rank, with100/70-digit independent
Decimal normal-equation references agreeing below1e-60. These tiny reference
calculations are restricted to these well-conditioned arrays and do not propose
normal equations as runtime. Extra observed design outputs preserve all fitted
coefficients in these controls. Numerical package and harness hashes match
across the four runs. Source/data/hardware provenance remains in each manifest.

| Device / callback | Maximum original NumPy mu error | Maximum native QR/SVD mu error | Maximum one-correction mu error | Original/native maximum response difference |
| --- | --- | --- | --- | --- |
|CPU / scalar|4.488e-16|4.155e-16|9.203e-17|1.110e-16|
|CPU / batch|4.488e-16|5.401e-16|9.203e-17|1.110e-16|
|GPU / scalar|4.488e-16|7.360e-16|4.215e-17|0.000e+00|
|GPU / batch|4.488e-16|7.360e-16|4.215e-17|0.000e+00|

The maximum design-array difference is1.03e-16. Each solver comparison uses the
same exact stored arrays; original/native input differences are preserved
separately. GPU responses are identical; CPU responses differ by at most1.11e-16.
One residual correction reduces the observed maximum coefficient error, but tiny
sign differences remain on CPU. The original and corrected solvers must still be
compared through the complete initializer, including every status, selected
center, event and target call. The diagnostic GPU complete-candidate evaluation
follows under unchanged original assertions. Runtime remains unmodified.

The coefficient clipping count is emitted by quadratic_geometry.py and consumed
as reporting/test evidence, while precision uses the actual clipped coefficients.
A future proposal must preserve that raw count and all numerical/selection gates,
explain any additional sign-resolution diagnostic, and obtain agreement before
changing its exact comparison criterion. No blanket near-tie exception or reused
ill-conditioned-precision exception is justified here.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Preserve independent attribution | High-precision agreement, full rank and observer controls pass | No failed reference control | Finite-program downstream choices | Complete GPU residual candidate and inspect every changed field | Isolated coefficient accuracy does not establish initializer equivalence |
| Keep public initializer migration gated | Original discrete comparison not yet satisfied | CPU02775 clipping count mismatch remains | Appropriate reporting of numerically unresolved signs | Prepare a concrete bounded repair/criterion only after full evidence | No runtime solver change, count waiver or initializer promotion |

Post-run review: fitted input differences alone do not explain the whole result,
because NumPy disagreements also occur on identical arrays. The weakest remaining
link is complete downstream behavior after correction, particularly on GPU.
A correction that changes another healthy decision fails the present repair gate.
Receipt: artifacts/filter-gradient-repair-20260917/initializer-coefficient-analysis-03251.json.

GPU03252 completes the full residual candidate and fails the unchanged strict
count assertion. Exhaustive saved-record inspection finds only two D3
mu_clipped_count differences per scalar/batch record and their copied events;
D1 records/events/calls match, and all remaining D3 fields and calls match.
The complete field-review receipt is
`initializer-residual-gpu-field-review-03252.json`. No runtime repair is installed.
The [concrete reporting-count proposal](filter_gradient_initializer_clip_reporting_proposal_20260923.md)
preserves raw counts and every actual decision; it remains pending agreement.
