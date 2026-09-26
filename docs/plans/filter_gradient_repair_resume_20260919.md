# Filter and gradient repair resume checkpoint

Checkpoint through 04187, September 26. Pushed commit `28cbdb536` contains
remote main `5eb6dcff5` through conflict-free merge `07996dd81`. Integration
04165--04167 passes (18 FAB, 6 canonical IAF, 129 policy checks). Main promotion
remains blocked and F01--F20 remain open.

The active numerical change is limited to fixed reverse-storage bounds on the
two GenUT loops. Unchanged current XLA derivatives failed compilation in 04171;
04172 exposed the zero-trip TensorList compiler case. Bounds preserve the loop
conditions and use one storage slot at zero steps. CPU FP64/FP32 04173/04174 and
GPU FP64 04177 pass zero/two/four steps, original records and finite differences.
GPU FP32 04178 fails unchanged gradient bounds; it is not qualified.

Highest-dot custom pullbacks using native reductions pass nine CPU/GPU primitive
checks (04169/04179), including mixed second derivatives and collection, and
full-program CPU/GPU comparisons (04175/04176/04180). The first recursive custom
pullback failed graph capture (04168). No dot candidate is installed in runtime.
Full FP32 GPU correctness is blocked by the underlying current gradient defect.

Independent rounded-input FP64 diagnostics 04181/04182 show healthy covariance
condition 1.3561, and source/weight/reset gradient errors in both GPU modes only
with TF32 enabled. 04183 passes eight primitive checks. 04184 identifies the
triangular-solve factor pullback as sufficient to repair GPU XLA, preserving
all forward output bits. Cholesky/matrix-solve/Gram/matvec pullback trials do not
repair graph mode (04184/04185). Graph forward values themselves vary with TF32;
XLA forward values do not. The next diagnostic replaces only the two diagonal
primal contractions plus the triangular-solve pullback, preserving equations,
controls and tolerances. All these substitutions remain diagnostic candidates.

04186 passes the combined diagonal-primal/triangular-pullback trial in both GPU
modes against independent FP64. XLA forward values are bitwise identical; every
graph forward field passes the unchanged bound. 04187 passes 129 policy checks.
No worker is active. Raw evidence/source snapshots are archived and verified in
`genut-reverse-verification-04187.json`. See the result note
`filter_gradient_genut_reverse_precision_result_20260926.md`. Plans:
`filter_gradient_genut_dot_pullback_unit_20260926.md`,
`filter_gradient_genut_reverse_precision_unit_20260926.md` and the prepared,
unexecuted `filter_gradient_genut_bounded_gradient_cost_20260926.md`.
The cost cohort is deferred until derivative correctness. No rejected derivative
may support speed ranking. The thresholded cap report remains a separate open
failure; no changed threshold, tolerance waiver or TF32 default is installed.

Charges through 04187: 86138.423980 CPU / 78093.035559 GPU seconds;
remaining 32.072660 CPU / 30.307490 GPU hours under unchanged
56/52-hour caps. The user's
added 24 CPU hours are already counted. No agents, package changes, external
MacroFinance edits or canonical LEDH rebuilding are in scope. Canonical NeuTra
remains `bayesfilter_neutra_iaf_author_v1`.

Next qualify the individual diagonal products and smallest sufficient precision repair,
then measure isolated before/after memory and costs, renew affected consumers,
archive evidence and commit/push. Broader staged/public API, initializer,
full-reset, DZ5, native-retention and terminal source-freeze gaps remain.
