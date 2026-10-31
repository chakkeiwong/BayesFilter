# Principal-angle XLA SVD accuracy repair

The actual CDF initializer investigation reproduced an accuracy defect in
`fixed_center_curvature._precision_geometry_kernel`. Its padded overlap matrix
uses the TensorFlow default SVD inside XLA. Run04578 measured maximum singular
error7.734272366999306e-8 against independent LAPACK on the identical operand.
Explicit XLA SVD with binary64 convergence gave3.1086244689504383e-15. The
selected covariance was unaffected in this case, but reported principal angles
were wrong at small angles. This is a repair trigger, not an ill-conditioning
waiver for the entire initializer.

Replace only this XLA SVD call with the existing repository pattern:
`xla_svd(max_iter=100, epsilon=sys.float_info.epsilon, precision_config='')`.
Keep graph-reference SVD, rank selection, positive-eigenvalue threshold,
unit-singular-value snap, angle conversion and admission thresholds unchanged.
The principal-angle mathematical target and factor covariance model do not
change. No anchor convention, optimizer, initialization or source admission is
changed. Other unresolved original-record differences remain vetoes.

Use the runtime at2c80ecbcc as the before baseline and a committed minimal
two-matrix fixture extracted from the hashed04578/04572 evidence. For the
actual D23 pair, compare cosines of reported angles with independent NumPy
eigendecomposition/SVD and with LAPACK on the saved overlap matrix. The stable
singular-value accuracy gate is1e-14 absolute/relative; acos near1 amplifies
roundoff, so preserve angle discrepancies under the existing1e-10 full-record
comparison separately. This solver gate does not relax those comparisons.
Include exact repeated subspaces, rank changes, and analytic rotations on both
sides of a5-degree gate. A failure of the analytic gate, source identity,
finite output, callback-free graph, stable signature or one-trace replay stops
qualification. Diagnostic-only NumPy supplies independent references and never
enters a runtime kernel.

Reserve16 workers and3600 combined charged seconds from the existing56 CPU/
52 GPU-hour totals. The prior initializer unit closed through04578 using28
workers and4427.858692 seconds; the extra24 CPU hours are already included.
One numerical worker at a time. CPU runs are explicit references; GPU tests
use an available non-display device with memory growth verified. Use the
existing campaign runner and fresh numbered output directories; no packages,
environment mutation or global cache manipulation. Workers normally get300s;
one complete saved-input fitter replay may get900s if needed. Localized
harness failures get at most three attempts per case within this allocation.

Run before/after CPU and GPU kernel probes, recording cold/steady durations,
process RSS/HWM, allocator current/peak, graph/HLO size and trace count. The
numerically defective baseline remains diagnostic timing only; do not rank
performance from it. Verify existing fixed-family stability and derivative
checks, a controlled saved-input fit replay, and final policy checks. Source
changes make the prior actual-consumer snapshot ineligible to qualify this
new source closure; later actual-consumer renewal/lifetime tests must preserve
that distinction. Archive complete records and update the master before
committing/pushing the repair. Main stays unmerged until the full master closes.

Skeptical review: a successful tiny Gaussian test would miss the padded D23
overlap failure, so the frozen actual overlap is mandatory. Comparing angle
decimal digits alone confounds SVD error with acos conditioning. Report both
without changing the old bound. Higher SVD accuracy may change a near-threshold
decision; analytic rotations exercise that behavior against its mathematical
target. Compile/runtime/memory regression thresholds remain attribution
triggers (20% warm time,2x device peak,256MiB/2x RSS), not permission to retain
an inaccurate kernel. The local review finds the proposed repair answers the
observed mechanism; no independent review or broad equivalence is claimed.
