# Highest-precision dot derivative and graph qualification

The uninstalled GenUT dot trial removes much of the observed large CPU cost,
but TensorFlow has no registered derivative for its raw XLA dot. Qualify a
TensorFlow custom pullback before considering runtime use. The current native
reduction implementation remains the numerical authority. This unit does not
alter the GenUT report predicate, tolerances, caps, controls, TF32 setting,
canonical LEDH score policy, or any admitted runtime route.

For rank-two real operands, orient the dot as `C=L R`, where `L` has its
contracting axis last and `R` has its contracting axis first. Given output
cotangent `G`, the pullback is `dL=G R^T`, `dR=L^T G`; transpose back to the
original operand orientations. Implement the same highest-precision dot in
the pullback, with a custom derivative on every recursive contraction so
higher derivatives remain defined. All four contraction orientations and
FP32/FP64 are in scope. Each program owns an explicitly shaped, default-XLA
function; no global callback cache, NumPy numerical path, Python data loop,
pfor or substitute LEDH score is introduced. Initially keep the candidate in
an explicitly diagnostic test module until the remaining adoption gates pass.

The primary correctness criteria are independent matrix-value/VJP checks,
centered directional finite differences, and mixed second derivatives on
nonsymmetric rectangular operands. FP32 value/VJP bounds stay at `2e-5`
absolute/relative; FP64 uses `1e-10`. Independent FP64 finite differences use
`1e-6` absolute/relative. Exercise changed operands, exact replay, one trace,
stable HLO, explicit graph-parent calls through the owned XLA function, and
actual owner/function collection. A graph parent containing this compiled dot
must be labeled as containing an XLA subprogram; it is not an all-non-JIT arm.

Next compare the complete reduced GenUT output and derivatives against the
current reductions on deterministic nondegenerate clouds. Keep every report
field in same-mode output comparisons. Compare VJPs to the current TensorFlow
program and directional derivatives to independent FP64 finite differences;
autodiff here is an independent diagnostic, never a canonical LEDH score.
Preserve failures before assertions. Any missing derivative, changed accepted
record, nonfinite result, retracing, or retained Python owner blocks adoption
and triggers a localized repair. Differentiate smooth numerical fields only;
discrete status/report predicates remain in complete value comparisons.

Run registered `genut_dot_pullback_*` groups with the existing tf-gpu Python
and campaign runner. CPU groups explicitly pass `--device CPU`; GPU groups use
trusted execution and the existing non-display selection/memory-growth policy.
Reserve at most 12 CPU workers / 3,600 seconds and four GPU workers / 1,200
seconds within the unchanged 56 CPU / 52 GPU process-hour caps. Each worker
has a 300-second timeout. Fresh campaign directories preserve commands,
versions, seeds, device settings, source hashes, output comparisons and logs.
Do not mutate source while a worker runs. Stop on source/input drift, invalid
independent reference or exhausted allocation; ordinary implementation failures
may be repaired and retried within the same allocation.

Skeptical review: a correct primitive VJP does not establish the derivative of
the enclosing iterative correction; check both. A custom first derivative can
still fail at second order; check that explicitly. Inner XLA functions may
change fusion or add overhead, so the prior raw-dot timing result cannot be
transferred to this implementation. Passing this unit only removes derivative
and graph/ownership blockers. Fresh-process host/device memory, size-dependent
CPU/GPU costs and full consumers remain prerequisites for adoption. No static
dispatch threshold is invented and no performance ranking is inferred here.

Run 04168 rejects the recursively custom dot pullback: all eight second-order
traces raise `InaccessibleTensorError` on an operand in the nested forward
function. Configuration rejection passes. Preserve this implementation under
that run's `diagnostic-source` directory. The repair computes `G R^T` and
`L^T G` with native FP32/FP64 broadcast products and reductions, which have
registered higher derivatives and do not use TF32 products. This changes the
trial's pullback implementation, not its mathematical derivative or forward
raw dot. It avoids a recursive custom-gradient scope dependency. It may cost
more backward memory/time, so the original raw-dot timings still cannot support
adoption. Repeat all primitive criteria before proceeding to the full program;
the failed attempt remains charged within the original allocation.

Run 04169 passes all nine primitive checks after that repair. Full-program
04170 then fails XLA reverse-mode compilation because the existing GenUT
`tf.while_loop` recurrences have no declared `maximum_iterations`. The failure
occurs after graph tracing, when XLA cannot size reverse-mode TensorLists.
First reproduce on the unchanged current reduction kernel using the registered
`genut_transitive_gradient_f64_cpu` group. If confirmed, add only the existing
fixed diagonal/pairwise step counts as `maximum_iterations`; do not change the
loop condition, body, control defaults or iteration count. This is a runtime
execution repair within the authorized campaign, not dot adoption or analytical
LEDH-score admission. These current-kernel regression groups are mandatory
runtime gates, unlike the uninstalled dot candidate groups.

Qualify zero, two and default four steps in FP32/FP64. Compare every value
against the pre-repair same-mode kernel at commit `28cbdb536`, compare complete
graph records and gradients against that original graph, and compare XLA
gradients and the smooth scalar loss against that graph derivative and FP64
directional finite differences. The known cross-mode thresholded report is
preserved separately and never silently waived. Repeat changed inputs without
retracing. These checks fit in the existing attempt/compute allocation; if a
new compiler or derivative defect appears, preserve it and localize it before
proceeding. Current-kernel GPU qualification also stays inside this unit.

04171 confirms the same unbounded TensorList failure on the unchanged current
kernel. Adding the fixed bounds lets the two-step FP64 case pass original
records, changed operands and independent finite differences in 04172. The
zero-step case then exposes XLA's zero-length update validation in the traced
reverse body (`Update dim size 1 greater than dynamic slice dimension: 0`).
For a zero configured step count, reserve one reverse-storage slot while
retaining the original false loop condition. Use `max(configured_steps, 1)`
as the declared bound; no extra iteration executes. This is a static allocation
bound, not an algorithmic floor on the configured steps. Preserve 04172 and
rerun all zero/two/four-step comparisons at unchanged criteria.
