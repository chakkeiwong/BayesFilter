# Transitive GenUT execution repair and import audit

Question: do the uncovered GenUT correction and model-callback operations
preserve their existing finite outputs when executed in one enclosing XLA
program, without Python numerical iteration? Import discovery through run
03993 found 17 modules outside the guard. This is a search overapproximation,
not proof that all 17 execute from every endpoint. Classify each construct and
record actual consumer relationships before extending coverage.

The first bounded numerical unit converts the diagonal and pairwise iteration
in `dual_cap_genut_primal_tf` to sequential `tf.while_loop`, and the Austria
callback's infectious-coordinate matrix to batched `tf.one_hot`. Baseline is
the exact Git source at `072959c00`, evaluated afresh. Preserve update order,
all controls, ridges, caps, validity thresholds, output fields and supplied
random inputs. The reduced primal correction is not the canonical LEDH
trust-region algorithm; repairing it grants no canonical admission. Its
existing callers and canonical-claim blocks must remain explicit.

Primary gates: complete frozen-source records for FP64 and FP32, same-mode
original/candidate parity, cross-mode diagnostics, scalar-state pairwise no-op,
zero iterations, nonuniform weights, changed operands, exact repeated calls,
one trace and enclosing HLO without host callbacks. Independently verify
restoration of weighted mean/covariance, the radial-cap bound and infectious
coordinate selection. The correction is primal-only: autodiff, if used in a
test, is an independent diagnostic and cannot establish an analytical score.
Exercise the real reset consumer with policy-compliant K=N and preserve its
candidate identity. Reference tolerances are fixed before execution at
2e-10 absolute/relative in FP64 and 2e-5 in FP32 for these full-record
comparisons; discrete outputs and exact replay must match exactly. These are
new local fixture bounds, not changes to any existing acceptance test.

Use N=72,d=3 and d=1 standard-normal primitive fixtures (seeds 101/102,
103/104,105/106) evaluated anew. These dimensions/seeds come from an existing
primitive fixture; no old LEDH result is reused. They are execution fixtures,
not calibrated numerical settings or evidence of scientific superiority.
Include a second changed cloud/weight operand. A small, full-rank covariance
keeps factorization failure from obscuring loop conversion; scalar and zero
iteration cases expose accidental body execution or changed accumulator
identities. For cost diagnostics, use identical frozen inputs and retained
owners in fresh original-graph/original-XLA/native-graph/native-XLA processes,
20 exact replays, recording cold/warm time, trace/HLO size, host RSS and TF
allocator current/peak. One cohort per device is descriptive, not a ranking.

Classify the remaining metadata, shape and diagnostic constructs individually.
Only exact host-schema/reporting allowances are eligible; no blanket module or
numerical-loop waiver. The SIR simulation needs its own source-anchor/time and
clip-semantics review before changing it. KR transport numerical host loops
remain open until actual runtime/reference consumers are established; a
diagnostic module title alone is insufficient. Neither route is silently
declared repaired by this first unit.

Budget: at most 16 CPU and 12 GPU supervised invocations, 3600 CPU and 3600 GPU
process-seconds, inside the unchanged cumulative 56/52-hour caps. Through
04004, 32.548484 CPU and 30.689271 GPU hours remain. Use the stable campaign
runner, one numerical worker at a time, immutable per-attempt source snapshots
and versioned output under the existing campaign root. Check non-display GPU
availability and verified memory growth. Freeze numerical sources during runs.
Preserve failures and localize them under the same gates; unexplained output
or decision changes block adoption. Source drift, invalid evidence, missing
provenance or exhausted budget stop the affected comparison.

Skeptical review: Python-unrolled and native-loop graphs may be optimized
differently; compare same-mode outputs as well as graph references and report
any TF32 effect rather than relaxing a failed gate. Zero-step bodies can still
be traced by XLA. Reduced primal-route parity must not imply a canonical
algorithm rebuild. Fixture throughput cannot answer target-scale capacity or
native cache eviction. Existing original precision, DZ5, initializer,
reporting/isotropic and terminal F01--F20 gaps remain open. This plan answers
the execution question with fresh original source comparisons and independent
moment checks; it does not promote any old LEDH result or change a numerical
method. Review passes for this bounded execution-only scope.

04005 passes the FP64 primitive cases. 04006 preserves an inherited FP32
cross-mode report discrepancy: `fraction_coordinatewise_cap_active` is
0.8888888955 in both original and repaired XLA and 0.8935185075 in original
graph. Other saved fields differ by less than 1e-5. The unchanged reporting
predicate uses displacement >1e-7, near single-precision resolution. Separate
same-mode repair checks from the cross-mode complete-record test, retaining the
latter as a mandatory unresolved gate with the original tolerance. Record all
cross-mode differences; no reporting waiver, changed threshold or canonical
promotion follows. This is localization of an inherited comparison problem.

04007 passes both reset-consumer dtypes but exposes a harness boundary error:
constructing the Austria callback inside XLA attempts to run the existing
host-only model static-spec validation. Construct the callback at the existing
host configuration boundary, then compile and inspect the actual observation,
Jacobian and residual evaluation. Save r1 source and failure; r2 changes tests
and registrations only. The numerical candidate is unchanged.

04010 passes the FP64 GPU 3 primitive cases after GPU 2 became busy.
04011 aborts inside TensorFlow while evaluating the FP32 group; pytest capture
hides the native fatal message. Preserve the failure and rerun with capture
disabled to identify the cause. No numerical source or tolerance changes.
GPU correctness under recorded sharing is not uncontended timing evidence.

04012 reproduces the abort and exposes its cause: the GPU GEMM fusion autotuner
cannot combine dimension orders for the pairwise projected-direction matrix
product, whose factor is `0.5*(cross+transpose(cross))`. Candidate r4 expresses
the product with `transpose_b=True` on this exactly symmetric factor. This
preserves the factors and contraction terms; no operation is redistributed,
precision changed or compiler optimization disabled. It is a local layout
repair hypothesis. Requalify all same-mode records, moments, exact replay and
consumers on both devices; reject the candidate if it changes the existing
comparison gates. This attempt stays within the unit's compute allowance.

04013 rejects r4: transposing the symmetric right operand produces the same
failing fusion. Candidate r5 instead expresses `U*S` as `transpose(S*U^T)`
for the same symmetric S, moving the factor to the left of the contraction.
This does not distribute a product over a sum, change cap controls, or add an
N-by-d-by-d intermediate. It remains a compiler-layout hypothesis subject to
all original record gates. No failed attempt is scientific admission evidence.

04014 also rejects r5: moving the symmetric factor to the left retains the
mixed-layout fusion defect. Candidate r6 uses explicit elementwise products
and reduction for this contraction. This preserves the mathematical product
and update order but may change internal floating-point reduction order; the
unchanged complete-record gates decide acceptance. It introduces a logical
N-by-d-by-d intermediate, which XLA may fuse; measure graph and XLA allocator
cost and preserve a target-capacity limitation rather than asserting no cost.
Do not use a compiler-wide optimization/TF32 disable or XLA-only barrier that
breaks explicit graph-reference execution. Re-run CPU numerics before GPU.
Allow up to 16 GPU supervised invocations inside the unchanged 3600-second
allowance to cover the preserved compiler failures and final qualification.

04015 passes r6 FP32 CPU. In 04016 the native GPU program compiles, then the
frozen original FP32 XLA program aborts with the same GEMM-layout defect
(`MatMul_7` in the unrolled original, after the native compiled-cluster log).
The original default FP32 GPU XLA arm is therefore unavailable. Do not keep
relaunching a fatal original or patch it and call the result the original.
Candidate r7 changes the harness only: explicitly use frozen graph for that
unavailable comparator and record cross-mode failures without conflating them
with same-mode qualification. Full strict cross-mode equality stays a separate
mandatory gate. FP64 and CPU retain original-XLA comparisons. On FP32 GPU,
independent moments, valid decisions, exact replay and native HLO can qualify
individually; complete original-XLA numerical equivalence cannot be claimed.

04017 confirms native FP32 GPU compilation but fails independent mean
restoration: maximum error 3.38e-5 exceeds the fixed 2e-5 bound. This is a
numerical acceptance failure, not just the thresholded-report issue. Preserve
the candidate and localize with one explicit TF32-disabled reference arm,
alongside default TF32 native-XLA and frozen original-graph arms, at zero and
four correction steps. Measure true moments independently in FP64. The
diagnostic TF32 setting cannot replace the default or waive a failed gate.

04018 localizes the moment error to TF32 contractions: it occurs with zero
correction iterations too (mean 4.26e-5, covariance 2.81e-3), versus roughly
4e-8/3e-7 with TF32 disabled and 4e-8/1e-7 in the original graph. This excludes
the new recurrence as the sole cause and makes moment-restoring contractions
the next repair target. Candidate r9 expresses weighted/uniform covariance and
the final affine map as products and reductions, preserving FP32 tensors,
global TF32 enabled, the same mathematical moments and unchanged comparison
bounds. It does not add a ridge, precision setting or change correction controls.
These reductions must pass CPU original-source comparisons and GPU independent
moments; cross-mode full records remain separately reported. Measure their
logical N*d*d temporary at the fixture scale; target capacity remains open.

04019--04021 pass r9 CPU FP32/FP64 and GPU moment invariants. The GPU complete
graph comparison passes zero-step, diagonal-only and scalar cases, but the
pairwise case retains a 9.58e-5 cloud difference and 1.45e-3 pre-cap RMS
difference. This isolates the remaining numerical drift to pairwise
contractions. Candidate r10 uses products/reductions for co-moments and the
three pairwise residual products as well. No control or tolerance changes;
the isolated zero-step/diagonal/scalar checks continue to guard unaffected
branches. Allow 20 CPU / 20 GPU supervised invocations, still at most 3600
process-seconds per device and inside global caps, for final qualification and
the preserved localized failures. Full target-size memory remains a separate
capacity gate even when these fixture comparisons pass.

04022 passes r10 FP32 CPU. Cost arms will use FP32, the default production
dtype, on the same N=72,d=3 fixture. Original GPU FP32 XLA aborts (04016) and is
excluded from timing; use its frozen graph comparison with that limitation
explicit. CPU has four fresh-process arms and GPU has three. Retain sharing
telemetry; a contended cohort cannot support an uncontended speed ranking.
Candidate r11 updates this cost registration/dtype only; numerical source is r10.
