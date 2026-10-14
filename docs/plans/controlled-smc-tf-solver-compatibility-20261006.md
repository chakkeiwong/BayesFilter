# TensorFlow solver compatibility on preserved author systems

Authorized continuation, 6 October 2026. Replication remains the sole target.
This diagnostic follows the completed offset localization; it does not change
the author's method, qualify the port, or launch KDM or E0.

## Question and evidence contract

Can TensorFlow reproduce the author's polynomial regression on identical
normal equations when given the author's solver branch? This tests whether
implementing branch selection alone could repair the observed likelihood gap.
The comparator is Octave's preserved answer, independently reproduced bitwise
by its linked system LAPACK. Use all 128 earlier matrix-crossover systems and
all 101 regressions from F=4.8801, seed20260930. The crossover systems include
mixed matrix/right-hand-side origins; retain those labels and do not describe
them as 128 independent stochastic trials.

Primary compatibility screen: every answer is finite, every SVD retains the
source rank, and both polynomial predictions and centered polynomial predictions
agree within the existing localization screen `1e-7 + 1e-9*abs(reference)`.
Coefficient differences, residuals, condition estimates, timing and rank
spectra are explanatory. This screen only licenses a subsequent autonomous
fitting comparison. It is not the P1 no-worse acceptance criterion, an estimate
of stochastic uncertainty, or evidence that a different solver is better.

An algebra/orientation error on known-solution controls, an altered input hash,
missing GPU/memory-growth evidence, or non-finite controls invalidates the
experiment and triggers harness repair. A compatibility mismatch is a candidate
promotion veto, not a research-direction or continuation veto. If GPU/XLA
compatibility fails, run the same systems through CPU TensorFlow graph mode
with CUDA intentionally hidden. This is an explicit non-default reference
exception to distinguish GPU/XLA behavior from a general solver-library
difference. Do not wire an already failing candidate into the full recursion.

No likelihood, robustness, default, superiority or complete replication claim
can follow from this diagnostic. Source branch choices are an oracle here;
the experiment does not implement DPOCON/DGECON branch selection. The raw
TensorFlow LU solve is a paired mechanism comparator, not the author baseline.
Heuristic dominance is not a promotion question in this arithmetic comparison;
known-solution systems supply the relevant cheap falsification controls.

## Source and mathematical anchors

The paper's Section4.2, equations20–21, defines the backward least-squares
fit; Section6.2 specializes the Lorenz model. The local tractable copy is
`.localresources/papers/heng2020-controlled-smc-published.txt`.
Author `StateSpaceModels/Lorenz/cSMC_Resample.m:193` computes
`(D'*D) \ (D'*Value)`. Octave6.4.0 `dMatrix.cc:1405–1567,1614–1628,2144–2219`
selects Cholesky, conditionally LU, then DGELSD. These anchors were inspected
in the preceding solver and localization audits and rechecked for this step.
The independent adapter is `docs/benchmarks/reference_octave_system_lapack.py`.

For `G=U diag(s) V'`, compute `beta=V diag(1/s) U' h` for singular values
above the source cutoff. DGELSD with `rcond=-1` uses this machine's LAPACK
epsilon `1.1102230246251565e-16`; do not substitute NumPy/TF epsilon, introduce
a ridge, or choose a cutoff from the observed discrepancy. The saved fallbacks
are full rank. Test the singular-rank control as well as full-rank controls.

## Default and assumption audit

| Choice | Provenance and status | Failure mode | Early diagnostic |
|---|---|---|---|
| Float64, TF32 off | Author double precision; reference comparator | GPU implementation differs despite dtype | Identical matrices, saved branch, known solutions |
| RTX4080SUPER, GPU/XLA | Same hardware as prior controlled-SMC comparison; baseline | Compiler or factorization error | Small controls, recorded device, single trace per kernel |
| Frozen source branch | Diagnostic oracle, not a runtime algorithm | Passing overstates autonomous capability | Explicit oracle field; no port promotion |
| Source DGELSD epsilon | Inspected source/system LAPACK; baseline | Rank selection differs by library | Save spectrum and retained rank for every SVD |
| Existing prediction tolerance | Localization convention, not calibrated P1 margin | Passing mistaken for no-worse qualification | Result labels and separate downstream requirement |
| CPU non-XLA follow-up | Debugging hypothesis only | Mistaken as a default GPU substitute | CUDA hidden before import; manifest exception |

## Execution, bounds and stop conditions

Prepare a diagnostic-only binary fixture without changing the preserved inputs;
verify all229 source answers again with the independent system adapter. NumPy
is confined to this independent reference preparation. The TensorFlow runner
uses TensorFlow and the standard library, stable `[45,45]` and `[45]` graph
signatures, no pfor or Python callbacks, and no numerical NumPy path. Its Python
outer loop dispatches independent saved diagnostic systems, not filter steps.

Remaining allocation at entry:849.218199 GPU seconds and151428.365656 CPU
seconds. At most2 GPU attempts,180seconds each, and one conditional CPU attempt,
180seconds. Reference preparation is bounded to120seconds. No new compute is
granted. Extend the bookkeeping attempt ceiling from157 to159 so all three
possible numbered attempts fit. Stop a timed-out worker through the existing
supervisor, preserve its evidence, and inspect before any retry. No concurrent
numerical campaign. A passing GPU result licenses planning the next fitting
step within the remaining allocation; a failure licenses only the specified
CPU localization and a reviewed next repair.

Commands use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B`:

1. `docs/benchmarks/prepare_lorenz_solver_compatibility.py --root /home/chakwong/BayesFilter --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/solver-compatibility-reference-20261006-01`
2. `scripts/run_controlled_smc_kdm.py run --phase p0_tf_solver_compatibility`
3. If needed: `scripts/run_controlled_smc_kdm.py run --phase p0_tf_solver_cpu_reference`

All GPU commands require escalated execution. Serious-run manifests and logs
use fresh numbered attempt directories under the campaign root. Result:
`docs/plans/controlled-smc-tf-solver-compatibility-result-20261006.md`.

## Skeptical review before execution

The exact author answers remain the baseline; neither a lower LS residual nor
the 1e-8 perturbation range substitutes for downstream replication. Forced
branches intentionally omit runtime branch selection and are labeled as such.
The cases are reused for diagnosis, not presented as independent validation.
Input checksums, known-answer controls, source-rank checks and stable signatures
catch the earliest invalidating failures. No protections or thresholds are
tuned against the discrepancy. Budgets and conditional follow-up are explicit.
Review passed for this bounded arithmetic question; autonomous fitting and P1
remain separate. The strongest misleading-success risk is confusing these
conditional solves with a working conditional filter; reporting forbids that.

## Localized repair after attempt157

Attempt157 stopped on the dense SPD known-solution control: GPU/XLA
`tf.linalg.svd` produced maximum coefficient error3.915788125e-7 versus the
predeclared1e-11 control threshold. It used5.829373 GPU seconds and yielded no
saved-system comparison. The installed TensorFlow XLA header
`include/xla/hlo/builder/lib/svd.h:46` declares100 sweeps and epsilon1e-6,
even for double tensors. This is a new candidate-path failure, not a claim that
SVD caused the previous LU/QR filter gap.

The remaining GPU attempt uses explicit `XlaSvd` with convergence epsilon
2^-52 (float64 machine precision),100 sweeps and HIGHEST dot precision. This
changes the convergence accuracy of our diagnostic candidate, not the author's
LS objective or its DGELSD rank cutoff2^-53. The known-solution controls must
still pass without changing their thresholds. Preserve the default-SVD control
and compiler HLO as explanatory evidence. The epsilon is derived from the
requested arithmetic precision, not tuned against author answers. The100-sweep
bound is the library default and remains a hypothesis: failed controls or
rank/prediction discrepancies prevent compatibility promotion. This is not an
author numerical repair and adds no ridge or damping. The predeclared CPU graph
follow-up uses its native SVD and records the execution difference. Skeptical
re-review: controls answer accuracy before author cases; no thresholds changed,
no unsupported autonomous or no-worse claim, remaining budgets unchanged.

## Serialization repair after attempt158

The explicit float64 SVD passed its known-answer controls and completed all229
comparisons. Final artifact serialization failed because `SerializeTensor` was
inside an explicit GPU device block with soft placement disabled. This is a
harness/device-placement failure, not a scientific or source failure. Move only
serialization/stacking for artifact output to CPU; preserve numerical kernels,
inputs and thresholds. Persist controls and each comparison row before final
serialization so another reporting failure cannot discard the diagnostic.
Allow a third GPU attempt under the same360-second phase total, and extend the
attempt bookkeeping ceiling to160 for that retry plus the conditional CPU
reference. No new compute or changed scientific target. Focused regression
checks the serialization block and dispatcher, followed by the same controls.

The local selfbuilt TensorFlow source was located at
`/home/chakwong/tf_dual_gpu_build/tensorflow/tensorflow/compiler/tf2xla/kernels/xla_svd_op.cc`.
Its `SvdOp::Compile` uses `2*Log2Ceiling(max(m,n))` sweeps and1e-6,
which is12 sweeps for45x45. Attempt158's preserved compiler HLO independently
contains s32 constant12 and f64 constant9.9999999747524271e-7. The earlier header's
100-sweep default is not the public `tf.linalg.svd` call's actual limit. The
explicit candidate deliberately uses the header's100-sweep bound and2^-52.

## Post-run explanatory analysis

Both GPU/XLA and CPU graph candidates failed the preserved-system compatibility
screen, despite passing all seven controls. Before interpreting that failure,
calculate normal-equation backward errors from the preserved coefficient
vectors, and capture the source DGELSD singular spectrum through the already
verified independent adapter. This answers whether small linear-system
residuals can coexist with the observed polynomial discrepancy and whether
nominally identical rank cutoffs act on different tiny singular values.
`docs/benchmarks/summarize_lorenz_solver_compatibility.py` uses NumPy only for
independent reference/reporting, decodes saved TF tensors, intentionally hides
CUDA, and must reproduce all229 source coefficient vectors again. It introduces
no new candidate, no tuning and no qualification threshold. Budget120 CPU
seconds, recorded separately as post-run analysis. Output:
`solver-compatibility-backward-error-20261006-01` under the campaign root.
Skeptical check passed: backward errors and spectra are explanatory only;
small backward error is explicitly not policy/likelihood agreement.

## Closed execution

The GPU retry and CPU follow-up completed as attempts159 and160. Both return
`FORCED_BRANCH_ARITHMETIC_MISMATCH`; all229 outputs are finite and72 pass the
strict screen in each lane. All seven controls pass with explicit GPU accuracy
settings. The independent residual/spectrum audit and terminal143-hash,
17-test,4-compile review also pass. This completes the plan's diagnostic work,
not replication. See the result for candidate vetoes, limits and next action.
No job is active. Remaining charged GPU allocation702.756496seconds; CPU
151423.790188seconds. No further run is queued by this plan.
