# Rejected dense precision: rejection is the required behavior

Owner decision, September 22: when the existing conditioning gate rejects a
fit, report the failure and prevent use of its geometry. Numerical equality of
the discarded diagnostic precision is not an admission criterion. This
supersedes the historical tolerance proposal below; no tolerance allowance is
being installed and a high-precision GPU reference for that discarded matrix
is no longer required for this execution repair.

The existing original, graph and XLA routes already reject the D3 fixture at
the fit stage: rank 2 rather than 3, infinite design condition, accepted=false,
status `curvature_fit_rejected`, and null public precision/covariance/factor.
The original record retains the separately approved deficient-rank condition
normalization; its raw retained-subspace condition is about 1.17e8.
The implementation gate and its 1e6 design-condition limit remain unchanged.
The engineering question is whether every route reports that same rejection
and prevents the failed fit from reaching consensus, audit, factorization or
proposal evaluation. Accepted and well-conditioned computations keep their
existing numerical checks.

Pre-execution review: requiring 1e-10 entrywise agreement for an unusable
precision matrix tests an unstable diagnostic, not the rejection behavior.
Conversely, simply skipping the failing assertion could hide an accepted bad
fit, a changed rank/condition decision, lost records, or a downstream use.
The replacement must check exact rejection status, rank and failure stage,
null public geometry, unchanged callback/row accounting and all remaining
record fields. It may omit value comparison only for
`diagnostics.replicates[0].precision_z` in this rejected fixture, while retaining
its complete diagnostic record and schema. Mutation tests must reject accepted
or well-conditioned records, exposed geometry, changed rank/status/accounting,
and missing or malformed diagnostic matrices. A poisoned-fit execution check
must confirm that discarded values cannot advance any later phase.

Use the original 3582b4ac record on identical inputs within each CPU/GPU run.
No equality is claimed between different CPU/GPU-generated design bytes.
Run the registered deficient-design and new rejection-contract checks on CPU
and an available GPU, then the complete posterior extras (including the newly
resolved case) and accepted original D3 records. Use the existing campaign
driver, one worker, verified memory growth, unique numbered output directories,
120/300-second job limits and remaining 32 CPU / 52 GPU-hour cumulative caps.
Runtime/tests/driver stay frozen during every worker. Any changed acceptance,
failure stage, usable output, callback count or unrelated numerical field
fails this criterion. No public XLA switch or terminal completion follows from
this bounded test repair. Record the raw matrix difference as explanatory
evidence rather than a promotion veto; preserve all previous failures.

## Validation and decision

The rejection criterion passes on CPU and GPU2. The complete extras group now
includes the formerly excluded ill-conditioned case. No runtime implementation,
conditioning threshold, accepted-result tolerance, or numerical-loop/NumPy
allow-list entry changed.

| Run | Check | Result | Process seconds |
| --- | --- | --- | ---: |
| 02621 | Focused CPU ill-conditioned fixture, graph and XLA | 1 passed | 10.241 |
| 02622 | Complete CPU posterior extras | 58 passed | 46.665 |
| 02623 | Complete GPU posterior extras | 58 passed | 97.574 |
| 02624 | Original D3 complete CPU records | 17 passed | 91.565 |
| 02625 | Original D3 complete GPU records | 17 passed | 169.896 |
| 02626 | Campaign, source-policy and GPU-selection checks | 102 passed | 8.735 |

All cases passed without failures, errors or skips. The 150 posterior checks
cover independent Gaussian geometry, all three deficient designs, 38 mutations
of the rejection boundary on each device, callback failures, ownership and
complete accepted/rejected original D3 records. The additional focused CPU
check repeats the critical fixture. Six poisoned-fit executions per device
(graph/XLA with precision scales 1, 1e200 and NaN) retain exactly 21 callback
batches and stop at the fit stage. No consensus, audit, reconstruction,
normalization or proposal completion is reported; all usable geometry stays
null. This checks execution isolation, not the numerical correctness of an
ill-conditioned discarded fit.

The original, graph and XLA records all report `curvature_fit_rejected`, rank 2,
and no usable geometry. The actual records still preserve the discarded matrix.
Its maximum absolute differences from the original are 2.912e-9 / 2.130e-9 for
CPU graph/XLA and 4.120e-9 / 7.178e-10 for GPU graph/XLA. All four still fail the
historical entrywise comparison. These are explanatory differences, not a claim
that the discarded precision is accurate or equivalent. Every other field
retains its original comparison, with numerical atol=rtol=1e-10 and exact
discrete decisions/accounting. The zero/rank-one fixtures keep full numerical
comparison, including their discarded matrices.

All six workers used an identical recorded source closure, with one worker at a
time and no runtime/test/driver edits during execution. CPU workers hid GPUs.
GPU workers selected non-desktop GPU2,
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`, and verified memory growth before
device initialization. The manifests record TensorFlow 2.19.1, TF32, environment,
exact commands, source hashes and bounded wall times. Graph execution is an
explicit reference arm. Focused Ruff and whitespace checks pass.

The shared campaign root contains `posterior-rejection-qualification-02626.json`,
its reproducible summary script, and every numbered manifest/log/full record.
Historical failures 02378, 02415 and 02456 remain preserved. Cumulative charges
through 02626 are 52,437.52665588881 CPU and 49,589.058448168376 GPU seconds,
leaving 17.434 CPU and 38.225 GPU process-hours under the unchanged caps.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Resolve this rejected-matrix comparison blocker | Rejection, reporting, no usable outputs and no downstream execution pass on CPU/GPU | No changed accepted output, failure stage, accounting or unrelated field | Coverage is the declared rejected fixture and controlled poison cases | Keep these checks in full qualification; retire the uninstalled allowance/reference requirement | No general waiver for ill-conditioned accepted results or discarded-fit accuracy claim |
| Keep the wider repair open | Focused posterior and policy checks pass | Public enclosure, costs, actual consumers and terminal evidence remain open | Complete production call chains and representative workloads | Continue the master program before integration and merge | No complete repair, posterior/HMC readiness or performance claim |

Post-run primary-agent review: merely suppressing the numerical assertion would
hide a behavior change. The strict rejection preconditions, mutation tests,
poisoned executions and unchanged accepted-record comparisons address that risk.
A changed rejection decision or any usable/downstream geometry would overturn
this resolution. Small fixtures do not establish behavior for every consumer,
and the public posterior endpoint remains unwired to the native controller.
No independent reviewer was used for this bounded test-only criterion change.

## Historical proposal — superseded, never installed

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
