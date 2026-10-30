# Stable principal-angle repair result

The shared precision-comparison kernel now computes principal angles using
overlap cosines and orthogonal-residual sines with `atan2`. This repairs loss of
accuracy from `acos` near one. Rank selection, the existing1e-12 unit-cosine
snap, padded shape, stability caps and other metrics are unchanged. The added
residual SVD uses the existing converged TensorFlow/XLA authority. No NumPy,
Python numerical loop, pfor or new policy exception enters runtime.

Baseline bf022dd90. Nine serialized workers04914--04922 pass, using166.662239
CPU /187.758706 GPU process-seconds. Independent candidate qualification and
readback precede runtime adoption. Installed tests pass49 checks per backend;
complete saved-fit replays pass on both; final source-bound readback/policy
passes161 checks; frozen diagnostic reconstruction and final combined readback04922
pass163 checks. The315-source/1457-exception guard is unchanged. GPU2 UUID
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba` uses trusted execution, float64,
TF32 off and verified memory growth; CPU is the explicit reference lane.

| Saved23-dimensional/rank7 fixture | CPU maximum degree error | GPU maximum degree error |
|---|---:|---:|
| Original acos formula | 3.11828e-9; fails strict bound | 1.38168e-10; fails strict bound |
| Stable formula | 4.19332e-12; passes | 3.83671e-12; passes |

The authority uses80-digit eigensystems and overlap SVD from the exact stored
binary64 matrices, preserving the original zero snap. Relative selected
eigengaps0.00197178/0.00199577 exclude repeated-eigenvalue ambiguity for this
fixture. Other metrics are exactly equal in the candidate comparison. Analytic
rotations from zero to90 degrees, snap cases, rank changes, full-rank spaces,
replay, one trace and HLO checks pass. Installed XLA and non-JIT reference
angles both meet the unchanged1e-10+1e-10*abs(angle) degree bound.

Exact saved fitted inputs are replayed in04919 CPU/04920 GPU. Relative to the
same-backend04614/04613 records, six CPU and three GPU strict differing leaves
are all angle diagnostics. There are zero non-angle differences under the
original complete-record comparison: matrices, statuses, selection, audit,
optimizer counters, thresholds and decisions retain their gates. Selected
geometry remains `consensus_diagonal_consensus`. The inherited payload status
is preserved; this execution repair does not confer HMC or scientific admission.

Every reported stability pair is also checked against80-digit arithmetic using
its actual fitted matrices. Maximum errors are3.96217e-12/3.66441e-12 degrees
for dense CPU/GPU and3.48166e-13/2.91287e-14 for factor2. Thus the remaining
factor2 cross-backend angle differences describe different fitted matrices;
they are not a remaining angle-evaluation error.

The full CPU/GPU record comparison now has4534 differing leaves, versus4539
previously. The five eliminated leaves were dense angle-report differences.
Remaining:57 selection/stability leaves and4477 leaves in unselected fits3/4/5
(1663/1150/1664). Eight angle leaves belong to the differing factor2 matrices;
their largest cross-backend difference is24.0053 degrees. No full-record
equivalence claim follows. Fits3/4 reject on holdout; fit5 is unselected and
unconverged. CPU diagnostics record200 iterations without convergence for fits3/5
and optimizer failure at iteration7 for fit4. Those outcomes need their own
matrix/optimizer/status investigation, not another small-angle arithmetic trial.

An isotropic identity-matrix control has zero selected eigengap when rank cuts
through repeated eigenvalues. Its chosen subspace is mathematically non-unique.
This limitation is reported; no new runtime eigengap rejection or arbitrary
subspace convention is silently installed. Isotropic reporting remains open.

Descriptive cost evidence: the same-process candidate screens use ten warm
calls after compilation. CPU mean warm time is11.946→13.713ms; GPU is
98.171→133.761ms. These sequential, retained-owner observations are not fresh
matched process comparisons, and their RSS includes both compiled owners.
They cannot establish a memory regression magnitude or terminal cost acceptance.
The added SVD and observed GPU warm increase require matched attribution.
Full saved-fit elapsed times are47.056s CPU /65.704s GPU, including compilation.
The repaired GPU allocator peak is2313216 bytes versus2306304 in the earlier
saved replay; differing historical environment/device scope prevents a clean
cost ranking. No native-executable eviction is claimed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Keep shared angle repair | Independent strict accuracy and installed/full-fit checks pass | Rank/snap/status/selection gates retained | Untested pathological eigenspaces | Commit scoped repair | Universal eigenspace identifiability |
| Preserve differing fits | Each backend's reported angle is independently accurate |4534 full-record leaves remain open | Unselected optimizer/matrix differences | Localize fits3/4/5 and isotropic reporting | Full CPU/GPU equivalence |
| Retain cost trigger | Extra SVD and descriptive warm increase recorded | No terminal acceptance | Fresh-process costs/native residency | Matched attribution | Performance ranking or main merge |

| Inference status | Result |
|---|---|
| Hard veto screen | Numerical and same-backend full-fit gates pass |
| Statistically supported ranking | None |
| Descriptive-only differences | Additional SVD; higher observed warm cost |
| Default-readiness | Scoped shared angle accuracy qualifies; broader gates open |
| Next evidence needed | Unselected-fit/isotropic diagnosis, matched costs, source dispositions |

Evidence: `run-04922/installed-angle-terminal.json`, candidate readback04916,
and exact commands/environment/source/fixture/operand hashes in raw run files.
Archive `principal-angle-stable-04922-evidence.tar.gz` contains53 verified
members,2970736 bytes, SHA-256
`0b6128152adaaf379affabc317aba840329b777cb7f7b6f80a2ef2c37c23c765`. The earlier04921 archive remains preserved.
The raw root remains artifacts/filter-gradient-repair-20260917. No failed
worker, numerical gate relaxation or new global compute allocation occurred.

Primary-agent review: the strongest competing explanation is eigenvector
inaccuracy or a non-unique selected subspace. Clear measured eigengaps and
independent full-pair references distinguish these from the repaired acos
conditioning error. Comparison against an old inaccurate angle is preserved
as a documented comparator defect, never silently relabeled equivalence.
The weakest evidence is terminal cost/memory and general isotropic reporting;
both remain open. No independent agent was used. iAPF and KDM stay deferred.

Final harness review04922 verifies that the diagnostic remains reproducible
after runtime adoption: Git-pinned original source and intervention hashes
match both saved probes, and a known-geometry execution matches the installed
and baseline programs. This check repairs diagnostic source reconstruction,
not runtime numerics.
