# BayesFilter V1 Nonlinear Performance NP6 Reference Path Decision

## Date

2026-05-16

## Governing artifacts

- Master program: `docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md`
- NP6 plan: `docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-plan-2026-05-15.md`
- NP0 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np0-inventory-result-2026-05-16.md`
- NP1 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md`
- NP2 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md`
- NP3 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-2026-05-16.md`
- NP4 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-2026-05-16.md`
- NP5 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-2026-05-16.md`

## Phase purpose

Decide, from the current NP0-NP5 record only, whether either NumPy reference path — `StructuralSVDSigmaPointFilter.filter` or `particle_filter_log_likelihood` — should be rewritten into TensorFlow now, while preserving the distinction between reference-only NumPy semantics and the existing production TensorFlow nonlinear paths.

## Skeptical plan audit

Audit target: whether NP6 can justify a TensorFlow rewrite for a NumPy reference path without silently converting narrow production-TensorFlow timing/support evidence into a broader reference-path, default-policy, or GPU claim.

Findings:

- Wrong-baseline risk: NP0 explicitly classifies `StructuralSVDSigmaPointFilter.filter` and `particle_filter_log_likelihood` as eager NumPy reference paths, not production TensorFlow surfaces, so NP6 must start from reference-only status rather than from the NP4/NP5 production TF results.
- Scope-leak risk: NP4 certifies only fixed-static-shape CPU XLA support for specific production TensorFlow value cells, and NP5 times only a narrow trusted Model B value ladder on those same production TF backends. Neither artifact evaluates a NumPy reference path or establishes a missing downstream use case for one.
- Proxy-metric risk: curiosity about possible TensorFlow speedups for the reference paths would be explanatory only. Under the NP6 plan, performance curiosity alone is a veto because a rewrite is allowed only if a concrete downstream need is unmet by existing production TensorFlow paths.
- Semantics-replacement risk: rewriting a reference path into TensorFlow now could blur the repository’s current separation between production TF nonlinear paths and reference NumPy semantics/oracles, especially because NP0 records the reference paths as non-compiled, non-GPU, non-default candidates absent a separate NP6 justification.
- Particle-filter-method risk: `particle_filter_log_likelihood` is a stochastic bootstrap particle reference path with resampling/ESS semantics. NP6 plan vetoes any rewrite proposal that lacks a separate evidence contract for randomness, resampling policy, and what correctness or reproducibility would mean after a TensorFlow port.
- Structural-mismatch risk: `StructuralSVDSigmaPointFilter.filter` is itself an eager NumPy sigma-point approximation, while the repo already contains production TensorFlow sigma-point value paths with NP4-certified narrow XLA support and NP5 narrow CPU/GPU timing evidence. NP6 therefore must ask whether the reference path provides a distinct downstream need, not whether TensorFlow sigma-point code exists in general.
- Hidden-default-change risk: initiating a rewrite now would invite later conflation with NP7 default-policy discussion even though NP2/NP3 already defer production fast-path promotion and NP4/NP5 preserve narrow non-claim boundaries.
- Artifact-mismatch risk: this bounded worker was instructed not to edit production code, tests, or benchmarks, and no new runtime evidence was collected. The only admissible NP6 output is a decision artifact grounded in NP0-NP5 evidence and explicit non-implications.

Audit outcome: pass for a decision-only NP6 artifact. The current record is sufficient to decide that no TensorFlow rewrite is justified now, and that any future rewrite must reopen under a new plan with a separate evidence contract.

## Evidence contract

Question:

- Is a TensorFlow rewrite justified now for either `StructuralSVDSigmaPointFilter.filter` or `particle_filter_log_likelihood`?

Baseline:

- NP0 inventory/result classification of production TensorFlow surfaces versus eager NumPy reference paths.
- NP1 benchmark-harness schema/result showing what later timing artifacts can and cannot certify.
- NP2 and NP3 deferral/blocker results showing that even production TensorFlow optimization claims remain narrowly gated.
- NP4 CPU-only XLA support matrix for specific production TensorFlow value cells.
- NP5 trusted CPU/GPU timing rows for the narrow NP4-supported Model B production TensorFlow value cells.

Primary criterion:

- A rewrite is admissible only if the NP0-NP5 record shows a concrete downstream need for compiled or GPU behavior that is not already served by the current production TensorFlow nonlinear paths.

Veto diagnostics:

- proposing a rewrite only because TensorFlow or GPU performance might be interesting;
- using NP4 or NP5 production-TensorFlow evidence as if it were evidence about NumPy reference paths;
- replacing a reference/oracle path with an unvalidated fast path;
- proposing a TensorFlow particle filter without a separate evidence contract for randomness, resampling policy, reproducibility, and correctness;
- collapsing the distinction between reference-only NumPy status and production TensorFlow nonlinear paths;
- implying a rewrite/default/GPU claim from the absence of a benchmark row rather than from a concrete unmet use case.

Explanatory diagnostics only:

- current eager NumPy status of the reference paths;
- existing narrow production TensorFlow XLA or CPU/GPU timing evidence;
- possible future convenience from unifying implementations.

What is not concluded:

- that either reference method is mathematically unnecessary;
- that a future TensorFlow rewrite would be impossible;
- that current production TensorFlow paths dominate every reference use case;
- that particle filtering should never exist in TensorFlow here;
- any production default change;
- any GPU readiness or default-GPU claim for a rewritten reference path;
- any HMC, Hessian, or sampler-readiness claim.

Artifact:

- `docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-2026-05-16.md`

## Current evidence summary from NP0-NP5

### NP0 classification boundary

NP0 classifies the two reference candidates as follows:

- `StructuralSVDSigmaPointFilter.filter`: eager NumPy reference value path, unsupported for graph/XLA/GPU, with explicit non-claim that it is not a production TensorFlow backend or default candidate without a separate NP6 justification.
- `particle_filter_log_likelihood`: eager NumPy bootstrap particle reference value path, unsupported for graph/XLA/GPU, stochastic, and likewise not a production TensorFlow/default candidate without a separate NP6 justification.

NP0 also records that the repository already has distinct production TensorFlow nonlinear value and score paths, and that helper/reference paths should not be promoted by implication.

### NP1-NP3 gating boundary

- NP1 upgrades the benchmark harness schema but states that its tiny CPU-only smoke rows are not promotion-grade performance or correctness evidence.
- NP2 defers production TensorFlow value fast-path work because even those production paths lack the promotion-grade ladder, derivation, and XLA-boundary evidence needed for optimization claims.
- NP3 similarly defers production TensorFlow score fast-path work because proof obligations, branch-preservation evidence, and score-correctness evidence remain missing.

These phases tighten, rather than relax, the evidentiary bar. If production TensorFlow optimizations are still deferred, NP6 cannot justify a reference-path rewrite from weaker evidence.

### NP4-NP5 narrow production-TensorFlow boundary

- NP4 certifies only fixed-static-shape CPU XLA support for six production TensorFlow value cells: Model B/C, cubature/UKF/CUT4, `return_filtered=False/True`. It explicitly retains non-claims for dynamic horizon, GPU XLA, and all score cells.
- NP5 benchmarks only trusted matched CPU/GPU timing for static Model B production TensorFlow value cells at `timesteps=3`, `dtype=tf.float64`, `return_filtered=False`, and explicitly says the results do not justify broad GPU/default/score/HMC claims.

Therefore NP4/NP5 answer only a narrow question about existing production TensorFlow value paths. They do not show that the NumPy reference paths block compiled execution or GPU use for a concrete downstream workflow.

## Candidate-by-candidate decision

### 1. `StructuralSVDSigmaPointFilter.filter`

Current decision: do not rewrite into TensorFlow now.

Reasoning from the record:

1. NP0 classifies this path as a NumPy reference sigma-point implementation kept for semantics/reference use, not as a production TensorFlow backend awaiting optimization.
2. The repository already contains production TensorFlow sigma-point value surfaces (`tf_svd_sigma_point_filter`, `tf_svd_sigma_point_log_likelihood`, `tf_svd_cut4_filter`, related low-level variants) that cover the TensorFlow nonlinear value role.
3. NP4 shows that a narrow subset of those production TensorFlow value cells already compiles under CPU XLA, and NP5 shows narrow exact-cell CPU/GPU timing behavior for the production Model B value paths.
4. Nothing in NP0-NP5 identifies a distinct downstream consumer that specifically requires the semantics of `StructuralSVDSigmaPointFilter.filter` but also requires TensorFlow compilation or GPU execution unavailable from the production TF nonlinear paths.
5. Rewriting now would risk duplicating or blurring an already-populated production TF sigma-point surface rather than addressing a demonstrated gap.

Decision status: deferred with no current rewrite plan.

### 2. `particle_filter_log_likelihood`

Current decision: do not rewrite into TensorFlow now.

Reasoning from the record:

1. NP0 classifies this path as an eager NumPy bootstrap particle reference path with Monte Carlo/resampling semantics and explicit non-claims for TensorFlow/graph/XLA/GPU readiness.
2. NP4 and NP5 contain no evidence about particle filtering at all; their scope is limited to deterministic production TensorFlow sigma-point value cells.
3. Under the NP6 plan, a TensorFlow particle rewrite is not admissible without a separate evidence contract covering randomness, resampling policy, ESS/resample diagnostics, reproducibility, correctness targets, and what role the port would play relative to existing reference semantics.
4. No such contract or unmet downstream need appears in NP0-NP5.
5. A TensorFlow particle implementation would be a new method-development effort, not a straightforward continuation of the current production TensorFlow sigma-point support evidence.

Decision status: blocked from rewrite under the current record; reopen only under a separate future plan.

## Decision

The current NP0-NP5 evidence does **not** justify rewriting either `StructuralSVDSigmaPointFilter.filter` or `particle_filter_log_likelihood` into TensorFlow now.

The repository should keep both paths reference-only NumPy implementations for the current program phase. Existing production TensorFlow nonlinear paths remain the only admissible basis for NP4/NP5 compiled and CPU/GPU statements, and those statements stay narrow. No current evidence supports a TensorFlow rewrite, a production default change, or a GPU claim for either reference path.

## Future reopen conditions

Reopen `StructuralSVDSigmaPointFilter.filter` only if all of the following become true:

1. A concrete downstream workflow is named that specifically needs the reference-path semantics in TensorFlow and cannot be met by existing production TensorFlow nonlinear value paths.
2. The required mismatch is stated precisely: value contract, diagnostics contract, shape regime, model family, and why the production TF path is insufficient.
3. A separate plan defines parity targets against the current NumPy reference path, including what constitutes acceptable agreement and what would remain a non-claim.
4. The plan states whether the rewrite would remain reference-only, become an optional backend, or seek eventual production status, with the corresponding evidence bar.

Reopen `particle_filter_log_likelihood` only if all of the following become true:

1. A concrete downstream workflow is named that needs a TensorFlow particle method rather than the existing NumPy reference path or existing production TF sigma-point methods.
2. A separate evidence contract specifies particle count regime, resampling policy, RNG/reproducibility policy, ESS/resample diagnostics, and the exact correctness/promotional targets.
3. The plan distinguishes method-development questions from implementation-performance questions.
4. The plan explicitly states what will not be concluded from any initial TensorFlow particle benchmark, especially around GPU/default/HMC claims.

## Continuation / repair labels

- Continuation label: `NP6_CONTINUE_WITH_REFERENCE_ONLY_NUMPY_STATUS`
- Continuation label: `NP6_CONTINUE_USING_NP4_NP5_ONLY_FOR_PRODUCTION_TF_VALUE_CELLS`
- Repair label: `NP6_REOPEN_ONLY_IF_CONCRETE_UNMET_DOWNSTREAM_NEED_IS_STATED`
- Repair label: `NP6_PARTICLE_FILTER_REQUIRES_SEPARATE_RANDOMNESS_AND_RESAMPLING_CONTRACT`
- Repair label: `NP6_STRUCTURAL_SIGMA_POINT_REQUIRES_DISTINCT_SEMANTIC_GAP_EVIDENCE`

## Non-implications

This NP6 decision does not say that the NumPy reference paths are poor methods, only that the current program record does not justify porting them into TensorFlow now. It does not imply that future TensorFlow particle filtering is forbidden, nor that `StructuralSVDSigmaPointFilter.filter` could never warrant a TensorFlow port. It does mean that the present evidence is insufficient for a rewrite, insufficient for any default-policy change, and insufficient for any GPU claim beyond the narrow production TensorFlow value cells already bounded by NP4 and NP5.

## Run manifest

| Field | Value |
| --- | --- |
| Git commit | `81c647b157612a233dde1a9fbd847647a5b03b8f` |
| Dirty worktree | `true` |
| Command | `N/A` |
| Environment | `N/A` |
| Python | `N/A` |
| TensorFlow | `N/A` |
| TensorFlow Probability | `N/A` |
| CPU/GPU status | `no runtime command executed` |
| GPU intentionally hidden | `N/A` |
| Device visibility | `N/A` |
| Random seeds | `N/A` |
| Wall time | `N/A` |
| Files read | `docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-plan-2026-05-15.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np0-inventory-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-2026-05-16.md` |
| Files changed | `docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-2026-05-16.md` |
| Output artifact | `docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-2026-05-16.md` |
| Governing plan | `docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-plan-2026-05-15.md` |
| Governing result | `docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-2026-05-16.md` |

## Commands run

No shell commands were run for NP6. This bounded worker only read governing artifacts and wrote this decision note.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep both NumPy reference paths reference-only for now; do not rewrite either into TensorFlow in this phase | passed: NP0-NP5 identify no concrete unmet downstream compiled/GPU need for either reference path | vetoes remain active and respected: no performance-curiosity rewrite, no production-TF evidence reused as reference-path evidence, no particle rewrite without separate randomness/resampling contract, no collapse of reference-vs-production distinction | whether a future user workflow will expose a genuine semantic gap not covered by the existing production TensorFlow nonlinear paths | continue to NP7 while preserving the reference-only NumPy status; reopen only through a new plan with a concrete unmet-use-case contract | no TensorFlow rewrite justification, no default-policy change, no GPU claim for reference paths, no particle-filter correctness or sampler-readiness claim |

## Phase exit label

`NP6_REFERENCE_PATH_DECISION_WRITTEN_NO_TF_REWRITE_JUSTIFIED_NOW`
