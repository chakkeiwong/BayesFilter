# BayesFilter V1 Nonlinear Performance NP3 Score Fast-Path Result

## Date

2026-05-16

## Governing artifacts

- Master program: `docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md`
- NP1 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md`
- NP2 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md`
- NP3 plan: `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-plan-2026-05-15.md`

## Phase purpose

Record NP3 as a structured deferral/blocker result. This note does not implement or benchmark a score fast path; it states why production score-path optimization is currently blocked or deferred under the master program, the NP1 baseline/result, the NP2 result, and the NP3 plan.

## Skeptical plan audit

Audit target: whether the current record is sufficient to implement or promote a production analytic-score optimization without weakening branch certification, fixed-support structure, or derivative semantics.

Findings:

- Wrong-baseline risk: NP3 promotion requires NP1 score benchmark rows for the relevant shape class, but NP1 currently provides only tiny CPU-only smoke rows whose score parity status is explicitly `measured_against_branch_precheck_only`, not finite-difference/reference-certified score evidence.
- Proxy-metric risk: branch-precheck success and tiny score timing rows are explanatory only. Using them as promotion evidence would silently upgrade branch certification into derivative-correctness certification.
- Proof-obligation risk: the NP3 plan requires a named preserved derivative expression, tensor-shape conventions, and a proof artifact for any parameter-axis vectorization or algebraic reuse candidate. No such proof artifact exists yet.
- Branch-contract risk: branch assertions cannot be treated as removable overhead. The NP3 veto contract requires that branch assertions either remain or be moved behind an explicit precheck artifact; current evidence does not justify changing their execution placement in production code.
- Model C structural-risk: the NP3 entry gate and veto rules require preserving the structural fixed-support branch for Model C where applicable. The current record contains no new evidence that a score fast path would preserve that structure.
- Hessian/HMC overclaim risk: score-path parity alone cannot be upgraded into Hessian readiness, HMC readiness, or sampler correctness. The master program and NP3 plan both forbid those implications.
- Artifact-mismatch risk: because this worker was instructed not to edit production code or run benchmarks/tests, there is no new parity, proof, timing, or structural-preservation artifact that could support even a narrow production acceptance claim.

Audit outcome: block production NP3 optimization work for now. The correct NP3 output at this stage is a structured deferral/blocker result documenting the missing proof and evidence obligations and the next admissible steps.

## Evidence contract

Question:

- Is there enough evidence today to implement or promote a production analytic score fast path for the nonlinear TensorFlow score filters?

Baseline:

- NP1 result rows and manifest from `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md`.
- NP2 deferral result from `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md`.
- NP3 candidate list, mathematical pre-gate, and veto conditions from `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-plan-2026-05-15.md`.
- NP3 master-program requirements from `docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md`.

Primary criterion:

- This deferral result passes only if it correctly shows that the current record is insufficient for production NP3 promotion and names the minimal future proof and evidence required to reopen NP3.

Veto diagnostics:

- Claiming a production score optimization from NP1 tiny CPU smoke rows alone.
- Treating branch-precheck success or branch-linked timing as sufficient score-correctness evidence.
- Accepting parameter-axis vectorization without a named preserved derivative expression, tensor-shape conventions, and a proof-obligation artifact.
- Weakening or relocating branch assertions without an explicit certified precheck artifact.
- Weakening Model C structural fixed-support behavior.
- Implying Hessian readiness, HMC readiness, or sampler validity from score-path work.

Explanatory diagnostics only:

- NP1 tiny CPU smoke score timing rows.
- NP1 branch precheck rows.
- NP1 schema readiness labels.
- NP2 deferral labels showing that earlier value-path work is also not promotion-grade.

What is not concluded:

- any production score fast-path improvement;
- any code-path acceptance or rejection on runtime grounds;
- any preservation of derivative semantics for vectorized parameter-axis updates;
- any graph or XLA support guarantee for NP3 candidates;
- any GPU statement;
- any default score backend policy change;
- any Hessian readiness claim;
- any HMC readiness or convergence claim.

Artifact:

- `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-2026-05-16.md`

## Current blocker summary

NP3 is blocked or deferred because the current record supports schema and branch-precheck readiness, not production score-fast-path promotion.

1. NP1 provides only tiny CPU smoke score rows, and those rows explicitly carry branch-precheck linkage rather than finite-difference/reference-certified score correctness evidence.
2. NP1 states that score timing rows are measured against branch precheck only, so current parity evidence is branch-contract evidence, not derivative-proof evidence.
3. The NP3 plan requires a proof obligation before parameter-axis vectorization, eigensystem reuse, or other derivative-structure-changing optimizations can be accepted. No such proof artifact has been produced.
4. Branch assertions and Model C structural fixed-support evidence must be preserved; the present record does not justify weakening, removing, or silently moving those contracts in production code.
5. Because this worker was instructed not to edit production code or run benchmarks/tests, there is no new parity, proof, timing, or structural artifact that could justify a narrower production acceptance claim.

## Candidate ledger

| Candidate | Risk class | Current status | Blocking reason | Required future proof/evidence | Continuation/repair label |
| --- | --- | --- | --- | --- | --- |
| vectorize parameter-axis loop for Kalman-gain derivatives | higher-risk | blocked | parameter-axis vectorization lacks the required proof obligation; NP1 score evidence is branch-precheck only | proof artifact naming the preserved derivative expression, tensor-shape conventions, equality proof between scalar-loop and batched form, then focused score parity plus finite-difference/reference evidence | `NP3_REPAIR_VECTORIZATION_PROOF_REQUIRED` |
| reuse eigensystem solves where algebraically identical | higher-risk | blocked | algebraic identity and derivative preservation conditions not recorded; no proof artifact | derivation/proof note stating when the eigensystem reuse is valid and how derivative semantics are preserved, plus focused score parity evidence | `NP3_REPAIR_EIGENSYSTEM_IDENTITY_PROOF_REQUIRED` |
| score-only fast path for callers not needing full diagnostics | lower-risk but blocked | deferred | branch assertions and branch provenance cannot be dropped; current evidence does not show diagnostics/branch contract preservation | explicit branch-precheck contract, score parity evidence, and API/diagnostic preservation evidence for touched cells | `NP3_CONTINUE_AFTER_BRANCH_CONTRACT_EVIDENCE` |
| separate branch precheck from steady-state timing while recording branch artifact per row | lower-risk for benchmarking, not yet a production optimization claim | deferred | NP1 proves schema linkage only; production execution placement still lacks preservation evidence | artifact showing explicit precheck lineage plus proof that production branch-certification behavior is preserved for affected cells | `NP3_CONTINUE_AFTER_PRECHECK_LINEAGE_EVIDENCE` |
| any Model C score fast path touching structural fixed-support branches | higher-risk | blocked | Model C structural fixed-support preservation evidence absent | explicit structural fixed-support preservation proof/evidence for Model C rows, plus score parity/reference evidence under those branches | `NP3_REPAIR_MODEL_C_FIXED_SUPPORT_EVIDENCE_REQUIRED` |

## Required future proof and evidence before NP3 can reopen

### Minimum mathematical proof obligation

- A proof artifact in project notation for every candidate that changes derivative structure, including parameter-axis vectorization and algebraic solve reuse.
- The artifact must:
  - name the original derivative expression and the proposed preserved expression;
  - state tensor shapes and parameter-axis conventions;
  - state assumptions or executable branch preconditions;
  - prove equality between the old scalar-parameter loop and the proposed batched/vectorized form where applicable.

### Minimum score-correctness evidence

- Focused score parity against the current implementation for every touched backend/model cell.
- Finite-difference or reference score checks for the affected cells. Branch-precheck success alone is not sufficient.
- Value parity where a score path also returns value information.

### Minimum branch-contract evidence

- Explicit evidence that branch assertions are preserved or moved only behind an explicit certified precheck artifact.
- Explicit evidence that Model C structural fixed-support behavior is preserved where applicable.
- Branch-precheck lineage recorded for any timing artifact that separates precheck from steady-state timing.

### Minimum compiled-mode/XLA evidence

- NP4 support-matrix certification for any cell on which NP3 wants to make graph/XLA preservation or XLA-ready claims.
- Until NP4 exists, XLA-related statements remain blocked or non-claims.

### Minimum performance evidence

- Tiny improvements alone remain explanatory only.
- Promotion requires at least one predeclared shape class with admissible steady-state improvement and no material worsening on required rows under the NP3 plan.

## Commands run

No shell commands were run for NP3. This worker only read governing artifacts and wrote this deferral/blocker result note.

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
| Dtype | `N/A` |
| Model/backend/path/shape/horizon/parameter dimension/point count | `N/A` |
| Random seeds | `N/A` |
| Wall time | `N/A` |
| Output artifact | `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-2026-05-16.md` |
| Governing plan | `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-plan-2026-05-15.md` |
| Governing result | `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-2026-05-16.md` |
| Derivation/proof-obligation artifact | `not yet produced; required before reopening NP3 candidates that change derivative structure` |

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Defer/block production NP3 score fast-path work | passed for deferral scope: this note identifies why current evidence is insufficient for promotion | vetoes remain active: NP1 score rows are tiny CPU smoke only, parity is branch-precheck only, proof obligations are missing, and branch/fixed-support preservation is not yet re-established for any candidate | whether any candidate survives proof, branch-contract preservation, and score-correctness checks once the missing artifacts exist | write the required proof-obligation artifact for the narrowest candidate first, then collect focused score parity/reference evidence before any production optimization claim | no production speedup claim, no derivative-structure acceptance claim, no XLA/GPU/default-policy claim, no Hessian claim, no HMC readiness claim |

## Non-implication text

This NP3 result does not imply that score-path optimization is impossible, only that it is not yet admissible as a production claim under the current evidence contract. It does not imply that branch assertions are unnecessary overhead or that current score code lacks optimization headroom. It means only that NP3 must wait for explicit proof obligations, focused score-correctness evidence, and preserved branch/fixed-support contracts before any production optimization can be accepted. It also does not imply Hessian readiness, HMC readiness, sampler correctness, or convergence evidence.

## Continuation/repair labels

- `NP3_DEFERRED_BASELINE_ONLY_TINY_CPU_SMOKE`
- `NP3_DEFERRED_SCORE_PARITY_IS_BRANCH_PRECHECK_ONLY`
- `NP3_REPAIR_VECTORIZATION_PROOF_REQUIRED`
- `NP3_REPAIR_BRANCH_ASSERTION_PRESERVATION_REQUIRED`
- `NP3_REPAIR_MODEL_C_FIXED_SUPPORT_EVIDENCE_REQUIRED`
- `NP3_DEFERRED_HESSIAN_AND_HMC_CLAIMS_FORBIDDEN`

## Files changed

- `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-2026-05-16.md`

## Phase exit label

`NP3_DEFERRED_BLOCKER_RESULT_WRITTEN`
