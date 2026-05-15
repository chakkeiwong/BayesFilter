# BayesFilter V1 Nonlinear Performance NP2 Value Fast-Path Result

## Date

2026-05-16

## Governing artifacts

- Master program: `docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md`
- NP1 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md`
- NP2 plan: `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-plan-2026-05-15.md`

## Phase purpose

Record the NP2 status as a structured deferral/blocker result. This note does not implement or benchmark a value fast path; it states why production value-path optimization is not yet promotion-eligible under the master program and NP2 plan.

## Skeptical plan audit

Audit target: whether NP2 can make or promote a production value-path optimization claim from the evidence currently available.

Findings:

- Wrong-baseline risk: NP2 promotion requires NP1 value benchmark rows for the relevant shape class, but NP1 currently provides a schema artifact plus tiny CPU-only smoke rows, not a promotion-grade tiny+small value ladder.
- Proxy-metric risk: tiny smoke timings and affine parity checks would be explanatory only if reused here; they are not sufficient evidence for accepting a production fast path.
- Missing-derivation risk: the higher-risk NP2 candidates named in the plan and master program still lack the required local derivation artifact in project notation.
- XLA-overclaim risk: NP4 has not yet produced the support matrix that defines which cells are certified for XLA claims, so NP2 cannot treat current paths as XLA-certified.
- Hidden-default-change risk: any production optimization landed before the required ladder and derivation evidence could leak into later default-policy discussion without the required gate.
- Artifact-mismatch risk: a result note claiming optimization progress without code changes, focused parity tests, or benchmark artifacts would not answer the NP2 scientific or engineering question.

Audit outcome: block production NP2 optimization work for now. The correct NP2 output at this stage is a structured deferral result documenting the missing evidence and the next admissible evidence-producing steps.

## Evidence contract

Question:

- Is there enough evidence today to implement or promote a production value fast path for the nonlinear TensorFlow value filters?

Baseline:

- NP1 result rows and manifest from `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md`.
- NP2 candidate list and mathematical pre-gate from `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-plan-2026-05-15.md`.
- NP2 master-program requirements from `docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md`.

Primary criterion:

- This deferral result passes only if it correctly shows that the current record is insufficient for production NP2 promotion and names the minimal future evidence required to reopen NP2.

Veto diagnostics:

- Claiming a production optimization without a promotion-grade tiny+small value ladder.
- Claiming XLA preservation before NP4 certifies the relevant support cells.
- Treating smoke timings or affine parity as sufficient promotion evidence for Models B-C or for broad value-path performance claims.
- Treating higher-risk algebra rewrites as admissible without a derivation artifact and stated executable preconditions.

Explanatory diagnostics only:

- NP1 tiny CPU smoke value rows.
- NP1 Model A affine parity.
- NP1 schema readiness labels.

What is not concluded:

- any production value fast-path improvement;
- any code-path acceptance or rejection on runtime grounds;
- any graph or XLA support guarantee for NP2 candidates;
- any GPU statement;
- any default backend or diagnostics-level policy change;
- any exact nonlinear likelihood claim for Models B-C.

Artifact:

- `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md`

## Current blocker summary

NP2 is deferred because the present record supports benchmark-schema readiness, not production value-fast-path promotion. NP1 established that the harness can emit the metadata NP2 will need, but NP1 did not yet establish the promotion baseline required by the NP2 plan:

1. There is no promotion-grade tiny+small value ladder baseline for the relevant value paths. NP1 explicitly records a CPU-only tiny smoke artifact and states that it makes no broad performance claim.
2. NP4 has not yet produced the XLA support matrix that the master program requires as the claim boundary for supported cells.
3. The higher-risk NP2 candidates still require a derivation artifact in project notation before they can be treated as production candidates.
4. Because this worker was instructed not to edit production code or run benchmarks/tests, there is no new parity, timing, memory, or wrapper-behavior evidence that could justify a narrower production acceptance claim.

## Candidate ledger

| Candidate | Risk class | Current status | Blocking reason | What future evidence is required | Continuation label |
| --- | --- | --- | --- | --- | --- |
| tensor-only log-likelihood helper beneath existing wrappers | lower-risk | deferred | no promotion-grade NP1 tiny+small value baseline; no NP2 parity/timing artifact produced | tiny+small value ladder baseline, focused eager/graph parity, wrapper-behavior check, benchmark artifact | `NP2_CONTINUE_AFTER_VALUE_LADDER_BASELINE` |
| diagnostics-level option with full wrapper diagnostics preserved | lower-risk | deferred | no evidence that diagnostics meaning would be preserved across touched wrappers | wrapper contract test, parity artifact, tiny+small timing rows, explicit non-implication text | `NP2_CONTINUE_AFTER_WRAPPER_CONTRACT_EVIDENCE` |
| sigma-point rule precomputation at call sites | lower-risk | deferred | no value ladder baseline and no parity/timing artifact | tiny+small baseline plus parity/timing comparison for touched rows | `NP2_CONTINUE_AFTER_VALUE_LADDER_BASELINE` |
| compiled-friendly filtered-state storage | lower-risk | deferred | `return_filtered=True` behavior would need targeted parity and timing evidence not yet produced | `return_filtered=False/True` parity, timing, and memory rows on tiny+small shapes | `NP2_CONTINUE_AFTER_FILTERED_STATE_EVIDENCE` |
| block-factor alternative to full augmented eigendecomposition | higher-risk | blocked | derivation artifact absent; no executable preconditions recorded | derivation note in project notation, stated assumptions/preconditions, parity tests, ladder benchmark rows | `NP2_REPAIR_DERIVATION_REQUIRED` |
| narrower solves instead of full innovation precision | higher-risk | blocked | derivation artifact absent; equality conditions not recorded | derivation note, executable assertions/preconditions, parity tests, ladder benchmark rows | `NP2_REPAIR_DERIVATION_REQUIRED` |
| covariance update algebra rewrite | higher-risk | blocked | explicit mathematical pre-gate not satisfied | derivation note proving equivalence under implemented covariance law, executable assertions, parity tests, ladder benchmark rows | `NP2_REPAIR_COVARIANCE_DERIVATION_REQUIRED` |
| `tf.while_loop` or `tf.scan` trace-structure rewrite | higher-risk | blocked | no derivation/preservation argument and no XLA support boundary from NP4 | trace-structure proof obligation or justification, parity evidence, NP4 support-cell certification, ladder benchmark rows | `NP2_REPAIR_TRACE_AND_XLA_GATE_REQUIRED` |

## Required future evidence before NP2 can reopen

### Minimum engineering evidence

- Promotion-grade NP1-style value benchmark baseline with at least tiny and small rows for the relevant value paths, using matched shape/mode/device metadata and explicit `return_filtered` status.
- Focused parity evidence against the current implementation for every touched value path.
- Wrapper-behavior evidence showing that diagnostics and high-level API meaning remain intact where required.

### Minimum mathematical evidence for higher-risk candidates

- A derivation artifact in project notation for any algebraic rewrite listed as higher-risk in the NP2 plan or master program.
- The derivation must state the original expression, proposed expression, assumptions on symmetry/flooring/positive definiteness/solve exactness, and the executable assertions or branch preconditions needed for validity.

### Minimum compiled-mode evidence

- NP4 support-matrix certification for any cell on which NP2 wants to make an XLA-preservation or XLA-ready claim.
- Until NP4 exists, XLA-related statements remain blocked or non-claims.

### Minimum performance evidence

- Tiny improvements alone remain explanatory only.
- Promotion requires at least one predeclared shape class with admissible steady-state improvement and no material worsening on required rows under the NP2 plan.

## Commands run

No shell commands were run for NP2. This worker only read governing artifacts and wrote this deferral result note.

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
| Output artifact | `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md` |
| Governing plan | `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-plan-2026-05-15.md` |
| Governing result | `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md` |
| Derivation artifact | `not yet produced; required for higher-risk candidates before reopening NP2` |

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Defer production NP2 value fast-path work | passed for deferral scope: this note identifies why current evidence is insufficient for promotion | vetoes remain active: no promotion-grade tiny+small value ladder, no NP4 XLA certification boundary, no derivation artifacts for higher-risk rewrites | whether any lower-risk candidate survives parity and tiny+small ladder evidence once the missing baselines and gates are produced | produce the missing tiny+small value ladder baseline, then reopen only the lowest-risk candidate first; keep higher-risk candidates blocked until derivation artifacts exist | no production speedup claim, no code-acceptance claim, no XLA/GPU/default-policy claim, no exact Model B-C likelihood claim |

## Non-implication text

This NP2 result does not imply that the value-path optimization ideas are wrong, only that they are not yet admissible as production claims under the current evidence contract. It also does not imply that current value paths lack optimization headroom. The result means only that NP2 must wait for the missing baseline ladder, derivation obligations, and XLA support boundary before any production optimization can be accepted or benchmark results interpreted as promotion evidence.

## Continuation labels

- `NP2_DEFERRED_BASELINE_NOT_PROMOTION_GRADE`
- `NP2_DEFERRED_XLA_MATRIX_NOT_CERTIFIED`
- `NP2_DEFERRED_HIGHER_RISK_DERIVATIONS_MISSING`

## Files changed

- `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md`

## Phase exit label

`NP2_DEFERRED_BLOCKER_RESULT_WRITTEN`
