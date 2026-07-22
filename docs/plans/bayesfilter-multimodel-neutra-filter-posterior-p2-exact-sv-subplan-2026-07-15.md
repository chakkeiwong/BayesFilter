# P2 Subplan: Exact Transformed SV

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `P2_TARGET_REPAIR_EXECUTING_AFTER_P1`

## Phase Objective

First repair and freeze complete posterior contracts for `SVX-SGQF` and
`SVX-ZC`, which both enter P2 as `TARGET_BLOCKED`. Only a cell that obtains a
repository-issued P1 typed identity may proceed independently through
likelihood/value/score admission, same-target plain HMC, a target-specific
training protocol and fresh 5,000-step GPU/XLA training, and NeuTra
confirmation.

## Inherited Entry Conditions

- P0 attempt 04 is valid but issued no target signature. Both P2 rows are
  `TARGET_BLOCKED`; their inventory `scope_identity` values are ineligible.
- P1 shared typed-identity, recomposition, batching, training, artifact, HMC,
  archive, state, seed, GPU/XLA, and memory-growth guards pass at
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p1/attempt-05-20260715T095202Z/`.
- Exact transformed SV is not KSC SV. No KSC mixture likelihood, training, or
  sampler evidence can satisfy an entry condition here.
- Zhao-Cui paper/math and author-source anchors for `SVX-ZC` are available.

## Target And Cell Scope

- `SVX-SGQF`: exact transformed SV with the fixed SGQF direct-likelihood route
  currently anchored near `exact_transformed_sv_independent_panel_fixed_sgqf_filter`
  and its score.
- `SVX-ZC`: the same exact transformed model with the fixed Zhao-Cui TT/SIRT
  route near `exact_transformed_sv_independent_panel_zhaocui_tt_filter` and its
  score, classified operation by operation under source governance.

The two filters induce two separate posterior signatures. Their evidence may
share data and priors but may not share comparator draws or transports.

## Required Artifacts

- A P2 target-repair ledger per cell freezing serious observation tensors and
  hashes, prior, unconstraining chart and full Jacobian, fixed filter settings,
  dtype, exact batched adapter route, independent recomposition functions and
  points, and repository-issued typed identity. Until this exists, that cell
  cannot execute R1-R4.
- Per-cell target replay and filter-route manifests.
- Exact/dense transformed-SV reference, value/score, branch/status, batch, XLA,
  and focused score-check artifacts.
- Zhao-Cui source-anchor/classification supplement.
- Per-cell tuned plain-HMC result and warm-up/retained archives.
- Per-cell training-protocol ledger, recipe-screen artifacts, selected fresh
  5,000-step training state, frozen transport, and GPU manifest.
- Per-cell fresh NeuTra tuning, warm-up/retained archives, convergence/health,
  comparator-agreement result, repair records, and final state transition.
- P2 phase result/run manifest and refreshed P3 subplan.

## Required Checks And Reviews

Execute R0-R4 of the runbook independently for each cell. Before R2:

0. Re-enter at R0/P0 target freeze. Close every P0 blocker and issue a typed
   identity using `bayesfilter.inference.neutra_campaign`; never promote the P0
   inventory scope identity. `SVX-ZC` must additionally replace or explicitly
   block its current extension/invention wrapper with a production-admissible
   paper/source-anchored fixed route.

1. Prove the transformed observation convention and Jacobian term match the
   registered posterior.
2. Check value and score against the exact/dense reference over the frozen
   parameter region, not only one center point; classify approximation gaps by
   their predeclared veto/explanatory role.
3. Require batch singleton/parity/permutation, deterministic fixed-design replay,
   finite/status, same-branch score FD or reviewed derivative reference, and
   trusted GPU/XLA checks.
4. For `SVX-ZC`, verify cited paper and author-code anchors and fail with
   `BLOCK_SOURCE_UNGROUNDED` if source-faithfulness language is unsupported.
5. Before plain HMC, pass R1B with an independent total posterior value/score
   recomposition including prior, transformed likelihood, and complete chart
   Jacobian plus wrong-substitution negative tests.
6. Use the exact same cell adapter for plain HMC and NeuTra HMC.
7. Apply target-specific capacity/optimizer/hyperparameter screens. A recipe
   transferred from LGSSM or the other SV cell is a warm-start arm only.
   Record every tried, rejected, selected, and untried candidate family.
8. Use fresh screen, final-training, plain-HMC, tuning, warm-up, and retained
   seeds with P0-frozen separation.

## Evidence Contract

| Field | P2 contract |
| --- | --- |
| Question | Can target-specific NeuTra sample each exact-transformed-SV filter posterior consistently with same-target plain HMC? |
| Same-target comparator | Tuned plain HMC bound to the identical SGQF or Zhao-Cui target signature |
| Filter comparator | Exact/dense transformed-SV likelihood/value/score reference in the P0-frozen region |
| Primary pass | Each cell separately passes filter admission, final GPU training validity, modern HMC diagnostics, health, and simultaneous comparator agreement |
| Hard vetoes | KSC substitution; missing Jacobian; source-ungrounded Zhao-Cui route; target/artifact mismatch; filter gate failure; nonfinite/status/health failure; R-hat/ESS/cap failure; comparator disagreement |
| Explanatory only | Loss, acceptance, runtime, truth distance, descriptive posterior/filter gaps |
| Not concluded | Exactness of SGQF/Zhao-Cui, ranking between filters, broad SV calibration/robustness, universal recipe, production readiness |

## Default And Assumption Audit

Audit per cell: transformed data offset/convention, Jacobian, prior/chart,
initial law, time/panel dimensions, SGQF sparse level/design, Zhao-Cui ranks,
bases, training/tuning schedule, fixed randomness, parameter region, affine
preconditioner, architecture/capacity, optimizer grid, batch size, heldout data,
seeds, and comparator margins. Existing repository settings are hypotheses until
the target-specific filter and recipe screens support them.

## Repair Triggers

- Value/score/branch failure: repair the affected filter route and replay R1.
- Zhao-Cui source mismatch: correct implementation/classification or block only
  `SVX-ZC`; do not relabel an invention faithful.
- Training failure: distinguish numerical/harness failure from valid recipe
  rejection; repair mechanics or mark the attempted family rejected.
- HMC failure: localize plain target, kernel, transport, or diagnostics; use only
  predeclared tuning/training repairs.
- Artifact/report defect: reconstruct only from preserved raw evidence or rerun
  the invalid rung in a fresh root.

## Forbidden Claims And Actions

- No KSC evidence in exact-SV admission.
- No ranking SGQF versus Zhao-Cui from descriptive metrics or individual passes.
- No tuning threshold/architecture expansion after viewing confirmation data.
- No NumPy, host callback, Python sample loop, CPU serious training, warm-up
  pooling, artifact overwrite, or source-faithfulness self-attestation.

## Handoff Conditions

P3 begins after both cells have an honest terminal phase state, all admitted
states bind their artifacts, P2 checks/result/manifest are complete, failures
are cell-local or repaired, remaining budget is recorded, and P3 is refreshed
with the actual KSC-only target and no exact-SV evidence substitution.

## Stop Conditions

Block only the affected cell for invalid target/reference, missing Zhao-Cui
source, unrepaired filter implementation, absent same-target comparator, three
identical failed repairs, or per-cell budget exhaustion. Stop P2/program-wide
only if the shared harness is invalid or evidence is corrupted across cells.

## Compute And Attempt Budget

Aggregate ceiling: 80 trusted GPU wall-hours plus 16 CPU reference hours. Each
cell reserves two 15-GPU-hour family arms: plain dense IAF and one P0-frozen
target-specific enhanced family. Each arm permits one screen, one selected fresh
5,000-step training, one NeuTra confirmation, and arm-local retries. A separate
6-hour bucket funds plain HMC and comparator retries; 4 hours fund trusted
R0/R1/R1B cell admission, cell-specific adapter/artifact emission, and their
repairs. Common harness/schema/reporting defects reopen and charge P1 only.
At most three localized repairs apply per identical failure within the owning
bucket. P1 refresh must set command-level timeouts before launch.

## Skeptical Pre-Execution Audit

P2 is executable only at its target-repair rung. It uses exact/dense evidence
for filter validity and same-target HMC for sampler validity, prevents KSC
substitution, and treats source faithfulness and target-specific defaults
explicitly. Serious R1-R4 commands, margins, and recipe budgets remain forbidden
until each repaired cell target ledger freezes them before result inspection.
