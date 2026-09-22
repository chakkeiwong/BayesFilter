# C2 Phase 8 Root Restoration

Date: 2026-09-07

Branch: `ledh-refactor-with-policy-fix`

Restoration commit: `21d5870f8320ba03ef017315ef3c62638e789fef`

## Outcome

The C2 Phase 8 integration call chain is executable again on the root branch.
The restoration adds the focused `HermiteBasis1D` implementation, the
base-mass-aware frozen APF evaluator, the exact-likelihood Laplace adapter and
proposal kernels, the Phase 8 drivers and fixtures, and their validated tests.
It does not alter or stage the pre-existing work outside the restoration
scope, including LEDH/RQMC changes and additional C2 diagnostic files.

The earlier
[`c2-phase8-recovery-integration-handoff-20260907.md`](c2-phase8-recovery-integration-handoff-20260907.md)
remains the historical record of the pre-restoration state.  Its statements
that the root imports fail and that the root worktree has not been changed are
superseded by this memo only for the state after commit `21d5870f`.

## Provenance

Repository history contains the introduction of `HermiteBasis1D` in
`e986d419` and no commit that deletes the class.  The recorded Claude cleanup
session restored the older branch copy of `bayesfilter/highdim/bases.py` and
removed untracked late C2 files.  The failure was therefore a working-tree and
branch-integration loss, not a mathematical or intentional source change.

The restored 66-file unit was compared against the clean integration reference
at `codex/c2-root-integration-20260907` and committed without staging the
pre-existing out-of-scope changes.

## Verification

| Check | Result |
| --- | --- |
| Root imports | `HermiteBasis1D`, exact-Laplace adapter, and base-mass APF fields present |
| Focused Phase 8 contracts | `36 passed, 2 warnings in 4.56s` |
| Broad C2 regression | `110 passed, 1 deselected, 2 warnings in 175.29s` |
| Patch integrity | staged 66-file set matched the clean integration reference; `git diff --check` passed |
| Root GPU/XLA mechanics smoke | `PASS_PHASE8A_GPU_XLA_MECHANICS`, all 15 checks passed |

The root GPU/XLA smoke ran on an NVIDIA GeForce RTX 4080 SUPER with TensorFlow
memory growth verified before device initialization.  It used XLA, produced
finite exact values and analytical scores, matched the central finite
difference score to `1.3641e-08`, matched XLA and non-XLA values exactly, and
had maximum proposal-density discrepancy `1.77636e-15`.  Its result is at
[`docs/benchmarks/artifacts/c2_root_restore_phase8a_20260907/attempt01/result.md`](../benchmarks/artifacts/c2_root_restore_phase8a_20260907/attempt01/result.md).

The explicitly deselected test is
`test_oracle_gate_degree12_rank6_n2_t12`.  Its prior 180-second CPU timeout is
an expensive Gaussian-oracle coverage gap, not evidence of a C2 proposal
failure.

## Remaining Work

No further source repair is required to recover this C2 Phase 8 call chain.
Before a new research-decision campaign, execution must use a clean worktree at
a committed integration point because the root worktree still contains
unstaged out-of-scope changes and the Phase 8E launcher fails closed on a dirty
tree.

The next scientific action is the bounded Phase 8E statistical replication in
[`docs/plans/c2-phase8e-statistical-replication-20260907.md`](../plans/c2-phase8e-statistical-replication-20260907.md).
Following the recovery handoff literally also calls for one fresh Phase 8D
repair replay at the committed integration point before Phase 8E.  Neither run
has been launched from this root commit by this restoration task.

TensorFlow schedule-kernel retracing cleanup, a clean custom-operation rebuild,
and the degree-12 oracle are separate engineering or coverage items.  They do
not invalidate the restored call chain, but they remain blockers for the
corresponding performance, clean-build, or complete-oracle claims.

## Scientific Boundary

The tests and Phase 8A smoke establish import, wiring, finite-program,
analytical-score, GPU-placement, and XLA-parity mechanics.  They do not
establish proposal efficiency, posterior correctness, statistical superiority,
general-model validity, HMC readiness, production readiness, or default
readiness.
