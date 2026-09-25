# C2 Phase 8 Recovery Integration Handoff

Date: 2026-09-07

Recovery branch: `codex/c2-phase8-recovery-writable-20260906`

Phase 8D code-close commit: `27c91fd7`

Phase 8E readiness commits: `54602019`, `b5c3b99c`

Clean integration validation branch: `codex/c2-root-integration-20260907`

Clean integration validation head: `1b8c1a1f`

Canonical worktree: `/home/chakwong/BayesFilter` on
`ledh-refactor-with-policy-fix` at `36f3d3b0`

## Decision

The recovered Phase 8D repair and Phase 8E readiness unit are valid on their
declared recovery branch.  The clean canonical-base integration branch now
executes the same call chain and passes its focused contracts, but the
canonical worktree is not yet an executable copy.  The root worktree was not
changed by this recovery and remains dirty from unrelated LEDH work.  A
conflict-reviewed integration patch is required before another claim-bearing
C2 run.

## Integration audit

| Component | Canonical root state | Recovery state | Consequence |
| --- | --- | --- | --- |
| `bayesfilter/highdim/bases.py` | committed file has no `HermiteBasis1D`; import fails | contains `RealLine`, `GaussianReferenceMeasure`, normalized `HermiteBasis1D`, and recurrences | apply the focused addition; do not replace the file wholesale |
| `bayesfilter/highdim/c2_exact_likelihood_laplace_adapter.py` | absent; import fails | restored and repaired | restore before invoking the exact-Laplace route |
| `bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` | untracked variant lacks the base-mass-aware finite program | recovery version carries normalized base-mass fields and exact increments | use the recovery version or perform a reviewed equivalent merge |
| `docs/benchmarks/run_c2_exact_likelihood_laplace_phase8b_20260904.py` | untracked older variant | failure-aware renderer, active-time screen, and opt-in repair registry | do not use the root copy for Phase 8D/8E evidence |
| `docs/benchmarks/aggregate_c2_phase8d_replication_20260905.py` | untracked older variant | preserves failed candidates and excludes `t=0` from active contrasts | use the repaired reader |
| `docs/benchmarks/run_c2_phase8d_schedule_repair_diagnostic_20260907.py` | absent | present and recorded | restore for reproducible schedule repair diagnostics |
| Phase 8 contract tests | most recovery contract files absent in root | present and passing | restore and run before any new campaign |
| Phase 8E launcher, analyzer, boundary artifact, and calibration fixtures | absent | present and passing on clean integration | use only from the clean integrated branch |

The root-side C2 source copies that happen to have the same hash as the
recovery copies are still untracked.  They are not evidence until they are
integrated into a clean commit and the call chain is rerun.  In particular,
the following root imports currently fail:

```
from bayesfilter.highdim.bases import HermiteBasis1D
# ImportError

import bayesfilter.highdim.c2_exact_likelihood_laplace_adapter
# ModuleNotFoundError
```

The provenance search finds the Hermite class introduced by `e986d419` and no
commit deleting it.  The missing root symbols therefore reflect branch and
working-tree divergence, not a mathematical change to the C2 method.

## Safe continuation sequence

1. Preserve `/home/chakwong/BayesFilter` as-is.  The clean validation branch
   `codex/c2-root-integration-20260907` at `1b8c1a1f` is the current executable
   integration reference.  Merge it into the canonical branch only through a
   conflict-aware, human-reviewed change.
2. Merge the focused `bases.py` additions and the recovered adapter/evaluator,
   runner, aggregate reader, schedule diagnostic, and tests.  Resolve any
   conflict against current LEDH code by inspecting the resulting call chain;
   never overwrite `bases.py` wholesale.
3. On the clean integrated commit, verify imports, source hashes, `py_compile`,
   and `git diff --check`.  Run the focused C2 contracts and the bounded broad
   suite (`96 passed, 1 deselected`); the deselected degree-12 oracle must be
   separately budgeted because its 180-second CPU bound timed out.
4. Re-run one GPU/XLA Phase 8A mechanics smoke and one clean Phase 8D repair
   replay.  New manifests must carry the new commit and plan hashes; old
   recovery artifacts remain historical evidence for the recovery branch.
5. After those checks, execute the bounded Phase 8E launcher specified by
   `docs/plans/c2-phase8e-statistical-replication-20260907.md`.  It creates
   fresh output roots, generates the twelve claim fixtures, runs two paired
   branches for each, and invokes the strict analyzer.  No Phase 8E claim
   fixture has been generated or analyzed yet.

## Open engineering work

- Clean up TensorFlow schedule-kernel retracing before making a performance or
  default-readiness claim.
- Rebuild the custom operation in a toolchain containing the required CUDA and
  LLVM headers; the existing ABI-matched `tftwogpu` binary is valid for the
  recorded runs but is not a clean-build proof.
- Allocate a bounded plan for the degree-12 Gaussian oracle and either finish
  it or record a justified coverage exception.
- Implement and audit the Student-T TT reference only as a separate method;
  it is not needed to repair the observed fixed-Newton failure.

## Scientific boundary

`quarter_long_12` repaired the observed fixed-iteration stationarity failure
on four C2 paths.  The ESS contrasts are one-branch-per-path descriptive
diagnostics, not statistical superiority.  The route remains an
`extension_or_invention_candidate_diagnostic_only` path with no posterior,
general-model, HMC, production, or default-readiness claim.

## Phase 8E readiness audit

The statistical analyzer, frozen observation-bin artifact, deterministic
branch-seed checks, and bounded launcher were added in recovery commits
`54602019` and `b5c3b99c`.  The recovery readiness suite passes `35 tests` with
two TensorFlow Probability deprecation warnings.  The clean integration clone
passes the same `35 tests`, `py_compile`, `git diff --check`, imports, and
boundary recomputation.  Its final broad C2 regression passes `110 tests` with
one explicit degree-12 oracle deselection; the run is recorded in
`docs/memos/c2-phase8-recovery-integration-regression-20260907.md`.  Its first launcher contract run exposed a missing
fixture-generator support file; restoring that file produced clean integration
commit `665d085e` and a passing rerun.  The current documentation-only head is
`1b8c1a1f`.  This is an integration repair, not
evidence about the proposal's scientific performance.
