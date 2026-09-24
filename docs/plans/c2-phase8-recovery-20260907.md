# C2 Phase 8 Recovery Plan

Date: 2026-09-07  
Status: `RECOVERY_PHASE8D_REPAIR_PASS_PHASE8E_LAUNCH_READY`  
Classification: `extension_or_invention_candidate_diagnostic_only`

## Purpose

Restore the C2 mixture-UKF and exact-likelihood Laplace proposal code after the
working-tree cleanup on 2026-09-05, without changing the dirty canonical LEDH
worktree or reusing a result as evidence for code that is not present.

## Provenance finding

The missing `HermiteBasis1D` was not deleted by a commit.  The C2 source is
present in commit `665d8ce0` (`codex/c2-phase2-generic-dmis-20260902`).  The
recorded Claude session executed a checkout/removal command on the
`ledh-refactor-with-policy-fix` worktree, restoring the LEDH version of
`bases.py` and removing late untracked C2 files.  The canonical worktree is
dirty and remains out of scope for this recovery.

The writable recovery checkout is
`/tmp/BayesFilter-c2-phase8-recovery-20260906`, branch
`codex/c2-phase8-recovery-writable-20260906`, based at `665d8ce0`.

## Compatibility decision

The recovery retains the C2 base-mass-aware
`zhao_cui_frozen_proposal_apf_tf.py` from `665d8ce0`.  Later C2 proposal callers
still pass `initial_log_base_mass` and `transition_log_base_mass`; pairing them
with the later evaluator that removed those fields would silently change the
finite importance program.  The later analytical C2 state score and negative
Hessian methods are added to `c2_sv_frozen_proposal_apf_tf.py` because the
generic Laplace kernel requires them.

The initial exact-Laplace adapter was recovered byte-for-byte from the recorded
creation patch.  Its pre-repair SHA-256 is
`da8c7d26df9f09dfb185e4f8680e49dbc097ccaabf29e76d3977fe39bba57dc6`.
The first fresh Phase 8A run then exposed a call-chain schema omission: the
generic kernel returned both Newton objective traces, while the adapter copied
only `iteration_objective_before_step`.  The recovery adds the missing
`iteration_objective_after_step` row and a regression assertion; it does not
change proposal construction, the exact target, or the proposal denominator.

## Evidence contract

| Item | Declaration |
| --- | --- |
| Question | Does the recovered C2 call chain construct valid observation-informed proposals and bind them to the exact frozen APF evaluator? |
| Comparator | C2 base-mass-aware `FrozenProposalAPFProgram` with the complete proposal denominator |
| Hard validity gate | import/compile success, finite target and proposal rows, exact score finite-difference checks, density recomposition, and stated lane parity |
| Promotion veto | candidate loss to a constructed heuristic at an active transition time (`t >= 1`) or unresolved proposal-validity failure; `t=0` is explanatory only because no transition proposal has been used; no promotion is sought in this recovery |
| Continuation veto | target/measure mismatch, missing source or fixture, corrupted artifact, unavailable required GPU/build environment, or budget exhaustion |
| Explanatory diagnostics | ESS, stationarity residual, ascent trace, runtime, trace count, and source/worktree hashes |
| Nonclaims | no posterior correctness, unbiased-likelihood, statistical ranking, general-model, HMC, production, or default-readiness claim |
| Artifact | this plan, focused test captures, fresh versioned Phase 8 output roots, and a recovery commit/manifest |

## Completed checks

All checks below were run in `tftwogpu` with `CUDA_VISIBLE_DEVICES=-1` unless
otherwise stated.  These are CPU-only mechanics/reference checks; they are not
GPU performance evidence.

| Suite | Result |
| --- | ---: |
| exact Laplace kernel and C2 adapter | `11 passed` |
| UKF Phase 0 | `7 passed` |
| UKF Phase 1 | `4 passed` |
| UKF Phase 4 and repair | `11 passed` |
| UKF Phase 7 | `3 passed` |
| Hermite and Gaussian engine oracle (first three completed tests) | `3 passed; degree-12 rung timed out in wrapper and is not promoted` |
| Gaussian-Hermite proposal | `6 passed` |
| frozen C2 proposal | `8 passed` |
| transformed Student proposal | `5 passed` |
| Student-floor tests | `4 passed` (the no-fire and lane-parity tests require approximately 60--100 seconds each) |
| recursive-map/RBF/hybrid mechanics | `18 passed` |
| C2 contract and diagnostic suites | `21 passed` |
| post-repair C2 exact-Laplace adapter regression | `4 passed` |

Phase 8A recovery attempt 01 reached the GPU/XLA kernel successfully and then
failed in the reporting harness with `KeyError:
'iteration_objective_after_step'`.  This is an adapter diagnostic-schema defect,
not a Laplace-validity, target, denominator, or GPU failure.  The failed output
is retained at
`docs/benchmarks/artifacts/c2_phase8_recovery_20260907/phase8a-mechanics-attempt01/`.

After the adapter repair, Phase 8A recovery attempt 02 passed on GPU/XLA.  The
artifact is
`docs/benchmarks/artifacts/c2_phase8_recovery_20260907/phase8a-mechanics-attempt02/`.
All 15 checks passed.  Notable values were analytical-score central-difference
error `1.1320e-08`, XLA/non-XLA proposal error `1.7764e-15`, maximum relative
stationarity residual `2.3432e-16`, minimum precision eigenvalue `1.0000`, and
one trace each for the Laplace kernel and sampler.  The artifact records commit
`7c33b869cd9747192442ce6a93f5800f19b492b1` and an empty workspace status.

Phase 8B calibration-only attempt 01 also passed.  It selected the declared
`quarter_long` schedule using independent ordinary and near-zero calibration
observations; the linear-Gaussian oracle and fail-closed indefinite-curvature
fixture behaved as required.  The artifact is
`docs/benchmarks/artifacts/c2_phase8_recovery_20260907/phase8b-calibration-attempt01/`.
The first four shorter schedules were rejected by the predeclared C2
stationarity screen, while `quarter_long` passed.  TensorFlow emitted bounded
retracing warnings while five distinct schedule-specific kernels were created;
each recorded kernel had trace count one and a fixed input signature.  This is
an engineering cleanup item before any default-readiness claim, not a validity
failure for this diagnostic run.

The recovery clone does not contain the ignored Phase 7 Gaussian-hint snapshot
bank.  Rather than copy an untracked stale comparator into the recovery or
silently change its provenance, the next comparison explicitly passes
`--exclude-gaussian-hint`.  The resulting five-family set is Laplace K=1,
transformed UKF K=1, bootstrap conditional, transformed Student `nu=8`, and
stationary independence.  This is a bounded recovery comparison, not a claim
that the six-family Phase 8B result has been reproduced.

The first five-family N=256 smoke reached the GPU/XLA kernels and passed every
finite-program validity check, but its reported promotion veto was not
interpretable.  The Laplace candidate used initial random-key offset `8101`
while the direct UKF K=1 comparator used `1001`; consequently the candidate's
time-zero cloud differed even though both branches had the same seed and prior
law.  The comparator table also treated this pre-transition ESS as a
promotion-veto observation.  The smoke therefore remains a mechanics artifact
but its dominance verdict is superseded, not evidence against the candidate.

The repair changes only comparison provenance: the Laplace adapter now uses the
UKF K=1 offsets `1001`, `5100 + 41t`, and `5200 + 43t`, names these constants,
and records them in its manifest.  The runner retains the time-zero comparison
with `decision_role=explanatory_only` and applies the heuristic veto only to
`screen_candidate_loses` at `t >= 1`.  The focused regression suite passes
(`7 passed`), including initial-cloud replay and both sides of the comparator
contract.  A fresh smoke is required because the old output was generated
before this repair.

The repaired N=256 five-family smoke passed at
`docs/benchmarks/artifacts/c2_phase8_recovery_20260907/phase8b-five-family-smoke-attempt02/`.
All five branches passed the finite-program checks and the GPU/memory-policy
checks.  The Laplace and UKF initial clouds now agree exactly (both initial
ESS `58.6644158938`), and the active-time heuristic table has zero losses, so
the result is `PASS_PHASE8B_VALIDITY_DIAGNOSTIC` with no candidate failure.
The candidate's minimum ESS was `58.6644158938` and its time-14 ESS was
`247.5982237744`; these contrasts are descriptive because this is one branch
per family.  The earlier attempt remains superseded only for its dominance
verdict; its finite mechanics evidence is retained.

The repaired N=1024 five-family comparison passed at
`docs/benchmarks/artifacts/c2_phase8_recovery_20260907/phase8b-five-family-n1024-attempt01/`.
All five branches and the GPU/memory checks passed, with zero active-time
heuristic losses.  The candidate minimum ESS was `197.6481573346` versus
`55.2852470232` for UKF, `120.3141094304` for bootstrap, `183.7590988259`
for transformed Student, and `36.5390025689` for stationary independence;
these are descriptive one-branch values only.  Its manifest binds commit
`0b6393c6` and the repaired plan hash.

The custom op used for these checks is the existing `tftwogpu` build,
SHA-256 `661f11b9db1f6e9ab9ce4aae8ae86591779cdbdfe6fda777a7c87956ec2686fa`.
It is ABI-incompatible with the separate `tf-gpu` TensorFlow 2.19.1
environment; all recovery tests therefore record `tftwogpu` (TensorFlow
`2.20.0-dev0+selfbuilt`).

A fresh custom-op build was attempted with the `tftwogpu` toolchain.  The
standard CUDA headers lacked `crt/host_defines.h`; using TensorFlow's vendored
CUDA headers progressed further but the environment lacked LLVM header
`llvm/ADT/ArrayRef.h`.  The existing ABI-matched binary therefore remains a
validated local runtime dependency.  Rebuilding it from a clean toolchain is a
separate infrastructure task, not evidence about the C2 algorithm.

## Remaining actions

1. Execute the reviewed
   [`docs/plans/c2-phase8e-statistical-replication-20260907.md`](c2-phase8e-statistical-replication-20260907.md)
   from the clean integration branch
`codex/c2-root-integration-20260907` at commit `1b8c1a1f`.  The bounded
   launcher and frozen observation-bin calibration artifact are now present;
   it specifies multiple proposal branches, fixture-level paired uncertainty
   calculations, and the opt-in `quarter_long_12` schedule.  Do not rank
   methods from the current one-branch-per-path ESS table.
2. Integrate the tested recovery unit into the canonical worktree only through a
   conflict-aware, human-reviewed change.  The canonical branch is dirty and
   its committed `bayesfilter/highdim/bases.py` does not contain
   `HermiteBasis1D`; the recovery branch supplies `RealLine`,
   `GaussianReferenceMeasure`, `HermiteBasis1D`, and their helper recurrences.
   Do not replace the canonical file wholesale: preserve unrelated LEDH work
   and apply the focused basis addition after the owner selects the target
   integration point.
3. Keep TensorFlow retracing cleanup and a clean custom-op rebuild as separate
   engineering tasks; neither changes the completed validity verdict.
4. Only after Phase 8E and a terminal call-chain/math audit should a human
   decide whether the candidate route can be promoted beyond diagnostic status.

## Regression closure and integration boundary

On 2026-09-07 the recovery clone ran the complete C2 test glob with the
degree-12 Gaussian oracle rung explicitly deselected:

```
96 passed, 1 deselected, 2 warnings in 166.92s
```

The deselected test, `test_oracle_gate_degree12_rank6_n2_t12`, was run alone
under the same CPU-only environment with a 180-second bound and timed out
without an assertion or finite result.  It is an expensive oracle-coverage
gap, not evidence of a C2 proposal failure.  `py_compile` over the C2 source,
drivers, and tests and `git diff --check` both pass.

The root worktree `/home/chakwong/BayesFilter` was not modified during this
recovery.  Its branch `ledh-refactor-with-policy-fix` remains dirty from
pre-existing LEDH work and its committed basis file lacks `HermiteBasis1D`;
the Phase 8D code-close commit is `27c91fd7`, the Phase 8E readiness commits
are `54602019` and `b5c3b99c`, and the current clean recovery head is
`b5c3b99c`.  The provenance search
finds the Hermite class being introduced in `e986d419` and no deletion commit.
This separation is deliberate: merging the recovery commits or applying the
focused basis patch is the remaining integration decision, not an automatic
cleanup operation.  The file-by-file state and safe merge sequence are in
[`docs/memos/c2-phase8-recovery-integration-handoff-20260907.md`](../memos/c2-phase8-recovery-integration-handoff-20260907.md).

The clean canonical-base integration branch
`codex/c2-root-integration-20260907` now contains the recovery unit plus the
Phase 8E launcher at `1b8c1a1f`.  It passes the integrated focused suite
(`35 passed, 2 warnings`), imports `HermiteBasis1D`, the exact-Laplace adapter,
and the base-mass-aware evaluator, and passes launcher help, compilation, and
boundary provenance checks.  This branch is validation evidence only; it has
not been merged into the dirty canonical worktree.

The final clean-integration C2 regression was rerun after the Phase 8E
hardening and support-file repair at commit `2157d088`:
`110 passed, 1 deselected, 2 warnings in 174.13s`.  The command and exclusion
are recorded in
`docs/memos/c2-phase8-recovery-integration-regression-20260907.md`.

## Phase 8D repair close

The original `quarter_long` fresh path failed at seed `424245` because two
rows at `t=5` exceeded the fixed stationarity tolerance; this was a candidate
schedule failure, not repository damage.  The renderer and aggregate-reader
repair preserved that failure as evidence.  The predeclared schedule
diagnostic then nominated `quarter_long_12`, and independent calibration
selected it on four clean replays: the confirming seed `424245` path plus
`(424246,20260909)`, `(424247,20260910)`, and `(424248,20260911)`.  Every
replay passed the exact target/complete denominator, analytical score,
APF-identity, parity, finite-program, and GPU/memory checks; all `76/76`
active-time contrasts were positive against each declared comparator.  The
full decision table and nonclaims are in
`docs/plans/c2-phase8d-repair-close-20260907.md`.

## Skeptical pre-run audit

The initial audit did not pass for the five-family smoke: the time-zero
random-stream mismatch and the all-time veto rule were material comparability
defects.  After repair, the bounded rerun passes the skeptical audit because
the baseline, finite target, random-key pairing, active-time screen,
branch/version identity, ignored-fixture treatment, and candidate-versus-
continuation failure distinction are explicit and tested.  A Phase 8D result
with one branch per seed remains descriptive and cannot establish a ranking.
The repaired N=256 and N=1024 comparisons passed this audit under the unchanged
target, budget, and nonclaim boundary.  The dedicated Phase 8D plan is now the
next controlled step; its fresh-path results must be audited before any
statistical or default-readiness interpretation.
