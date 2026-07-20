# Reset Memo: LEDH Score Wiring Repair

Date: 2026-07-10

Last updated: 2026-07-11

## Current State

The active program remains:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-master-program-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-visible-gated-execution-runbook-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-visible-execution-ledger-2026-07-10.md`

Current status:
`PHASE9_FD_POLICY_CORRECTED_PARTIAL_CONTINUATION_REVIEW_REQUIRED_LGSSM_NOT_RUN`.

Phases 0 through 8 completed their scoped wiring, precision, provenance, and
cross-model test work. Phase 9 Gate A built and validated the shared trusted
GPU/XLA score/FD harness. Its original absolute-or-relative FD gate was later
audited and found unsupported: the `0.005` values were inherited CLI defaults,
not calibrated production or HMC thresholds. All hard-veto and
candidate-rejection statements based on that policy are superseded; the raw
trusted GPU/XLA measurements remain valid.

The current owner-directed rule applies only to the finite-difference
diagnostic. For each stored direction,
`r_j=abs(score_j-FD_j)/max(abs(score_j),abs(FD_j),1e-12)`. A comparison passes
iff `max_j(r_j) <= 0.05*sqrt(p)`. There is no RSS/RMS aggregation or
absolute-error escape branch. The `5%` constant mirrors the conventional 95%
threshold, but this calculation is not itself a calibrated confidence interval.

Offline reclassification covered all 11 completed live Gate B/Gate C
comparisons. Nine pass and two fail:

- predator-prey fails Gate B `T=1,N=2`: maximum direction error `1.0` versus
  threshold `0.122474487139`;
- generalized-SV passes Gate B but fails Gate C `T=4,N=10000`: `0.442753962161`
  versus `0.0866025403784`;
- fixed-SIR passes its historical terminal Gate C `T=20,N=10000` comparison:
  `0.0566700085587 <= 0.0866025403784`;
- Actual-SV (`p=2`) passes Gate C `T=4,N=10000`:
  `0.0602924688125 <= 0.0707106781187`;
- KSC-SV passes Gate C `T=4,N=10000`:
  `0.0369351492982 <= 0.0707106781187`.

No shared harness, source binding, or original-artifact integrity veto fired.
No nonlinear Gate D or aggregate artifact was created, and the separate LGSSM
lane was not run. Fixed-SIR, Actual-SV, and KSC-SV have this FD veto removed,
but require a new reviewed v3 continuation manifest before GPU execution.

Consolidated artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-result-2026-07-11.md`;
- `docs/plans/artifacts/ledh-score-wiring-repair-phase9-fd-policy-correction/phase9-fd-policy-reclassification-2026-07-11.json`,
  with all 11 source pairs SHA-bound;
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-codex-review-2026-07-11.md`,
  which replaces the earlier review of the wrong `2%` RSS/RMS policy.

Final authority:

- correction result SHA-256:
  `12108b9dc32283c2a42ddcb72937a87853ba381cd97416a53d718e52a327bbaf`;
- reclassification JSON SHA-256:
  `1ffa3fd9fdf74050d667b4205c8545e56657f0102b81fb28933894bd3644a4dd`;
- correction review SHA-256:
  `5c3e983baea6d283fa9e8b1590ff5e518f4715f7958b9a881f60339ba24bfaee`;
- corrected review verdict: `VERDICT: AGREE`.

The prior consolidated result
`docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-result-2026-07-10.md`
is historical only and now carries a supersession notice.

## Historical State At Prior Reset

### Phase 4: Predator-Prey

Status at prior reset: `PASSED`

Changed files:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase4-predator-prey-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-subplan-2026-07-10.md`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase4-result-phase5-subplan-review-bundle-2026-07-10.md`

What changed:

- Predator-prey score defaults now use `float32` and TF32 enabled.
- `_coordinate_fd_score_diagnostic` now uses
  `_compact_value_and_score_from_components` as the score base.
- Finite differences use a value-only same-scalar objective.
- Score artifacts include explicit `score_precision`.
- Full-admission artifact construction rejects nested historical/manual
  relabeling and tiny-shape promotion.
- Historical reverse/manual score routes remain diagnostic-only.

Checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py
```

Passed.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Result: `70 passed, 2 warnings`.

Review:

- Claude review gate was attempted and rejected by execution policy as external
  repository data disclosure. No workaround was attempted.
- Fresh Codex substitute read-only review returned `VERDICT: AGREE`.

### Phase 5: Actual-SV

Status at prior reset: `LOCAL_CHECKS_PASSED_REVIEW_PENDING`

Changed files:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-subplan-2026-07-10.md`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase5-result-phase6-subplan-review-bundle-2026-07-10.md`

What changed:

- Actual-SV score defaults now use `float32` and TF32 enabled.
- `_coordinate_fd_score_diagnostic` now uses
  `_compact_value_and_score_from_components` as the score base.
- Finite differences use a value-only same-scalar objective.
- Score artifacts include explicit `score_precision`.
- Full-admission artifact construction rejects nested historical/manual
  relabeling and tiny-shape promotion.
- Historical reverse/manual score routes remain diagnostic-only.
- The transformed actual-SV target policy
  `transformed_actual_sv_log_y_square` is preserved.
- `claims_exact_native_actual_sv_likelihood` remains false.

Checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py
```

Passed.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Result: `70 passed, 2 warnings`.

Review status at prior reset:

- Claude review gate for Phase 5 result plus Phase 6 subplan was attempted and
  rejected by execution policy as external repository data disclosure.
- No substitute review had been completed yet for Phase 5. That next step was
  later completed and is superseded by the current Phase 9 state above.

## Boundaries Recorded At Prior Reset

- CPU-hidden local tests are wiring evidence only.
- At that prior reset, no trusted GPU `N=10000` score-memory run had been
  performed.
- No full score admission was claimed for predator-prey or actual-SV.
- No leaderboard rebuild was performed.
- No HMC readiness, posterior correctness, exact native likelihood, or
  scientific-superiority claim was made.
- Claude calls are currently policy-blocked as external data disclosure. Do not
  attempt workarounds. Use a fresh Codex substitute read-only review unless
  policy changes.

## Exact Next Step

Do not run any command from the frozen Phase 9 exact-command manifest. It uses
superseded v1 output paths and predates the corrected v3 runner schema.

The next permissible action is to write and review one of these artifacts:

1. A narrow continuation subplan plus a new exact-command manifest using new
   paths and the v3 runner. It may resume fixed-SIR at Gate D and Actual-SV/KSC-SV
   at Gate C `T=50`, preserving all original shapes, seeds, transport, precision,
   memory, and row-local stop rules.
2. A diagnostic subplan for predator-prey and generalized-SV that preserves
   each row's scalar and predeclares precision/step arms capable of
   distinguishing compact-score error from float32 FD resolution.
3. An explicit closeout/leaderboard subplan that records only the two supported
   FD vetoes and does not promote the three passing rows without their remaining
   ladder evidence.

Phase 10 has no scoped subplan in the workspace and is not authorized by the
negative Phase 9 result.

## Important Boundaries

- Candidate rejection for predator-prey/generalized-SV is not rejection of the
  shared harness or compact-score research direction.
- The exact derivative cause is unsupported and not checked; the two FD
  mismatches establish only FD diagnostic failures at their measured rungs.
- The `5%` constant mirrors the conventional 95% threshold, but no confidence
  interval or coverage calibration was computed.
- Prefix SV memory peaks are not full-time memory evidence.
- No score default-readiness, HMC readiness, posterior correctness, native
  actual-SV correctness for KSC, runtime superiority, or statistical ranking is
  established.
- Preserve unrelated dirty work, especially
  `bayesfilter/linear/kalman_qr_tf.py` and
  `docs/plans/bayesfilter-post-integration-reboot-reset-memo-2026-07-10.md`.
