# Zhao-Cui Austria SIR Fixed-Variant Parameter Extension Reset Memo

Date: 2026-07-30

Current status: `LANE_A_EXHAUSTED_OWNER_DECISION_REQUIRED_FOR_NEW_BASELINE`

Canonical plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`.

## Correct Baseline

The active baseline is the exact P86/P88 training-base fixed-variant fit
artifact, starting with:

`docs/plans/bayesfilter-highdim-zhao-cui-p88-phase2-degree-order3-rank4-lr3e-4-l1-0-fit-2026-06-27.json`.

Required SHA-256:
`ea5fc7434f328b95e3c2c53bca3e1a7bee6b35a452a81acce8230407ea11ef8e`.
This is the strongest serialized fit artifact, not proof of a complete filter,
correct value, T2/T20 execution, or observation binding. The fit is explicitly
`time_index=1`; the prior P90 T2 bridge is a separate deterministic fixture.

P76/P77 UKF is a separate one-step warm-start experiment, not a proved part of
the P88 T1 artifact or a proved end-to-end P88 retained-object pipeline. Do not
insert or remove UKF while claiming unchanged baseline behavior.

## Phase 0 Result

Phase 0 executed. It established:

1. all 36 serialized core hashes and shapes pass;
2. the exact order-3/eight-element Lagrange basis, reference measure, rank
   tuple, `tau=1e-8`, and floors reconstruct;
3. reconstructed square normalizer `4.544027196172014e-06` and full normalizer
   `4.554027196172014e-06` exactly match P88;
4. the reconstructed density branch hash matches P88; and
5. P88 does not bind the affine frame arrays, transport CDF configuration,
   frozen reference samples, retained branch identity, observation hash, or
   source dependency closure required to prove the same T1 retained branch and
   T2 previous-marginal boundary.

The matching P86 predecessor was also checked. It contains the same density
branch/core identity and the same omissions, so it does not repair the blocker.

Classify inherited target assembly, squared-TT marginalization, conditional KR
and normalizer accumulation against the cited paper/author source. Classify the
P88 training-base optimizer and the new conditioned parameter child as
`extension_or_invention`. Do not use source anchors to replace the local
trainer with the author estimator.

Therefore Phase 0 stopped with
`BLOCK_FIXED_VARIANT_BASELINE_NOT_RECONSTRUCTIBLE`. Density reconstruction is a
pass; complete fixed-variant filter reconstruction is blocked. Phase 1 is not
authorized.

Terminal artifact:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-result-attempt02-2026-07-30.json`.

Terminal result:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-phase0-result-2026-07-30.md`.

## Proposed Recovery Plan

The proposed two-lane recovery plan is:

`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-baseline-recovery-plan-2026-07-30.md`.

Lane A performs a bounded exact historical-artifact recovery audit. Lane B
constructs a newly named, fully serialized training-base fixed variant only
after an explicit owner decision accepting that the target is no longer exact
P88 reconstruction. The proposal does not authorize Lane B, training, GPU,
parameter extension, score work, or HMC.

Lane A has now executed and stopped with
`BLOCK_EXACT_P88_RECOVERY_EXHAUSTED`. The terminal result is:

`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-baseline-recovery-result-2026-07-30.md`.

No local workspace, Git, snapshot, worktree, ignored artifact, log, or
checkpoint candidate bound the missing exact identity fields. The next action
is the owner decision whether to approve Lane B under the new identity
`zhao_cui_austria_sir_fixed_variant_training_base_v1`.

## Parameter Direction After Blocker Repair

Theta will be an external conditioning input. Add three TT conditioning cores,
evaluate them at theta, and integrate only state cores. If Lane A recovers the
exact P88 baseline, the origin slice must preserve its P88 state cores,
`tau=1e-8`, defensive reference measure, and fixed value. If the owner instead
approves Lane B, the origin slice must preserve the newly admitted fixed
baseline and must not claim P88 identity. Extend the TT amplitude `phi`, not
the square root of the full `phi^2 + tau*lambda` density. The manual total
score must carry
`D_theta R_t` through the previous retained
marginal and every applicable normalizer/conditional/transport owner. A local
transition-plus-observation score is wrong for this target.

## Forbidden Next Steps

- no original-author TT-cross/ALS reconstruction;
- no July 30 source-replica or frozen-proposal APF continuation;
- no generic retained-grid fallback;
- no local complete-data score promoted as the observed-data score;
- no HMC.

Historical July 30 candidate failures do not block this baseline extension.

## Reopen Condition

Reopen Phase 0 only if one of these occurs:

1. recover a historical artifact that binds the exact P88 `mu`, frame matrix,
   CDF configuration, frozen reference samples, retained identity, and source
   dependency closure; or
2. obtain an explicit owner decision defining a new fixed-variant baseline and
   accepting that it is not exact P88 reconstruction.

Do not recompute the frame with the current unbound code and call it P88. Do not
borrow diagnostic `grid_size=5`/`bisection_steps=4` settings, insert UKF,
synthesize T2 cores, or return to author TT-cross/ALS.
