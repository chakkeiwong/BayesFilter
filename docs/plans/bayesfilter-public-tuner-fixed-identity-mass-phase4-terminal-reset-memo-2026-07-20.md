# Phase 4 Public-Tuner Replay-Gate Terminal Reset Memo

Snapshot time: `2026-07-20T06:09:31+08:00`

Status: `PHASE4_CLOSED_TUNING_REPLAY_HASH_MISMATCH`

Read this memo first after a reboot. It supersedes the active-run state in
`bayesfilter-public-tuner-fixed-identity-mass-phase4-reboot-reset-memo-2026-07-19.md`.
It does not supersede the master plan or preserved artifacts.

## Terminal Verdict

Attempt 03 completed in the trusted GPU/XLA context and stopped correctly at
`TUNING_REPLAY_HASH_MISMATCH`. No process remains to resume. Do not relaunch
Attempt 02 or Attempt 03, and do not start another sampling seed under the
current Phase 4 budget.

Expected public final-kernel hash:
`e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc`.

Observed public final-kernel hash:
`e1d61cd46e9e65cd510bad14669619c7c9854348bee6a3a659e065a26a4ce0b6`.

The replay gate recorded `sequential_sampling_authorized=false`. No warm-up,
retained sample, convergence, ESS, or truth-tail artifact exists for Attempt
03. The second sampling seed was never run.

## Important Interpretation

The tuner passed fresh fixed-kernel verification, and its mechanics-visible
fields match Attempt 01 exactly:

- step size `0.7779889586003162`;
- six leapfrog steps;
- selected-step hash
  `c39e59a4ec867b98594e40b2a1551fbe92eabf4f05b4826e0a0e8bdd1631a9ec`;
- fixed identity mass signature
  `25eb272b3f8b1e742173a12ea1ae6a07ba8a203dfdba3e6f67deebc30a7598fe`;
- verification acceptance `0.7109046801795739`.

The top-level public hash differs because its payload also includes run-specific
intermediate artifact hashes and the nested Phase 7 hash, while the public
payload intentionally omits private mechanics. Classify the public-summary hash
as insufficient to witness executable-kernel identity for the intended
comparison. Do not classify it as proven numerical kernel drift or proven
numerical equality, and do not override the failed gate: exact top-level
equality was the active Phase 4 criterion.

## Preserved Roots

- Attempt 01 posterior result:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt01/`;
- interrupted Attempt 02:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt02/`;
- terminal Attempt 03:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt03-reboot-replacement/`;
- terminal result note:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-attempt03-result.md`.

Preserve all roots exactly. They are in the shared dirty worktree and may be
untracked. Do not clean, reset, checkout, restore, overwrite, or relaunch into
them.

## Next Work Boundary

Phase 5 is blocked under the current plan. The next justified engineering task
is to design and implement a repository-owned mechanics-stable replay identity
that binds the actual executable kernel:

- target and adapter identity;
- step size and leapfrog count;
- full fixed mass payload/signature;
- HMC execution settings that affect transitions;
- dtype/backend/XLA/TF32 policy as applicable.

Run-instance provenance such as intermediate artifact hashes should remain in
the audit lineage. The replacement identity must fail when step size, leapfrog
count, mass, target, adapter, or another transition-affecting private setting
changes, and pass when only versioned run-instance lineage changes.

Before new serious compute:

1. write a focused repair plan and skeptical audit;
2. add regression tests for true mechanics drift and lineage-only differences;
3. obtain one bounded implementation review if material;
4. request fresh user authorization and compute budget for a new second-seed
   attempt;
5. use a new versioned output root.

Do not reuse the observed Attempt 03 hash as a new expected value, because that
would not answer the original pure-replication question and would tune the gate
to a consumed holdout attempt.
