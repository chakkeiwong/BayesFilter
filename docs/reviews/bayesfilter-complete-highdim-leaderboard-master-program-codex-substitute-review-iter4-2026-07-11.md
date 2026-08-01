# Complete Leaderboard Master Program Substitute Review, Iteration 4

Date: 2026-07-11

Review type: `fresh_codex_readonly_substitute`

Reviewed path SHA-256:
`f8eb573d69eca8dbb841f440a8dc86fe365eb8cfc702ba451201f2c28a39ab3e`

## Verdict

`VERDICT: REVISE`

## Material Findings And Repairs

1. Computational dependency changes could use documentary re-admission instead
   of recomputation. Repair: all computation-relevant changes now force rerun;
   documentary re-admission is limited to reviewed non-computational metadata.
2. FD could trust collapsed or mismatched endpoint records. Repair: every
   per-seed/coordinate endpoint must be finite, representably distinct,
   same-route/config/randomness, and FD is recomputed from endpoint scalars.
3. Dependency completeness was declarative. Repair: Phase 8 reconciles static
   dependency closure with runtime imports/loads, generated configs, data
   opens, and transitive dependencies; unknown dependencies veto release.
4. Zhao-Cui anchor unavailability could be discovered after expensive LEDH
   runs. Repair: Phase 1 screens all six routes before Phase 2.

