# Contract E--TP All-Model Gradient Comparison Reset Memo

metadata_date: 2026-07-15
program_status: COMPLETE_WITH_NEGATIVE_RESULTS_AND_BLOCKERS

## Resume Point

The program is closed in
`docs/plans/bayesfilter-contract-e-tp-phase10-terminal-synthesis-result-2026-07-15.md`.
Do not rerun completed rungs merely because an interactive terminal or VS Code
session disappeared. The first Phase 9 GPU job completed despite the session
interruption, and the resumed `T=50` job also completed successfully.

## Current Truth

- Contract E--Chol remains canonical. Contract E--TP remains experimental.
- LGSSM `finite_lookahead=8` passes the frozen center through `T=50` and passes
  float64 GPU/XLA parity. Its `T=50` compile takes about 28.25 minutes; warmed
  execution takes about 1.52 seconds.
- Actual SV, KSC-SV, and predator--prey have viable short-prefix evidence but
  no completed target-horizon GPU/XLA evidence.
- The tested generalized-SV feature family is a negative result, not a wiring
  failure.
- SIR observed-data comparison is blocked by clipped-push versus Gaussian-
  density measure mismatch and missing total derivative owners.
- The adjacent-state squared-TT route is an extension/invention, not Zhao--Cui.
- No source-route Zhao--Cui parameter-learning comparator is available.
- Structural support passes on the finite fixture; no executable NAWM client
  likelihood is registered.
- No cross-method equivalence margin or statistically supported ranking exists.

## Controlling Paths

- Master plan:
  `docs/plans/bayesfilter-contract-e-tp-all-model-gradient-comparison-master-plan-2026-07-15.md`;
- Phase 7 result:
  `docs/plans/bayesfilter-contract-e-tp-phase7-same-target-all-model-comparison-result-2026-07-15.md`;
- Phase 8 result:
  `docs/plans/bayesfilter-contract-e-tp-phase8-one-factor-refinement-result-2026-07-15.md`;
- Phase 9 result:
  `docs/plans/bayesfilter-contract-e-tp-phase9-gpu-xla-scaling-result-2026-07-15.md`;
- terminal result:
  `docs/plans/bayesfilter-contract-e-tp-phase10-terminal-synthesis-result-2026-07-15.md`.

## If Work Resumes

Start a new blocker-specific plan rather than reopening the completed campaign.
The smallest defensible next programs are:

1. staged/loop-native LGSSM compilation plus dtype-generic float32/TF32 audit;
2. target-specific nonlinear full-horizon recursive XLA factories and
   parameter-region validation;
3. a materially new generalized-SV distributional feature family;
4. explicit binding of a mathematically complete SIR target law and all total
   derivative owners; or
5. a client-owned DSGE/NAWM adapter with observations, structural metadata,
   parameter chart, and same-target scalar.

Do not claim canonical, default, HMC, leaderboard, Zhao--Cui source parity, or
complete nonlinear validation from the existing evidence.
