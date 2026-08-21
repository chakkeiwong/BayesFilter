# NeuTra curriculum search reset memo (2026-08-15)

## Current state

The target-specific curriculum-search plan and pure policy mechanics are
complete and tested. No GPU campaign was launched.

Key paths:

- plan: `docs/plans/bayesfilter-neutra-curriculum-search-plan-2026-08-15.md`;
- review result: `docs/plans/bayesfilter-neutra-curriculum-search-plan-review-result-2026-08-15.md`;
- search policy: `bayesfilter/inference/neutra_curriculum_search.py`;
- tests: `tests/test_neutra_curriculum_search.py`.

## Required next sequence

1. Write a bounded Gaussian/banana control-campaign plan with explicit group
   adapters, probe threshold calibration, full-protocol practical tolerance,
   GPU count, update budget, seed partitions, and artifact root.
2. Implement probe callbacks that restore common parent model/optimizer states
   and call the existing batch-native GPU/XLA staged controller.
3. Run a tiny trusted GPU/XLA mechanics canary.
4. Run beam search, then the mandatory paired equal-budget full-protocol
   tournament.
5. Freeze the nominated protocol, retrain on fresh data, and apply the untouched
   exact-law Gaussian/banana validation.
6. Only if the controls demonstrate useful target-dependent selection, inspect
   and document SSL-LSTM parameter-group ownership and write a separate
   SSL-LSTM curriculum-search/predictive-audit plan.

## Cautions

- Local probe LCBs are comparable only among children of the same parent.
- `representative_sequence` is bookkeeping, not scientific selection.
- Search loss, protocol-selection loss, ESS, and moments cannot replace the
  final predictive-output distribution test.
- Minimum improvement and practical loss tolerance require target-specific
  provenance before a campaign.
- Preserve unrelated dirty-worktree changes from other agents.
