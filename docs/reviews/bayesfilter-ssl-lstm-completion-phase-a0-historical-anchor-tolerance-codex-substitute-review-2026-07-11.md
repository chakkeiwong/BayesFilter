# SSL-LSTM Completion Phase A0 Historical-Anchor Tolerance Review

Date: 2026-07-11

Reviewer type: `CODEX_SUBSTITUTE_REVIEW`

Claude status: policy-unavailable; no Claude process or liveness probe ran and
no repository content was sent. These reviews are weaker than Claude review.

Exact reviewed paths, independently:

- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-governance-target-lock-subplan-2026-07-11.md`
- `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py`

## Prompt Recovery

The initial prompts forbade commands without providing file contents. Both
reviewers requested read access rather than guessing. Revised prompts permitted
only a numbered read of the exact target, followed by one targeted read of the
tool-truncated middle range. These no-verdict prompt-recovery attempts are not
substantive review rounds.

## Plan Review

No material findings. The plan limits scale-aware tolerance to the two
historical JSON value/score comparisons and separately retains the existing
decomposition checks. Fresh-process generated-lock equality, hashes, dependency
closure, and immutable projections remain exact. Handoff and stop clauses do
not contradict that scope.

`VERDICT: AGREE`

## Harness Review

No material findings. The historical scalar comparison implements exactly
`8 * eps64 * max(1, abs(current), abs(historical))`. The score comparison uses
the corresponding maximum-absolute-coordinate infinity-norm formula. The
relaxation occurs only after loading the two historical JSON anchors. Fresh
observation/probe replay and all byte, fingerprint, dependency, geometry, and
signature checks remain exact.

`VERDICT: AGREE`

## Nonclaims

This review resolves a historical decimal-round-trip verifier defect only. It
is not Claude convergence, target validation beyond A0 replay, posterior
correctness, HMC or NeuTra evidence, forecasting evidence, or scientific
promotion.

VERDICT: AGREE
