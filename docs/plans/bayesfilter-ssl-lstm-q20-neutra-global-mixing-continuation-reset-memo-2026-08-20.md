# SSL-LSTM q=20 NeuTra global-mixing continuation reset memo (2026-08-20)

This memo supersedes
docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-reset-memo-2026-08-19.md
for work after the 18-hour continuation. The 2026-08-19 memo remains binding
historical provenance for r1.

## Terminal state

- The 2026-08-20 continuation is terminal with
  HMC_NO_CANDIDATE_ADMITTED; its HMC wall was 23,280.603976539 s.
- Both predeclared candidates ran full four-chain, 2,000-draw-per-chain
  raw-coordinate verification and were rejected by the same required modern
  folded R-hat gate: seed 2 L=5 had 1.0875996310350042, and seed 3 L=3 had
  1.1020661342469682, against <= 1.01.
- Finite state, target, score, target-status, and log-acceptance checks passed
  for both verifications. Native divergence telemetry was unavailable, so there
  is no zero-divergence claim. Acceptance is explanatory only.
- Neither candidate selected a kernel. Mechanics, post-selection consumer
  parity, canonical sequential warm-up/retention, ESS, direct sign traversal,
  and predictive validation were not run.
- No predictive root exists. No posterior bank, mode weights, predictive
  decision, model-adequacy claim, scientific promotion, or default-readiness
  result exists.
- The fresh HMC artifact inventory has 15 entries and all rehash correctly.

## Binding files

- Continuation plan:
  docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-continuation-plan-2026-08-20.md
- Terminal result:
  docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-continuation-result-2026-08-20.md
- Terminal HMC result:
  docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r2/hmc/result.json
- Terminal HMC manifest:
  docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r2/hmc/manifest.json
- HMC SHA-256 inventory:
  docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r2/hmc/artifact-hashes.json
- Continuation runner:
  docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_continuation_2026_08_20.py
- Shared fixed-transport tuner:
  bayesfilter/inference/fixed_transport_hmc_tuning_tf.py
- Canonical sequential controller: bayesfilter/inference/neutra_hmc.py
- Route ledger:
  docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/c0/route_ledger.json

## Resume boundary

The r2/hmc root is terminal evidence and must not be overwritten, resumed, or
modified. The schedule in the executed plan is exhausted: it authorized exactly
seed 2 L=5 followed by seed 3 L=3 after a genuine diagnostic rejection.
Remaining wall time within the original upper cap is not an automatic
authorization to try another kernel, rerun a failed pair, retrain a transport,
or start predictive work.

Any future serious continuation must start from a new plan and a new versioned
output root. It must preserve the target, immutable input hashes, canonical
sequential route, GPU memory-growth policy, and anti-pooling boundaries unless
it explicitly changes one of them and justifies that change. It must also:

1. State the new scientific or engineering question separately from the
   implementation checklist.
2. Define a genuinely new target-specific kernel/tuning investigation rather
   than treating a failed pair as admitted or silently reusing its result.
3. Audit every material numerical change with provenance, a failure mode, and
   an early discriminating diagnostic.
4. Predeclare candidate ordering, full answer-path affordability, stop
   conditions, nonclaims, and a fresh budget that includes required sequential
   evidence if any candidate passes verification.
5. Keep calibration/tuning and any later claim evidence disjoint, retain failed
   verification artifacts, and make no ranking without suitable uncertainty
   evidence.

## Do not conclude or reuse

- Do not describe either failed pair as globally mixed, posterior-converged, or
  a viable HMC kernel.
- Do not reject either full frozen transport, the SSL-LSTM target, or NeuTra
  from these scoped pair failures.
- Do not compare the two attempts as better/worse based on acceptance, R-hat,
  or runtime alone.
- Do not use tuning traces, mechanics diagnostics, or conditional chain data
  as posterior samples or mode-weight evidence.
- Do not claim native divergence count zero; it was not exposed.
- Do not launch predictive validation without a future artifact whose terminal
  status is HMC_ADMITTED_FOR_PREDICTIVE and whose retained archive identity has
  been verified.
