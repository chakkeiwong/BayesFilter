# SSL-LSTM q=20 NeuTra global-mixing execution reset memo (2026-08-19)

This memo supersedes
`bayesfilter-ssl-lstm-q20-neutra-global-mixing-repair-reset-memo-2026-08-19.md`
for future continuation work. The earlier memo remains historical provenance
for the pre-gateway state.

## Current state

- The reviewed execution plan is terminal at `UNDER_BUDGETED_HMC`.
- Trusted GPU preflight, anti-pooling canary, eight-cell batch-native training,
  and both frozen audits completed successfully.
- Training nominees are seed 2 and seed 3, both width 64, 3 stages, learning
  rate `3e-4`. They are not statistically ranked.
- Seed 2 with `L=3` is rejected: observation-weight rank R-hat `1.134678` and
  folded R-hat `1.129265` exceed `1.01`.
- Seed 2 with `L=5` passed a finite short screen at step `0.249446` and
  descriptive acceptance `0.763694`, but its long verification was refused by
  the compute callback. It is unadjudicated, not failed.
- `L=10`, `L=15`, transport seed 3, canonical sequential HMC, and predictive
  validation were not run.
- The HMC terminal artifact understates prior GPU wall by `19.8 s` because it
  omits the failed preflight (`9.5 s`) and launch-invalid canary (`10.3 s`).
  Correct cumulative GPU wall is `12,499.045 s`; no cap was breached and the
  resource-stop decision is unchanged.
- No posterior bank, mode weights, predictive result, scientific promotion, or
  default-readiness result exists.

## Binding files

- Execution plan:
  `docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-plan-2026-08-19.md`
- Terminal result:
  `docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-result-2026-08-19.md`
- HMC terminal artifact:
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc/result.json`
- HMC manifest:
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc/manifest.json`
- HMC hash inventory:
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc/artifact-hashes.json`
- Training result:
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/training-screen/result.json`
- HMC runner:
  `docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_hmc_2026_08_19.py`
- Shared sequential controller: `bayesfilter/inference/neutra_hmc.py`
- Route ledger:
  `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/c0/route_ledger.json`

## Resume boundary

Do not rerun into `r1/hmc`; it is a terminal immutable output. Any continuation
needs a fresh reviewed plan and versioned root. The next smallest discriminating
run is seed 2 `L=5` full verification. Its conservative measured/canary-derived
forecast is `15,546.542 s` plus closeout. A complete path through the canonical
minimum sequential evidence is approximately `45,676 s` (`12.69 h`) plus
orchestration overhead, excluding fallback lengths and seed 3. Those numbers
are planning evidence, not authorization or a guaranteed runtime.

Before a new serious launch:

1. Bind the unchanged target, training state, route policy, GPU/XLA mode, and
   memory-growth receipt.
2. Count every GPU-initializing failed attempt in aggregate wall accounting;
   do not reuse the HMC runner's successful-launch-only prior-wall constants.
3. Decide whether to preserve the existing fixed tuner or repair its resource
   exception reporting so a refused verification is not rendered as synthetic
   numerical hard vetoes inside the lower-level artifact.
4. Use a fresh root and a budget that covers the next indivisible call plus
   closeout. Do not borrow from a predictive reserve.
5. If `L=5` passes tuning verification, use the canonical shared sequential
   controller; do not shorten warmup/retention or pool conditional chains.

## Do not conclude or reuse

- Do not call the `L=5` resource refusal a convergence or candidate failure.
- Do not reject seed 2, seed 3, or NeuTra from the `L=3` kernel failure.
- Do not use the mechanics canary, training losses, SMC replay weights, pooled
  occupancy, acceptance, or the `L=5` screen as posterior evidence.
- Do not run the predictive endpoint without an admitted common-kernel
  posterior bank.
- Do not silently resume or overwrite any `r1` artifact.
