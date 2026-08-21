# Reset memo: frozen weighted NeuTra HMC seed replications (2026-08-12)

## Terminal state

The four-root frozen-kernel analytic HMC campaign is complete. All four roots
passed the declared sequential and analytic primary gates. Raw warm-up and
retained tensors, per-root results, receipt hashes, consolidated JSON, runner,
tests, plan, and terminal result are preserved.

Canonical result:
`docs/plans/bayesfilter-defensive-weighted-neutra-analytic-hmc-replication-result-2026-08-12.md`.

Canonical artifact root:
`docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-replications-2026-08-12/`.

Summary SHA-256:
`33f5b59b831dea76818a88e9c161c19e02cb3a0e3405de3b036221494b0211e8`.

## What is established

- The frozen transport and fixed `L=20`, epsilon `0.14091138276334744`
  kernel passed four independent HMC root seeds on the known analytic target.
- All roots passed maximum-over-latent-and-physical R-hat `<=1.01` and bulk/tail ESS `>=400`.
- All roots' analytic minority-mass 99% intervals contain `0.2`.
- Every chain observed and transitioned between both hard-assignment modes.
- No hard numerical, movement, target-status, or exposed-divergence veto fired.
- Warm-up was archived and excluded; all 492 checked receipts passed.

## What is not established

- Native divergence count is unavailable, not zero.
- Finite energy-error proxies have extreme tails (`~4.39e14` to `~1.04e17`).
- One transport-training seed does not establish training-seed robustness.
- One analytic target does not establish cross-target or SSL-LSTM validity.
- No stationarity/equality proof, sampler superiority, or default promotion follows.

## Next phase

1. Diagnose the finite energy-error tail on the current immutable runs without changing their verdict.
2. Select fresh frozen transports by a neutral pre-HMC rule from the completed training confirmations.
3. Freeze target, tuning policy, kernel-selection rule, roots, gates, and compute cap in a new reviewed plan.
4. Tune each changed transport scope as required; do not reuse `L=20`/epsilon merely because this transport passed.
5. Run independent HMC roots and report each transport separately before any cross-transport conclusion.
6. Only after cross-transport robustness, continue the remaining analytic target suite. Do not advance this evidence directly to SSL-LSTM.

## Recovery notes

- Serious GPU route: TensorFlow 2.20.0, TFP 0.25.0, float64, TF32 off, XLA on.
- Mandatory launch environment: `TF_FORCE_GPU_ALLOW_GROWTH=true` before Python import.
- Canonical controller: `bayesfilter_neutra_sequential_hmc_v1`.
- Frozen checkpoint SHA-256:
  `af961871dcc3b626216d7500e695534f147ecfd9ba4fe0f9907f59018d40e8e5`.
- Frozen tuning SHA-256:
  `6dfe2b8145040a18831a08032bfd61854189f2651e76c70842e59d4e4e12eb4f`.
- Passed canary:
  `docs/plans/artifacts/defensive-weighted-neutra-analytic-hmc-replications-2026-08-12-canary-v3/`.
- Pre-transition provenance failures are preserved under roots ending in
  `launch-invalid-v1`, `canary-launch-invalid-v1`, and
  `canary-launch-invalid-v2`; they contain no HMC samples.
- Final focused tests: `30 passed`.
- Run manifests record the run-time Git state; the terminal manifest separately
  records the later merged repository state.
