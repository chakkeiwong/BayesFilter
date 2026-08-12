# SSL-LSTM q=20 seed-B NeuTra mode-failure reset memo (2026-08-10)

## State

The root cause is identified. Do not rerun the old seed-B transport with the old
kernel or interpret its R-hat/ESS as global convergence.

- Reverse-KL transport proposal coverage: 3 negative-half-space draws among
  100,000 base draws.
- Exact pullback: two distinct stationary regions, latent distance 23.707.
- Original starts: all positive, 13.1--13.9 latent units from inverse negative
  source MAP.
- Frozen `epsilon=0.811521`, `L=3`: zero negative-region acceptance at both the
  inverse source MAP and exact transformed stationary point.
- Negative transformed maximum precision eigenvalue: 91.72047; harmonic
  stability scale 0.208832.
- Causal control `epsilon=0.1`, `L=3`: negative acceptance 31/32 and positive
  acceptance 32/32; no cross-mode transitions expected at trajectory length
  0.3.

Terminal result:

`docs/plans/bayesfilter-ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-result-2026-08-10.md`

Primary artifacts:

`docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/`

## Next plan

The next campaign should not be “run more HMC.” It needs, in order:

1. A multimodal posterior-mass authority or weighted mode-discovery mechanism,
   such as tempered SMC/AIS/parallel tempering, on the exact target.
2. NeuTra training whose batches explicitly represent the discovered modes with
   validated weights, or a multimodal base/mixture-of-transports design.
3. Region-aware HMC tuning using starts in every retained region. A common
   global step is admissible only if it passes each region's numerical gates;
   otherwise use a method whose kernel/tempering design addresses the geometry.
4. A global transition test and uncertainty-aware mode-mass comparison.
5. Only after sampler validity, rerun posterior-predictive distribution tests.

The `epsilon=0.1` control is not a selected replacement kernel. It establishes
local cause only and has insufficient trajectory length for global transitions.

## Reproduction notes

- Historical tracked code commit: `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`.
- The original August 7 run used a dirty worktree; its historical transformed
  manifest hash cannot be reconstructed from the commit alone. Final diagnosis
  used checkpoint/source identities plus archived-state numerical parity under
  a measured-derived `5e-7` tolerance; artifact records
  `historical_identity_exact=false`.
- A detached code worktree was used at
  `/tmp/BayesFilter-seed-b-root-cause-historical`; this is disposable and not a
  scientific artifact.
- CPU only, GPU hidden, XLA enabled. Total admitted runtime 5,436.6 seconds,
  below the 12,000-second cap.

