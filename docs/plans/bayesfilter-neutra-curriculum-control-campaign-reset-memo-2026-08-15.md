# NeuTra Curriculum Control Campaign Reset Memo (2026-08-15)

## State At Reset

The Gaussian/banana curriculum-search control campaign is complete. Durable
artifacts are under
`docs/plans/artifacts/neutra-curriculum-control-campaign-2026-08-15/` and all
campaign and per-target SHA-256 manifests verify. The process exited normally;
no campaign process remains.

Run provenance: commit `3030d86df9cb00346df82c7c19f015c09c7c6e1f`, TensorFlow
2.20.0, GPU 0, float64, XLA enabled, TF32 disabled, batch size 4096, memory
growth enabled before logical-device initialization, and no scalar/sample-wise
fallback. Total wall time was `1929.60 s` under the `3600 s` cap.

## Scientific Decision

Do not promote the curriculum-search procedure and do not begin SSL-LSTM
adapter work from this result. The selected protocol failed the untouched
exact-law gate on both controls. This is a protocol-selection/training result,
not a harness invalidation and not evidence against NeuTra as a method.

- Gaussian: search selected cold `LR=2e-4`; it failed 0/2. Cold `LR=1e-3`,
  retained as an explicit comparator, passed 2/2.
- Banana: search selected cold `LR=2e-4`; selected and comparator both failed
  0/2, with the first two coordinate second moments distorted.

## Required Next Work

1. Repair and retest Gaussian learning-rate selection. The current
   uncertainty-set shortest-sequence rule is a nomination mechanism, not a
   reliable exact-law selector when cold arms differ by LR. Keep the equal-work
   tournament, but define a fresh-seed nomination rule that cannot select a
   lower-LR arm solely because it lies in a broad practical-loss tie.
2. Run a separate banana target-specific plan covering initialization/order,
   direct cold-joint capacity, and a longer-budget arm. Keep exact-law audit
   data untouched and do not transfer Gaussian settings as defaults.
3. Only after both controls have a predeclared viable target-specific protocol
   should an SSL-LSTM adapter/group-ownership review be written. No HMC or
   posterior claim follows from this campaign.

## Nonclaims And Evidence Classes

Search probes, tournament loss, ESS, ratio SD, terminal gradients, and runtime
are descriptive or explanatory. They do not establish predictive equivalence,
convergence, HMC readiness, multimodal coverage, or default readiness. Hard
vetoes are the exact-law screens, nonfinite state, invalid partitions/budget,
and missing GPU/XLA/memory provenance.
