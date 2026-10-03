# q20 short fit canary checkpoint

Owner authorized the proposed short test. Active plan:
`docs/plans/bayesfilter-q20-short-fit-canary-plan-2026-09-23.md`.
Original depth-four map and target source are fixed. Two fresh-slot Adam
corrections receive 64 updates each: affine, then scalar+affine. A fresh
128-point bank measures loss and score residual before/after; a separate
16-batch probe compares standard/path gradient variability at the fixed initial
scalar checkpoint. No HMC or complete training claim.

23 analytic CPU/XLA tests passed. Runner is
`docs/benchmarks/diagnose_q20_mechanism_fit_2026_09_23.py`; the executed copy and
new optional numerical module are preserved under
`docs/plans/artifacts/q20-short-fit-canary-2026-09-23/run-01/source/`.
Trusted GPU1 run completed in `run-01/worker-01/`. Supervisor charges the original
training-repair campaign as stage `short-mechanism-fit-canary`. The owner's new
test authorization provides a dedicated 1800-second diagnostic allowance inside
the unchanged total campaign; both attempts, if needed, share this cap.

The 128-point pilot left the scalar-vs-original loss inconclusive. One planned
fresh 512-point confirmation (`run-02/worker-01/`) then passed: loss difference
-0.31008, approximate 95% CI [-0.46889,-0.15127]. Score RMS 4.32079 -> 3.42853;
left latent-tail RMS 4.94179 -> 5.56382 remains an unresolved conditional concern.
No extra optimizer updates were made. Affine-only had no detectable loss benefit.
The fixed-checkpoint path gradient had descriptively 22.3% lower covariance trace
over 16 batches; no path training or estimator ranking was established.

Both independent saved-data audits passed. Results and interpretation:
`docs/plans/bayesfilter-q20-short-fit-canary-results-2026-09-23.md`.
Total worker charge: 645.1695993356407 seconds. Remaining total campaign:
143330.11637298454 seconds. The unused dedicated test allowance was retired;
the prior diagnostic balance remains 7.711412891243526 seconds. No process is
pending. Raw confirmation accounting mislabeled a stage remainder as total
canary remainder; `accounting-summary.json` preserves the corrected aggregate.

The requested test is complete. Next training work should retain the promising
nonlinear correction and examine the unresolved left-slice geometry under a
bounded continuation protocol. Full production optimizer/checkpoint integration,
1000-point post-training verification and HMC qualification remain separate.
Do not repeat a completed run or promote this canary to completed training.
