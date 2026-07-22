# Structural UKF NeuTra Phase 1 Canary Result

Date: 2026-07-17

Decision: `PASS_PHASE1_CONTINUE_TARGET_SPECIFIC_TRAINING`

The structural UKF target passed a batched 128-row GPU/XLA canary on an NVIDIA
GeForce RTX 4080 SUPER. All target values and scores were finite, target status
was valid for every row, value and score outputs were placed on GPU, TensorFlow
memory growth was enabled before logical-device initialization, and the replayed
typed target signature was
`e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665`.

Two earlier canary attempts stopped before target evaluation. Attempt 1 exposed
a generic identity comparison mismatch. Instrumented attempt 2 isolated the
known audit-point device drift: typed identity binds serialized audit points,
so constructing them on GPU changes their byte representation. Pinning audit
point construction to CPU restored the admitted identity exactly. This was a
harness repair; the data, mathematical target, model, criteria, and compute
class did not change.

Passing artifacts:

- result SHA-256:
  `de391ec2c64e94a0d51816a238fdc4642fdef080d11174d5b60044e2592aa481`;
- run-manifest SHA-256:
  `f87a322e6d49bb8170ade83d9bf98a4eebca73832e14420dfd7a3858b862871d`;
- recursive-hash ledger SHA-256:
  `b782d11975c397a23d065e4e0d6e9f96c4659de4d42787c520238c4e33b97c4b`;
- focused CPU-only regression before the canary: `64 passed`.

Phase 2 entry conditions are satisfied. Run the four declared 500-step fresh
screens in recipe order, finalize the common-heldout selection, and train one
fresh 5,000-step transport from the selected recipe. Screen weights may not be
reused, and heldout reverse KL remains nomination evidence only.

No HMC, truth recovery, filter exactness, calibration, superiority, or
readiness claim is made by this phase.
