# Structural UKF NeuTra Phase 2 Session Repair

Date: 2026-07-17

Classification: `INFRASTRUCTURE_MANAGED_SESSION_RESTART`

The selected fresh 5,000-step training attempt lost its process when the
managed Codex session restarted. No kernel OOM or CUDA fault was observed, and
the attempt wrote only its configuration: there is no checkpoint, frozen
transport, or scientific result to interpret.

Attempt 2 preserves the selected recipe, target, seed, optimizer, learning-rate
schedule, batch size, XLA/GPU route, and total 5,000-step budget. It partitions
the same deterministic stateless trajectory into five 1,000-step compiled
`tf.while_loop` segments. Each later segment uses the repository's tested
`infrastructure_resume_same_config_fresh_output_v1` path, which restores all
trainable variables and both Adam moments from a hash-checked checkpoint in a
fresh directory. Existing unit evidence proves equality with an uninterrupted
run for this resume mode.

This is a localized infrastructure repair. It adds no sample-axis or training-
step Python loop, changes no scientific default, and does not reuse screen
weights. The final segment alone may freeze the transport and hand it to HMC.
