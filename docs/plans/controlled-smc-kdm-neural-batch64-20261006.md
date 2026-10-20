# KDM batch-size-64 training cost pilot — 6 October 2026

Authority: controlled-smc-kdm-replication-master-program-20260928 and accepted
criteria. The neural replay and native sampling diagnostics are complete; full
training and qualification remain incomplete.

Question: can the source-faithful TensorFlow neural consumer execute one actual
batch-size-64 update at the source horizon and architecture within the remaining
grant, and does its fixed replay still match the author at the existing fixture
bounds?

Comparator: pinned author checkpoint loader, source full-sequence loss,
truncated-BPTT cadence, global norm clipping and Torch Adam. Use 64 saved source
training rows from the verified 5,000-row dataset, source checkpoint
85cb07f3f46a2efe1b37b4469c1bafed57cf5440489cfadee9362865bddb3d1c, 25 particles,
17 states, sparse masks and one update. This is a cost/wiring pilot, not a
training-stage or stochastic qualification claim.

Primary checks: source fixture is complete and hash verified; all MDPF 31 and
A-MDPF 38 trainable tensors are covered; no frozen initializer changes; losses,
raw/clipped gradients, parameters and Adam moments pass the prior dtype bounds;
TF uses GPU/XLA, memory growth, one trace and no pfor/eager fallback. Record
source CPU time, TF compile and steady-state times, host/device memory and
per-batch throughput. Compare float64 and float32 separately. A failure localizes
the consumer or cost; it does not reject KDM.

Default/assumption audit: batch size 64 is the released source training default,
not a newly selected value. One fixed contiguous development batch is a cost
fixture, not evidence about shuffled training or generalization. The source
checkpoint and rows are development data, never untouched qualification data.
The existing two-update batch-2 results remain the detailed parity authority;
this pilot tests scaling and call-chain coverage. No new clipping, damping,
projection, tolerance or source repair is introduced.

Budget: two attempts, at most 3,600 CPU wall seconds and 3,600 GPU wall seconds
combined for this phase, fresh directories
artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-batch64-20261006-NN.
Attempt 01 is source export plus GPU/XLA; a localized harness repair may consume
attempt 02 under the same bound. Existing grants are not increased. CPU source
runs hide CUDA; GPU runs use trusted access, memory growth and TF32 off.

Skeptical audit PASS: the pilot has a source comparator, real released batch
size, explicit cost question, hard wiring checks, stop conditions and no
promotion claim. It cannot answer the full eleven-trial paper scope or
downstream no-worse likelihood/RMSE/NLL margins; those remain future evidence.

## Terminal accounting — 7 October

Fixed replay passed 896/896 in attempt 02; see controlled-smc-kdm-neural-batch64-result-20261007.md. This phase does not establish steady-state cost. Two source process wall times/logs were not captured; conservative CPU charge 120 seconds, GPU measured 55.778 seconds. Training-controller and native integration work has its own bounded next-phase plan. No acceptance margins were changed.
