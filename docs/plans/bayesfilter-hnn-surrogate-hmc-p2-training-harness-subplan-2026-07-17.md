# P2 Subplan: Scalar-Force Training, Tuning, And Campaign Harness

Phase objective: add target-specific residual-potential training, force
freezing, disjoint tuning, sequential sampling, cost accounting, and Tier A
adapter integration around the admitted P1 kernel.

Entry conditions: P1 kernel and trace contract pass.

Measured P1 entry evidence: 15 CPU/reference tests and two trusted GPU/XLA
canaries pass.  The repository pytest harness hides GPUs unless
`BAYESFILTER_TEST_DEVICE_SCOPE=visible`; every P2 GPU pytest command must carry
that opt-in.  P2 reuses `neutra_campaign.py` typed identities and archival
contracts plus `hmc_convergence.py` modern diagnostics.  It does not create a
parallel identity or R-hat implementation.

Required artifacts:

- scalar residual-potential trainer and frozen artifact loader;
- target/chart binding and normalization identities;
- P0-frozen architecture/optimizer screen and heldout diagnostics;
- HNN-specific tune/verify/freeze adapter using modern R-hat/energy health;
- warm-up/retained archive, truth-tail calculator reuse, run manifest, and cost
  ledger;
- generic Tier A campaign driver with dry-run mode.

The new implementation scope is limited to the missing scalar residual
potential trainer, frozen force artifact/loader, corrected-kernel binding, and
thin orchestration adapters.  Existing dense-IAF NeuTra transport training is
not a substitute for scalar-force training and is not duplicated.

Required checks/tests/reviews:

- analytic Gaussian and banana/funnel training fixtures;
- scalar gradient equals exported force; offset invariance; finite shell/tail
  heldout checks; frozen reload parity;
- disjoint training/tuning/warm-up/retained seeds and no tuning on retained data;
- tuning selection cannot use acceptance alone and fails closed on health veto;
- target/chart/force substitution-negative tests;
- transformed target-value/score parity including the complete chart
  log-Jacobian at training, tuning, and endpoint correction boundaries;
- batch-native GPU/XLA training with memory growth, no active-path NumPy or
  Python sample-axis loop;
- dry-run manifests for all five Tier A cells.

Evidence contract: P2 admits the mechanics and evidence pipeline. Training loss
and heldout force quality are nomination/veto diagnostics only.

Forbidden claims/actions: no model confirmation from a smoke; no global
architecture default; no cross-target force reuse; no training-data leakage;
no silent fallback from learned to true gradients inside the symmetric map.

Exact P3 handoff: analytic fixtures pass; each Tier A dry run resolves its
target/chart identity; command-level LGSSM pilot budget and stop rules are
refreshed.

Stop conditions: target-specific scalar force cannot be bound safely to the P1
kernel, training path violates GPU/XLA policy, or shared archival/tuning cannot
preserve disjoint evidence. Local mechanics failures permit three repairs.

Skeptical audit, refreshed 2026-07-17: passed.  The exact baseline remains the
same-chart true-gradient HMC; loss is not a promotion criterion; target and
chart identities are substitution-negative tested; disjoint seed domains and
the 10,000-draw cap remain prospective.  Analytic fixture commands answer
training mechanics only and cannot be used as model confirmation.

Phase-end duties: run checks; write P2 result; refresh P3 with measured canary
cost; review P3; continue if no real blocker.
