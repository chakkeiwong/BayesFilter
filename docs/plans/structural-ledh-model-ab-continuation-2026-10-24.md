# Structural Model A/B continuation, 24 October 2026

The 48-GPU-hour master remains active. At 18:06 UTC on 23 October (02:06 Hong
Kong on 24 October), approximately 46.3 GPU hours remained. This note records
existing receipts; it does not change the numerical method, seeds, targets,
stopping rules or execution budget.

Model B's first T=50 NeuTra arm, width18/seed17, completed 256 accepted updates
in 263 attempts. Its inverse error was 6.66e-16 and log-determinant finite-
difference error 1.38e-10. Its heldout mean loss was 54.8131. These establish a
valid training candidate and map implementation, not sufficient whitening or
posterior convergence. Width18/seed29 is running. The two width36 arms, selected
arm extension to 1,024 accepted updates, and adaptive HMC are already master
stages. They do not require a new direction or approval.

The training placement repair passed 21 shared native-shard regressions,
including full Adam update and checkpoint/resume parity, both precisions,
padding, invalid rollback and mixed map/target precision. Execution is GPU/XLA
for the full numerical update; host initialization and serialization remain CPU.
The handoff uses the same data/design/controls target signature, restores Adam
state, freezes the selected map, and includes its Jacobian in the MH endpoint.
Fresh and resumed sampling check sampler, transformed and physical coordinates.
Actual serious-run checkpoint restoration and HMC convergence remain pending.

## Reference evidence

`artifacts/structural-ledh-ab-20261023-01/continuation-reference-audit-20261024.json`
records 28 existing final-partition parameter probes by model and horizon,
including EKF, cubature and available bootstrap comparisons. These are reporting
calculations from saved receipts; they do not feed tuning or training.

At Model A T=50 truth, LEDH's log-likelihood difference from Kalman is -0.06641;
the maximum centered difference over its four probes is 0.02935. Exact Kalman
values still provide its HMC acceptance endpoint.

At Model B T=50 truth, LEDH minus the 65,536-particle reference is +0.31647 in
log likelihood and approximately (0.49935, -1.06118, -0.16776) in score. Reference
score MCSEs are (0.02745, 0.10027, 0.14300); the particle-ladder shifts are
(0.04296, 0.19743, 0.63970). Reference bias remains unresolved. Cubature has a
smaller descriptive score-error norm at truth and the positive local shift.
These are unresolved weaknesses in the finite LEDH approximation. Good HMC
mixing would not remove them.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue existing master | Final numerical probes valid; sampling pending | No new continuation veto; no approximation/advantage promotion | Fixed-design approximation and long-B reference bias | Finish training and adaptive HMC; interpret with independent references | Exact B posterior or superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | All current final probe workers returned PROBE_VALID; no retained posterior yet |
| Statistically supported ranking | None |
| Descriptive-only differences | Conditional score errors, tuning loss, runtime |
| Default readiness | Not established |
| Next evidence | Retained-sample convergence, A exact-posterior comparison, separate B approximation assessment |

The strongest alternative explanation for the B discrepancies is a combination
of finite-design approximation and particle-reference bias. More accurate
independent references and replicated LEDH designs could change that diagnosis.
Current probes cover local shifts and prior center, not boundary or tail regions.
