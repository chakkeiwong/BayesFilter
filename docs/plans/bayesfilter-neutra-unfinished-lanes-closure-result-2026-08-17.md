# NeuTra Unfinished-Lanes Closeout Result (2026-08-17)

This result records the bounded execution of
`bayesfilter-neutra-unfinished-lanes-closeout-plan-2026-08-17.md`. It closes
the executable Banana and German diagnostics, preserves the KSC handoff veto,
and records the two policy-blocked lanes without inventing replacement routes.

## Decision Table

| Lane | Primary criterion | Veto status | Decision | Next justified action | Nonclaim |
|---|---|---|---|---|---|
| Banana feature decomposition | Complete finite feature-family and exact-control diagnostic | No nonfinite values or hash mismatch | `DIAGNOSTIC_COMPLETE` | Use only as localization input to a separately reviewed repair | No equality proof, retraining decision, or SSL-LSTM transfer |
| KSC-UKF sequential HMC | Valid complete private broad-grid handoff, then sequential convergence/ESS/energy gates | Handoff veto fired before sampling | `INCOMPLETE_HANDOFF` | Recreate a complete private broad-grid artifact under a fresh plan/budget | No KSC posterior or sampler conclusion |
| German reverse-KL repair | Global and median-batch proposal ESS fraction >= 0.0625 | Support veto: global `0.0360679`, median `0.0361022` | `FAILED_SUPPORT_SCREEN` | Fresh target-specific support repair or stop; do not launch HMC | No HMC, posterior, weighted-objective, or ranking claim |
| PP-ZC | Source-route policy admission | Generic retained-grid route is not source-faithful | `BLOCKED_POLICY_SOURCE_ROUTE_MISMATCH` | Author-source-anchored fixed-variant route or owner-approved extension | Existing route is not production/HMC evidence |
| Austria SIR | Same finite scalar for tangent-carrying and tangent-free paths | Finite-program value mismatch | `BLOCKED_FINITE_PROGRAM_VALUE_MISMATCH` | Repair the value program and rerun parity | No NeuTra/HMC conclusion for the mismatched route |

## Banana

Artifact: `docs/plans/artifacts/neutra-unfinished-closeout-banana-2026-08-17/`.

The frozen seed-15, 6,000-update, `(32,32)`, `L=10` archive was hash checked.
The run used 4,096 draws, 256 bootstrap replicates, 64 exact-control
replications, blocks 32/64/128, offsets 0/904, float64 GPU/XLA, and GPU memory
growth. It completed in 652.03 seconds.

Across both windows, raw and latent feature upper intervals were close to the
exact-control envelope (ratios approximately 0.96--1.13). The banana-pair,
banana-residual, nonlinear-tail, and coordinate families were below the
envelope. This is a decomposition diagnostic, not an equality test. No family
was promoted as a sufficient law check.

## KSC-UKF

The frozen transport SHA-256 was checked as
`dbbaba3735404d9dd98b233e9419ab4fd3d82c8ac9a5922c9e47712d42e8bddb` and the
target signature was `727718ec8c4b4a68e2bc59c5f88d33be8e24cc4b77095f9197a360f6c9e7114d`.
The broad-grid result SHA-256 was checked as
`bab4ddb01a6b1f1f6e87197d8b72c7eae13809ff7ec0dbb49e8fe776c1227f5e`.
The shared sequential controller rejected the supplied public result because
the complete private viable-set payload was absent. The referenced private
artifact does not exist. No retraining, inferred epsilon, or hand-built
candidate was used.

## German Credit

The target data and logistic-gamma reference remained source-bound. The
200-update `(128,128)`, six-stage GPU/XLA canary was finite. The declared
3,000-update repair completed with the best heldout checkpoint at update 2,900.
The disjoint CPU proposal audit used 8,192 rows in 4,096-row batches and was
finite, but failed the predeclared support screen:

| Metric | Observed | Required |
|---|---:|---:|
| Global ESS fraction | 0.0360679 | >= 0.0625 |
| Median batch ESS fraction | 0.0361022 | >= 0.0625 |
| Minimum batch ESS fraction | 0.0339490 | descriptive |
| Maximum normalized weight | 0.0243022 | descriptive |

Because proposal support failed, no German HMC run was launched. The failure
weakens this architecture/budget hypothesis, not the NeuTra research
direction as a whole.

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | KSC handoff veto; German proposal-support veto; PP-ZC/Austria policy/value blockers |
| Statistically supported ranking | None; no ranking experiment was run |
| Descriptive-only differences | Banana feature ratios, German loss/checkpoint, ESS/tail values |
| Default-readiness | Not established for any closeout lane |
| Next evidence needed | Complete KSC private handoff; German support repair; policy-compliant PP-ZC/Austria routes |

## Artifact Map

- Banana: `docs/plans/artifacts/neutra-unfinished-closeout-banana-2026-08-17/`
- KSC attempted handoff: `docs/plans/artifacts/neutra-unfinished-closeout-ksc-2026-08-17/`
- German canary: `docs/plans/artifacts/neutra-unfinished-closeout-german-canary-2026-08-17/`
- German repair: `docs/plans/artifacts/neutra-unfinished-closeout-german-2026-08-17/`
- German proposal audit: `docs/plans/artifacts/neutra-unfinished-closeout-german-proposal-2026-08-17/`
