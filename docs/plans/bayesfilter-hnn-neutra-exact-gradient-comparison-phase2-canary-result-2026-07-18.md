# HNN-NeuTra Exact-Gradient Comparison Phase 2 Canary Result

Decision: `PASS_PHASE2_FOUR_OF_FOUR_TWO_ARM_GPU_XLA_CANARIES`.

## Results

| Cell | HNN cold compile + 2 transitions | Exact gradient cold compile + 2 transitions | Both passed |
| --- | ---: | ---: | --- |
| PP-UKF | 7.400 s | 27.785 s | yes |
| PP-SGQF | 5.724 s | 14.852 s | yes |
| SIR-SGQF | 5.588 s | 15.596 s | yes |
| STR-UKF | 6.448 s | 22.307 s | yes |

These cold canary timings are compilation/mechanics diagnostics only. They are
not sampling-speed or ESS-normalized performance results.

Every passing canary established:

- trusted RTX 4080 SUPER GPU execution with memory growth;
- XLA compilation for both HNN and exact transformed-filter gradient forces;
- exactly three force calls per transition for `L=2`;
- exactly one new exact endpoint value per transition;
- finite energies and exact energy-identity replay;
- common target/chart identity and value-only endpoint parity.

STR-UKF additionally preserved
`k_t=phi*k_(t-1)+gamma*m_t^2` and
`artificial_k_noise_allowed=false`.

## Repairs

PP-UKF attempts 01 and 02 stopped before any chain:

1. Memory growth was configured after a TensorFlow-initializing module import.
2. The generic runner assumed a context-stored grid that predator-prey had
   historically hardcoded.

Both were classified as localized harness failures, preserved, repaired with
focused regressions, and retried under the unchanged scientific contract.

## Verification

- all four result JSON files parse;
- all four result/run-manifest hash ledgers exist;
- focused comparison tests: `8 passed`;
- Python compilation and `git diff --check`: pass;
- failed-arm cost reporting returns explicit unavailable efficiency rather than
  crashing or fabricating a metric.

## Decision And Inference Status

| Field | Status |
| --- | --- |
| Primary criterion | four of four cells pass both force routes |
| Hard veto status | none in passing attempts |
| Viable serious cells | PP-UKF, PP-SGQF, SIR-SGQF, STR-UKF |
| Ranking | none; canary timings are explanatory only |
| Default readiness | not established |
| Main uncertainty | fresh HNN training, independently tuned convergence, accuracy, and synchronized warm speed remain unrun |
| Next justified action | execute PP-UKF serious comparison, then PP-SGQF if no shared veto |

Phase 3 was reviewed against the frozen target/chart, exact two-arm set, fresh
HNN requirement, common seed domains, adaptive caps, synchronized timing,
truth/direct-agreement gates, and six-hour per-cell ceiling. No continuation
veto fired.
