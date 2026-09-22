# Phase 0F: coupled finite differences and frozen design

The three requested stencils now call the actual verified filter value endpoints. The 40-row CPU FP64/XLA mechanics run completed in 31.288 seconds. Its seven directions reconstruct all six parameters; full covariance propagation agreed with the observed score covariance to at most 1.82e-13 in absolute Frobenius norm. Shared innovations and independent-node sampling both preserve the marginal filter law. This check establishes the calculation, not a universal noise curve or stochastic stencil order.

The first pilot selected three-point h=0.02 on its calibration dataset. Validation vector MSE was 0.215895, compared with 0.215729 for the analytical score of the same finite program. The independent-node version was 839.48. Two particle replicates per dataset make these descriptive observations only.

A separate eight-row selection fixture used fresh calibration/validation datasets and selected h=0.01 from h=0.01 and h=0.02. The repository-issued selection binds the full candidate family, scope, executed controls, directions, coupling and step. It validates the original completed results and rejects source changes, altered selections, changed designs, reused calibration data and promotion of mechanics evidence into a scientific claim. Calibration vector MSEs were 4.907907 and 4.909713; their difference has no supported ranking.

The initial held-out attempt exposed a coordinator bug before numerical execution: it demanded the standard LEDH selection for the separate FD estimator. The corrected coordinator dispatches to the FD selection contract. The complete eight-row selection was repeated under the repaired source in `fd-selection-cpu-02` (23.729 seconds), then two held-out plumbing rows completed in `fd-frozen-claim-cpu-02` (20.241 seconds). Previous artifacts remain unchanged. The failed invocation of a nonexistent test path was repaired to the actual coordinator test file; 19 selection/coordinator tests passed in 9.92 seconds. Ten numerical/selection tests had already passed in 20.30 seconds.

| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Admit stochastic FD mechanics | All three stencils, actual consumers, covariance propagation and frozen-design consumption checked | No failed completed-row validity checks; pre-run entry bug repaired | Tiny datasets/replications and Gaussian model only | Extend model coverage and evaluate under independent scope tuning | FD improves the model score |
| Retain both h candidates | Calibration choice executes and claim consumes it | No data leakage permitted | Choices differ across tiny fixtures | More calibration data in a later scientific slice | h=0.01 is a universal default |
| Keep normalization/consistency phase open | Stencil error attribution exists | Missing broader ratio and consistency studies | Bias attribution beyond the oracle fixture remains incomplete | Implement Phase 4/4B reports as methods become available | Full Phase 0F or the master is complete |

| Inference status | Finding |
|---|---|
| Hard veto screen | Verified runs finite and complete; stale/forged/scope-mismatched selections rejected |
| Statistically supported ranking | None |
| Descriptive differences | Shared-innovation noise is much smaller here; calibrated h changes across datasets |
| Default readiness | No default or HMC promotion |
| Next evidence | Multiple observation datasets, many independent particle replicates, paired uncertainty and matched-cost accounting |

The strongest alternative explanation for the apparent differences is particle and observation variation. Larger independent replication could reverse the ordering. The weakest evidence is the two-replicate covariance estimate. Continue the implementation program: KDM covariance lifecycle, fitted iAPF, nonlinear references, capacity/timeout pilots and ratio/consistency reporting remain open. The refreshed finite budget and stop rules are in `phase-0g-repair-execution-plan.md`.
