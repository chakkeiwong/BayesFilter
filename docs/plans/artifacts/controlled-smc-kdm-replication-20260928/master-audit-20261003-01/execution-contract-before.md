# Replication-first execution plan: controlled SMC and KDM/IWSG

Date: 3 October 2026. This is the active execution plan for the revised
master. It replaces the repair-first ordering; earlier repair plans remain
historical evidence.

## Research question and claim boundaries

The primary question is whether the selected controlled-SMC and KDM/IWSG
experiments can be reproduced from the inspected paper, author source, data,
settings and runtime conventions. A separate engineering question is whether
the TensorFlow/TFP implementation is numerically no worse than the original
source on source-valid cases. Numerical improvements are a third question and
cannot close either replication claim.

The replication claim does not require outperforming the paper's baselines or
repairing an instability in the author implementation. Correcting our own
translation, call-chain or solver-compatibility mistake remains replication
work. Source-prescribed safeguards must be reproduced; a source failure is
reported as a source result. It is not silently removed, repaired, or counted
as a successful replication.

## Evidence contract

| Lane | Comparator and primary criterion | Vetoes | Explanatory only | Non-claim |
|---|---|---|---|---|
| Author source | Unchanged pinned author source, driver, data and declared runtime; reproduce the selected paper-level outputs and summaries with predeclared uncertainty | Wrong source/data/driver, missing required output, corrupted artifact, or unrecorded failure | Warnings, conditioning, ESS, runtime, individual trajectories | Does not prove MATLAB and Octave are identical; does not prove superiority |
| TensorFlow port | Same frozen inputs/control decisions where possible, then matched repeated runs; no extra nonfinite/invalid outputs or failed source-valid cases, and differences stay within predeclared scaled tolerances | Extra failure on a source-valid case, wrong target/control flow, nonfinite accepted state/weight, invalid density/precision, or missing required term | Solver condition number, coefficient differences in ill-conditioned systems, compilation time | Does not require bitwise equality across RNG/solver stacks; does not certify robustness |
| Enhancement | Explicitly changed solve, projection, damping, floor, clipping, training or optimizer route | Changed route used to close a source or port gap; unreported target change | Safety curves and failure reduction | Cannot be called author replication or the default without its own evidence |

For the port, “no worse” is evaluated against the source run record. A
source-valid run has the source's declared finite state, weights, normalizer,
precision/covariance and stopping behavior. A source-invalid or source-failed
run remains a classified failure; a port that avoids it through an unapproved
change does not establish source parity. Tolerances are frozen before the
untouched comparison and scaled to the reported quantity and justified
numerical sensitivity. Conditioning does not license an arbitrary tolerance.
Fixed-input checks compare outputs and invariants, not coefficients alone when
a fit is nonunique; repeated runs compare failure frequency and downstream
likelihood/state accuracy. A nonsignificant difference is not equivalence.

## Phase graph

| Phase | Work | Completion condition |
|---|---|---|
| G0 | Freeze the source/port/enhancement contract, source commit and paper version; preserve earlier evidence and live budgets | All later runs name one lane and one claim boundary |
| C0 | Lock each controlled-SMC source matrix: model, data provenance, driver, runtime (MATLAB or explicitly labeled Octave), RNG, solver, particle count, refinements, comparator and output statistic | Every selected experiment is executable or visibly blocked with the missing item named |
| C1 | Run the unchanged author source: Lorenz Figure 6/Table 1 scope, neuroscience Figure 4 right scope, and the selected multimodal source scope after C0 resolves its driver/horizon issue | Complete source run ledger, including warnings, failures, costs and all retained outputs; paper-level comparison artifact |
| C2 | Assemble source replication results and uncertainty; separate reproduction from heuristic performance and from source-runtime limitations | Each selected C experiment is replicated, partial with a named gap, or rejected as unavailable; no silent reconstruction |
| P0 | Replay source inputs, localize and correct translation or solver-compatibility errors without changing the method, and verify the endpoint call chain | Required outputs and invariants pass frozen scaled tolerances; localization alone is partial progress |
| P1 | Run the matched TensorFlow matrix and compare source-valid outcomes and downstream accuracy under the no-worse contract | No extra source-valid fixed-case failure and declared non-inferiority is supported, or the port remains incomplete/inconclusive |
| K0 | Recover KDM Appendix B/bearings settings, data, initialization, estimator and training protocol; distinguish source facts from reconstruction hypotheses | Exact source settings are recovered or a reconstruction is explicitly labeled and cannot support exact-author claims |
| K1 | Validate normalized mixture-Kalman references and the actual IWSG/KDM consumer call chain | Reference and estimator semantics pass their declared checks; fixed-sample parity is not treated as training replication |
| K2a | Run unchanged author KDM training/filtering with independent seeds, costs and uncertainty | Complete source scope and source verdict, or a named source/compute gap |
| K2b | After fixed-input value/gradient/update parity, run the matched TensorFlow training/filtering comparison | Separate port verdict based on downstream density/state accuracy and failures; K2a alone cannot close K2b |
| E0 | Optional robustness work: high precision, projections, floors, damping, KL/model-adequacy controls and other changed routes | Separate result note, separate target identity, non-harm evidence and no effect on source/port status |
| J | Terminal review of source faithfulness, port non-regression, KDM evidence and remaining gaps | One decision table per lane; no result from one lane closes another |

Dependencies are `G0 → C0 → C1 → C2`, `C1 → P0 → P1`,
`G0 → K0 → K1 → K2a → K2b`, and `C2 + P1 + K2a + K2b → J`. `E0` may
start only after the affected replication lane has a recorded baseline and
must never be a prerequisite for it.

## Stop and continuation rules

Stop the affected phase for wrong target or data, corrupted or missing
required artifacts, an invalid reference/oracle, or exhausted budget. Continue
to the next planned diagnosis for a failed candidate when the source contract,
target, hardware class, privacy boundary and campaign budget are unchanged.
Do not spend the remaining replication budget on high-precision or KL repairs
before the source baseline and port non-regression lanes are closed.

## Artifact requirements

Each serious run preserves the source snapshot, exact command, commit, runtime,
device, dtype, RNG/data hashes, seeds, warnings, failures, wall time, budget
charge and output paths. Source and port outputs use different versioned
directories. Result notes include a decision table with hard vetoes, viable
lanes, descriptive differences, statistical support and the next evidence.

## Current readiness

G0 is the restructuring review dated 3 October 2026. C1 is partial: six small
Lorenz author-native Octave cases pass, but the selected full source matrices
are not complete. P0 is partial: source captures exist, while fixed-input
TensorFlow coefficient parity fails in ill-conditioned cases. C2, P1, K2a and
K2b remain incomplete. E0 is deferred and cannot block the replication lanes.

No numerical run is launched by this plan. A new source driver and bounded
phase specification must be reviewed for source/data completeness before the
full matrix is executed.
