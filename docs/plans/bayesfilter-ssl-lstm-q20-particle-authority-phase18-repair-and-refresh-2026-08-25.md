# Phase 18 Repair and Refresh Note

Status: `PASS_SOURCE_FAITHFUL_ETPF_FIXTURE_REFRESHED_PHASE19`

The first fixture attempt reached the numerical computation but failed while
building its manifest because the required Phase 18 subplan path did not yet
exist. The failure is a harness/documentation issue; no numerical or method
conclusion is drawn and the partial output directory is preserved.

Repair: create the exact subplan named by the runner, then rerun the unchanged
fixture in a fresh `phase18-attempt2-etpf-fixture` directory. The source-derived
Riccati tolerance and all gates remain unchanged.

After the rerun, classify any failure as Sinkhorn marginal, Riccati convergence,
moment identity, source/hash, or artifact construction. Refresh Phase 19 only
after a complete result receipt.

Attempt 2 passed all gates. The first-order transport remained nonnegative and
the Riccati-corrected transform satisfied the row/column and moment receipts.
The corrected negative fraction (`0.46875`) is preserved as a known
second-order-support diagnostic. Refresh Phase 19 to a small q=20 integration
probe; do not scale directly to N=300 until that probe has target/status
evidence.
