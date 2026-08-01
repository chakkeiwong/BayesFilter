# Codex Substitute Review: Phase 9 Gate B Fixed-SIR Repair, Iteration 1

Date: 2026-07-11

## Scope And Limitation

Fresh local read-only review of exactly the fixed-SIR Gate B attempt-1 failure,
the archived artifacts/logs, the covariance repair, its focused tests, the
repair result, and the frozen Phase 9 execution contract. Claude remains
policy-blocked as external repository disclosure. No GPU/CUDA command ran
during this review.

## Findings

### Blocking: post-repair shards would not bind this repair authorization

`benchmark_ledh_compact_score_gpu_xla.py` still records only
`GATE_B_REVIEW_PATH`, which points to the pre-repair Gate A/manifest iteration-2
review. The governance hash set and shard validator likewise bind only that
older review. A retry after a source repair therefore would not prove which
fresh review authorized the changed code.

This is a provenance defect, not a defect in the covariance repair. Before a
trusted retry, add a fixed repair-review path to the run manifest, governance
hash set, and shard validation contract. Add a focused test that rejects a
missing or changed repair-review path/hash. The exact runtime argv must remain
unchanged.

## Functional Repair Assessment

The scoped code repair is correct relative to the prepared fixed-SIR tensor
contract. `_build_actual_sir_tensors` obtains the fixed callback covariance
before tracing and tiles it over the singleton batch. The removed graph-time
constructor recovered that same fixed covariance. Cholesky of
`transition_covariance[0]` therefore preserves the matrix shape and value used
by the existing `einsum` without changing target or transport math.

The new CPU-hidden XLA/eager parity test directly covers the failed route, and
the broader `151`-test and `53`-test suites pass. Attempt-1 evidence is archived
with recorded hashes. The failure is correctly classified as extraction
failure rather than numerical FD evidence.

## Authorization Boundary

No GPU retry is authorized by this verdict. Gate B, Gate C, Gate D,
aggregation, and LGSSM remain blocked.

VERDICT: REVISE
