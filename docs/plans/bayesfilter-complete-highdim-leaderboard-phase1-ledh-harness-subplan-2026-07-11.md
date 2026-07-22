# Complete High-Dimensional Leaderboard Phase 1 Subplan

Date: 2026-07-11

Status: `DRAFT_REFRESHED_FROM_PASS_PHASE0_BOUNDARY_FREEZE_REVIEW_REQUIRED`

## Material Review Question

Does this Phase 1 subplan safely sequence canonical-target and Zhao-Cui
feasibility pre-gates before scoped six-row harness implementation, while
closing the paired total-value/score, individual FD-endpoint, and five-seed
aggregation defects without authorizing runtime execution?

## Phase Objective

Extend the current endpoint-rich schema-v4 score/FD harness to all six LEDH
main rows, including LGSSM; generate byte-level canonical target signatures;
and repair aggregation to preserve paired total values/scores and all five
individual seed-level FD verdicts. Do not change row targets, parameter
coordinates, seeds, transport settings, or the FD-only policy. July 7 forward
artifacts remain target/shape and historical-forward evidence only.

## Entry Conditions

- Phase 0 result status is `PASS_PHASE0_BOUNDARY_FREEZE`.
- Phase 0 JSON SHA-256 is
  `4115ef55114ffd73255363f0c62c4a19dd85d7ca3241d002c48409cb9004f878`
  and hashes/normalizes every Phase 0 input named here.
- Master and Phase 0 material reviews converged.
- Current root-cause repair tests and all six row contract tests pass.
- Phase 0 classifies all nine July 3 non-LEDH rows as candidates, not admitted
  cells, and requires Phase 1 canonical target signatures before runtime.
- No GPU execution is authorized by this Phase 1 implementation subplan.

## Sequenced Gates

1. `P1-A canonical target freeze`: without editing the harness, materialize
   exact observation tensors and row semantics for all six rows; hash
   observations, initial/time conventions, target densities/normalization,
   evaluation theta/order, generator/data/config paths, and their code hashes.
   Cross-check existing July 3/7 values only as diagnostics. Write a hashed
   `P1-A` gate receipt that also re-hashes the Phase 0 harness and proves that
   it is byte-identical to its Phase 0 frozen hash. Any contradiction or
   harness drift stops before implementation.
2. `P1-B Zhao-Cui availability`: inspect the paper/math and local author source
   for all six rows; record candidate fixed-variant route, exact source
   availability, preliminary classification, and whether an unapproved
   extension/invention is already necessary. Write a hashed `P1-B` gate
   receipt. Any missing/contradictory anchor or required invention stops before
   implementation.
3. `P1-C harness implementation`: only after both P1-A and P1-B pass and their
   receipt hashes are recorded, add LGSSM and the repaired schema
   fields/validators/tests. This gate cannot change target signatures or FD
   thresholds. The implementation result must bind both pre-gate receipts.
4. `P1-D command freeze`: generate, inspect, and check exact future Phase 2/3
   commands. Do not execute them.

## Required Artifacts

- Scoped harness implementation and focused tests.
- Six canonical row-target signature records that hash exact observations,
  initial/time conventions, target and normalization semantics, theta order and
  values, and generator/config sources independently of algorithm settings.
- Separate hashed `P1-A` and `P1-B` pre-implementation gate receipts. The
  `P1-A` receipt must include the Phase 0 frozen harness path, Phase 0 expected
  SHA-256, observed pre-edit SHA-256, and equality verdict; the `P1-B` receipt
  must bind every paper and author-source anchor it inspected.
- A six-row Zhao-Cui source-anchor availability and route-feasibility ledger
  that can veto expensive Phase 2 execution when source grounding is already
  unavailable or would require an unapproved invention.
- A current-source exact command-manifest builder for later Phase 2/3 runs.
- Phase 1 result and refreshed Phase 2 subplan.
- A material implementation/result review artifact.

Exact source and test files may be narrowed after Phase 0 inventory, but the
primary candidate is
`docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py` plus dedicated tests.

Required namespace artifacts:

- `docs/plans/artifacts/complete-highdim-leaderboard/phase1-canonical-targets-2026-07-11.json`;
- `docs/plans/artifacts/complete-highdim-leaderboard/phase1-p1a-gate-receipt-2026-07-11.json`;
- `docs/plans/artifacts/complete-highdim-leaderboard/phase1-p1b-gate-receipt-2026-07-11.json`;
- `docs/plans/bayesfilter-complete-highdim-leaderboard-phase1-zhao-cui-anchor-availability-2026-07-11.md`;
- `docs/plans/complete-highdim-leaderboard-ledh-phase2-phase3-exact-commands-2026-07-11.json`.

### Canonical Byte Encoding

The target-signature builder must not rely on framework serialization or
implicit dtype inference. For every target-defining tensor it must receive and
record an explicit semantic field name, row-specific declared dtype, exact
shape, exact source slice/index range, preprocessing steps in execution order,
and a `C` memory-order marker. Each row must have an authoritative ordered
field-name ledger stored in the signature artifact; a checker-owned independent
constant must state the expected ordered field-name ledger for that row and
reject omissions, additions, or reordering before hashing. Numeric payloads
are encoded as contiguous raw
bytes in canonical little-endian byte order without a value or precision
conversion; strings are UTF-8. Every length prefix is exactly one unsigned
64-bit integer encoded in eight bytes in network/big-endian byte order, giving
the following frame's byte count. Before each payload, hash a length-prefixed
UTF-8 header produced by canonical JSON (`sort_keys=True`, separators
`(',', ':')`, `allow_nan=False`) containing the field name, NumPy-style dtype
descriptor including kind and item size, shape, byte order, memory order,
source slice, and preprocessing ledger. Hash a length-prefixed payload after
that header. The row digest is the SHA-256 of the ordered sequence of these
header/payload frames plus a final length-prefixed canonical-JSON semantics
frame covering initial and time conventions, density and normalization
definitions, theta names, order and exact values, generator/data/config paths
and hashes. Header/payload alternation and the final semantics-frame position
are fixed; no untagged optional frame is permitted. The artifact must retain
the authoritative ordered-field ledger, each field digest, and row digest.
Missing metadata, an implicit or object dtype, nonfinite numeric data,
unrecorded preprocessing, a field-ledger mismatch, or a
byte-order/memory-order ambiguity is a hard failure.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can one fail-closed harness bind all six canonical row targets, represent paired total values and total scores, preserve actual FD endpoints and all seed-level verdicts, and aggregate exactly five paired seeds? |
| Baseline | Phase 0 row/source freeze and current schema-v4 five nonlinear-row harness. |
| Primary criterion | Before harness edits, canonical target signatures and the six-row Zhao-Cui anchor-availability screen both pass, their hashed gate receipts exist, and the P1-A receipt proves that the Phase 0 harness hash did not drift. CPU-hidden parser/contract/adversarial tests cover all six rows; every full-row aggregate requires exactly one paired value/score record for each frozen seed `81120`, `81121`, `81122`, `81123`, and `81124`, with no extra seed; every seed and parameter direction passes the explicit FD-only policy below and is retained; aggregate total value is the arithmetic mean of the five per-seed total values and aggregate score is the arithmetic mean of the paired five per-seed total scores; LGSSM uses the row-matched compact route. |
| Vetoes | Target or coordinate substitution, missing/mismatched canonical target signature, missing/contradictory Zhao-Cui source availability, unapproved extension/invention required, stale hashes, historical value reuse, non-XLA production fallback, average-value/total-score scaling mismatch, endpoint arithmetic not reconstructed from finite representably distinct same-route endpoints, aggregate-only FD masking a failed seed, mixed source/config shards, missing seed accepted, or source artifact mismatch. |
| Explanatory only | CPU XLA compilation, tiny numerical values, and test runtime. |
| Nonclaims | No trusted GPU execution, row admission, full-time feasibility, HMC/posterior correctness, ranking, or complete leaderboard. |

## Required Checks And Review

- syntax/compile checks;
- all harness tests;
- all six model-specific score contracts;
- cross-model provenance tests;
- endpoint tamper and aggregate missing/duplicate seed tests;
- substituted-five-seed and extra-seed rejection tests;
- within-seed value/score mispair and cross-seed shard/fingerprint mispair
  rejection tests;
- collapsed, nonfinite, wrong-route, wrong-randomness, and wrong-config FD
  endpoint tests that recompute each FD from endpoint scalars;
- wrong-step, off-center, swapped-endpoint-role, and wrong-direction endpoint
  vector rejection tests;
- paired-score theta substitution rejection test;
- target-signature tamper and cross-row/cross-algorithm mismatch tests;
- average-vs-total scalar scaling tests;
- adversarial test where aggregate FD passes but one seed fails;
- exact-command manifest `--check` mode;
- scoped `git diff --check`;
- narrow material implementation review and result/next-subplan review.

The canonical-target generator/checker and focused tests must encode their
expected identities independently enough to reject self-consistent wrong row
labels or hashes. No production framework run occurs in Phase 1; CPU-hidden
TensorFlow is permitted only for deterministic target materialization and
engineering checks, with `CUDA_VISIBLE_DEVICES=-1` set before import.

For every frozen seed, the value and score records must have identical row,
canonical-target signature, source hash, configuration hash, exact-command
hash for that seed, randomness fingerprint, and route fingerprint. Across all
five seed pairs, the row, target, source, configuration, route, and
seed-invariant command-template-family hashes must be identical. The declared
seed, seed-derived randomness fingerprint, output path, deterministic argv,
and resulting per-seed exact-command hash must match the manifest entry for
that one frozen seed and may differ only in those declared seed-specific
fields. Aggregation must reject a substituted five-seed set, an extra seed, any
within-seed value/score mismatch, or any cross-seed
shard/config/template-family/route mismatch even when all scalar values are
finite.

The FD policy is scoped only to finite-difference validation and is not a
general score-error tolerance. For a row with `p` evaluation parameters, where
`p` is exactly the length of that row's frozen evaluation-theta order, each
direction `j` must reconstruct
`FD_j = (L(theta_plus_j) - L(theta_minus_j)) /
(theta_plus_j[j] - theta_minus_j[j])` solely from the two stored endpoint
total log likelihoods and the actual stored parameter separation. Both
endpoint scalars must be finite; the endpoint coordinates must be finite and
representably distinct; and both endpoints must bind the same canonical
target, frozen seed/randomness, source, configuration, route, FD scheme, and
all non-`j` coordinates as the paired score. The paired score record must store
its complete theta vector, and that vector must exactly equal the per-seed
manifest's frozen score-evaluation center before any endpoint arithmetic.
Stored FD values and stored pass flags are ignored and recomputed. The endpoint
records must have explicit
`plus` and `minus` roles, and their complete theta vectors must exactly match
the corresponding per-seed exact-command-manifest vectors for direction `j`.
The manifest must declare the frozen score-evaluation theta as the FD center;
both endpoint vectors must equal that center in every non-`j` coordinate and
must use the manifest's declared signed perturbations in coordinate `j`.
Swapped roles, a wrong step, an off-center pair, a wrong direction, or any
endpoint vector not identical to its manifest entry is rejected before
arithmetic. Every seed and direction must satisfy
`abs(score_j - FD_j) / max(abs(score_j), abs(FD_j), 1e-12) <=
0.05 * sqrt(p)`. This owner-selected `5% * sqrt(p)` rule replaces the
arbitrary historical `0.005` setting; neither threshold may be silently
changed after results are seen.

The exact-command manifest must store deterministic argv arrays, never only
shell strings. Each entry must bind the repository cwd, exact environment and
conda interpreter, row, frozen seed, `N`, `T`, unique output path, canonical
target SHA-256, source and configuration SHA-256 values, GPU device policy,
XLA JIT setting, TF32 setting, FD policy version, FD scheme, frozen direction
set, exact perturbation/endpoints, and command timeout. It must store both the
seed-invariant command-template-family hash and each seed-specific exact-command
hash. Its `--check` mode must rebuild the manifest from current sources and
reject any field/hash drift, missing frozen row/seed entry, duplicate argv, or
output-path collision.

## Forbidden Claims And Actions

- Do not run Phase 2 GPU ladders.
- Do not modify public/default policy or FD thresholds.
- Do not reuse July 6 admitted score labels.
- Do not use July 7 forward values as released LEDH values.
- Do not widen Zhao-Cui or non-LEDH scope.
- Do not overwrite historical artifacts or unrelated dirty files.
- Do not describe P1-B as final source-faithfulness approval.
- Do not admit any row or cell in Phase 1.

## Handoff Conditions

Advance only when the six-row harness and adversarial gates pass, exact future
commands are generated but not run, the result and Phase 2 subplan are reviewed,
and human/trusted execution approvals for Phase 2 are separately available.
Both pre-implementation gate receipt hashes must appear in the Phase 1 result
and Phase 2 handoff.

The Phase 1 result must write a run manifest, decision table, inference-status
table, target-vs-computed classification, post-run red team, review receipts,
and exact artifact hashes. The Phase 2 subplan must inherit the canonical
target and anchor-availability hashes.

## Stop Conditions

- Any canonical target signature cannot be deterministically materialized or
  contradicts the Phase 0 target identity.
- The Phase 0 frozen harness hash has drifted before the P1-A receipt is
  written.
- Any of the six rows lacks the required Zhao-Cui paper/author-source anchors,
  has contradictory anchors, or requires an unapproved
  `extension_or_invention` for the candidate fixed-variant route.
- Phase 0 identity cannot be represented without changing a target.
- LGSSM cannot share the artifact contract without a material public/default
  API change not authorized here.
- Tests expose a shared score/value mismatch requiring mathematical redesign.
- Five review rounds do not converge.
