# Codex Substitute Re-Review: Phase 9 Gate B Result, Iteration 2

Date: 2026-07-11

## Scope And Limitation

Fresh local read-only re-review after the iteration-1 Gate C decision-provenance
blocker was repaired. The review inspected the complete Gate B result, final
live shards and hashes, runner validators, new Gate B result/review bindings,
adversarial tests, and frozen Gate C commands. Claude remains policy-blocked as
external repository disclosure. No Gate C, Gate D, aggregate, or LGSSM command
ran during this review.

This verdict authorizes Gate C only for fixed-SIR, actual-SV, generalized-SV,
and KSC-SV. Predator-prey is excluded because its frozen Gate B FD rule failed.
For each eligible row, execute the frozen Gate C score/FD pairs in ascending
prefix order. Inspect score before FD and each completed pair before the next
prefix. A row-local hard failure stops that row. Gate D, aggregation, and LGSSM
remain blocked.

## Iteration-1 Blocker Resolution

| Blocker | Resolution checked |
| --- | --- |
| Future prefix shards would not bind the Gate B result that determines row eligibility. | `GATE_B_RESULT_PATH` is included in governance SHA-256, emitted in every manifest, and required by common shard validation. |
| Future prefix shards would not bind the review authorizing Gate C. | `GATE_B_RESULT_REVIEW_PATH` identifies this exact review and is included in the same hash and validation contract. |
| A relabeled or edited decision artifact could pass unnoticed. | Independent adversarial mutations cover result path, review path, result hash, and review hash. The focused governance suite passed `35` tests. |
| Provenance hardening could alter frozen experiments. | Exact-command manifest `--check` passes. No Gate C argv, target, seed, prefix, chunk, transport, precision, memory budget, FD step, or tolerance changed. |

## Gate B Decision Assessment

The Gate B result is correct and appropriately bounded:

- fixed-SIR passes its declared OR rule by relative tolerance;
- actual-SV and generalized-SV pass by absolute tolerance;
- KSC-SV passes by both tolerance branches;
- predator-prey fails both branches and its terminal `failed_fd` artifact is
  correctly excluded;
- all score shards and four passing FD shards validate through the runner's own
  acceptance code;
- score-reference hashes and prepared-input fingerprints match;
- all live shards have common GPU/XLA/TF32/trust/source/review identity;
- no shared continuation veto fired.

The predator-prey zero `a` FD makes float32 cancellation plausible, but that is
explanatory only. It does not override the frozen rule, and no post-result
threshold or step change is authorized.

## Gate C Evidence Contract

- Gate C remains a one-seed `N=10000` prefix feasibility/correctness ladder,
  not full five-seed admission.
- Every score must be terminal, finite, trusted GPU/XLA/TF32, source/review
  matched, and at or below `14000 MiB` reset peak before FD.
- Every FD must reference that exact score, match prepared inputs, and pass the
  frozen row rule before the next prefix.
- A prefix failure is a row-local veto unless it exposes shared artifact or
  harness invalidity.
- Full-time seed `81120` passage does not authorize Gate D until that row's Gate
  C result receives the separately required review.

## Verification

- Focused Gate B decision-provenance tests:
  `35 passed, 53 deselected, 2 warnings in 3.19s`.
- Full harness/cross-model/shared contract with all-row XLA:
  `161 passed, 2 warnings in 89.80s`.
- Syntax, exact-command currentness, and `git diff --check` pass.

CPU-hidden tests are engineering evidence only. Gate B tiny results and future
prefix results do not establish full score admission, posterior correctness,
HMC readiness, statistical ranking, runtime superiority, native actual-SV
correctness for KSC, or broad scientific validity.

No material blocker remains for the frozen Gate C ladders of fixed-SIR,
actual-SV, generalized-SV, and KSC-SV.

VERDICT: AGREE
