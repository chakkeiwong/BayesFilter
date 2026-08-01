# Codex Substitute Review: Phase 9 Gate A And GPU Manifest, Iteration 1

Date: 2026-07-10

## Scope And Limitation

Fresh local read-only review of the Gate A result, shared runner/tests, five
tensor-only adapters, Phase 9 subplan, and proposed GPU execution manifest.
Claude review remains policy-blocked as external repository disclosure. This
review did not execute a GPU/CUDA command.

## Findings

| Severity | Finding | Required repair |
| --- | --- | --- |
| Blocking | Gate C/D were described by substitution templates rather than a complete exact command set, contrary to the post-Gate-A review requirement. | Generate a deterministic frozen command artifact covering every B/C/D/aggregate argv and path; parser-test it and bind it into shard provenance. |
| Blocking | Score-only and FD-only rebuilt fixed observations/randomness in separate processes, but artifacts did not prove those tensor inputs were identical. Same seed/shape metadata alone was insufficient. | Serialize and hash every prepared tensor leaf, record the tree fingerprint, require FD to match its score reference before evaluation, and revalidate during aggregation. |
| High | Dirty-worktree disclosure did not content-address all local data/model/transport dependencies; helper edits could change a run without changing HEAD. | Recursively hash local imported modules and freeze hashes for process lifetime. |
| High | Trusted execution accepted parser-equivalent commands outside a reviewed literal command line, including alternate output/device variants. | Enforce the frozen command artifact, repository root, CUDA/device fields, and literal argv. |
| Medium | Source-value/governance hashes could be recomputed after a long process started, allowing mid-process file changes to alter provenance. | Freeze source, code, governance, and command hashes at process start and validate against the frozen snapshot. |
| Medium | Shell `python ... | tee ...` could mask a Python failure without `pipefail`. | Use direct stdout/stderr redirection so the shell exit status is the Python runner's. |

These issues blocked trusted Gate B execution even though the initial CPU-hidden
tests passed. They were harness/artifact-identity defects, not evidence against
the compact score recurrence or any row.

## Nonclaims

No GPU viability, memory, FD correctness, score admission, HMC, posterior, or
scientific claim follows from this review.

VERDICT: REVISE
