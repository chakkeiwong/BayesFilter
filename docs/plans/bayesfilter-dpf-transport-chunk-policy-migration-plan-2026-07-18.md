# DPF Transport Chunk Policy Migration Plan

Date: 2026-07-18  
Status: execution authorized by owner request

## Research intent ledger

| Field | Contract |
|---|---|
| Main question | Can active DPF Contract E canonical/candidate routes be made unable to reuse the superseded tiny-fixture chunk settings? |
| Mechanism | One repository-owned exact-divisor selector, enforced before graph construction and guarded by focused and discovery tests. |
| Expected failure mode | A lower-rung fixture constant or CLI default is copied into a larger run, causing an unnecessarily large `(N/K) x (N/K)` block grid and misleading runtime evidence. |
| Promotion criterion | Every active Contract E canonical/candidate boundary selects or validates the one policy, and focused tests cover the required examples and rejection behavior. |
| Promotion veto | Any active claim-bearing route can construct with a conflicting row/column chunk, or any independent active tiny-chunk default remains. |
| Continuation veto | The evidence anchors do not support the rule, or enforcement would change transport mathematics rather than only exact tiling/traversal. |
| Repair trigger | A focused or discovery test exposes an unguarded active route or stale hard-coded setting. |
| Explanatory diagnostic | Source inventory of chunk arguments and constants. It cannot establish numerical or scientific correctness. |
| Nonclaims | This migration does not establish score correctness, Sinkhorn convergence, HMC readiness, GPU memory safety, posterior correctness, or leaderboard completion. |

## Evidence contract

- Exact baseline: the current active behavior, including `N=128,K=16` and
  `N=1024,K=16` descendants of the Phase 8 lower-rung fixture.
- Required policy: for particle count `N` and row/column chunk extent `K`,
  require `K | N`, equal row/column extents, `K=N` for `N<=3000`, and for
  `N>3000` the largest divisor of `N` not exceeding 3000. If the only eligible
  divisor is 1, fail closed.
- Required witnesses: `N=1000 -> K=1000`, `N=1024 -> K=1024`,
  `N=10000 -> K=2500`, and `N=10240 -> K=2560`.
- Artifact: this plan, the migration note/result, source changes, and focused
  test output.

The timing basis is preserved in:

- `docs/plans/bayesfilter-ledh-pfpf-ot-autodiff-free-adjoint-n10000-xla-chunk-ladder-result-2026-06-24.md`;
- `docs/plans/bayesfilter-ledh-pfpf-ot-autodiff-free-adjoint-n10000-xla-chunk2500-result-2026-06-24.md`;
- `docs/plans/bayesfilter-ledh-pfpf-ot-autodiff-free-adjoint-n10240-xla-chunk2560-binary-boundary-result-2026-06-24.md`; and
- the generic and SIR chunk comparisons cited by the migration result.

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Cap 3000 | June 24 GPU/XLA chunk ladder and exact-tiling results | The best tested exact chunks were 2500 and 2560; larger 3334 was slower | Hardware/model transfer may not be universal | Keep policy versioned and do not claim universal optimality | Evidence-backed repository policy |
| Exact divisibility | Exact-tiling timing artifacts and owner clarification | Removes padded blocks and defines the grid exactly | A prime `N>3000` has no useful divisor | Selector rejection test | Reviewed default |
| Equal row/column chunks | Owner clarification and square transport layout | Prevents asymmetric independent policy drift | A future kernel may benefit from asymmetry | Require a new reviewed policy version rather than a local override | Reviewed default |
| Static selection before tracing | XLA compilation policy | Avoids dynamic policy logic in the graph | A dynamic `N` cannot be selected safely | Reject unknown/static-invalid `N` at route construction | Engineering requirement |

## Skeptical pre-execution audit

- Wrong baseline: checked. The offending `K=16` values trace to an explicitly
  lower-rung-only `N=32` comparison, not to the June GPU timing evidence.
- Proxy promotion: checked. Timing evidence supports chunk configuration only;
  no score, posterior, or scientific claim is inferred.
- Missing stop conditions: checked. Stop if the selector cannot bind static
  `N`, if enforcement changes mathematical outputs beyond traversal roundoff,
  or if active routes remain independently configurable.
- Unfair comparison: not applicable to the source-enforcement step; no new
  method ranking is performed.
- Hidden assumptions: the 3000 cap, exact divisor rule, and static-`N`
  requirement are explicit above.
- Stale context: checked against active source on 2026-07-18; seven active
  benchmark/default violations were found, not merely the four newest LGSSM
  arms.
- Environment mismatch: focused enforcement tests are deliberate CPU-only
  checks with GPU hidden. No GPU performance result will be claimed.
- Artifact sufficiency: selector tests plus an active-source discovery guard
  directly answer whether the regression can re-enter the covered routes.

Verdict: PASS. The plan may execute.

## Execution phases

1. Add binding policy text to `AGENTS.md` and reviewer guidance to `CLAUDE.md`.
2. Add a repository-owned selector and fail-closed validator.
3. Enforce it at canonical/candidate factories and prepared-input boundaries.
4. Replace independent active constants/defaults with selector output.
5. Add behavioral and source-discovery regression tests.
6. Run focused CPU-only tests and write the migration result.

## Stop conditions

- Stop without promotion if a conflicting chunk still reaches an active
  canonical/candidate graph.
- Stop and report if tests reveal that exact-divisor traversal changes the
  mathematical target rather than only block traversal/roundoff.
- Preserve old artifacts byte-for-byte. They are archival provenance only and
  are explicitly ineligible to govern, diagnose, compare, or support new runs.
