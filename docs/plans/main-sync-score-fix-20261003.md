# Main score-fix synchronization plan — 2026-10-03

Research question: does the score-computation repair now on main change the
active surrogate-HMC M13 proposal force or its interpretation?

Comparator: this branch at commit 8a4cffa2fd085f43ab5710c24d6e7b48770aea0d,
with its local edits preserved exactly.

Action: preserve local tracked and untracked work, merge main at the fetched
tip, restore the tracked edits with explicit conflict inspection, then run
focused score call-chain/parity tests and a short CPU-only surrogate-HMC smoke
diagnostic. No new long sampling campaign is authorized by this sync request.

Pass criteria: merge completes without loss of preserved files; score source
and the M13 consumer call chain are inspected; focused tests pass or failures
are classified; artifacts record commit, command, environment, and paths.

Vetoes: missing preserved patch/inventory, unresolved merge conflict,
non-finite score/force, changed target/binding without an explicit scope note,
or a test failure that cannot be classified.

Interpretation boundary: a corrected analytical score can alter a surrogate
proposal force, acceptance, and mixing only if M13 calls that repaired path.
It does not by itself prove posterior correctness or convergence; endpoint
target and MH correction must still be checked.

## Merge audit and bounded verification

Skeptical audit: wholesale selection of main for overlapping score files would
remove branch failure-status and structural-innovation interfaces used by the
campaign. Resolve these semantically, keeping compatible main numerical fixes
and local status propagation. The preserved stash and three-way source copies
are the baseline. Tests are engineering/derivative checks, not sampler promotion.
Run CPU-only diagnostics with GPU hidden, up to 20 minutes per focused test batch;
no new posterior sampling. Preserve complete test logs in the sync artifact root.
The KSC observation tangent fix must be traced separately from shared changes: a
KSC-only correction cannot be assumed to explain M13 range-bearing mixing.

## Completion — 2026-10-04

Main merged; original work preserved; integration repaired and checked.
Final validation: 134 passed, 1 skipped, CPU only. Result and exact commands:
docs/plans/artifacts/main-sync-20261003-01/{result.md,run-manifest.json}.
The KSC-specific correction does not directly affect the M13 model callback.
No new sampling run or performance/convergence claim.
