# Complete High-Dimensional Leaderboard Launch Implementation Review, Iteration 5

Date: 2026-07-12

Reviewer: fresh Codex read-only substitute after Claude was classified
unavailable. This is weaker evidence and grants no execution or scientific
authority.

Reviewed surface: schema-v6 detached launch control, exporter verification,
watchdog, finalizer, isolation preflight, and focused tests.

## Material Findings

1. `BLOCK_PRIMARY_EXPORT_COMPLETENESS`: supervisor, watchdog, and finalizer
   validate records listed in the export hash ledger but do not require the
   complete primary payload set. An empty or partial `files` list can pass. The
   watchdog fixture currently accepts a ledger containing only a `{}` manifest.
2. `BLOCK_SEAL_TO_REMOUNT_TOCTOU`: the finalizer hashes the handoff and writes a
   seal claiming post-seal writes are forbidden, then returns. The outer shell
   chmods/remounts aliases read-only afterward without revalidating every sealed
   byte during that interval. A mutation through the host-visible alias can
   make the written seal stale before lock enforcement.
3. `BLOCK_PREFLIGHT_SCOPE_MISMATCH`: the trusted preflight validates the inner
   Codex/GPU namespace, supervisor, primary export, and namespace closure using
   a synthetic preparer and direct external launcher. It does not execute the
   exact wrapper, outer boundary, production preparer, watchdog, or finalizer.
   It cannot clear the repaired outer export/seal controls; those currently have
   local tests only.

VERDICT: REVISE
