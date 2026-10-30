# SIR controller exit-status repair and pending-method recovery

18 October 2026. Local infrastructure repair under the existing SIR evidence
phase and modern-HMC master authorization. No new compute grant or method.

## Diagnosis and question

The immutable numerical supervisor correctly records the killed worker return
code -9 and calls sys.exit(-9). On POSIX its parent observes 247. The outer
controller compared these distinct quantities directly, rejected a legitimate
resource stop, and failed before charging 83767.59229301522 seconds. Numerical
results are terminal and preserved. This is orchestration failure, not evidence
against the sampler or a reason to retry a failed posterior member.

Question: can terminal evidence be audited, charged once, and the untouched
remaining method sequence continued without changing numerical results?
Baseline: existing four-method controller, frozen source e68cdf1c4, GPU 1,
original designs, seeds, membership and posterior criteria.

## Repair and evidence contract

1. Normalize a recorded worker exit into the POSIX launcher's exit byte; retain
   rejection of mismatched or directly signalled launchers. Persist both exit
   codes for new children. Verify with a real tiny subprocess and pipeline tests.
2. Add terminal-prefix recovery into a fresh output directory. Require a stopped
   prior service, identical inputs/prices/allocation/device, independently audited
   terminal prefix, no holes or partial unknown children. Preserve all old files.
   Existing historical launcher code was not persisted: record that limitation;
   recover from strict terminal numerical evidence, never invent that code.
3. Carry old engineering and numerical charges exactly once; reserve the same
   future method shares using existing overrun handling. Never rerun the prefix.
4. Freeze only repaired orchestration and launch randomized HMC, SNAPER, MAMS.
   Check the first admitted child, update master/checkpoint and cost receipts.

Pass: focused success/timeout/mismatch/infrastructure/recovery regressions pass;
real fixed terminal audit passes; exact saved identities/costs reconcile; next
method starts with unchanged numerical source/design. Corrupt/ambiguous evidence,
live prior service, exhausted budget or true infrastructure failure stops launch.
Posterior/reference failure vetoes delivery only; resource censoring is explicit.
No statistical ranking, full-matrix completion or release readiness follows.

## Budget, defaults and skeptical audit

Use the existing 1785.234760601311-second engineering allowance; 38.50532025285065
was spent, leaving 1746.7294403484605 seconds for focused checks and orchestration.
These are inherited reservations, not scientific defaults. Test early that carry
charges reconcile; do not grant a fresh engineering cap. Idle time is not compute.
Numerical caps retain original reservations; the fixed overrun of approximately
0.51 seconds proportionally reduces remaining shares under existing next_cap.
Output: sir-cohort-20261017-01/continuation-04; new orchestration snapshot
continuation-source-03; checks under controller-recovery-20261018-01.

Skeptical audit: do not equate launcher and worker statuses, treat timeout as
scientific failure, hide censored members, charge only successful children,
reprice to favor a method, or resume into old evidence. The proposed tests must
exercise the actual continuation call chain, not just a modulo helper. Original
resource caps are bounded evidence allocations and do not promise completion.
The plan passes for infrastructure recovery only; numerical methods unchanged.

## Execution result — 18 October, 10:21 UTC

Recovery is implemented and the untouched suffix is running. Evidence root
`controller-recovery-20261018-01` preserves 115 focused checkout tests, the strict
fixed terminal audit over 1,890 files, 185 exact-snapshot tests with executable
import/CLI provenance checks, and a successful real saved-prefix recovery.
No numerical replay was needed. Repaired orchestration is frozen in
`continuation-source-03`; all six hashes still match at the latest observation.

Service `bayesfilter-modern-sir-continuation-20261018-04.service` launched at
10:08:58 UTC with `Restart=no`; it is active with zero restarts. Exact command,
environment and watchdog are in `launch-manifest-01.json` and
`launch-submission-01.json`. Output is fresh `continuation-04`. Recovery preserves
fixed's `83767.59229301522`-second charge and its full member denominator.
Remaining numerical pool is `134264.42241627054` seconds before the live child
settles. Randomized cap is `103707.94928282853` seconds; SNAPER and MAMS remain
pending. The deployed worker records GPU 1, XLA and memory growth configured
before logical-device initialization. Its committed qualification chunks grew
from eight to 23; this is progress, not completion.

New prelaunch engineering receipts total `41.66309461835772` seconds; prior
`38.50532025285065` seconds are carried separately within the original hold.
At 10:21 UTC the controller's inclusive live engineering charge was
`97.75383217213675` seconds with `1687.4809284291744` still held. This is an
observation, not terminal settlement. Exclusive submission cost
`.04695963021367788` seconds remains due exactly once at closeout; it is outside
the initial aggregate. Fresh pricing's `6423.1472200709395` seconds is not charged
again. Diagnostic-study funds are separate. The exact live observation is
`controller-recovery-20261018-01/continuation-observation-20261018-1021.json`.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue untouched methods | Regression, real-prefix audit, exact costs/source and first child pass | No corrupt prefix, live predecessor or source mismatch found | Remaining methods may hit their resource caps | Observe, terminal-audit and settle each method; preserve censoring | Full posterior delivery or release readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Fixed member 1 fails reference; member 8 resource-censored. Neither changes tuning membership. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Seven posterior passes and six reference deliveries out of eight verified fixed members. |
| Default readiness | No scientific or numerical default changed. |
| Next evidence needed | Terminal randomized/SNAPER/MAMS inventories and target-specific posterior/reference checks. |

Red-team check: live heartbeats alone could hide a stalled kernel; committed
partial chunks and checkpoint times demonstrate actual progress. Historical
launcher status is unavailable and has not been invented. The repair resolves
orchestration failure, not fixed member 1's reference discrepancy or member 8's
insufficient allocated time. No candidate ranking or full-matrix claim follows.
