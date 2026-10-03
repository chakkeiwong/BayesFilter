# Derived replay diagnostic result

04654 correctly derives the replay count while removing its variable, but
does not restore the original score. The first score still differs from the
historical original by1.5276668818842154e-13 (maximum1952 ULPs on the reported
small component). Against the unmodified one-iteration candidate04636, every
callback position/value/score/validity byte and every returned record field
matches exactly. The original04635 short record retains47 differing leaves;
callback positions, values and scores differ after the identical first inputs.

The probe has one trace, three optimizer objective batches, no host callbacks,
and no captured int32 resource. Counts remain3 attempts/3 optimizer calls/
2 replays, equal to both controls. Exact five-site AST change, source/HLO/
callback hashes, all callbacks and complete short records pass the diagnostic
readback;04655 passes161 evidence/policy checks. The unit closes at2/4 workers,
233.688754/1200 CPU process-seconds. This is a valid negative remedy result,
not a harness failure or a reason to reject the analytical filtering method.

| Decision | Primary criterion | Veto | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain the derived-count diagnostic only | Count identity and all candidate short records agree | Original score/record restoration fails | Exact optimized-code difference is unknown | Compare optimized HLO/lowering of the same genuine one-iteration controls, including the positive replay-int32 intervention | GPU-safe runtime remedy, full trajectory or convergence |
| Keep the master open | All new evidence/policy checks pass |121 strict full-trajectory leaves and other listed gates remain open | Broader consumer/cost applicability still needs review | Follow `filter_gradient_terminal_gap_queue_20260928.md`; preserve completed cohorts | Whole-program completion or main merge |

Post-run skeptical review: the strongest alternative explanation remains
whole-graph optimization/scheduling affected by resource signatures and this
observer context. Neither a positive int32-storage probe nor a negative
resource-elimination probe identifies the exact compiler arithmetic. A
source/input mismatch would invalidate this attribution; a later demonstrated
GPU-safe, complete-record remedy would supersede this candidate rejection.
Only the frozen r1 CPU fixture was tested. Runtime source, tolerances, allowlist
and canonical architecture remain unchanged; no independent reviewer was used.
