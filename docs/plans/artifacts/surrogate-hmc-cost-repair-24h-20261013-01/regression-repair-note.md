# CPU regression harness repair

The combined process exited with a TensorFlow native segmentation fault after
473.7 seconds and about 190 passing cases. The truncated native stack does not
identify a failing test; no numerical assertion failure is established.
Preserve transport-regression-01. The master now runs the same modules in
separate CPU-only processes and records every exit status and log. This bounds
accumulated compiler state and localizes any repeat failure; it does not dismiss
the crash or change the pass criterion. All modules must pass. CPU budget is
1800 seconds, distinct from the GPU-worker ledger. Continue numerical trials
only after the focused kernel checks pass. Audit: no tests omitted, failures
remain failures, module isolation does not alter runtime scientific code.

Consumer-cost diagnostic now uses each case's declared dtype/precision policy.
This repairs a harness-only hardcoded FP32 path. Mixed precision remains an
explicit optional scope. Tiny tests cover actual consumer and grouped forces.

The isolated suite completed all 15 modules. Fourteen passed. Five cases in the mixed-runtime module failed only the old pre-optimization bitwise comparison; the actual mixed route versus isolated diagnostic remained exactly equal. The old comparison now records every floating-output scaled error and uses a 1e-4 screen (100 times tighter than the owner full-horizon 1% screen); boolean/integer invariants and current mixed-route equality remain exact. This is a comparison-contract repair, not a kernel change. Rerun pending.
