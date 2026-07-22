# NeuTra HMC Program Phase A Missed-Item Repair Record

Date: 2026-07-15  
Status: `REPAIRED_AND_CLOSED`

The terminal local audit found closeout omissions rather than failed scientific
runs.

| Omission | Classification | Repair | Verification |
| --- | --- | --- | --- |
| Master and completed subplans retained ready-state labels | documentation drift | Marked C1, C2, S1, F0, F1, and F2 closed; master and Phase A now show terminal-audit state | status scan |
| F0 fixture identity ledger lacked a stable self-hash | evidence metadata omission | Added `artifact_hash` computed by `bayesfilter.runtime.stable_config_hash` after excluding both hash fields | recomputation must equal `sha256:1936ff2a29f46d60930ae2c02bde850da82ef364e63625922fac74f61ffabe56` |
| F1 facts were distributed across four immutable GPU result files and selection ledger with no consolidated command manifest | provenance consolidation omission | Added `phase-a/serious_run_manifest.json` with environment, seeds, hardware, wall time, result paths, file hashes, artifact hashes, plan, and result links | JSON parse, file-hash audit, and manual cross-check |
| F1 result artifacts did not store command strings | residual provenance limitation | Reconstructed exact invocations from the frozen CLI and artifact routes, and labeled each as reconstructed rather than contemporaneous | CLI argument audit; no scientific rerun performed |
| Crash handoff reported duplicated C2 prose | unconfirmed report | Scanned current master; duplicate was not present | no edit made |

The repairs do not change target, data, method, weights, samples, diagnostics,
thresholds, kernels, decisions, or scientific claims. Re-running completed GPU
training solely to create a shell transcript would add cost without improving
the already hashed numerical evidence, so it was not done.
