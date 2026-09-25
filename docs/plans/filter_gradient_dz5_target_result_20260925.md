# Frozen DZ5 target integration result

The merged BayesFilter CDF target passes the planned value/analytical-score
comparison against both its graph reference and the independent archived CDF
target at batches 1, 4, 46 and 68 on CPU and GPU. All eight source/device pairs
and four cross-device comparisons pass the original `atol=1e-8, rtol=1e-7` gates.
Every discrete status matches; batch 4 isolates the two original invalid rows
with exact rejection values, zero scores and unchanged valid rows. Initial,
changed and replay calls retain one trace and unchanged HLO without host
callbacks. Loaded-source checks and forbidden-local-runtime import scans pass.

CPU merged runs are 03790/03793/03795/03797, archived runs
03792/03794/03796/03798. GPU merged runs are 03800/03802/03804/03806,
archived runs 03801/03803/03805/03807. The physical device is GPU2,
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`, with verified memory growth and
FP64/TF32-disabled target execution. Every GPU output is device-checked.
Both source trees remain read-only. The snapshot binds repair commit
`9d8202b77`, the unchanged frozen MacroFinance closure, fixture identity
`e116fe853c8579369036ab2ce57724ba07544524d0920fa0a706d76714f75d8a`,
96 observations and all 23 parameters.

| Comparison | Largest fraction of the allowed error | Result |
|---|---:|---|
| Merged versus archived, CPU | 0.000000747 | Pass; other three batch sizes are exactly equal |
| Merged versus archived, GPU | 0 | Exactly equal in saved numerical records |
| CPU versus GPU, either source | 0.0000302 | Pass at unchanged tolerances |

These comparisons include scores, values and chart/support diagnostics; discrete
statuses and input arrays are checked exactly. The post-run analyzer validates
run/JUnit success, snapshot/module/command/HLO checksums, device identity and
growth, and rejects five corrupted-record mutations. The final record is
`artifacts/filter-gradient-repair-20260917/dz5-target-comparison-03808.json`.
Its source and raw evidence, including full HLO and frozen source snapshots, are
preserved in the archives named by `dz5-target-verification-03808.json`.

The unit used all 24 numerical worker slots and 1,929.137249 charged seconds,
including four preserved infrastructure/harness failures and 60 seconds charged
for two driver-only CUDA initialization probes. These were repaired by restoring
the pinned import binary, using the existing numeric adapter, matching the
original metadata-class import sequence, and mounting `/proc` for CUDA thread
naming. No numerical algorithm, tolerance, archive or old admission was changed.
Policy renewal 03808 passes all 129 checks in its separate routine allocation.
Cumulative charges are 75,306.143808 CPU / 75,188.859048 GPU seconds, leaving
11.081627 CPU / 31.114206 GPU hours. No numerical worker remains active.

| Decision | Criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Accept the frozen target execution comparison | Values, analytical scores, statuses and replay pass on both devices | No outstanding mismatch in these fixtures | Broader states, independent fresh score oracle and actual initializer behavior | Finish E5 adapter/initializer/staged-supervisor qualification | Adapter admission, HMC or posterior readiness |
| Preserve the existing admission as stale | Candidate dependency bytes differ from its original qualification | No hash refresh or self-issued admission | Full admission evidence has not been renewed | Complete the stated evidence before any new engineering admission | Eligibility from parity alone |
| Continue memory investigation | Growth/device/allocator telemetry is valid | No whole-device preallocation | These child processes contain both graph and XLA modes | Use matched separate-process cost/capacity units | Memory attribution, leak freedom or speed ranking |
| Keep main promotion blocked | New remote execution violations and remaining integration gates are open | No terminal F01--F20 closure | New LEDH call chains are only partially audited | Execute the linked merged-endpoint repair unit | Repository-wide compliance |

Review: the strongest misleading explanation would be both implementations
sharing the same derivative defect. Independent archived bytes, changed operands
and device comparisons constrain execution changes but do not replace a fresh
value-difference/closed-form score oracle for scientific admission. The original
CDF five-point oracle is preserved as historical engineering evidence; this unit
does not silently renew it. The next admission check must repeat that oracle on
the changed closure. The weakest integration evidence remains the complete
public initializer and supervised lifecycle. No learned NeuTra map is trained
or promoted, and all superseded recipes retain their historical status.
