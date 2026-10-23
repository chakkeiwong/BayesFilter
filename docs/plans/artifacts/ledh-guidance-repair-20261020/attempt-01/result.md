# Phase A: guided LEDH implementation repairs

20 October 2026. COMPLETE on `ledh-refactor-with-policy-fix`, base
`4d6915cc66e701c4c7b657e094306d298fdd3e76`, with preserved working changes.
Plan: `docs/plans/ledh-guidance-repair-plan-20261020.md`.

The physical consumer now reports actual guide/proposal validity. Both guide
consumers retain the Gaussian-product conditioning margin and a separate
measurement mask through the moment endpoint. The campaign assembler validates
the complete paired grid before inference. Campaign startup snapshots every
Python source under `bayesfilter`, including untracked sources, alongside the
Git diff/status; finalization detects source changes. These are reporting and
evidence repairs. Accepted proposal, reset and analytical-score formulas did
not change.

## Checks and reproducibility

- `cpu-tests.log`: 59 focused tests passed in 153.76 seconds; two upstream TFP
  deprecation warnings. GPU devices deliberately hidden with
  `CUDA_VISIBLE_DEVICES=-1`. Interpreter: `tftwogpu/bin/python`.
- `check_healthy_values.py`, `healthy-before.json`, `healthy-after.json`: three
  healthy consumers have bitwise-identical CPU values and analytical scores.
- `check_saved_result.py`, `saved-result-check.json`: the original 544 rows
  remain complete and paired, with unchanged decisions and summaries within
  CPU/GPU reduction roundoff (maximum difference 1.1102230246251565e-16).
  Missing, duplicate and unpaired copies are rejected.
- The initial historical-summary diagnostic required exact CPU/GPU equality
  and failed. Its retry uses 128 double-precision epsilons times `1+abs(value)`;
  discrete decisions remain exact. This changes only that independent check,
  not any algorithm tolerance. Both logs are preserved.
- `check_gpu.py`, `gpu-check.json`, `gpu-manifest.json`, `gpu.log`: escalated
  GPU 0, FP64, TF32 disabled, verified memory growth, shared consumer under XLA.
  Healthy flags pass; inadmissible guides reject. Graph/XLA discrepancies are
  8.88e-16 in value and 2.22e-16 in score. Measured/inactive margins remain
  distinguishable. The complete package snapshot in `source/` stayed unchanged.

Commands, source provenance, device, environment and paths are retained in the
scripts, logs and GPU manifest. GPU main elapsed time was 40.796203379984945
seconds. Charge 60 GPU-process seconds conservatively including import/setup;
remaining parent allowance is **23766.197895057 seconds**. Peak measured TF
allocator use was 9515008 bytes. No training or new efficacy evaluation ran.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the engineering repairs | Consumer regressions, unchanged healthy calculation and strict evidence validation pass | No unresolved defect in the tested paths | Small fixtures do not cover every model | Reconcile the learner with the mathematics | Better moments or scores |
| Preserve the earlier negative result | Original grid remains complete and paired | No new invalidity found in the saved rows | Missing historical source provenance cannot be reconstructed by assertion | Keep the exposed holdout read-only | Retroactive source certification |
| Continue the learning design | Reporting defects do not explain unsuccessful training | No research-direction veto | Native KDM conditioning and SIR adaptation remain to implement | Phase B and a fresh Phase-C protocol | Completion of the larger integration |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Invalid guides and malformed evidence reject; healthy checks pass |
| Statistically supported ranking | None evaluated in this repair |
| Descriptive-only differences | Floating-point reduction differences only |
| Default readiness | Numerical defaults unchanged; no new efficacy claim |
| Next evidence needed | Real learner integration, fresh conditional moment evaluation and separate downstream score evaluation |

Post-run red team: the earlier unsuccessful training remains unexplained by
these reporting defects. A three-state bearings filtering density cannot be
inserted as an eighteen-state SIR future-likelihood message. That distinction
requires a specified adaptation, not a renamed import. The user has now chosen
a SIR-specific neural KDM proposal, explicitly an adaptation of the replicated
method; Phase B must define its conditioning and full sampling density.
