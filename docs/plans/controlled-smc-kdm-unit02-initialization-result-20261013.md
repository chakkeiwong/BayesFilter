# Unit02 MDPF initialization result

Updated 2026-10-13T16:16:12.779798+00:00. Status: FAILED.

A stage failed; preserve its logs and inspect the first failure.

Evidence: docs/plans/artifacts/controlled-smc-kdm-replication-20260928/unit02-initialization-20261013/attempt01/manifest.json, decision.json, training and evaluation logs.

The intervention changed only initial measurement weights. Source data, source stopping, optimizer and TF random schedule stayed fixed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Limits |
|---|---|---|---|---|---|
| FAILED diagnostic | Controlled attribution, descriptive NLL contrasts | See decision and audit | One fixed training unit | Inspect the first failed stage | No whole-KDM completion |

| Inference item | Finding |
|---|---|
| Hard veto screen | Failed/incomplete; inspect first failed stage |
| Statistically supported ranking | None; draws are conditional on fixed trained checkpoints |
| Descriptive differences | Validation and endpoint means, per-draw values, RMSE and histories are preserved |
| Default readiness | No algorithm/default changed or promoted |
| Next evidence needed | Root-cause interpretation of remaining gaps; population studies are separate from replication |

Post-run red-team: a favorable result can still be specific to this initialization and TF random schedule. A mismatch in any non-measurement parameter or a failed replay would invalidate attribution. The weakest evidence is generalization beyond these fixed development units.

Charged resources: 5115.153827 GPU seconds and 90.740936 CPU-reference seconds. Prior results remain preserved.
