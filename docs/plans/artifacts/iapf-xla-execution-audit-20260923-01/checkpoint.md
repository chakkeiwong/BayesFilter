<!-- BEGIN_REPLICATION_FIRST_AUTHORITY_20260924 -->
> **Authority update — 2026-09-24:** the [unified iAPF and MDPF/IWSG replication master](/home/chakwong/BayesFilter/docs/plans/iapf-kdm-replication-master-program-2026-09-24.md) now governs the next work. Use its [current checkpoint](/home/chakwong/BayesFilter/docs/plans/iapf-kdm-replication-checkpoint.md). Both scoped replications precede extensions and integration. The next stage is source/reference closure; the old precision-repair queue is deferred. Execution-state text below is preserved historical evidence, not a current launch instruction or live-process report.
<!-- END_REPLICATION_FIRST_AUTHORITY_20260924 -->

# iAPF XLA investigation checkpoint

Question: distinguish compiler wrong results, retracing, NumPy/host loops and the Phase21 timeout. Stage: investigation complete; see result.md.

Checkout: /home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922
Branch ledh-refactor-with-policy-fix; commit89065bc6354801cb368d5e163ba49fd9c4372d10.
Relevant runtime and worker source matches Phase21 snapshot. No runtime edits.

Checked: prior center-reuse TF32 wrong result already repaired. Phase21 FP32 was killed at2100s; compilation succeeded. Fixed signatures and tf.while_loop in the core; Python replicate loop runs ordinary and traced kernels for every stream. No NumPy numerical path found. All8 kernel instances retained1 trace after192 streams per mode. FP64/FP32 diagnostics each passed18 controls.

Evidence: result.md, timing-summary.json, audit-manifest.json, backend-comparison.json.
Diagnostic time49.429s of separate180s cap. Two filter stages77–78% of measured stream time; JSON/gzip about8%. Failed-run FP32 slowdown did not recur and remains unexplained.

Phase21 has3334.712 GPU seconds remaining of7200, two attempts used; FP64 complete; FP32 partial; TF32 not run. Two original2100s reservations would exceed phase balance. No full retry launched.

Owner allowlist request completed: /home/chakwong/.codex/rules/bayesfilter-iapf-execution.rules, plus3 matching live-session approvals. Seven exact command forms passed in allowlist-validation.json. Use git -C <exact checkout> status, rev-parse, or diff --no-ext-diff --no-textconv; reuse direct approved interpreter/read tools. GPU always escalated. New permissions upfront; no broad shell rules added.

Latest owner clarification: first establish two filter replications, iAPF and KDM/IWSG, before extensions and broader integration. See replication-first-status.md for recovered instructions, exact governing plan paths and scope gaps.

Historical session confirms a September18 two-method public-reference comparison request and a September20 explicit iAPF paper-first request. The joint plan completed component checks; its result excludes full author training and the outer adaptive iAPF experiment. KDM has66 matched-target component checks, not a completed original-filter replication. iAPF paper replication remains open: the later score/QR ladder uses alternatives to Equation15, and TF/R local parity does not prove paper identity. No dedicated complete standalone KDM paper-replication plan was identified in the inspected plans.

Master: docs/plans/younis-kdm-score-master-program-2026-09-14.md. CLI: scripts/run_younis_score_master.py; campaign wrapper: scripts/run_younis_score_campaign.py. These schedule the broader study and cannot establish replication by execution status alone.

Next action: explain the two-method priority and map the existing plans accurately. Subsequent planning should give iAPF and KDM/IWSG separate explicit replication completion criteria before extensions. Investigation has not rewritten the master, launched more experiments or expanded a budget.
