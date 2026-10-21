# Active surrogate-HMC checkpoint: main synchronization completed

## KSC prioritization question — 2026-10-04

The user asks why the same KSC model cannot be used. It can: KSC is already
within the original all-model campaign. Its existing T2 implementation has
one state, one observation and two parameters, with a 49-component independent
mixture reference. Important distinction: the branch KSC consumer replaces
actual_sv_model's observation density/mean but retains its parameter-observation
tangent dx + 2*d_beta/beta, which already includes the direct beta term.
Main's repaired ksc_sv_canonical_model uses a separate callback and coordinates
(probit gamma, log_beta). A test of that specific repair must verify the
consumer call chain and match the target and coordinate transformations;
rerunning the old KSC consumer alone does not demonstrate the fix's effect.
No sampling or implementation change was launched in response to this question.
Source: bayesfilter/testing/surrogate_hmc_ksc_consumer.py::ksc_model;
bayesfilter/testing/surrogate_hmc_sv_consumer.py::actual_sv_model;
bayesfilter/highdim/ledh_canonical_models_tf.py::ksc_sv_canonical_model.

Active request: sync main's score repair into this branch and assess its
relevance to surrogate HMC. Main a925f67a19d8c5d265b8ff71bfd2c78ac1964ff4 is
merged as a55c6be7a16e9401242ee3449248fa347eb58d90. Original branch work is
preserved in stash@{0}, the tracked patch, and the untracked inventory under
docs/plans/artifacts/main-sync-20261003-01/. Do not drop that stash.

Checked finding: b7ed96ec7 corrects the KSC observation tangent by including
2*d(log_beta). The active M13 range-bearing target uses a different callback.
Its proposal force comes from canonical_batch_fused_value_score, and its
acceptance target remains the separate frozen importance likelihood.
No measured acceptance, mixing, or whitening improvement follows from this sync.

Integration repairs preserve local failure-status/structural interfaces plus
main's SQMC ancestry, stage diagnostics, and higher-moment safety behavior.
Final score/safety and integrated M13/structural/batch/oracle/dispatcher checks:
134 passed, 1 skipped. All pre-existing untracked entries still exist.
Result and manifest: docs/plans/artifacts/main-sync-20261003-01/{result.md,run-manifest.json}.
The sync request is complete. Next action: report the model-specific impact.
No further computation is pending under this synchronization task.

Plan: docs/plans/main-sync-score-fix-20261003.md. CPU-only diagnostics hide GPUs;
20-minute cap per focused batch. No new sampling or default-policy promotion.
The sampling allocation expired 2026-10-01T16:00:00Z; 12,750 retained draws per
chain exist, with a 20,000 draw ceiling. No current sampling job.

Previous completed warmup-covariance result:
docs/plans/artifacts/surrogate-hmc-m13-warmup-covariance-check-20261002-02/result.md.
The old checkpoint is preserved as pre-sync-active-checkpoint.md in the sync
artifact directory. All-model campaign remains incomplete; no Claude review pending.
