# Saved precision and tail diagnosis, 28 October 2026

The two successful audits check saved scalar evidence, not raw trajectories.
precision-01 preserves the failed reader; precision-02 checks 1,152 identities
and all 288 generated validation endpoints. tail-01 checks 1,792 records and
5,376 profiles. No failed precision result or original interval is changed.

Results use deterministic gzip compression (mtime=0); decode using Python's
gzip module. The hash inventory covers compressed files and manifests.
Commands and source hashes are preserved in each manifest. CPU-only reference
execution used tfgpu Python; no sampler or GPU workload was launched. Saved
seeds and data identities are inherited from the input compact records.

See [the plan](../../plans/bayesfilter-modern-hmc-c4-saved-precision-audit-2026-10-28.md)
and [result](../../plans/bayesfilter-modern-hmc-c4-saved-evidence-result-2026-10-28.md).
The 1,792 original raw paths for the tail study are absent. These scalar checks
cannot reconstruct or replay those trajectories. The variance counterfactual
is explanatory only, unavailable as a real-world estimator, and establishes no
interval calibration, ranking, posterior validity, default or release claim.
