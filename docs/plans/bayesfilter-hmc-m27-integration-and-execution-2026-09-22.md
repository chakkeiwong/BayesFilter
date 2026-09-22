# M27 integration and execution note

The owner requested commit, remote integration, conflict resolution, push to
main and continued execution. Publish the completed M25/M26 HMC changes and
their compact evidence, preserving unrelated Q20/training/governance work in
the shared workspace. Raw tensors, checkpoints and full source snapshots stay
in the versioned local evidence directories; committed manifests and audits
identify them. Merge remote main before freezing M27 numerical source.

The active scientific plan is
[the post-M26 program](bayesfilter-hmc-post-m26-next-phase-2026-09-22.md).
The opening ledger leaves 79635.40248619507 CPU and 84803.5201060148 GPU
worker-seconds. M27 uses at most 3600 CPU and 4800 GPU seconds, including
integration tests, failed attempts and teardown, with at most two numerical
workers concurrently. No compute allowance increases.

Skeptical integration audit: whole-workspace staging would mix unrelated work;
select the HMC source, tests, documentation and compact result records
explicitly. Preserve both sides of numerical conflicts and run the affected
tests before publication. Use an isolated integration worktree if remote
changes overlap unrelated dirty files. No stale frozen experiment is rerun or
relabeled as merged-source evidence. The git commit is provenance, not evidence
that numerical or statistical checks passed.

Skeptical execution audit: the new sibling rule must nominate its fixed list
before posterior data exist, preserve every verified member, keep member
results separate and count complete fits as independent replications. A
failed first sibling must not suppress the second. A list shortfall and a
missing posterior stay in their declared denominators. R-hat, ESS and MCSE
remain posterior-only requirements. GPU pairs need the same source, numerical
design and device settings, ordinary process exits and exact tensor parity.
Neither development delivery nor successful GPU execution establishes coverage
or a default policy. The bounded plan is suitable for execution once its
concrete designs and source are frozen.
