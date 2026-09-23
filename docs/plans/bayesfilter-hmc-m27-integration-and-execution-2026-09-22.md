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

## September 23 execution record and audit correction

Merge `0643b0adc` is pushed to main. The clean integration tree plus the five
M27 validation modules and the GPU study harness form source-r1 at
`/tmp/bayesfilter-hmc-m27-source-r1-20260923`; `source_snapshot.json` binds every
Python module. Shared Q20/training/AGENTS edits are excluded. Integration tests
passed 150 cases; the sibling/fit/doc checks passed 69, then 29 focused cases.
These overlapping counts are not added as distinct tests.

The concrete designs are in `m27-r1/rotated-sibling-suite.json` and
`m27-r1/m27-gpu-{gaussian,beta-binomial}-lugsail-202609228{3,4}.json`.
CPU seeds 2026092281/82 and GPU seeds 2026092283/84 are convenience identifiers,
not optimized seeds. The GPU design is identical between persistent and isolated
execution. The CPU suite uses the public plan/run CLI with one worker; GPU
pairs use `docs/benchmarks/run_hmc_fit_process_study_2026_09_23.py` with the
same design, `--mode persistent|isolated`, and fresh nested `fits/` roots.
The meter writes the exact command/environment/source in each attempt manifest.

The first CPU attempt `rotated-sibling-r1` failed before numerical work because
the meter-created output directory was also supplied as the CLI's fresh root.
The nested `fits/` repair succeeded in r2. A sandbox process-list observation
was then wrongly interpreted as worker interruption; r3 duplicated r2 while
r2 was still running. Both exited normally. There were at most two numerical
workers, all costs are charged, and r3 is a replay rather than a new fit.
Terminal receipts, not absence from a sandbox PID namespace, determine exit.

The generated CPU suite also mistakenly declared warmup maximum 10000 instead
of the plan's 30000. Preserve this deviation in the original design. Every
selected member stopped successfully at the first 10000-transition readiness
check, so these observations answer the bounded sibling-delivery question.
They do not test continuation through 30000 warmup transitions. No retrospective
change to the frozen design or rerun is needed to report those limited facts.
The 10000 minimum/window, 5000 chunk, 30000 retained minimum and 60000 retained
maximum are the plan's explicit allocation hypotheses, not promoted defaults.

Each fit retained every verified member and selected L=3 and L=4 before sampling.
The independent slot summaries passed, but the aggregate `finding` remained
`incomplete`: the reused single-member comparison sorted an unassessed lower
candidate ID before the selected IDs. Repair the summary filter and test an
unassessed ID that sorts first. Save corrected summaries in a new audit file;
keep original run artifacts and source-r1 unchanged. No tuning or posterior
calculation changes. Freeze source-r2 for the GPU experiments after this repair.

Skeptical re-audit: never pool duplicate runs or siblings as independent fits;
never claim the unused 30000 warmup cap was exercised; and never overwrite a
run to conceal a reporting error. Both GPUs pairs still require exact tensor,
receipt and candidate parity, verified growth/placement, and normal exits.
The unchanged M27 ceilings leave about 1106 CPU seconds after recorded work,
before terminal audit/tests, and 4800 GPU seconds before readiness allowance.
Reserve 60 GPU seconds for earlier unmetered readiness probes and cap each
new GPU execution at 1180 seconds (1160 fit timeout) so four executions plus
probes remain within 4800. These are resource ceilings, not scientific criteria.

This tranche tests implementation, delivery and cost. It cannot promote a
selection policy or rank samplers. Its constructed sanity comparators are the
independent fixed-discard/fixed-retained arm for each member, exact analytic
means/medians for each target, and the matched persistent complete-fit process
for GPU isolation. Report them separately by model and selected slot; aggregate
averages cannot conceal unavailable or failed cells. Any future method-quality
promotion must add target-specific cheap sampler baselines and sufficient
replication; current heuristic-dominance status is not established.

## Terminal reconciliation

Both GPU pairs have exact candidate, receipt and saved tensor parity and normal
exits. The final saved-evidence audit passed for both CPU fit identities and
both GPU pairs. The final result is
[the M27 result](bayesfilter-hmc-m27-sibling-and-gpu-result-2026-09-23.md);
`m27-r1/reconciliation-terminal.json` records every charge, including the duplicate
and test timeout. No M27 numerical worker remains. The reviewed
[post-M27 program](bayesfilter-hmc-post-m27-next-phase-2026-09-23.md) is the
active next-phase design. No scientific coverage or default was promoted.
