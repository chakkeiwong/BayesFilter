# Surrogate-HMC command approvals — 2026-09-23

Owner requested consolidated approvals upfront and commands that match them.
Installed ~/.codex/rules/bayesfilter-surrogate-hmc.rules with six reusable
inspection prefixes: rg -n, rg --files, checkout-specific git status,
git diff --no-ext-diff --no-textconv, git rev-parse, and codex execpolicy check.
Existing approvals, including the absolute conda Python executable, remain.

Nine prospective command forms were checked against both installed rule files;
all returned allow. See artifacts/surrogate-hmc-command-approvals-20260923.json.
This validates rule matching, not correctness or implementation of a proposed
scientific stage. At that September 23 checkpoint, the M13 stage remained to be implemented and reviewed.

Use explicit cwd, the absolute /home/chakwong/anaconda3/envs/tftwogpu/bin/python,
and plain single-line invocations. Put environment configuration and logging
inside repository Python/master code rather than shell assignments, redirection
or variable wrappers. GPU commands still request escalated execution. New
command families need their reusable approvals before the affected phase.

Manually installed rule files load on Codex startup. Reuse the active saved
prefixes during this session; no research restart is required to continue.
Official source: https://learn.chatgpt.com/docs/agent-configuration/rules


Amendment, 2026-09-24: the active rule file now also contains three exact
campaign prefixes:

- /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest
- /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py
- /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m bayesfilter.testing.surrogate_hmc_closure_diagnostics

The exact closure dry-run completed successfully with a 1,260 CPU / 1,200
GPU reservation, previous charged totals 42,156.263748 CPU / 33,507.6772886668
GPU, pass kind m13-repair, and inference_launched=false.

The exact proposed M13 repaired-default replay command, its dry-run form, the
focused pytest command, and the worker form were checked with codex execpolicy
check; every decision was allow and each matched the corresponding narrow
prefix. The original nine checks are preserved in the machine-readable record.
The rules file was edited directly because this Codex installation exposes
execpolicy check but no execpolicy install command. This is a command-matching
result. The owner approved the exact repaired-default command on 2026-09-24;
it ran once. The required 76-test review passed, but the unchanged FP32
regression criterion failed. The command/result and charges are recorded in
the machine-readable approval record and
[saved result](surrogate-hmc-m13-consumer-repair-result-2026-09-24.md).

Follow-up preparation, 2026-09-24: m13-repeatability is now implemented.
The exact master command, its dry-run form, focused-test command, numerical
worker and CPU summary worker all returned allow against the same narrow
prefixes. No additional shell/interpreter prefix was installed. The dry run
passed with a 1,620 CPU / 1,200 GPU reservation; 100 focused CPU tests passed.
The complete command arrays and policy matches are recorded in
artifacts/surrogate-hmc-closure-20260924-repeatability-command-preflight-01/report.json.
The four-process numerical batch is prepared but awaits the owner's approval
under the [comparison plan](surrogate-hmc-m13-repeatability-2026-09-24.md).
The earlier exact replay approval is recorded as executed, not reused.
