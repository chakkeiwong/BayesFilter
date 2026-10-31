> Update, 4 October: attempt 143 was interrupted after source-process memory exhaustion. Its evidence is preserved. The active [memory repair](controlled-smc-kdm-source-memory-repair-20261004.md) and [C2 continuation](controlled-smc-kdm-source-continuation-repair-20261004.md) supersede the launch state below.

# Approved source replication execution, 4 October 2026

The owner approved the prepared 72-hour CPU wall-time reservation. The run
executes the four original-source Figure 6 / Figure 4 right matrices, with
100 repetitions per setting and 12000 cases in total. The remaining campaign
balance before launch is 259293.880125249 CPU seconds; the new grant contributes
259200 seconds. No GPU hours were added or converted.

The scientific specification remains
[the reviewed C1 plan](controlled-smc-kdm-c1-full-source-20261003.md).
It fixes source, data, settings, seeds, comparisons and stopping limits.
C1 establishes source execution and complete accounting. C2 evaluates the
paper comparison with its frozen uncertainty analysis. The TensorFlow
comparison and KDM training remain incomplete; Task 3 / E0 stays downstream.

## Review and permission changes

[The final launch review](artifacts/controlled-smc-kdm-replication-20260928/approved-source-launch-20261003/final-launch-review.json)
passed all seven check groups: authorization and balances, source/dependency
integrity, unchanged generated drivers and 12000-case accounting, six negative
launch conditions, finite/nonfinite/partial progress records, detached launch
wiring, and failed-attempt status. All 52 pinned source/data files and runtime
dependency hashes matched. All four drivers matched the reviewed preparation.
This verifies the launch mechanism; actual source behavior and runtime still
require execution, and MATLAB/Octave numerical equivalence is not established.

The installed narrow rule file is
/home/chakwong/.codex/rules/bayesfilter-controlled-smc-replication.rules.
[Command checks](artifacts/controlled-smc-kdm-replication-20260928/approved-source-launch-20261003/command-rules-checks.json)
passed seven intended forms and rejected three out-of-scope forms against
these new rules. The already active interpreter rule also permits the current
launch. Startup-loaded rules may require a later session restart for new
permissions; this launch does not depend on that restart. No sandbox or
approval safeguards were disabled.

Commands use absolute executable/script paths and separate arguments. The
wrapper manages environment variables, output logs and process lifetime
internally, avoiding shell wrappers, pipelines and redirection:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py start --phase c1_source_full
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py progress
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py report --phase c1_source_full

The detached supervisor retains the existing campaign lock, timeout, process
identity and resource accounting. This is an explicit single-thread CPU
original-source reference run with GPU devices intentionally hidden. Source
code, numerical settings and original results remain preserved. Local
infrastructure repair and retry are authorized within the unchanged scope and
remaining aggregate budget; a retry must use a fresh output directory.

## Original launch record (superseded by the memory repair)

Attempt 143 was launched at 2026-10-04 02:44 Hong Kong time
(2026-10-03 18:44 UTC). Its detached supervisor and worker are alive.
The startup check at 2026-10-03T18:46:45.700552+00:00 recorded
76/12000 completed cases, with
0 nonfinite recorded likelihoods.
This is early execution evidence, not a replication verdict.

[Attempt manifest](artifacts/controlled-smc-kdm-replication-20260928/attempt143_c1_source_full/manifest.json)
and [startup verification](artifacts/controlled-smc-kdm-replication-20260928/approved-source-launch-20261003/startup-verification.json)
preserve the exact command, CPU-only environment, 259200-second cap and progress.
The progress command and checkpoint are the live authorities for subsequent
counts. No new numerical launch should occur while this attempt is running.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Execute the approved C1 matrices | Launch review passed; settings unchanged | No preflight veto | Source outcomes, full runtime, MATLAB/Octave equivalence | Monitor attempt 143; assemble C2 after completion | Replication success, numerical superiority or TensorFlow parity |

After C1, assemble C2 using the predeclared uncertainty procedure before
judging numerical replication. KDM source-setting recovery and author training
remain separate unfinished replication work.
