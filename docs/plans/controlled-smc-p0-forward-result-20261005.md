# Source batch and fixed-policy forward result

The scheduled batch is terminal. Both neuroscience source runs completed2000
cases; Lorenz retained its known3537-case prefix and intrinsic covariance
failure. The failure replay reproduced the author outcome without repair.
The source draw capture preserved all returned results and random states
exactly for six cases and22 stages.

All22 TensorFlow GPU/XLA forward stages passed the declared diagnostic screen
and matched the source ancestry exactly. The overall diagnostic remains
incomplete because the failed source policy contains complex/nonfinite values:
its precision was rejected, but end-to-end forward rejection was not tested.
This is missing evidence, not an observed mismatch on the22 valid stages.

## Reporting repair review before execution

The dispatcher dropped comparison_verdict and forward_screen_passed from
terminal manifests. The watchdog consequently could not distinguish an absent
verdict from a failed comparison, though its request for review was justified.
The checkpoint also incorrectly described the terminal source run as ready.

Repair only reporting: preserve comparison fields in future manifests; for
existing manifests, read the checksum-verified summary without editing old
evidence; report terminal source state accurately. The criterion is exact
preservation of the recorded verdict, including false and missing values.
Checksum mismatch must fail closed. No numerical run, threshold, source policy,
budget or acceptance rule changes. Focused tests and live artifact reads will
verify the reporting call chain before refreshing the checkpoint.

## Checked results

| Quantity | Result |
|---|---|
|Source completion|Both neuroscience matrices2000/2000; Lorenz3537/4000 and APF blocked|
|Exact failing source replay|Same positive-semidefinite error in both arms; covariance minimum eigenvalue−0.00458167399141278|
|Valid forward stages|22/22 pass; ancestry identical in22/22|
|Maximum particle error|1.4210854715202004e-14|
|Maximum incremental log-weight error|1.04046193882823e-9|
|Maximum logZ error|1.5279510989785194e-9|
|Maximum ESS error|3.1195881433632167e-8|
|TensorFlow execution|GPU/XLA float64, TF32 off; one trace per forcing and one covariance trace|
|Failed-policy check|Precision rejection true; full forward rejection not evaluated|
|Overall verdict|FAILED_POLICY_FORWARD_NOT_CONFIRMED; full replication incomplete|

These are deterministic comparisons against preserved source draws under the
predeclared diagnostic tolerances (atol1e-7, rtol1e-9). They do not establish
P1 statistical non-inferiority or full fitting parity. The guard outcome is
consistent with the indefinite source precision; the cause of the fitted
complex coefficients remains unchecked by this diagnostic. Preserving an
author failure is acceptable; changing or discarding its complex values to
force this test to run would change the reference input.

|Decision|Primary criterion|Veto/limitation|Main uncertainty|Next justified action|What is not concluded|
|---|---|---|---|---|---|
|Retain valid forward evidence|22/22 source comparisons pass|No extra invalid valid-case output|Only six cases; fixed policies|Complete fitting call-chain comparison|Full P0/P1 acceptance|
|Keep failed-policy check open|Whole-forward failure behavior not evaluated|Complex/nonfinite input cannot be silently replaced|Earliest invalid coefficient and ingress behavior|Trace saved A/b/c and specify explicit source-domain rejection test|Port numerical regression or author-method repair|
|Close scheduled execution batch|All four scheduled phases terminal|Known source failure and incomplete failure-domain comparison retained|Further replication scope still open|Continue from master gaps within a reviewed allocation|Completed CSMC/KDM replication|

|Inference status|Finding|
|---|---|
|Hard veto screen|No invalid result among22 valid forward stages; failed source precision rejected; full failed-policy check missing|
|Statistically supported ranking|None requested or supported|
|Descriptive-only differences|Maximum absolute discrepancies above; no speed/accuracy superiority claim|
|Default readiness|Not established|
|Next evidence needed|Source-to-port fitting and invalid-domain coverage, predeclared P1 margins and repeated-run evidence, KDM source/training qualification|

The reporting repair passed39 focused tests. Live reads through both
dispatcher.progress and monitor.sample now expose the checksum-verified
comparison verdict from attempt148; historical manifests and result files
were not rewritten. Future terminal manifests retain these fields directly.
The regenerated checkpoint no longer calls the source batch ready to run.

Evidence under docs/plans/artifacts/controlled-smc-kdm-replication-20260928:
attempt145_c1_source_full/summary.json;
attempt146_c1_source_failure_replay/summary.json;
attempt147_p0_forward_capture/summary.json;
attempt148_p0_forward_compare/summary.json.
Review tests and live checks: terminal-review-20261005-01/tests.log and
terminal-review-20261005-01/live-report-check.json.

Post-run review: the strongest limitation is frozen-policy coverage: excellent
forward agreement says nothing yet about the learned policy produced by our
fitting procedure. A fitted-policy mismatch or extra invalid result in the
next source-to-port comparison would prevent full replication acceptance.
The terminal notification was submitted but Windows global notifications
remain disabled; no desktop display or user-read receipt is established.
