# Confirmation monitoring and bounded resource recovery

The user requested unattended health checks, completion/error reports and repair
of recoverable failures. The primary 96-slot source-23 confirmation is already
running. At inspection, two QR searches completed, two other slots deferred
before TensorFlow import because foreign GPU processes occupied the selected
device beyond the frozen 60-second readiness cap, and the next nonlinear
search was progressing. No numerical failure was observed in the two completed
searches. The competing PIDs have exited.

The existing supervisor owns timeouts, per-slot readiness and the fixed outcome
denominator. It has no separate observer, completion notification or generic
restart support. Add an independent local observer without editing the running
supervisor, numerical source, design or original outcomes.

## Evidence contract and permitted actions

The engineering question is whether completion, stalled progress, malformed
results, source/memory-policy violations, resource deferrals and service exits
are detected and reported while preserving the frozen experiment. The baseline
is the running tested supervisor. Focused fake-process/file tests must show
correct classification, bounded retries, unchanged seed/configuration, retained
original failures, and no recovery while the primary process remains active.
These tests are engineering checks, not sampler-validity evidence.

Every minute, inspect the user service, original outcome inventory, current
worker checkpoint/stage timestamps, free storage, and completed-slot identity,
memory policy and member inventory. Source/runner/design hashes must match.
Checkpoints with no recent change are explanatory warnings, not automatic
numerical failures: use the inherited 1,200-second closeout allowance as the
warning interval. Existing process and campaign caps retain termination
authority. Foreign processes are never stopped or modified.

An invalid source/design, malformed completed evidence or memory-policy
violation is an infrastructure/evidence veto and stops the owned primary
service. An observed failed candidate or statistical criterion does not stop
the research direction. Disk exhaustion is an infrastructure stop: use twice
the largest measured family allocated footprint as a conservative free-space
floor, recorded as a capacity hypothesis, not a model default. Report larger
projected storage pressure without treating a forecast as an automatic veto.

Write atomic `monitoring-01/status.json` and `status.md`, plus an append-only
event log. Report new failures and terminal state using desktop notifications
when the local notification service is available. Notification failure must be
recorded and cannot kill the numerical run. This session has no callable
automation or chat-notification tool; a local observer cannot promise to create
new messages in this chat. The official automation documentation fetch returned
HTTP 403, so no unsupported product-level notification claim is made.

## Recovery boundary

After the original service has terminated normally, a resource-deferred slot
may receive **one** fresh-directory recovery attempt if its original receipt
says `gpu_initialized=false`, no numerical model directory exists, the source,
seed and configuration are intact, and the remaining original campaign budget
can fund its full 3,060-second process allowance plus 10 seconds cleanup. These
are pre-import resource retries, not reruns selected from numerical outcomes.

Wait for resource availability within that recovery process's inherited
60-second pre-import limit. Use the original frozen worker and design, the
same original configuration and seed, and a fresh output path. Do not rewrite
primary progress or results, widen a numerical cap, change the criterion,
retry a sampled failure, start a second copy of a live worker, or infer that a
generic crashed supervisor is safe to restart. Preserve original unsuccessful
slots in the primary delivery report. Recovery results supply separately named
numerical evidence only; they cannot rescue the frozen primary release gate.
Any other failure needs concrete diagnosis before a new execution attempt.

## Budget and audit

The primary service has 189,073 reserved GPU seconds including cleanup. The
observer settles its enclosing duration once using service timestamps and the
terminal result, never adding child times. Each separate recovery attempt is
charged once to the unused part of that same reservation, with its actual
enclosing duration including waits. Do not transfer the outside GPU balance or
count a grant again. Total primary plus recovery cost may not exceed the
original reservation. After settlement, release unused reservation funds.

Transfer 3,600 earlier-authorized CPU seconds from the 48,936.183646 seconds
outside the release allocation to fund focused tests, active monitoring and
closeout. This is a convenience cap, not a new grant. Charge active observation
and notification wall time, excluding idle timer sleep and GPU repair work
already charged to GPU. The observer may use at most 1,800 active host seconds;
bounded tests may use at most 180. Keep the remainder for terminal scientific
audit. The 60-second polling interval is an operational responsiveness choice;
it changes neither numerical work nor its statistical interpretation.

Pre-mortem: a live process can be stalled, a stale checkpoint can be healthy
closeout, an exit-zero service can contain unsuccessful slots, and a successful
retry can obscure a primary failure. Inspect artifacts and classify each case;
never use a green service state or member count alone to declare release.
Missing source/evidence, doubled accounting, or any ability to overwrite primary
results fails the engineering audit. Test interrupted/missing-result states,
wrong seeds, false memory policy, sampled resource failures, live-primary retry
attempts, insufficient budgets, already-consumed recovery attempts, and invalid
source identity. No new numerical policy or experiment family is introduced.

Skeptical review: the important risk is accidentally changing the confirmation
estimand through retries. The design therefore keeps the original 96 outcomes
authoritative and makes all recovery evidence secondary. The actual deferrals
did not initialize TensorFlow, so bounded same-seed retries can repair missing
numerical work without rerunning a failed sampled outcome. The monitor is
proportionate local process/artifact supervision; it has no authority to patch
scientific code or declare a release. Implementation may proceed subject to the
focused fault-injection checks above.
