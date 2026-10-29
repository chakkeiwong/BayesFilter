# Replication run monitor

Heartbeat UTC: 2026-10-05T07:11:49.447688+00:00
Monitor: BATCH_TERMINAL_REVIEW_REQUIRED; interval 60s. Full replication remains incomplete.

## Source progress

- lorenz_csmc: 3537/4000; nonfinite=0; incomplete/invalid rows=0
- lorenz_apf: 0/4000; nonfinite=0; incomplete/invalid rows=0
- neuroscience_csmc: 2000/2000; nonfinite=0; incomplete/invalid rows=0
- neuroscience_bpf: 2000/2000; nonfinite=0; incomplete/invalid rows=0

## Scheduled continuation

- c1_source_failure_replay: COMPLETE; supervisor alive=False
- p0_forward_capture: COMPLETE; supervisor alive=False
- p0_forward_compare: COMPLETE; supervisor alive=False

Active health alerts: []

Terminal decisions, errors and repairs: [events](events.jsonl).
Notification submission records: [notifications](notifications.jsonl).
Desktop submission does not confirm display/read receipt. This monitor does not post chat messages.
