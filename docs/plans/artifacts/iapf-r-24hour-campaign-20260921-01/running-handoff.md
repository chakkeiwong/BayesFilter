# Active unattended continuation

The master and phase plans are updated. The real phase2 driver and the separate
unattended supervisor are running. The supervisor waits for phase2 validation,
then runs the frozen five-dimension1000-replica phase and terminal reporting.
Live authority: manifest.json, attempts.jsonl, checkpoint.md, supervisor.json.
Exact plans and phase specifications: phase02.json and phase03.json.
Validated source differences: phase03-numerical-source-check-v2.json.
Focused regression: five tests pass in supervisor-regression-v3.log.
Phase1 science: phase01-controller-pending/analysis-v2/result.md.

Do not start a second worker driver while manifest.json lists inflight work.
Do not reset prior attempts or tune on these datasets. A candidate rejection
continues the planned study; genuine evidence or resource failures are recorded
as continuation boundaries. The deadline is2026-09-21 20:04:26 UTC. Both resource
ceilings apply. No old budget is transferred and no numerical default changed.
