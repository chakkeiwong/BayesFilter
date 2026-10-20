# Completed independent R 24-hour campaign

Status: frozen campaign and final analysis completed at 2026-09-21 10:32:17 UTC
(18:32 Hong Kong). All three phases are complete; no active or queued work.
Question and criteria: docs/plans/iapf-r-first-study-1000-replication-2026-09-21.md.
Study coverage: 1,000 paired replicas at each of d=5,10,20,40,80; 25,000 method
runs. Fit, tail, source and record-identity checks pass with no missing pairs.
Worker use, including preliminary phases: 107291.716/172800 seconds;
5136/10000 launches. Deadline was 2026-09-21T20:04:26+00:00.
Both QR variants pass practical and heuristic screens through d=40.
At d=80, window6 fails the particle-count allowance; both trigger the
conservative ordinary-innovation prefix-error veto, with difference intervals
including zero. Window5 passes terminal practical checks at all dimensions.
Neither variant clears every reference screen. QR differs from paper Eq.15;
no full paper replication, statistical inferiority or LEDH/KDM/HMC claim follows.
Evidence: result.md, terminal-decision.json, supervisor.json and
phase03-first-study-1000/analysis-v1/{result.md,summary.json}.
Next: inspect d80 prefix-error concentration and conditional uncertainty before
selecting a repair; source settings and multivariate R/TF parity remain open.
No experiment is currently running. Preserve all completed evidence.

Post-campaign gap audit: remaining-replication-gaps.md. Published replication
counts are complete. Both large-study arms use QR instead of Eq.15; window5 is
an explicit controller extension. Current timers include additional iAPF
diagnostic/I/O work and cannot establish the paper's efficiency claim. Prefix
errors are correctly untwisted but are an added project criterion, separate
from paper replication. Resolve the numerical procedure before another scale run.

Status reconciliation: the worker driver left its intermediate "analysis
pending" label after the supervisor finished. Reconciled against the terminal
decision and complete reports; no numerical results or criteria changed.
