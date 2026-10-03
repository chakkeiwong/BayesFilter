# Progress-counter split result

CPU/XLA04646 changes only the round-budget counter and preserves the
unmodified candidate score. CPU/XLA04647 changes only the three reporting
counters and restores the historical original score exactly. Input, value
and validity bytes remain identical against04584/04585/04635/04636. Both
workers have one trace, three optimizer objective batches and no host callbacks;
saved source/AST, HLO and callback hashes pass readback04648 (163 checks).

The unit closes at3/5 workers,451.586176/1500 CPU process-seconds. No runtime
source or policy exception changed. The positive family is now `attempts`,
`optimizer_calls`, `replays`; none intentionally feeds target arithmetic or
the callback-budget decision. This establishes a compiler-context effect for
the bounded fixture, not an analytical formula error.

Next: individual reporting-counter interventions and one isolated candidate
retaining all int64 resources while doing reporting increments in int32.
Record the counts to exclude overflow and verify resource storage explicitly.
No full-trajectory, GPU compatibility, convergence or runtime repair is claimed.
The121 strict trajectory differences and other master gates remain open.
Primary-agent skeptical review confirms these limits; no independent reviewer.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Localize to reporting counters | Reporting-only intervention restores original first-score bytes | No GPU storage qualification | Which counter or interaction changes compilation | Individual counters and storage-preserving arithmetic | Runtime repair, full trajectory or convergence |
