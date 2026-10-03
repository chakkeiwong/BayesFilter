# 72-core canary reset memo

Date: 2026-09-03  
Controlling plan: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`

Attempt 01 was a harness failure: an absolute-path launch omitted the
repository root from `sys.path`, and the controller inspected a child return
code before joining it.  The child had produced a valid task result, but the
controller did not consume it.  The repair was limited to import-path setup
and process joining.  No target, chart, seed, topology, or numerical code was
changed.  The focused tests and compile checks passed, and attempt 01 remains
archived rather than overwritten.

Attempt 02 passed the composite canary.  Its summary and per-worker records
are under
`docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03/canary/attempt-02/`.
The measured wall time was `581.3556926490273` seconds.  All CPU IDs were
disjoint within each sequential barrier, all children hid CUDA before
TensorFlow import, and fixed-seed serial/process parity passed.  The
historical fixture chart was used only to test mechanics; it cannot be used
as a fresh tuning result.

The full-run entry gate is therefore open.  Fresh chart preparation must use a
new seed namespace and a new output root.  The provisional full cap remains
`14,400` seconds, and the total campaign cap remains `15,600` seconds.  These
numbers are resource hypotheses, not scientific thresholds; the full result
must report actual barrier timings and stop on the cap without promoting
partial work.

