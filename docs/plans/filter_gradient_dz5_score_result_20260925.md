# Fresh DZ5 score-oracle result

The current CDF analytical score passes the independent five-point value oracle
for all 23 parameters at both original steps on CPU/XLA 03866 and GPU/XLA 03867.
Both repeat values/scores/statuses exactly, retain one trace, export enclosing
HLO without host callbacks, and pass the frozen-source/import audit. The
96-observation fixture and unchanged 1e-8 absolute / 1e-7 relative gates match the
original qualifier. This qualifies the tested XLA scope, not a new adapter
admission, public initializer, posterior-wide derivative guarantee or main merge.

The source snapshot binds committed BayesFilter 4c37f9f40 and the unchanged
MacroFinance closure/inputs. It contains 725 sources and 3 inputs; manifest SHA-256
is `cbe75052fe85fcff58a7cbdc337e77e6507cf4240340308347993e88445c516d`.
Old admissions remain stale. No MacroFinance or package/environment source changed.

| XLA backend | Error / allowed error at 1e-3 | At 5e-4 | Exact replay |
|---|---:|---:|---|
| CPU |0.391720|0.767832|Pass|
| GPU |0.344380|0.625857|Pass|

Cross-device value/score differences pass the same original tolerances; the
largest normalized score difference is 2.918e-5. A separate standard-library
analyzer reconstructs every coordinate from saved values and verifies banks,
status, replay, source/snapshot/qualifier hashes, device/memory policy and HLO.
It rejects six deliberate corruptions (score, value, false pass, stale snapshot,
changed replay and invalid row). Receipt:
`artifacts/filter-gradient-repair-20260917/dz5-score-verification-03870.json`.
Its archive contains every 03863--03870 attempt, isolated executed script,
reference/source snapshot, logs, raw records and HLO.

The explicit graph references have two unresolved failures. CPU 03863/03864 pass
the finite-difference oracle but fail exact score replay by 1.0914e-11 / 1.3188e-11;
values and statuses repeat exactly. TensorFlow op determinism did not repair it.
Two-observation diagnostic 03869 repeats all initial/final values, factors and
tangents exactly on both 1 and 2 CPU threads. That shorter case does not explain
the 96-step discrepancy; the first differing longer prefix/operator remains open.

GPU graph 03868 passes the larger step but misses the smaller-step gate with
normalized error 1.00448179 at `log_foreign_equity_growth_diffusion_sd`. Recombining
the saved FP64 values in 80-digit Decimal gives 1.09914, so a more accurate final
combination alone does not rescue the comparison. Rounding sensitivity is an
explanation to investigate, not an error certificate or gate waiver. Raw failed
records are preserved; neither graph arm is silently accepted.

03865 already passed the CPU/XLA numerical/replay checks but failed diagnostic
HLO export: TensorFlow's device-inference helper made an unseeded random scalar
under op determinism. Supplying the known device removed that diagnostic-only
RNG; complete rerun 03866 passes. The target uses no such random operation.

Single-process capacity observations, including imports, are retained below.
These are not repeated cost rankings; failed graph arms remain unqualified.

| Arm | Cold seconds | Replay seconds | Sampled peak RSS GiB | GPU allocator peak MiB |
|---|---:|---:|---:|---:|
| CPU XLA 03866 |123.853|106.325|2.470|unavailable|
| GPU XLA 03867 |28.398|1.788|1.855|512.104|
| CPU graph 03864, replay failed |225.168|213.136|1.698|unavailable|
| GPU graph 03868, oracle failed |15.547|not run|1.466|826.485|

Memory growth is verified. CPU allocator telemetry is unavailable and is not
reported as zero. Parent supervisor RSS is separate: the observed 2.5 GiB is
queued for a history-record retention investigation. It is not target compiler
memory. No executable eviction, leak freedom or compiler-memory cause is proved.

The unit consumed 8 workers / 1578.815253 charged seconds within its 7200-second bound.
03870 renews all 129 policy tests. Ruff and whitespace checks pass. Remaining
campaign budget is 34.520592 CPU / 30.976836 GPU hours within 56/52-hour caps.

| Decision | Evidence/veto | Next action | Not concluded |
|---|---|---|---|
| Accept bounded fresh XLA gradient check | All 23 coordinates, both steps, CPU/GPU, replay and source checks pass | Actual initializer/staged-supervisor integration | Adapter admission or posterior-wide validity |
| Preserve graph failures | CPU replay and GPU smaller-step oracle remain outside gates | Longer-prefix/operator and value-error localization | Graph qualification or tolerance change |
| Investigate supervisor memory | Source retains all historical manifests while a worker runs | Stream metadata with exact budget/retry/provenance checks | Attribution of XLA memory |
| Keep main unmerged | LEDH wrapper, initializer, reporting/isotropic and F01--F20 gaps remain | Continue their bounded repairs | Whole-program completion |

Review: independent value differences address the shared-derivative risk that
archived/graph parity could miss. They still test one local parameter bank and
finite step sizes. Their largest normalized errors are not far below 1, so the
failed graph comparison is meaningful preserved evidence, not something to
round away. The full public workflow and unresolved replay mechanism remain the
weakest parts of the current engineering evidence.
