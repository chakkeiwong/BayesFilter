# M29 execution: numerical-child profiling

Status: bounded execution complete and audited. The governing
plan is `bayesfilter-hmc-post-m28-next-phase-2026-09-23.md`. The terminal M28
ledger opens this phase at 75647.83128280654 CPU and 79820.0906220735 GPU
worker-seconds; M29's ceilings remain 1800 CPU and 4800 GPU seconds, at most
two numerical workers, with serial GPU fits. No new compute is authorized.

## Preimplementation skeptical audit

The existing profile surrounds the framework-free isolated coordinator, and
the design validator explicitly rejects isolated profiling. Numerical work
must be profiled in `fit_process.fit_worker`. A coordinator profile cannot
answer the cost question.

Seeds use `design_id`, and a prepared binding also uses the full design digest.
Changing a design option for paired profiling therefore risks changing scope
or streams. Add an execution-only `--profile-execution` flag and propagate it
to each numerical child. Use one unchanged design file and ID for both arms;
record the flag in execution manifests. Preserve the legacy design option for
compatibility, but do not use it for paired evidence.

Completed fits must not be replayed to manufacture missing profiles. Inspect
profile availability on resume, preserve the original numerical receipt and
report unavailable instrumentation separately. Profiling setup, persistence,
or reporting failures must never replace the original numerical outcome.

The audit passes with these repairs. Exact replay is the primary numerical
criterion; profile contents are the instrumentation criterion. Timing is
descriptive. Runtime caps can truncate a profiled fit and are an explicit
affordability risk; do not count an incomplete pair as a parity pass. Historical
M27 values are target-specific development baselines, not calibrated defaults.
CPU tests are deliberately small non-XLA reference checks with GPUs hidden.
GPU fits retain XLA, verified memory growth, source isolation and public CLI
execution. Unrelated Q20/NeuTra edits are excluded from the frozen source.

## Concrete GPU count table

All counts are per chain; these are inherited M27 allocations, not new
adequacy claims. The remaining numerical fields must equal the respective M27
design. Allowed changes are design ID, seed, purpose/provenance/plan path,
budget (1000 seconds) and child timeout (980 seconds). Profiling is outside
the design. A full field diff and this table are checked before launch.

| Field | gaussian | beta_binomial |
| --- | ---: | ---: |
| draws | 500 | 500 |
| measurement_draws | 128 | 128 |
| posterior_cap | 20000 | 10000 |
| warmup_min_results | 30000 | 2000 |
| warmup_check_window_results | 30000 | 1000 |
| warmup_max_results | 60000 | 10000 |
| warmup_chunk_results | 5000 | 500 |
| retained_min_results | 4000 | 5000 |
| retained_max_results | 20000 | 10000 |
| retained_chunk_results | 2000 | 1000 |
| fixed_warmup_results | 30000 | 2000 |
| fixed_retained_results | 20000 | 10000 |

Gaussian retains its explicit 60000 posterior count ceiling. Beta-binomial
retains the existing implicit 10000 ceiling. Root seeds 2026092391 and
2026092392 are predeclared convenience identifiers. No sibling/profile replay
contributes a new independent-fit replication.

## Evidence and stop decisions

Numerical receipts, observations, candidate membership, archived tensors and
posterior decisions must match, excluding declared clocks and paths. Lost or
corrupt evidence, source mismatch, improper GPU policy and abnormal exit veto
the affected comparison. A posterior member failure triggers classification,
not removal from the verified candidate set. Profiling supplies host call
attribution including synchronization; it cannot resolve GPU kernel timing.
No speed ranking, coverage, posterior default or learned-map claim follows.

Attempt manifests and receipts live in
`docs/plans/artifacts/hmc-repair-master-2026-09-16/m29-r1/`. Charge only outer
worker receipts, including failures. Finish with a terminal audit, result,
budget reconciliation and a concrete refreshed next phase.

## Focused-test repair

The first frozen-source run passed 40 tests and failed both profile-content
integration assertions. Both ordinary fits completed and saved their numerical
evidence, but a profile started before TensorFlow import omitted the outer
pipeline functions. This is instrumentation invalidity, not a sampler failure.
The repaired boundary starts profiling after framework/device and pipeline
imports, recording setup time separately in the child manifest. Recheck actual
function coverage and numerical parity before GPU execution; never interpret
the incomplete first profiles as cost attribution. The first attempt's 133.05
CPU seconds remain charged.

The second probe reproduced omitted outer calls with Python 3.13's default
`cProfile` builtins tracing on a tiny TensorFlow graph. The underlying
C-extension event mechanism is not established. M29 therefore uses
`Profile.enable(builtins=False)`: it attributes Python orchestration and
TensorFlow call boundaries while deliberately excluding native/device timing.
The probe and failed pair remain diagnostic evidence, not a promotion result.

## Terminal checkpoint

All four GPU fits completed and both pairs passed the full saved-evidence
audit. Gaussian retained 22 verified candidates from 92 proposals and
beta-binomial retained 18 from 100. All 326 numerical observations/evidence
records and 178 tensor records matched between profile modes. Both selected
posteriors passed their declared checks; unselected members remain retained.

The final regression suite passed 43 tests, and all 15 documentation-contract
tests passed. The official book rebuilt with bibliography checks and a visual
inspection of PDF page 432 (printed page 414). A LaTeX path-spacing defect
in the new CLI example was repaired after that inspection. Supplemental
snapshot inputs and the final guide build have separate hashes.

The saved cost audit initially confused work-creation ordinals with scheduler
execution order. Its assertion was corrected to use the recorded observation
sequence; no sampler was rerun. First-call chunk time totals 313.707/369.978
seconds over 29/33 cache keys; warmed-call totals are 41.240/29.693 seconds.
These are descriptive times, not a decomposition of compilation and execution.

The terminal ledger charges 668.153 CPU and 2525.602 GPU worker-seconds over
32 outer attempts, including failures, all book-build retries and the
declared 60-second CPU bookkeeping allowance. Remaining campaign allowance is
74979.679 CPU and 77294.489 GPU seconds. There are no live M29 reservations.
The [result](bayesfilter-hmc-m29-profiling-result-2026-09-23.md) and reviewed
[M30 plan](bayesfilter-hmc-post-m29-next-phase-2026-09-23.md) govern continuation.
