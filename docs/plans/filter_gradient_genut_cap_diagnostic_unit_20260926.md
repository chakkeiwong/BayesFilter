# GenUT FP32 coordinate-cap report localization

Question: does the one-of-216 graph/XLA cap-active count difference arise
from different correction trajectories, from rounding/cancellation in the
coordinate-cap arithmetic, or from instrumentation changing compiler fusion?
The unresolved predicate is `abs(capped - pre_coordinate_cap) > 1e-7`.
This is explanatory localization of an existing failed gate, not a threshold
waiver or a canonical LEDH experiment.

Use the exact N=72,d=3 fixed inputs/seeds and default controls from
`test_filter_repair_genut_transitive.py`; keep the original source at
`072959c00`. Run original and candidate CPU graph/XLA, and candidate GPU
graph/XLA plus original GPU graph. Do not retry the preserved original FP32
GPU XLA compiler abort. TF32 remains enabled and memory growth is verified by
the campaign worker. Input clouds are generated on CPU exactly as before.

Make an explicitly diagnostic in-memory AST copy of the function that adds
pre-cap, powered ratio, denominator, capped value, displacement, predicate and
constant operands to its return dictionary. Save its exact source and hash.
Compare every untouched return field against the uninstrumented function,
including bitwise equality and the unchanged 2e-5 numerical bound. Instrumented
operands are evidence for the original evaluation only if those original
outputs match exactly; otherwise preserve the discrepancy and restrict the
interpretation to the instrumented computation.

Replay the cap formula on identical saved pre-cap operands in graph/XLA, then
replay only division/subtraction on identical saved pre-cap/denominator operands.
Use diagnostic Python Decimal at 80-digit precision for the smooth cap and for
the exact difference of rounded operands. Compare the original threshold and
record neighboring FP32 spacing without adjusting it. Capture full arrays,
point indices, returned fractions, per-field errors, source hashes and HLO.
These references diagnose rounding; they do not issue a replacement runtime
score or change any algorithmic parameter.

Promotion criterion: a reproducible numerical mechanism supported by matched
operands and an independent reference. Veto on attributing untouched behavior
from instrumented outputs that changed. Preserve failed equivalence as a
repair trigger; never reinterpret a diagnostic pass as closure of that gate.
Continuation stops on corrupted inputs, unexplained source drift, unavailable
required provenance, or budget exhaustion. The strongest misleading outcome
would be a seemingly obvious cancellation example created by instrumentation;
the mandatory untouched comparison tests that risk first.

Local review: the baseline, seeds, dtype, TF32 setting and 1e-7 predicate are
frozen to reproduce the existing failure, not promoted as good defaults.
Decimal is an independent diagnostic only. No NumPy is introduced into runtime
code, and no runtime source is changed in this unit. Default scientific and
HMC admission remain blocked under the canonical LEDH exclusion.

Execute `scripts/run_filter_repair_campaign.py test --group
genut_transitive_cap_diagnostic_cpu --device CPU --test-timeout-seconds 300`,
followed by the corresponding `_gpu` group with `--device GPU`.
Use the unchanged environment and automatically selected
available non-display GPU. Artifacts go into new campaign `run-NNNNN` folders.
Reserve at most eight CPU/four GPU invocations, each at most 300 seconds and
at most 1,200 seconds per backend, within the existing 56 CPU / 52 GPU
process-hour caps. Local harness repairs/retries keep the same contract.

Localization 04110 GPU / 04111 CPU passes exact instrumentation comparisons.
04109 accidentally omitted `--device CPU` and therefore used the CLI's default
GPU; it is preserved/charged as a duplicate GPU diagnostic, never CPU evidence.
Same pre-cap operands reproduce the count change, while replaying the same
rounded denominator through division/subtraction restores the graph count.
The next discriminating probe adds an XLA optimization barrier after the
denominator in a diagnostic copy and saves optimized HLO for both versions.
Use `genut_transitive_cap_lowering_cpu/gpu` with explicit device arguments under
the same invocation budget. No barrier is installed in runtime: reproducing
graph rounding would disagree with the frozen original XLA report, and this
diagnostic cannot waive either frozen comparison.
