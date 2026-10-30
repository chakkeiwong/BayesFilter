# Dense spectral and extreme-precision qualification

The near-identity dense XLA eigenvalue error is repaired using the existing
residual-refined eigensystem. QR reduction computes the design SVD on the
triangular factor for static tall XLA designs. It changes neither the fitted
precision nor the rank threshold. Public uniform-cloud refinement remains on
its old execution route pending complete-controller qualification.

The owner approved `huge_dense_d3_d5_matrix_relative_v1` on 2026-09-21. Only
`raw_precision` in the two frozen huge D3/D5 stress fixtures uses matrix-relative
error <=1e-12. Every other floating field keeps atol=rtol=1e-10; schemas,
shapes, dtypes, ranks and decisions remain exact. Both solver outputs must also
match independent100/160-digit references and satisfy response-residual bounds.
The implementation verifies that the input is the declared diagonal-response
fixture. No production numerical decision uses this diagnostic comparison.

Mutation attempt02231 retained a fixture mistake: an independently generated
D5 design fell outside the approved scope. The corrected fixture replays the
original design;02232 passes all22 mutation checks.02233/02234 pass all14 dense
spectral cases per device;02235/02236 pass the independent high-precision and
original self-sensitivity checks. 02237/02238 pass6 independent derivative cases per device;02239/02240
pass28 dense rank/consumer cases per device;02241/02242 pass50 quadratic numerical
cases per device.02244 passes72 policy/controller checks. CPU component02243
passes2 diagnostic cases; GPU attribution remains pending contention. In total,
296 checks passed through02244. The original strict failures02173--02176 remain archived.

All36 fresh-process costs02195--02230 preserve the initial and changed complete
records. The original graph source3582b4ac is the authority; no original XLA
failure was used as a baseline. All GPU workers used GPU3 idle preflight and
verified memory growth; the prelaunch contention stop after02212 is archived
in `dense-cost-preflight-pause-02212.json`. Numerical sources were frozen.

| Device/dimension | Original warm ms | Current graph ms | Current XLA ms | D5 investigation |
| --- | ---: | ---: | ---: | --- |
| CPU D3 | 0.574 | 1.377 | 0.592 | None |
| CPU D5 | 0.580 | 1.834 | 0.720 | +24.2% versus original |
| GPU3 D3 | 2.075 | 6.275 | 1.632 | None |
| GPU3 D5 | 2.335 | 8.732 | 3.736 | +60.0% versus original |

These are medians of three fresh-process medians. The GPU original ranges are
1.58--2.11ms D3 and1.61--2.66ms D5. Measurements are descriptive; this sample
supports no statistical ranking. The QR change has not closed the D5 cost gate.
Component attribution must distinguish its reduction overhead from the newly
corrected spectral computation, then whole-controller costs must establish the
practical effect. Separate component timings cannot be added together.

No declared memory trigger fires in these dependency costs. Observed CPU host
RSS increases225/231MiB; GPU host RSS changes about-2MiB. GPU allocator peaks
are18,944/25,600 bytes versus approximately8.4MB originally. Cold XLA calls are
0.739/0.831s CPU and2.278/2.459s GPU. Small warm RSS increments persist over this
short sample; it establishes neither exact allocation attribution nor general
leak freedom. Full initializer/native executable retention remains separate.

Primary-agent review: numerical checks use the original dependency closure and
independent references; no missing field or failed timing is discarded. The
weakest evidence is the tiny, well-conditioned artificial design. It supports
no general extreme-scale or ill-conditioned accuracy claim. Full enclosing
controller and actual DZ5 consumers remain separate required evidence.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue dense/controller qualification | Boundary and comparison checks pass | D5 performance trigger open | Full controller impact unmeasured | Component attribution and native uniform record checks | No public default promotion or merge |

| Inference status | Finding |
| --- | --- |
| Hard screen | Complete dense/quadratic numerical records, derivatives and policy checks pass |
| Statistical ranking | None established |
| Descriptive differences | Warm medians/ranges and memory above |
| Default readiness | Public uniform integration remains gated |
| Next evidence | Full controller numerical and cost qualification |

Commands, environment, source hashes, input hashes, seeds, hardware and exact
wall times are in each numbered `run.json`, worker metadata, log and numerical
JSON under `docs/plans/artifacts/filter-gradient-repair-20260917/` in the main
worktree. Analysis: `dense-numerics-repeats-02230.json`; comparison script:
`/tmp/analyze_dense_numerics_repeats_20260921.py`. Shared caps remain32 CPU/52 GPU
process-hours; one worker at a time and300-second focused limits are unchanged.

Checkpoint inventory02245 covers2951 working Python files,2950 parsed and one
unchanged vendor-reference parse error. The source guard is still partial at
204 sources/1289 exact exceptions; none added here. Focused runtime/test Ruff and
whitespace checks pass. The driver has five pre-existing lint findings already
recorded at d263830c; no repository-wide lint claim. Charged process time is
46542.280774255734/115200 CPU seconds and43955.581166660864/187200 GPU seconds.
No worker is active. Remote fetched: repair branch synchronized at the previous
checkpoint, remote main an ancestor; main remains unmerged. Native uniform
controller qualification is the next independent work while GPU3 is busy.
