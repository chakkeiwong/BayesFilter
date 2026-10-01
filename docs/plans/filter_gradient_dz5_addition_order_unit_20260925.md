# DZ5 transition-tangent addition order investigation

Continue the authorized execution repair from `ab431169d`. The question is
whether protecting the written binary addition order in the rectangular SRUKF
transition tangent removes default CPU graph replay variation. The equation is
unchanged: `(F_x dx + F_q dq) + F_theta`. This is an execution hypothesis, not
attribution of the previously observed failure to that expression.

Baseline evidence is the unchanged 23-parameter/96-observation source snapshot
`dz5-candidate-source-score-4c37f9f40-r1`, the failed normal-optimizer prefix and
full graph replays, and the qualified explicit arithmetic-off reference 03901.
The 03900 AddN experiment proves buffer-sensitive summation in the installed
primitive; it does not prove which filter addition caused 03896. The two
multi-input additions in that optimized loop motivate this narrow intervention.

First test a diagnostic source overlay, saved verbatim with SHA-256 alongside
the immutable snapshot identity. Wrap each of the two binary additions in one
local `tf.function` with a stable rank-four signature and `_noinline=True`. The
sigma-point extent alone is polymorphic to permit a singleton broadcast; the
enclosing branch fixes and restores the final extent. It inherits
the enclosing JIT choice; there is no optimizer-default change, stop-gradient,
rounding, tolerance change, new derivative engine or external source edit.
The function is created once per enclosing trace, outside the time loop. If this
fails, retain the negative result and use actual optimized graph/operand evidence
to select the next bounded intervention. Do not broaden barriers speculatively.

Evidence contract and execution ladder:

1. Check cancellation against an independent exact arithmetic example and
   derivative propagation, for different fetched intermediates. Inspect executed
   optimized graphs to verify the function boundary and binary AddV2 body.
2. With ordinary optimizer settings and reused input tensors, execute the actual
   original 185-row oracle harness on 48 observations and at least four replays.
   Save every value, score and status output before asserting equality. A prefix
   success is nomination only; any discrepancy vetoes the proposed repair.
3. If nominated, repeat with all 96 observations, the original five-point steps
   1e-3 and 5e-4 and unchanged atol=1e-8 / rtol=1e-7. Exact replay, original
   oracle, full import/snapshot checks and absent host callbacks are required.
4. Only after diagnostic success install the small runtime change, commit it,
   freeze a fresh snapshot and qualify CPU/GPU XLA and graph, affected filter
   regressions, compilation/steady costs, RSS and device allocator telemetry.
   A diagnostic overlay alone cannot qualify runtime or close a master finding.

Use the stable runner `/home/ubuntu/miniforge3/envs/tf-gpu/bin/python
scripts/run_filter_repair_campaign.py test --group GROUP --device CPU
--test-timeout-seconds SECONDS`; register distinct diagnostic groups. Raw outputs
are new `run-NNNNN` directories under the existing campaign root. One numerical
worker at a time, frozen runtime/scripts/tests during each worker. This unit is
bounded to 18 workers / 7200 charged CPU seconds and, conditionally, 4 workers /
3600 GPU seconds inside existing 56/52-hour caps. No budget extension is implied.
Stop on cap exhaustion, source drift, missing raw evidence or invalid harness.
Local harness repairs and retries consume the same unit budget.

Exact replay and the original finite-difference gates are pass criteria;
source/import failures, host callbacks, lost derivatives, changed statuses,
unbounded retracing or failed XLA are vetoes. Costs, AddN counts and changed
rounding order are explanatory until matched repeated measurements are valid.
External MacroFinance callback autodiff/pfor debt remains explicit. No claim
about completely analytical whole-target execution, adapter admission, HMC,
posterior correctness, performance ranking or whole-program completion follows.

Skeptical review before implementation: a successful primitive could miss the
actual failing filter node; output retention can suppress the symptom, so the
target oracle keeps its original tuple outputs. Five calls only bound observed
replay and do not prove all-call determinism. `_noinline` may be ignored or
handled differently by XLA; graph inspection plus separate XLA regression is
required. The overlay must be labeled as executable modification even while
loaded module files retain their snapshot checksums. Default optimizer settings
are recorded, not inferred. The cancellation case has a known exact answer but
is not a substitute for the actual target. These controls make the first narrow
diagnostic informative; runtime promotion remains conditional on later gates.

Results and next action will be recorded in a linked result note and the concise
master/recovery checkpoint before leaving this unit.

Pre-install compatibility review during 03908: ordinary TensorFlow addition
also permits direct derivatives broadcast over batch, parameter or other axes.
The installed candidate must preserve that input behavior. Explicit native
broadcast of the direct derivative to the already declared transition-tangent
shape can satisfy the helper signature without changing the equation; test
singleton and scalar direct derivatives against full-shape equivalents with
graph/XLA and independent value differences. This is a compatibility repair,
not a new numerical method. Full snapshot qualification must execute the actual
installed bytes, since the diagnostic overlay does not include that broadcast.
