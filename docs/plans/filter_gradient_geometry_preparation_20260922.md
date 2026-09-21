# Geometry direction and finite-design preparation

Continue from c7eba61b within the existing campaign, single-worker rule and
32 CPU / 52 GPU process-hour caps. GPU3 keeps its unchanged idle preflight;
CPU reference workers hide GPUs. No new method, tolerance, seed stream, package
or external consumer change is proposed.

Enclose three missing numerical operations in stable XLA programs: normalization
and ordered compaction of raw pilot directions; design point construction and
exact scalar/batched evaluation; and finite-row compaction plus the original
training/holdout partition. Fixed-capacity outputs carry explicit active counts.
Padding must never become an extra target call, training row or holdout row.
The partition receives the exact prepared permutation of the retained rows as
a runtime tensor, with its unused tail ignored. RNG remains outside this scope.

The baseline is isolated 3582b4ac. Extract the unchanged normalization and
partition statements for direct comparison, and verify the extracted partition
against values captured from the full original initializer before fitting.
Frozen directions, offsets and permutations are identical across arms. Check
every retained direction/index/value/score/point, active count, callback order
and row partition at unchanged 1e-10 tolerance and exact discrete comparisons.
Include zeros, empty directions, nonfinite directions, all/partial invalid
targets, ties, insufficient finite rows, exact sample threshold, holdout cap,
zero holdout and changed same-shape inputs. The original predicate is norm > 0,
not a new finiteness filter. Invalid caller permutations fail closed explicitly;
they cannot silently duplicate or omit training rows.

Review before implementation: fixed-capacity padding permits XLA compilation
but does not justify using a padded QR or a padded batched callback. These
programs expose counts only; the existing fixed-extent pilot/fit programs cannot
yet consume dynamic counts without additional qualification. In particular,
variable-size batch callbacks and post-evaluation permutation generation remain
whole-initializer obligations. No CPU dependency result closes that call chain.
No callback executes inside the normalization or partition programs.

Use the bounded campaign driver for one small smoke, then dimension/lane and
boundary checks, runtime-operand/HLO and resource checks. Fresh D3/D5 matched
original/graph/XLA costs must include the same input preparation and output
materialization on each side; retain cold, warm, observed host RSS and device
allocator metrics separately. Existing investigation triggers apply and cannot
be waived by smaller dependency timings. At most three attempts per unchanged
job, with 120/300-second ceilings and unique numbered artifacts in the shared
campaign root. Freeze runtime/tests/driver during each worker/matrix.

Numerical disagreement triggers localization; unsupported callbacks, malformed
permutations, contention, invalid evidence or budget exhaustion veto the affected
launch/result. Successful checks permit retaining this internal candidate only.
Whole initializer, iterative control, GPU/default readiness, HMC and terminal
repair claims remain unsupported. This bounded review finds the baselines and
omitted boundaries explicit and the next CPU diagnostic justified.

02555 passes the first partition smoke; 02556 passes all 46 initial checks.
02557--02580 pass all 24 original/graph/XLA CPU cost processes. Extended review
adds full-original insufficient-row exits, combined design/partition enclosure
and underflow/holdout rounding boundaries. 02581 passes 49 checks and exposes a
real direction-retention difference: CPU TensorFlow flushes subnormal squared
components, retaining two rows where the original retains four.

Repair the original binary64 arithmetic, not its norm > 0 predicate. For small
rows, express squared components in units of 2^-1074 before summation and recover
the norm as sqrt(unit_sum) * 2^-537. Subnormal products require exact rounding:
decode each 53-bit significand, square it in two uint64 words, then shift/round
to nearest with ties to even. Merely rounding a floating scaled product can
double-round at half-unit boundaries and is not sufficient. Normal products
can use power-of-two scaling directly. Ordinary rows keep their original path;
mask the unused scaled branch before multiplication to avoid artificial overflow.
Verify original NumPy products plus an independent integer-significand reference,
including adjacent values around half-unit boundaries, before renewing direction
costs. Other original fields/tolerances and the GPU gate remain unchanged.

02582 restores the failing direction and exact holdout rounding boundaries.
02583 passes all 51 checks, including 2,092 binary64 products compared exactly
with Python integer-ratio rounding and NumPy, plus the combined design/partition
HLO and original insufficient-row exits. Renew six direction cost processes
because its arithmetic changed. Investigate the standalone direction/partition
warm triggers using separate synchronized native-call and completed-report timers,
then 3,000 alternating-input calls at each D3/D5 extent. These component timers
are explanatory, not a comparison with a differently bounded original kernel.
Retain allocation checkpoints at build, trace, first execution and 1,000-call
intervals; absence of late growth in this bounded run cannot prove leak freedom.

02584--02589 pass all six renewed direction cost processes. 02590--02593 pass
four component/reuse processes, each with 3,000 alternating-input calls. The
final 1,000-call RSS change is 0/4/4/0 KiB (direction D3/D5, partition D3/D5).
All retain one trace and complete original records. Most allocation occurs
after tracing and before the first measured warm phase. These observations do
not demonstrate native executable eviction or general leak freedom.

| Operation | Dimension | Original / graph / XLA warm ms | XLA cold s | Extra RSS MiB |
| --- | --- | --- | --- | --- |
| normalization | 3 | .692 / 1.386 / 1.153 | .244 | 113.5 |
| normalization | 5 | 1.013 / 1.668 / 1.525 | .207 | 112.8 |
| partition | 3 | .932 / 2.239 / 2.349 | .173 | 130.5 |
| partition | 5 | 1.362 / 2.709 / 2.650 | .173 | 130.7 |
| scalar design | 3 | 24.510 / 2.602 / 1.898 | .168 | 112.9 |
| scalar design | 5 | 24.041 / 3.180 / 2.507 | .162 | 113.1 |
| batched design | 3 | 2.020 / 1.776 / 1.875 | .133 | 98.4 |
| batched design | 5 | 2.439 / 2.506 / 2.493 | .129 | 98.5 |

These are descriptive one-process comparisons with identical prepared inputs
and complete common outputs. Fixed-capacity bookkeeping is additional candidate
work. Direction costs use the renewed implementation; the other operations
retain their unchanged 02557--02580 numerical source. The old direction costs
are preserved but superseded. Every XLA arm fires the cold-time trigger;
normalization and partition also fire the warm-time trigger. No 256 MiB host
trigger fires. Component measurements find native/report median costs of
.385/.877 and .365/1.111 ms for direction D3/D5, and .592/2.012 and .588/2.449 ms
for partition D3/D5. This explains reporting's contribution but does not waive
the complete-cost regressions or compare a native kernel against original
reporting. Enclosing integration must avoid unnecessary intermediate reports.

Artifacts under the shared campaign root:

- `geometry-preparation-directions-costs-02589.json`;
- `geometry-preparation-{partition,design_scalar,design_batch}-costs-02580.json`;
- `geometry-preparation-components-reuse-02593.json`;
- raw run directories 02555--02595, retaining failure 02581.

02594 passes all 72 policy/controller checks. Audit 02595 covers 2,986 working
Python files, 2,985 parsed and one unchanged vendor-reference parse error.
The guard remains partial at 213 sources / 1,306 exact exceptions; this module
adds no exceptions, Python iteration or NumPy dependency. Focused Ruff and
whitespace pass. The refreshed remote main remains c7adbda7; no main merge.

| Decision | Primary criterion | Vetoes / uncertainty | Next action | Unsupported conclusion |
| --- | --- | --- | --- | --- |
| Retain internal preparation candidate | All 51 original/boundary/enclosing checks pass; exact underflow reference passes | GPU not qualified; variable active extents and RNG boundary remain outside full initializer | Qualify GPU, then consume active counts inside pilot/fit control | Public GPU, whole initializer, HMC or terminal repair readiness |
| Keep timing investigations open | All matched numerical cost records pass | Standalone normalization/partition warm regressions and all cold triggers | Measure full enclosing execution with completed reporting at its final boundary | General speed or memory improvement |

Post-run review finds no relaxed predicate or comparison. The strongest
alternative explanation for the standalone warm penalties is dispatch and
reporting overhead; component timings support that contribution, but only
whole-initializer costs can settle its practical impact. GPU behavior and larger
capacities remain the weakest evidence. Source and callbacks stayed frozen in
each matrix. No worker is active at this checkpoint.
