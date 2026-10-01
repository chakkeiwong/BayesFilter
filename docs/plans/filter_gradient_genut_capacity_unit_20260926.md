# GenUT reduction capacity and execution cost

Question: do the native N*d*d tensor-product reductions introduce material
host/device memory growth at the particle counts and state dimensions used by
the repaired filtering consumers? The previous evidence covered only N=72,d=3.
This unit tests the reduced primal correction alone, not canonical LEDH, full
transport/reset memory, analytical scores or target statistical accuracy.

Use FP32, global TF32 enabled, unchanged four diagonal/four pairwise iterations,
strengths, floors and caps. Deterministic CPU-generated inputs use the existing
101/102, 103/104 and 105/106 seeds. Test (N,d)=(1000,3),(10000,3),(10000,18).
The first two isolate N scaling; d=18 covers the nine-location SIR state
dimension as a synthetic capacity envelope. The generated well-conditioned
cloud is not an Austria observation/data qualification. No transport is called;
these array dimensions do not introduce an alternative transport chunk policy.

Fresh-process arms: frozen `072959c00` graph/XLA and candidate graph/XLA on
CPU; original graph and candidate graph/XLA on GPU. The original FP32 GPU XLA
abort is preserved and is not retried. Use 10 synchronized replays, exact
within-owner replay, one trace, independent FP64 mean/covariance restoration
at the existing 2e-5 bounds, finite particles, and the original radial cap.
Record full returned arrays, hashes, input identity, cold and median warm time,
RSS before/after compilation and replay, high-water RSS, GPU allocator
current/peak, device identity and sampled foreign-process activity.

Complete cross-arm records are compared under unchanged bounds. Preserve the
known thresholded cap-report mismatch separately as an unresolved veto; do not
call the complete-record gate passed. A baseline that fails independent moments
is recorded as invalid comparison evidence for performance ranking. Candidate
nonfinite output, failed validity/moment/radial-cap/replay or source drift stops
its affected capacity arm for localization. Memory increase triggers explicit
attribution and retention checks; passing a small peak does not establish a
universal memory bound. There is no stochastic speed ranking from this cohort.

Default/assumption review: Gaussian clouds and fixed seeds are controlled
capacity probes; they avoid confusing ill-conditioned input with allocation
growth but leave real-target validity open. Original graph and original CPU
XLA have different finite cap reporting, so neither is silently chosen as a
universal scalar authority. Independent moments prevent graph/TF32 rounding
from being mistaken for correct restored covariance. The old and new methods
have identical controls; this experiment changes execution only.

Local skeptical review: RSS includes compiler/context and host-return arrays,
while TensorFlow allocator counters do not. Capture both; compare identical
return formats and device UUIDs. Returning intermediate arrays would perturb
memory, so this capacity test uses untouched runtime functions. The earlier
operand-instrumentation diagnostic is not a benchmark kernel.

Use registered `genut_transitive_capacity_{arm}_{mode}_{N}_{d}_{device}` groups
with explicit `--device CPU/GPU --test-timeout-seconds 300`. The tf-gpu
interpreter, two CPU threads, automatic available non-display GPU and verified
memory growth are unchanged. Reserve at most 18 CPU / 12 GPU invocations,
300 seconds each and 1,800 seconds per backend in aggregate inside the existing
56/52-hour campaign caps. Run the 21 initial arms once; retries require a
localized finding and remain in this reserve. Stop on insufficient remaining
reservation, corrupted evidence or source drift. All outputs use new campaign
`run-NNNNN` directories. No runtime sources change during the cohort.

Post-cohort trigger through 04134: all candidate moment/replay checks pass,
but complete records exceed their bounds against original graph in several
fields. Original GPU graph independently fails moments at d=3; d=18 still
passes moments but its full particles differ. Generate three CPU FP64 frozen
original reference runs on the exact saved FP32 operands, using the registered
`genut_transitive_capacity_fp64_{N}_{d}_cpu` groups and the same 300-second
limit. These three invocations remain inside the 18-CPU/1,800-second reserve.
Compare all saved FP32 arms with this full-program reference and preserve the
original 2e-5 per-field bounds. Reference diagnostics do not waive the original
complete-record gate or certify a new numerical policy. The 18-dimensional CPU
XLA warm-time increase (76.8 versus 31.5 ms in one cohort) is a separate
performance trigger, not a reason to ignore numerical differences.

The FP64 references 04135--04137 attribute every non-report original/candidate
comparison failure to an inaccurate original comparator at the unchanged
bounds; all candidate non-report fields pass. Retain cap-report failure.
Use one CPU matched timing diagnostic with both XLA owners retained, 20 paired
replays in alternating order, exact replay, and optimized HLO captures. It
tests whether the d=18 warm-cost increase persists with matching process and
inputs; it is not a memory comparison or universal speed ranking. The registered
`genut_transitive_capacity_matched_cpu` group plus final policy checks remain
within the original CPU invocation/time reserve. No runtime source changes.
