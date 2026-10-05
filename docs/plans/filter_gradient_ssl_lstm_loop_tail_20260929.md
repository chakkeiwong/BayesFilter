# Fixed replay native-loop tail intervention

Question: does removing the per-date terminal conditional reduce the measured
compiled-context overhead while preserving the entire replay? Source baseline
c05f2c2cc has a native observation loop with a conditional transition at every
date; GPU T8 explicit-XLA warm ratio against the original is1.235. Its graph
and runtime evidence is preserved through05059. This is an execution-cost
follow-up to a specific rewrite regression, not algorithm or optimizer research.

Change only control scheduling: execute observation and transition for dates
0 throughT-2 unconditionally in one native loop, then evaluate observationT-1
once. Preserve ordering, all RNG draws, analytical equations, final states,
histories, manifest/error behavior and both RNG contexts. Share one observation
helper between the body and tail; no Python numerical iteration is introduced.
At T1 the loop is empty and only the original first observation executes.
Numerical authority and source-route/nonclaim classification stay unchanged.

First run the existing17-case qualification group on CPU reference and trusted
GPU3. Failure against pinned c4950a827 full records, independent derivatives,
seed streams, original errors, default HLO or changed-input reuse blocks cost
execution. The intermediate c05f2c2cc source is also retained in Git for exact
review; qualification against the same original establishes the numerical
contract without making the candidate its own oracle.

If qualified, freeze source/harnesses and repeat the same72-worker matched cost
matrix: CPU/GPU, T2/T8, before/after, actual default/explicit graph/explicit XLA,
three repetitions with reversed order in repetition1. Existing runner groups,
commands, inputs, physical GPU UUID, memory growth, provenance, thresholds and
independent readbacks apply. Compare paired final ratios and absolute costs with
the preserved earlier cohort, keeping environment/time-of-run uncertainty
explicit. A remaining or worsened trigger requires recorded disposition; speed
of the new public default cannot silently waive an already-XLA regression.

Renew the two declared2000-call/20-specialization lifetime checks on final
source, then the independent cost/lifetime readback and policy suite. Record
cold cost, graph size, host and live device memory. Process-lifetime native
residency is a known limitation; this intervention does not claim eviction.
Do not alter numerical bounds, streams, algorithm, compiler flags or allocator
policy to improve timings.

Allocate at most90 serialized workers,6000 CPU and6000 GPU process-seconds,
300-second worker timeouts, inside the existing56CPU/52GPU-hour global cap.
Use existing `ssl_lstm_replay_` groups and unique numbered artifacts after05059.
One numerical worker at a time, no subagents/HMC/training. Stop on numerical or
provenance failure, exceeded allocation or unusable timing evidence. Preserve
failures; only local harness repair within the same contract may retry.

Primary-agent skeptical review: the branch executes inside the recurrent
program, so this is a concrete control-flow hypothesis rather than attributing
the regression from graph counts alone. Separating the final observation
preserves exact chronological operation order and noise indexing. Two extents
and existing T1 tests detect off-by-one/final-state errors; independent gradient
tests detect missing terms. Three process pairs are limited timing evidence,
so report conditional intervals and absolute overhead, not broad superiority.
The plan passes review without changing any scientific method or claim.
