# Controlled-SMC independent-repetition batching

The Figure 4 cost pilot passed but estimates 23180 seconds (6.44 hours) for
sequential execution. Its warm combined cSMC/BPF cost is at most 7.683 seconds
per repetition in the three observed cells. The authorized next engineering
step adds a native repetition dimension to the same scalar consumer. The scalar
interface wraps that common implementation with a size-one repetition axis;
there is no separate reduced batch algorithm or sample-wise map.

Question: does batching independent particle filters preserve each filter's
source-validated computation and make the intended empirical run less costly?
Baseline: the immutable scalar core snapshot from attempt12_c2_full_parity.
Candidate: the shared batched core, retaining source normal equations,
precision projection, FP64, TF32 disabled, XLA and all numerical guards.

Each replication receives distinct fixed noise and resampling uniforms. No
particle, weight or fitted policy may cross repetition boundaries. Tensor
linear algebra uses a repetition axis; only time and refinement remain loops.
Expected failures are wrong broadcast axes, cross-repetition normalization,
lost validity diagnostics, changed source results, compilation or memory cost.

Engineering pass: compare every short fixture's clouds, incremental weights,
normalizers, ESS and fitted coefficients to independent scalar calls under
1e-7 absolute + 1e-7 relative tolerance; ancestry must match exactly. Include
three independent replicas at H64/N32/I2. Check the scalar wrapper against the
same immutable endpoint. For the full H3000/N128/I3 batch of ten, compare its
first replica's forward and fit outputs to the scalar reference at every
refinement. Compare all four full-horizon BPF replicas with their scalar
reference at N5529. A failed identity, invalid accepted value or missing check
vetoes this candidate and triggers repair; budget/data corruption is a
continuation veto. One trace per invoked stable signature is required.

Run two full-sized batches for timing: cSMC batch ten, BPF batch four, variance
.10. Sizes divide 100 and are convenience engineering hypotheses. Warm per-
replication time is descriptive only; no runtime superiority claim follows
from two batches. Report GPU allocator peak, host RSS, compilation-inclusive
and subsequent execution time. Decline batching if it worsens the measured
per-replication cost or cannot meet numerical/memory checks. No new optimizer,
fitting objective, precision margin or resampling scheme may be introduced.

Source paper experiments, empirical uncertainty and heuristic comparisons
remain in their own phase. These checks support mechanical equivalence only;
they do not establish likelihood accuracy, Figure 4 replication, multidimensional
generality or improved scientific performance. The constructed heuristic set
and conditional noise regimes are unchanged from the master. This engineering
phase evaluates no scientific ranking against that set.

Pre-mortem/audit: a speedup could conceal replicas sharing weights or RNG. Use
different fixed inputs and compare each short replica, plus full-horizon
consumer outputs, to a preserved implementation. A scalar-only parity check
would be insufficient. Input layout and common call chain must be inspected.
The cost baseline is the same GPU execution and includes all four forward
passes and three fits. Skeptical audit PASS for this bounded engineering test.

At most two attempts and 240 GPU-charged seconds total under the existing
campaign allocation. Each attempt cap 180 seconds. No parameter fitting to
claim data, package changes or new scientific target. Preserve snapshots,
input hashes, fixed streams and every check in a unique attempt directory.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_batch

The manifest, checks.json, timings.json and summary.json are the result record.
