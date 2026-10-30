# Exact-tile scheduling candidate within computational phase2

Question: can batching the exact transport tiles reduce launch/scheduling cost
without changing the selected finite filter or any analytical score term?
Comparator: the frozen current shared batch source (SHA256
933135f1be600510c5e26b587fb3f8476802ed77c1837f076a75b4e6231190f2).
Candidate keeps N4032, K2016, all reset/flow iterations and arithmetic policy.
It batches the existing 2x2 tiles along leading tensor dimensions. It does not
change chunk extents or cache P*N*N tangents. Largest new temporary is primal
weights of 4*B*K*K numbers, about130MB at B2 FP32. The fixed cost cache already
contains this many entries. Packed analytical contractions remain O(N*P*D).

For each row tile r and column tile c, compute the same local log-normalizer
L_rc and derivative D_rc as the serial reducer. Merge column tiles with
L_r=logsumexp_c L_rc and D_r=sum_c softmax_c(L_rc) D_rc. This is the same
finite tile reduction in real arithmetic; FP32 reassociation must be checked.
For column normalization transpose the two block axes and both within-tile
axes. All parameter directions share one contraction, with the original
centered-position and centered-tangent terms retained. The final barycentric
transport and moment reset stay byte-for-byte identical in a generated full
candidate module; the diagnostic archives the exact replacement and source.
No selected working module is changed during candidate testing.

First run CPU FP64 primitive and full-transport value/score equivalence and
finite-difference checks, with K=N per policy. Then N4032 KSC and LGSSM GPU
transport primitives with five alternating warm pairs, cap600seconds each.
Pass: finite outputs, same row/column errors and <=1% scaled values/scores;
no new health failures. Nominate if at least10% faster in either shape without
>10% slowdown in the other. This engineering threshold avoids noise-sized
nomination; it is not posterior evidence. If nominated, run full-horizon
paired frozen-control comparisons and off-centre starts for all five models,
cap1800seconds per model; actual four-chain consumer, same-program FP64 and
fresh exact-scope tuning remain required before retaining scientific source.

The same86,400-second/120-attempt cumulative ledger counts compile/failures.
One GPU worker at a time, memory growth, TF32off, stable XLA signatures, no
pfor or scalar-row fallback. New output roots preserve every attempt. No OOM
or nonfinite may be silently hidden; stop this candidate on new health failure,
material arithmetic discrepancy or useful-cost failure. Memory growth must
remain within the existing16GB device; record allocator peak and host peak.

Default audit: tile sizes come from repository policy; batch scheduling is a
hypothesis with bounded extra primal memory. Off-centre starts are explanatory
coverage, not posterior coverage. Candidate cannot certify canonical LEDH or
posterior correctness. No model-specific implementation fork will be retained.
Skeptical audit: batch matmul may be bandwidth bound or increase memory enough
to lose performance; cheap transport probe decides. Tiled reduction may alter
rounding amplified by recursion; whole-sequence score and FP64 tests decide.
Generated diagnostic source is checked against frozen source and the only
changed region is the reducer. It is not an admitted runtime implementation.
