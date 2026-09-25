# A04 execution readiness

Independent protocol review AGREE; 18 CPU-only tests pass (14 initializer/pair,
4 observation-guided consumer). CPU devices were deliberately hidden from
CUDA. Exact historical target implementation hashes match campaign-02. The
only runtime edit extracts the existing generic initializer without changing
its values and reports explicit-initialization provenance correctly; an
executable call-chain/default-parity test passes.

GPU/XLA smoke-01 completes in 8.247 seconds with positive mass and no validity
failure. CUDA's ordinal 0 selected the RTX 4080 SUPER, as its full log records;
this is a smoke hardware exception, not RTX 5080 timing evidence. The full
run therefore selects RTX 5080 by UUID and records its framework device name.
This is a hardware selection repair, not a numerical-method change.

One-time coefficient TT-SVD runs as a CPU TensorFlow reference exception;
repeated recurrence and fitting use GPU/XLA. The recurrence's graph/compiled
parity is checked in each case. Separate d1/d4 scopes, saved target hashes,
fresh row partitions, L1 selection and all planned comparisons are wired.
Audit rows are generated after the validation decision is written to disk.
Numerical attempt 01 is ready under A04's unchanged evidence contract.
