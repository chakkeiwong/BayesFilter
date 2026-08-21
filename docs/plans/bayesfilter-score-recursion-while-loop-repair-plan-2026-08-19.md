# Score-Recursion XLA While-Loop Repair Plan

Date: 2026-08-19
Owner authorization: user directive "We should not have python loop in an XLA
path. Fully audit the code to find similar design issues. Create a plan to fix
the problem. Review the plan. Execute the plan and test again."
Parent campaign: `docs/plans/bayesfilter-genut-sqmc-streaming-n16128-plan-2026-08-18.md`
Parent result: `docs/plans/bayesfilter-genut-sqmc-streaming-n16128-result-2026-08-18.md`

## Question

Does converting the all-parent backward score recursion from an unrolled
Python block loop to a sequential `tf.while_loop` bound peak GPU memory to
O(block x N) per step, preserve small-N numerical behavior, and make the
`N=16128` streaming feasibility row executable on a 16 GB device?

## Audit of Python-level loops in the XLA route (2026-08-19)

The compiled scope is `evaluate` in
`docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py`
(`jit_compile=True, autograph=False`), which calls
`finite_value_standard_score_initial_rqmc`. Every Python-level loop reachable
from that scope was classified:

| Site | Loop | Trip count | Live tensor scale | Verdict |
|---|---|---:|---|---|
| `ledh_pfpf_genut_initial_rqmc_tf.py:198` `standard_pairwise_backward_marks` | Python `for` over child blocks | `N/126` (128 at `N=16128`) | `[126, N, d/p]` per block, all blocks co-live after XLA fusion → O(N^2) | **DEFECT — the OOM. Fix in this plan** |
| `ledh_pfpf_genut_initial_rqmc_tf.py:985` horizon loop | Python `for`, fallback branch | `T=20` | full per-step state x 20 | Benign on active path: harness passes `functional_time_loop=True`, taking the `tf.while_loop` branch (line 977). Fallback retained for eager debugging only |
| `sqmc_tf.py:76,88,112-113,141-142` Hilbert bit loops | Python `for` over bits x dimension | `<= 12 x 3` | O(N) int32 elementwise per iteration, no growth | Benign: constant small trip count, no O(N^2) intermediates |
| `sqmc_tf.py:168` word sort loop | Python `for` over 30-bit words | ~2 | O(N) argsort per iteration | Benign |
| `genut_guided_proposal_tf.py` streaming transport | already `tf.while_loop` + `TensorArray`, `parallel_iterations=1` | — | one K x K tile | Correct pattern; reference idiom for this repair |
| Harness `_row`/`main` loops over counts/routes | Python `for` | small | outside `@tf.function` | Benign (not in XLA scope) |

Conclusion: exactly one defective site. The rule "no Python loop in an XLA
path" is refined to: no Python loop whose per-iteration live tensors scale
with N and whose trip count scales with N; constant-trip-count elementwise
bit loops are acceptable and standard.

Evidence the defect explains the OOM: allocator peaks track
`N^2 x 126 x 4 B` (`0.68 GB` at `N=1008`, `8.29 GB` at `N=4032`, both plans
identically), and the `N=16128` failure was a single fused 56.9 GB
allocation, the signature of XLA fusing across unrolled sibling blocks.

## Mechanism under test

`tf.while_loop` (with `parallel_iterations=1`) compiles to a genuine XLA
`While` region: buffers from iteration `i` are dead before iteration `i+1`
allocates. A `tf.TensorArray` with static `element_shape` accumulates the
per-block `[126, p]` mark outputs. This is the same idiom already proven in
the streaming transport of `genut_guided_proposal_tf.py`.

Per-step live memory after repair (at `N=16128`, `d=3`, `p=3`):
one block's `[126, N]` float64 transition log/score intermediates
(~16-50 MB) + O(N) state, instead of ~131 GB of co-live block intermediates.
Transport is already streamed (`K=2688`, 29 MB/tile). Predicted peak: low
single-digit GB, dominated by O(N) route state and the `6 x 6` tile loop.

## Change

One function: `standard_pairwise_backward_marks` in
`bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`. Replace the Python
block loop with `tf.while_loop` + `TensorArray`, preserving per-block
arithmetic exactly: same `[block, N]` broadcast grids, same float64 density
casts, same softmax normalization, same einsum, same block concatenation
order (TensorArray stack = concat of blocks in index order). The
`DiagonalLGSSMAnalyticalModel` early-return branch, validation, and signature
are unchanged. No new controls; `score_child_block_size=126` stays frozen.

## Success criteria and evidence contract

- Primary: the existing focused suite passes unchanged, in particular
  `test_austria_score_child_blocking_matches_dense` (blocked-vs-dense marks
  parity) and `test_austria_streaming_route_matches_dense_value_and_score`
  now exercise the while-loop path. These tests compare blocked against
  single-block dense reference; the repair must not change that parity.
- GPU regression: fresh `N=1008` and `N=4032` dense/streamed pairs pass the
  owner-revised parity basis (value relative diff `<= 0.1%`, all hard
  validity screens; scores descriptive). Cross-check against attempts 04-07
  values is descriptive only (while-loop scheduling may perturb rounding; the
  route is chaotic, so trajectory equality with the old graph is not a
  criterion — the same basis as the 2026-08-19 gate revision).
- Feasibility: the `N=16128` streaming repaired-permutation row completes
  finite/program-valid/row-valid with `TV<=1e-4`, zero saturation, 16128
  unique ancestors, allocator peak recorded and well under 14 GB.
- Veto diagnostics: any focused-test failure, nonfinite output, validity
  failure, or allocator peak that still scales as N^2 vetoes the repair.
- Explanatory only: runtime changes (a sequential loop may be somewhat
  slower than fused unrolled blocks at small N), exact score values.

## Failure modes and pre-mortem

- XLA rejects `TensorArray` inside nested while loops → fall back to a
  dense `[N, p]` accumulator updated with `tf.tensor_scatter_nd_update` or
  a stacked `tf.foldl`; same memory bound.
- Hidden second N^2 site emerges at `N=16128` (e.g. inside model callbacks)
  → the audit table above says no; if the run still OOMs, record the new
  allocation signature — that is a new finding, not a failed repair.
- Silent numerical drift at small N → caught by the focused blocked-vs-dense
  parity test at strict tolerance.
- Run passes but misleads: only if validity screens are weaker than believed;
  unchanged screens, unchanged harness, so no new risk introduced.

## What will not be concluded

No score-estimator correctness, no variant ranking, no variance rate, no
promotion of any route, no HMC/NeuTra/default readiness. One seed;
continuous metrics descriptive.

## Exact commands

Environment: `/home/chakwong/anaconda3/envs/tftwogpu`, GPU1
(RTX 4080 SUPER) via `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, memory growth verified in-harness.

1. `CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/highdim/test_genut_sqmc_score_blocking.py -q`
2. `python -m py_compile bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`
3. Fresh GPU pairs: harness `--stage smoke --particle-counts {1008,4032}`
   `--routes repaired_permutation --resets trust_region`
   `--transport-plan {dense,streaming}` (four rows, fresh attempt dirs).
4. `--particle-counts 16128 --transport-plan streaming` repaired_permutation.
5. If 4 passes resource/validity gates: the remaining three variants
   (`iid_dual_cap previous_inverse_cdf repaired_fixed_previous_controls`).

Budget: inherit the parent campaign's remaining ceilings (~7.8 GPU-hours for
`N=16128` rows; repair 2 of 2 is consumed by this implementation change being
authorized as a new owner-directed task, so subsequent localized failures
stop for direction). Artifacts continue under
`docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/` with fresh
attempt directories.

## Skeptical pre-execution audit

- Wrong baseline? No: blocked-vs-dense marks parity test is the correct
  invariant; old-graph trajectory equality is correctly excluded (chaotic
  route, established 2026-08-19).
- Proxy promotion? No promotion exists; feasibility only.
- Missing stop condition? OOM, invalidity, and test failure all stop.
- Unfair comparison? Dense/streamed pairs rerun fresh on the same source so
  both sides include the repaired recursion.
- Hidden assumption: "TensorArray-in-while compiles under jit" — already
  true in this repo's transport module on this exact TF build.
- Environment mismatch? Same env, device, and controls as the parent runs.

Audit verdict: PASS. Proceed.

## Execution log

- 2026-08-19: Implementation applied to
  `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`
  (`standard_pairwise_backward_marks`): Python block loop replaced by
  `tf.while_loop` + static-shape `TensorArray`, `parallel_iterations=1`,
  per-block arithmetic preserved verbatim (verified by read-back); block
  order preserved (`stack` + reshape = former `concat`). Static
  `_block_slice` helper mirrors the transport module's `_static_row_slice`.
- 2026-08-19: First verification attempts blocked before process creation by
  a Claude Code permission-classifier timeout
  (`claude-sonnet-5 temporarily unavailable`). Same operational class as the
  Codex approval-gateway failures: no process created, no evidence about the
  code. Retrying; if persistent, verification resumes when the gateway
  recovers.
