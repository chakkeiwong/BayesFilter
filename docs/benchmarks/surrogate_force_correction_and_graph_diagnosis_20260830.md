# Surrogate-force LEDH HMC: graph-size diagnosis and correction of the record

Date: 2026-08-30
Scope: LGSSM canonical batch-fused target, N=252, dim=3, horizon=50, substeps=12
(the horizon=10 truncation used by the third diagnostic is marked where it
appears).
Diagnostic scripts, all run this session:
`docs/benchmarks/diagnose_graph_size_20260830.py` (node counts, unroll scaling),
`docs/benchmarks/diagnose_eval_time_20260830.py` (warm eval time, monotonic
clock), `docs/benchmarks/diagnose_direction_cost_scaling_20260830.py`
(direction-count cost scaling).
Device policy: CPU-only, `CUDA_VISIBLE_DEVICES=-1` set before TF import. GPU
devices were intentionally hidden; these are host graph-construction and CPU
execution numbers, and no GPU claim is made from them.

Measurement caveat: the first version of the graph-size diagnostic reported
node-count ratios of 1.00 for all three cases and a negative warm time. Both
were defects in the diagnostic, not findings — `theta` was reshaped outside the
`GradientTape` block so no backward pass was traced at all, and `time.time()`
stepped backwards on this WSL2 host. Fixed by moving the reshape inside the tape
and switching to `time.perf_counter()`. The numbers below are from the corrected
scripts.

This note answers three questions about the surrogate-force implementation and
retracts unsupported claims in six earlier documents.

## 1. Why does each gradient issue five LEDH calls?

Because the kernel returns one directional derivative per call. From
`bayesfilter/highdim/ledh_canonical_batch_fused_tf.py:65`:

> `theta: [B, P]; theta_directions: [B, P]` (the score direction per row —
> multi-parameter scores sweep directions across calls, matching the
> analytical-score convention).

A 5-parameter gradient therefore needs five directions. My target wrapper swept
them with a Python `for p in range(param_dim)` loop, giving 5 score calls plus 1
value call = 6 LEDH calls per HMC gradient evaluation. This is the API
convention, not a defect in the kernel.

The five calls can be collapsed to one by tiling theta to `B=5` against
`tf.eye(5)` directions. Measured parity, same target and theta:

| | value | gradient |
|---|---|---|
| swept (6 calls) | +1.277910975060e+02 | `[7.79103373, 1.4332902, -1.58852186, -0., -0.]` |
| batched (2 calls) | +1.277910975060e+02 | `[7.79103373, 1.4332902, -1.58852186, -0., -0.]` |

Max absolute gradient difference 2.665e-15 — agreement at float64 round-off.

Important, and the opposite of what I first assumed: **batching the directions
is an arithmetic pessimization.** Measured, full horizon=50, N=252, CPU:

| variant | trace + first eval | warm eval (median of 5) |
|---|---|---|
| swept (6 LEDH calls) | 281.6 s | **0.185 s** |
| batched (2 LEDH calls) | 115.3 s | **0.485 s** |

Batching cuts trace time 2.4× and the graph 3×, but costs 2.62× more per warm
evaluation. The mechanism is common-subexpression elimination, confirmed by a
direction-count scaling probe at horizon=10
(`docs/benchmarks/diagnose_direction_cost_scaling_20260830.py`), where the
reference unit is one score call (one primal + one tangent):

| variant | D | nodes / base | warm / base |
|---|---|---|---|
| swept | 1 | 1.00 | 0.97 |
| swept | 2 | 2.00 | 1.05 |
| swept | 3 | 3.00 | 1.14 |
| swept | 5 | 5.00 | **1.23** |
| batched | 1 | 1.00 | 0.97 |
| batched | 2 | 1.00 | 1.59 |
| batched | 3 | 1.00 | 1.98 |
| batched | 5 | 1.00 | **3.51** |

The five swept score calls share a bitwise identical primal recursion (UKF sigma
propagation, flow substeps with their Cholesky solves, weights) and differ only
in the tangent, so graph optimization deduplicates four of the five primals: the
swept graph grows exactly linearly in D (1.00, 2.00, 3.00, 5.00) while its
executed arithmetic grows only 1.23× at D=5. The marginal cost of an extra
direction is ≈(1.23−0.97)/4 ≈ 0.065 of a full call, i.e. the tangent recursion
is roughly 6.5% of a call and the primal dominates. Tiling theta to `B=5` makes
the rows distinct data, so nothing can be deduped and the batched variant pays
five full primals (3.51× at D=5, sublinear in D only because larger tensors
vectorize better).

Consequence for design: the swept form's arithmetic is already close to the
theoretical floor for this kernel. A fused multi-direction tangent recursion
(tangents carried as `[m, P, dim]`, primal computed once) would cost about
1 + 5×0.065 ≈ 1.3× a single call — essentially what CSE already delivers at
1.23×. So the multi-direction kernel change buys graph size and independence
from a compiler optimization, not arithmetic. The redundancy worth removing is
in the *graph*, not the FLOPs.

## 2. Analytical recursion or autodiff?

Analytical forward-mode recursion. No autodiff tape is involved in the score.
Evidence in the kernel:

- Module docstring: "NO autodiff (C-9): score is the analytical recursion,
  fused."
- `PerPointScoreModel` carries explicit tangent callbacks:
  `transition_mean_tangent_fn`, `observation_tangent_fn`,
  `observation_log_density_tangent_fn`.
- `chol_diff` (line 132) is a hand-written Cholesky derivative, which is what
  you write precisely when you are *not* differentiating through
  `tf.linalg.cholesky` with a tape.
- Tangent state `d_states`, `d_covariances`, `d_total` is carried forward
  alongside the primal state through the recursion.

The `tf.custom_gradient` in my HMC target does not create a tape over the
recursion either. It binds an exact-config forward value to a damped-config
analytical score, so TensorFlow never traverses the 50-step filter backwards.

## 3. The user's inference was right: recursion memory is not the problem

Forward-mode recursion carries fixed-shape state, so its *runtime* memory is
O(1) in horizon — for N=252, dim=3, float64, the carried state
(`states`, `covariances`, `d_states`, `d_covariances`, plus accumulators) is on
the order of 100 KB. It cannot explain a 45 GB process.

The cost is **host graph memory from Python-loop unrolling**, multiplied by the
call count. `ledh_canonical_batch_fused_tf.py:146` is `for time_index in
range(horizon)` and line 219 is `for step_index in range(substeps)` — both
Python, so all 50 timesteps and all 600 flow-substep bodies are emitted as
static graph nodes. Measured:

| case | nodes | GraphDef | trace | RSS after |
|---|---|---|---|---|
| A: one LEDH call (value+score), B=1 | 110,628 | 9.98 MB | 16.6 s | 1,336 MB |
| B: value + 5 swept directions | 663,766 | 61.24 MB | 101.8 s | 5,245 MB |
| C: value + 1 batched-direction call | 221,484 | 20.17 MB | 41.5 s | 6,644 MB |

B/A = 6.00 exactly, C/A = 2.00 exactly — the node count is exactly the LEDH
call count, confirming the mechanism. Batching directions shrinks the graph 3×.

Unroll scaling of a single call confirms the loops are the driver:

| horizon | substeps | bodies | nodes | nodes/body |
|---|---|---|---|---|
| 5 | 12 | 60 | 11,133 | 185.6 |
| 10 | 12 | 120 | 22,188 | 184.9 |
| 25 | 12 | 300 | 55,353 | 184.5 |
| 10 | 3 | 30 | 9,768 | 325.6 |
| 10 | 6 | 60 | 13,908 | 231.8 |

Node count is linear in horizon at fixed substeps (≈2,213 nodes per timestep)
and linear in substeps at fixed horizon (≈138 nodes per flow substep, plus
≈563 fixed nodes per timestep for UKF predict and weights). Nothing here scales
with a tape.

Peak RSS for the graph-size diagnostic alone, which traces eight graphs (three
cases plus five unroll probes), was 11,940 MB. The production HMC target is
traced more than once by `tfp.mcmc.sample_chain` (current state, proposed state,
and step-size adaptation contexts), each instance carrying a 663,766-node graph.
That is the 45 GB, and it is also why `jit_compile=True` died in LLVM with
"Unable to allocate section memory" — that was the **XLA compiler running out of
host memory on a 663,766-node graph**, not a device-memory failure. Note that
the XLA failure is a compile-time consequence of graph size only; the executed
arithmetic is 1.23× a single score call (§1), so nothing about the *work* was
too large for the device.

### Correction: the N=10000 artifact does not contradict this

I earlier read `gpu_memory_info_after.peak: 20504576` in
`experimental-batched-ledh-pfpf-ot-streaming-lgssm-gpu0-b1-t100-np10000-d20-m20-activeall-callback-2026-06-15.json`
as "20.5 GB". It is ~20 MB. The correct reading strengthens rather than weakens
the user's point: N=10000, d=20, T=100 ran under `jit_compile: true` in about
20 MB of device memory.

The reason that run was fine and this one is not has nothing to do with particle
count:

- it is a different route (LEDH-PFPF-OT streaming transport, not the canonical
  batch-fused analytical-score kernel);
- it was **one isolated call**, not a target embedded in an HMC chain graph
  traced 6× per gradient across multiple TFP contexts;
- it used `particle_chunk_size: 128`, `row_chunk_size: 512`,
  `col_chunk_size: 512`, `plan_mode: "streaming"` — and
  `make_canonical_neutra_target(model, *, particle_count, noise_seed, substeps)`
  exposes no chunking parameters at all.

My earlier framing, that "batch-native LEDH is too memory-intensive at N=1008",
was wrong. Particle count is not the binding constraint; graph node count is.
N=1008 vs N=252 changes tensor extents, not node count, which is why the N=252
reduction bought only a 2.17× wall-time change and did not solve the memory
problem.

## 4. Retractions

The following claims appear in `phase2_implementation_summary.md`,
`phase2_memory_analysis.md`, `phase2_performance_analysis.md`,
`surrogate_force_progress_report.md`, `surrogate_force_final_results.md`, and
`surrogate_force_investigation_complete.md`. They are withdrawn.

**Retracted: "validated", "SUCCESS - All objectives achieved", "samples the
exact posterior (theory + empirical validation)", "Theory confirmed".**
The evidence is 1 chain × 10 retained draws per arm at 100% acceptance. A 100%
acceptance rate with a step size that small means the chain barely moved; it is
not evidence of correct sampling, it is evidence the sampler was not exercised.
There is no R-hat, no ESS, no seed matching across arms (seeds were
`42 + hash(arm_name) % 1000`, which is not even reproducible across processes),
and no uncertainty interval. Under the repo's Statistical Evidence Discipline
these runs support at most: the implementation runs end to end without
divergence or non-finite values. No posterior-agreement claim, no ranking, and
no correctness claim is supported. The Arm C vs Arm A RMS difference of 0.1201
is descriptive only, and calling it "reasonable" was an unsupported judgement.

**Retracted: exact arm at "λ=1e-5, δ=1e-5".** Wrong. Source
(`ledh_canonical_neutra_targets_tf.py:_diagonal_lgssm_fused_model`) sets
`process_covariance = 0.35² I` and `observation_covariance = 0.45² I` with no
ridge term. The exact arm is λ=δ=0. Step 1's own printout said
`Exact: λ=0.0e+00, δ=0.0e+00` and I documented the opposite. The damped arm's
`+1e-3 * I` on both covariances is correct as recorded.

**Newly reported, previously unflagged: the force is 3-parameter, not
5-parameter.** `transition_mean_fn` reads only `theta_rows[:, :3]`, and q, r
enter only through theta-independent frozen covariances. So ∂/∂q and ∂/∂r of the
value are structurally zero, which is what `[7.79, 1.43, -1.59, -0., -0.]`
shows. HMC was therefore moving q and r by momentum alone with no force, and any
q/r posterior numbers in the earlier documents describe a random walk in those
two coordinates, not a gradient-informed one. This invalidates the q and r
columns of the three-arm comparison specifically, independent of the sample-size
problem above.

**Also non-compliant with repo policy, and now fixed in the diagnostics:** the
step1-3 scripts used `@tf.function` with no explicit `input_signature`, contrary
to the TensorFlow Graph And Compilation Policy. The two diagnostics in this note
declare explicit signatures.

## 5. What is actually supported

- The surrogate-force pattern is wired correctly: exact value forward, damped
  analytical score backward, via `tf.custom_gradient`. Verified by parity of the
  swept and batched direction paths to 2.7e-15.
- The score is analytical forward-mode recursion, not autodiff. Verified from
  the kernel source.
- The memory and XLA failures are host graph size from Python-loop unrolling
  times LEDH call count. Verified by exact 6.00 and 2.00 node ratios and linear
  horizon/substep scaling.
- Batching the five directions into one call is numerically equivalent
  (2.7e-15) and shrinks the graph 3×, but costs 2.62× more per warm evaluation
  because it defeats the common-subexpression elimination that dedupes the five
  shared primal recursions in the swept form. Verified.
- The swept form's executed arithmetic is 1.23× a single score call at D=5, so
  it is already near this kernel's floor. The tangent recursion is ≈6.5% of a
  call; the primal dominates. Verified.
- Nothing about posterior correctness, sampler validity, or arm ranking is
  established.

## 6. Next justified action

The binding constraint is graph node count, not arithmetic. That narrows the
options to one with leverage.

1. **`tf.while_loop` over the horizon** inside the kernel, replacing
   `for time_index in range(horizon)` at
   `ledh_canonical_batch_fused_tf.py:146`. The loop carries fixed-shape state
   (`states`, `covariances`, `d_states`, `d_covariances`, `total`, `d_total`),
   and `observations`/`noises` can be indexed dynamically, so this is
   structurally feasible; it would take node count from linear in horizon to
   O(1) — roughly 110,628 → ~2,200 plus loop overhead for one call at
   horizon=50. Whether it holds numerical parity, and whether
   common-subexpression elimination still dedupes the shared primal across
   swept directions once the body is a loop rather than an unroll, are both
   **not checked**. The second question matters: if CSE cannot see into the
   loop body, the swept form loses its 1.23× arithmetic and the fused
   multi-direction tangent recursion becomes necessary rather than optional.
   This modifies the canonical kernel, so it must be added as an alternative
   route with a parity test against the unrolled implementation, not as a
   silent replacement, per the anti-fork rule.

2. **Batched directions: rejected as a graph-size repair.** It does shrink the
   graph 3×, but at 2.62× warm-evaluation cost, and it does not address the
   horizon unroll, which is the dominant term (a batched call is still 221,484
   nodes). Recording the question it failed, per the safety-guardrail rule: it
   was rejected as *the* graph-size repair on cost grounds. It is not rejected
   as a component of a multi-direction tangent recursion, where the primal
   would be computed once and the tiling redundancy would not exist — that
   remains open and untested.

Also worth fixing regardless: `substeps=12` contributes ≈138 nodes per body and
600 bodies per call. The substep loop at line 219 is the larger of the two
unrolls by body count, so a `tf.while_loop` there compounds with (1).

Only after the graph fits should the sampling question be re-opened, and then
with multiple chains, real retained draws, R-hat and ESS, arms sharing matched
seeds, and a target whose q and r coordinates actually receive a force. Until
then the surrogate-force question is open, not answered.
