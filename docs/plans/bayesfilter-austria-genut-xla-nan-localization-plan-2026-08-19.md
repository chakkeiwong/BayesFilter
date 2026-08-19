# Austria GenUT XLA Nonfiniteness Localization Plan

Date: 2026-08-19 (living document: updated after each phase per user
instruction; phase-result annotations are appended, never rewritten)

> **Execution status 2026-08-19 (updated): RESOLVED — the harness classifier
> outage cleared after ~14 rejected launch attempts and execution proceeded.**
> Historical record of the blocker (accurate for its window): the Claude Code
> permission classifier became unavailable and rejected consecutive P0 launch
> attempts before process creation; read-only operations worked; no process,
> artifact, GPU compute, or scientific budget was consumed. Same
> infrastructure class as the historical approval 502/404 rejections — not
> evidence about the GPU, TensorFlow, or the endpoint. Phase results are
> appended at the end of this document.

Prior evidence:
`docs/plans/bayesfilter-austria-genut-graph-mode-divergence-localization-result-2026-08-19.md`
(F3: XLA `T=20,steps=4` returned NaN value AND score with
`program_valid=false` on both endpoints;
`repair_validation_attempt09/endpoint_gpu0_xla.json`).

Authorization: user requested plan + ladder execution with minimal stopping,
2026-08-19. Priority ruling: this XLA NaN is problem #1.

## Code-Trace Foundation (established before this plan)

The NaN is fail-closed masking, not necessarily raw arithmetic NaN:
`batch_finite_value` returns `tf.where(valid, total, NaN)`
(`cubature_genut_batch_tf.py:1560`); the score endpoint masks identically
(`:1963`). Therefore the question is which validity conjunct went false.
Per-step conjuncts, ANDed over 20 steps:

- Stage A filtering (`:1444`): particles/log-likelihood/logsumexp/weights
  finite; `|sum(weights)-1| <= 1e-4`.
- Stage B Sinkhorn (`:337`): row masses finite and `> 1e-7`; barycentric
  finite; column TV `<= 1e-4` after 16 fixed iterations.
- Stage C Contract-E restore (`:392`, `ledh_contract_e_reset_tf.py:44`):
  `min eigvalsh(target_cov - transported_cov) + ridge(1e-5) > 0`; three
  ridged Choleskys finite with positive diagonals; particles finite.
- Stage D higher-moment (`:1336`): output finite; contains two UNRIDGED
  Choleskys (`:1273`, `:1288`) and the ill-conditioned LM moment solve.

Key facts constraining hypotheses: XLA `T=1,steps=0` passed bitwise equal to
eager (single-step XLA machinery is clean, including eigvalsh + 3 Choleskys);
value-only and score endpoints failed together (`finite_pattern_equal=true`,
primal validity failure, not a tangent pathology); eager `T=20` passes but the
endpoint runner discards all margin diagnostics; XLA score compile showed
40+ min mega-fusion with register spills; the grappler campaign measured
~4-orders ULP amplification over 20 steps in this recursion.

## Hypotheses

- H-A thin-margin guard flip (highest prior): XLA-reassociated (and
  TF32-seeded) drift accumulates along the horizon until a threshold conjunct
  crosses. Sub-candidates by fragility: H-A1 gap-eigenvalue test (`+1e-5`
  headroom on the smallest eigenvalue of a covariance DIFFERENCE); H-A2
  Sinkhorn column TV `<=1e-4`; H-A3 weight-sum `<=1e-4`.
- H-B raw decomposition NaN: XLA's HLO cholesky/eigvalsh (not cuSOLVER) hits
  a non-positive pivot on a drift-degraded matrix at some later step.
- H-C correction-loop-specific: the unridged Choleskys / LM solve in Stage D
  are the failure site; discriminated by `T=20,steps=0` under XLA.
- H-D XLA miscompilation (low prior): the register-spilled fusion computes a
  logically wrong result.
- H-E TF32-as-seed (modifier): the drift seed is TF32 matmul precision;
  tested by a TF32-off XLA arm. (Not checked previously for XLA; the graph
  grappler verdict does not transfer.)

## Evidence Contract

- Question: which validity conjunct fails first under XLA at the frozen
  scope, at which smallest (horizon, steps), and is it removed by TF32-off?
- Comparator: eager same-case diagnostics (margins in the passing mode).
- Primary criteria: per-case boolean `program_valid` plus, in
  diagnostic-serialized arms, which recorded margin sits nearest or across
  its threshold. Localization only; nothing passes any confirmation gate.
- Hard vetoes per process: frozen identity mismatch, source hash drift,
  wrong build/device/dtype, unverified memory growth, `status != COMPLETE`.
  An XLA case returning invalid/NaN is a RESULT here, not a veto.
- Explanatory only: margin magnitudes, compile/wall times, TF32 state
  effects, cross-mode diagnostic differences, per-step traces from any
  re-implemented diagnostic program (a separately compiled program is
  "consistent with" evidence about the endpoint program, never identity
  evidence — XLA may compile the two differently).
- Not concluded regardless of outcome: any tolerance/ridge change (repair
  proposals go to a fresh reviewed scope), XLA acceptance or permanent
  rejection, graph-mode scope, dual-cap/NeuTra/HMC/tuning/posterior/default
  claims.
- Artifacts: fresh JSON per process under
  `docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/xla_nan_attempt*/`.
  Terminal interpretation appended here and to the checkpoint.

## Default/Assumption Audit

| Choice | Provenance | Failure mode | Mitigation |
|---|---|---|---|
| New diagnostic runner serializes the endpoint's own diagnostics dict | Frozen runner discards it | Wrapper drift from frozen arithmetic | Wrapper reuses `batch_finite_value`/`_score` and base guards/manifest by import; value SHA must match frozen-runner artifacts for repeated cases |
| Small-T XLA first | Compile cost scales with unrolled horizon | Defect may need T=20 to reproduce | Ladder ends with `T=20,steps=0` and, only if needed, diagnostic-serialized `T=20,steps=4` |
| TF32-off is a legitimate arm | Repo policy keeps FP32-no-TF32 as explicit comparison mode | Misreading it as a default change | Recorded as reference arm; manifest records TF32 state after the flag is set |
| Aggregated (min/max over steps) diagnostics may under-localize | Value program aggregates | Cannot name the failing STEP, only the failing GUARD class | Acceptable for phase goals; per-step trace deferred to a conditional phase with its compilation-sensitivity caveat declared |

## Phases And Budget

Exact P0 command (blocked at the harness boundary on 2026-08-19; reuse
verbatim on resume, then proceed to P1a/P1b with the case lists below and
fresh `xla_nan_attempt02/`, `attempt03/` output files):

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_DETERMINISTIC_OPS=1 \
MPLCONFIGDIR=/tmp/bayesfilter-matplotlib CUDA_VISIBLE_DEVICES=0 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
docs/benchmarks/run_genut_austria_xla_nan_localization_20260819.py \
--device gpu --gpu-index 0 \
--cases 20:4:eager 1:0:eager \
--output docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/xla_nan_attempt01/eager_margin_audit.json
```

P1a cases: `2:0:xla 3:0:xla 5:0:xla 2:4:xla 5:4:xla`. P1b case: `20:0:xla`.
P2 adds `--tf32 off` on the smallest failing case. Summarizer:
`python3 <scratchpad>/summarize_xla_nan.py <artifact.json>` (margins vs
thresholds per case).

GPU lane unchanged: RTX 5080 (GPU 0), `tftwogpu`, deterministic ops, memory
growth; TF32 on except the declared TF32-off arm. Trusted execution.

- P0 eager margin audit: eager `T=20,steps=4` + `T=1,steps=0` with full
  diagnostics serialized. Answers: which margins are thin in the passing
  mode. Cap 10 min.
- P1a XLA small ladder: `(2,0) (3,0) (5,0) (2,4) (5,4)` xla with
  diagnostics. Smallest failing case + guard class. Cap 60 min.
- P1b XLA `(20,0)`: discriminates H-C. Cap 60 min.
- P2 (conditional): TF32-off XLA rerun of the smallest failing case (or of
  `(20,4)` if nothing smaller fails). Tests H-E. Cap 90 min.
- P3 (conditional, only if P0–P2 leave the guard unidentified): per-step
  conjunct trace, separately compiled, caveat declared. Cap 60 min.
- Campaign ceiling 4.5 h. Stop after 3 consecutive launch failures. Kill any
  process exceeding its cap; preserve `RUNNING` JSON as non-evidence.

## Pre-Mortem

Small-T XLA passes could be compile-scale artifacts (fusion decisions differ
at T=20) — a clean small ladder plus failing `(20,0)`/`(20,4)` does NOT prove
Stage D innocence at small T. Diagnostics aggregated over steps can hide
which step fails. TF32-off changing compile decisions (not just precision)
would confound H-E — record as "consistent with". The wrapper could
accidentally diverge from the frozen endpoint — mitigated by SHA comparison
on repeated cases (eager `T=1` and `T=20` values must match attempt07
bitwise).

## Skeptical Audit

Passed 2026-08-19: comparator declared (eager margins); criteria are booleans
plus recorded margins, no proxy promotion; XLA failure is data, not veto;
budget/caps/stop conditions explicit; wrapper validated against frozen
artifacts on overlapping cases; per-step trace's compilation-sensitivity
caveat predeclared; all repair decisions explicitly out of scope.

## Phase Annotations (appended per execution)

### P0 eager margin audit — COMPLETE (xla_nan_attempt01, 85 s)

The harness classifier outage resolved; P0 ran on retry ~15. Eager `T=20,4`
margins in the passing mode (both endpoints identical and valid):

- `minimum_covariance_gap_eigenvalue = 0.2418` — NOT thin. The gap guard has
  ~4 orders of headroom over the `1e-5` ridge. H-A1's "1e-5 headroom"
  fragility framing was wrong for the trajectory-minimum: the aggregated
  minimum over 20 steps is 0.24. (Caveat: aggregation hides per-step values,
  but the MINIMUM is the binding one, so the guard is robust in eager.)
- `max_col_residual = 2.79e-6` vs `1e-4` TV guard — ~36x headroom (H-A2
  plausible only under >30x XLA-induced degradation).
- `max_mean_residual = 1.68e-4`; skew/kurtosis residuals O(3)/O(35);
  pre/post-cap RMS 56.45 (large displacements, but explanatory).
- `T=1,0` margins similarly healthy (`min_gap = 1.297`).

Consequence: the H-A sub-ranking is revised. No eager margin is within an
order of magnitude of its threshold, so a guard flip under XLA requires
either large drift (not ULP-level) or a raw-NaN path (H-B/H-C), or an
XLA-specific computation of the guard quantity itself diverging far more
than the value does. H-A1-as-thin-margin is DEMOTED; H-B/H-C rise.

### P1a XLA small ladder — COMPLETE (xla_nan_attempt02, 810 s)

`(2,0) (3,0) (5,0) (2,4) (5,4)` all COMPLETE, finite, program_valid=true on
BOTH endpoints. min_gap values healthy (0.26–1.30). Register-spill ptxas
warnings already present at these small scopes (loop_add_fusion_*).

- The XLA nonfiniteness does NOT reproduce at `T<=5` with 0 or 4 steps.
  Small-scope XLA decomposition machinery is clean — consistent with the
  earlier `T=1` pass, now extended through `T=5` including the correction
  loop under XLA.
- NEW explanatory observation: `(5,0)` has `exact_equal=false` WITHIN XLA
  (value-only vs score-carried value differ) while `(2,0) (3,0) (2,4) (5,4)`
  are exact. XLA therefore also exhibits scope-dependent within-mode value
  discrepancy, previously seen only in non-XLA graph mode. Recorded; no
  tolerance created; does not affect the NaN question directly.
- H-D (generic small-scale miscompilation) further weakened. The failure is
  horizon-accumulation-dependent: next discriminator is `T=10` and
  `T=20,steps=0` (P1b, launched).

### P1b XLA horizon escalation — COMPLETE (xla_nan_attempt03, 2939 s)

`(10,0) (10,4) (20,0)` all COMPLETE, finite, program_valid=true on BOTH
endpoints under XLA. `(20,0)` alone took 1847 s (compile-dominated).

Consequences for the hypothesis set:

- H-C REFUTED in its strong form: `T=20, steps=0` is VALID under XLA, so the
  horizon recursion alone does not trigger the NaN. The failure requires the
  four-step correction loop AT the full `T=20` horizon (`(10,4)` and `(5,4)`
  pass). The interaction is horizon-depth x correction-loop, not either
  alone.
- H-A demoted further (P0 showed fat margins; every intermediate scope
  passes cleanly rather than degrading toward a threshold).
- Remaining candidates: H-B (raw decomposition NaN inside the correction
  loop's unridged Choleskys at late-horizon conditioning, XLA HLO
  decomposition), H-D (fusion-scale-specific miscompile at the largest
  program), H-E (TF32 seed pushing a late-step matrix indefinite).
- Explanatory: within-XLA `exact_equal` is scope-dependent — false at
  `(5,0)` and `(10,0)`, true at `(2,0) (3,0) (20,0)` and all steps-4 passing
  cases. XLA shows the same compiler-rewrite value/JVP asymmetry class as
  non-XLA graph mode, at ULP scale, not tied to the NaN.

Plan refresh: P2 is now a diagnostic-serialized XLA `(20,4)` rerun
(xla_nan_attempt04, launched) to name the failing guard class from the
endpoint's own diagnostics dict — the aggregated min/max values survive
NaN-masking of the scalar because they are returned alongside it. If the
diagnostics themselves are NaN-poisoned, the failing stage is still
identifiable by which aggregates are finite. P3 becomes the TF32-off arm on
`(20,4)`.

### P2 diagnostic-serialized XLA (20,4) — COMPLETE (xla_nan_attempt04, 3175 s)

The NaN reproduced with full diagnostics captured. Identical pattern on both
endpoints:

| Diagnostic | Value | Verdict |
|---|---|---|
| `minimum_covariance_gap_eigenvalue` | 0.6619, finite | Stage C gap guard HEALTHY |
| `max_col_residual` | 2.65e-5, finite | Stage B Sinkhorn HEALTHY (under 1e-4) |
| `max_mean_residual`, `max_row_residual` | 2.4e-4 / 1.5e-7, finite | Stage B/C residuals healthy |
| `minimum_pearson_feasibility_margin` | 0.1654, finite | feasibility margin positive but 7x smaller than eager's 1.178 |
| `minimum_finite_particle_upper_margin` | 958.6, finite | healthy |
| `maximum_skew_residual`, `maximum_kurtosis_residual` | NaN | Stage D POISONED |
| `maximum_diagonal_pre/post_cap_particle_rms` | NaN | Stage D correction-iteration displacement POISONED |
| `maximum_diagonal_scaled_system_condition` | 0.0 | LM branch NOT taken (lm_damping=0 normal-equation path) |

Localization: the raw NaN originates INSIDE the Stage D higher-moment
correction loop (`_higher_moment_batch_value` /
`_shape_iteration_batch_primal`), specifically in or before the displacement
computation (pre_cap_rms is already NaN), while every Stage A/B/C aggregate
along the whole 20-step trajectory stays finite and healthy. The candidate
raw-NaN sites in that lane are the unridged Choleskys
(`cubature_genut_batch_tf.py:1273,:1288,:746`) and the floor-ridged
2x2 `tf.linalg.solve` normal-equation path (`:705-714`, active since
condition diagnostic is 0.0 => lm_damping=0).

Hypothesis update: H-B (raw decomposition/solve NaN inside Stage D under
XLA-compiled arithmetic at full horizon) is now the leading and only
strongly supported hypothesis. H-A is refuted for all guard classes (all
threshold-guard aggregates finite and far from thresholds in the failing
run itself). H-C is revived in a narrowed form: not "correction loop alone"
(small-T corrections pass) but "correction loop fed by late-horizon
particle states under XLA arithmetic". H-D remains unexcluded only insofar
as the Stage-D NaN could itself be a miscompiled fusion; H-E (TF32 seed)
now under direct test in P3 (xla_nan_attempt05, launched).

Interpretive note (predeclared aggregation caveat applies): the diagnostics
are max/min aggregates over 20 steps, so the step at which Stage D first
produces NaN is not identified. The explanatory Pearson-margin drop
(1.178 eager -> 0.165 XLA) shows XLA arithmetic degrades the corrected
higher-moment state well before the NaN, consistent with drift-then-blowup
inside the unprotected Stage D solves rather than an abrupt miscompile.

### P3 TF32-off XLA (20,4) — COMPLETE (xla_nan_attempt05, 3159 s) — DECISIVE

With `enable_tensor_float_32_execution(False)` and everything else identical
(same XLA pipeline, same fusion/register-spill warnings, same frozen scope):
`T=20, steps=4` is FINITE and program_valid=true on BOTH endpoints.
Value `-680.6785889`. Every Stage D diagnostic that was NaN under TF32 is
healthy: skew residual 1.26, kurtosis residual 16.59, pre/post-cap RMS 35.15,
Pearson margin 1.044 (vs 0.165 TF32-on before the blowup).

Hypothesis resolution:

- H-E CONFIRMED as the seed: the XLA T=20 NaN requires TF32. TF32 matmul
  precision inside XLA-compiled arithmetic degrades the late-horizon
  higher-moment state until a Stage D solve/Cholesky produces raw NaN.
- H-B CONFIRMED as the mechanism/site: raw nonfiniteness inside Stage D's
  unprotected solves (unridged Choleskys :1273/:1288/:746 or the 2x2
  normal-equation solve), per P2's poisoned-aggregate pattern.
- H-D effectively refuted: the same XLA mega-fusion pipeline compiles and
  executes correctly TF32-off — consistent with precision seeding, not
  miscompilation. (Same ptxas spills, valid output.)
- H-A refuted; H-C holds only in the narrowed interaction form.

Explanatory (recorded, no tolerance): TF32-off XLA still has
`exact_equal=false` within-mode (value-only `-680.6785889` differs from the
score-carried value) — the compiler value/JVP program-identity issue is
INDEPENDENT of the NaN and persists TF32-off. Also note eager TF32-on value
`-683.0019` vs XLA TF32-off `-680.6786` vs CPU `-680.7359`: the XLA TF32-off
value is far closer to the CPU value; the ~2.3 log-unit eager-GPU offset is
TF32-dominated arithmetic difference, not checked further here.

Chain of causation (established): TF32 seed -> horizon amplification
(needs T=20; T<=10 passes) -> Stage D unprotected solves blow up ->
fail-closed guard correctly masks to NaN. The guard system worked as
designed; the failure is real arithmetic degradation, not a guard bug.
