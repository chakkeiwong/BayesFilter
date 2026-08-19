# Austria GenUT Graph-Mode Divergence Localization Result

Date: 2026-08-19

Plan:
`docs/plans/bayesfilter-austria-genut-graph-mode-divergence-localization-plan-2026-08-18.md`

Wrapper runner (new, diagnostic lane only):
`docs/benchmarks/run_genut_austria_graph_mode_bisect_20260818.py`
(reuses the frozen root-cause runner's `_endpoint`, identity guards, and
manifest by import; compile check passed).

Frozen scope unchanged throughout: commit
`dae37183bf4421682b2ad991e2dc0d0f3c53f260`, current source
`cubature_genut_batch_tf.py` = `ae8cbfb...a976e`, Austria SIR `N=1008`,
FP32, TF32 on, deterministic ops, memory growth verified, RTX 5080 (GPU 0),
`tftwogpu` TensorFlow `2.20.0-dev0+selfbuilt`, targets constructed on CPU.
Every process: `status=COMPLETE`, frozen identity guard `PASS`, finite unless
stated, fresh output directory, nothing overwritten.

## Run Manifest

| Process | Artifact | Cases | Wall |
|---|---|---|---|
| P1 ladder | `graph_bisect_attempt01/ladder_t2_t5_graph.json` | `(2,0) (2,4) (5,0) (5,4)` graph | 282 s |
| P3 bisection | `graph_bisect_attempt02/bisect_t3_t4_steps.json` | `(3,0) (4,0) (2,1) (2,2)` graph | 164 s |
| P4 grappler probe | `graph_bisect_attempt03/grappler_meta_off.json` | `(3,0) (2,1) (2,4) (5,0)` graph, `disable_meta_optimizer=True` | 340 s |
| P5 XLA arm | `repair_validation_attempt09/endpoint_gpu0_xla.json` | `(20,4) (1,0)` xla (memo step-6 command) | 4173 s |

All artifacts under
`docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/`.

## Findings

### F1 (Q2): the graph-mode within-mode inequality reproduces at tiny scopes and does NOT require the higher-moment correction

Within-mode `value_only` vs score-carried value, default grappler, graph mode:

| Horizon | Steps | exact_equal | max abs err | ~ULP |
|---:|---:|---|---:|---:|
| 2 | 0 | true | 0 | 0 |
| 2 | 1 | false | 3.05e-5 | 4 |
| 2 | 2 | false | 3.05e-5 | 4 |
| 2 | 4 | false | 1.53e-5 | 2 |
| 3 | 0 | **false** | 2.59e-4 | 34 |
| 4 | 0 | **false** | 1.37e-4 | 9 |
| 5 | 0 | false | 9.25e-3 | 606 |
| 5 | 4 | false | 7.15e-2 | 4688 |
| 20 | 4 (prior attempt08) | false | 5.62e-1 | 9209 |

Smallest reproducing cases: `T=2, steps=1` (correction lane) and
`T=3, steps=0` (recursion lane, no correction at all). The prior session's
claim that the graph divergence "localizes to the four-step higher-moment
correction" was **wrong relative to that claim**: the `T=1` control was too
weak to detect the recursion-lane divergence, which needs `T>=3` to appear at
FP32 visibility. Both the horizon recursion and the correction loop are
affected. The discrepancy grows roughly monotonically with horizon and steps,
consistent with the known ill-conditioned amplification.

### F2 (Q3): disabling the grappler meta-optimizer restores bitwise identity in every probed failing case

Same cases, same process pattern, `tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})`
applied before any tracing:

| Horizon | Steps | default grappler | meta_off |
|---:|---:|---|---|
| 3 | 0 | unequal (34 ULP) | **exact, 0 ULP** |
| 2 | 1 | unequal (4 ULP) | **exact, 0 ULP** |
| 2 | 4 | unequal (2 ULP) | **exact, 0 ULP** |
| 5 | 0 | unequal (606 ULP) | **exact, 0 ULP** |

Classification: the observed non-XLA graph divergence is **consistent with**
grappler graph rewrites (e.g. arithmetic simplification/layout/remapper)
transforming the value-only graph and the JVP-carrying graph differently, so
the two graphs execute different FP32 primal arithmetic. It is **not**
consistent with `ForwardAccumulator` tracing emitting a different primal op
sequence, since with rewrites disabled the two traced graphs are bitwise
identical on every probed case. Per the plan's pre-mortem, this is stated as
"consistent with", not proof; the specific responsible grappler pass was not
isolated (that would be a follow-up ablation over individual optimizer
toggles, which the plan did not include).

### F3 (Q1): the XLA arm FAILS at the frozen scope — nonfinite and invalid

`repair_validation_attempt09/endpoint_gpu0_xla.json`, memo step-6 command:

- `T=1`, zero steps: value `-31.12767029`, `exact_equal=true`, 0 ULP, finite,
  valid — and bitwise equal to the eager and graph `T=1` values.
- `T=20`, four steps: **both** the value-only endpoint and the score endpoint
  returned nonfinite values and scores with `program_valid=[false]`.
  `finite_pattern_equal=true` between the two endpoints — the fail-closed
  contract behaved coherently on both routes.
- Score compile alone took >40 min (XLA "very slow compile" alarm, 10m+
  single-module compile, ptxas register spills); total wall 69.5 min, inside
  the 90-min cap.

This is a hard veto observation for XLA at the frozen claim scope on the
current source: the repository-default execution mode cannot currently
produce a valid `T=20` Austria endpoint, independent of the identity
question. Note the historical stale attempt06 XLA artifact (older source
`606897...`) was finite (`-682.3775024`); the nonfinite XLA outcome is
specific to the current source, the current XLA pipeline, or their
interaction — not checked which.

Explanatory only: the `T=1` score vectors differ slightly across all three
modes (eager/graph/XLA min component `-5.2127638 / -5.2127652 / -5.2127647`);
no reviewed score tolerance exists, recorded without interpretation.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Classify non-XLA graph inequality as a grappler-rewrite arithmetic difference between value and JVP graphs | Reproduced at `T=2..5`; restored to bitwise identity in 4/4 cases with meta-optimizer off | No hard veto in P1–P4; all processes COMPLETE/finite/valid | Which specific grappler pass; whether meta_off restores identity at full `T=20` (not run — cost) | Owner scope decision on graph mode's claim-bearing status; optionally a per-pass ablation and/or one full-scope `T=20` meta_off confirmation | No repair promoted; no tolerance created; graph mode NOT accepted |
| Record XLA `T=20` as nonfinite/invalid on current source | Within-mode identity untestable — validity veto fires first | HARD VETO: nonfinite value and score, `program_valid=false`, both endpoints | Whether nonfiniteness is XLA compile arithmetic vs a genuinely invalid intermediate (e.g. Cholesky failure) under XLA reductions | Bounded XLA localization (e.g. correction-step/horizon ladder under XLA; inspect which validity guard fires) under a fresh plan | XLA failure is NOT evidence against eager/CPU correctness, the repair, or the research direction |
| Prior session's "localizes to the correction loop" claim corrected | `T=3, steps=0` fails identity with no correction steps | — | — | Corrected in this note and the checkpoint | — |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | P1–P4 clean. P5 XLA `T=20`: nonfinite/invalid — hard veto for XLA at frozen scope. |
| Statistically supported ranking | Not applicable — deterministic compiler-mode diagnostics. |
| Descriptive-only differences | Gap magnitudes, ULP counts, growth with `T`/steps, cross-mode `T=1` score spread, compile/trace wall times. |
| Default readiness | Worse than before this campaign: the repository-default XLA target is invalid at the frozen scope on current source. Nothing is promotable. |
| Next evidence needed | (a) owner decision on graph-mode scope; (b) XLA nonfiniteness localization; (c) optional grappler per-pass ablation; (d) only after those: revisit the three-mode confirmation contract. |

## Post-Run Red Team

Strongest alternative explanation for F2: meta_off changes op layout such
that both graphs coincidentally fall into the same kernels, rather than the
meta-optimizer being the mechanism that splits them; 4/4 restoration across
both lanes (recursion-only and correction) makes coincidence less likely but
the per-pass ablation was not run. For F3, the weakest part of the evidence
is that no intermediate diagnostics exist under XLA (endpoint-only by
design), so "which guard fired" is unknown; the ptxas register spills and the
extreme compile time suggest an unusually aggressive fusion of the unrolled
recursion, but that is speculation, marked as such. An observation that would
overturn F2: a failing case that stays unequal with the meta-optimizer off.

## Nonclaims

No repair is proposed or promoted. No tolerance was created. Graph mode is
not accepted or rejected as a claim-bearing arm — that is a human scope
decision. XLA correctness, dual-cap correctness, posterior correctness,
NeuTra/HMC/tuning readiness, and default readiness are all not concluded.
The eager GPU pass and exact CPU derivative authority are unaffected.
