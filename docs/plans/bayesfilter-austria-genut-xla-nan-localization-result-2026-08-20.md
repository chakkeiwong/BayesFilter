# Austria GenUT XLA Nonfiniteness Localization Result

Date: 2026-08-20

Plan (with per-phase annotations):
`docs/plans/bayesfilter-austria-genut-xla-nan-localization-plan-2026-08-19.md`

Runner: `docs/benchmarks/run_genut_austria_xla_nan_localization_20260819.py`
(reuses the frozen root-cause runner's endpoints/guards/manifest by import;
serializes the full diagnostics dict that the frozen runner discards).

Frozen scope throughout: commit `dae37183bf4421682b2ad991e2dc0d0f3c53f260`,
source `cubature_genut_batch_tf.py` = `ae8cbfb...a976e`, Austria SIR
`N=1008`, FP32, RTX 5080 (GPU 0), `tftwogpu` TF `2.20.0-dev0+selfbuilt`,
deterministic ops, memory growth verified, CPU target construction, frozen
identity guard PASS in every process. TF32 on except the declared P3 arm.

## Headline

**The XLA `T=20` NaN is TF32-seeded arithmetic degradation that blows up
inside the higher-moment correction loop's unprotected solves; with TF32
disabled, the identical XLA pipeline at the identical scope is finite and
valid.** The fail-closed guard system behaved correctly throughout — the
NaN output was the guard masking a genuine raw-NaN event in Stage D, not a
guard bug and not an XLA miscompilation.

## Run Manifest

| Phase | Artifact (`xla_nan_attempt*/`) | Cases | Wall |
|---|---|---|---|
| P0 eager margins | `01/eager_margin_audit.json` | `(20,4) (1,0)` eager | 85 s |
| P1a XLA small ladder | `02/xla_small_ladder.json` | `(2,0) (3,0) (5,0) (2,4) (5,4)` xla | 810 s |
| P1b XLA escalation | `03/xla_t10_t20_ladder.json` | `(10,0) (10,4) (20,0)` xla | 2939 s |
| P2 failing case + diagnostics | `04/xla_t20_s4_diagnostics.json` | `(20,4)` xla | 3175 s |
| P3 TF32-off arm | `05/xla_t20_s4_tf32off.json` | `(20,4)` xla, TF32 off | 3159 s |

All COMPLETE; ~2.8 h total against the 4.5 h ceiling. One infrastructure
blocker (harness permission-classifier outage, ~14 rejected launches before
P0) resolved without consuming scientific budget.

## Findings

### F1. Reproduction boundary: the NaN needs BOTH full horizon and the correction loop, under TF32

| Case (xla, TF32 on) | Valid | Finite |
|---|---|---|
| `(2,0) (3,0) (5,0) (10,0) (20,0)` | true | true |
| `(2,4) (5,4) (10,4)` | true | true |
| `(20,4)` | **false** | **false** |

`T=20, steps=0` passing refutes the horizon recursion alone as the site;
`T<=10, steps=4` passing refutes the correction loop alone. The failure is
the interaction: correction iterations applied to late-horizon particle
states.

### F2. The failing stage is Stage D (higher-moment correction), by direct diagnostic evidence

In the failing case, every Stage A/B/C aggregate over all 20 steps is finite
and healthy — Sinkhorn column residual `2.6e-5` (guard `1e-4`), gap
eigenvalue `0.662` (guard `-1e-5`), mean/row residuals normal — while all
four Stage D aggregates are NaN (skew residual, kurtosis residual, pre-cap
RMS, post-cap RMS). The NaN therefore originates in or before the correction
displacement computation: candidate sites are the unridged Choleskys at
`cubature_genut_batch_tf.py:1273`, `:1288`, `:746` and the floor-ridged 2x2
normal-equation `tf.linalg.solve` (`:705-714`; the LM branch is inactive —
condition diagnostic 0.0, `lm_damping=0`).

### F3. TF32 is the seed (H-E confirmed); XLA miscompilation refuted (H-D)

TF32-off, same XLA pipeline, same scope: finite, valid, both endpoints.
Stage D diagnostics healthy (skew 1.26, kurtosis 16.6, RMS 35.2, Pearson
margin 1.044). The same register-spilled mega-fusions compiled and executed
correctly, so the compile pathology is a cost issue, not a correctness
issue. Under TF32, the Pearson feasibility margin in the failing run had
already collapsed to 0.165 before the blowup — consistent with progressive
TF32-arithmetic degradation of the corrected higher-moment state, then
raw NaN in an unprotected Stage D solve.

### F4. Thin-margin guard-flip hypothesis refuted (H-A)

Eager `T=20,4` margins are fat (gap eigenvalue 0.242 vs `1e-5` headroom
needed; column TV 36x under threshold), and in the failing XLA run every
threshold-guarded aggregate remained finite and far from its threshold. No
tolerance was or should be adjusted in response to this campaign.

### F5. Explanatory: the within-mode value/JVP identity issue is independent of the NaN

XLA exhibits scope-dependent `exact_equal=false` (seen at `(5,0)`, `(10,0)`,
and TF32-off `(20,4)`), the same compiler value/JVP program-split class as
non-XLA graph mode. It persists TF32-off and is therefore NOT fixed by the
TF32 finding. Also explanatory: XLA TF32-off `T=20` value `-680.6786` is
near the CPU value `-680.7359`, while eager GPU TF32-on gives `-683.0019`
— the ~2.3 log-unit offset between eager-GPU and CPU lanes is
TF32-dominated; its interpretation is not checked here.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Classify the XLA T=20 NaN as TF32-seeded Stage D arithmetic blowup, guard functioning correctly | Reproduction boundary mapped (F1); failing stage directly identified (F2); TF32-off finite at identical scope (F3) | No hard veto in any process; the NaN case is the object of study | Which exact Stage D solve emits the first NaN, and at which step (aggregates cannot say) | Owner decision on the repair/scope question below; optional per-step localization only if the repair choice needs it | No repair promoted, no tolerance changed, XLA not accepted or rejected, no NeuTra/HMC/tuning/posterior/default claim |
| Record the within-mode value/JVP split as TF32-independent and NaN-independent | Persists TF32-off (F5) | — | Same as the graph-mode program-split issue | Same owner scope decision as the graph-mode question | — |

## The Decision Now Owed To The Owner (not taken by this campaign)

The repository default is "GPU + TF32 + XLA" (LEDH-PFPF-OT TF32 route). This
campaign establishes that at the frozen Austria scope those three choices
are jointly incompatible with the current Stage D implementation: TF32+XLA
produces NaN at `T=20,steps=4`; TF32-off XLA works; TF32-on eager works.
Candidate resolutions, each a reviewed contract change with different costs:

1. Harden Stage D numerically (e.g. ridge the three unridged Choleskys, or
   compute Stage D moments/solves in float64 islands) — preserves the TF32
   default; changes the numerical program; needs fresh confirmation ladder.
2. Force full FP32 (TF32 off) for claim-bearing Austria XLA runs — smallest
   code change; contradicts the repo-level TF32 default directive and costs
   matmul throughput.
3. Exclude Stage D (or the whole route) from XLA claim-bearing scope.

Option 1 is the only one that addresses the mechanism rather than avoiding
it, and the same hardening plausibly helps every mode; but it edits
production source and is explicitly outside this campaign's authority.

## Post-Run Red Team

Strongest alternative reading of F3: TF32-off changes XLA fusion choices
(compile time and spill signatures differed slightly), so the pass could in
principle be layout luck rather than precision. Weakened by the eager
evidence: eager TF32-on also passes, so no single-mode arithmetic is
NaN-prone except the TF32+XLA combination — precision-as-seed is the only
explanation consistent with all four (mode, TF32) cells. Weakest evidence:
which specific Stage D solve fails first (not measured; aggregates only).
An observation that would overturn the classification: a TF32-off XLA NaN
at the same scope on a rerun, or a TF32-on XLA NaN at a scope where eager
TF32-on also fails.

## Nonclaims

No production source was edited. No tolerance, ridge, or default was
changed. XLA is neither accepted nor rejected for claim-bearing work. The
graph-mode program-split issue and its scope decision are unchanged. NeuTra,
HMC, tuning, cross-model, dual-cap, and posterior claims remain blocked.
