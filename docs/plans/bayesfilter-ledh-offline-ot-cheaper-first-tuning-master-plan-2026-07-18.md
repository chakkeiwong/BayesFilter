# LEDH Offline OT Cheaper-First Tuning Master Plan

Date: 2026-07-18  
Campaign ID: `ledh-offline-ot-cheaper-first-tuning-20260718`  
Status: `SUPERSEDED_WRONG_CROSS_SCOPE_TRANSFER_DESIGN_2026-07-19`

> Supersession note, 2026-07-19: this plan is historical. Its design was wrong
> because it tuned at LGSSM `T=10` and treated `T=50` as a claim-bearing
> transfer test without first tuning the `T=50` scope. Under the active
> per-scope tuning policy, each model/route/horizon/data regime requires its own
> tuning artifact. The executed artifacts remain useful baseline evidence but
> cannot certify or reject a properly tuned T=50 run.

## Research Intent Ledger

| Field | Frozen statement |
| --- | --- |
| Main question | Can an offline, cheaper-first search select one fixed annealed-Sinkhorn/terminal-balance pair whose canonical Contract E--Chol value-and-score program satisfies direct probability-marginal gates at LGSSM `T=10,N=1024`, and does that pair remain valid at held-out `T=50`? |
| Mechanism | Hold `sinkhorn_steps=20`; search terminal `balance_steps=(2,3,5,8,12,16,25,32)` in order. Only when the full balance ladder has no calibration-and-validation pass, advance `sinkhorn_steps` through `(25,30,40)` and rerun the full balance ladder for each value. |
| Expected failure | A low balance count may leave `E_row > 0.01`; a complete balance ladder may expose an inadequate annealed initialization; a pair tuned at `T=10` may fail a fresh T=10 claim or the longer T=50 resource/claim gate. |
| Promotion criterion | The first lexicographic pair passes every active seed/time on both tuning partitions at `TV_col <= 1e-4` and `E_row <= 0.01`, then passes a fresh 16-seed T=10 claim and a fresh one-seed plus 16-seed T=50 holdout with finite/replayable value and score, valid charts/resets, exact one-solve work counts, GPU/XLA/TF32 identity, `K=N=1024`, and the 8 GiB cap. |
| Promotion veto | A fresh T=10 or T=50 claim marginal/chart/reset/finiteness/replay/work/identity failure, OOM, non-XLA graph, Python horizon unrolling, wrong chunks/dtype/device, artifact corruption, or cap breach. |
| Continuation veto | No pair passes within the declared finite grid; a selected pair fails the fresh T=10 claim; the T=50 resource witness fails; the serious artifact is incomplete; or the campaign budget is exhausted. A tuning-candidate failure is not a continuation veto. |
| Repair trigger | A localized harness, serialization, exception handling, or identity-binding failure under unchanged scientific settings triggers one focused repair and a fresh versioned attempt. |
| Explanatory only | Kalman value/score differences, continuous runtimes, allocator peaks, and residual trajectories below the hard gates. They cannot select a pair or prove score correctness. |
| Forbidden conclusions | No universal/optimal OT schedule, HMC readiness, posterior correctness, statistical superiority, all-model correctness, or nonlinear validity follows from the LGSSM ladder. |

The tuning procedure is offline. Runtime adaptation, tolerance-driven stopping,
or parameter-dependent iteration counts inside HMC are forbidden. The selected
integer pair is frozen before either claim run.

## Evidence Contract

The exact comparator is the previously failed production-shaped LGSSM pair
`(sinkhorn_steps,balance_steps)=(20,2)` at `T=10,N=1024`. Selection uses only
the two declared direct probability errors and engineering hard checks from the
same fused value-and-total-score graph. It must not use Kalman agreement.

- Calibration seeds: `81520..81527`.
- Tuning-validation seeds: `81528..81535`.
- Fresh T=10 claim seeds: `81700..81715`.
- Fresh T=50 witness/claim seeds: `81720` and `81720..81735`.
- Gates: `TV_col <= 1e-4`; `E_row <= 0.01`.
- Target: TensorFlow float32, TF32 enabled, XLA JIT, one `StatelessWhile`
  horizon, `N=K=1024`, and an 8192 MiB logical GPU limit.

The tuning-validation partition participates in selection: a validation
failure advances to the next declared pair. It is therefore not called a final
audit. The fresh T=10 and T=50 seed sets are the untouched claim evidence.
Every attempt uses a fresh versioned output directory.

For efficiency the two tuning partitions are concatenated into one fixed
16-seed TensorFlow batch for each candidate, then summarized separately by
their frozen index slices. This does not mix the partitions statistically and
avoids compiling and traversing the same candidate twice. Failed tuning
candidates do not run a second replay traversal; replay is mandatory on the
selected pair's fresh T=10 and T=50 claim nodes.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| Sinkhorn start `20` | Existing finite-program baseline | Isolates the cheaper terminal correction first | Initialization may be inadequate; only a complete failed balance ladder advances Sinkhorn |
| Balance ladder `2,3,5,8,12,16,25,32` | Bounded search hypothesis, extending the inadequate old `2,3,5,8` grid | Dense near the observed low-count failure and capped well below obsolete 50/100 roundoff schedules | Cap may exclude a valid schedule; report `NO_PAIR_WITHIN_GRID`, never relax gates |
| Sinkhorn ladder `20,25,30,40` | Bounded second-stage hypothesis | Tests initialization only after terminal balancing is exhausted | It is not a proof of global optimality; selection is lexicographic, not a runtime ranking |
| First passing pair | Reviewed selection rule | Implements cheaper-control-first tuning and minimizes balance within the first successful Sinkhorn rung | Does not guarantee globally minimal softmin count; no optimality claim |
| Two 8-seed tuning partitions | Existing failed-seed block split without overlap | Directly includes the known T=10 failure while checking a disjoint partition | Limited tail coverage; fresh 16-seed claims remain mandatory |
| T=50 inherits T=10 pair | **Wrong historical design** | No scientific justification: the horizon changes the encountered filtering geometries | This choice caused an untuned T=50 baseline to be mislabeled as a claim run; it is superseded by mandatory T=50 tuning |
| `K=N` for `N<=3000` | Binding artifact-supported repository policy | Avoids the repeatedly regressed tiny-tile route | Any `K<1024` is an identity veto |
| Fixed hard thresholds | Owner-approved probability-scale gates | They measure total column probability error and maximum row error directly | Passing them does not establish value/score oracle agreement |

## Skeptical Plan Audit

Verdict: `PASS_AFTER_REVISION`.

1. The superseded plan stopped at `balance_steps=8` and could not test the
   requested second-stage Sinkhorn repair. This plan declares both bounded
   ladders and exhausts the cheaper one first.
2. Treating a validation failure as a final audit failure would prevent the
   requested adaptive tuning. The second tuning partition is now honestly
   labeled validation, while new T=10/T=50 seeds remain untouched claims.
3. The prior node executor bound Sinkhorn to a module constant and reported
   only aggregate residuals. The repair must bind both integers in preparation,
   graph construction, node output, selection artifact, and claim identity,
   and preserve per-seed/time diagnostics.
4. Reusing Kalman error for tuning would select for the diagnostic model rather
   than the general OT invariant. Kalman values/scores are disabled during
   tuning and remain explanatory in claim nodes only.
5. A Python search loop is allowed in the offline supervisor. The differentiable
   horizon and both numerical solvers remain TensorFlow `while_loop` bodies
   compiled by XLA; no Python loop enters the HMC-facing finite program.
6. Compiling calibration and validation as separate candidate calls and
   replaying every failed candidate would double avoidable OT work. The two
   partitions therefore share one 16-seed candidate execution and retain
   separate summaries; only selected-pair claim nodes pay the replay cost.
7. “All nonlinear models” was too broad: scalar SV, generalized SV, KSC-SV,
   and predator-prey currently use Contract E--TP controls, not this annealed
   OT contract. Applying Sinkhorn labels to their feature/order settings would
   answer the wrong question. The nonlinear phase must first emit a route
   compatibility inventory. Only a route exposing the same solver, direct
   marginal metrics, fixed-count binding, TF32/XLA path, and claim schema may
   run this tuner.

The commands and artifacts answer the frozen question without proxy promotion,
hidden horizon retuning, or an unrestricted grid.

## Phases

### Phase 0: Harness And Plan Repair

Objective: parameterize both iteration counts, emit complete marginal
trajectories, add structured exceptions, and implement the offline supervisor.

Entry: pushed fused one-solve implementation and preserved failed T=10
artifact. Required artifacts: this plan, Python tuner, focused tests. Checks:
CPU-hidden import/compile/tests, source inspection for `StatelessWhile`, and
`git diff --check`. Forbidden: changing tolerances, chunks, reset semantics,
particle count, or score target. Handoff: only when the tuner can prove that
preparation, callable, result, and selection all bind the same pair. Stop:
unresolved identity or schema defect.

### Phase 1: T=10 Offline Tuning

Objective: select the first pair passing both tuning partitions. Required
artifact: candidate-by-candidate JSON plus a frozen selection record and
manifest. Checks: every active seed/time residual, finiteness, replay, charts,
resets, work counts, graph/device identity, runtime and memory. Forbidden:
Kalman-based selection, dynamic runtime stopping, threshold relaxation, or
skipping the remaining balance candidates when a rung fails. Handoff: selected
pair and its exact artifact hash. Stop: no pair within the grid or budget cap.

### Phase 2: Fresh T=10 And Held-Out T=50

Objective: test the frozen pair. Run fresh T=10 16-seed claim, then T=50
one-seed resource witness and 16-seed claim. Required artifacts: one JSON per
node, supervisor summary, result note, and run manifest. Checks: all promotion
and veto fields in the ledger. Forbidden: retuning after claim output or
describing Kalman agreement as the marginal selection criterion. Handoff:
`LGSSM_T50_PASS` only if every node passes. Stop: first declared claim/resource
veto.

### Phase 3: Nonlinear Compatibility And Conditional Continuation

Objective: inventory actual nonlinear LEDH routes and run the same tuner only
where its mathematical/numerical contract is present. Required artifact:
route table with model, callable, reset family, solver controls, marginal
metrics, dtype, XLA status, and disposition. Latent SIR is the first candidate
because it exposes `steps` and `balance_steps`; it is not executable under this
plan until its current float64/roundoff-marginal/separate-traversal gaps are
repaired and reviewed. Contract E--TP models are `INCOMPATIBLE_SEPARATE_TUNER`,
not failures. Forbidden: calling TP order/features Sinkhorn settings or claiming
all-model coverage from LGSSM/SIR. Handoff: a bounded nonlinear adapter plan or
an executed compatible-route result. Stop: no route meets the frozen interface.

## Compute Budget And Stop Conditions

At most 32 T=10 tuning candidate evaluations, three claim/resource nodes, two
localized harness retries, 90 minutes total GPU wall time, and 8192 MiB logical
GPU memory. Candidate evaluations stop at the first calibration-and-validation
pass; no full grid is run after selection. A resource/harness retry uses a new
output directory and does not change scientific settings.

## Exact Launch

```bash
/home/chakwong/anaconda3/bin/conda run -n tf-gpu python \
  docs/benchmarks/run_ledh_offline_ot_tuning_campaign.py \
  --output-root docs/benchmarks/artifacts/ledh_offline_ot_cheaper_first_tuning_20260718/attempt01 \
  --attempt-id attempt01
```

GPU execution requires trusted/escalated device access. Success means the
supervisor writes `LGSSM_T50_PASS`; it does not itself authorize applying the
same integers to an incompatible nonlinear route.

## Execution Close Record

- Plan audit passed after the harness identity, shared tuning-batch, and
  nonlinear-compatibility revisions.
- Both iteration counts were bound through preparation and the XLA callable;
  complete per-seed/time residual histories and structured failures were
  emitted.
- Local checks passed: tuner tests `5/5` before launch and `7/7` continuation
  tests plus `16/16` chunk-policy tests after launch; Python compilation and
  `git diff --check` also passed.
- The GPU campaign selected `(20,3)` at T=10. The fresh T=10 claim and T=50
  resource witness passed. The fresh T=50 claim failed the row gate at four
  seed-time states.
- Corrected handoff: the T=50 run is an untuned-baseline failure. Tune T=50 on
  fresh calibration/validation partitions, then use a new untouched T=50 claim
  set. Independently, every nonlinear model/route must receive its own tuning
  phase; a failure in one scope does not waive or block tuning in another scope
  unless an implementation-validity or total-budget veto fires.
