# Canonical LGSSM TF32 Balance And Horizon Continuation Plan

Date: 2026-07-18
Campaign ID: `canonical-lgssm-tf32-balance-horizon-continuation-20260718`
Status: `AUDITED_READY_FOR_EXECUTION`

## Research Intent Ledger

| Field | Frozen statement |
| --- | --- |
| Main question | Does a target-specific fixed terminal-balance count make the canonical float32/TF32 Contract E--Chol LGSSM route pass the declared probability-marginal gates at `T=2,N=1024`, and, if so, does the repaired one-solve XLA route remain valid and feasible at `T=10` and `T=50`? |
| Mechanism under test | Select the smallest count in `2,3,5,8` using only direct fused-route marginal errors on TF32 design seeds, audit it without retuning on disjoint seeds, then run a same-count float64/TF32 T=2 precision pair before the conditional horizon ladder. |
| Expected failure mode | Float32 accumulation may require more terminal balancing than float64; a count may pass design but fail audit; a longer horizon may expose marginal, numerical, memory, or runtime failure even after T=2 passes. |
| Promotion criterion | A positive count passes every active reset on both design and audit with `TV_col <= 1e-4` and `E_row <= 0.01`; same-count T=2 float64 and TF32 nodes are finite, hard-valid, replayable, use exact one-solve work counts, and have no sign reversal or order-one precision drift; each longer node then passes its frozen hard/resource gates. |
| Promotion veto | Marginal failure, nonfinite or invalid reset/chart, wrong chunk identity, non-XLA or wrong dtype/device, Python horizon unrolling, wrong work counts, diagnostic solver/sweep work, replay failure, score sign reversal, order-one float32 drift, OOM, corrupt artifact, or node-cap breach. |
| Continuation veto | No candidate passes the disjoint audit; corrected T=2 fails; a longer resource witness fails; a serious artifact is incomplete; or the bounded launch budget is exhausted. A candidate-count failure merely advances selection to the next declared candidate. |
| Repair trigger | A localized harness, serialization, XLA, resource, or reporting failure under the unchanged target triggers one focused repair and a fresh versioned attempt. |
| Explanatory diagnostics | Exact marginal trajectories, Kalman differences, compile/warm timing, graph size, allocator peak, and continuous float64/TF32 drift. These do not replace hard gates or prove HMC readiness. |
| Forbidden conclusions | No nonlinear-model validity, HMC readiness, posterior correctness, statistical superiority, universal balance count, or complete-leaderboard claim follows from this LGSSM continuation. |

## Evidence Contract

The selection target is the production dtype and workload: TensorFlow
float32 with TF32 enabled, XLA JIT, `T=2`, `N=1024`, and `K=N=1024` under
`dpf_transport_exact_divisor_cap3000_v1`. Selection may inspect only direct
fused-route `TV_col` and `E_row` plus hard implementation-validity checks. It
must not inspect Kalman values or scores. Design seeds are `81600..81607` and
audit seeds are `81620..81627`; they are disjoint from one another and from
the T=2 comparison seeds `81500..81515`.

After selection, the float64 and TF32 T=2 comparison must use the same selected
balance count, observations, parameters, particles, reset schedule, and seeds.
Comparing TF32 count `b` to the historical float64 count 2 artifact when
`b != 2` is forbidden because those are different finite programs. Continuous
Kalman and cross-precision differences remain descriptive; the frozen precision
veto is sign consistency and absence of order-one coordinate/value drift.

Every node writes a fresh JSON artifact under
`docs/benchmarks/artifacts/canonical_lgssm_tf32_balance_horizon_continuation_20260718/`.
No prior artifact may be overwritten.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode / early diagnostic |
| --- | --- | --- | --- |
| Counts `2,3,5,8` | Predeclared bounded continuation hypothesis | Starts at the failed transferred count and tests modest extra fixed work without reviving 50-step roundoff balancing | Grid may not contain a passing count; stop rather than extrapolate |
| `TV_col <= 1e-4`, `E_row <= 0.01` | Owner-approved frozen gates | Direct probability-scaled marginal errors | May not guarantee downstream score quality; same-count precision and Kalman diagnostics remain separate |
| `T=2,N=1024` selection | Target-specific reviewed choice | The failed TF32 gate occurred here; `N=128` float64 transfer was insufficient | Eight simultaneous seeds may miss tails; disjoint audit is required and no universal claim is allowed |
| Balance count fixed across T=2/10/50 | Algorithm identity requirement | Avoids horizon-wise retuning from claim results | T=2 count may fail longer horizons; that is a declared longer-horizon veto, not permission to retune |
| `K=N=1024` | Binding supported repository policy | One exact tile for `N<=3000` | Any smaller chunk repeats the retired performance defect; identity/work checks veto it |
| 20 Sinkhorn steps | Frozen finite-program baseline | Isolates terminal balance and horizon feasibility | Could be independently suboptimal; no tuning or universal claim is made |
| 8 GiB logical GPU cap | Owner-approved campaign boundary | Preserves machine stability and prior comparability | OOM is recorded as a resource veto; the limit is not raised |
| One and 16 seed longer nodes | Resource witness then frozen claim node | Prevents launching the larger batch after a resource failure | Sixteen seeds are descriptive, not population-level uncertainty evidence |

## Skeptical Pre-Execution Audit

Verdict: `PASS_AFTER_REVISION`.

1. The prior campaign driver hard-coded balance count 2. That is stale after a
   TF32-specific selection and could silently repeat the known failure. The
   driver must require the selected count as an explicit argument.
2. The prior precision gate pointed to a float64 count-2 artifact. Reusing it
   after selecting another TF32 count would compare different finite programs.
   A fresh same-count float64 T=2 artifact is mandatory.
3. Selecting at float64 `N=128` was an inadequate proxy for TF32 `N=1024`.
   This continuation selects at the actual dtype, particle count, chunk policy,
   horizon, and marginal target.
4. Kalman accuracy cannot select the numerical balance count. The selection
   driver imports no Kalman result and emits marginal/work/validity evidence
   only.
5. Longer-horizon success cannot repair a failed T=2 gate. The ladder remains
   conditional: selection, audit, same-count float64 T=2, TF32 T=2, inactive
   zero-OT witness, T=10 witness/claim, then T=50 witness/claim.
6. A T=10/T=50 run can succeed while answering only an engineering question.
   The result must keep engineering, numerical, and scientific interpretations
   separate and retain the forbidden conclusions above.

The revised commands and artifacts directly answer the question. No material
unexamined default remains.

## Execution Ladder And Budget

GPU launch budget: at most 12 process launches, at most 90 minutes total GPU
wall time. Per-node caps remain 20 minutes for T=2/T=10 and 45 minutes for
T=50. The TensorFlow logical memory limit remains 8192 MiB.

1. Run TF32 marginal-only selection over `2,3,5,8`; stop at the smallest design
   pass, then audit once without retuning.
2. Run a float64 T=2,N=1024,16-seed reference at the selected count.
3. Run the TF32 conditional supervisor at the same count: T=2 claim, T=2
   all-inactive witness, T=10 one-seed and 16-seed nodes, then T=50 one-seed and
   16-seed nodes.
4. Validate schemas, work counts, marginal gates, graph/device identity,
   precision gate, timing/memory, and absence of overwrite.
5. Write a terminal result and attempt ledger. Run focused CPU-hidden tests,
   Python compilation, and `git diff --check`.

## Exact Commands

All GPU commands use the `tf-gpu` environment with trusted/escalated device
access.

```bash
/home/chakwong/anaconda3/bin/conda run -n tf-gpu python \
  docs/benchmarks/run_canonical_lgssm_tf32_balance_selection.py \
  --output <root>/tf32_balance_selection.json \
  --attempt-id continuation-selection-01

/home/chakwong/anaconda3/bin/conda run -n tf-gpu python \
  docs/benchmarks/run_canonical_lgssm_fused_ot_loop_repair.py \
  --output <root>/t2_float64_reference_b<selected>.json \
  --attempt-id continuation-t2-float64-b<selected> \
  --time-steps 2 --num-particles 1024 \
  --seeds 81500,81501,81502,81503,81504,81505,81506,81507,81508,81509,81510,81511,81512,81513,81514,81515 \
  --arm all_active_contract_e --balance-steps <selected> --dtype float64 \
  --campaign-id canonical-lgssm-tf32-balance-horizon-continuation-20260718 \
  --plan-path docs/plans/bayesfilter-canonical-lgssm-tf32-balance-and-horizon-continuation-plan-2026-07-18.md

/home/chakwong/anaconda3/bin/conda run -n tf-gpu python \
  docs/benchmarks/run_canonical_lgssm_fused_ot_loop_campaign.py \
  --output-root <root>/campaign_nodes \
  --summary-output <root>/campaign_summary.json \
  --float64-t2-reference <root>/t2_float64_reference_b<selected>.json \
  --selection-artifact <root>/tf32_balance_selection.json \
  --balance-steps <selected> \
  --attempt-prefix continuation-01
```

## Stop And Handoff Conditions

Stop on a declared continuation veto, not on a failed selection candidate that
has a later declared candidate. Do not change `N`, `K`, dtype, thresholds,
seeds, horizon, reset schedule, Sinkhorn steps, memory limit, or balance count
after observing claim-node output. The terminal result must state exactly which
nodes ran, which did not, why the ladder stopped or passed, and what remains
unproved.
