# Actual-SV Overcomplete Analytical Chart Phase 1 Result

Date: 2026-07-17

Status: `PASS_PHASE_1_SPECIFICATION`

Plan:
`docs/plans/bayesfilter-actual-sv-overcomplete-analytic-chart-repair-plan-2026-07-17.md`

## Result

The candidate design is frozen before candidate execution in:

`docs/benchmarks/artifacts/actual_sv_overcomplete_analytic_chart_repair_20260717/phase-01-specification/design_specification.json`

It fixes the Actual-SV row and target hashes, center and parameter geometry,
four-feature finite target, deterministic TensorFlow anchor/reference
construction, global capacity ladder `K=5,...,25`, ordered design and held-out
sets, numerical gates, derivative contract, pilot costing rule, budgets, and
fresh-artifact policy.  Held-out points cannot select another capacity.

No candidate capacity, design outcome, or held-out outcome was evaluated while
freezing this specification.

## Frozen Interfaces And Commands

The implementation interfaces are:

- preparation: `docs/benchmarks/prepare_actual_sv_overcomplete_analytic_chart.py`;
- evaluation: `docs/benchmarks/run_actual_sv_overcomplete_analytic_chart.py`;
- preparation schema:
  `bayesfilter.contract_e_tp.scalar_sv_overcomplete_preparation.v3`;
- result schema:
  `bayesfilter.contract_e_tp.actual_sv_overcomplete_result.v1`.

The preparation command shape is:

```text
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 timeout <T-limit> python docs/benchmarks/prepare_actual_sv_overcomplete_analytic_chart.py --time-steps <T> --anchor-count <K> --output <fresh-preparation.json>
```

The CPU evaluation command shape is:

```text
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 timeout <T-limit> python docs/benchmarks/run_actual_sv_overcomplete_analytic_chart.py --device cpu --preparation <preparation.json> --point-set <center|design|held-out|fd> --evaluation-mode <chart|score> --output <fresh-result.json>
```

The trusted GPU command shape is:

```text
timeout 1800 python docs/benchmarks/run_actual_sv_overcomplete_analytic_chart.py --device gpu --gpu-memory-limit-mib 8192 --preparation <preparation.json> --point-set fd --evaluation-mode score --cpu-result <cpu-result.json> --output <fresh-result.json>
```

The GPU process must configure one TensorFlow logical GPU with an 8192 MiB
limit before logical-device initialization and must not enable memory growth.
Phase 3 uses `--evaluation-mode chart`; Phases 5 and 7 use
`--evaluation-mode score`.  This separation keeps score compilation out of the
capacity screen without changing its finite-program chart gates.

## Consistency Review

The JSON is sufficient to derive every candidate-defining numerical input
without a hidden scientific default.  The command shapes preserve deliberate
CPU hiding, fixed thread limits, fixed model/feature settings, phase timeouts,
and the hard GPU cap.  Candidate preparation is TensorFlow-only; NumPy/SciPy
may appear only in independent reporting or high-precision audits.  The
existing fixed-square v1 preparation and runner are historical references and
will not be silently reinterpreted as v3.

Review verdict: `CONSISTENT_FOR_PHASE_2_IMPLEMENTATION`.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Pass Phase 1 | Reproducible specification and interfaces are frozen before candidate evaluation | No Phase 1 handoff veto fired | Feasibility, positivity, derivatives, and cost remain untested | Implement the specialized primitive and recursive route | No chart, score, GPU, HMC, canonical, or leaderboard claim |

## Handoff

Phase 2 may implement the frozen design.  Phase 3, not Phase 1, owns the pilot
and capacity observations.
