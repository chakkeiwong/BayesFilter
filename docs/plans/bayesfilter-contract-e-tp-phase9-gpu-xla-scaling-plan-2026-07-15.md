# Contract E--TP Phase 9 GPU/XLA Scaling Plan

metadata_date: 2026-07-15
status: READY_AFTER_SKEPTICAL_CAPABILITY_AUDIT
program_id: contract-e-tp-all-model-gradient-comparison
phase: 9
execution_target: trusted NVIDIA GPU with TensorFlow XLA
budget: 8 trusted GPU-hours; at most three attempts per rung

## Phase Objective

Compile and execute the correctness-eligible Contract E--TP recursive LGSSM
`finite_lookahead=8` program on the real GPU under XLA, first at `T=10` and then
at `T=50`. Record value, total score, chart validity, CPU-artifact drift,
compile/warm runtime, and allocator peak memory. Audit rather than hide the
remaining gap to the repository's float32/TF32 production target.

## Entry Conditions

- Phase 8 is complete; inaccurate adjacent-state extensions do not enter GPU
  scaling.
- LGSSM Contract E--TP `T=10,50` float64 CPU artifacts pass value, score,
  chart, and same-scalar FD gates against Kalman.
- Phase 2 streaming forward/JVP/VJP GPU/XLA smoke passed, including compiled
  fail-closed poisoning.
- Nonlinear recursive Contract E--TP modules do not yet expose XLA-default
  factories and are hard-coded float64. They remain explicit capability gaps.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Main question | Does the actual successful LGSSM finite-lookahead recursion compile and execute correctly with GPU/XLA at `T=10,50`? |
| Baseline | Controlling CPU float64 Phase 7 LGSSM artifacts on identical preparations/data/theta |
| Primary engineering gate | visible GPU, XLA compiled, finite value/score, valid charts, no CPU fallback, exact preparation hash |
| Numerical diagnostic | GPU float64 versus CPU float64 value/score differences, reported without an invented threshold |
| Hard veto | wrong feature core, wrong preparation hash, nonfinite, invalid chart, XLA failure, CPU placement/fallback, or value/score disagreement indicating different execution |
| Explanatory diagnostics | compile runtime, warmed runtime, peak allocator bytes, graph operation count |
| What is not concluded | float32/TF32 readiness, nonlinear GPU readiness, HMC readiness, canonical/default status, statistical superiority |
| Artifact root | `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase9_gpu_xla_20260715/` |

## Capability Ladder

1. Run trusted `nvidia-smi` and a trusted TensorFlow GPU device probe.
2. Add an XLA-default factory for
   `contract_e_tp_lgssm_score_informed_recursive_core`; it must bind
   `feature_mode` and `lookahead_steps`, not call the older base-feature core.
3. Run `T=10` float64 on GPU/XLA, once for compile and once warmed.
4. If `T=10` passes, run `T=50` float64 on GPU/XLA.
5. Record float32/TF32 as `blocked_dtype_generic_refactor_required` unless the
   owned recursive core, preparation tensors, numerical vetoes, and reference
   checks are made dtype-generic and independently tested. Do not cast only the
   outer inputs while inner constants remain float64.
6. Record scalar-SV and predator recursive GPU/XLA as
   `blocked_xla_factory_not_implemented`; their CPU scientific evidence remains
   valid but is not production-target scaling evidence.

## Skeptical Plan Audit

Status: `PASS_AFTER_CORE_AND_DTYPE_SCOPE_REPAIR`.

The inherited Phase 9 wording implied all nonlinear full horizons could run on
float32/TF32 immediately. That is false for the current code. The only existing
recursive factory calls the older base-feature core, not the successful
finite-lookahead core, and all model adapters are currently float64. The
repaired plan first establishes actual LGSSM GPU/XLA execution in the checked
scientific dtype and exposes, rather than conceals, the float32/TF32 and
nonlinear factory gaps. A float64 pass cannot close those gaps.

The plan preserves the research question: scaling evidence is engineering
evidence, not a substitute for the Phase 7/8 scientific comparisons.

## Exact Commands

GPU/CUDA commands require trusted/escalated execution under repository policy:

```bash
nvidia-smi
/home/chakwong/anaconda3/envs/tf-gpu/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
/home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/run_contract_e_tp_phase9_lgssm_gpu_xla.py --preparation docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8b_lgssm_t10_order5_lookahead8_attempt1_20260715/charts.json --cpu-result docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8b_lgssm_t10_order5_lookahead8_attempt1_20260715/result.json --output docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase9_gpu_xla_20260715/lgssm_t10_float64_result.json
/home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/run_contract_e_tp_phase9_lgssm_gpu_xla.py --preparation docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8b_lgssm_t50_order5_lookahead8_attempt1_20260715/charts.json --cpu-result docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8b_lgssm_t50_order5_lookahead8_attempt1_20260715/result_aggregate.json --output docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase9_gpu_xla_20260715/lgssm_t50_float64_result.json
```

## Handoff And Stop Conditions

`T=50` begins only after `T=10` uses the correct feature core and passes device,
XLA, finite, chart, and CPU-drift checks. A localized XLA harness failure is a
repair trigger within budget. Stop the rung on OOM, CPU fallback, nonfinite,
invalid chart, or material CPU/GPU disagreement requiring root-cause analysis.

Phase 10 terminal synthesis begins after `T=50` passes or after a bounded
capability blocker is recorded. The report must keep float64 GPU/XLA success
separate from unresolved float32/TF32 and nonlinear-route gaps.
