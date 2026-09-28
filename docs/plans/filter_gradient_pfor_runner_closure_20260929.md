# Runner and benchmark pfor disposition

Continue the execution repair after library qualification 04711–04719. The
engineering question is whether all 28 remaining discovered sites in 17 owned
runners/benchmarks can have an explicit non-pfor implementation or an enforced
retirement. The exact baseline sites, functions, AST and file hashes are in
`filter_gradient_remaining_pfor_discovery_20260929.json`; preserve that catalog.
No old pfor implementation will execute as a comparator. No historical LEDH
fixture, result, wrong-chunk benchmark, training, or HMC run supplies new evidence.

## Reviewed implementation scope

| Paths (under docs/benchmarks unless qualified) | Disposition and call-chain scope |
|---|---|
| experiments/dpf_implementation/tf_tfp/runners/run_ledh_pfpf_source_faithful_repair_tf.py | Retire the June P44 auxiliary-flow repair before framework import under the August21 invalidation. Its only Python reference is a historical classification reader, not a runtime import. Preserve source and remove its implicit pfor selection; do not revive the algorithm. |
| scripts/p91_gpu_xla_jit_check.py; scripts/p91_performance_benchmark.py | Explicit non-pfor Jacobian inside the existing enclosing fixed-signature XLA functions. These differentiate complete-data diagnostic targets, not analytical filtering scores. Preserve the scalar and batched density authority. |
| scripts/p91_hmc_smoke.py | Replace the scalar-map pfor with a sequential TensorFlow map and scalar output signature. Test only the target function on fresh inputs, with no sampler launch, tuning or admission. This existing scalar map is not batch-native training. |
| benchmark_identifiable_ssl_lstm_oracle_geometry_2026_07_08.py; benchmark_minimal_ssl_lstm_zhaocui_hmc_tuning_phase5_2026_07_06.py | Disable pfor in the diagnostic Hessian/score Jacobian, make the owning tape persistent where eager execution needs it, and release it. Test the actual helper with an independent analytic quadratic adapter; do not launch tuning. |
| benchmark_minimal_ssl_lstm_zhaocui_hmc_oracle_2026_07_06.py | Replace scalar-map pfor with TensorFlow map_fn. Preserve the explicitly CPU reference role and host diagnostic output; no training eligibility. |
| benchmark_p8p_parameterized_sir_gradient.py; diagnose_contract_e_phase8_common_path_identity.py | Preserve existing wrong-chunk CLI retirement. Remove implicit pfor in preserved helpers; do not run these old LEDH fixtures. Source/control checks establish the disposition, not numerical LEDH qualification. |
| diagnose_p8p_sir_rk4_sensitivity_vjp.py; run_genut_austria_endpoint_root_cause_20260817.py | Explicit non-pfor engine and correct tape lifetime. Only fresh generic derivative mechanics may be tested; no old P8p/Austria artifacts, route execution or canonical claims. |
| contract_e_score_aware_teacher_projection_2d_lgssm.py; run_contract_e_tp_scalar_sv_prefix.py | Non-pfor diagnostic derivative wrappers. Test generic callable wrappers using new analytic functions, not historical teacher/prefix data. Candidate selection and canonical reset/scientific gates are outside this unit. |
| run_cubature_exact_sv_score_ladder.py; run_exact_sv_fixed_gaussian_genut_paired.py; run_zhao_cui_moment_teacher_actual_sv.py | Disable implicit pfor in the independent dense reference arm. Preserve current DGP/history gates. Qualify derivative mechanics on fresh exact-SV primitive inputs if the existing call contract permits a small bounded case; do not run their campaign entry points. |
| run_kalman_qr_cpu_xla_formulation_shootout_2026_07_15.py | Remove unapproved vectorized_strict/vectorized_fallback candidates and reject explicit requests before building kernels or worker commands. Preserve legitimate native/map comparisons and numerical thresholds. Tests must also reject historical pfor records from nomination. |

The primary agent inspected each call site and its immediate tape/map owner,
plus textual Python consumers. P91 source-route behavior and mathematical
formulas are unchanged; there is no new Zhao-Cui source-faithfulness claim.
The HMC interface reference has been consulted; no tuner or sampler is changed.
Whole-consumer admission is not inferred from an extracted helper test.

## Evidence contract and sequence

1. Freeze the completed library sources and evidence, archive and commit them.
2. Make only the engine/map/retirement changes above. For multiple Jacobians,
   retain the persistent tape until the final use; release it afterward.
3. Add a current disposition catalog and discovery guard. All roots in the
   original catalog must be scanned again; unexpected sites fail. The sole
   custom log-coordinate Jacobian in ValidationTarget remains a verified false
   positive. Existing NumPy/host-loop exceptions must not grow. Runner pfor
   coverage is explicit; unreviewed numerical/selection code is not relabeled
   diagnostic to evade wider policy repair.
4. Run focused CPU reference checks of actual generic derivative wrappers,
   target-map outputs and derivatives, rejected formulations and retirement.
   New primitive inputs and exact analytic derivatives are the controls.
   Use healthy float64 comparisons at 1e-9, five-point derivatives at 2e-6;
   preserve existing tests' tighter or domain-specific thresholds. Do not
   loosen tolerances after a failure. For P91 use the shared current density
   and independent finite differences on fresh complete-data inputs.
5. Qualify eligible fixed-signature enclosing XLA target functions on CPU and
   trusted GPU. Record changed inputs, one trace, HLO, no pfor/host callbacks,
   placements and verified memory growth. CPU-only diagnostic helpers remain
   explicitly reference-only. Do not import CPU-hiding historical harnesses
   into a GPU worker; bounded AST extraction must be labeled a helper check.
6. Read back source hashes, artifacts and existing policy tests. Reuse unchanged
   04713–04719 library evidence in its bound scope. Preserve failures and record
   what was executed versus inspected only. Commit/push the repair checkpoint.

Pass criteria: every discovered site has a mechanically enforced disposition;
all affected eligible helper/default/rejection tests pass without numerical
gate changes. Missing gradients, lost axes, disconnected batch rows, pfor or
host callbacks in tested graphs, changed equations, stale source records or
early framework import before retirement veto that unit and trigger repair.
Unexpected scientific-target or algorithm changes stop dependent work for plan
revision. Compilation/runtime/RSS observations are explanatory here; no speed
ranking against unapproved pfor follows from them. Full consumer costs, memory
capacity, terminal F01–F20 admission and main merge remain separate gates.

At most 12 sequential workers, 2400 CPU / 1800 GPU process-seconds, including
two localized harness retries, inside the unchanged 56 CPU / 52 GPU hour caps.
Initial worker timeout 300 seconds. Use the existing approved command prefix:

```
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python scripts/run_filter_repair_campaign.py test --group pfor_runner_<group> --device CPU --test-timeout-seconds 300
```

GPU groups use `--device GPU --test-gpu-index <available non-display index>`
with trusted access after checking availability. One numerical worker at a
time; no other jobs stopped. Preserve per-run source/environment/command/seed
and elapsed-time manifests under the versioned campaign artifact root. No
package, environment, cache/system-limit, canonical NeuTra, LEDH RNG, training,
HMC, live MacroFinance or main-merge changes are authorized by this unit.

Skeptical pre-execution review: diagnostic names and old dates are not pfor
approval. Changing only Jacobian keywords can fail in eager mode unless tape
lifetime is repaired; test both owning contexts. Removing CLI options alone
can leave direct builders and historical-record nomination open; test those
boundaries too. A generic helper check cannot establish a historical LEDH
algorithm or HMC consumer, and no such qualification is sought. The comparison
uses independent formulas/current authorities, never an unapproved pfor run.
The bounded scope, controls, exclusions and stop conditions answer this specific
execution question. Primary-agent review passes; no independent review asserted.

04720 passes23 helper/rejection checks.04721 passes the mapped target but both
P91 batched Jacobian owners fail at tf2xla conversion: `Cannot find body function
while_1_body_*_const_0 for While node while_1`. The shared complete-data density
contains a four-step RK4 TensorFlow loop. Inspection of TensorFlow backprop.py
1150–1185 shows jacobian(False) re-enters its tape to reshape/gather inside
parallel_for/control_flow_ops.py's TensorArray loop. This identifies the failing
generated derivative/loop path; it does not prove a general TensorFlow defect.
No numerical mismatch or old pfor comparator follows from this compile failure.

Localized repair: replace those two Jacobian wrappers with one repository-owned
diagnostic helper that evaluates the same batch-native density once, then uses
tf.while_loop and one-hot output cotangents with GradientTape.gradient to obtain
each value row's derivative. No scalar density fallback, analytical filtering
score, source equation or new source-faithfulness claim is introduced. Test the
same fixed inputs, two step sizes, scalar reference gradients and enclosing
XLA/HLO criteria. The shared helper is fully guarded; no new allowance. Record
the failed04721 and retry under the unchanged12-worker/compute caps. If the same
compiler failure persists, inspect that graph before another implementation.

04722 repeats exactly the missing-body compile failure for both captured-tape
owners; the scalar-map target still passes. Before another runtime change,
export the failing traced GraphDef and verify loop-body/condition definitions.
This diagnostic differentiates an already missing graph function from a later
compiler rewrite failure. Preserve the complete GraphDef. Then, if definitions
are intact, test the smallest ownership change: create the batch density and
its tape inside the score-direction loop body. This repeats the same batched
forward calculation for each direction but avoids capturing a tape whose
saved RK4 loop lives outside the score loop. It changes diagnostic execution
cost, not the finite scalar or its derivative, and cannot support a speedup
claim. Two-step finite differences, unchanged scalar checks and full enclosing
XLA remain required; no row-wise scalar-density fallback is introduced.

04723 exports the failing graph: all three While body/condition pairs are
defined before XLA, with ten function definitions. The missing `_const_0`
variant therefore appears during compilation; the saved graph itself is not
missing a referenced body. Proceed with the loop-local tape ownership change
above. This evidence narrows the compiler stage, not the exact compiler pass.

04724 CPU and04725 trusted GPU pass all three P91 cases after loop-local tape
ownership.04726 passes162 readback/policy checks and fails one registration
test because the new GPU group was absent from TEST_DEVICES. The GPU command
explicitly used --device GPU, and saved tensor placement, HLO and verified
growth establish actual GPU execution; the omission affected automatic group
dispatch, not those numerical results. Add the group to TEST_DEVICES and renew
readback/policy only. This is localized harness repair1/2 with unchanged scope,
controls, tolerances and budget; preserve04726.
