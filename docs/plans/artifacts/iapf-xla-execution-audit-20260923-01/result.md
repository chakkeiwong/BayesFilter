# iAPF execution and XLA investigation, 23 September 2026

Phase21 stopped because its ordinary-FP32 worker exceeded the supervisor's 2,100-second limit. The inspected log shows successful CUDA/XLA initialization and compilation, not an XLA compilation exception. A separate earlier TF32 wrong-result case is real and has an existing source repair. The old agent also encountered a session compaction failure, distinct from the worker timeout.

Checkout: /home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922.
Branch: ledh-refactor-with-policy-fix. Commit: 89065bc6354801cb368d5e163ba49fd9c4372d10.
The supervisor, shared worker, fitted-twist kernel and three numerical helper modules match Phase21's frozen source. This audit changed no numerical source, environment or default.

## Traced call chain

The executable path is diagnose_iapf_tf32_precision_confirmation.py → diagnose_iapf_tf32_revalidation.py::worker → cached make_fitted_twist_kernel → twisted_transition and the TensorFlow Gaussian/conditional-mean/score helpers. The timing diagnostic executes this actual frozen worker with host timers around its stages.

| Question | Checked finding |
|---|---|
| Python loops? | Yes: diagnose_iapf_tf32_revalidation.py:143 loops over independent replicates. Each iteration generates inputs, calls the ordinary filter, checks/materializes outputs, calls the traced filter, verifies neutrality, transfers traces, encodes JSON/gzip and writes CSV. |
| Python recursion inside the filter? | No runtime Python time loop: fitted_twist_tf.py:96 has a fixed input_signature and jit_compile; line 171 uses tf.while_loop. Particles are tensor dimensions. Static branches and feature-building comprehensions execute during graph construction. |
| Repeated retracing? | Not observed in checked runs. All ten saved FP64 compilation records have trace_count=1. The new FP64 and FP32 diagnostics retain one trace for each of eight function instances after 192 streams/mode. Eight cached variants remain below the factory limit of 12. |
| Why the warning? | Installed TensorFlow polymorphic_function.py::_get_key_for_call_stats groups warning statistics by Python code object. Multiple function instances created at the same nested factory location can trigger the warning while each instance traces only once. The warning does not establish per-replicate retracing. |
| NumPy? | No NumPy arithmetic/import was found in the inspected worker-to-kernel chain or score_study/runtime imports. Tensor.numpy() synchronizes/materializes data at diagnostic/control/serialization boundaries; it does not imply that NumPy implements the kernel. |
| Callbacks or pfor? | No tf.py_function, tf.numpy_function, tf.vectorized_map, tf.map_fn or pfor was found in the inspected numerical path. |
| Adaptive consumer? | iapf_adapter.py:112 separately uses a host loop over adaptive fitting iterations and host stopping decisions. It calls compiled filter/fit kernels. Optimizer/backward recursion in iapf_fit_tf.py uses TensorFlow loops. This is distinct from Phase21's fixed-guide replication driver. |

These findings cover the inspected call chain, not every BayesFilter execution lane.

## Earlier TF32 wrong-result case

The saved center probe localized a time-dependent guide-center error in the GPU/XLA/TF32 transition: an expression inside the TensorFlow time loop could behave as though it reused the first guide center. The existing repair at fitted_twist_tf.py:37 separates K(center-mean) into K center - K mean, including tangent terms. These are equal in real arithmetic but present different operations to the compiler and can round differently.

Phase18's center diagnostic and reference checks support this localized repair. Phase21 already used the repaired source. This is separate from small finite-precision CDF changes that alter resampling ancestors and amplify concentrated weights, as recorded in Phase19. The inspected current HLO has cuBLAS GEMM/triangular solves and cuSOLVER Cholesky; it does not support blaming the current timeout on Triton. Duplicate CUDA-factory and oneDNN startup messages also do not identify the timeout's cause.

## Executable timing evidence

The debugging-only diagnostic used the frozen numerical source, original reference warmups, assigned RTX4080 SUPER, XLA, verified memory growth, and 16 streams for each of 12 d2 cells. FP64 took 25.514 s; FP32 took 23.914 s. Each passed 18 reference checks, wrote 192 records and ended with one trace per function instance. Combined process time 49.429 s was below the separate 180 s diagnostic cap. See [manifest](audit-manifest.json), [timing summary](timing-summary.json) and [backend comparison](backend-comparison.json).

| Stage, mean milliseconds per later stream | FP64 | Ordinary FP32 |
|---|---:|---:|
| CPU random inputs |0.889|0.815|
| Ordinary filter and synchronization |12.007|10.104|
| Finite/normalized checks and initial HLO capture |1.779|1.738|
| Traced filter and neutrality checks |11.622|9.625|
| Trace transfer/materialization |1.517|1.160|
| JSON/gzip |2.362|2.177|
| CSV row |0.037|0.034|

Two filter stages consume 77–78% of measured stream time; JSON/gzip about 8%. These timings include synchronization and host checks and are not a GPU-kernel profiler. Full trace recording deliberately doubles filter execution to check bitwise neutrality on every stream. Removing that comparison would change the evidence contract.

The failed attempt's unusually slow FP32 throughput did not recur. Its cause is unresolved. Device contention, clocks, compilation choices or long-run host/IO effects need contemporaneous measurements; none is established by the current logs. A 192-stream diagnostic cannot certify the runtime of a 49,152-stream worker or rank precision modes by speed.

## Master program and next step

The execution checkout's [master](../../younis-kdm-score-master-program-2026-09-14.md) records Phase21 REPAIR_REQUIRED. The main checkout's older master status is stale for this branch. Phase21 is a conditional precision confirmation: d2/N64/T5, three datasets, four guides, 4096 streams/cell/mode, identical common FP32 inputs and complete contemporaneous traces. Its 2% simultaneous mean-difference criterion, 20% tail-contribution veto, FP64/FP32 controls and exact Kalman reference are unchanged. Phase20 passed 18/24 comparisons with six tail vetoes. Broader Phase 18/19 limitations, Equation 15 reconstruction, paper replication, marginal scores and default/HMC claims remain open.

Phase21 FP64 completed 49,152 records in 1,765.068 s. FP32 timed out at 2,100.220 s after 28,352 records: six complete cells and 3,776 rows of the next. The partial attempt lacks terminal result metadata; it is not a completed control. TF32 and terminal reporting had not run. This investigation launched no full retry.

The phase ledger charged 3,865.288 of 7,200 GPU-process seconds, leaving 3,334.712 s (55.58 min), with two of six attempts used. The supervisor reserves 2,100 s before each missing worker. Two worst-case reservations need 4,200 s. Specifically, a fresh FP32 retry must finish within 1,234.712 s for the unchanged supervisor to reserve another 2,100 s for TF32. The larger campaign balance does not automatically expand this phase allowance.

The next repair should add frequent progress/rate/resource measurements and recoverable stream-block outputs with source/input/completeness checks, preserving the frozen numerical call and every neutrality check. Current progress advances only after a whole cell, and a failed mode restarts as a whole. Any batching or loop rewrite is a separate numerical-equivalence candidate because compiler inputs and trajectories can change. A bounded preflight must establish parity, stable tracing, bounded memory and a viable time allowance before a full retry. No phase budget was increased here.

| Decision | Evidence | Main uncertainty | Next justified action |
|---|---|---|---|
| Diagnose host-driven duplicate filter execution and a supervisor timeout | Source plus executable timing | Original FP32 slowdown not reproduced | Instrument progress/resources and recoverability |
| Reject repeated per-stream retracing as demonstrated cause | Saved FP64 and new FP64/FP32 counters | TF32 long-run counts not remeasured here | Keep per-instance counters |
| Retain existing center repair | Source identity and saved center/control checks | Broader precision/resampling failures remain | Preserve numerical source pending distinct validation |
| Make no scientific/default or speed-ranking claim | Debugging-only audit | Short conditional observations | Complete valid phase before interpretation |

The saved session /home/chakwong/.codex/sessions/2026/09/23/rollout-2026-09-23T16-30-21-01a0cd63-2237-74f3-aace-6e9c415dd4a1.jsonl repeatedly reports "remote compaction v2 expected exactly one compaction output item". That is the agent-session failure, not a TensorFlow exception. Its detached supervisor continued until the worker timeout.

## Allowlist and execution

At the owner's request, installed /home/chakwong/.codex/rules/bayesfilter-iapf-execution.rules adds exact checkout-specific read-only Git status/diff/revision prefixes. All three were also approved and saved in the current session. Existing rg, sed, nvidia-smi and the absolute tftwogpu interpreter approvals are reused. [Seven command forms](allowlist-validation.json) passed policy checks; inline examples also check nonmatching mutations.

Use git -C followed by the exact execution-checkout path, then status or rev-parse, or diff --no-ext-diff --no-textconv in that order. Use the absolute interpreter directly. Avoid adding env, shell wrappers, assignments or redirection around approved command prefixes. GPU commands still use trusted/escalated execution. Request genuinely new prefixes before dependent work, with a concrete command and scope ready for review. The allowlist does not expand campaign budgets.

[Official Codex rules](https://learn.chatgpt.com/docs/agent-configuration/rules) describe exact argument-prefix matching, startup loading of manual rules, and persistence of live approvals. The sandbox also returned "mountinfo path is not absolute" before several commands executed; approved escalated inspection succeeded. That launch failure is separate from GPU health.
