# SSL-LSTM q=20 Phase 9B M4-P0 Executable Readiness Result

Date: 2026-09-06  
Updated: 2026-09-07  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md`  
Parent master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Status: `M4_P0_BUDGET_INFEASIBLE_P1_BLOCKED_P2_BLOCKED`

## Latest result: September 7 multi-GPU execution

The GPU-0-only blocker below is historical and superseded by the owner's
three-GPU placement policy. A real diagnostic ran on eligible non-display
GPU 1 and completed factor first/steady calls in 1,360.86/1,390.70 seconds.
The six-chunk factor extrapolation is 8,314.36 seconds before overhead,
incompatible with the 2,600-second arm cap. The strict arm was interrupted.
All diagnostic time was settled once; 183.80 seconds remain, with no active
reservation. No complete M4-P0 or P1 pass exists, and P2 remains blocked.

The detailed result, decision/inference tables, termination record, GPU
provenance, reporting gaps, and next Phase 0 work are in
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-multigpu-continuation-result-2026-09-07.md`.
The earlier sections below preserve the pre-launch static-repair history and
are not current launch instructions.

## Historical pre-launch decision

The non-GPU executable-readiness repairs pass focused validation, but M4-P0 does not
pass. Two independent prerequisites remain unresolved:

1. **GPU 0 availability:** the trusted probe at `2026-09-07T07:45:41Z`
   returned `requested_gpu_compute_busy:0`. The accompanying process snapshot
   identifies `/usr/NX/bin/nxnode.bin`, PID `6826`, using 312 MiB. The process
   was not stopped, and GPU 1 was not substituted because the reviewed P1
   contract specifies GPU 0. This is an external resource blocker.
2. **Campaign budget:** the authorized aggregate is 5,200 seconds, with an
   estimated historical debit of 1,832.61 seconds and at most 3,367.39 nominal
   seconds remaining. No reviewed budget decision funds both the two-arm
   readiness diagnostic and a complete fresh P1 schedule. This is a
   compute/governance boundary, not a request for another launch token. The
   diagnostic and P1 share one ledger; the closed factor campaign's
   11,800-second budget cannot be transferred.

Neither blocker is evidence against the q=20 target, factor backend, strict
comparator, or HMC method. No BayesFilter GPU workload was launched during this
remediation and no P1 posterior or sampler evidence was produced. An idle GPU
would resolve only the first prerequisite. Measured two-arm timing, a feasible
complete-schedule forecast, and the M4-P0 closeout would still be required.

## Completed repairs

- The shared batched HMC controller now uses an explicit fixed state/seed
  `input_signature`, with a repeated-shape single-trace regression.
- The full NeuTra route-policy audit passes after classifying the Phase 9B
  runner/diagnostic and preserving legacy non-HMC or historical dispositions.
- The persistent campaign budget ledger is implemented, tested, and wired into
  the P1 runner for attempt, chunk reservation, settlement, and source/plan
  binding.
- The source-owned two-arm readiness diagnostic is present and its direct CLI,
  compilation, and static route classification checks pass.
- P1 rejects a missing, stale, or non-passing executable-readiness closeout
  before TensorFlow import or campaign reservation. Source-only verification
  remains available to the diagnostic without creating a circular dependency.
- Diagnostic timing now includes first-call and steady-state measurements;
  whole-attempt budget settlement includes startup and cleanup, including
  failure paths, without minting an allowance per retry.
- Run-start, success, and failure receipts include Python/conda/platform/host
  and bounded GPU-environment provenance with the managed-session trust basis.
- Both controller calls enforce the shared per-chain movement check; synthetic
  stationary calls are rejected. Real q=20 movement remains unmeasured.
- The fresh source audit `r12` passes against the current P1 plan and runner,
  with status `PASS_PHASE9B_P1_PLAN_AUDIT` and no findings.

## Evidence boundary

The September 6 static suite passed with `29 passed`, the route-policy suite
with `6 passed`, and compilation/whitespace checks passed. Those are historical
repair checks, not validation of every September 7 change. On September 7 the
expanded standalone readiness regression passed with `12 passed`, followed by
the fresh source audit and combined checks:

| Check | Actual result | Evidence |
|---|---|---|
| P1 source-only audit `r12` | `PASS_PHASE9B_P1_PLAN_AUDIT`, no findings | `docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-plan-audit-20260907-r12/run_manifest.json` |
| Combined focused CPU suite | `72 passed, 191 warnings in 8.60s`; shell wall time `11.024s` | `validation-20260907-r12/focused-cpu-tests.log` under the M4-P0 artifact root |
| Full NeuTra route-policy suite | `6 passed in 2.10s`; shell wall time `3.290s` | `validation-20260907-r12/route-policy-tests.log` under the M4-P0 artifact root |
| Python compilation and whitespace | `py_compile` and `git diff --check` passed | `validation-20260907-r12/compile-whitespace.log` under the M4-P0 artifact root |

The exact audit, focused-suite, route-policy, and compilation commands are in
the plan's Required command set and were executed with
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. The test commands set
`CUDA_VISIBLE_DEVICES=-1` and `TF_CPP_MIN_LOG_LEVEL=2` before framework import:
GPU devices were intentionally hidden. The warnings are TFP/distutils and
Python AST deprecations, not test failures. These short synthetic/reference
checks provide engineering evidence only, not new target-level sampling data.

The September 6 blocked probe receipt is preserved at:

`docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/gpu-preflight-blocked-20260906/run_manifest.json`

The September 7 trusted probe and process snapshot are:

- `docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/resumption-20260907/gpu-probe.json`
- `docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/resumption-20260907/gpu-processes.csv`

The shared ledger implementation is tested, but its live campaign file has not
yet been initialized because no new runtime attempt has launched. Initialization
must retain the historical debit as an estimate, not silently reset spending.

The result establishes static engineering readiness only. It does not establish
GPU/XLA execution, posterior correctness, convergence, whitening, sampler
ranking, factor superiority, default readiness, or scientific validity.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| M4-P0 static repair | Passed: source audit, 72 focused CPU tests, and 6 route-policy tests | GPU availability and budget resolution still block runtime measurement | Synthetic checks do not establish target GPU behavior | Resolve both runtime prerequisites | No runtime readiness |
| P1 launchability | Not passed | No passing runtime diagnostic, feasible measured schedule, or M4-P0 closeout | Compile versus steady-state cost for factor and strict | Fund diagnostic plus P1, obtain fresh GPU 0 availability, measure, reconcile, then close P0 | No sequential sampler claim |
| P2 scientific validation | Blocked | Chart thresholds and downstream posterior/reference gates remain unresolved | Chart quality and posterior behavior | Keep P2 blocked independently of P1 mechanics | No posterior or default claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | GPU occupancy and unresolved budget block execution; synthetic tests reject stationary controller calls |
| Statistically supported ranking | None; no new stochastic comparison ran |
| Descriptive-only differences | The historical first-chunk time is a budget-planning observation, not a factor-versus-strict ranking |
| Default-readiness | Not established; narrow prior factor-backend admission is unchanged |
| Next evidence needed | Funded two-arm runtime measurements for readiness, followed by separate sequential and posterior/uncertainty evidence |

## Next action

Do not kill PID `6826` and do not launch on GPU 1 under the current contract.
First obtain a reviewed campaign-budget decision that allocates both diagnostic
and complete P1 work. Any expanded compute needs plain-language user approval,
not hash-bound wording or a review chain. Then obtain a fresh trusted GPU 0
availability check. Only when those conditions and the static checks pass may
the following conditional diagnostic command run; its allocation is deliberately
not invented here:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/diagnose_ssl_lstm_q20_phase9b_executable_readiness_2026_09_06.py \
  --output-dir \
  docs/plans/artifacts/ssl-lstm-q20-phase9b-executable-readiness-2026-09-06/runtime-diagnostic-<fresh-id> \
  --max-seconds <authorized-diagnostic-allocation-seconds>
```

Only a passing runtime diagnostic, complete-schedule budget forecast, and
Phase 0 closeout can open a fresh P1 attempt.

## Post-run red team

The strongest alternative explanation for the expensive historical chunk is
first-call compilation rather than equally expensive steady-state sampling.
That possibility motivates measurement; it does not justify a larger cap or
claim that the remaining budget suffices. The weakest evidence is the missing
strict-arm and steady-state runtime data. A funded, valid two-arm diagnostic
and complete-schedule reconciliation, together with a fresh free-GPU check,
would resolve the present readiness blockers. No current result rejects the
factor candidate or the research direction.
