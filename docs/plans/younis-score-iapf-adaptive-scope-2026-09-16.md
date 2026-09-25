# Adaptive iAPF selection and cost reporting

Completed: [result and phase refresh](artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-adaptive-scope-result-and-refresh.md). Ten GPU rows and eleven main consumer tests pass. Allocation closes at 46/60 attempts, 2/3 GPU launches, 52.191/1800 GPU seconds and 219.98/5400 CPU process seconds. Seven implementation/test files are integrated into main; the source snapshot remains frozen.

This is Phase 0E implementation inside the authorized score master, following the completed control-safety screen. It closes the specific gap that keeps the existing bounded scalar Gaussian iAPF consumer restricted to mechanics. It does not broaden its mathematical model class or promote a default.

## Skeptical audit and research intent

The existing adapter records an actual particle count but selection assumes a fixed count and does not accept iAPF. Merely adding iAPF to the tunable list would conceal variable work and permit misleading equal-N comparisons. The preceding phase-refresh sentence forbidding fitting on evaluation observations was also overstrict: fitting a proposal to those observations is part of this declared iAPF procedure. Hyperparameters, nominal fit parameter, optimizer precision, initial count and resource caps are selected using calibration only. At each new dataset the procedure fits from independent offline streams, then freezes its coefficients and realized count before final evaluation. No oracle score, final likelihood sample or heldout error enters that fitting procedure. This correction preserves the existing algorithm; it does not authorize test-error tuning.

For each dataset y, write the fitting result as H(y,U_fit)=(psi,N). The final finite likelihood program is L(theta;y,U_final,psi,N), where U_final is independent of U_fit. The reported score differentiates L with psi, N, ancestor labels and mixture labels fixed. It is not a derivative through the adaptive stopping/count decisions, and no unbiased marginal-score conclusion follows. Selection compares the declared procedure, with an initial count and finite upper count, rather than falsely claiming a common realized N. Changes to either requested scope or selected procedure invalidate consumption.

| Role | Requirement |
|---|---|
| Main question | Can a repository-issued selection be consumed by the real iAPF endpoint with explicit adaptive-count and offline/online cost evidence? |
| Baseline | Existing mechanics iAPF at frozen `cbcfea1e`; shared fitted-twist kernel and existing fixed-count selection semantics |
| Engineering pass | Actual calibration/validation → selection → fresh final evaluation executes; scope, data separation, count history, fit digest and final random-stream independence are checked |
| Promotion criterion | None: this is an implementation/selection-plumbing study, not a scientific efficacy comparison |
| Promotion veto | Missing fit cost/count evidence, wrong scope, modified selection, reused tuning data, invalid fit/count, or a claim of fixed-N fairness from varying-N results |
| Continuation veto | Source drift, broken old-consumer behavior, invalid derivatives, unsupported backend, malformed evidence or budget exhaustion |
| Repair trigger | A localized validation, artifact or consumer failure; preserve it and repair within the same allocation |
| Explanatory diagnostics | Realized N, offline iteration counts, fitting/pass work counts, measured offline/final wall times, fit margins and descriptive error to Kalman |
| Nonconclusions | No method superiority, unbiased model score, nonlinear coverage, exact author-code replication, fully tuned control family, matched-cost claim or default readiness |

## Implementation and assumptions

Add a versioned adaptive-count contract to iAPF scope only. Preserve all settings and candidate-family fields already checked by selection. Record initial and realized counts, each offline count/action, final fixed count, fit observation digest and seed separation. Validate that count changes follow the existing controller and all counts remain within declared limits. Group candidate results by the selected algorithm controls; report actual count distributions and total offline/online work separately. Keep failed candidates visible through the existing complete-evidence requirement. Non-iAPF selection artifacts retain their current shape and semantics.

Add cost observability at the actual fitting/filter calls: synchronous offline/final elapsed time including compilation, particle-time points in offline filter passes, points supplied to fits, final particle-time points, and actual optimizer iteration totals. These counts describe operations, not FLOPs or equivalent runtime. Shared fits are currently recomputed by the adapter across replicates; report actual paid work and do not claim cache amortization. Runtime comparisons remain unavailable with concurrent workloads.

Fit settings are explicit mechanics hypotheses inherited from the already checked scalar fixture: k=1, tau=100, finite iteration/count caps, mean box 4, standard-deviation interval [0.2,4], floor ratio 0.01, tolerance 1e-7, 2,000 fit steps and 30 backtracks. The large tau deliberately reaches a short controller path and is not a quality default. A second tolerance/window arm or bounded controller fixture exercises count growth; no scientific conclusion may depend on forced controller outcomes. Offline FP64 fitting is explicit; final GPU filtering remains FP32/TF32/XLA. CPU FP64 tests are reference exceptions. Reference tests must show that requested N differs from realized N on at least one valid history and that invalid histories cannot enter selection.

Use the existing scalar Gaussian model, N_start=16 and T=2, six parameters. Use fresh calibration/validation/evaluation dataset IDs 800/810/820 and distinct fit/final seed roles. Tiny partitions test plumbing only. Any wider quality study must supply its own calibration, uncertainty and cheap-comparator design. Kalman and untwisted/one-step/fitted-twist endpoints remain the eventual comparison ladder; this implementation phase adds no ranking from their errors.

## Validation and allocation

First run focused CPU/XLA tests of the real consumer and selection lifecycle, including missing/mismatched selection, data leakage, changed initial/maximum count, edited count/fit evidence, independent final replicates, finite cost fields and fixed coefficient derivative semantics. Reuse existing iAPF objective/controller/derivative tests where code changes affect them. No pfor or runtime NumPy is allowed. A bounded GPU run then checks actual final endpoint consumption and memory growth with complete provenance. Freeze source before that run.

Allocation: at most **90 CPU process-minutes, 30 GPU wall-minutes, three GPU launches and 60 numerical row/fit attempts**, including localized failures and repairs. This is a fresh finite Phase 0E allocation within the master envelope, not reuse of exhausted control-screen attempts. Check prior cumulative manifest costs before GPU launch. The first GPU command must be preceded by a refreshed exact command, source commit, generated row list and successful CPU checks. Preserve outputs under `docs/plans/artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-adaptive-scope-*`; every retry gets a new directory. Stop unchanged retries after three failures of one cause. No environment mutation or new hardware is needed.

Pre-mortem: implementation could appear successful while silently hiding offline fitting cost, interpreting starting N as realized N, differentiating through adaptive decisions, fitting with the final random stream, or choosing hyperparameters from validation/claim errors. Actual consumer tests and replay of the count ledger address these early. An adaptive count limit failure rejects a candidate at the declared budget; it is not evidence against iAPF as a research direction. At exit preserve a decision/inference table, actual budget, source/command manifests and the next master obligation.

## Frozen GPU execution

Source `1c12eefa2ed55b81c4336fca4c006b9877c2df5f` is frozen in `.localresources/worktrees/younis-score-iapf-adaptive-scope-20260916`. CPU tests passed: 23 focused tests, eight after the final fit-evidence validation change, and three mandatory commit oracle checks. The sandbox commit attempt failed before execution because Git metadata was read-only; the trusted retry succeeded. No numerical failure occurred in that attempt.

Driver: `artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/run-iapf-adaptive-scope.py`. It generates four calibration/validation rows (k=1 or 2, datasets 800/810), two independent final samples of dataset 820 using repository-issued controls, and four mechanics comparators on dataset 820 (Kalman, bootstrap, fixed one-step twist, log-quadratic fitted twist). Initial N=16; final comparator N equals the selected procedure's realized N, without claiming equal total cost. Maximum row/recursive-fit charge is 36, below the 60-attempt allocation. The realized charge is recorded. Every row runs GPU/FP32/TF32/XLA; offline iAPF fitting explicitly uses FP64.

Run from the frozen worktree, using the exact full source revision above:

```bash
env CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/younis-score-mpl /usr/bin/time -p -o /home/chakwong/BayesFilter/docs/plans/artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-adaptive-scope-gpu-01.time timeout 1800 /home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter/docs/plans/artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/run-iapf-adaptive-scope.py /home/chakwong/BayesFilter/docs/plans/artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-adaptive-scope-gpu-01 --source-revision 1c12eefa2ed55b81c4336fca4c006b9877c2df5f
```

Trusted preflight found the selected RTX 4080 SUPER available but busy (14,925/16,376 MiB used, 80% utilization). The tiny mechanics run uses verified memory growth; timings cannot support comparisons. The conservative cumulative GPU budget check, including double-counted wrappers and the prior whole iAPF allocation, is 4,433.874 seconds before this 1,800-second allocation, below the master's 28,800-second envelope. Numerical failures remain repairable within this phase's remaining attempts and wall time.

Attempt 01 completed all six iAPF rows, with 14 recursive fits (20 charged attempts) in 21.851 seconds. The comparator batch was rejected before numerical execution because Kalman inherited the particle estimator label. This is a driver metadata failure, not a failed iAPF candidate. The original driver is preserved in the attempt directory. Repaired comparator metadata passes all four registry validations. Retry the same command and frozen source, changing every `gpu-01` path to `gpu-02` and the timeout to 1700 seconds. The fixed worst-case retry charge of 36 plus 20 already used is below 60; the remaining GPU time exceeds 1700 seconds. No scientific settings or selection inputs change.
