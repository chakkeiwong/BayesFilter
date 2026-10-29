> Current execution status, 2026-09-18: shared engineering repairs pass the
> exercised checks; LG1 failed all three permitted calibration attempts.
> Posterior/default validation and dependent model execution are blocked.
> See [execution result](ledh-graceful-failure-execution-result-2026-09-18.md).
> The earlier account below is preserved as history, not current completion evidence.

# LEDH graceful-failure execution handoff — 2026-09-17

Ready to begin Phase 0 and the shared Phase 1 repair in this isolated worktree. The master program has not been executed. This checkpoint supersedes the old proposal/restoration checkpoint for the next execution session.

## Working location and governing program

- Worktree: `/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917`.
- Branch: `ledh-graceful-failure-execution-20260917`. Verified HEAD: `6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef`; local `main` and `surrogate-hmc` matched at this check.
- Master program: [revised program](ledh-graceful-failure-revised-program-2026-09-17.md). SHA-256: `3b1725f1f2114e764d306b9090e80c05411f36f590c6abc942f58f14048d92de`.
- Review: [Claude assessment](ledh-graceful-failure-revised-program-claude-review.md), which recommends adoption and Phase 0–1 followed by LG1.
- Evidence: [readiness record](artifacts/ledh-graceful-failure-execution-readiness-20260917-01/readiness.json). Full setup/test logs are in that directory.

Use this worktree as the explicit working directory for all commands and output paths. The source checkout has concurrent uncommitted documentation edits, including the old LEDH proposal checkpoint; those edits were preserved and are not needed for this program's code. The revised program itself matches the source copy. Git metadata, the conda environments, CPU resources, and GPUs remain shared.

## Checked findings and repaired setup gap

All 25 reviewed source entries match the technical review manifest. All eight local links in the program resolve, its five initial test files exist, and Python resolves `bayesfilter` from this worktree. The public HMC tuning documentation and active capability registry are present; the appropriate endpoint-corrected binding still needs executable qualification during the program.

The new worktree initially lacked `bayesfilter/ops/_symmetric_sylvester_ops.so`, an ignored build product. This caused four test-collection errors. A fresh private build now provides it. The first build selected incomplete CUDA headers; explicitly selecting TensorFlow's complete CUDA headers and the existing LLVM headers resolved the build. No package or shared environment was changed. The library is local to this worktree; no dependency on the original checkout's binary was introduced.

Verification after the repair: **37 tests collected successfully; four focused native-op tests passed**, including reference/residual and CPU/XLA comparisons for the symmetric Sylvester and principal-square-root operations. These checks deliberately used `CUDA_VISIBLE_DEVICES=-1`; they establish setup/import and primitive compatibility only. The LEDH regressions were collected, not run, and known containment defects remain the work of Phase 1.

## Skeptical readiness audit

The current code still matches the reviewed defects, so the correct starting point is repair, not repeating historical completion claims. The program uses current healthy canonical computations and independent references as comparators; separates smoke/acceptance diagnostics from scientific promotion; gives explicit vetoes, repair triggers, and budgets; and requires scoped tuning and disjoint claim data. No missing source or checkout prerequisite was found. Adapter feasibility, statistical thresholds, posterior-reference cost, and GPU/XLA call-chain behavior remain explicit early-stage questions. Claude's confidence and success-probability estimates are advisory, not measured feasibility evidence.

The old `ledh-graceful-failure-STATUS.md`, completion summaries, and proposal checkpoint are stale. Correct active status in Phase 0 while preserving the dated historical results. The proposed `scripts/run_ledh_graceful_failure_program.py` and `docs/benchmarks/ledh_graceful_failure_reference_diagnostics.py` do not yet exist; they are planned implementation deliverables, not runnable launch commands.

## Exact next action and bounds

1. Read current governing instructions and master-program §§2–5 and §12. Verify this branch and relevant changes. Initialize one versioned campaign directory and a budget/attempt ledger; retain this readiness record as setup evidence.
2. Complete Phase 0: correct active status, map the actual consumer endpoints and reviewed defects, and preserve a fixed healthy full-feature fixture.
3. Reproduce R1/R2/R6 through the smallest actual consumers. Repair shared containment and its focused regressions before dependent inference.
4. Take LG1 through derivatives, model specification, scoped LEDH/HMC tuning, and the controlled posterior comparison before expanding.

The program ceiling is **64 GPU device-hours and 16 aggregate CPU worker-hours**, one GPU campaign at a time, initially at most four GPU-hours per job and at most three substantive attempts per case. No scientific campaign has started. The setup collection/smoke timings are recorded; compiler CPU usage was not measured. Account for setup work in the initial ledger rather than treating readiness checks as scientific evidence.

At the trusted inventory check, GPU UUID `GPU-68251639-fe82-8f81-3ccc-2953c32e805b` (RTX 4080 SUPER) was idle; the RTX 5080 had an existing allocation. Recheck availability before launching. Use trusted GPU access, select by UUID, set and verify TensorFlow memory growth before initialization, and preserve GPU/XLA provenance. Worktree isolation does not reserve a GPU or isolate shared package installations.

## Rebuilding the private native library

Run from the execution worktree; the build requires write permission there. These commands use installed headers and runtime libraries and deliberately hide GPU devices during configuration:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 cmake \
  -S . -B .localresources/builds/ledh-graceful-failure-native \
  -DPython3_EXECUTABLE=/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  -DPython3_ROOT_DIR=/home/chakwong/anaconda3/envs/tftwogpu \
  -DCUDA_INCLUDE=/home/chakwong/anaconda3/envs/tftwogpu/lib/python3.11/site-packages/tensorflow/include/third_party/gpus/cuda/include \
  -DLLVM_INCLUDE_DIR=/home/chakwong/anaconda3/envs/tf-gpu/include
cmake --build .localresources/builds/ledh-graceful-failure-native \
  --target symmetric_sylvester --parallel 2
```

The actual readiness build used `/tmp/ledh-execution-readiness-20260917-t34qizb_/cmake-build`; its configure/build logs, compiler flags, link command, binary hash, and test commands are preserved with this handoff. Recompilation is unnecessary unless relevant sources or the TensorFlow environment change.
