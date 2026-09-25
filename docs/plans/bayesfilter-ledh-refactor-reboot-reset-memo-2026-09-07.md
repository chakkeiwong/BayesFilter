# BayesFilter Reboot Reset Memo

Date: 2026-09-07
Status: `REBOOT_BASELINE_DOCUMENTED`

## Purpose

This memo is the restart point for the current BayesFilter worktree.  It
records the branch split, the code and evidence that are being saved, the
scientific conclusions that are already supported, and the next actions that
are safe to perform after a fresh checkout.  It is a state record, not a new
promotion decision.

## Repository state at the reset

- Checkout: `/home/chakwong/BayesFilter`
- Active branch: `ledh-refactor-with-policy-fix`
- HEAD before this save: `167e10de` (`Record C2 Phase 8 root restoration`)
- Local `main`: `d26edcdf` (`Phase 2 driver: 45-run RQMC campaign with LGSSM-gated continuation veto`)
- Divergence from local `main`: 96 commits reachable only from this branch and
  99 commits reachable only from `main` (`git rev-list --left-right --count
  main...HEAD` reported `99 96`).
- Remote: `origin = git@github.com:chakkeiwong/BayesFilter.git`
- Upstream before this save: none.  The intended push target is
  `origin/ledh-refactor-with-policy-fix`.

This branch is intentionally separate from `main`.  Do not rebase, merge, or
switch the dirty root checkout during reboot.  Use a new clean worktree and a
new working branch based on the pushed save point when a new experiment is
needed.

## Save scope

At the initial inventory, `git status --porcelain=v1` reported the existing
dirty LEDH/RQMC/C2 set.  The save operation stages the complete visible set at
commit time (51 paths, including this memo and the Phase 4A timing pilot),
using ordinary Git history as the record.  No unrelated working-tree change is
discarded.
The two deleted RQMC helper scripts are intentional correction-state changes;
they are not C2 source deletion:

- `docs/benchmarks/run_rqmc_phase2b_repair.py`
- `docs/benchmarks/salvage_predator_prey_phase1_artifact.py`

The previous C2 Phase 8 restoration is already committed in
`21d5870f` and `167e10de`; this save does not replace it or restore an older
copy of the canonical LEDH files.

### Implementation and test paths saved

Canonical LEDH score and governance changes:

- `bayesfilter/highdim/ledh_alg1_contract.py`
- `bayesfilter/highdim/ledh_canonical_batch_tf.py`
- `bayesfilter/highdim/ledh_canonical_score_stages_tf.py`
- `bayesfilter/highdim/ledh_canonical_score_tf.py`
- `tests/highdim/test_ledh_canonical_batch.py`
- `tests/highdim/test_ledh_canonical_governance.py`
- `tests/highdim/test_ledh_canonical_total_gaussian_tf.py`

Younis/KDM diagnostic implementation and seed adapter:

- `bayesfilter/highdim/ledh_younis_kdm_tf.py`
- `bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py`
- `bayesfilter/inference/ledh_dual_force_adapter.py`
- `tests/highdim/test_ledh_younis_kdm_tf.py`
- `tests/highdim/test_ledh_younis_kdm_integrated_tf.py`
- `tests/inference/test_ledh_seed_policy.py`

RQMC correction, diagnostics, and campaign state:

- `docs/benchmarks/analyze_rqmc_ledh_phase3.py`
- `docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/campaign_summary.json`
- `docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/phase3_analysis.json`
- `docs/benchmarks/diagnose_rqmc_process_noise_confound.py`
- `docs/benchmarks/rebuild_rqmc_campaign_summary.py`
- `docs/benchmarks/run_phase2b_repair_campaign.sh`
- `docs/benchmarks/run_rqmc_ledh_initialization.py`
- `docs/memos/rqmc-ledh-phase2b-blocked-2026-09-04.md`
- `docs/memos/rqmc-ledh-correction-and-phase2b-completion-2026-09-05.md`
- `docs/memos/rqmc-ledh-confound-analysis-2026-09-06.md`
- `docs/plans/rqmc-ledh-init-v2-design-2026-09-06.md`
- `docs/plans/rqmc-ledh-v2-mathematical-audit.md`

C2 mixture-UKF/APF diagnostic scripts and plans:

- `docs/benchmarks/analyze_c2_mixture_ukf_apf_phase2_20260903.py`
- `docs/benchmarks/run_c2_mixture_ukf_apf_phase5_recursive_map_20260904.py`
- `docs/benchmarks/run_c2_mixture_ukf_apf_phase5b_rbf_20260904.py`
- `docs/benchmarks/run_c2_mixture_ukf_apf_phase5c_hybrid_20260904.py`
- `docs/benchmarks/run_c2_mixture_ukf_apf_phase5c_replication_20260904.py`
- `docs/benchmarks/run_sqmc_rerun_corrected_filter_20260906.py`
- `docs/memos/c2-mixture-ukf-apf-master-plan-refresh-2026-09-03.md`
- `docs/memos/genut-filter-initialization-clarification-2026-09-06.md`
- `docs/memos/genut-guided-identity-proof-2026-09-06.md`
- `docs/memos/section-3-6-prior-weight-sidecar-reset-2026-09-03.md`
- `docs/plans/sqmc-rerun-corrected-filter-2026-09-06.md`

Younis/KDM plans and bounded result:

- `docs/benchmarks/run_ledh_younis_kdm_phase4a_gpu_smoke.py`
- `docs/benchmarks/run_ledh_younis_kdm_phase4a_tf32_calibration.py`
- `docs/benchmarks/run_ledh_younis_kdm_phase4a_timing_pilot.py`
- `docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md`
- `docs/plans/bayesfilter-ledh-younis-kdm-phase4a-gpu-smoke-result-2026-09-07.md`
- `docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md`
- `docs/plans/bayesfilter-score-target-preserving-rewrite-proposal-2026-09-06.md`
- `results/ledh_younis_kdm_phase4a_gate_20260907.md`

Surrogate-force HMC execution records:

- `results/phase0-coverage-tests.txt`
- `results/phase0-tolerance-tests.txt`
- `results/phase2-output.txt`
- `results/phase3-seed-tests.txt`

## Evidence already established

### C2 Phase 8 recovery

The missing `HermiteBasis1D` and related Phase 8 call chain were restored and
committed.  The authoritative recovery record is
`docs/memos/c2-phase8-root-restoration-20260907.md`.  Its checked results are:

- focused Phase 8 contracts: 36 passed;
- broad C2 regression: 110 passed, one explicitly deselected expensive degree-
  12 oracle, and two warnings; and
- GPU/XLA Phase 8A mechanics: all 15 checks passed, including finite value and
  analytical score, central finite-difference error `1.3641e-08`, exact
  XLA/non-XLA value agreement, and proposal-density discrepancy
  `1.77636e-15`.

Those checks establish wiring and finite-program mechanics only.  They do not
establish proposal efficiency, posterior correctness, statistical superiority,
generality, HMC readiness, or default readiness.  Phase 8E must run from a
clean committed worktree after the save point; do not use the dirty root for
that launcher.

### RQMC initialization

The V1 RQMC campaign is invalid as evidence for initialization quality.  The
runner used one random stream for initial and process noise, so MC and RQMC
arms received different process-noise trajectories for the same seed.  The
`genut_guided` arm also had severe replicated-design degeneracy and reset-aware
pre-alignment.  The corrected verdict is `MIXED_REJECT` relative to the
confounded data, but the campaign does not answer the intended question.

The V2 mathematical design removes the process-noise confound with independent
streams, but its audit recommends dropping `genut_guided` and adding Latin
hypercube sampling.  No V2 claim campaign has been run.  Any reboot must use
the corrected seed contract, a nondegenerate arm set, and a fresh evidence
contract rather than reusing V1 results.

### Surrogate-force HMC

The executable program records the following bounded results:

- Phase 0: tolerance derivation, frozen seed policy, and joint coverage tests
  passed (21/21 CPU-only).
- Phase 1: JVP parity, Sinkhorn/moment convergence, and GPU memory-growth
  checks passed (54/54 CPU-only checks plus the RTX 4080 SUPER smoke).
- Phase 2: exact-force versus damped-force HMC on the 3D quadratic passed the
  predeclared W2 criterion (`0.069 < 0.100`).
- Phase 3 seed-policy tests passed: deterministic replay, different-seed
  control, reversibility, and no call-count dependence (5/5).

These results validate the mechanism on the declared fixtures.  They do not
yet validate the full LEDH score on the target model suite.

### Younis/KDM score investigation

Phase 4A is implemented as a diagnostic, full-feedback
`kernelized_observation_weighting` route sharing the canonical executor.  It
is labelled `KDM-FINITE`, not a full Younis mixture posterior and not the
canonical atom target for positive bandwidth.

The focused CPU reference suite in the project environment passed:

```text
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python \
  -m pytest -q \
  tests/highdim/test_ledh_canonical_batch.py \
  tests/highdim/test_ledh_canonical_governance.py \
  tests/highdim/test_ledh_canonical_total_gaussian_tf.py \
  tests/highdim/test_ledh_younis_kdm_tf.py \
  tests/highdim/test_ledh_younis_kdm_integrated_tf.py \
  tests/inference/test_ledh_seed_policy.py
```

Observed result: `40 passed, 2 warnings in 64.70s`.  The warnings are the
existing TensorFlow Probability deprecation warnings.

The prior trusted GPU/XLA gate passed with FP32 and TF32 disabled.  The TF32
arm failed the exact identity and parity gates; it is not admitted for this
route.  No paired model-score error campaign has been run, and Phase 4B is
mathematically blocked until the complete sequential proposal law and
derivative are derived.

## Mechanical checks in this save pass

- Existing changed Python files were compiled with the `tftwogpu` Python 3.11
  interpreter; intentionally deleted Python files were excluded.
- `git diff --check` passed.
- `bash -n docs/benchmarks/run_phase2b_repair_campaign.sh` passed.
- An initial system-Python pytest attempt failed at collection because that
  interpreter has no TensorFlow.  It is an environment diagnostic, not a code
  result; the same suite passed in `tftwogpu` above.
- The first syntax command also included the two intentionally deleted RQMC
  files and reported `No such file or directory`; the corrected existing-file
  check passed.

## Active blockers and boundaries

1. The root worktree is the integration/save checkout, not the execution
   checkout for a new long experiment.  Use a clean worktree after the push.
2. Phase 4A is restricted to the no-TF32 route unless a new, route-specific
   numerical contract is reviewed.  Do not silently inherit the DPF TF32
   default for score-sensitive KDM work.
3. Phase 4B KDM mixture-resampling remains math-blocked.  Existing density and
   anchored-weight helpers are diagnostic primitives only.
4. RQMC V1 is quarantined.  V2 is design/audit state only and needs a fresh
   campaign; no old V1 value should be used as a baseline.
5. C2 Phase 8E requires clean integration, the committed observation-bin
   artifact, exact complete-mixture denominators, and its stated fixture-level
   uncertainty.  A one-branch ESS table is not promotion evidence.
6. The `tftwogpu` TensorFlow 2.20 development build is the ABI-matched runtime
   for the restored C2 custom operation.  The separate `tf-gpu` 2.19.1
   environment is not interchangeable for that path.

## Reboot procedure

After the push, start a new task without altering this root checkout:

```bash
git fetch origin
git worktree add -b reboot/20260907 \
  /tmp/bayesfilter-reboot-20260907 \
  origin/ledh-refactor-with-policy-fix
cd /tmp/bayesfilter-reboot-20260907
git status --short --branch
git log -1 --oneline --decorate
```

Choose exactly one research lane before executing:

- **C2:** run the Phase 8E preflight and replication from the clean worktree;
  preserve all source hashes and use the reviewed statistical aggregation.
- **Younis/KDM:** first freeze the no-TF32 Phase 4A route decision, run the
  bounded timing gate, then run the paired LGSSM/Kalman oracle pilot.  Do not
  implement Phase 4B from the current incomplete proposal expression.
- **RQMC:** revise the V2 arm set to independent-seed MC/Sobol/Halton/LHS
  comparisons, retain the process-noise hash check, and run only after the
  plan survives its skeptical audit.
- **Surrogate-force HMC:** continue the executable master program under its
  frozen estimand, comparator, vetoes, repair budget, and seed policy.  A
  failing candidate is a result or repair trigger, not permission to change
  the contract.

Before a serious run, create a fresh versioned artifact directory and record
the branch commit, environment, device/XLA/TF32 state, memory policy, seeds,
command, wall time, and plan path.  Do not overwrite historical artifacts or
run a dirty-tree launcher that fails closed.

## Nonclaims

This reset memo does not claim that the current LEDH, RQMC, C2, or KDM method
is statistically superior, posterior-correct, generally valid, HMC-ready,
production-ready, or default-ready.  It records implementation checks,
negative findings, and the conditions required for the next discriminating
experiment.

## Authoritative restart references

- C2 recovery: `docs/memos/c2-phase8-root-restoration-20260907.md`
- C2 replication: `docs/plans/c2-phase8e-statistical-replication-20260907.md`
- RQMC V2 design: `docs/plans/rqmc-ledh-init-v2-design-2026-09-06.md`
- RQMC mathematical audit: `docs/plans/rqmc-ledh-v2-mathematical-audit.md`
- Younis/KDM reset: `docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md`
- Younis/KDM integrated plan: `docs/plans/bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md`
- Surrogate-force execution contract: `docs/plans/execution-contract-2026-09-07.md`
- Surrogate-force Phase 0-2 completion records: `docs/plans/phase0-complete.md`,
  `docs/plans/phase1-complete.md`, and `docs/plans/phase2-complete.md`

## Supersession checklist

- [x] The current C2 root-restoration memo supersedes the earlier statement
  that the root worktree had not been modified after restoration.
- [ ] No historical RQMC or KDM result is silently promoted by this memo.
- [ ] If a later terminal result materially changes one of the four lanes, it
  must add a scoped supersession or correction banner to the affected plan.
