# Zhao-Cui Austria SIR Score Completion Reset Memo

Date: 2026-08-03

Status: `STOPPED_AT_T3; FRESH_PROPOSAL_REPAIR_PLAN_REQUIRED`

## Reboot Summary

The Zhao-Cui-derived Austria SIR `T=20` score is **not complete**.

The manual frozen finite-score implementation is working on the scopes that
were checked. It passed same-scalar derivative, finite-difference, replay,
additivity, prefix-identity, GPU, and XLA tests. The blocker is the proposal:
the fixed nine-branch rank-one guide family passes T1 and T2 but fails the
predeclared proposal-quality gate at T3.

The stopped campaign must not continue to T5, T10, T20, untouched claims, or
HMC. The next task is to write and review a fresh proposal-repair plan.

## Read In Order

1. `AGENTS.md` from the active session.
2. `docs/plans/bayesfilter-zhao-cui-austria-sir-score-completion-result-2026-08-03.md`
3. `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/terminal-block-t3-01/manifest.json`
4. `docs/plans/bayesfilter-zhao-cui-austria-sir-score-completion-plan-2026-08-02.md`
5. `docs/plans/bayesfilter-zhao-cui-austria-sir-score-completion-plan-review-result-2026-08-02.md`
6. `bayesfilter/highdim/zhao_cui_austria_sir_rank_one_proposal_tf.py`
7. `scripts/run_zhao_cui_austria_sir_frozen_score_tuning.py`
8. `tests/highdim/test_zhao_cui_austria_sir_rank_one_proposal_tf.py`

The August 2 plan is historical execution authority and is now stopped. Read
it for the mathematical/evidence contract; do not resume its launch sequence.

## Exact Scientific Target

The selected target is the value and manual total derivative of a repository-
defined frozen importance-filter scalar with event order

```text
x0 -> transition -> y1 -> ... -> transition -> y20
```

and parameters

```text
(log_kappa_scale, log_nu_scale, log_obs_noise_scale).
```

This target was explicitly accepted as a replacement for the historical
trained-TT normalizer. It is classified `extension_or_invention`. It is not the
source-faithful Zhao-Cui Austria parameter score: the author Austria example
fixes kappa and nu and sets parameter dimension `d=0`.

## What Is Verified

- The FP64 manual score recursion differentiates the same frozen finite scalar
  on the checked tiny/T2 cases.
- The material agreement rule is frozen as

  ```text
  abs(a-b) <= 5e-6 + 5e-6*max(abs(a),abs(b)).
  ```

- The proposal compiler, RK4/time recursion, score evaluator, and theta sweep
  use TensorFlow batching and `tf.while_loop`.
- No NumPy numerical path or Python numerical loop contributes to the claim-
  owned numerical path.
- The outer tuning graph contains `StatelessWhile` and has no `PyFunc`,
  `EagerPyFunc`, or `MapDefun` host callback.
- Stateless streams are time-major. Repository-owned prefix views make T1,
  T2, and T3 literal prefixes of one frozen T20 parent program.
- Focused CPU-hidden result: `15 passed, 2 warnings in 39.12s`.
- GPU evidence used an NVIDIA GeForce RTX 4080 SUPER, FP64, TF32 disabled,
  XLA JIT, and a 6,144 MiB logical-device limit configured before GPU
  initialization.

Parent T20 program ID:

```text
936216cd0a8c4cab2b4551b6d44d99821a662046441c2cfb07200c2e23438fad
```

## Exact Blocker

The frozen gate requires all branches to be finite and, at every parameter
point, at least one guide satisfying both

```text
ESS/N >= 0.10
maximum normalized particle weight <= 0.10.
```

T1 and T2 pass only the `0.03` half-width box. The wider `0.10`, `0.25`, and
`0.50` boxes fail. At T3, three mixed points in the `0.03` box have no viable
guide:

| Theta | Best ESS/N | Max weight for best-ESS guide | Veto |
|---|---:|---:|---|
| `(0.015, -0.0075, 0.0225)` | 0.0960390 | 0.0516172 | ESS |
| `(-0.015, 0.0075, -0.0225)` | 0.0703704 | 0.0526825 | ESS |
| `(0.018, -0.021, -0.012)` | 0.0456736 | 0.102963 | ESS and maximum weight |

All T3 values and scores are finite. The graph, identity, and allocator gates
pass. This is a proposal-concentration failure, not a score-kernel, target,
data, GPU, or XLA failure.

## Preserved Artifacts

- Preflight:
  `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/phase2-fp64-gpu-xla-preflight-01/result.json`
- T1 pass:
  `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/phase3-t1-persistent-guide-tuning-01/tuning.json`
- T2 pass:
  `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/phase3-t2-persistent-guide-tuning-01/tuning.json`
- T3 block:
  `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/phase3-t3-persistent-guide-tuning-01/tuning.json`
- Terminal result:
  `docs/plans/bayesfilter-zhao-cui-austria-sir-score-completion-result-2026-08-03.md`
- Terminal manifest:
  `docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/terminal-block-t3-01/manifest.json`

Do not delete, overwrite, or reuse these output directories.

## First Action After Reboot

Do not launch an experiment immediately. First write a fresh bounded proposal-
repair plan under `docs/plans/` and subject it to the required skeptical audit.
The plan must preserve the scientific target and classify defaults before
choosing a candidate.

The smallest useful next experiment is a T3 discrimination screen for a richer
proposal. Plausible hypotheses are:

1. an XLA-native higher-rank squared-TT/KR proposal with target-specific L1
   tuning and disjoint calibration/validation data; or
2. a predeclared broader persistent-guide family whose locations, branch count,
   and proposal seeds are selected without tuning on the three failed points.

The three failed points are now observed negative evidence. Preserve them as
fixed regression/admission points, but do not use them to choose the repair.
They are not untouched claim points. Reserve new, predeclared untouched points
for a later claim only after the repaired candidate passes calibration and
validation.

Any learned repair must be batch-native TensorFlow/XLA throughout training,
proposal compilation, tuning, and selection. Python optimizer/sample/axis/
microbatch loops, NumPy numerical work, `tf.map_fn`, scalar fallbacks, and
runtime autodiff remain ineligible.

## Do Not Do These Things

- Do not weaken the ESS or maximum-weight gates.
- Do not shrink the `0.03` domain after seeing the T3 result.
- Do not select a replacement seed post hoc to erase the failure.
- Do not revive the generic retained full tensor-product grid as a production
  Zhao-Cui route.
- Do not call the route source-faithful, exact physical likelihood, HMC-ready,
  posterior-correct, default-ready, or production-ready.
- Do not run T5/T10/T20 or untouched claims before a fresh candidate passes the
  staged T1/T2/T3 contract.
- Do not modify or clean unrelated KSC/UKF worktree changes.

## Environment And Commands

Environment:

```text
repo: /home/chakwong/BayesFilter
conda prefix: /home/chakwong/anaconda3/envs/tf-gpu
Python: 3.11.14
TensorFlow: 2.19.1
TensorFlow Probability: 0.25.0
base git commit: f3ca5aa990fa0997414359983da2e93be8bee40c
worktree: dirty; Zhao-Cui changes are not committed
```

Focused reference verification, if needed:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run \
  -p /home/chakwong/anaconda3/envs/tf-gpu \
  python -m pytest -q \
  tests/highdim/test_zhao_cui_austria_sir_fixed_variant_tf.py \
  tests/highdim/test_zhao_cui_austria_sir_rank_one_proposal_tf.py
```

All GPU detection, compilation, tuning, or execution commands must run with
trusted/escalated GPU access. The 6,144 MiB logical-device cap remains fixed.

## Required Next Verdict

The next plan must end its first stage with one of:

```text
PASS_REPAIRED_PROPOSAL_T1_T2_T3
BLOCK_REPAIRED_PROPOSAL_QUALITY
BLOCK_REPAIRED_PROPOSAL_IMPLEMENTATION_OR_XLA
```

Only the first verdict permits planning T5/T10/T20. It still does not establish
physical-likelihood accuracy or any HMC/posterior claim.

## Update 2026-08-03: First Action Completed

The required first action ("write a fresh bounded proposal-repair plan under
`docs/plans/` and subject it to the required skeptical audit") is **done**:

```text
docs/plans/bayesfilter-zhao-cui-austria-sir-proposal-repair-plan-2026-08-03.md
```

A future agent should not repeat it. Nothing was launched; no new artifact
directory was created.

Verification performed against this memo before writing that plan, all clean:
all five source-file SHA-256 hashes and all four artifact hashes match the
terminal manifest byte-for-byte; the five preserved artifact directories are
present; `run_zhao_cui_austria_sir_frozen_score_claim.py` is absent, consistent
with stopping before Phase 5. This memo was not stale.

**Base commit has advanced.** The "Environment And Commands" section above
records `f3ca5aa990fa0997414359983da2e93be8bee40c`. During this planning session
a concurrent session committed `efce62b5aaf5b540811286511905a7765efe952c` ("Add
repaired KSC gaussian-sum NeuTra route, campaign evidence, and broad-grid
manifest provenance fixes"), which is why the KSC/UKF worktree modifications
listed at session start no longer appear as dirty. That commit touches **no**
Zhao-Cui or Austria file, and all four relevant Zhao-Cui source hashes remain
byte-identical to the terminal manifest, so this campaign's evidence and line
citations are unaffected. Zhao-Cui changes remain uncommitted.

Three findings from that work that this memo did not contain:

1. **The T3 block is `kappa/nu` guide-lattice resolution, not the
   observation-noise coordinate.** Stratifying the 23 design points by distance
   `d` in the `(log_kappa_scale, log_nu_scale)` plane to the nearest guide node:
   `d=0` (15 points) gives `ESS/N` 0.990-1.000, all pass; `d~0.0106` (2 points)
   gives 0.353 and 0.550; `d~0.0150-0.0168` (6 points) gives 0.046-0.367 and
   contains all three failures. Every `theta_3 = +/-0.03` point that is on-node
   in `(kappa, nu)` reaches `ESS/N ~ 0.99`. The mechanism is visible in the 20
   *passing* points alone, so it does not depend on the holdout failures.
2. **Per-step ESS decay is bimodal**: roughly 1% loss per step on-node, versus
   tens of percent per step off-node. The worst point `(0.018, -0.021, -0.012)`
   runs `0.8056 -> 0.3565 -> 0.0457` (per-step ratios `0.443` then `0.128`),
   while the best off-node passer sits near ratio `0.72`. A repair that only
   rescues T3 would plausibly move the block to T5. The new plan measures T5
   directly rather than extrapolating; the geometric-decay reading is an
   unverified extrapolation from three horizons and is not promoted to evidence.
3. **The exact-density mixture route in-repo is `not checked`, not refuted.**
   `make_rank_one_mixture_branch_tensor_compiler` evaluates the exact conditional
   mixture density `logsumexp_c[log P(c|a) + log q_c(x)]`, so the August 2 plan's
   `K^-T` matching-path rejection (line 412) does not apply to it: the weight
   never references the realized component. It has **zero callers** repository-
   wide and **no artifact** under `docs/`, so line 521's "failed per-time
   mixtures retained as negative evidence" is an **unsupported claim**.

Open owner decision recorded in the new plan: August 2 plan line 413 predeclares
rejecting "the analytic proposal route" on T3 failure. Under a broad reading that
rejects all analytic proposals and the repair must be a learned higher-rank
squared-TT/KR proposal; under a narrow reading it names only the persistent-guide
route. The new plan relies on the narrow reading, consistent with this memo's own
allowance of an analytic "broader persistent-guide family", and gates its Stage A
screening on owner confirmation.
