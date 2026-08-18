# Mild-smooth weighted NeuTra regression plan (2026-08-13)

Status: `MILD_HMC_COMPLETE_GERMAN_NEXT`

## Research intent and evidence contract

| Item | Contract |
|---|---|
| Question | Does the repaired weighted forward-KL route remain viable on the milder source-bound varying-Hessian target where historical plain NeuTra had a credible one-seed result? |
| Mechanism | Reuse only the source formula and frozen affine lift; fit a target-specific full-support replay proposal; then use disjoint target-specific training, selection, audit, and corrected fixed-length HMC stages. |
| Comparator | Historical plain result is context only. A matched reverse-KL comparator is required before any objective ranking; this lane first establishes weighted-arm viability. |
| Promotion criterion | Proposal support screen passes, then a frozen weighted candidate passes canonical sequential HMC R-hat/ESS and numerical status gates. |
| Promotion veto | Source value/score parity failure, nonfinite proposal/target, invalid artifact, failed sequential HMC gates, or missing hash-bound constants. |
| Repair trigger | Proposal ESS shortfall triggers proposal calibration. The one target-specific capacity rung has now been consumed; HMC rejection stops this mild rung and moves the campaign to German credit. |
| Explanatory only | Training NLL, clipping, acceptance, energy tails, and runtime. No ranking without paired uncertainty. |
| Nonclaims | No posterior correctness claim for the unnormalized source-bound target, no default promotion, no general NeuTra claim. |
| Artifact | Training: `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/varying-hessian/mild-smooth-serious-r1/`; HMC: `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/varying-hessian/mild-smooth-serious-hmc-r1/` |

## Source and default audit

| Choice | Provenance/status | Risk and early diagnostic |
|---|---|---|
| Source constants | Local read-only copy of historical replay-state JSON, SHA-256 `0aa2eb...0894b5`; nested `target_constants` only | Historical artifact could drift or be misbound; target name, full file hash, and source formula parity are required |
| Mild parameters | Source-bound `rot_alpha=0.35`, `weak_collapse=0.6`, `stiff_growth=0.25` | Wrong target variant; spec validation and source TensorFlow probe catch this |
| Proposal | Same affine-lift ridge family and reflection repair as strong, but re-screened for mild | Strong proposal numbers cannot transfer; fresh ESS/max-weight screen required |
| Capacity | `(64,64)`, 3 stages, 200-update canary first; `(128,128)`, 6 stages, 10,000 updates only if HMC or canary diagnostics trigger the reviewed repair | Under/over-capacity; disjoint selection/audit and downstream HMC decide |
| Learning rate | `1e-3` is a warm start from strong only, not a mild default | Short target-specific canary records clipping and NLL before serious training |

## Skeptical audit

The plan does not treat the historical plain result or training NLL as a promotion baseline. It binds a local constants copy, preserves the source file hash, runs proposal support before training, separates all replay partitions, and requires the actual downstream sequential HMC screen. The unnormalized target has no independent posterior authority, so the strongest allowed claim is sampler-route evidence, not posterior correctness. A valid failure is a repair trigger, not a research-direction veto; repeated failure after the one reviewed capacity repair stops this target rung.

Audit verdict: `PASS_FOR_STAGED_MILD_EXECUTION`.

### HMC execution addendum after serious training

The serious `(128,128)`, six-stage, 10,000-update capacity rung completed and
selected update 9000. Therefore the earlier one-repair allowance is exhausted:
the next action is one fixed-length HMC attempt, and HMC rejection is a stop for
the mild target rather than authority for another optimizer/capacity repair.

The HMC procedure is target-specific retuning, not transfer of the strong
kernel. The leapfrog grid `(3, 5, 10, 15, 20, 25)`, initial step size `0.10`,
and tuning budgets `(32, 64, 128)` are inherited baseline hypotheses from the
successful strong harness; every grid arm is evaluated anew on mild. `L=1` is
forbidden. Four chains, the canonical recent-window warm-up R-hat gate
`<=1.05`, retained R-hat gate `<=1.01`, and bulk/tail ESS gates `>=400` come
from `bayesfilter_neutra_sequential_hmc_v1`. The 3,600-second sequential cap is
a bounded-compute convenience limit; hitting it rejects this attempt but does
not establish a scientific defect. Fixed seeds are reproducibility choices, so
one passing run is viability evidence only and does not support comparative
ranking or default promotion.

Pre-mortem: a run could pass while the wrong checkpoint/target pair was loaded,
so the runner must verify both the checkpoint hash and training-manifest target
name/constants hash before tracing. It could fail from a nonviable inherited
grid rather than transport quality; the full predeclared grid and tuning
diagnostics preserve that explanation, but the mild rung still stops because
its repair budget is spent. Numerical nonfiniteness, invalid hashes, or a
broken GPU/XLA/memory-growth launch invalidate the attempt; ordinary HMC gate
failure is valid negative candidate evidence.

Addendum audit verdict: `PASS_FOR_ONE_MILD_FIXED_LENGTH_HMC_ATTEMPT`.

### Completion

The one authorized mild fixed-length HMC attempt completed with
`candidate_sampler_evidence_passed`. The detailed evidence and nonclaims are
recorded in
`docs/plans/bayesfilter-weighted-forward-kl-mild-smooth-result-2026-08-13.md`.

## Execution order

1. Run source/probe parity and focused target tests with the mild constants.
2. Run the CPU-only proposal diagnostic with target-specific constants and fresh output.
3. If proposal support passes, generate disjoint CPU training/selection/audit replay tensors.
4. Run a short GPU/XLA weighted canary, then a serious capacity rung only if the target-specific screen supports it.
5. Retune fixed-length HMC and run canonical sequential sampling; record sampler-only evidence and nonclaims.

## Mild canary nomination

The `(64,64)`, three-stage, 200-update GPU/XLA canary completed in `18.1 s` with a finite selected checkpoint. Disjoint selection NLL was `17.6335`; untouched audit NLL was `17.4013`, audit importance ESS fraction `0.47062`, max normalized weight `0.001953`, and `66/200` updates clipped. These are descriptive route diagnostics only. The finite canary nominates the reviewed serious capacity rung `(128,128)`, six stages, 10,000 updates, with the strong-screen warm-start learning rate `1e-3`; this is not a mild-target default promotion.
