# Strong-smooth reflected-proposal repair (2026-08-12)

Status: `AUDITED_EXECUTION_AUTHORIZED`

## Research intent and evidence contract

| Item | Contract |
|---|---|
| Question | Can a replay proposal for the source-bound `nk_like_strong_smooth` target attain adequate target-weight coverage after correcting a pilot's one-branch support failure? |
| Mechanism | Verify exact local reflection symmetry, fit the well-supported positive branch by self-normalized pilot weights, reflect its mean/covariance into the negative branch, and retain a 5% full-support defensive mixture. |
| Comparator | Existing disjoint `strong-smooth-proposal-r3` diagnostic, whose unconstrained two-branch pilot fit had ESS fraction `0.03647` and maximum weight `0.01344`. |
| Promotion criterion | A fresh 65,536-row CPU-only validation cloud is finite and meets the provisional proposal-nomination screen: ESS fraction `>=0.05`, max normalized weight `<=0.01`. This permits a GPU training canary only. |
| Promotion veto | Failed source-formula reflection value/score parity, nonfinite target/proposal, or failed independent validation screen. |
| Repair trigger | A screening shortfall is evidence to change proposal geometry/calibration; it is not evidence against weighted NeuTra or the source target. |
| Explanatory only | Pilot branch ESS, target weight concentration, and comparison with r3. Neither validates HMC nor posterior correctness. |
| Nonclaims | This run does not train a transport, evaluate reverse KL, tune HMC, establish posterior correctness, or promote a default. |
| Artifact | `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/varying-hessian/strong-smooth-proposal-r7-reflected/` with manifest, frozen proposal payload, result, and hashes. The incomplete r4/r5 directories record pre-result harness-gate repairs. |

## Assumption audit

| Choice | Provenance/status | Risk and earliest check |
|---|---|---|
| `x0` local reflection | Derived from the checked source formula: radius, weak/stiff terms are even in `x0`; angle and `y0` change sign while squared energy is invariant. | An implementation/sign error would create a false negative branch. Check source-bound value and transformed local-score parity on the entire pilot cloud before fitting. Float64 tail parity uses `|delta value| <= 1e-9 + 1e-10*scale` and `|delta score| <= 1e-10 + 1e-10*scale`: the pre-repair pilot measured relative maxima `6.69e-14` and `8.07e-13`, while absolute maxima arose from cancellation at large tail magnitudes. |
| Equal fitted branch masses | Derived from exact reflection invariance, not inferred from the asymmetric pilot. | If the symmetry check fails, do not fit/reflect. |
| 5% defensive mixture | Inherited proposal convenience hypothesis from r3, not a target default. | It can waste proposal mass. The independent ESS/max-weight screen exposes this. |
| ESS `>=5%`, max weight `<=1%` | Provisional calibration hypotheses, not literature-derived thresholds or posterior criteria. | Report raw values and preserve that passing only nominates a training canary. |
| 262,144 pilot / 65,536 validation rows | Existing r3 scale, bounded CPU diagnostic. | Independence is preserved by distinct stateless seeds and no reused validation rows. |

## Skeptical audit

The repair does not substitute proposal ESS for a posterior or sampler criterion. The r3 failure was a one-branch proposal fit despite a geometry with an exact source symmetry, so fitting from the positive branch and reflecting it directly tests the identifiable mechanism. The baseline, target constants, source hash, CPU-only boundary, and validation row count are unchanged. A pass authorizes only the already-planned GPU/XLA transport canary; a failure retains r3/r4 as proposal-calibration evidence and triggers a new proposal geometry rather than a scientific conclusion.

Audit verdict: `PASS_FOR_CPU_PROPOSAL_DIAGNOSTIC`.

## Execution

1. Add TensorFlow-only physical reflection and reflected-positive pilot proposal helpers.
2. Add source value and transformed-score symmetry tests plus proposal shape/mass tests.
3. Run the focused CPU-only test module.
4. Run the fresh r7 proposal diagnostic with `CUDA_VISIBLE_DEVICES=-1`; record the result and a hash-bound selected mixture payload.
5. A passing r7 proposal freezes its hash-bound mixture and proceeds to a separate 200-update GPU/XLA affine-local weighted-training canary. This canary requires finite batch-native updates and a selected heldout-NLL checkpoint but does not tune HMC or make a posterior claim.

Execution note: the initial CPU replay attempts (`strong-smooth-training-canary-r1-replay` and `r2-replay`) stopped before producing valid replay artifacts: first the target saw `float32` rows; then the seven-component probabilities were normalized in `float32` and failed the strict sum-to-one check. The producer now constructs JSON proposal arrays as `float64` before validation; the retry uses a fresh r3 root. The first HMC attempt stopped before tuning because its loader included the outer state schema in a semantic hash that the canary writer intentionally excludes; the loader now matches the declared `{config, selected_update, variables}` payload and the retry uses a fresh HMC root.

## Post-HMC capacity repair

The 200-update `(64,64)`, three-stage candidate was valid training-route evidence but failed all corrected-HMC tuning candidates. `L=3` and `L=5` had finite traces but maximum modern R-hat `1.782` and `1.114`; `L=10,15,20` also produced nonfinite proposal/log-accept traces, and `L=25` had no viable ladder step. This rejects that candidate, not the weighted objective.

The planned repair is a `(128,128)`, six-stage, 10,000-update GPU/XLA arm, inherited only as a capacity hypothesis from the three-mode repair. It trains from a 1,048,576-row CPU replay cloud, selects checkpoints on a disjoint 65,536-row cloud, and evaluates the frozen selection on an untouched 65,536-row audit cloud. Optimizer, selection, and audit rows use distinct stateless seeds. A finite post-update selected checkpoint and finite audit nominate exactly one fresh corrected-HMC retuning run; loss alone remains non-promotional. A failed audit or a second corrected-HMC rejection stops the strong-smooth capacity rung and triggers cross-target continuation to mild rather than further unplanned scaling.

Skeptical audit: `PASS_FOR_ONE_CAPACITY_REPAIR`. The repair addresses the observed transport-conditioning failure, removes the canary's training/selection leakage, preserves the source target, proposal, affine chart, GPU/XLA/float64 route, HMC gates, and total campaign direction, and has an explicit one-attempt stop condition.

### Learning-rate screen decision

| Arm | Selection NLL | Untouched audit NLL | Audit ESS fraction | Audit max weight | Clipped updates / 200 |
|---|---:|---:|---:|---:|---:|
| `(128,128)`, 6 stages, `1e-3` | 16.5446 | 16.5139 | 0.36196 | 0.001258 | 32 |
| `(128,128)`, 6 stages, `3e-4` | 16.9017 | 16.8780 | 0.36196 | 0.001258 | 76 |

Both arms were finite and XLA compiled on the same disjoint replay partitions. The `1e-3` arm is frozen for the serious 10,000-update run because it is descriptively lower on both NLL partitions and clipped fewer updates. This is a target-specific warm-start decision, not a statistically supported ranking or a repository default promotion.

### Serious capacity result

The `(128,128)`, six-stage, 10,000-update GPU/XLA arm completed in `912.3 s` and selected update 8000 by minimum disjoint selection NLL. Selection NLL fell from `186.376` at initialization to `14.7479`; the untouched audit NLL was `14.7468`, with audit importance ESS fraction `0.36196` and max normalized weight `0.001258`. All artifacts and transport variables are finite and their recorded hashes verify.

Gradient clipping occurred on `7,737/10,000` updates. This is an explanatory optimization risk and means loss behavior alone is not convergence or transport-validity evidence. Under the predeclared one-attempt rule, the finite selected checkpoint nominates exactly one fresh corrected-HMC retuning run. If that run rejects all candidates, stop the strong-smooth capacity rung and continue to mild rather than silently adding another optimizer/capacity repair.
