# Zhao-Cui Austria SIR Lane-B T1 Result

Date: 2026-07-31

Status: `PASS_NEW_FIXED_VARIANT_T1_VALUE_BASELINE`

Baseline identity:
`zhao_cui_austria_sir_fixed_variant_training_base_v1`

Selected artifact identity:
`e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`

## Verdict

The newly named Lane-B T1 fixed-variant baseline is admitted for the coherent
latent pre-clipping Austria SIR T1 scalar. It has a sealed target and event
order, the exact normalized-reference-measure conversion including `36 log 2`,
a batch-native GPU/XLA training kernel, independently calibrated absolute
normalizer scale, deterministic artifact identity, fresh reload, bounded
memory, and an untouched value check within predeclared Monte Carlo
uncertainty.

This is not P88 reconstruction and does not establish a correct T2/T20 value,
score, HMC target, exact nonlinear likelihood theorem, source-faithful assembled
route, production readiness, or superiority over GenUT/SGQF/UKF.

## Primary Evidence

| Field | Result |
|---|---:|
| Selected arm | `p05_r4_b5_lr3e4_l1_1e9` |
| Rank / basis order / elements | `4 / 2 / 2` |
| Learning rate / L1 | `3e-4 / 1e-9` |
| Calibrated artifact value | `-31.1290512231882` |
| Untouched log-evidence estimate | `-31.131520868973524` |
| Untouched log standard error | `0.0016009228535994377` |
| Absolute value difference | `0.0024696457853252696` |
| Predeclared combined tolerance | `0.008351738395320572` |
| Claim allocator peak | `134301952` bytes |
| Memory cap | `6 GiB` |
| Fresh identity reload | passed |
| Untouched seed | `73401`, read once |

The claim artifact is
`docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/untouched-claim-01/result.json`.
Its SHA-256 is
`bb731507dd35cf45d346d6dba8d753309954f2bd8b5fee8affbe049b38405df3`.

The validation-only selection ledger is
`docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/pilot-final-02/selection-v2.json`.
Its SHA-256 is
`8821fce40877f4d3e225fe1c308696a129f10ac13cde0697785def7b836e59d4`.

## Pilot Screen

All six final arms passed hard finite-value, independent-normalizer, shape,
GPU/XLA, and memory screens. Validation RMS values were descriptive selection
metrics, not statistically supported rankings.

| Arm | Validation normalized-log-density RMS | Peak bytes | Status |
|---|---:|---:|---|
| `p01_r2_b3_lr3e4_l1_0` | `22.80600035085222` | `71791872` | viable zero-L1 comparator |
| `p02_r2_b3_lr3e4_l1_1e8` | `22.80599313832715` | `71791872` | viable |
| `p03_r4_b3_lr3e4_l1_1e9` | `22.478280656578946` | `71791872` | viable |
| `p04_r4_b3_lr3e4_l1_1e8` | `22.478280224158013` | `71791872` | viable |
| `p05_r4_b5_lr3e4_l1_1e9` | `21.43754899517971` | `171080960` | selected descriptively |
| `p06_r4_b5_lr1e4_l1_1e9` | `22.88844380202675` | `171080960` | viable |

No ranking is statistically supported. The screen establishes that all six
remain viable under their hard gates and deterministically chooses `p05` under
the frozen validation rule.

## Repairs And Negative Evidence

1. The first `p01` attempt exposed a generic random-initialization collapse:
   multiplying `0.05`-scale cores over 36 axes left `rho=tau`, near-zero
   gradients, and a `3.10e46` post-fit rescale. This was an implementation
   failure, not a candidate or research-direction rejection. A Lane-B unit
   constant path plus connected `1e-6` rank channels repaired it.
2. The first order-2 `p05` attempt dispatched a tiny FP64 Vandermonde solve to
   GPU and failed before training. CPU-precomputed setup-static basis mass and
   integral tensors repaired that backend placement without changing the basis
   or objective.
3. Fresh reload exposed JSON tuple erasure in the v1 artifact manifest. A
   separate compatibility decoder restored only the declared `("z1","z0")`
   tuple, verified every tensor and bound source hash, and reproduced the
   original identity exactly. The fitted numerical artifact was not changed.

All failed attempts remain preserved. They consumed campaign budget but did not
consume the untouched claim seed.

## Decision Table

| Field | Decision |
|---|---|
| Decision | Admit the T1 fixed-value baseline and open B3 only. |
| Primary criterion | Passed: fresh reload plus untouched value agreement under the declared uncertainty rule. |
| Veto diagnostics | Passed: target/hash/event order, measure, finite mass, identity, XLA/GPU, and 6 GiB memory cap. |
| Main uncertainty | Monte Carlo uncertainty in calibration and untouched estimates; finite TT shape remains an approximation. |
| Next justified action | Implement the T2 previous-marginal boundary and independently tie out the T1 retained marginal before any T2 fit. |
| Not concluded | No T2/T20 value, score, HMC, exact nonlinear likelihood theorem, source-faithful assembled route, statistical superiority, or production readiness. |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for T1. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Per-arm validation RMS and runtime/memory differences. |
| Default readiness | Not established. |
| Next evidence needed | B3 independent T1 marginal tie-out, then a separate scope-specific T2 tuning plan. |

## Post-Run Red Team

The strongest alternative explanation is that a scale-correct but weak TT shape
can pass the T1 value gate because the absolute normalizer is calibrated from
independent Monte Carlo. That is why validation shape remains a hard pilot
screen and why T2 must independently verify the retained marginal rather than
assuming a correct filtering density from the scalar pass. A failed B3 marginal
tie-out would invalidate T2 continuation even though this T1 scalar result
remains valid for its declared finite program.

## B3 Previous-Marginal Boundary

Status: `PASS_T2_PREVIOUS_MARGINAL_BOUNDARY`

The selected T1 artifact was reloaded and its retained `z1` density was
evaluated through two independent paired-core contractions at the cut after
axis 17. Both routes applied the exact physical conversion

`log p(z1)=log p_nu(u1)-18 log 2+log|du1/dr1|-log|det L11|`.

At 64 deterministic sealed T2 points:

| Diagnostic | Result | Tolerance |
|---|---:|---:|
| Maximum API/independent log-marginal residual | `1.4210854715202004e-14` | `2e-12` |
| Maximum recomposed T2 log-target residual | `2.842170943040401e-14` | `2e-12` |
| Independent cut/direct normalizer residual | `1.7041923427996153e-13` | `2e-12` |

The boundary artifact is
`docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/t2-boundary-01/result.json`,
SHA-256
`ae116ab1b2938d695b19017075bbdbbefe4b63cb95a6442b8ff2b3495b0d73ce`.

This opens a new scope-specific T2 tuning plan. It is not a T2 fit, value,
score, T20, HMC, or production result.
