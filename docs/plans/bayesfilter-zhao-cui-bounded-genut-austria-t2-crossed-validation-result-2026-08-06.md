# Zhao-Cui Bounded GenUT Austria T2 Crossed Validation Result

Date: 2026-08-06

Status: `STOPPED_NO_CALIBRATION_CANDIDATE_PASSED`

Plans:

- `docs/plans/bayesfilter-zhao-cui-bounded-genut-austria-t2-crossed-validation-plan-2026-08-06.md`
- `docs/plans/bayesfilter-zhao-cui-bounded-genut-austria-t2-joint-control-repair-plan-2026-08-06.md`

Primary calibration artifact:
`docs/benchmarks/artifacts/zhao_cui_bounded_genut_austria_t2_crossed_validation_20260806/joint-repair-attempt01/calibration.json`

Primary artifact SHA-256:
`c402a5a6e93d78925d496a0de103daf8a9e6b6d618e5020952f9b4b90e18fad1`

## Verdict

The proposed crossed validation did not run because no calibration candidate
produced a valid bounded-coordinate finite program. This is the correct stop.
All 36 joint diagonal/pairwise candidates and all 72 calibration rows left the
required `u in (-1,1)` chart. The maximum corrected absolute coordinate ranged
from `1.01880` to `1.31226`. The candidate with the smallest worst-seed value
still reached `1.03222`.

The failure persists when diagonal strength is zero. The mechanism is therefore
not just an overly strong inherited diagonal target: once a bounded-teacher
shape route is active, the current standardize/correct/affine-restore operation
in `u` does not preserve compact support. Ordinary controls in the tested grid
cannot make that operation admissible on both calibration particle seeds.

No Zhao-Cui value/score comparison and no teacher-to-particle SD ratio were
computed. The three validation teachers and six validation particle seeds were
never evaluated. Consequently, the user-proposed `0.5` adequacy criterion is
still scientifically appropriate but untested.

## Claimed and computed quantities

| Item | Verdict |
|---|---|
| Claimed candidate score | Total JVP of the same finite bounded-teacher GenUT scalar |
| Quantity reached in calibration | Invalid finite program: corrected bounded coordinate leaves `(-1,1)` before the inverse algebraic map |
| Same-program finite difference | Not computable for any candidate because the finite program correctly returned invalid/NaN |
| Exact Austria score accuracy | Not tested; no independent exact authority exists at this target |
| Teacher quantity | Independent self-normalized 128-sample Zhao-Cui bounded-coordinate moment/JVP estimator with exact TT/proposal correction |
| Physical third/fourth moments | Not claimed; they diverge for the Lane-B defensive component |
| Source classification | Bounded-teacher GenUT composition is `extension_or_invention` |

## Teacher construction

Four strict T1/T2 teacher artifacts were built deliberately CPU-only. Total
recorded build time was `266.17 s`.

| Role | T1/T2 seeds | ESS range out of 128 | Log-correction range | Manifest SHA-256 |
|---|---|---:|---|---|
| calibration | `98541,98542` | `127.719--127.729` | `[-0.1668,0.1435]` | `a18fba74dd6e4957e356045f43c8f286e8de205fc99985b1374e4800cffb2234` |
| validation 1 | `98611,98612` | `127.694--127.728` | `[-0.1762,0.1389]` | `47c0dd0cb655deb0ef4e2723ebb0dd66d65ac8f817214dd87e51b824eb5482dd` |
| validation 2 | `98621,98622` | `127.750--127.786` | `[-0.1584,0.1315]` | `ba0d1af88a7e1ad25267c3d39dd320e6b350a86ad818aa9cdbd850a356d7446e` |
| validation 3 | `98631,98632` | `127.685--127.798` | `[-0.1492,0.0986]` | `58db5d11a485fa8100345b3537be6bafe2dc6f8187758379ec9175483e7216b7` |

All four artifacts strictly reload, use eight distinct time-specific seeds, and
bind one consistent Zhao-Cui parent/issued-child identity set. High ESS supports
proposal coverage only; it does not prove that 128 samples precisely estimate
high moments.

## Calibration attempts

### Fixed-diagonal attempt

The first grid held diagonal strength at the inherited `0.2` and varied
pairwise strength `{0.005,0.01,0.02}` and cap `{1,2,4}`. Across nine candidates
and 18 rows:

- all programs were invalid because maximum corrected `|u|` was
  `1.02597--1.18580`;
- normalized physical affine mean residuals were at most `1.25e-7`;
- normalized physical affine covariance residuals were at most `4.63e-6`;
- no same-program FD was computable; and
- no validation teacher was evaluated by the finite filter.

The first launch failed without preserving candidate rows. A localized
unchanged-contract retry added only the calibration ledger and reproduced the
same veto. Artifacts:

- opaque failure: `crossed-attempt01/failure.json`;
- preserved calibration: `crossed-attempt02/calibration.json`, SHA-256
  `da7e874af97073bf95b04ec1b025ce3393da4d4fb0ee731a420cca5a4b068882`.

### Joint-control repair

The repair grid varied diagonal strength `{0,0.02,0.05,0.1}`, pairwise strength
`{0.01,0.02,0.04}`, and cap `{disabled,2,4}`. All routes used four diagonal and
four pairwise loop steps. Results:

| Diagnostic | Result |
|---|---:|
| Candidates | 36 |
| Calibration rows | 72 |
| Finite/program-valid rows | `0/72` |
| Rows with corrected `|u|<1` | `0/72` |
| Candidate with smallest worst-seed `|u|` | `diag_s0p0_pair_s0p02_cap2` |
| Smallest candidate worst-seed `|u|` | `1.03222` |
| Overall corrected `|u|` range | `1.01880--1.31226` |
| Eligible candidates | `0/36` |
| Validation rows run | 0 |

For the smallest-boundary-violation candidate, normalized covariance
restoration residual reached `0.00551`, so it also failed the `2e-4` affine
gate. Many stronger-diagonal rows restored affine moments within the gate, but
every one still failed bounded support. This confirms boundary validity as the
universal veto while also showing that near-boundary inverse conditioning can
damage affine restoration for some weak-control candidates.

## Run manifest

| Field | Value |
|---|---|
| Git commit | `6a11b689295bfb0e58de6e6d2f84918671b5a685` |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu` |
| TensorFlow | `2.19.1` |
| GPU | TensorFlow `/GPU:0`, NVIDIA GeForce RTX 4080 SUPER, compute capability 8.9 |
| Dtype / TF32 / XLA | FP32 / disabled / compiled |
| Deterministic ops | enabled |
| Memory policy | `bayesfilter.tensorflow.gpu_memory_policy.v1`, growth verified before initialization on both visible GPUs |
| Model scope | Austria SIR Lane-B latent-preclip, sealed observations, `T=2,N=1008` |
| Calibration teacher | 128 samples, seeds `98541,98542` |
| Calibration particles | `98701,98702` |
| Untouched validation teachers | three 128-sample artifacts listed above |
| Untouched validation particles | `98801..98806` |
| Fixed-diagonal launch wall observations | approximately `112 s` and `97 s` from first TensorFlow log to failure artifact |
| Joint-grid launch wall observation | approximately `312 s` from first TensorFlow log to failure artifact |
| Exact command | recorded in each `failure.json`; the joint launch invoked `run_zhao_cui_bounded_genut_austria_t2_crossed_validation.py` with the four teacher directories and output `joint-repair-attempt01` |

The stopped-run harness did not serialize an exact process timer before raising
the continuation veto. The wall observations above are derived from logged
process start and artifact timestamps and are therefore approximate. This is an
artifact-quality limitation, not a reason to reinterpret the numerical veto.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject the current bounded-`u` correction map for crossed validation | not reached | boundary veto on every calibration row | whether a support-preserving map can retain useful moment correction | derive and test a boundary-preserving update before another campaign | no rejection of Zhao-Cui teacher direction |
| Do not run validation or T20 | not reached | calibration continuation veto | teacher sensitivity remains unknown | keep validation partitions untouched | no value/score comparison or `0.5` result |
| Preserve the 128-sample teachers | teacher identity/coverage checks pass | no teacher-artifact veto | high-moment sampling precision still finite | reuse only under a newly reviewed support-preserving repair | no exact-moment claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | failed: `0/72` joint-calibration rows preserve bounded support |
| Viable candidates | none in the tested finite-program family |
| Statistically supported ranking | none; validation never ran |
| Descriptive-only differences | calibration residual objectives, boundary excesses, affine residuals, and runtimes |
| Default readiness | no |
| Next evidence needed | derivation/JVP parity for a support-preserving bounded correction, fresh calibration, then the untouched 3-teacher by 6-particle validation |

## Negative-result classification

- Teacher implementation failure: not supported. Strict identities, tensor
  hashes, ESS, and proposal-correction diagnostics pass.
- Harness failure: repaired for calibration-ledger preservation; the numerical
  veto reproduced.
- Tuning failure: the fixed-diagonal grid was under-tuned. Joint tuning repaired
  that flaw but still found no valid candidate.
- Candidate-algorithm failure: supported for the tested map. Standardizing and
  affinely restoring points in bounded `u`, followed by additive moment
  correction, does not preserve compact support.
- Zhao-Cui teacher-direction rejection: unsupported. A boundary-preserving
  composition has not been implemented or tested.

## Post-run red team

Strongest alternative explanation: a much smaller pairwise strength or fewer
active loop steps might happen to remain inside the chart. That would approach
the exact no-shape bypass, however, and would not fix the mathematical fact that
the active affine/additive map does not preserve `(-1,1)`. It would need its own
calibration and evidence that the moment correction is nontrivial.

The most direct repair is to perform correction through a support-preserving
chart, for example an update of the form

\[
 u' = \tanh\{\operatorname{atanh}(u)+\Delta(u)\},
\]

with the complete tangent of the chart and with physical mean/covariance
restoration checked afterward. This is a new algorithmic extension, not a
localized retry. Its derivation, near-boundary conditioning, residual objective,
and total JVP must be tested before another stochastic campaign.

The result would be overturned if a correctly implemented support-preserving
map passed boundary, affine, and same-program FD gates and then met the frozen
teacher-sensitivity screen on untouched validation data. The weakest evidence
remains score accuracy because no exact score comparison was reached. The
strongest evidence is the universal, directly observed compact-support veto.
