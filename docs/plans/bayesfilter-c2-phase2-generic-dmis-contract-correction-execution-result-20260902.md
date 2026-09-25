# C2 Phase 2 DMIS Contract-Correction Execution Result

Date: 2026-09-02  
Correction plan: `bayesfilter-c2-phase2-generic-dmis-contract-correction-plan-20260902.md`  
Correction review: `bayesfilter-c2-phase2-generic-dmis-contract-correction-plan-review-20260902.md`  
Parent plan: `bayesfilter-c2-phase2-generic-dmis-recursive-repair-plan-20260902.md`  
Status: `PLAIN_DMIS_PRECISION_VETO; RECURSIVE_STAGE_NOT_EXECUTED`

## Executive result

The parent plan declared the plain complete-DMIS log-normalizer as the
precision estimator. A prior driver version used its `log_normalizer` alias,
which is the squared-TT control-variate estimate, for calibration and the
precision boolean. Attempts 02--08 are preserved, but they are not
contract-faithful executions of that primary screen. Attempt09 was run after
the source correction and explicitly records
`plain_complete_dmis_log_normalizer` as the estimator.

Attempt09 passes the finite-program validity checks but fails the declared
`0.00125`-nat 95% cross-scramble half-width screen. At 32,768 rows per
component, the plain-DMIS half-widths are `0.00359` (t=2), `0.000791` (t=3),
and `0.0870` (t=4); at least t=2 and t=4 therefore veto the candidate. The
largest half-width on the full ladder is `0.4101` nats at t=4 with 8,192 rows.
The control-variate estimate is retained as a diagnostic, not as a replacement
for the declared criterion.

This is evidence that the tested Gaussian-plus-Student proposal bank does not
meet the frozen precision requirement at this scope. It is not evidence that
the importance identity, target convention, TT basis, or recursive moment map
is mathematically wrong.

## Contract and provenance audit

The target in every run is the finite carried-density target

\[
  \gamma_t(u)=\eta_{2n}(u)E_t(u),
\]

where `E_t` is the exact branch-summed snapshot evaluator. It is not the true
C2 marginal likelihood. The complete mixture contains both proposal families,
with deterministic row masses `(1-alpha)/N` and `alpha/N`.

The source correction made two changes only:

1. alpha calibration now minimizes the plain-DMIS log half-width; and
2. `precision_pass` now reads `plain_log_normalizer['half_width_95']`.

The target assembly, proposal densities, rows, seeds, snapshots, row counts,
control variate, and tangent calculation were unchanged. Attempt09 used source
SHA-256
`ebc36e978d737ce6dbc88df14903da18ccee0d41fc5c9e2452364456bba2c147`.

Attempt08 was launched before this corrected source was present and used the
old source hash `b3bde906ffb7e20ee7ab8d2847ee39306803f6763b37effd4a6dda8505d0970c`.
It remains a duplicate diagnostic and is excluded from the contract-faithful
decision.

## Corrective command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
MPLCONFIGDIR=/tmp/mpl-c2-phase2-dmis-20260902-corrected2 \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_phase2_generic_dmis_repair_20260902.py \
  --output-root docs/benchmarks/artifacts/c2_phase2_generic_dmis_repair_20260902/attempt09 \
  --row-counts 8192,16384,32768 --scrambles 8 --calibration-scrambles 2
```

The run used float64 TensorFlow/XLA, TF32 enabled, two visible GPUs, and
verified memory growth before logical-device creation. Wall time was about 23
seconds. The fresh output directory is
`docs/benchmarks/artifacts/c2_phase2_generic_dmis_repair_20260902/attempt09/`.

## Validity checks

All 72 claim records (three row counts, eight scrambles, three captured times)
are finite and have valid support and complete-mixture closure. The snapshot
fingerprints and serialized tensor hashes agree with the preserved source.
The one-component convention check agrees with direct reference means to
machine precision; the largest independent row-wise target difference is
`8.882e-16`. The maximum checked frozen analytical tangent error is
`1.857e-11`, below the `2e-6` tolerance. The GPU placement probe ran on
`GPU:0`, and both physical devices reported memory growth enabled.

The focused CPU-hidden regression suite, including the executable generic-DMIS
call-chain check, passed:

```text
27 passed, 2 warnings
```

The warnings are TensorFlow Probability deprecation warnings. Python
byte-compilation and `git diff --check` also pass.

## Numerical results

The table reports the plain-DMIS estimator used for the gate and the squared-TT
control-variate (CV) diagnostic. Values are means and 95% half-widths over the
eight claim scrambles; they are descriptive, not a ranking of proposal
families.

| rows/component | t | plain DMIS log Z | CV log Z | ESS fraction |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 2 | -1.769565 +/- 0.001493 | -1.759617 +/- 0.0322 | 0.673805 |
| 8,192 | 3 | -4.013104 +/- 0.001038 | -2.543562 +/- 0.1040 | 0.666751 |
| 8,192 | 4 | -8.379732 +/- 0.410114 | -0.939807 +/- 0.0560 | 0.002067 |
| 16,384 | 2 | -1.769491 +/- 0.001065 | -1.748952 +/- 0.0149 | 0.672120 |
| 16,384 | 3 | -4.012955 +/- 0.001182 | -2.528405 +/- 0.0491 | 0.663641 |
| 16,384 | 4 | -8.370134 +/- 0.262411 | -0.939497 +/- 0.0327 | 0.000924 |
| 32,768 | 2 | -1.768476 +/- 0.003593 | -1.748655 +/- 0.0138 | 0.605990 |
| 32,768 | 3 | -4.012660 +/- 0.000791 | -2.595494 +/- 0.1861 | 0.658826 |
| 32,768 | 4 | -8.341742 +/- 0.087044 | -0.945291 +/- 0.0265 | 0.000477 |

The calibration selected `alpha=0.25` using the plain-DMIS statistic on two
calibration scrambles. This is a frozen selection, not evidence that 0.25 is a
general default. At t=4, both the standard-normal and Student-only arms have
large uncertainty, and the target ESS fraction is below `0.0021` throughout
the ladder. These are explanatory overlap diagnostics; they do not alter the
primary veto.

## Decision and inference status

| Question | Status | Evidence | Consequence |
| --- | --- | --- | --- |
| Was the corrected estimator actually used? | yes | Result metadata and source hash identify plain complete-DMIS | Attempt09 is the valid contract run |
| Did the target convention or mixture closure fail? | no | Machine-precision convention check; all 72 records valid | No convention fix is indicated |
| Did the plain-DMIS precision screen pass? | no | t=2/t=4 half-widths exceed `0.00125` | Candidate veto at this scope |
| Did the CV remove all integration uncertainty? | no | CV half-width reaches `0.1861` at t=3 | CV remains diagnostic only |
| Did recursive UKF/lagged-map feedback improve Phase 2? | not tested | No executable retained-moment/map callable and the precision gate failed | No recursive success/failure conclusion |
| Is proposal-family ranking statistically supported? | no | Eight scrambles are a bounded pilot without a paired ranking model | Differences are descriptive only |
| Is a production/default change justified? | no | Precision and recursive gates remain open | Keep this route diagnostic-only |

## What is and is not invalidated

The run does not invalidate the finite-mixture importance identity, the
`gamma=eta*E` target convention, the model-independent TensorFlow kernel, or
the frozen analytical tangent. It does show that this particular fixed
Gaussian-plus-Student bank does not cover the frozen target well enough for the
declared precision, especially at t=4. The data do not distinguish tail-family
undercoverage from coordinate-map or finite-row effects.

The recursive stage was intentionally not relabeled or run with external GH9
hints. A future experiment must expose actual TensorFlow moment contractions
and map construction, then repeat the same target/support/tangent checks on a
tail-aware proposal or coordinate map. That future experiment needs its own
predeclared contract; this result cannot select its basis or tune its proposal.

## Artifact integrity

The attempt09 manifest records the command, environment, GPU policy, snapshots,
and output hashes. After launch, it was reconciled to add the correction-plan
links, the explicit precision-estimator field, and a post-run workspace digest;
result and Markdown hashes remain valid. The following checks pass:

```text
result.json: True
result.md: True
correction plan and review hashes: True
result script hash: True
```

The MathDevMCP bounded derivation audit used for the surrounding finite-program
algebra returned `inconclusive/source_label_missing` rather than a mismatch;
that parser limitation is not a proof certificate. No claim of formal Lean
certification is made for this numerical candidate.

## Next justified action

Do not proceed to the C2 recursive comparison under this contract. Write and
review a separate tail-aware proposal/map plan that distinguishes (a) target
tail coverage, (b) proposal variance, and (c) the missing recursive moment-map
callable. Retain DMIS with complete denominators as the correctness comparator,
and require the same plain-DMIS precision, support, and tangent gates before
making any recursive claim.
