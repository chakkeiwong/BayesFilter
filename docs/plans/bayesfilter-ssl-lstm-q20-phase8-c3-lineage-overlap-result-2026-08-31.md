# Phase 8 C3 lineage and temperature-overlap result

Date: 2026-08-31  
Status: `PASS_C3_LINEAGE_OVERLAP_WITH_DIVERSITY_REPAIR_NO_ARM_NOMINATION`

Subplan:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-lineage-overlap-subplan-2026-08-30.md`

## Result

The C3A GPU pilot completed all eight declared rows (two architectures, two
lineage arms, and two fresh roots) in 1,695.482 seconds. Every row passed the
finite/status-valid training screen, immutable checkpoint replay, proper-bridge
overlap evaluation, and learned-map reliability screen. The strict backend was
`tensorflow_eigh_strict`; the frozen q=20 target signature was unchanged; one
GPU was used with memory growth configured before device initialization; and
the largest row allocator peak was 1,410,923,264 bytes.

The required covariance and sign-occupancy summaries were absent from the first
runner. The artifact-only repair reconstructed all sixteen beta-one charts from
their hashed checkpoints and evaluated fresh, disjoint 256-row Gaussian banks.
It completed in 4.992 seconds; all checkpoint contexts, finite map outputs,
sign partitions, and allocator caps passed. The repair did not retrain a map or
evaluate target samples.

Primary artifacts:

- C3A manifest:
  `docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3-lineage-overlap/attempt-01/run_manifest.json`
- diversity repair manifest:
  `docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3-lineage-overlap/diversity-repair-2026-08-31/attempt-01/run_manifest.json`
- repair subplan:
  `docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-diversity-repair-subplan-2026-08-31.md`

## Evidence table

| Architecture | Arm | Root | Adjacent acceptance `(0-.5, .5-1)` | beta-one mean distance | covariance Frobenius distance | sign-occupancy L2 distance |
|---|---|---:|---:|---:|---:|---:|
| compact-high | pure | 0 | `(0.3020, 0.3214)` | 1.0407 | 5.5402 | 0.0497 |
| compact-high | pure | 1 | `(0.4451, 0.3385)` | 1.9248 | 4.2696 | 0.0110 |
| compact-high | branching | 0 | `(0.2350, 0.2997)` | 1.0177 | 6.8121 | 0.0387 |
| compact-high | branching | 1 | `(0.2742, 0.2597)` | 0.7527 | 6.4779 | 0.0829 |
| compact-low | pure | 0 | `(0.4014, 0.4627)` | 0.4857 | 10.1615 | 0.0663 |
| compact-low | pure | 1 | `(0.3777, 0.3817)` | 0.8743 | 9.6596 | 0.1160 |
| compact-low | branching | 0 | `(0.4168, 0.3807)` | 1.0566 | 3.6356 | 0.0276 |
| compact-low | branching | 1 | `(0.3034, 0.3579)` | 0.9812 | 6.1260 | 0.0331 |

The swap values are proper-bridge diagnostics from 64-chain banks. They are
positive finite overlap screens, not estimates of mixing or posterior mass.
Every sign-occupancy vector had zero boundary fraction and positive/negative
fractions close to one half. No chart showed evidence of a sign-separated
posterior mode through this diagnostic.

## Decision

The predeclared nomination rule required a branching arm to be hard-valid on
both roots, have nonzero adjacent overlap, and show a larger cross-lineage
distance than pure continuation. Hard validity and nonzero overlap passed, but
the distance condition did not hold consistently: branching mean distance was
smaller for both compact-high roots and larger for both compact-low roots;
covariance distance changed in the opposite direction between architectures;
sign-occupancy distance was not consistently larger. With two roots and short
training, these are descriptive contrasts only. Neither arm is promoted and no
architecture is ranked.

The result does not establish that tempering or branching fails. It shows that
the L3, 16-update, K=2 calibration is insufficient to nominate a branching
policy. The C2 pullback-score residual problem remains unchanged, so C3 does
not provide whitening evidence.

| Decision | Primary criterion | Hard-veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close C3A hard screen | 8/8 rows and repair rows complete | Pass | Short calibration updates | Preserve maps and run a finer ladder diagnostic | No posterior or HMC result |
| Nominate positive branching | Larger distance than pure on the declared comparison | Not met consistently | Two roots and finite-bank variability | Test L5 spacing with the same branch event | No claim that branching is invalid |
| Nominate pure continuation | Valid overlap and reproducibility | Mechanically viable, not a promotion | No statistical ranking | Carry as comparator in C3B | No default selection |
| Whitening | Pullback score residual criterion | Not tested in C3; C2 residuals remain large | Training capacity and objective geometry | Keep whitening gate closed | No whitening claim |

## Inference status

| Evidence class | Status | Interpretation |
|---|---|---|
| Hard veto screen | Pass for all C3A rows and the diversity repair | The implementation and artifact paths are valid for this bounded diagnostic |
| Statistically supported ranking | None | Two roots and short runs do not support a ranking |
| Descriptive-only differences | Acceptance, mean/covariance distance, sign occupancy | Useful for selecting the next diagnostic, not superiority evidence |
| Default-readiness | Not established | No scope-specific tuning, long chains, ESS/R-hat, or posterior validation |
| Next evidence needed | L5 adjacent-overlap and repeated-branching ablation, then a fresh review before any Phase 9 work | Phase 9 remains closed |

## Post-run red-team

The strongest alternative explanation is finite training and bank variability:
the 16 updates at each positive temperature may not have moved either chart to
its local reverse-KL optimum, and 256-row summaries have non-negligible Monte
Carlo noise. A second explanation is that the sign coordinate is not a valid
basin label for these learned charts. The result would be overturned as a
mechanics result by a checkpoint hash/context mismatch, nonfinite proper-bridge
value, failed inverse/logdet/score screen, or a memory-growth violation; none
occurred. The weakest evidence is the apparent arm contrast itself, which is
descriptive and not statistically supported.

There is an additional comparability limitation: C3A folded the arm index into
its training and diagnostic seeds. Thus its pure-versus-branching contrast was
not perfectly paired even though the target and architecture were shared. C3B
removes that confounder by using arm-neutral folded seeds for corresponding
training and fresh diagnostic banks; C3A remains useful as hard-valid lineage
evidence, but its arm differences should not be treated as a clean causal
ablation.

## Nonclaims

This result does not claim mode discovery, exhaustive mode coverage, posterior
regional masses, IID Gaussian whitening, convergence, HMC readiness, scaling
to higher dimension, architecture superiority, or statistical superiority.
