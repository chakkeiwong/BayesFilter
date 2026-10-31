# Phase18 result: the center repair passes; TF32 precision is only partly established

Completed 2026-09-23 in the isolated LEDH execution checkout, numerical base
89065bc6354801cb368d5e163ba49fd9c4372d10. The repaired shared consumer passes
the time-varying-center reproducer and ordinary FP32 controls. Its TF32
likelihood precision meets the declared criterion in51/88 conditional
comparisons. The other37 remain unresolved because their paired errors are
dominated by rare observations;22 also have intervals extending outside the
2% margin. This does not establish systematic TF32 bias, and it does not
establish acceptable precision for those37 comparisons.

The scientific target is the local six-parameter linear Gaussian model,
d=1,2,5, N64/T5 and d5/N128/T25, with frozen guides. It is not the first-study
paper model or an adaptive guide-fitting comparison. All22 comparisons using
the exact future Gaussian guide pass. The unresolved contrasts comprise16
stress-guide,12 BPF and9 observation-guide comparisons.

## Verified engineering evidence

The selected result contains67,584 likelihood evaluations:512 streams in44
dataset/guide cells in each of three isolated GPU modes. Eleven R Kalman/full
joint-Gaussian comparisons agree to5.6844e-14. All18 independent center cases
pass; maximum R discrepancy is7.55e-15 in FP64,4.76e-7 in ordinary FP32,
and8.071e-4 in TF32, within its unchanged0.003 bound. FP64 and ordinary FP32
each pass18/18 original value/score/cloud checks. TF32 passes7/18; its other11
strict path/score checks remain failed at the original tolerances.

Every worker ran the actual `make_fitted_twist_kernel` with verified GPU
placement, XLA and pre-initialization memory growth. Source/output/fixture
verification checks528/305/211 hashes. All1056 recorded traces are bitwise
neutral for the value, score and cloud. The review compares704 trace pairs;
296 have a changed discrete decision, of which266 can affect the returned
likelihood. Maximum cloud discrepancy before the first changed decision is
0.00167203; after a changed decision it reaches2.18450. The latter is not a
pure arithmetic comparison: the runs are then propagating different ancestors.

The returned analytical derivative holds guides and discrete labels fixed.
These checks do not validate the marginal model score.

## Tail evidence and interpretation

For the first T25 stress dataset, the TF32-minus-FP64 normalized-likelihood
mean difference is0.9683, with simultaneous approximate99% interval
[-3.0318,4.9685]. One stream contributes99.39% of its centered squared paired
variation. For the second it is0.1891, interval[-0.5515,0.9297], with98.88%
from one stream. The TF32 stress likelihood ratio reaches558.22 in the first
dataset. Means at this replication count cannot resolve these comparisons.

The random stress guide has larger observed squared likelihood error in90/99
conditional comparisons with BPF, observation-adapted or exact-future guides.
That is a guide-quality promotion veto, not a statistically supported ranking;
the stress guide was never a trained candidate. All existing heuristic rows
and uncertainty intervals remain in the saved summary.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Retain algebraic center repair | Independent center/control checks pass | No engineering validity veto | Tested small local scopes only | Preserve shared implementation/reproducer | All historical TF32 results repaired |
| Partial precision evidence | 51/88 pass | 37 tail vetoes;11 strict path/score failures retained | Rare changed paths and importance-weight tails | Replay influential seeds with exact original inputs | Global TF32 equivalence or bias |
| Reject stress-guide quality promotion | 90/99 observed heuristic losses | Quality veto | No ranking inference from descriptive errors | Use controls when interpreting tails | General iAPF failure |

| Inference status | Finding |
|---|---|
| Hard veto screen | Finite/GPU/XLA/growth/independent controls pass; precision/path vetoes above remain |
| Statistically supported ranking | None sought or established;51 contrasts meet the predeclared approximate mean-precision interval criterion |
| Descriptive-only differences | Means, RMSE, maxima, heuristic losses and trace frequencies |
| Default-readiness | Not established; no numerical defaults changed |
| Next evidence needed | First localize influential original seeds; any later equivalence claim needs fresh, adequately informative replications at unchanged criteria |

## Repairs, provenance and budget

`repair-log.md` preserves the failed R matrix serialization and the subsequent
FP64 decimal-representation repair. All9096 common input values pass the CPU
loader regression, and FP32 inputs are unchanged. Original FP64 attempt07 and
summary10 are preserved and superseded; selected FP64 attempt11 uses exact
promoted FP32 inputs, paired with unchanged FP32 attempt08 and TF32 attempt09.
Summary12 is authoritative. There were90,112 actual distributional evaluations
including the superseded FP64 arm;67,584 are eligible for the final comparison.

The execution command was `/home/chakwong/anaconda3/envs/tftwogpu/bin/python
docs/benchmarks/diagnose_iapf_tf32_revalidation.py --execute`; final CPU-only
inspection was `CUDA_VISIBLE_DEVICES=-1 python3
docs/benchmarks/review_iapf_tf32_revalidation.py`. Detailed commands, seeds,
TensorFlow/compiler/device information and logs are in the12 attempt manifests.
Source snapshots01–03 and `completed.json` bind the selected inputs/results.
`review-01/verification.json`, `decision.json`, `path-localization.json`,
`tail-influence.json` and its manifest preserve the terminal checks.

Phase charges:14.711536 CPU seconds and1872.783823 GPU process seconds,
including failed/superseded attempts and the initial probe allowance. Remaining
campaign balance:64909.137640 CPU seconds (18.0303h),170009.736506 GPU seconds
(47.2249h), no active reservations. This phase's12-attempt allowance is settled.

Post-run skeptical review: the strongest alternative explanation for large
paired discrepancies is ordinary threshold crossing amplified by inefficient
importance sampling. The sampled trace evidence is consistent with it but does
not inspect the largest influential streams. Fresh-compilation exact replays
of those streams are therefore the next discriminating diagnostic. Persistent
large cloud differences before any changed decision would weaken that
explanation. Passing oracle-guide comparisons cannot rescue the unresolved
arms, and the current data cannot prove absence of systematic TF32 bias.

Engineering correctness, numerical precision and scientific interpretation
remain separate. The harness is valid; candidate precision remains partially
unresolved. Continue phase19 localization under the existing campaign budget.
