# LG1 finite-endpoint posterior comparisons

The question remains whether imperfect full-DCTR forces can drive valid
endpoint-corrected HMC. B/R attempt4 has completed its two planned looks and is
INCONCLUSIVE: healthy chains, insufficient precision at all frozen margins.
That is neither a discrepancy nor a continuation veto. Complete the independent
finite-target comparisons on the same untouched attempt4 observations before
considering the separately authorized fresh-data repair attempt5.

Use only the qualified endpoint `lg1_dctr_physical_fp32_chart_fp64_v1`, with the
same controls, N1024, T10, random inputs, prior and data as the preserved endpoint
qualification. This endpoint is distinct from the old unconstrained FP32
scalar. Proposal and endpoint identities stay separate. Endpoint coordinate
rounding must survive the HMC transition and archive; otherwise C cannot launch.

First compute D, independent physical-coordinate quadrature of that complete
finite endpoint, using uniform trapezoidal grids with 128,256,512,1024
intervals. The density is constant in a small one-sided neighborhood of each
physical support boundary because of the declared FP32 coordinate rounding.
Evaluate its boundary limit at the adjacent interior FP64 coordinate; include
the full support interval in quadrature. This introduces no support truncation.
Normalize the density and evaluate exactly the six R-frozen functions, using
R's centers, scales, CDF thresholds, margins and predictive functional.
The predictive functional remains the same exact conditional LG1 function;
this comparison concerns uncertainty about phi, not DCTR state prediction.

An empirical integration allowance is twice the maximum of the last two grid
changes, separately per function. Require every allowance <= one tenth of its
frozen margin and the final change <= half the penultimate change plus 1e-5.
These are convergence screens, not rigorous bounds. Every grid node must have
a finite value and original value-valid status. Preserve force-valid failures
separately; they cannot invalidate an otherwise valid endpoint. Save every
grid, values, flags, normalizers, expectations and timing. D/R intervals are
the difference plus/minus the sum of D and R empirical allowances. All inside
the frozen margins means EQUIVALENT under these empirical reference allowances;
any wholly outside means DISCREPANCY; otherwise INCONCLUSIVE. This is a
finite-likelihood approximation comparison, not sampler accuracy.

After D qualifies, C must independently tune through the public verified TF
tuner using that exact endpoint and the unchanged imperfect proposal field.
Use the same nomination grid, four dispersed starts, 250 discarded warmup and
1000 independent verification draws as A/B, with a separate seed family.
Changing the endpoint means a new tuning scope: B's selected kernel is not
admission for C. Before retained C, forecast from measured C tuning. At most
1000 and 2000 retained draws per chain, in archived 250-transition chunks,
with the same batch-MCSE, R-hat, ESS and two-look equivalence rules. Test C/D
and C/R separately. Use Bonferroni over both six-function comparisons and
both looks, so C does not acquire twice the familywise error allowance.

The cheap adversaries remain the master program's frozen exact-reference,
zero-force/random-walk and simple approximate-posterior checks. They have not
been evaluated here: heuristic dominance is NOT_RUN and efficiency/default
promotion remains vetoed. No accuracy result here establishes superiority.

Default audit: the quadrature grid is a bounded numerical hypothesis, tested
by its refinement ladder; inherited R function definitions ensure comparable
quantities; C nomination settings are warm starts re-adapted on C, not transferred
tuning. The model/prior and full DCTR protection settings have already passed
their own checks and are frozen. No posterior error is a tuning objective.
Potential misleading success is agreement from compensating sampler/endpoint
errors; the separate C/D,D/R,C/R comparisons prevent that inference. A grid
failure could be numerical roughness or inadequate refinement, not evidence
against surrogate HMC. A mixing failure is a C tuning repair trigger, not
authority to change the endpoint after seeing agreement.

Budget: D at most900 GPU seconds and1200 CPU-worker seconds; C tuning at most
2400 GPU seconds with its actual CPU usage charged; C retention at most3600
GPU seconds and4800 CPU seconds, subject to a measured forecast and cumulative
LG1 allocation of10800 GPU/9000 CPU seconds. These are maxima, not added
allocations. The master checks remaining budget before each launch. One D
preparation and one C inference on attempt4; local harness repairs are charged.
No A/B extra look and no extension of frozen margins. New versioned output
roots under `docs/plans/artifacts/surrogate-hmc-all-models-*` preserve failures.
If the remaining allocation cannot fund a qualified comparison, record
INCONCLUSIVE_BUDGET and return to the remaining model queue.

Skeptical audit: exact endpoint binding, physical/chart measure, immutable
functions and separate tuning are explicit; GPU memory growth and XLA remain
required. Unresolved references, numerical endpoint failures, mutable bindings,
missing archive diagnostics or a forecast over budget block the affected
launch. Candidate disagreement alone does not block the other comparisons.

## Execution continuation, 2026-09-20

D qualified on all four grids. D/R fails the median-CDF margin; this is a
finite-likelihood accuracy veto, not a C/D continuation veto. The flag-aware
public tuner passed its fresh Gaussian GPU qualification, including exact and
imperfect rotational forces, archived rounding flags, and negative binding
checks (`surrogate-hmc-all-models-20260920-tuner-repair-07`). CPU suites passed
47 core/trace tests and25 master/equivalence/reference tests. Qualified endpoint
and reference source hashes remain unchanged. Skeptical re-audit found no
unresolved target, data, baseline, coordinate, diagnostic or permission issue
for separate C tuning. Tuning has a3000 CPU-second watchdog in addition to the
2400 GPU-second limit; both remain inside the cumulative LG1 allocation.

Execute with `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, trusted GPU
access, memory growth, and the master watchdog:

```bash
python scripts/run_surrogate_hmc_program.py --stage P2-C-tune --scope LG1 \
  --output docs/plans/artifacts/surrogate-hmc-all-models-20260920-lg1-C-tune-01 \
  --max-gpu-seconds 10800 --max-cpu-worker-seconds 9000 \
  --reference-root docs/plans/artifacts/surrogate-hmc-all-models-20260919-lg1-reference-master-01/P2-reference \
  --selection docs/plans/artifacts/surrogate-hmc-all-models-20260919-lg1-controls-02/P2-controls/selection.json \
  --endpoint-root docs/plans/artifacts/surrogate-hmc-all-models-20260919-lg1-finite-endpoint-01/P2-endpoint \
  --finite-reference-root docs/plans/artifacts/surrogate-hmc-all-models-20260920-lg1-finite-reference-01/P2-finite-reference
```

Retained C uses `--stage P2-C-sample`, its fresh `--tuning-root`, the same frozen
inputs, a new output directory, and a measured forecast checked against both
the job watchdogs and remaining cumulative allocation before launch.

## Retained C launch audit

C tuning01 completed in1823.94 GPU wall seconds /1750.37 child CPU seconds.
All fresh verification checks passed, including ordinary R-hat admission.
The selected kernel has L=1, epsilon=1.4319353352 and chart covariance
0.9420811274. Its own measured forecast is4738.34 GPU seconds and5686.01
CPU-worker seconds for the complete planned retained schedule. Neither
nomination acceptance nor this tuning pass is a posterior-equivalence result.

The master reallocates45 minutes of unused P0/P1 CPU allowance to LG1, within
the original15-hour total. Retained C uses cumulative ceilings10800 GPU
seconds /11700 CPU-worker seconds and job limits4800 /5800 seconds. The full
forecast and watchdog reservation must fit after charging the latest tests.
No margin, posterior function, retained look, seed policy, endpoint, force,
data, hardware or total campaign limit changes. Attempt4 remains the active
inference attempt; attempt5 is not used by this continuation.

Skeptical audit: C/D and C/R use different reference expectations but the
same six frozen functions and uncertainty method, with adjustment over both
comparisons. The2000-draw cap can leave the CDF margins unresolved even for
healthy chains; this must be reported as INCONCLUSIVE and cannot justify a
third look. D/R's discrepancy remains an approximation result and cannot
cancel C/D. The master dependency repair will receive focused regression
checks before launch. The qualified worker, endpoint, force and tuner sources
remain frozen.
