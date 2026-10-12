# D4R4b preconfirmation execution record

Plan: [D4R4b preconfirmation](bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md).
Date: 12 October 2026. Status: first discarded-window rung prepared; no
posterior or tuning-membership authority follows from this cost diagnostic.

## Prelaunch review and source

The skeptical plan audit checked the archived-reference baseline, the
independence of reference compatibility from cost pricing, the unproven
64/16-to-256/64 window hypothesis, complete four-target denominator, explicit
stop conditions, GPU contention, and the risk that a healthy local path might
still miss posterior mass. It found one code defect: caller-supplied subset
indices could be distinct yet identify the wrong targets. The consumer now
requires exact original target positions. The four-model isolated CPU-hidden-
GPU suite passed seven tests in 453.26 seconds from staged tree
`59cb47ad1bb47288b33afcec81e17525d3a525cd`; the focused index check
passed after correction in 2.03 seconds in the working tree and 2.77 seconds
from the committed archive. These are mechanics checks, not sampler evidence.

The first-rung source is immutable archive `N5/generic-adaptation/source-163d24b34/`
from commit `163d24b34b1350f3db0681867ea4d03c580f87c2`, tree
`e18a3988442bff81d406c8bf6ef0985ca6481513`, limited to committed
`bayesfilter`, `scripts` and `tests`. Its compiled Sylvester operator has
SHA-256 `2feb93135f2359e60aca446f52f8468b104199b4d398d4fe6057030df20f23d6`.
The archive's `.validation-source.json` binds these identities. The source
contains both the consumer and the tested SIR status repair.

GPU 1 was occupied by a separate NeuTra process (PID 1990789), using about
454 MiB at the capacity check. The trusted GPU probe selected otherwise idle
host GPU 0, also an RTX 4080 SUPER, and verified TensorFlow 2.20.0 device
placement and memory growth before initialization at 2026-10-11 18:33:46 UTC.
This cost-only fallback preserves the hardware class, source, settings and
allocation. It cannot support a cross-device speed ranking.

## First rung command and bounds

Run from `N5/generic-adaptation/source-163d24b34/`:

```bash
env CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/price_mams_adaptation_full_models.py \
  --geometry-root /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/window-256 \
  --targets NL-PP-SGQF NL-PP-UKF NL-SIR NL-STR --seed 5400 \
  --fixed-window-transitions 256 --max-autocorrelation-lag 64 \
  --plan docs/plans/bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md \
  --max-wall-seconds 7200
```

The output path is fresh. Its sibling `window-256.log` preserves console
output. The supervisor cap is 7,200 seconds; the full phase separately reserves
at most 3,600 local-check seconds and a conditional second 7,200-second GPU
rung. Charge elapsed supervisor time once, preserve failures and unstarted
targets, and release unused reserve at terminal. No R-hat, ESS or MCSE is used
to choose a rung or a tuning candidate. The separate reference-compatibility
bridge must pass before any posterior comparison.

The first rung started at 2026-10-11 23:58:25 UTC after tool approval, parent
PID 2058439 and worker PID 2058441, session `86753`. Its manifest confirms
GPU 0, XLA, verified memory growth and the intended commit. At approximately
14 minutes, PP-SGQF remained in discarded adaptation with advancing completed
work; no target failure had been reported. The earlier readiness probe and
actual start timestamps are distinct and preserved.

## B0 reference audit preparation

The [source audit](bayesfilter-mams-reference-source-audit-2026-10-12.md)
records the exact nine source-difference arguments and the distinction between
an independent scalar reference and the old native-parity binary. The new
diagnostic reader reaggregates all eight scramble rows, binds current and
archived source, checks data and quantity charts, and keeps old references
unchanged. Thirty-one corruption/identity tests passed in 0.21 seconds; import
and compile checks passed. These tests use synthetic evidence and establish
reader mechanics only. The real four-archive summary/source precheck passed.

The CPU parity command is bounded to 2700 wall seconds, charged within the
3600-second local allowance (453.26 seconds already used by the isolated
four-model suite). This ceiling is an allocation choice, not a predicted
runtime. Two CPU threads inherit the original reference-driver diagnostic
setting and are not used for performance comparisons. Execute the new
diagnostic driver from its next scoped immutable commit, while passing
`source-163d24b34` as the exact current target source. Preserve four original
reference points and four starts using the first-rung seeds for every target.
The output root is `N5/generic-adaptation/preconfirmation-01/reference-01/`;
the sibling log preserves failures and progress. No reference compatibility
is claimed until the current native numerical checks finish.
