# M13 arithmetic precision and timing diagnostic

The owner asks whether FP32 arithmetic error is small relative to the observed
N1008-to-N2016 change, and requests FP32/FP64 timings at N1008, N2016 and N4032.
This bounded measurement extends the completed particle ladder; it does not
restart surrogate-HMC sampling. The candidate is the current shared
`covariance_guided_defensive_mixture_finite_v1` implementation, including the
already-tested reset recentering and exact tiled transport.

## Evidence contract and research intent

- Compare FP64/TF32-off, FP32/TF32-off, and FP32/TF32-on using identical
  observations, initial particles, innovations, uniforms, parameter values and
  mixture weights. Inputs are generated once by the existing FP64 generator,
  rounded to FP32, then cast to the requested arithmetic dtype. Thus different
  RNG implementations cannot masquerade as precision error.
- Freeze each N's saved FP64-tuned controls and mixture weights. These are a
  matched diagnostic comparator, not scope-tuned FP32 controls. N4032 has 8 flow
  and 40 reset steps; N1008/N2016 have 16 and 80. Cross-N differences therefore
  describe the previously tuned configurations, not an N-only ablation.
- Use M13 T20, state dimension 4, observation dimension 2, parameter dimension
  2; data seeds 1704/18020/18021; truth and the existing difficult parameter
  point; eight design seeds per case, exactly as in the particle ladder.
- Primary engineering quantities are absolute log-likelihood difference and
  Euclidean score difference from the matched FP64 computation. Compare their
  casewise RMS to the RMS change between the N1008/N2016 FP64 configurations.
  A ratio below 0.1 is the predeclared materiality screen, a diagnostic
  convenience rather than a theorem or an HMC correctness criterion.
- Also compare to the earlier FP64 score spread and likelihood SD, and report
  per-case means/maxima. Direct precision differences require no likelihood
  oracle; particle-count changes do not by themselves imply improvement.
- Finite value/score, both directional primal values agreeing, all 20 steps,
  and existing reset/quadrature/rank guards are hard candidate vetoes. Do not
  time an early rejection as a successful full evaluation. A failed precision
  arm does not veto evaluation of the other arms. Do not relax guards.
- Runtime is synchronized wall time for one likelihood plus both analytical
  score components, after two warm-up calls. Report cold compile separately;
  median of 48 complete evaluations is descriptive. Same RTX4080 SUPER UUID,
  one worker at a time, memory growth verified, GPU/XLA throughout. TF32-off
  is a deliberate precision exception, not a new production default.
- Promotion is limited to the arithmetic-materiality screen in these cases.
  No default change, FP32 tuning admission, posterior convergence, HMC accuracy,
  all-model claim or statistically supported method ranking follows.

The main expected failure is TF32 or FP32 accumulated error triggering reset
guards or materially perturbing scores. The first truth/difficult checks expose
this before spending on the full set. If either preflight is invalid, retain
that arm's failures and classify it as failed preflight; do not report a complete
timing for it. Run all other precision/count combinations. Implementation,
input-parity, missing-source, device, or artifact corruption is a continuation
veto requiring a localized repair within the budget. An isolated candidate
failure does not reject the method or the use of FP32 without TF32.

## Assumption audit, sanity checks and skeptical review

| Choice | Provenance and role | Failure mode and earliest check |
|---|---|---|
| FP32/TF32-on | Repository production direction; hypothesis here | Reduced matmul precision; inspect preflight reset error and validity |
| FP64 | Earlier controlled ladder baseline | FP64 is not an exact filtering oracle; claim only arithmetic comparison |
| Controls and beta | Saved per-N offline tuning; frozen comparator | Not tuned for FP32; no FP32 admission claim |
| Rounded common inputs | Existing paired-design generator | Dtype-dependent fixture creation; generate with FP64 specification in every arm and hash input bytes |
| Eight designs/six cases | Existing ladder evidence | Limited generalization/tail resolution; retain separate cases and maxima |
| 10% materiality screen | Explicit practical hypothesis | Small N change makes ratio unstable; also report numerator/denominator and design spread |
| XLA and chunk sizes | Existing tested shared kernel; K1008/2016/2016 | Retracing or accidental alternate route; record graph traces and source hashes |
| Reset controls/correction off | Frozen earlier measured per-N candidates | Inherited protection behavior; preserve all guards, report residuals, do not tune on these cases |

The cheap adversaries for filtering quality are the same-particle bootstrap,
prior-only proposal, and global Gaussian guide already constructed in the
particle ladder. They remain in its conditional six-case evidence. They are
not re-run here because arithmetic parity is the question, and these results
cannot promote filtering quality over those adversaries. The direct FP64
computation is the arithmetic comparator. The original Gaussian/prior heuristic
losses remain unresolved.

Skeptical audit: matched precision pairs answer this question only if random
inputs, selected controls, theta and beta agree. Source and input hashes enforce
that; saved FP64 rows provide an additional replay comparison (tiny changes
from rounding theta/beta are explicitly reported). Runtime excludes compile
and early exits. No oracle uncertainty is confused with arithmetic error, and
no FP64 controls are relabelled as FP32 tuning. The plan passes this audit.

## Execution and budget

One serial GPU campaign, maximum 5,400 wall seconds and nine workers, plus up
to two localized infrastructure retries within the same wall budget. No
algorithm repair or tuning sweep is implicit. Initial output is
`docs/plans/artifacts/ledh-range-bearing-precision-20261010-01`; never overwrite
an earlier campaign. Complete command, source/data/control/input hashes, seeds,
memory policy, device, graph trace count, cold/warm times, all rows, failures,
and decisions are preserved there. Result note is
`docs/plans/ledh-range-bearing-precision-results-2026-10-10.md`.

Exact launch from the repository, with escalated GPU permissions:

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/range_bearing_precision_diagnostic.py run --output docs/plans/artifacts/ledh-range-bearing-precision-20261010-01 --gpu GPU-68251639-fe82-8f81-3ccc-2953c32e805b
```

The same entry point's `status` and `report` commands use only Python standard
library reporting. This is diagnostic code, not a new numerical authority.
