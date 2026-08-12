# SSL-LSTM q=20 multimodal repair reset memo (2026-08-10)

## Current state

The old seed-B NeuTra/fixed-HMC archive remains globally invalid: ordinary reverse
KL omitted the negative mode, all starts were positive, and the selected kernel was
locally unusable in the negative transformed region.

Stage 1 implemented and validated an exact TFP replica-exchange fixed-HMC diagnostic:

- eight focused tests passed;
- equal and unequal analytic Gaussian mixtures were recovered from all-positive
  starts under XLA;
- ordinary HMC stayed trapped;
- accepted swap matrices and initial replica identities are fully archived; and
- an exact SSL transformed-target four-step canary passed finite/status mechanics in
  `503.928 s`.

The SSL canary had accepted swaps but zero cold sign transitions and zero complete
round trips.  It is not a posterior sample and does not resolve weights.

Terminal result:

`docs/plans/bayesfilter-ssl-lstm-q20-multimodal-repair-result-2026-08-10.md`

Artifacts:

`docs/plans/artifacts/ssl-lstm-q20-multimodal-repair-2026-08-10/r1/`

## Critical next correction

Do not run a long tempering ladder in the failed NeuTra `z` coordinates.  The source
MAPs are separated by about 1.28 in the original four parameters, while NeuTra moved
the transformed stationary regions 23.707 units apart and produced severe
region-dependent curvature.  The independent global authority should operate on the
original batch-native target.  NeuTra repair comes after weighted multimodal coverage
exists.

## Next plan

1. Derive or calibrate one physical-coordinate mass/chart and step ladder using both
   known source modes.  A scalar identity-mass step is only a baseline because source
   curvature eigenvalues span roughly `0.045` to `64`.
2. Run the smallest physical-coordinate replica-exchange canary with both modes and
   enough steps to observe replica travel; freeze settings before a validation run.
3. Require repeated cold-hot-cold round trips, hot-chain basin forgetting, finite
   status, and both cold-region transitions.  Raw occupancy remains explanatory.
4. Run independent AIS on a normalized multimodal proposal constructed from the two
   local physical-mode approximations.  Validate first on analytic unequal-scale and
   unequal-weight mixtures.  Report weight ESS, maximum normalized weight, repeated
   independent batches, schedule sensitivity, and uncertainty.
5. Only if transition and weighted evidence agree, issue a weighted posterior archive
   and repair/retrain NeuTra using global coverage.  Then run sequential HMC and the
   existing posterior-predictive distribution diagnostic.

## Provenance warning

Historical checkpoint reconstruction requires:

`BAYESFILTER_CODE_ROOT=/tmp/BayesFilter-seed-b-root-cause-historical`

at clean commit `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`.  The live shared
worktree now has trainer-schema drift and correctly fails restore.  Load new
lane-specific helper code by exact path, as the stage-1 runner does.  Historical
identity remains `historical_identity_exact=false`; do not claim exact reconstruction
of the original dirty August 7 executable.

## Resilient launch procedure

The interactive terminal stream disconnected during the accepted canary's long
target evaluation.  The process completed, but future runs must not depend on an
open foreground stream.  Use one unique user-service unit per attempt with:

- `WorkingDirectory=/home/ubuntu/python/BayesFilter`;
- `Environment=CUDA_VISIBLE_DEVICES=-1` before TensorFlow import;
- `Environment=BAYESFILTER_CODE_ROOT=/tmp/BayesFilter-seed-b-root-cause-historical`
  when restoring the historical checkpoint;
- an absolute command and versioned `--output-root`;
- `TimeoutStartSec` no larger than the plan cap;
- stdout/stderr redirected to the versioned artifact root; and
- a runner-side wall/attempt cap and overwrite refusal.

Monitor with short `systemctl --user show`, `systemctl --user status`, `tail`, and
artifact-manifest reads.  A disconnected client is then only a monitoring event,
not an experiment interruption.  Do not infer completion from a missing stream;
require the structured terminal JSON and receipt hashes.
