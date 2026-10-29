# Completed-checkpoint development evaluation — 8 October 2026

Question: does the descriptive validation gap between the completed author and
TF training runs persist when both best checkpoints are evaluated by the same
actual author evaluator on fresh, long, densely labelled trajectories?
The target is diagnosis of learned-model quality versus evaluation-backend
differences. The candidate is the unchanged TF-trained checkpoint; comparator
is the source-trained checkpoint from the completed development pair. No
optimizer, training objective, runtime policy or acceptance margin changes.

## Evidence contract and research intent

Execute the pinned author full-sequence forward method and quantitative NLL/RMSE
metric classes on each checkpoint. The author's NLL includes the existing 1e-8
density floor. Its RMSE is mean over trajectories of the square root of each
trajectory's mean squared state distance, with source circular means/errors.
Use the actual NLL-stage best checkpoints selected by each source controller.
A checkpoint conversion must cover every named parameter, preserve its exact
float32 value/shape and preserve every source buffer. Replay an author checkpoint
through the same conversion as a negative-control test: identical random seeds
must produce identical state outputs and metrics. Failed conversion or source
drift invalidates the comparison; repair the harness under this budget.

Primary diagnostic: per-variant NLL difference and RMSE ratio under the common
source evaluator, reporting source and TF-trained models separately.
These are explanatory development observations, never promotion criteria.
Report all outcomes and finite/probability/bandwidth validity checks. A learned
candidate with worse observations remains unqualified and triggers localization;
it does not stop the replication direction. Invalid data, changed pinned source,
bad checkpoint conversion, non-finite metrics, missing required outputs, or
budget exhaustion are continuation vetoes for this diagnostic. Original float32
comparison failures remain unresolved. Source-intrinsic failures are preserved,
not repaired. Do not infer trained-model non-inferiority, a population ranking,
TF/GPU evaluation parity, either replication, default or HMC readiness.

The accepted qualification margins remain 0.01 nat NLL and 1% RMSE, with the
previously accepted one-sided 95% limits and independent trained models as units.
This one training pair cannot satisfy that test. No interval using trajectories
as if they were independent trained models will be reported.

## Fixed choices, defaults and source anchors

- Source commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18; inspect/verify
  src/evaluations/full_sequence_evaluations/full_sequence_evaluation_base.py:
  63–121 and 730–829; src/loss_functions/loss_functions.py:15–79,
  146–192 and 290–347. Execute these functions/classes, not a local metric rewrite.
- Released bearings evaluation settings: N=25, T=150, dynamics skip 3, dense
  labels, same car/sensor distributions. Generate 256 fresh trajectories using
  the unchanged source dataset class and seed 202610081. This smaller size is a
  bounded development sample, not the source's 5000-row paper evaluation.
- Evaluation batches 64 instead of source 1024 are an explicit CPU-memory/cost
  convenience. No batch-normalized network is present; require a deterministic
  model-state immutability check and do not claim exact stochastic equivalence
  across batch shapes. Separate trajectory moments and retain the batch choice.
- Filter RNG seeds 202610082 and 202610083 are fixed before execution. Reset each
  checkpoint to the same starting seed; this does not promise identical later
  random draws when rejection sampling consumes different numbers of draws.
- CPU-only pinned PyTorch/NumPy is an independent author comparison exception.
  It cannot establish the TF production implementation. CUDA is hidden before
  import; one CPU thread; no GPU resources or environment mutation.
- Data are development-only immediately on inspection. Existing training data,
  held-out qualification data and checkpoints remain unchanged.
- Heuristic adversaries: no new superiority gate is added to source replication,
  as the accepted criteria explicitly require matching author outcomes. The exact
  source-checkpoint conversion replay is the cheap sanity adversary here.
- Conditional situations: report first 17 versus remaining steps, observed bearing
  residual <=0.5 versus >0.5 rad, and true angle within 0.1 rad of the circular seam.
  These are descriptive localization strata fixed from truth/data before scoring.
  Residual >0.5 is an observed large-error category, not a claim of known latent
  uniform-mixture membership. Empty strata are unavailable, never silently pooled.

## Budget, commands and artifacts

At most 1800 CPU seconds across three attempts, zero GPU seconds; reserve 120 CPU
seconds within that cap for implementation/smoke/audit. Existing active allocation
covers this phase. Each subprocess is bounded by the remaining phase budget.
Stop after both fixed seeds; no optional extension based on observed metrics.

Exact supervisor command from /home/chakwong/BayesFilter:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B docs/benchmarks/reference_kdm_development_evaluation.py --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-development-evaluation-20261008-01

The supervisor launches the CPU-only worker using
/home/chakwong/anaconda3/envs/tf-gpu2/bin/python -I -B, records the exact command,
environment, git/source/helper/checkpoint hashes, seeds, CPU status, elapsed time,
logs and outputs. Unique versioned output directory, no overwrite. Preserve
per-trajectory/time NLL and squared errors, source metric aggregation, conditional
summaries and conversion checks. Result note:
controlled-smc-kdm-development-evaluation-result-20261008.md.

## Skeptical pre-run review

The wrong baseline would be a fresh or final-epoch model; bind the actual best
checkpoints and audited stage transfer. Proxy validation loss is not acceptance:
the new data and source evaluation measure the intended consumers, but still
only diagnose one training pair. A successful command could mislead if RMSE is
aggregated differently, sparse labels are retained, the short training horizon
is reused, checkpoint conversion changes orientation, or samples are treated as
independent training units. Executing the pinned evaluator, checking T=150 and
dense labels, exact checkpoint round-trip replay, and retaining per-sequence
metrics catch these mistakes early. Different stopping epochs/random streams
are expected, not translation bugs. Pre-run verdict: proceed to this bounded
diagnostic; scientific numerical and qualification vetoes remain in force.


Localized infrastructure amendment before attempt 03: attempts 01/02 consumed
5.9782 and 5.4238 CPU seconds. Attempt 02 passed the actual source checkpoint
round-trip but its validity harness treated an author-optional None resampling
bandwidth as a tensor. Require the primary particle/weight/bandwidth tensors;
check optional resampling tensors only when present. The source model and
data/numerical contract are unchanged. Increase the attempt cap from two to
three under the unchanged 1800-second total cap, consistent with authorized
localized infrastructure repair. No additional compute or scientific approval
is required. The original attempt records and failures remain preserved.
