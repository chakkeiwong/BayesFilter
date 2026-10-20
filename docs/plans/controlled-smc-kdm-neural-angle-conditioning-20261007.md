# KDM neural angle conditioning — bounded verification, 7 October 2026

Authority: the active master and owner-approved acceptance criteria. This is
replication diagnostics within the remaining allocation. No author-method change.

## Question and evidence contract

Do the two preserved float32 particle-angle failures reflect the conditioning of
the author's normalization/atan2 map, while the upstream proposal, actual loss,
gradients, clipping and Adam updates remain compatible? Compare the actual pinned
author neural consumer and TensorFlow GPU/XLA consumer on replayed common draws.
Use the preserved failing fixture plus two prospectively fixed diagnostic fixtures.

Promotion here means passing this explicitly conditional numerical screen only.
It cannot accept KDM replication or trained-model quality. The circular-gradient
failures, cSMC fitting and independent-trained-unit qualification remain open.

The primary criterion is simultaneous success of: all existing non-angle checks;
raw four-coordinate proposals within the unchanged 128*2^-23*(1+abs(source))
float32 (1e-9 in float64) screen; the derived circular-angle bound below; local
normalization/atan2 checks for both implementations; all actual neural gradient,
clipping, Adam moment/parameter and call-chain checks at unchanged bounds.
Preserve original particle-angle screen failures separately in every new report.

Nonfinite values, invalid updates, missing source inputs/gradients, changed source
hashes, extra traces, wrong device, or failed non-angle checks veto this screen.
A proposal-error ball reaching the origin makes that coordinate unresolved,
never automatically passing. Corrupt/missing fixtures or budget exhaustion stop
execution; a candidate numerical failure triggers localization, not abandonment.
Runtime and observed error/bound ratios are explanatory only.

## Independent geometric derivation

The author uses a raw proposal a=(s,c), divides by its positive Euclidean norm,
then returns atan2(s,c). Exact positive normalization leaves the angle unchanged.
If each proposal coordinate differs by at most b_j=tau*(1+abs(a_j)), set
E=sqrt(b_s^2+b_c^2), r=sqrt(s^2+c^2). For E<r, the tangent from the origin to
the radius-E ball around a gives circular angular error at most asin(E/r).
This is an exact finite-distance bound, not a first-order approximation.

Check each saved source/TF angle against double-precision atan2 of its OWN raw
proposal with the existing local tolerance tau*(1+pi). Thus the cross-framework
bound is asin(E/r)+2*tau*(1+pi). Calculate all diagnostic geometry in float64.
Use circular distance abs(atan2(sin(delta),cos(delta))) at the branch cut.

The raw-proposal tolerance is a reviewed diagnostic convention inherited from the
original screen, not a theorem bounding every floating-point instruction. The
new bound is CONDITIONAL on its observed success at every coordinate and on both
local-map checks. It is independently derived from that unchanged convention,
not fitted to the observed 4.05e-5 angle difference. Report E/r and the raw
coordinate errors so its conservatism remains visible. Do not claim a universal
IEEE error bound or silently replace downstream performance criteria.

Anchors inspected: pinned author b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18,
src/models/particle_transformer.py:112-160 (BearingsOnlyTransformer);
kde_particle_filter.py:830-846 and 1023 (proposal normalization and transform).
Existing neural reference extracts actual trainer/loss and captures actual
proposal outputs. The TF consumer returns the raw proposal at outputs[5].
The geometric argument is local derivation; no new paper claim is made.

## Defaults, diagnostic baselines and scope

| Choice | Provenance and reason | Failure mode / earliest check | Status |
|---|---|---|---|
| Original source checkpoint and architecture | Existing source neural fixture; preserves comparator | Wrong transfer/wiring; source hashes and trainable call-chain check | Diagnostic baseline |
| Two updates, T=17, N=25; batch 4 for new fixtures | Reuses source loss/TBPTT/optimizer contract and exercises stateful Adam | Few inputs miss edge cases; retain original failing fixture | Bounded diagnostic hypothesis |
| Seeds 202610171 and 202610181; row offsets 256 and 288 | Fixed before launch, distinct from old two-/64-row replay fixtures | These are development rows, not qualification data | Convenience choices, no training-quality claim |
| Raw/local tolerances | Existing 128 eps / 1e-9 screen | May be conservative; report all raw ratios and E/r | Reviewed diagnostic convention |
| Circular distance and finite tangent bound | Geometry above; source angle semantics | Origin crossing; explicit unresolved veto | Derived |
| GPU/XLA, TF32 off, memory growth | Existing comparison environment | Backend/config mismatch; manifest and single-trace checks | Frozen comparison baseline |

This is an implementation/conditioning check, not a learned-method ranking.
The cheap diagnostic adversaries are the actual author raw proposal, float64
atan2 of each raw proposal, and deliberately corrupted angles beyond the bound.
No tracking-quality claim bypasses the later heuristic and oracle comparisons.

## Execution, budget and artifacts

At most 600 CPU seconds and 900 GPU seconds; at most three launch attempts.
Use unique kdm-neural-angle-20261007-NN directories under
docs/plans/artifacts/controlled-smc-kdm-replication-20260928. Record commands,
source/helper/plan/data/checkpoint hashes, devices, memory policy, seeds, wall time,
all check results and complete subprocess logs. Snapshot inputs/helpers per run.
No overwrite or runtime-module changes. Original evidence stays intact.

One supervisor generates the two CPU-only source fixtures, then checks those
and the preserved fixture on GPU/XLA. Reuse is permitted only after artifact
verification for a localized harness repair. Bound subprocesses by remaining
resource budgets and retain incomplete attempts. The author's existing full CPU
training continues on one thread; these short source diagnostics use one
additional thread. The TF full run has ended, so there is no competing campaign
GPU kernel. This explicit narrow concurrency exception supersedes the generic
checkpoint 'no overlapping numerical launch' wording for this phase only.

Exact entry command:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B docs/benchmarks/run_kdm_neural_angle.py --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-angle-20261007-01
Run with escalated GPU permission. The supervisor hides GPU devices in source
subprocesses and records their conda library path. It invokes reference and TF
diagnostic helpers with fixed seeds/row offsets and independent output paths.

## Skeptical review before implementation/execution

PASS for this bounded conditional question. A naive increase in the angle
tolerance would not answer it; raw proposals, both local maps, origin separation
and the real downstream optimizer are checked separately. The math uses a
finite bound and circular distance. A successful run cannot hide a non-angle
regression or be reused as independent-trained-model evidence. Source precision,
source stopping times and TF32 are not conflated. The main residual risk is a
conservative upstream diagnostic threshold; expose it and retain its limited
interpretation. The source fixture code and GPU check will be reviewed against
this contract before launch.

Pre-mortem: a changed random stream without replay, wrong array time alignment,
or using post-normalized vectors as raw proposals would make a clean result
meaningless. Verify capture lengths/shapes and compare the original failing
coordinate. A local-map mismatch, unchanged-bound upstream failure, or corrupted
angle passing its negative control requires repairing the harness or candidate.

## Implementation review before launch

Reviewed the actual source-helper capture, GPU consumer update call, raw proposal
output index, time alignment, angle checks, retained old checks, trainable and
frozen parameter checks, and actual gradient/clipping/Adam checks. No algorithmic
runtime file changed. The original reference is merged with fresh source runs
only after checkpoint, dataset and author-source hash verification. The author
reference uses the already verified tf-gpu2 interpreter and its conda library
path, with GPU hidden and one thread. All three new helpers compile. Independent
standard-library tangent-point checks cover small and near-origin error balls
and recover both the declared error radius and asin(E/r) angle.

Skeptical code review: PASS for the stated conditional screen. GPU execution is
bounded to the remaining 900-second phase allowance; CPU reference execution to
600 seconds across at most three attempts. Parent completion is distinguished
from numerical screen success. Legacy angle failures are retained, origin overlap
vetoes, and source/TF actual optimizer checks retain their previous tolerances.
The checkpoint's old no-overlap sentence is superseded only by this short
diagnostic exception. All post-run geometry runs on CPU in float64 as an explicit
diagnostic; the actual neural consumer remains GPU/XLA.
