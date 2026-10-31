# SIR neural training continuation: terminal result

21 October 2026. The reviewed continuation completed on
`ledh-refactor-with-policy-fix`, base commit
`4d6915cc66e701c4c7b657e094306d298fdd3e76`. The nominated initialization now has a
validated 512-update checkpoint. Recovery engineering passes; independent moment
improvement and model-score accuracy were not evaluated in this continuation.

## Question, repair and checked result

The original optional 512-update process stopped after 362 accepted updates,
but saved neural/Adam state only after completed validation. Its last resumable
checkpoint therefore contained 128 updates. Logged losses cannot reconstruct
the missing optimizer state. The repair saves all parameters, Adam slots and
iteration, the next data/particle indices, configuration and source identity
every eight updates. These recovery files are explicitly unselectable and have
no measured validation result. Validated checkpoints remain separate.

The new runner inherits the actual `NeuralKDMStudy.fit` training loop. It does
not replace its numerical kernel. The target is equality of the resumed finite
training program, including the next random-data indices and Adam update.
Forty-three focused CPU checks pass, including interrupted/uninterrupted tensor,
optimizer and next-seed parity and actual-consumer wiring. CPU devices were
intentionally selected by setting `CUDA_VISIBLE_DEVICES=-1` before import.

Attempt21 resumed 128 and completed 256 updates; attempt22 resumed 256 and
completed 512. All 384 continuation updates were accepted, with finite losses
and six finite positive gradient norms. All 234 updates overlapping the earlier
interrupted process exactly match its recorded losses, block losses, six
gradient norms and optimizer/validity flags. This proves equality of those
recorded quantities; the lost original terminal tensors cannot be compared.
Each new validated checkpoint equals its corresponding recovery state, and each
attempt starts from exactly its declared resume state.

The terminal inspector passes all 30 checks for attempt21 and all 46 for
attempt22. All 834 numerical package file hashes match. Each saved runner copy
matches its own manifest, and neither runner changed during execution. The
schedule was extended between attempts without changing numerical package code.
Both processes exit zero and preserve the final accepted state. See
[terminal-inspection.json](terminal-inspection.json) and
[inspection code](inspect_continuation.py).

## Validation and scientific interpretation

Each row uses the same 32 validation datasets and two particle replications.
These are scaled simulated-label losses, including irreducible label variance;
they are not measured posterior-moment MSEs or independent improvement evidence.

| Accepted updates | Validation loss | Source |
|---:|---:|---|
| 32 | 0.678633411917 | Original frozen selection |
| 128 | 0.682256043134 | Attempt14 resume checkpoint |
| 192 | 0.682556397305 | Attempt21 |
| 256 | 0.671551450052 | Attempt21 |
| 384 | 0.682223242252 | Attempt22 |
| 512 | 0.707064472846 | Attempt22 |

The observed loss falls at 256 and rises at 384 and 512. Longer training is
therefore not a monotone improvement on this validation set. These reused
validation measurements do not rank generalization performance, demonstrate
overfitting, or show that optimization has converged. No newly trained candidate
was evaluated on the exposed holdout, and the original frozen 32-update
candidate was not replaced retrospectively.

The earlier independent comparison remains negative: trained loss 0.770637,
untrained 0.774066, plain LEDH 0.765211 and recursive Gaussian 0.533512.
The trained-minus-untrained and trained-minus-plain paired intervals include
zero; the 10% improvement rule fails, and the recursive-Gaussian conditional
comparison vetoes promotion. The two boundary datasets limit precision in that
stratum. Those findings concern the original frozen candidate, not the new
256- or 512-update checkpoints. See the
[original independent result](../attempt-01/neural-holdout-result.md).

## Execution, provenance and budget

Both attempts use `tftwogpu`, TensorFlow `2.20.0-dev0+selfbuilt`, escalated access
to the RTX4080SUPER, verified memory growth before device initialization, FP64,
TF32 disabled and XLA enabled. The recorded CUDA command-buffer workaround is
unchanged. Scope remains SIR d18, T5, N64, batch4, OT(1,5), balance32, K=N64 and
the shared UKF/GenUT/Contract-E consumer. This is a diagnostic training scope,
not a canonical tuning artifact. The numerical data version is
`canonical_sir_d18_x0_then_transition_observe_v1`.

Configuration is K2, width16, beta0.9, Adam rate0.001, initialization263044001.
Calibration seed263040001 and reset seed263040002 are unchanged. Training and
particle namespaces are263041000 and283010000; each accepted-update index
determines the next seeds. Full commands, environment, source closure, resume
hashes, device and memory provenance are in the two manifests and external
accounting files:

- [Attempt21 manifest](../attempt-21-continuation/manifest.json),
  [external accounting](../attempt-21-continuation/outer-accounting.json),
  [log](continuation.log).
- [Attempt22 manifest](../attempt-22-continuation/manifest.json),
  [external accounting](../attempt-22-continuation/outer-accounting.json),
  [log](continuation-512.log).
- [43-check CPU log](cpu-tests-512.log) and
  [reviewed plan](../../../ledh-sir-neural-kdm-continuation-20261021.md).

External process accounting includes imports and initialization. Attempt21 is
charged1107seconds and attempt22 is charged1802seconds; the GPU inventory probe
is charged1second. Child elapsed times exclude imports and must not replace
these charges. All previous failed attempts remain charged. Total campaign use
is21785 of23766.197895057seconds, leaving1981.197895057seconds, about33minutes.

Both initializations completed the mandatory128-update stage. The second
initialization's optional512 rung is under-budgeted: its last20 update times
have90th percentile7.69309246796seconds. The predeclared conservative projection
is at least `1.25 * 384 * 7.69309246796 + 180 = 3872.68438462`seconds before
adding validation time, exceeding the1981.197895057seconds remaining. No new
training launch or independent comparison is started. This is a resource limit
on that optional rung, not a numerical veto or rejection of the research idea.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept recovery repair | Exact state/seed checks and replay pass | No source, numerical, device or budget veto | Lost original terminal tensors are unavailable | Retain validated512 and periodic recovery states | Scientific improvement |
| Close this bounded continuation | Nominated initialization reaches512 | Second initialization's optional512 exceeds remaining budget | Initialization and optimization sensitivity | Plan adequate replication and fresh evaluation before any new claim | Training adequacy or direction failure |
| Keep promotion closed | Independent10% criterion previously failed | Conditional heuristic veto remains | Fresh checkpoints have no independent evaluation | Freeze any later selection on validation, then use untouched data | Default readiness or score accuracy |

| Inference question | Status |
|---|---|
| Hard veto screen | No new engineering/numerical veto; earlier heuristic promotion veto remains |
| Statistically supported ranking | None for the continuation; no new independent comparison |
| Descriptive-only differences | Validation loss is lowest at256 among these checkpoints and higher at512 |
| Default readiness | Not established; no promotion permitted |
| Next evidence needed | Adequate seed/budget coverage, frozen validation selection, fresh paired comparison against the original baseline and conditional heuristic set |

Post-run red-team: successful state recovery cannot diagnose why the proposal
fails independently. The strongest alternative explanation for the validation
pattern is finite validation/particle noise combined with a single optimization
trajectory; overfitting or a harmful learning rate remains possible but is not
identified. Repeated training trajectories and a new untouched paired evaluation
could overturn the candidate-level negative result. The weakest evidence remains
generalization of later checkpoints and the sparse boundary stratum. Neither
the population moment-risk identity nor model-score improvement is tested by
successful recovery.

## Document closure

The master program, adaptation/continuation plans and concise checkpoint now
record this terminal state. The manuscript adds Section15.15, page98, with the
complete validation trajectory and its interpretation. The updated105-page
PDF builds twice with no warnings, undefined references or box overflow.
Rendered pages3,95,96,98,99 were checked. Comparison with protected source
copies finds no removed inline/display mathematics, equation environments or
citations. The existing compact two-digit subsection numbering in the contents
is retained; the factual update does not change global document styling.
See [document check](document-check.json),
[second build log](latex-continuation-pass2.log), and
[updated PDF](../../../../papers/ledh_younis_kdm_score/ledh_younis_kdm_score.pdf).
This is a factual documentation update, not a new formal mathematical audit or
an independent review of the scientific method.
