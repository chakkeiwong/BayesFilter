# Streaming RNG and compiler profile result

Runs 04750–04755 complete the bounded explanatory plan
`filter_gradient_streaming_profile_20260929.md`: six sequential CPU-reference
workers, 40.675878 process-seconds, zero GPU seconds. All passed, including 161
final source-bound readback and policy checks. GPU remains the default target;
these measurements are explicit CPU references. The numerical runtime was not
changed in this unit. Commands, source hashes, environment, seeds, HLO and
memory observations are preserved in each numbered run.

The saved optimized HLO from the matched full-filter cohort puts buffered RNG
in its own loop and streaming RNG inside the observation loop. The global loop
count falls from 32 to 31, and computation counts fall from 878 to 872 at T32
and 881 to 875 at T128. The observation body's unique reachable computations
contain three additional copies and six additional fusions (357 to 360 copies,
596 to 602 fusions). Global sine, cosine, log and dot counts are unchanged.
These are static counts, not executed multiplicities or causal timing evidence.

| CPU RNG component | Buffered warm median | Streaming warm median |
|---|---:|---:|
| T32, N64, d2, float64 | 0.429479 ms | 0.413554 ms |
| T128, N64, d2, float64 | 0.924752 ms | 0.792708 ms |

Each component owner used a stable enclosing XLA signature, three conditioning
calls and 30 synchronized timed calls. Full noise clouds and final Philox states
agree exactly at seeds 123 and 124; each owner traced once. These are one-process
component measurements. Buffered materialization and streaming summaries differ
in allocation, and neither component includes the full filter. Therefore the
measurements do not rank full-filter performance or establish its slowdown's
cause. Raw RNG component cost alone does not explain the observed regression.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Close bounded profiling | HLO provenance and component clouds/states checked | All six workers pass | Component/full-filter compiler interaction | Preserve result; continue concrete score-consumer repair | Full-filter cost acceptance |
| Keep streaming cost gate open | Prior paired upper bounds 1.11687/1.15608 exceed 1.10 | Performance acceptance blocked | RNG state placement, scheduling/fusion, timing noise | One reviewed full-filter intervention preserving all records and RNG scheduling | Three extra copies caused the slowdown |

| Inference status | Finding |
|---|---|
| Hard veto screen | Component numerical and execution checks pass; existing full-filter performance gate remains open |
| Statistically supported ranking | None from this component study |
| Descriptive differences | Streaming component medians are lower at both horizons |
| Default readiness | Not established; CPU reference, no new GPU cost/capacity evidence |
| Next evidence | Matched full-filter causal intervention and uncontended GPU costs |

Post-run skeptical review: compiler optimization of the reduced component is
the strongest alternative explanation for its lower times. An intervention on
the full filter, with identical outputs, validity-gated draws and seeds, could
overturn the working compiler-interaction hypothesis. The weakest evidence is
the single-process component timing; no promotion follows from it. Do not change
streams, prefetch beyond prediction validity, relax tolerances or accept the
slowdown based on static opcode counts. The separate score-study enclosing
direction repair remains the next concrete implementation task.
