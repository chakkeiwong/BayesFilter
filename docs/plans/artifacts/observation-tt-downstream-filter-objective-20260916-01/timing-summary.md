# A08 observed computation times

Date: 2026-09-16. Post-run descriptive extraction of existing A08 timers;
no new benchmark, fitting, tuning or numerical execution was performed.

Source: `docs/benchmarks/artifacts/observation_tt_downstream_filter_objective_20260916/attempt-full-01/result.json`,
`sequences[].times`, from the completed RTX 5080 TensorFlow float64/XLA run.
Every particle repetition has 512 particles and 20 observations. Within each
dimension, all four methods use the same complete sequences: twelve for d=1,
eleven for d=4. The failed d=4 sequence 8 is excluded from every row.

The runner records proposal construction as `fit_seconds` and the whole
method block as `total_seconds`. Each block constructs proposals once and
executes four particle-filter repetitions. Reported particle-run time is
`(total_seconds - fit_seconds) / 4`; it includes residual artifact writing
and any compilation occurring inside that block. It is not an individually
timed, warmed kernel latency. Guide construction is recorded separately as
`times.guide`. All values below are arithmetic means across matched sequences.

## Four-dimensional model

Mean shared guide construction: **1.3128 seconds** per 20 observations.

| Method | Proposal construction / fitting (s) | Particle run and recording (s), four-run average | Observed method block: one fit + four runs (s) | Estimated guide + one fit + one average run (s) |
| --- | ---: | ---: | ---: | ---: |
| Earlier guided TT | 18.3514 | 0.5010 | 20.3553 | 20.1652 |
| Repaired pair TT | 31.5516 | 0.9004 | 35.1531 | 33.7648 |
| Selected TT/SGQF | 54.1781 | 0.6982 | 56.9707 | 56.1890 |
| SGQF joint | 0.2369 | 0.1593 | 0.8742 | 1.7090 |

Particle-run averages correspond to 25.05, 45.02, 34.91 and 7.97 milliseconds
per observation respectively, after proposal construction. The corresponding
guide-plus-fit-plus-average-run estimates are 1.01, 1.69, 2.81 and 0.0855
seconds per observation. These latter values include fitting at each step;
they are arithmetic estimates, not separately measured standalone launches.

## One-dimensional model

Mean shared guide construction: **0.6459 seconds** per 20 observations.

| Method | Proposal construction / fitting (s) | Particle run and recording (s), four-run average | Observed method block: one fit + four runs (s) | Estimated guide + one fit + one average run (s) |
| --- | ---: | ---: | ---: | ---: |
| Earlier guided TT | 4.3261 | 0.2747 | 5.4249 | 5.2467 |
| Repaired pair TT | 2.9773 | 0.3752 | 4.4780 | 3.9984 |
| Selected TT/SGQF | 6.4263 | 0.3074 | 7.6559 | 7.3797 |
| SGQF joint | 0.2020 | 0.1430 | 0.7739 | 0.9909 |

## Interpretation and limits

Proposal fitting dominates the TT cost in this implementation. The selected
route evaluates generic-start and SGQF-start TT candidates over an L1 grid,
with validation, audit diagnostics and artifact writing included in its setup.
This is more work than a single TT fit. Its particle stage also sometimes uses
SGQF directly, so its sampling time does not describe a TT-only consumer.

Shared kernels are cached and methods run in a fixed order. Compilation and
serialization are not isolated, and common setup can be paid by an earlier
method. The guide is built once and shared by all methods and repetitions.
The final column adds its mean once to make setup visible; it does not include
independent-reference generation or process startup. No isolated cold-start,
steady-state GPU throughput, scaling law or statistically supported speed
ranking is established. A dedicated timing protocol would need explicit
warm-up and synchronization, separate compilation/I/O timers, and repeated
measurements before supporting those claims.

The full 2023.805-second campaign time includes all eight methods, references,
fitting, repetitions and reporting; it is not any individual filter's runtime.
