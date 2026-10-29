# Fixed-particle stress repair — 18 October 2026

## Question

Does the positive-tail score error fall when particle count increases, with
the message, consumer, controls and model held fixed? This is a mechanism
diagnostic following the message/consumer comparison; it is not a default,
canonical, HMC or tuning decision.

## Evidence contract

Focus the positive-future-tail observation because it has the largest
remaining score error. Use its checked KSC numerical reference and the same
eight saved fit clusters as the preceding campaign. Compare the unguided
global, fitted product/global and exact product/global arms, along with the
six declared heuristics, on 48 fresh clouds. Generate each cloud at `N=64`,
then evaluate its first 32 particles and all 64 particles, so the two counts
are nested and paired under common randomness. Report squared error, bias and
variance for both score coordinates. The primary diagnostic is the paired
change in stress-tail second-coordinate MSE from `N=32` to `N=64`; this
conditional comparison is not a fresh-data population ranking.

The cheap adversaries are identity, global, local, equal, Gaussian-future
global and Gaussian-future equal. Compare each of the two product/global
candidates against all six at `N=64`, and compare each candidate at `N=64`
against its paired `N=32` result. These 14 contrasts use paired cluster
differences, seven degrees of freedom and a Bonferroni simultaneous 95%
interval family. Any heuristic loss remains a promotion veto.
Nonfinite values, invalid traces, broken references, source drift or unpaired
clouds are continuation vetoes. A lower MSE is descriptive unless the paired
uncertainty supports a ranking.

## Frozen assumptions and budget

Controls remain OT epsilon `1`, five Sinkhorn iterations, five reset steps,
two correction steps, four correction features, rate `0.1` and radius `0.25`.
Use FP64 GPU/XLA with TF32 disabled as a diagnostic reference exception and
the repository GPU memory-growth policy. Reuse the eight positive-tail fits
only as a conditional diagnostic; do not claim fresh-fit generalization. Use
fresh cloud seeds disjoint from all prior campaigns. The remaining campaign
budget is 714.570 GPU-process seconds, with one attempt capped at 600 seconds
in a fresh output directory. Preserve any incomplete attempt. Pre-run audit:
the earlier particle-count study used different fits/clouds and cannot answer
this crossed message/consumer comparison; nested clouds make particle count
the only changed factor. This remains a single-condition mechanism check and
cannot support broad model claims.

## Stop and repair rule

If the run is valid, update the result note, master and checkpoint. If `N=64`
does not reduce the positive-tail error, close the particle-count hypothesis
and prepare a consumer/reset design diagnostic; do not enlarge KDM fitting.
If infrastructure fails, repair only the harness and retry with unchanged
target and budget. Do not relax the validity or heuristic screens.

## Execution and skeptical audit

Run `scripts/run_ledh_particle_count_stress_diagnostic.py --output
docs/plans/artifacts/ledh-particle-count-stress-20261018/attempt-01
--budget-seconds 600` with the `tftwogpu` Python, escalated GPU access and
`TF_FORCE_GPU_ALLOW_GROWTH=true`. Then run
`scripts/analyze_ledh_particle_count_stress_diagnostic.py --run` on that
directory with `CUDA_VISIBLE_DEVICES=-1`. Preserve both complete logs.
Fresh cloud seeds are `861800 + 100*cluster + replicate`, for eight clusters
and six replicates. The eight frozen fit seeds are `822000 + cluster`.
There are 768 rows and 16 finite-difference checks. The remaining CPU
reference/analysis allowance is 178.241358376 seconds; references are capped
at 60 seconds. The parent campaign permits three attempts total, of which
one has completed; this repair uses the second attempt.

The pre-run skeptical audit passes for this narrow diagnostic: it compares
the same fits, observations, controls and nested randomness, includes six
constructed heuristic adversaries, checks the actual analytical score against
the finite value program, and preserves the independent numerical reference.
The inherited OT and correction settings are frozen comparator hypotheses,
not selected settings or evidence of canonical conformance. Their failure
mode is discretization or reset error that more particles cannot repair;
the exact-message arm and bias/variance decomposition expose that possibility.
The inherited 256-component fit is a fixed conditional input, not an original
KDM replication claim. The particle counts are a two-point mechanism probe,
not a convergence ladder. Six clouds per fit and eight fixed fits can leave
the intervals inconclusive; no ranking follows in that case.

Pre-mortem: a lower observed MSE can result from paired Monte Carlo noise or
bias cancellation. Retain all per-cloud errors and simultaneous intervals.
A valid run with persistent bias closes only the particle-count repair and
triggers a consumer/reset diagnostic. A broken pairing, invalid numerical
trace or derivative mismatch invalidates the run and triggers harness repair
before scientific interpretation. Each completed phase automatically audits,
records budget use, refreshes the checkpoint and executes the next authorized
repair while the scientific scope and remaining budget permit it.
