# NeuTra geometry and checkpoint selection repair

Status: executed and reviewed, including the two documented validation amendments.
The code repair is complete; rare-region sampling remains unresolved. See
`bayesfilter-neutra-geometry-selection-results-2026-10-01.md`. This is a new bounded
cycle following the completed gap-closure cycle, under the remaining shared
allocation (82,690.797 GPU process-seconds; 172,323.215 conservative CPU
core-seconds). The owner requested code/math tracing, a reviewed repair plan,
and execution. Existing passing and failed evidence remains unchanged.

## Findings and mathematical question

`neutra_warm_start_closure.refine` saves RKL checkpoints at 256, 1024 and
2048 updates but nominates only the lowest forward-KL map. It never assesses
the other shape-passing maps downstream. Its terminal probe covers only that
nominee. `neutra_warm_start_diagnostics.features` checks coordinates, squares
and mode half-space for mixtures, omitting the CDF/valley/conditional-coordinate
checks used in development. Passing that smaller set does not establish those
omitted properties. These are selection and diagnostic-coverage gaps, not
evidence that the previously reported posterior screens were computed wrongly.

For x=T(z), the actual score residual is

    r(z) = J_T(z)^T grad_x log p(T(z)) + grad_z log|det J_T(z)| + z.

The inspected manual pullback implements these terms; verify it against full
autodiff and finite differences at the observed outliers before interpreting
them. A bounded conditional log scale controls neither conditioner derivatives
nor off-diagonal Jacobian entries. Large shear and cancellation can therefore
coexist with apparently acceptable scale-saturation summaries.

Neither KL controls this residual without extra regularity assumptions. For
p_k(z)=phi(z) exp(a_k sin(k z_1))/Z_k and a_k=k^(-1/2), log Z_k is between
-a_k and a_k, so both KL directions are at most 2a_k and tend to zero.
The residual is a_k k cos(k z_1)e_1; under phi its squared expectation is
k(1+exp(-2k^2))/2 and diverges. Thus forward-KL selection is a density-fit
nomination, not an HMC geometry decision. This counterexample does not claim
that the learned maps have sinusoidal errors.

## Evidence contract and intent ledger

Question: are the observed residual tails correct, which chain-rule terms
explain them, and does preserving admissible RKL checkpoints allow the actual
HMC consumer to find a map satisfying a wider declared posterior screen?
Baselines: the saved warm maps for mixture seeds11/37, their three existing
RKL checkpoints, and the previously selected wiggle seed23 map for geometry
diagnosis only. No training, architecture, objective, mass or target change.

Primary engineering criterion: independent derivative/roundtrip checks and
consumer regressions. Primary candidate criterion: fresh public fixed-map
tuning and shared sequential HMC, with named physical moments, binary CDF
events, valley and (for warped mixtures) unwarped second-coordinate events.
The existing R-hat, ESS and MCSE/SD limits remain unchanged. Final reference
is opened once after map and member selection, never to choose another map.
The additional event screen is a new versioned scope, not a retrospective
reclassification of earlier evidence or a claim of exhaustive coverage.

Numerical invalidity, derivative mismatch, invalid artifact or missing required
probe vetoes a candidate; a demonstrated shared score/harness defect vetoes
further numerical interpretation until repaired. Budget exhaustion stops work.
A finite checkpoint/HMC failure triggers the next declared checkpoint. Residual
quantiles, Jacobian condition numbers, term norms, losses, clipping and timing
are explanatory only. They receive no new finite acceptance cutoff. RKL is not
claimed to improve geometry merely because it is later in the training path.

No method superiority, universal defaults, uniform whitening, exhaustive mode
coverage, q20 readiness or reliability rate will be concluded. Two selected
training seeds and selected historical failures cannot establish those claims.

## Phases and numerical provenance

1. Probe all eight preserved mixture maps and the selected wiggle map. Use two
   common 1000-row Gaussian banks per map (1000 is the owner's standard; two
   banks is a bounded replication hypothesis), plus two disjoint 1000-row
   subsets of the existing development posterior reference, mapped inversely.
   Save actual rows, residual vectors, log ratios, Jacobians, singular values,
   chain-rule terms and hashes. The posterior subsets are explanatory and do
   not serve as final holdout. Check top eight residual rows plus eight ordinary
   rows per bank; eight is a cheap coverage hypothesis. Autodiff relative
   tolerance 1e-9 and central-difference steps 1e-4,1e-5,1e-6,1e-7 with minimum
   scaled error <=1e-5 are FP64 diagnostic hypotheses. Preserve the full step
   ladder; an inconclusive difference requires localization, not a claim that
   the score is wrong. Roundtrip tolerance 1e-8 is inherited. All repeated
   numerical evaluations use stable batch-native TF/XLA graphs without pfor.
2. Repair refinement to preserve a checked shortlist of every shape-passing,
   numerically valid checkpoint and the warm fallback; each gets a standard
   1000-point probe. Declare order RKL2048, RKL1024, RKL256, warm: a finite
   preference to test the requested RKL completion, not a quality ranking.
   FKL remains recorded for explanation and compatibility nomination only.
   The actual qualification consumer tries the shortlist, each with fresh
   tuning and at most three verified members, without reading final reference.
   It selects the first posterior-screen pass and applies the final reference
   once. A final-reference failure ends the attempt. Invalid maps are excluded;
   no failed map silently masquerades as the warm fallback.
3. Add an explicit broader mixture diagnostic profile with named binary events:
   x0 CDF cuts [-7,-5,-3,-2,0,2,3,5,7], |x0|<2, and second-coordinate CDF cuts
   [-2,0,2]. These are inherited development quantities, not thresholds tuned
   on final results. Binary tail-quantile ESS remains inapplicable; use the
   shared typed binary probability diagnostics. Rare-event precision here is
   absolute MCSE relative to event SD, not relative precision of its probability.
   Test sensitivity using distributions with matching low moments but wrong
   valley mass; test holdout isolation, fallback and numerical exclusion at the
   actual consumer boundary. Preserve legacy diagnostic profile for old readers.
4. Build reviewed shortlists from the saved maps after phase1. Execute the two
   mixture cases with the broader profile. Stop each case at the first map/member
   passing health, convergence and precision; failed candidates are retained.
   A fresh exact reference of32768 rows is generated for each case with a new
   deterministic seed and consumed once. HMC controls are inherited from the
   last repair: fixed latent identity, L=[3,9,18], initial epsilon .5, bounded
   public tuner, 2000 minimum warmup, at most10000 retained/chain, Rhat<=1.01,
   ESS>=400, MCSE/SD<=.03. No mass adaptation or acceptance-band relaxation.
5. Review artifacts, exact source/input provenance, all exclusions and costs.
   Record terminal decisions, uncertainty, strongest alternative explanation,
   remaining gaps and shared master next phase. Completion may mean the bounded
   repair was tested but some candidates remain scientifically inadequate.

Environment: existing tfgpu Python, FP64 analytic benchmark path, XLA, two CPU
threads, GPU1 after trusted readiness check, growth enabled before TF import
and verified. CPU-only regression tests explicitly hide GPUs. Numerical sources
are frozen for workers. Output cycle: campaign-r1/geometry-selection-20261001-r1;
fresh attempts have prefix geometry-20261001. Sub-cap: 4 GPU process-hours and
8 CPU core-hours, taken from the existing allocation; at most16 numerical
workers plus two final-reference workers and two localized infrastructure
retries. Per-worker ceiling1200s for diagnostics and qualification; selection
has an overall finite checkpoint count and per-case wall ceiling3600s, inside
the sub-cap. Ceilings are operational bounds, informed by the preceding
roughly100-second HMC case, not convergence constants. Expected work below one
GPU hour; compilation and new event diagnostics may increase it. The controller
uses the existing shared run ledger and refreshes status between workers.

## Skeptical review before implementation

The first proposed shortcut would have selected maps by residual q99 or by
forward KL again. Rejected: neither is the downstream criterion and1000-row
tails have high sampling variability. Another shortcut would have reused final
reference for each candidate; rejected because it leaks holdout into selection.
The reviewed plan retains the baseline warm maps, uses common recorded probe
rows, tries finite candidates in a fixed order, and separates selection from
confirmation. Different target/reference laws are labeled explicitly. The
additional posterior events use the same definitions as development, including
the warped-coordinate transform. Numerical caps, thresholds and inherited
settings are declared hypotheses; no new optimizer or architecture is promoted.

Premortem: a candidate can pass finite moments/events with localized stiffness;
this remains a reported limitation. Rare events may yield degenerate diagnostics;
that is insufficient evidence, never permission to mark an unobserved event
converged. A larger screen can consume more draws or reject all candidates;
retain that result rather than weakening the screen. A runtime/source change
may change streams; compare new runs within their frozen scope. The plan passes
skeptical review for this bounded repair and these qualified conclusions.

Implementation review:43 focused geometry, consumer, continuation and controller
tests passed in68.32 seconds, CPU-only with GPUs intentionally hidden. The
corrupted-pullback injection is detected; scalar finite differences agree on a
nontrivial configured map. Checkpoint selection defers final reference until
both map and member pass. Probe records bind the saved transport hash. GPU1 was
idle at trusted readiness inspection; GPUs0/2 were occupied. Command:
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_neutra_geometry_repair.py campaign`.

## Diagnostic-control amendment before terminal interpretation

Review of the first seed's failures identified an assumption needing an explicit
control: inherited R-hat/ESS thresholds have not been calibrated for the newly
added rare valley event. Before attributing those failures to the maps, run64
independent exact-mixture replications through the **actual** broader posterior
assessment:1000 draws/chain for the warm-up window and2000 independent
draws/chain for retained assessment, four chains. These sizes match the existing
controller's earliest assessment sizes. Sixty-four is a bounded reference
replication count, not a precision guarantee; report Wilson95% intervals and
individual failure reasons. Seeds [38101,replication] are predetermined and
disjoint from training, probes and final reference. This control does not tune
thresholds or replace HMC validation. It asks whether exact sampling ordinarily
passes this finite screen; frequent failures require diagnostic localization
before interpreting candidate rejection. It cannot establish validity under
dependent MCMC or a universal false-rejection rate.

Run CPU-only with GPUs deliberately hidden, unchanged numerical source snapshot,
two threads,300s wall cap and600 CPU core-second cap, charged to this cycle.
Artifact: `geometry-selection-20261001-r1/iid-screen-control/`. This late control
corrects an omission in the first plan review; preserve that limitation rather
than implying the initial review calibrated the new event diagnostics.
The skeptical amendment review passes: known exact baseline, actual consumer,
independent replications, unchanged thresholds and explicit inference limits.

## Rare-event reference repair after terminal inspection

The v3 candidate for seed37 passes with only2 valley visits among24000 retained
draws, all in one chain. Its estimate is.00008333 against the fresh reference
.00122070. The existing agreement tolerance is.00188801, larger than the
reference probability itself. Algebra explains this: a zero estimated event
probability can pass whenever approximately
`p <= sqrt(p(1-p))*(4/sqrt(n_reference)+.03)`, before accounting for chain MCSE.
The .03 reference-SD allowance dominates a sufficiently rare probability.
Also, MCSE/SD=.03 means relative MCSE about `.03*sqrt((1-p)/p)`, which is about
.816 at the true valley mass. It was wrong to let the word coverage suggest a
stronger result; the declared coarse rule itself was applied as written.

Repair an explicit prospective profile `moments_regions_shape_v4`:
(a) retain the same event definitions and moment thresholds;
(b) require both event outcomes in **each** retained chain before selection,
without imposing this condition on the1000-step warm-up window;
(c) remove the unmotivated .03 reference-SD allowance for binary event
probabilities, leaving the inherited4-combined-SE comparison. Continuous
moment tolerances and all HMC controls remain unchanged. Per-chain observations
are a necessary finite evidence condition, not a precision or convergence proof.
No positive event-count constant is introduced beyond requiring both outcomes.
The4-SE rule remains an operational CLT-based screen, not exact small-count,
simultaneous or sequential coverage. Relative rare-probability precision remains
unestablished and must be reported explicitly.

Do not overwrite v3 outcomes or reinterpret them as v4 confirmations. Reassess
the already frozen selected draws once as a **retrospective diagnostic**, with
no new map selection or holdout-driven retry. Add consumer regressions proving
the retained callback is enforced only for v4, and that the saved low-count
case fails the new event agreement. Check64 exact iid controls at2000 and4000
retained draws/chain, seeds [38401,replication], with the actual new callback
and comparison; report finite pass rates, not an acceptance guarantee. Same
CPU-only reference bounds as above. No new HMC campaign is justified merely
to obtain a pass; the next scientific experiment must discriminate map
geometry from rare-region kernel resolution.

Review: the amendment removes an unsupported additive probability tolerance,
keeps the old evidence intact, names the stronger scope and exposes the
small-count limitation. It repairs a validation defect without asserting that
it repairs the maps. A stricter screen may leave both cases unresolved; that
is the correct report if the saved runs lack the required evidence.
