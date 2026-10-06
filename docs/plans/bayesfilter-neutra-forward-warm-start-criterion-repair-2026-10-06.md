# Forward warm-start criterion repair and continuation

The owner explicitly rejected requiring a fully accurate forward checkpoint and
authorized a more reasonable rule followed by continued execution. The intended
question is whether approximate forward fitting initializes successful reverse
training. An intermediate feature-accuracy veto was answering a stronger question
than that intended computation. The final RKL accuracy screen stays unchanged.

## Revised evidence contract

Version `bayesfilter_neutra_forward_warm_start_then_final_v2` requires:

1. The native teacher passes its existing independent-bank checks.
2. The forward checkpoint reloads, its existing density/moment calculations are
   finite, and its 1,000-point probe is complete and finite. Keep the inherited
   identity-Gaussian cross-entropy difference <= .25 nats as a weak gross-fit
   guard, not an absolute KL bound.
3. The forward map retains the inherited maximum absolute component-mass error
   <= .15. In addition, every evaluated component must retain at least half its
   reference mass. This stops the absolute tolerance from admitting complete
   loss of a .10-mass component. The factor two is an explicit heuristic bound
   on acceptable underrepresentation for a warm start, not a fitted threshold
   on the observed .2-sigma mean shifts. It implies at least half the nominal
   expected component occupancy in the current synthetic evaluation; it is not
   a discovery, tail-weight, RKL recovery or HMC guarantee.
4. Forward feature z-scores and KL estimates are explanatory only. Record the
   old full-screen verdict separately, even when warm-start eligibility passes.
5. Each final RKL endpoint must pass the same full distribution and finite-probe
   screen as before, and its complete training path must remain finite without
   the existing majority-clipping veto. No final threshold changes.

The exact component labels and masses are evaluator information for these
synthetic cases only; this rule is not transferable to q20 without a separate
coverage diagnostic. Half-mass is a convenience risk limit with an interpretable
failure mode. Synthetic missing/underrepresented-mode tests are the early check.
Its role is a forward warm-start veto; final feature accuracy remains an endpoint
promotion veto. A failed final fixed case still blocks randomized continuation.

## Preservation and continuation

The six fixed cases have been inspected. Reassessing them under the owner's
revised rule is explicitly retrospective and cannot be called fresh confirmation.
Archive the old queue/result/selection before updating decisions; never edit
the worker result files or their manifests. Write a separate criterion-revision
record containing old and new decisions and exact source-result hashes.
Recheck the existing selected setting on all six calibration cases. Keep its
rate and update count frozen; changing an intermediate screen does not retune
the sampler or optimizer. Reassess fixed results from their saved artifacts.

If all fixed pairs satisfy the revised rule, resume the 12 previously untouched
random geometry cases (four geometries, three fitting seeds). Their results use
the same v2 criterion from the outset. Keep the existing shared/local budgets,
versioned outputs, GPU 2 assignment subject to availability, CPU teacher lane,
memory growth, FP64 diagnostic exception, XLA, and stage reservations. No
training rerun is needed to reevaluate saved metrics.

## Skeptical audit and verification

The old baseline requirement confused warm-start suitability with final fit
accuracy. Simply raising z from 5 to an observed value would tune a precision
cutoff to these failures and would still answer the wrong question. The revised
rule changes the diagnostic's role instead. It preserves gross mode loss and
numerical failures as vetoes and keeps final accuracy strict. The main remaining
risk is an approximate map passing this coarse screen but failing to improve;
the final screen explicitly measures that outcome.

Verify rejection of missing modes, nonfinite/missing probes, checkpoint failure,
bad density guard, majority clipping and bad final endpoints. Verify that a
finite shifted warm start can pass while retaining its old rejection label,
that saved artifacts are not rewritten, and that resume uses the frozen recipe
and launches only outstanding random cases. Focused CPU tests and an artifact
reassessment suffice because the numerical training kernels are unchanged.
Record any code or artifact inconsistency as a continuation veto before launch.

Review verdict: the change answers the intended warm-start question and is
authorized by the owner's current instruction. It makes no retrospective
confirmatory or downstream posterior claim. Verification must pass before the
new numerical phase starts.

## Verification and preserved-map reassessment

All 46 focused tests pass across the completed runs: 41 criterion/controller
checks in `forward-warm-start-criterion-check-r2`, plus the five actual numerical
call-chain tests that passed in r1. The r1 fixture initially compared a JSON list
with a Python tuple; correcting its seed representation repaired that sole test
failure. Both attempts remain charged and preserved. No numerical kernel changed.

The read-only reassessment checked worker artifact hashes and applies v2 to the
selected reverse branch on all six calibration and six fixed cases. All pass.
The minimum per-component mass ratio is .9333 across calibration cases and
.9801 across fixed cases. The selected SMC profile, forward schedule and .0001 /
256 reverse recipe remain frozen. The two old forward feature-screen failures
remain recorded as failures of that stricter diagnostic, while their warm-start
eligibility and final RKL maps pass. Reassessment is retrospective.

GPU 2 was available at the continuation check; GPU 1 contained unrelated work.
The next exact action is the existing trusted wrapper's `forward-reverse`
command, which archives old controller decisions, verifies the frozen selection
again, reserves the complete random stage, and starts only the 12 new random
cases. No previous training or teacher generation is repeated.

## Completed execution

The master completed the 12 reserved randomized geometry/seed cases under v2.
All teachers, forward warm starts and final RKL maps passed. Final criteria and
the frozen training recipe were unchanged. Every random forward map also passed
the old full diagnostic; the criterion revision enabled continuation past the
two preserved fixed-case intermediate failures without rewriting them.

The r2 terminal audit verified 936 artifact hashes, 1,439 source hashes, 78
complete finite probes and all 24 original result hashes from the revision.
The completed controller launches zero workers on a copied-state resume test.
Accounting matched across local/shared ledgers with no unsettled charges.
See `bayesfilter-neutra-naf-forward-reverse-results-2026-10-06.md` for per-case
results, cost, uncertainty and remaining tail-geometry limitations. This repair
and continuation are complete; downstream HMC/q20 readiness is not established.
