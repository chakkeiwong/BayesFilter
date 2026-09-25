# A06: frozen SGQF/TT safeguard on independent filtering sequences

Date: 2026-09-15. Status: REVIEWED AGREE; authorized within H6 balance.
Review: `artifacts/observation-tt-defense-consumer-20260915-01/review-a06-01.txt`.
It assesses the protocol, not uninspected implementation.
This is master phase 9, following A04 initialization and A05 safety calibration.
Existing H6 authorization has about 160 minutes remaining; no new budget is
requested. The target remains the same SV model, d=1,4 and T=20. Pair gradients,
HMC, larger dimensions, coupled charts and default promotion remain outside scope.

## Research intent and amendment

Question: does a frozen per-step procedure that may preserve exact SGQF improve
actual particle-filter accuracy, and does it avoid losses to cheap proposals?
Mechanism: fit from generic and SGQF initial cores, with A05's optional 1e-5
defense, but retain the analytic SGQF joint as a validation-selectable candidate.
The empirical selector is not a performance guarantee. Expected failure: row
coverage or finite polynomial capacity produces a TT that validation selects
but that loses in independent filtering. This rejects the candidate and records
the failure; it does not invalidate the filtering question.

Changes from A04/A05 are fresh observation sequences, actual recursion of the
selected retained marginal, and downstream filtering measurement. Fitting
capacity, controls, row design and selection rule remain fixed below. A05's
Gaussian conditional is added to the comparator ladder. The prior maximum
observation regime definition could miss small observed channels in d4; the
primary conditional analysis here classifies each state coordinate by its
own contemporaneous observation. Whole-vector maximum regimes are also saved
as explanatory context. This change is explicit and pre-outcome.

## Frozen procedure and comparator ladder

Generate 12 independent sequences for each dimension using the existing A,P0,
beta=.4,sigma=1 model, stationary initial state, T=20. Data seeds are
916000+100*d+sequence_index (0–11), with stateless stream 0 for state noise and
1 for observation noise. Generate only after this protocol is reviewed and
controls are written to disk. No previous observations are reused. Observations
are naturally sampled, not selected for a desired method outcome.

Eight methods at N=512, four particle replicates each:

1. Transition bootstrap: cheap, preserves local state dynamics.
2. Stationary Gaussian: cheap broad coverage, useful when observations are weak.
3. SGQF current marginal: the existing observation-aware independent proposal.
4. Exact SGQF joint conditional: preserves state dependence without TT conversion.
5. Predictive scalar TT: existing plain fitted baseline.
6. Guided scalar TT: existing observation-aware scalar baseline.
7. Generic pair TT with 5% defense: existing paired fitted baseline.
8. SGQF/TT safeguard: the frozen enhanced procedure below.

All methods use identical model/data, N, resampling policy and exact proposal
correction. Fits/SGQF setup, compilation and failures count in cost. Baseline
methods retain their reviewed original numerical controls, with their known
limitations; they are not described as optimally tuned classical authorities.
The four Gaussian/transition arms constitute the constructed heuristic set.
No comparator outcome may tune the enhanced procedure.

For the enhanced procedure, t0 uses the unconverted SGQF current marginal and
retains it. Each t>=1 builds the true model transition and likelihood times the
previous *selected retained approximation*. Current/conditioning charts remain
the SGQF guide marginals. Draw 1024 training, 4096 validation and 8192 audit rows
from .2 product-reference + .8 exact SGQF joint. Use training-only scale; degree
3 per coordinate, pair rank 3, four sweeps and 128 proximal steps. Both the
generic and analytic Gaussian-projected starts are scaled by their optimal
positive training-only scalar. Tune L1 in [0,1e-5,1e-3] separately for each start,
time and dimension using validation signed RMS. No dimension transfers a
selected L1 to another. Select the minimum validation Hellinger discrepancy
among the two fitted densities (defense 1e-5) and the exact SGQF joint; ties
favor SGQF. Save selection before generating audit rows. Audit never changes
the selection or controls. The selected analytic or polynomial current
marginal is retained for the next target. There is a separate fit each time.

This fits the *specified algorithm* on each observed sequence, as filtering
requires. Its controls and selection policy are frozen before holdout data;
sequence-level outcomes cannot retune them. Full Gaussian coefficient projection
is diagnostic d<=4 only; no scalable implementation claim. Four-sweep optimization
and row coverage remain hypotheses, not established convergence. Save objective,
validation/audit, L1, target-weight ESS/max share, selected family and fit time.
Poor coverage is explanatory and a promotion concern, not permission to retune.

Numerical seeds: fit 917000+10000*d+100*sequence_index, time offsets as in A04;
train/validation/audit streams add 0/100000/200000. Baseline scalar/pair seeds
add 1000000/2000000/3000000 to the fit base. Particle seeds
918000+10000*d+100*sequence_index+10*replicate (0–3), shared across methods;
sharing seeds aids pairing but need not produce identical draws. Reference
seeds are separate: 919000+10000*d+100*sequence_index+10*replicate.

## References, primary criterion and uncertainty

Reference d1: physical-grid filtering at 801 and 1201 nodes on the existing
12-stationary-SD domain. Require max normalized mean gap <=1e-6 and log-evidence
gap <=1e-6. This is a deterministic independent reference check, not TT evidence.
Reference d4: four bootstrap runs at Nref=32768, plus four independent runs at
65536. Use the higher-count reference, record both and their difference.
Require every reference mean MCSE <=.02 stationary SD, log-evidence MCSE <=.10,
and low/high mean differences <=3.182446 times their combined MCSE +.01 SD,
with the analogous log-evidence allowance .10. If this precision screen fails,
one predeclared enlargement to four runs at 131072 is allowed and must compare
against 65536 under the same screen. Failure remains reference-inconclusive;
no method ranking for that scope. These screens assess resolution/MC precision,
not exact high-dimensional filtering or all bootstrap bias.

Primary error for each method/sequence is mean squared filtering-mean error,
normalized per coordinate by P0[j,j], averaged over its four particle replicates.
Report all states/times and three conditional classes: |y_j|/beta<=.5,
.5<|y_j|/beta<2, and >=2. A class needs at least eight of twelve sequences and
at least 24 coordinate-times overall; otherwise its inference is inconclusive.
For a class, use only sequences containing that class, equally weighted.

The paired unit is the independent observation sequence, never particles,
coordinates, time steps or fitting-row designs. Use 9999 sequence bootstrap
resamples (seed 920000), keeping all methods/replicates/regimes together.
Construct simultaneous max-standardized-deviation 95% intervals across all
candidate-minus-heuristic contrasts, both dimensions and all/three regimes.
Use the observed sequence SE as fixed scale; zero-SE contrasts use an exact
point interval and are explicitly labeled. Require interval half-width <=.01
normalized MSE for a precise comparison. Twelve sequences can be insufficient;
this is an explicitly finite-sample bootstrap approximation, not guaranteed
coverage or publication-grade certification.

Candidate advancement requires accurate references, adequate regime coverage,
all simultaneous upper contrast limits <=0 and the precision target, plus no
validity/heuristic veto. A negative upper limit supports a ranking only for
that declared metric/scope under the bootstrap assumptions. Any positive
observed candidate-minus-heuristic mean is a heuristic promotion veto even
without significance. Failing a precision requirement is inconclusive, not
equivalence. Report evidence bias/MCSE, conditional mean error, ESS, runtime and
row diagnostics separately; none replaces the primary error comparison.

## Defaults, pre-mortem, vetoes and budget

Skeptical pre-execution audit passes: exact conditional density and recursive
retained law are specified, the ladder includes inexpensive Gaussian methods,
and no fitting/ESS proxy can promote the filter. The .2/.8 row mixture is an
inherited A04 hypothesis: it bounds rho/s by 5 while allocating most rows to
the joint guide. Its demonstrated late-tail weakness is recorded through
target-row concentration; the number is frozen, not treated as optimal.

The inherited SV parameters, T20/N512, degree/rank, row counts and solver
schedule preserve the previous comparison scope. They are baselines with
limited capacity/optimization evidence. A05 supplies the safety rationale for
1e-5; no primary-error tuning of that value occurs. SGQF closure, L1 grid and
minimum validation selection remain empirical approximation hypotheses.
Fresh per-step validation implements the frozen rule, not post-hoc control
selection. Reference counts and interval precision are explicit budgeted
resolution hypotheses; fail inconclusive if they are inadequate.

Pre-mortem: aggregate improvement may conceal a loss in large or near-zero
observations, references may miss tails, or audit may reveal a selector mistake.
Conditional tables, independent reference scales and immutable selections
expose these. The strongest remaining risk is inadequate sequence/MC precision.

Continuation vetoes: source/data corruption, wrong density or target, missing
required evidence, nonlocal implementation invalidity, exhausted budget.
Candidate-specific nonfinite/SPD/CDF failures are recorded as invalid candidates;
continue independent remaining arms/sequences when their computation is valid.
No fallback silently changes a failed method. A shared guide failure invalidates
that sequence for guided methods and blocks their promotion, not bootstrap
reference collection. Preserve every failure, and do not retune on holdout data.

A06 active-work ceiling 6000 seconds (includes design/review/code/checks/results),
numerical ceiling 2700 seconds total, at most three launch attempts. Require
at least 1800 overall seconds remaining for phase-10 closeout. Per-launch
internal limit 2400 seconds, external 2500; charge all attempts against totals.
Pilot is one bounded mechanics test on synthetic data, not holdout tuning.
If measured cost prevents completing the ladder, record under-budgeted rather
than reduce methods, sequences, gates or reference accuracy after seeing data.

After review, implementation and focused checks, run:

    bash docs/plans/artifacts/observation-tt-independent-filtering-20260915-01/run_campaign.sh attempt-01

The wrapper uses the tftwogpu environment, float64 RTX 5080 by UUID, verified
memory growth and stable GPU/XLA numerical kernels. One-time chart/projection
setup, scalar grid reference and post-run TensorFlow/stdlib summaries are
explicit reference/reporting exceptions. No NumPy numerical runtime or pfor.
The versioned output is
`docs/benchmarks/artifacts/observation_tt_independent_filtering_20260915/attempt-01/`.
Save command, Git/source/input hashes, controls before data, observations/seeds,
all proposal selections/cores, filter/reference records, per-stage elapsed time,
GPU/environment/allocator fields, plan/result paths and stop status. Serious
result notes include decision/inference tables and candidate-vs-direction
rejection. Phase 10 updates the master, checkpoint, budget and manuscript.
