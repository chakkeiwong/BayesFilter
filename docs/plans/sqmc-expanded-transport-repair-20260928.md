# Expanded SQMC transport-iteration repair

The original two-flow-step calibration exposed invalid candidates, not a
broken oracle or infrastructure. CPU diagnostic localization suggested a reset mass failure, but its regenerated
inputs did not reproduce the saved GPU case. Repair-check-01 observed GPU
column TV 1.280e-3 above the unchanged 1e-4 guard; the next check must verify
saved observations/input hashes and compare backends on identical tensors.
Higher-moment validity, finite states/tangents and positive covariance were
healthy in that trace. This sufficiently explains rejection; it does not
establish that every rejected case has the same cause.

## Question and bounded repair

Can exact-scope calibration of terminal transport balancing produce valid
full-model comparisons without relaxing guards or changing the algorithm?
Keep all six full-A/full-SPD-Q scopes, four routes, N1020, FP64 GPU/XLA,
TF32 off, every score coordinate and exact Kalman comparator. Retain the
completed P44 comparisons unchanged. Original failed candidates remain
separate evidence. No production, default, HMC or superiority claim follows.

For each full scope/route, nominate the smallest balance count in 12,48,192
that passes the shared finite-program validity check and has maximum reset
column TV <=1e-6 and row error <=1e-8 on both fresh calibration seeds.
Use flow 8 for this scalar nomination. The stricter nomination margin is
100-fold inside the existing column guard; neither guard is relaxed.
Nomination uses one analytical direction and full observation horizon only
to expose transport validity, not to establish score accuracy. Save computed
transport, higher-moment and program diagnostics from the shared trace.

With the nominated balance count, run the existing complete-score flow grid
2 versus 8, validate the selected controls, then freeze them for untouched
final evaluation. All other controls and the numerical implementation are
unchanged. A failed nomination or full-score validation remains a candidate
failure; do not evaluate final cells with an unissued tuning artifact.

Use calibration seeds 199001,199002; validation 200001; final data 201001,
201002 crossed with filter seeds 202001,202002. These are disjoint from the
original partitions. All four routes share observations within each scope.
Preserved P44 results use the original partitions and must be labeled so.

## Evidence roles and assumptions

Primary engineering criterion: complete all coordinates with exact-scope
issued controls and intact observations/oracle provenance. Primary scientific
measurements remain raw likelihood/score and absolute error, not a threshold
or ranking. Guard failures veto the affected candidate. Wrong data, broken
oracle or derivative, unsafe resources, exhausted budget or deadline veto
continuation. Heuristic checks remain conditional promotion vetoes only.

Balance 12 is the rejected original baseline. Counts 48 and 192 are a bounded
geometric numerical-resolution ladder, not transferred optima. Direct
full-horizon calibration supplies target-specific evidence. Fixed epsilon,
ridge, correction caps and damping remain documented hypotheses; this repair
does not establish their adequacy. Flow-8 nomination could miss a flow-2
pathology; the subsequent full-score calibration catches it. Calibration
margin can fail on final data; preserve that failure without tuning on it.

The scalar trace uses the same shared analytical implementation. It cannot
replace all-coordinate calibration or final scores. Maximum column/row error,
higher-moment flags, finite-program validity and wall time are observability
records; numerical results are never altered by the diagnostic.

## Budget, execution and stop conditions

Original 43,200-second aggregate GPU cap and 2026-09-28 07:25:28.470267 UTC
deadline remain. After original run-01, 39,546.277 seconds remain. Charge all
new GPU wall time, including probes, compilation, rejected candidates and
retries. At most two infrastructure retries per unit; a planned stop for this
numerical repair is recorded separately from an infrastructure crash.

Before the repair ladder, use <=1,200 GPU seconds for trace compatibility and
worst-resolution full-d10 timing. Compare CPU/GPU guard localization on identical saved-input tensors from the
failed original case. Project remaining six full scopes at balance 192 with a
50% margin; launch only if the projection fits. Preserve unique repair-check
and repair-run directories under the existing renewal root. Per-worker
90-minute limits remain. No package/environment changes or external actions.

Skeptical audit: original fixed balancing was an unproved numerical-resolution
hypothesis and failed its mass-conservation check. Increasing resolution
addresses that check without weakening it. New seed partitions protect
validation/final separation. Final score tuning uses all coordinates; scalar
validity is not promoted to score accuracy. The cost check must precede the
expensive repair. If repair cost does not fit, stop as under-budgeted and
report the rejected comparison rather than silently shrinking the request.

Focused preflight audit: nine CPU control tests passed (2.46 seconds), including disjoint repair partitions, unchanged numerical source closure, original budget checks, and rejection of insufficient mass margin. GPU trace compatibility and repair cost remain untested until repair-check-01 completes.

## Recorded rejection and amended smoothing calibration

Repair-check-02 reproduced the saved GPU observations and random-design hashes.
CPU/non-XLA and GPU/XLA on identical tensors agree: maximum column TV is
0.00128003128469953 at balance12, 0.00120558876340008 at48, and
0.000977795499045041 at192. All fail the unchanged1e-4 validity guard.
This rejects the iteration-only repair, not the research direction. Check01
was a diagnostic-fixture mismatch and consumes one infrastructure retry;
check02 executed correctly and is a rejected numerical candidate.

Source inspection of ledh_unified_reset_tf.py:136 shows both counts feed the
same loop. Lines119–149 use an exponentiated scaled cost and add1e-7 to both
divisors. Thus increased iterations alone need not eliminate a mass residual:
at a fixed point the unnormalized row/column masses equal their targets less
1e-7 times the corresponding scaling vector. Subsequent row normalization
restores rows but need not restore columns. This derivation explains a possible
residual, without proving it is the sole cause of every failed candidate.

Amend the bounded repair to existing epsilon values0.4,1.6,6.4,25.6, keeping
Sinkhorn24+balance12 and all shared numerical source unchanged. These form a
predeclared geometric conditioning ladder: larger epsilon reduces exponent
contrast; it changes the finite transport and is explicitly a tuned setting,
not an implementation-equivalence claim. Nominate the smallest epsilon passing
both full-horizon calibration seeds with the existing1e-6 column margin.
Preserve epsilon0.4 exactly where it qualifies. The endpoint25.6 is a bounded
large-smoothing adversary, not a recommended default. It could lose useful
particle structure; all-coordinate calibration, untouched Kalman comparisons
and conditional heuristic vetoes will expose that risk. No primary-error
improvement is required for this numerical-validity nomination.

Before launch, the amended check must reproduce the failed fixture, test the
largest epsilon through full-d10 T120, and measure full-score cost. Budget
projection includes at most eight scalar calibration evaluations per unit and
a50% margin. Fresh partitions199/200/201/202 remain unused for evaluation.
All other evidence roles, six full scopes, four routes, disjoint partitions,
aggregate budget, deadline and infrastructure-retry cap remain unchanged.
Skeptical audit: the previous premise of separate balancing stages was wrong;
this amendment addresses kernel conditioning with an existing method control,
keeps the exact target/oracle, and tests validity before a long comparison.

Check03 completed as a numerical-candidate test in41.39GPU seconds. Epsilon1.6
and6.4 passed the original guard but failed the predeclared nomination margin;
25.6 also missed that margin. Extend the geometric ladder by one endpoint102.4,
keeping the1e-6 margin unchanged. The diminishing residual across the measured
ladder motivates this bounded endpoint, but does not prove it will qualify.
Do not repeat intermediate checked fixtures: check04 compares the original0.4
and new102.4 endpoint, then attempts the full-horizon and full-score cost checks.
Projection now charges at most ten scalar nominations per unit. If the new
endpoint fails, do not expand epsilon again without reassessing this strategy.

Check04 passed numerical validity: full-d3 failed fixture column TV4.092e-7,
full-d10 T120 maximum1.050e-7 at epsilon102.4. Its conservative projection
50,916seconds exceeds remaining39,265seconds, so the ladder was not launched.
The cost model charged ten cold scalar compilations per unit. Replace static
epsilon in diagnostic nomination only with a scalar float32 graph input,
matching the existing tf.cast(Python-float,float64) rounding before its use.
Check05 must establish identical static/dynamic outputs within1e-10, one trace
across epsilon values, and cold/warm full-horizon timing. Shared numerical
source and all-score kernels are unchanged; reuse checked full-score timings
only after numerical dependency hashes match.

Revised projection follows actual work: one scalar cold plus nine warm calls
per unit; two full-score calibration calls at flow2, two at flow8, and five
validation/final calls at the slower flow8 bound; two cold full-score charges
remain conservative extras. Keep50% margin, all scopes/coordinates/partitions,
and strict launch-if-fits rule. This is an implementation/cost repair, not a
smaller statistical design or a relaxed margin.

Prelaunch result: check05 passed static/dynamic diagnostic parity within1e-10
and exactly one trace per scope. Full-d10 T120 scalar timing39.874s cold,
3.273s warm. Reused full-score timings have matching numerical dependency
hashes. Projection23,849.069s; with50% margin35,773.603s, below remaining
39,201.966GPU seconds. Aggregate charged3,998.034s. The complete six-scope
repair is authorized under the unchanged original cap and deadline.
