# Qualification criteria review, 26 September 2026

Verdict: REVISE BEFORE REUSE. The selector correctly implements the recorded
screen, but that screen does not establish whether an iAPF reconstruction is
faithful or scientifically successful. No candidate is newly promoted by this
review. The numerical results and original decision remain preserved.

The review traced the live consumer to its frozen executed source, reconstructed
all 266 row decisions and all 14 policy decisions from saved CSV fields, and
verified exact agreement. This checks selection wiring, not numerical accuracy.
The [executable audit result](qualification-review-20260926.json) preserves the
counts and outcome. No new experiment ran and no GPU budget was used.

## What the program actually required

1. Status exact_start or converged, with no numerical bound, local bound,
   escape or underflow flag.
2. For exact diagonal Gaussian fixtures: training shape error at most 1e-12
   and maximum mean/log-variance parameter error at most 1e-5.
3. In every one of nine generated fixtures: held-out shape error no greater
   than the best QR, target-weighted-moment or available diagonal-Laplace
   baseline plus 1e-8.
4. In all ten saved failure inputs: the numerical condition in item 1.
5. Full-filter work additionally required confirmation and a justified floor.
   The floor study used center base-mixture probability at most 1/N and later
   reported conditional proposal losses as promotion vetoes.

Items 1–4 are implemented in
[fit selection](/home/chakwong/BayesFilter/docs/benchmarks/diagnose_iapf_choice_calibration_reference.R:229)
and [policy aggregation](/home/chakwong/BayesFilter/docs/benchmarks/diagnose_iapf_choice_calibration_reference.R:306).
The all-saved-input requirement was stronger than the plan's explicit wording
about every generated class. It must be stated and justified if reused.

## What is sensible and what is not

Derivative checks, exact normalization, finite probabilities, positive
covariances, explicit failure accounting and separation of calibration from
confirmation are appropriate. The absolute numeric tolerances are local
engineering hypotheses, not source-established replication thresholds.

Exact-start preservation is also useful, but an explicit early return protects
the exact QR fixtures. It tests no-harm at an exact start, not optimizer recovery.
Nearby perturbed exact starts and conditioned cases are required for that claim.

The global shape-dominance rule is too strong as a replication prerequisite.
For density samples p and positive target samples y, the code's shape statistic is
\[
S(p,y)=1-\frac{(p^\top y)^2}{(p^\top p)(y^\top y)}.
\]
The profiled paper objective is instead
\[
L^*(p,y)=p^\top p-\frac{(p^\top y)^2}{y^\top y}
       =(p^\top p)S(p,y).
\]
Reducing L* therefore need not reduce S. The statistic is valuable for exposing
amplitude escape, but dominating it on independent points is an extra empirical
condition, not mathematical correctness of Equation (15). The best baseline is
also chosen separately on each held-out fixture; it is an envelope, not one
deployable comparator. The 1e-8 slack is a numeric tolerance, not uncertainty
over generated clouds. Only one cloud pair was used per generated case.

Correlated Gaussian cases are relevant to the paper's immediate linear-Gaussian
experiment, but these standard-normal clouds are not actual backward clouds
from that filter. Broad-tail and bimodal cases are additional stress tests.
Failure there must remain visible and blocks unsupported generality claims;
it does not alone invalidate a Section 5.2 reconstruction or prohibit a bounded
diagnostic pilot. The meaningful heuristic-dominance check must still be made
conditionally on the actual filtering task before a performance promotion.

Conversely, the numerical rule is too weak: it trusts convergence flags that
the terminal audit showed can stop on small absolute gradients after density
collapse. The escape threshold log-density change below -20 is heuristic, and
the underflow flag tests only zero loss with nonzero shape. Passing these flags
is not proof of stationarity. A revised rule must calibrate scale-aware
stationarity for the declared problem and distinguish safe rejection of a
pathological input from a successful solution.

The floor assessment mixed protection with proposal quality. In the centered
underdispersed example, a 1/N mixture floor reduces an infinite second moment
to 14.42, while the original transition achieves 1.15. Thus the floor supplies
tail protection and the combined proposal remains poor. The second conclusion
does not invalidate the first or isolate the floor as the cause of poor quality.
Class C protection should be judged for non-harm in healthy cases and bounded,
reported behavior in pathological cases; complete-proposal quality remains a
separate downstream veto. The 1/N center rule has an expected-component-count
interpretation but is not an author rule or calibrated acceptable variance
increase. It cannot be a universal threshold.

The inspected paper Section 5.1 specifies diagonal density least squares and
positive c, with protection against large or infinite variance as its purpose.
It does not specify these local numerical, shape-dominance or mixture thresholds.
Source: [local text](/home/chakwong/BayesFilter/.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.txt:911).

## Corrected decision and next work

| Layer | Correct criterion | Current evidence |
| --- | --- | --- |
| Source fidelity | Same equations, model and controller; numerical reconstruction choices disclosed | Optimizer/floor provenance unresolved |
| Numerical validity | Derivatives/normalizers, perturbed exact recovery, calibrated stationarity and explicit collapse/failure handling | Several controls pass; stopping needs repair |
| Protection | Floor non-harm and tail protection assessed separately from fit quality | Protection demonstrated locally; no global choice justified |
| Filter usefulness | Predeclared Kalman-relative likelihood measures, failure rates, cost and conditional simple-filter comparisons on the actual study | Not evaluated by this fixture study |
| Replication | Complete source-matched experiment accounting and uncertainty, with source gaps disclosed | Incomplete |

The correct statement is: none passed the original screening rule. It is
unsupported to infer that no tested choice could contribute to a faithful or
useful iAPF reconstruction. Nevertheless, amplitude escape, misleading stopping
and the reported local proposal losses remain real findings.

Revise the evidence contract before using it to block or admit the next run.
Do not retroactively change the old tests to manufacture a pass. Preserve all
stress failures, avoid tuning on heuristic challenges, and continue bounded
repair/diagnostic work when no genuine numerical continuation veto applies.
