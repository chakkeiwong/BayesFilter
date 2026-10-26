# Diagnostic extension: current evidence and next execution

Updated 26 October 2026, Asia/Shanghai. This note supersedes the previous no-allocation disposition,
not the preserved failures. The added 24 hours plus inherited 3539.096492208382
seconds form one combined 89939.09649220838-second allocation. Separate SIR
funds remain untouched. Exact charges and manifests are in
artifacts/modern-hmc-implementation-2026-10-07/N5/diagnostic-extension-20261025-01.

## Terminal update — 26 October 2026, 15:52 Asia/Shanghai

Qualification-03 stopped at 11:37:22 with shared source invalidity, not a timeout
or budget exhaustion. Work43 records execution source changed for
bayesfilter/inference/hmc_candidate_set_execution.py. That module and
hmc_candidate_set_tuning.py have modified timestamps at 11:36:41 and differ
from the launch hashes. Both are real dependencies; their changes cannot be
excluded or silently incorporated into the original binding. No worker is active.

The actual inventory is **981 unique completed trials and 981 charges**, with
42 cumulative observations and no missing returned chunks. Earlier counts of
1,557 and the terminal cumulative sum of 1,717 include repeated prefixes across
evidence rungs. Those reported totals were wrong as counts of executed trials.
The corrected progress code counts each work/trial interval once. All raw
chunks and charged work are preserved; this was a reporting defect, not missing
samples. The original failed attempt remains immutable.

Three candidates previously had verified states, but final shared invalidity
cleared the authoritative verified set. **Zero members are authorized for
export from this attempt.** Its failures remain diagnostic evidence; neither
posterior validity nor a qualified consumer handoff is established. This does
not reject the numerical isolation repair or the research direction.

The final wrapper raised incomplete independent trial coverage because its
native audit rejected the correctly recorded five-of-sixteen early hard veto.
This masked the prior source failure. A localized repair now validates that
stop using the existing numerical evidence validator, saves controller status
before postprocessing, and refuses export after shared invalidity. Fifteen
focused CPU regressions passed in 17.10085040400736 inclusive seconds. Tests
cover real modern-kernel trial execution with an injected health veto, missing
ordinary trials, malformed stop metadata, cost/health/schema/analysis tampering,
unique progress counts and preservation of source failure. They are engineering
checks only and do not rehabilitate the failed campaign.

The supervisor consumed 36449.98963719199 seconds. After settlement and repair
tests, **51604.53847091534 seconds remain**, with no live reservation. Separate
SIR funds are unchanged. The previous deadline is retired with the stopped
attempt. Next requires isolated execution sources and a reviewed full-cohort
price: the previous 87768.77-second estimate exceeds remaining funds. A plain
resource resume is unavailable for this source-invalid checkpoint. Do not
weaken its source checks, use favorable seeds or overwrite another task's edits.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve qualification-03 as failed diagnostic evidence | Final qualification incomplete | Bound-source drift invalidates handoff; earlier candidate health failures preserved | Exact original runtime reproducibility | Isolate sources and review a funded fresh run or exact-source recovery | Three intermediate states are not export authority |
| Accept localized reporting repair | 15 focused checks passed; saved 981-trial inventory reconciles | Corrupted/unsupported inventories still rejected | Broader campaign behavior remains untested after repair | Use repaired diagnostics in a source-isolated next attempt | Target qualification, posterior validity or default readiness |

No statistically supported ranking is available. Intermediate acceptance and
runtime differences remain descriptive. Default readiness remains unestablished.
The strongest alternative to a statistical failure is now directly evidenced:
concurrent source edits stopped the campaign. A fresh, completely source-isolated
run would separate this infrastructure failure from candidate qualification.

## Historical live status — 26 October 2026, 10:44 Asia/Shanghai

The following live review predates the source failure. Its 1,557 trial total is
the cumulative-observation reporting error corrected above.


Qualification-03 remains active. Its fresh checkpoint contains 36 observations
and 1,557 completed trials. Three nominees have completed measurement and
verification and are currently verified: `L=3/.4`, `L=5/.4`, and `L=25/.2`.
The two `.8` nominees are promotion failures under their recorded health
vetoes. `L=9/.4` and `L=13/.4` are validating; `L=18/.4` and `L=25/.4` are
screened. Four work items remain. The checkpoint heartbeat is fresh, the worker
has no result file yet, and all 741 bound source hashes match. The run is
healthy and progressing, but cohort qualification, export/reload and final
settlement are unfinished.

## Completed findings

- Twelve full-horizon root GPU/XLA prices passed their numerical screens.
- The existing controller completed 512 known-law searches across eight cells.
  All six families were delivered in 64/64 interior, rare-two-sided and
  shared-start searches; heterogeneous-valid starts delivered 0/64 and
  persistence .995 delivered 19/64. No false-verified search was observed, but
  the simultaneous upper limit .1136 is too weak to certify rare-event control.
  These are mechanism/power diagnostics, not target-power evidence.
- The existing discarded proposer retained nine healthy pairs across all six
  L values. Short pilot health is not full-horizon qualification.
- Nine further full-trial prices measured the previously unpriced epsilons
  and separated cold compilation from warm execution. L5/.8 failed on
  chain 2, transition 56, leapfrog substep 5. Its finite proposal was outside
  the numerically supported chart: the transition rounds to a unit root.
  CPU saved-point replay confirmed an invalid row, with finite healthy
  companions matching duplicate and permuted calls. This supports a
  candidate-specific numerical-range veto, not a recurrence of cross-row
  contamination. L3/.8 and L5/.8 also triggered movement vetoes.

All nine nominees remain recorded. Earlier health vetoes remain binding even
if a later independent stream happens to pass. The complete frozen cohort is
priced at 57752.51424665749 warm seconds, or 87768.77136998624 seconds including
the inherited 1.5 operational margin and 1140 seconds of build/export/closeout
reserves. This is an estimate, not a runtime guarantee. A frozen nine-member
cohort uses statistical cap 24 without increasing its error allowance; optional
adaptive expansion is a separate, still-open phase.

## Historical running qualification and continuation review

Qualification-03 launched on 26 October at 01:29:52 Asia/Shanghai, using GPU1,
TensorFlow 2.20.0/TFP 0.25.0, float64, XLA, TF32 disabled and verified memory
growth. It is a direct diagnostic composition, not an admitted public
MacroFinance route. At 02:02:31 local, the checkpoint held four measurement
observations (53 completed trials) plus two persisted trials for L9/.4. L3/.4
and L5/.4 were inconclusive at 16 trials. L3/.8 failed movement; L5/.8 stopped
after five trials with numerical-health and movement vetoes. Other candidates
continued. No candidate was yet verified. The later failures of the two .8
pairs are consistent with their earlier vetoes; they do not justify another
seed search for those pairs.

The subsequent CPU saved-trace inspection identified the first invalid proposal
at zero-based transition34, chain2 (the third chain), leapfrog substep5/stage4.
Its position and incoming momentum were finite, but chart support, target value
and target score failed. The other three chains retained finite proposal
values and log-acceptance ratios; all four chains' recorded final target values
remained finite. The controller recorded five completed trials, one invalid
trial, eleven unstarted trials and no attempted-but-unreturned trial. These
observations support local rejection and truthful accounting. The inspection
does not reproduce the target or establish duplicate/permutation parity for
this new point. Its tiny relative Lyapunov residual does not override failed
chart support. Exact values, saved evidence hash and seed are preserved in
qualification-03-failure-localization.json; inclusive CPU cost was
2.6217534620082006 seconds. GPUs were deliberately hidden for this decoding.

Qualification-01 and qualification-02 consumed 340.16724209999666 and
205.15157336101402 inclusive seconds before failing closed on concurrent edits
to the unrelated q20 helper. Both checkpoints and negative findings remain;
neither supplies qualification evidence to qualification-03. A direct-reference
audit supported excluding that single helper from the fresh binding. This
audit did not establish a complete dynamic or transitive import graph. Its
observed hash is recorded separately; all 741 bound paths matched at
continuation review. Further collisions should trigger isolated source
execution instead of a growing exclusion list. The post-run repair note
qualification-01/source-drift-repair.json was subsequently revised; it is not
an immutable record of the first attempt. Numerical evidence and original
attempt manifests/checkpoints were preserved.

The final focused suite passed 52 tests. The CPU/XLA integration completed
464 independent trials, verified all nine fixture candidates, and exported and
reloaded all nine with unique seeds and statistical cap 24 unchanged. A preceding
CPU non-XLA fixture timed out and remains a preserved engineering result; it
does not count as target invalidity or as the successful integration.

Accounting review restored 81.25 seconds omitted from the previous settlement
(80.92-second failed validator suite plus .17/.16-second focused checks).
After the additional saved-trace inspection and before charging qualification-03,
88071.62895851134 seconds remain; 88000 are reserved for its inclusive
supervisor cap. Only 71.6289585113409 seconds are unreserved. Live elapsed time
is inside that reservation, not a second charge.
The supervisor deadline is 27 October 01:56:32 Asia/Shanghai. Export/reload
and closeout reserves are inside the cap. Separate SIR funds remain untouched.

Warm L3 trials now take about 27–28 seconds, versus about 8.5–8.7 seconds in
pricing. Trusted inspection shows a second Python process sharing GPU1, ample
free device memory, and an advancing worker. Contention is a plausible
contributor, not a demonstrated sole cause. Sustained timing above the inherited
1.5 margin creates a risk of incomplete maximum-rung evaluation; it does not
invalidate completed numerical evidence or authorize a larger budget. Continue
under the fixed cap, preserve progress and assess actual remaining work as
candidates resolve. The frozen cohort must not be reduced retroactively.

Skeptical continuation review passes for the unchanged run: the comparator,
target, data, evidence thresholds and all nominees are preserved; short pilots
and observed timings do not qualify candidates. Progress and source checks are
healthy. A live partial_budget/interrupted checkpoint label is not a terminal
stop. Candidate rejection triggers continued evaluation of the other nominees;
source drift, corrupted evidence or exhausted allocation remains a continuation
stop. The detached supervisor survives editor exit but does not provide
automatic machine-reboot recovery.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Numerical isolation repair remains supported | Saved-point companion parity passes | One large-step pair invalid; isolated | Other extreme regions untested | Preserve failure; continue other nominees | Universal numerical robustness |
| Continue all nine frozen nominees | Qualification-03 running; integration passed | Both .8 pairs fail current health screens; earlier vetoes retained | Actual power and slower shared-device runtime | Continue other nominees; audit final export/reload within cap | Cohort completion, ranking or adaptive-domain completion |
| Leave external admission open | Direct consumer diagnostics only | Historical clean-source pin rejects public entry | Successor admission policy/source | Resolve separately from numerical evidence | Public consumer readiness |

## Inference status

| Evidence role | Status |
|---|---|
| Hard veto screen | L5/.8 has nonfinite target/proposal evidence; both .8 pairs have observed movement vetoes |
| Statistically supported ranking | None |
| Descriptive-only differences | Trial timings, proposal acceptances and per-cell delivery frequencies |
| Default readiness | Not established; direct diagnostic composition only |
| Next evidence needed | Complete fresh cohort measurement/verification, all-member export/reload; later posterior/reference checks |

The strongest alternative explanation is that short proposals missed rare
candidate failures, as the longer price already demonstrated. Healthy root
prices cannot establish long-run target health. A fresh collateral-row failure
would reopen the implementation repair; candidate-local failures do not justify
abandoning the other nominees. The weakest evidence is target-specific power:
known-law results cannot certify the MacroFinance acceptance-score law.
