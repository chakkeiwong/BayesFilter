# N5 D4: method-specific proposals, qualification, and posterior delivery

This is the active continuation of the [master program](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md)
and its [48-hour supplement](bayesfilter-modern-hmc-n5-additional-48h-2026-10-10.md).
It replaces D4's previously unspecified full-fit study. No new campaign grant is
requested. NUTS is excluded from every executable phase and denominator.

**Current stage, 11 October:** D4a finished all eight cells and sixty
hypotheses. D4b1's first full SGQF cohort is terminal. All six SNAPER and all
nine MAMS proposals received fresh tuning verification and export/reload.
SNAPER delivered 6/6 posterior and independent reference screens. MAMS
delivered 5/9; one member had inconclusive warm-up and no retained draws, and
three reached the existing 10,000-draw cap without meeting precision. A
post-run reference diagnostic crash on the empty archive was repaired and
reassessed without rerunning sampling; see the
[repair record](bayesfilter-modern-hmc-d4b-reference-repair-2026-10-11.md).
The original queue remains a partial historical result. A fresh 20-cell matrix
snapshot and conservative budget ledger are in `N5/method-reference-repair-01/`.
After charging the repair, 43.98 campaign hours remain after overhead reserve.
The other 45 D4a proposals still await scoped full-model cohorts. The remaining
grant cannot be assumed sufficient for all 45 based on this cohort's measured
cost; choose the next cohort under a refreshed, skeptical budget audit.

## Settled starting point

D2f recovered UKF randomized L=9 without retuning. D3 completed six modern
price cells and twelve healthy short paths. E3 is terminal: all fifteen
structural quantities pass the unchanged .002 reference-SD empirical refinement
criteria. Maximum half-width/SD is .0018873181816258845; maximum last-order
change/SD is .0014673134087065422. This is a reference for the full 100-observation,
five-parameter finite UKF posterior, not exact nonlinear-generative inference.
The failed old reference and all earlier sampler failures remain evidence.

The settled observer `N5/supplement-watch-04/budget-ledger-live.json` charges
193403.4474935018 seconds against 432000, with no active reservations. After the
existing 600-second overhead reserve, 237996.5525064982 seconds (66.11 hours)
remain. Tests and failed attempts count. Serial parent/children are charged once;
parallel parent durations add. Passive capacity waiting is reported separately.

## Research intent and evidence contract

Question: do target-specific trajectory hypotheses avoid the under-explored
duration/step-size regimes in the existing full nonlinear comparison, and can
their freshly verified members deliver the declared posterior quantities?
Mechanisms: discarded fixed-geometry SNAPER adaptation and a native MAMS
step-size/duration neighborhood. Baselines: the recorded 32-step SNAPER pilots
(all L=1), HMC-transferred MAMS L=3/9, epsilon=.2 prices, and the completed fixed
and randomized HMC fits. Historical fits are descriptive comparators, not fresh
paired performance evidence or adequately searched classical baselines.

| Diagnostic | Role and consequence |
| --- | --- |
| Exact target/data/source, valid native trace and full internal filter status | Engineering pass criterion; corrupted identity or changed computation stops the affected cell and dependent comparison |
| Nonfinite state/value/score, filter failure, motion failure | Hard numerical veto for that candidate; retain its record and continue valid peers |
| Fresh replicated acceptance measurement and verification | Only tuning membership criterion, with existing numerical/movement vetoes |
| R-hat, bulk/tail ESS, lugsail MCSE, reference agreement | Separate posterior delivery criteria and promotion vetoes; never tuning membership, ordering, repair, or search stopping rules |
| Short acceptance, duration, gradient work, runtime, pilot objective | Explanatory prices/proposal diagnostics only; cannot establish mixing, qualification, or ranking |
| Wall/attempt budget exhaustion | Resource stop preserving partial evidence; repair trigger, not mathematical failure |

Pilot failure is not failure of randomized HMC or of the research direction.
Candidate numerical failure does not invalidate a different candidate. Invalid
target, reference, source or checkpoint stops dependent execution. No short price
establishes adaptation adequacy, exact posterior correctness, superiority,
optional release, or default readiness. Keep engineering correctness, tuning
membership, posterior delivery, and comparative interpretation as distinct ledgers.

## D4a — executable proposal/duration pricing

Eight cells: SNAPER and MAMS on full NL-PP-SGQF, NL-PP-UKF, NL-SIR and NL-STR.
Use the saved local factors, four starts and full datasets; neither proposal
fitting nor sampler geometry changes. Seed 4300 is a fresh convenience-selected
block, shared across methods; all derived stateless streams are recorded. Keep
every generated, omitted, failed and unstarted hypothesis. This phase issues no
tuning artifact. Prices reset to the original bound starts, not pilot endpoints.

SNAPER uses 1000 discarded adaptation steps, .2 initial epsilon, .8 adaptation
acceptance target, max32 steps, zero MALA, trajectory learning rate .05, EMA
factors8, epsilon factors(.8,1), and duration factors(.5,1,2). These settings are
explicit hypotheses inherited from the successful *small-target* mechanics
consumer, not validated nonlinear defaults. The duration neighborhood is a new
factor-two exploration hypothesis. Record deduplication and support omissions;
an all-omitted proposal fails its cell honestly, without clipping epsilon or L.
TFP's shared IID pilot jitter differs from the frozen independent per-chain IID
law. Pilot status checks cover accepted states; frozen native prices restore
the full per-chain internal-step status coverage. Every pilot draw is discarded.

MAMS prices the Cartesian neighborhood L=(3,9,18), epsilon=(.2,.5,.8), nine
hypotheses. L3/9 and epsilon.2 are inherited warm starts from D3, not method
tuning; L18 doubles the upper duration scale. Epsilon.5/.8 are convenience
exploration points within the already declared [.001,1] support, not assertions
about optimal MAMS scaling. A healthy path retains a hypothesis for later fresh
tuning, not a qualification label. The broad grid tests sensitivity to both
controls rather than assuming an HMC epsilon transfers.

Each hypothesis receives 32 transitions/chain with four chains, inherited D3
pricing lengths (insufficient for posterior inference). Durable native chunks
preserve realized lengths, work and health. Explicit shapes/signatures and XLA
come from the shared numerical authorities; no sampler or tuner is copied.
Reference preflight verifies evidence identity only; reference values do not
enter proposals, trajectories or membership. Structural quantity/data wiring
must validate dimension5/horizon100, not masquerade as a twenty-observation fit.

Reserve 36060 occupied seconds for the serial queue (36000 hard queue ceiling,
60 cleanup allowance) and 600 for focused regressions. Each SNAPER cell has a
7200-second ceiling; each MAMS cell1800. These are convenience ceilings inside
the authorized grant, not deadlock tests or predicted completion times. D3
measured 32-step pilot durations were18–23 seconds and native L9 prices13–35
seconds; longer pilot/duration costs remain uncertain. The queue records active
cell/candidate and durable progress. It does not kill healthy work for a missing
heartbeat during compilation. A ceiling expiry triggers workload/cost diagnosis;
a later extension needs a fresh allocation from the same remaining budget.

Commands (from the repository, `tfgpu` environment):

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_modern_hmc_method_proposals.py tests/test_hmc_trajectory_proposals.py tests/test_modern_hmc_full_comparator_prices.py
/home/ubuntu/.codex/bin/codex-gpu-probe --framework nvidia --gpu auto
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/price_modern_hmc_method_proposals.py --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/method-proposals-02 --gpu 1
```

GPU commands run in trusted/escalated context, with growth set before import and
verified at runtime. Manifests record source hashes, commit, actual commands,
environment, full target/data identity, seeds, GPU/XLA/allocator and elapsed time.
Device selection is refreshed automatically before **each** cell, following
the installed trusted probe's GPU1 preference. The initial GPU is only the
fallback for the existing shared-memory admission when no permitted device is
idle; driver/visibility failures remain vetoes. Each row records its actual
device and probe. Changing between identical GPU hardware classes is resource
recovery, not a new scientific arm; runtime observations remain descriptive.
Prior outputs are immutable. Localized harness/resource retries use new output
roots and consume the same grant. No automatic new scientific method is launched.

## D4b — refresh from terminal prices, then public qualification and delivery

Before launching D4b, append its exact cohort, commands, measured-cost model,
caps and audit here; this phase cannot use an unspecified full-fit command.
Start with at most 40 aggregate hours; keep at least 6 hours for repair and
closeout. Use a new consumer capable of variable candidate counts and structural
quantities, without changing historical source closures. Budget-feasible cohort
selection is predeclared by target/method order and cost, not posterior metrics.
Preserve the complete D4a inventory and distinguish outside-cohort proposals
from tuned candidates. Price all hypotheses needed by a cohort before tuning;
do not silently select a speed/ESS winner from short runs.

For every declared cohort, retune its full frozen proposal set with the public
`tune_hmc_kernel`, the typed exact-score binding and a correctly sized replicated
family error allocation. Reset starts; measurement/verification streams are fresh
and disjoint from prices/pilots. Retain/export/reload every verified member,
including higher-acceptance candidates. Report planned, measured, verified,
posterior-started and delivered denominators separately. A single member veto
cannot abandon remaining valid members; genuine common-source/data/reference
failure stops the affected comparison. Wall stops preserve original denominators.

Posterior policy remains the shared controller: warm-up min2000/max10000,
1000-result windows/chunks, retained max10000, modern R-hat, bulk/tail ESS400,
lugsail MCSE/SD.02 on every declared quantity. Independent agreement remains
`(abs(error)+3*MCSE+reference_uncertainty)/reference_SD <= .10`. These inherited
campaign hypotheses are separate from tuning, and may yield honest delivery
failure. Any future precision-budget change must be labeled a new study and
preserve original failures; do not relax checks to produce a pass.

## D4c/F — fair comparison, replication, and closure

Extend classical fixed/randomized duration/epsilon hypotheses under the same
target/geometry and pricing contract before a strong comparison. A narrow weak
classical grid is insufficient for a performance claim. Then freeze a *fresh*
paired-seed replication cohort and uncertainty method within measured remaining
cost. If adequately replicated full nonlinear evidence cannot fit, state that
N5 statistical qualification is under-budgeted; never convert single-seed prices
into superiority or erase missing cells. Refresh the twenty-cell core matrix,
budget, result decision/inference tables and the next phase after every terminal
stage. Keep NL-GROWTH adapter/reference and NL-SV eligibility explicit in N8.

N6 LAPS needs its own source-resolved ensemble contract and budget; N7 MALT and
later branches remain conditional. They are not reasons to postpone repair of
the implemented profiles, nor requirements for scoped N2–N5 optional release.
N8 consumer/registry/reference/book consistency and terminal numerical audit
remain required for any scoped release. No default change is authorized here.

## D4b1 — concrete first full-set qualification and delivery cohort

The predeclared target/method order begins with full twenty-observation
predator–prey SGQF, SNAPER then MAMS. Execute **all six frozen SNAPER hypotheses
and all nine MAMS hypotheses** from `method-proposals-02`, not a representative
winner. This limits the number of funded target/method cells, not the membership
of a tuned candidate set. The other45 priced proposals remain pending for later
cohorts, never failed or silently discarded. Pricing acceptance, R-hat, ESS,
reference means, and posterior outcomes cannot choose the cohort's candidates.

The exact pairs are read from each preserved `epsilon_by_l` and cross-checked
against its complete planned and measured inventory, transition family, full
target/data, geometry hash, source closure and health records. The SNAPER set is
L2:(.6269862024585631,.7837327530732038),
L4:(.6269862024585631,.7837327530732038), L7:.7837327530732038,
L8:.6269862024585631. MAMS uses L3/9/18 × epsilon.2/.5/.8. Every pair is passed
unchanged to the public tuner. New fit seed4400 is a convenience-selected fresh
block shared across methods, with fresh start jitter and purpose-separated
measurement, verification, warm-up and retained streams. No adaptation or price
draw is reused as tuning or posterior evidence. Proposal transfer is within the
same declared target/data/frozen geometry; the new initial positions are recorded.

The exact-score prepared binding uses the existing replicated lower-bound
acceptance contract:64 transitions/trial,32 measurement repetitions and32 fresh
verification repetitions, qualification region[.5,1], and .05 family error
allocations. The candidate-family size is6 or9, not the legacy two-member size.
The controller reserves three units/candidate plus one repair reserve and no
additional candidates; the binding permits no repairs in this frozen diagnostic
cohort. Its work ceiling derives from64×64×4×sum(2×(L+1)) with1.25 inherited
headroom, rather than a two-candidate constant. Qualification, replay and
work accounting use existing public authorities; no new tuner is implemented.

Every verified member is exported and reloaded before any posterior work. The
consumer saves all planned candidates before tuning and preserves every later
disposition. Posterior chunks use the existing controller/policy described
above, with all18 physical/source-square/tail quantities. Member hard vetoes
are retained and valid peers continue. Changed common source/data/reference or
corrupted archives veto the affected cell. A cooperative resource stop saves
committed evidence and unstarted members; it never produces all-member success.
The public tuner's checkpoints and durable posterior chunks preserve completed
work. Consumer callbacks record replay/export/reload progress and a ten-second
runtime-boundary heartbeat (a convenience reporting cadence, not a timeout or
proof of completed transitions), so long compiled calls are not diagnosed as
failed merely from silence.

The independent CPU diagnostic runs only after each terminal fit. It assesses
every saved member using the same lugsail precision and unchanged .10 agreement
screen, including unsuccessful posterior members. Missing/unverified/unstarted
members remain in the diagnostic denominator with explicit dispositions. No
posterior/reference diagnostic changes tuning status or selects the next member.

### Cost, numerical provenance and launch

Short-call prices are descriptive, not runtime guarantees. Summing measured
32-transition native calls gives2.017386 sec/transition for the six SNAPER
members together and4.687961 for the nine MAMS members together, including the first
compile in each set. Multiplying by4096 tuning transitions/member gives about
2.30 and5.33 hours. Adding the previously observed2000 warm-up plus a full10000
retained allowance predicts about9.02 and20.96 hours, before export/diagnostics.
If every member instead needs both10000-count maxima, that rises to about13.50
and31.38 hours. These are workload extrapolations, not assumptions that warm-up
has converged after2000 or authority to shorten the sequential policy.

Allocate43200 seconds to SNAPER and97200 to MAMS (12+27 hours), plus120 seconds
for each independent assessment and60 seconds queue overhead:140700 seconds.
Reserve140760 including60 seconds service cleanup; at most600 additional seconds
for focused mechanics tests. These caps give approximately30% headroom over the
observed-warm-up/max-retained extrapolation and respect D4's40-hour initial ceiling.
They do not guarantee every member can reach both count maxima. Running calls
are cooperatively stopped only at boundaries, with a process-group hard cap as
the final guard. A cap expiry is a resource repair trigger with preserved chunks,
not a mixing verdict or a reason to remove an otherwise verified candidate.
About26 hours remain outside this reservation for recovery, later cohorts,
baseline work and closeout. Do not spend the whole grant on unbounded retries.

Create `scripts/run_modern_hmc_method_fit.py` as a new consumer and
`scripts/diagnose_modern_hmc_method_reference.py` as a separate post-run diagnostic;
keep the hashed D4a and original two-member scripts unchanged. The new consumer
supports variable counts and all four target charts, though this launch admits
only the declared first SGQF cohort. Use:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_modern_hmc_method_fit.py tests/test_modern_hmc_method_proposals.py
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_modern_hmc_method_fit.py --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/method-full-fit-01 --gpu 1
```

Launch the GPU queue through a trusted service with memory growth, automatic
per-cell device selection,144000-second outer wall guard, and existing bounded
child-group cleanup. The manifest records the actual commands, source closure,
scope, observations, seeds, versions, device/growth/XLA and wall time. The
read-only supplement observer tracks the additive ledger; terminal results
refresh the master and next phase. No automatic expansion to another target or
method is authorized by a successful process exit alone.

### D4b1 skeptical audit and acceptance

Question is scoped delivery of the frozen sets through the public lifecycle,
not superiority. The comparison baseline remains the earlier same-target
fixed/randomized fits, descriptive only because seed/search costs differ and a
stronger classical search is unfinished. Hard numerical vetoes come first;
posterior precision/reference failure rejects delivery only. Timing, acceptance
and pilot objectives remain explanatory. This avoids a wrong baseline, proxy
promotion and a weak-baseline ranking claim. The structural pilot's tiny epsilon
(.0030–.0038) is a recorded proposal concern, not a reason to transfer SGQF
settings or drop structural from the matrix.

Tests must exercise >2 public candidates, multiple epsilon values per L,
native export/reload, full quantity shapes, interrupted tuning with the original
denominator, one posterior veto followed by successful peer execution, changed
source/reference rejection, and separate reference assessment with missing or
failed members. Tiny Gaussian fixtures test this control flow; the real full
SGQF fit is the default GPU/XLA integration evidence. Existing reference/chart
corruption tests remain required. Successful process exit cannot establish
posterior validity; final result and all member/reference records must be read.

**Audit disposition:** proceed with this first complete-proposal-set cohort
after focused checks pass. All15 proposals remain represented; every verified
member remains retained. Uncertainty about actual stopping cost is explicit,
funded peers continue after candidate failure, hard ceilings preserve evidence,
and the remaining65-hour budget cannot be misrepresented as enough for every
full nonlinear comparison. No optional release or ranking is inferred from
this one-seed cohort.

## Default audit, pre-mortem, and skeptical review

| Choice/provenance | Failure mode | Earliest check and status |
| --- | --- | --- |
| Saved local curvature, previous geometry work | Locally healthy but globally poor coverage | Starts/data/source parity and later independent posterior quantities; unadmitted geometry proposal |
| 1000-step pilot, small-model inherited settings | Target-specific adaptation may fail or remain L1 | Preserve complete objective/epsilon/duration trace and fresh native prices; hypothesis only |
| MAMS expanded grid, convenience factors | Too broad/expensive or still misses useful duration | Nine short prices and measured work; exploration only |
| 32-transition prices, D3 inheritance | Compile cost dominates; short acceptance hides long-run failure | Separate cold/warm costs, do not promote; pricing convenience |
| 7200/1800-second caps, new convenience allocations | Contention mistaken for algorithm failure | Trusted resource probe, active progress and capped cost record; resource limit only |
| Existing acceptance/precision thresholds | Numerical health may pass without mixing, or posterior fail for precision | Separate fresh tuning and posterior ledgers; inherited campaign contract |
| Full finite-filter independent references | Agreement mistaken for exact nonlinear inference | Target/parity/hash/order checks and explicit finite-target scope |

The pre-run audit rejects the previous implicit assumptions that a 32-step L1
pilot is adequate or HMC epsilon.2 is tuned MAMS. D4a instead tests proposals
without claiming tuning or posterior success. It avoids wrong horizon baselines,
proxy promotion, unfair unqualified rankings, unspecified caps, GPU allocation
mismatch, and mutation of preserved source closures. Missing posterior outcomes
are explicit unfinished work. Focused tests cover corruption, chart and horizon,
variable proposal denominators, reset/profile/status wiring, excluded methods,
candidate-veto continuation and bounded-process cleanup. Full actual-model
GPU/XLA prices provide integration evidence after these mechanics tests.
**Audit: proceed with D4a only after focused checks pass.** D4b must be refreshed
from its terminal evidence before execution. This is a self-audit, not independent
review certification or scientific promotion.

### Localized resource admission repair

Attempt `method-proposals-01` was stopped during initial setup after GPU1 became
idle. The installed probe then forbids explicitly admitting GPU0; a fixed-device
queue would fail its next cell despite spare capacity. The original source is
preserved in `method-study-refresh-01/historical-source/`, and the stopped attempt
has no tuning/posterior outcome. Repair: refresh automatic selection per cell,
record the selected device in command/environment/row, and use only the existing
headroom-checked shared policy if no permitted GPU is idle. Focused tests cover
GPU0-to-GPU1 changes and unexpected probe failures. Restart the same eight-cell
study in `method-proposals-02`; charge the stopped work and both test attempts,
release the old reservation once, and preserve the original attempt. Target,
data, method, geometry, criteria, nominal seeds, hardware class and cumulative
budget do not change. New source-bound streams and closure are recorded; no
partial price is reused as a tuning artifact. This is a localized harness repair
under the existing authorization, not a request for another grant.
