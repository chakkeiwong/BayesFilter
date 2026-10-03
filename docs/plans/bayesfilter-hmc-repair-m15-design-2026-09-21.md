# M15: stopping and whole-fit calibration

This is the next design under the HMC repair master program. Numerical execution
begins after M14's terminal reconciliation and phase refresh. Its ceilings remain
84,000 CPU-reference and 24,000 GPU worker seconds (GPU allocation
refreshed below from the original 15,000); actual failed attempts and
resource repairs count. No new compute is requested. Outputs use `m15-r1/` and
an immutable copy of the repaired source. CPU work deliberately hides GPUs;
GPU work uses trusted TF/TFP/XLA and verified growth.

## Question and evidence contract

Do the declared posterior checks exercise later looks and caps, and what errors
and reported uncertainty occur when the full public tuner and posterior procedure
are repeated on new observations? The baseline is M8's mostly-first-look stopping
design. Each new stopping fit has its own independent fixed-length comparator,
using the same preselected verified member. SBC compares ranks against the exact
discrete-uniform null, using independently simulated datasets and independent
complete fits conditional on each dataset. One fixed dataset or candidate siblings
cannot supply those independent replications.

Every fit uses `tune_hmc_kernel`, standard preparation with the optional
finite-window schedule, startup backoff20, probe16, recovery3, bound expansion1,
and the native broad L search. These are explicit M9--M14 hypotheses, not newly
promoted defaults. Preserve all verified candidates; choose the first verified
identity before posterior draws. No truth/reference draw enters tuning or member
selection. Missing fits, timeouts and caps remain in planned denominators.

Models are normal-conjugate (tau2, sigma1, n6), beta-binomial (Beta(2,3), n12),
and LGSSM location (tau2, sigma0.5, stationary latent variance1, rho0.6, n6).
These are the existing proper generative fixtures. Independent reference code
provides conjugate means/medians and data-dependent rank quantities. They cover
Gaussian, constrained and correlated-observation mechanisms, not all Bayesian
models. M14's nonlinear and matched-reference integrations provide a separate
layer of evidence.

Primary engineering criteria: correct whole-fit independence and data separation;
all-member retention; verified replay; no warmup leakage; honest availability and
cap reporting; complete fixed/stopped pairs or explicit missing arms. Numerical
corruption, broken target/reference identity or missing required evidence stops
the affected experiment for repair. A candidate failure or a posterior cap is
an observed outcome and repair trigger, not a veto on later planned phases.
R-hat, ESS and MCSE remain exclusively posterior checks.

Statistical outcomes are rank-test discrepancies and interval coverage with
uncertainty. A nonrejection does not establish calibration. Report all planned
and available denominators, exact binomial intervals, mean/median errors, and
paired differences. Pointwise intervals are exploratory; report family size and
do not use them to rank methods or change defaults. Earlier minima, warmup
readiness and stopping-time coverage are not inferred from the lugsail formula.

## Allocations and escalation by measured cost

First execute one CPU SBC pilot per model: one dataset and three complete fits,
900 seconds per job (2,700 total), separately seeded and excluded from fresh
calibration. Pilot status and cost resolve whether the following fixed fresh
design is affordable. Unfinished pilot fits trigger a localized resource/profile
review, not replacement with successful seeds.

Fresh CPU SBC has 32 normal datasets, 32 beta-binomial datasets and eight LGSSM
datasets. Each dataset requires three independently tuned fits, with one output
from the first chain's last retained transition, irrespective of posterior pass
status; absent retained output is missing. Each dataset has a 900-second ceiling:
72 datasets, 216 complete fits, at most 64,800 worker seconds. Each model is a
predeclared shard group on identical source/numerical settings. Root seed
2026092192 and separate pilot/fresh/model/shard domains prevent accidental reuse.
Three rank draws gives four rank categories: this is a budgeted calibration
design with limited subtle-defect power, explicitly assessed in M16.

Fresh CPU stopping has four new datasets per model/start combination, with
dispersed and remote initial regimes: 24 complete fits, 600 seconds each
(14,400 total). Its absolute MCSE requests are .005 for normal, .001 for beta,
and .01 for LGSSM, applied to both means and medians. These are explicit precision
hypotheses, approximately 1--2% of typical fixture posterior scale, intended to
exercise later looks. The exact posterior scale varies with data; no universal
relative-precision claim is made. Warmup minimum2000, recent window1000, chunks500,
maximum10000; retained minimum1000, chunks500, maximum10000. These are inherited
operational baselines, not known sufficient counts. The independent fixed arm
discards2000 and retains10000 per chain. Truth is assessor-only. SBC uses the
same posterior counts with the older .1 absolute request, because its question
is the actual output law of that explicitly defined procedure.

The CPU worst case is81,900 worker seconds. Reserve900 for focused regressions,
900 for analysis/bookkeeping and300 for local repair. If pilot costs show a design
cannot finish within its declared caps, revise its allocation from remaining
phase/campaign funds before launching that design; preserve the original pilot.
Do not describe an underfunded design as calibrated. Up to eight CPU workers,
each with two intra-op and one inter-op threads, are allowed; begin with the
three pilots and inspect host memory before eight-worker execution.

The initial GPU allocation, superseded by the measured-cost amendment below,
first runs one stopping fit per model, at900 seconds each. Then
run four normal and six beta SBC datasets, three complete fits per dataset,
at1200 seconds per dataset. Total worst case14,700 seconds leaves300 for
analysis/resource margin. GPU pilot wall costs and memory determine concurrency
(at most two workers sharing a selected GPU with incremental allocation).
These GPU samples test the declared route on current code; their small dataset
counts cannot establish device-wide calibration or justify pooling CPU/GPU
outputs. LGSSM GPU stopping is present even though its GPU SBC is unfunded.

The generated JSON suites, source snapshot, exact commands, run indices, per-fit
candidate inventories, posterior chunks and aggregate tables are the evidence.
Development pilot outcomes remain separate from fresh assessments. No
package/environment change or external communication is involved.

## Skeptical review and phase exit

Wrong baseline: M8 mostly stopped at its first eligible look; its favorable
coverage cannot validate repeated stopping. The new requests deliberately seek
later looks and report how many occurred. Proxies: preparation success and
acceptance do not establish posterior accuracy. Unfair comparisons: the fixed
arm uses the same member and independent streams, with its larger sample cost
shown. Hidden assumptions: normal intervals require applicable CLTs and useful
variance estimation; beta/logit and LGSSM data transformations need existing
independent oracles. Independence: every SBC draw comes from a whole separate
fit and shared observed data only; seed domains include dataset and fit.
Missingness: incomplete datasets cannot be silently removed to claim unconditional
calibration. Power: the four-bin rank design may miss subtle defects, so M16 must
measure defect sensitivity. Environment: CPU/non-XLA is reference evidence;
GPU/XLA is separately executed. Count and time ceilings prevent unbounded repair.

This design passes the skeptical review for bounded execution and measurement,
not default promotion. Before M16, reconcile attempts, inspect cap/failure causes,
test localized fixes, preserve decision/inference-status tables and refresh the
exact acceptance-boundary and defect size/power designs. An observed failure in
one procedure motivates that next discriminating experiment; it does not by
itself invalidate the harness or stop the program.

## Pilot review and fresh launch, September 21

All three CPU pilot datasets completed their three independently tuned fits:
normal580.07, beta751.47 and LGSSM624.00 worker seconds. All nine produced
retained output; all nine separate posterior assessments passed. These are
development observations, excluded from fresh calibration and insufficient to
estimate reliability. The slowest observed dataset fits the declared900-second
cap with148.53 seconds margin. No statistical or numerical setting is changed.
The first GPU stopping pair completed in511.58 seconds; the other GPU pilots
remain in progress and must be reviewed before launching GPU SBC.

The fresh CPU SBC and stopping jobs share one eight-worker queue, preserving
each previously resolved design, data, seed, cap and SBC aggregation group.
Interleaving the first24 stopping jobs with SBC jobs exposes stopping failures
early without changing their statistical experiment. Host inspection before
launch showed207GB available; every worker retains two intra-op and one
inter-op threads and deliberately hides GPUs. Actual resource caps or missing
fits remain outcomes; source/checkpoint continuation cannot silently replace
scientific failures or erase the planned denominator. The immutable numerical
source is M14 `source-schedule-r1`, identity
`9a8d4da7c5e4706477433a379cacc049445e6233ba6e527d93fdfe93cf8f8b37`.
Pilot cost and invariants support this bounded fresh launch; no default or
calibration claim follows from the pilot pass.

### GPU allocation repair before fresh SBC

The first two stopping pilots used417.75 and516.46 seconds for preparation
and search alone, plus roughly45--57 seconds for member construction and
posterior assessment. A three-fit dataset could therefore need about1720
seconds before contention or data-dependent variation. The original1200-second
GPU cap is underfunded relative to these measurements. Increase each of the
ten still-unstarted GPU SBC dataset caps to2100 seconds (roughly22% above that
measured three-fit projection), preserving models, seeds, data law, fit count,
numerical settings and criteria. This is a resource repair, not a replacement
experiment or threshold relaxation. Use a new resolved suite file.

Increase M15's GPU ceiling from15,000 to24,000 worker seconds, covering the
21,000-second fresh inventory, the2700-second pilot reservation and300 seconds
of margin. M14 left48,626.24 seconds; the revised M15 ceiling still funds
M16/M17/M18's7500/10000/2000 ceilings and leaves5126.24 seconds of reserve.
Actual unused charges carry forward. No total campaign allocation is increased.
Use at most two fresh SBC GPU workers after the third pilot and device review.
The beta stopping pilot reached its10,000 retained cap with precision unmet;
the fixed comparison and both interval calculations remain available. This is
a posterior outcome that stays in the denominator, not a tuning rejection or
an infrastructure failure.

### Terminal CPU resource continuation

The fresh queue completed 95 of 96 jobs. Beta-binomial dataset 003 reached its
900-second worker ceiling after two complete fits; its third fit retained
175 tuning observations and a partial checkpoint with verified members.
No numerical or target failure appears in its log. Allocate 600 additional
CPU seconds to resume this exact dataset and source, preserving the original
root and copying only its failed job into `fresh-cpu-continuation-r3` with the
original run index. Completed jobs still point to their immutable original
results. This is a conservative allowance relative to observed 200--300-second
fit costs and checkpoint reload, funded by the unspent 84,000-second phase cap.
Charge both attempts; count the dataset once. If the resumed procedure has no
retained output, keep it missing rather than selecting another seed. The
unchanged design/source/checkpoint checks and absence of a numerical veto
support this localized resource repair.
