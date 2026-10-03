# M21/M22 continuation after remote integration

This plan supersedes the source and launch details of the unlaunched M21
public-HMC and M22 null confirmation inventories. Their scientific targets,
screens, comparator counts, caps and reporting denominators remain unchanged.
The old waiting queue was stopped with no numerical task started. Preserve it.

First commit the tested integration and bounded continuation code. Freeze the
committed `bayesfilter/` package using Git, excluding unrelated dirty worktree
files. Record commit, per-file hashes, aggregate identity and environment.
Use `m21-r2/source-r1` for all new studies; `m21-r1` and `m22-r1` remain intact.
Before confirmation, run one new Gaussian and one beta-binomial public fit
with the existing M21 pilot settings and independent fixed-count comparators.
Their root seed is 2026092241, with distinct design identities. Reserve 600
CPU seconds each. These check complete execution, diagnostics, sibling
retention and cost under the merged implementation; neither is calibration.

The incoming arithmetic intentionally changes bulk/tail ESS and preserved-
transition mean reporting. The identified TFP precision-policy estimator and
lugsail calculation remain unchanged. No bitwise posterior-stop parity is
required across a changed diagnostic; report both the method and outcome.
Posterior caps do not invalidate the public fit or remove tuning siblings.
Broken law/score, missing evidence or unexplained source mismatch stops the
affected experiment for repair. Runtime cost exceeding a group's reservation
requires an affordability revision before confirmation, not a partial sample
silently reported as the planned sample.

Subject to that pilot, retain the frozen M21 inventory of 128 fresh Gaussian,
128 beta-binomial, eight rotated Gaussian and eight LGSSM fits. Use root seed
2026092242 and new output roots under `m21-r2/public-confirmation-cpu-r1`.
Reserve 28000, 35000, 3000 and 3000 CPU seconds respectively. The two larger
groups use the previously declared exact pointwise lower-bound 0.90 delivery
and interval screens; the eight-fit groups are stress repetitions only.
Preserve missing/capped outcomes in the fixed denominators. Do not rank viable
members using posterior diagnostics. All verified siblings remain retained.

For M22, run 512 independent complete null experiments each for baseline and
no-op, root seed 2026092243, under `m22-r2/null-confirmation-cpu-r1`, retaining
the 10800-second reservation. Every experiment must be valid. The unchanged
pointwise 95% Clopper--Pearson upper rejection-rate bound must be <=0.10 in
each arm to pass the declared diagnostic screen. This cannot close whole-fit
defect power. The pilot's measured cost makes repeated full-fit SBC power
under-budgeted within this phase; retain that gap explicitly.

Before the long queue, independently confirm the estimator hypotheses nominated
by M21. Use 400 fresh independent fixed-count arms each at stationary rho=0.98
and deterministic starts (-8,-4,4,8), rho=0.995. Four chains, 10000 discarded
transitions and 10000 retained transitions per chain are inherited comparators.
Compare the unchanged lugsail baseline (b=100), declared alternative b=500,
and existing TFP autocorrelation precision estimator on each same fresh array.
Keep r=3, c=0.5 and minimum 20 batches. No bandwidth is chosen from these
outcomes. Root seed 2026092244 is a disjoint convenience seed, not a favorable
seed selected from data. Reserve 900 CPU seconds, including raw array storage
and summaries, under `m21-r2/estimator-confirmation-cpu-r1`.

This fixed-count confirmation asks whether each nominated estimator meets the
existing diagnostic coverage floor in these two exact laws. Report all 400
outcomes, unavailable estimates, exact pointwise 95% binomial intervals and
estimated/exact standard-error ratios. Eligibility requires all estimates
available and the coverage lower bound >=0.90 per case. Compare exact Gaussian
oracle coverage with 0.95 using a Bonferroni interval across the two cases;
an oracle failure triggers diagnosis. Method differences are descriptive:
no superiority test or anytime-valid interval is claimed. Even a passing
alternative needs separate sequential-controller confirmation before a
default change. This is an optional-estimator validation, not default promotion.

All work uses the tfgpu environment with GPUs hidden and one TensorFlow
intra/inter-op, OMP and OpenBLAS thread. At most two numerical workers run
across the campaign. Per-worker ceilings and versioned outputs are mandatory;
coordinator waiting time is not charged again. M21's revised 79200-second
ceiling covers completed work, the pilot, estimator confirmation and 69000
seconds of public confirmation. M22's 43200-second ceiling covers its pilots,
null confirmation and remaining planning/repairs. GPU charges remain zero
until trusted permitted capacity is available. The total stays within the
owner's 48 CPU / 24 GPU worker-hours.

Skeptical audit: a changed diagnostic can change stopping, so old-source
calibration cannot be relabeled. Fixed-count estimator calibration is not
random-stop calibration, and AR(1) transitions are not the public HMC pipeline.
The plan has separate experiments for those questions. Replica counts are
fixed before launch, pilot observations excluded, reference laws exact and
truth assessor-only. The 0.90 and 0.10 screens are inherited adequacy screens,
not nominal-coverage or nominal-size theorems. The most likely misleading
"pass" would be dropping caps or presenting conditional coverage as delivery;
the reporting denominator forbids that. Candidate failure continues the next
funded repair; invalid evidence stops only its affected study. No new global
default or transition family is introduced.
