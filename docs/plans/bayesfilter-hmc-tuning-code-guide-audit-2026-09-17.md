# HMC tuning and guide audit

Date: 2026-09-17. Program:
[HMC repair master](bayesfilter-hmc-repair-master-program-2026-09-16.md).
Execution evidence is recorded in the
[master result](bayesfilter-hmc-repair-master-result-2026-09-17.md).

The common procedure is implemented, and the reproduced defects below have
been repaired. Complete consistency across every target and failure history is
not established. In particular, the new normal-conjugate experiments are small
development checks; they cannot certify automatic preparation, posterior
coverage, or a universal search envelope.

## Answers to the recurring questions

| Question | Finding |
| --- | --- |
| Is there one clear entry point per case? | Yes. Exact ordinary HMC uses `tune_hmc_kernel`; supported frozen transport uses `tune_fixed_transport_hmc_kernel`; conditional position-field mechanics uses the ordinary facade with its typed config/binding. Prepared geometry enters the same controller. Unsupported combinations fail before numerical work. |
| Is the internal logic fully consistent? | The main candidate lifecycle is shared, but this audit reproduced additional reporting, checkpoint, retained-replay and independent-assessor defects. Those counterexamples now pass. Passing regressions is not proof that no further defect exists. |
| Can reasonable problems still fail? | Yes. Preparation can fail; finite domains, candidate/repair/evidence caps and compilation costs can prevent qualification; frozen geometry may be unsuitable; and a qualified kernel may fail subsequent posterior assessment. These failures have different remedies. |
| Is the implementation well structured? | Several active responsibilities have been extracted, but preparation still depends on a 30,053-line historical module. Further staged extraction and measured runtime work are justified. A replacement tuner or a simultaneous rewrite of all stages would add risk. |
| Is burn-in sufficiency measured? | The separate posterior controller checks a declared recent warmup window and can require persistent passes and information floors. It reports cap exhaustion. These are finite-window diagnostics, not an estimator proving that all initialization bias has disappeared. |
| Is lugsail MCSE available? | Yes, for declared posterior means. Quantile precision has a separate estimator. Neither determines burn-in length, proves discovery of modes, or controls repeated-look coverage automatically. |

R-hat, ESS and MCSE do not qualify, rank, repair or delay tuning candidates.
They belong to posterior assessment. Every verified candidate remains recorded;
an explicitly predeclared posterior subset does not discard its siblings.

## Scope and skeptical review

The review followed public dispatch, preparation and affine composition,
candidate proposals, acceptance evidence, health vetoes, cohort scheduling,
reservations and deadlines, seed inventories, durable replay, retained sampling,
posterior assessment, validation engines and rendered guide text. It inspected
the boundaries between active code and historical implementations as well as
the capability registry and examples. It does not claim a line-by-line proof of
all historical helper code or an audit of concurrent work by other agents.

The comparator was the stated numerical and persistence behavior, not prior
plan completion labels. Deterministic counterexamples preceded the terminal
repairs. Actual Gaussian transitions tested recovery and replay; synthetic
observations tested scheduling and reporting only. Saved experiments used a
frozen package, while terminal regression tests used the changing checkout.
Neither source class is silently relabeled as the other.

The review checked wrong baselines, proxy promotion, missing stop conditions,
unfair comparisons, inherited numerical choices, environment mismatch and
artifacts that would fail to answer the question. The main correction to the
experimental plan was to measure complete-fit cost before allocating fresh
shards. CPU and GPU timeouts were resource failures, with prior costs and data
preserved during bounded checkpoint retries.

## Reproduced defects and repairs

| Defect | Why it matters | Repair and evidence |
| --- | --- | --- |
| Preparation epsilon treated as a restrictive exploration ceiling | In a saved native Gaussian failure, every candidate wanted a larger step but the domain ended at the preparation probe value. This did not establish that no valid pair existed. | Explicit finite expansion under unchanged geometry; a new search identity; fresh measurement and verification for every pair. Saved development comparison: zero versus 18 verified members. The default remains zero expansion. |
| `serious()` enabled a retired option | A public preset was rejected by its own dispatcher before preparation. | Removed that override. Tests cover all five public presets reaching preparation. |
| Preparation discarded useful failure causes | A generic `hard_veto` exception did not explain the failed bootstrap/windowed stage. | `HMCPreparationFailure.details` and progress records preserve stage, public diagnostics, round vetoes and repair causes. Tests cover both failure stages and the actual public bootstrap round's `classification` field, which was initially lost through a mismatched field name. |
| Nonfinite diagnostic numbers masked the preparation failure | Raw bootstrap/windowed diagnostics can contain NaN or infinity, which the strict progress-file writer rejected before preserving the original exception. | Encode nonfinite values explicitly at the diagnostic JSON boundary. A nested nonfinite counterexample failed before repair; the 33-test preparation/documentation batch passes afterward. Runtime evidence and failure classification are unchanged. |
| Incomplete shard work was called unstarted | A killed fit with saved numerical work was indistinguishable from an untouched dataset in aggregate reports. | Separate missing-record and known-unstarted counts; unknown unstarted totals remain null. Partial records remain available separately from the statistical aggregate. |
| Declared global quantities did not reach the validation posterior policy | Local diagnostics could pass with a missed mode; reporting mode occupancy only afterward did not test requested precision. | A stable mixture-quantity identity and a declared mean precision target reach `run_hmc_posterior`. The saved missed-mode output fails that assessment. |
| Candidate-cap exhaustion reported `epsilon_domain` | A consumer could enlarge epsilon despite having exhausted the candidate allocation. | Distinct candidate, family, domain and unvisited-proposal reasons. Refinement now counts only eligible unadmitted pairs. |
| Malformed observation persisted before receipt validation | Interrupted work could retain an unusable observation and acquire a duplicate on retry. | Validate before appending to the observation history. Preserve valid shared-failure observations and attempted costs. A real numerical checkpoint can fail, resume and reload successfully. |
| Older result bypassed later shared invalidity in a live binding | A previously verified member could still export after another transition exposed shared execution invalidity. | Recompute/check the entire live evidence inventory, including evidence outside the older result's receipts. Known shared invalidity blocks both export and new member construction. Candidate-local failures retain their distinct treatment. |
| Stopped-interval assessor assumed all quantity names were coordinates | A successful mode-probability report would raise at lookup; a warmup cap omitted its missing interval. | Independent mixture-CDF lookup and an explicit denominator for every declared global mean. Tests cover a valid probability interval and entirely missing posteriors. |
| Inventory oracle rejected a valid shared-failure observation | Shared invalidity is recorded on interrupted work without a normal receipt; the oracle incorrectly required every observation to belong to completed work. | Allow that exact terminal case, preserving missing/extra/duplicate observation checks. |

The terminal counterexample files retain the failing results. One refinement
test initially used an invalid factor of one; correcting that fixture exposed
the actual overcount (five remaining proposals instead of two). Two test
commands named nonexistent test files and ran no tests; the corrected commands
and all charges remain recorded. These harness errors are not code failures or
statistical defect detections.

The retained-replay repair applies to evidence available in the live or loaded
binding. An exported historical bundle cannot know about a later failure that
was never included in it. No global revocation service or new launch mechanism
is introduced for this trusted local repository.

## Invariants checked across stages

| Stage | Checked behavior and principal source |
| --- | --- |
| Dispatch | Config/route checks precede preparation; exact-score and conditional position-field authority remain distinct. `hmc_tuning_dispatch.py`, `tuning_contract.py`, `fixed_transport_hmc_tuning_tf.py`. |
| Preparation | Both affine layers and actual post-warmup starts survive the ordinary handoff; expansion changes the search domain, not geometry or qualification. `hmc_preparation.py`, `hmc_candidate_set_public.py`, preparation binding factory. |
| Proposals and evidence | Exact pairs are immutable and deduplicated; repairs preserve L/geometry and require fresh receipts; no monotonic-acceptance assumption grants qualification. `hmc_candidate_proposals.py`, `hmc_candidate_decisions.py`, `hmc_verification.py`. |
| Scheduling | Current cohorts close before child admission; reservations protect mandatory work; released resources can fund deferred work; finite rungs can end inconclusively. `hmc_candidate_set_tuning.py`. |
| Numerical execution | Native state/proposal consistency and declared health checks include discarded evidence warmup. Chunks use recorded independent streams and charge attempted work before execution. `hmc_candidate_set_execution.py`, `hmc_candidate_runtime.py`. |
| Persistence and retained sampling | Immutable evidence, checked geometry/source, preserved failed ancestors, exact child kernels and fresh posterior seeds; interrupted native attempts remain charged. `hmc_candidate_set_checkpoint.py`, `hmc_candidate_set_retained.py`. |
| Posterior assessment | Active versus model coordinates remain distinct; extra quantities preserve draw/chain axes; warmup is excluded; invalid precision cannot pass. `neutra_hmc.py`, `hmc_posterior_assessment.py`, `hmc_precision.py`. |
| Independent validation | Original fit denominators survive failures; candidate siblings do not become independent replications; source identity separates evidence versions. `testing/inference_validation/engines`, `aggregation.py`, reporting and accounting scripts. |

For lugsail means, the checked quantity is the per-chain long-run variance
estimate `(V_b - c V_floor(b/r))/(1-c)`, pooled as `sum(V_i)/(m*m*n)` for
independent chains of equal length. This matches the guide's stated stationary
CLT approximation. Incomplete terminal batches do not enter the long-run
variance estimate but remain in the reported mean; this is documented.
Nonpositive per-chain estimates are unavailable, not clipped to apparent
precision. Existing independent formula/rank/quantile regressions passed; no
new arithmetic defect was established. Finite-window diagnostics do not verify
the assumptions under which those quantities describe full-posterior error.

## Remaining gaps and next repairs

| Priority | Gap still open | Next discriminating work |
| --- | --- | --- |
| High | Automatic preparation/expanded search is not broadly calibrated | On a fresh frozen source, pilot ordinary beta-binomial, LGSSM location, funnel and rotated Gaussian. Preserve preparation failures and empty sets. Use measured cost and defect sensitivity to fund a complete independent-fit design. The normal-conjugate result cannot fill these cells. |
| High | Subtle-defect detection remains weak | Revisit actual reversed-MH-ratio kernels at the previously weak epsilon settings, with baseline/no-op controls. Change the observable or experimental horizon under a new fixed design, then estimate size and power with intervals before scaling full SBC. More repetitions of an insensitive test do not solve the mechanism. |
| High | Known-mode monitoring is not mode exploration | The new indicator prevents the saved false precision claim, but both declared start regimes exhausted posterior warmup. Target-specific geometry/transport and global reference checks are needed. Do not feed mode failure back into tuning membership or choose posterior winners after looking at results. |
| Medium | Acceptance operating characteristics are only partially measured | Synthetic Beta marks isolate screen arithmetic and dependence sensitivity. Full multivariate, nonstationary and actual-HMC measurement/verification paths still need calibration; the intervals are operational compatibility screens, not sequential confidence guarantees. |
| Medium | Stopping/default accuracy has limited evidence | Warmup counts, recent windows, persistence counts, ESS floors and batch choices remain operational baselines. Evaluate reported intervals at actual stopping, including caps, under matched fixed-length comparisons and adequate replication. Lugsail alone does not justify a new stopping default. |
| Medium | External consumer coverage is unavailable in the registered suite | Regression, eight-schools and MacroFinance suite entries have no bound matched observations/reference bundles. Preserve this status until exact target/data/prior/coordinate identities, reference uncertainty and consumer outputs are supplied. Repository mechanics and literature examples are not substitutes. |
| Medium | Optional sequential invariance wrapper is absent | The current Gandy–Scott random-position test is a fixed-look construction. Algorithm 3 is the sequential wrapper; Algorithm 2 constructs ranks and Algorithm 1 is the two-sample procedure. Use fresh independent p-value vectors at successive looks, audit the complete mcunit wrapper source, and check p-value resolution, size and power before claiming its guarantee. |
| Medium | Preparation and historical code remain entangled | Extract active geometry, bootstrap, windowed preparation and config translation into bounded modules while preserving public imports, serialized identities and tests. Keep historical readers explicitly separate. |
| Medium | Search/serialization/compilation costs grow materially | A widened search often reaches 100 candidates. Profile source checks, evidence hashing, analysis caches, exports and graphs keyed by `(L, count)` separately. Optimize only with matched numerical results and measured construction/compile/steady-state costs. |

The module sizes help locate refactoring pressure: the controller is 1,777
lines; numerical binding 828; retained bridge 459; public translation 288;
preparation facade 294; historical ordinary module 30,053; TensorFlow historical
tuning module 1,966; fixed-transport tuning module 2,773; posterior controller
3,227. Public dispatch is already separate at 173 lines. File size is not a
correctness metric; the active facade's dependency on the historical module is
the concrete maintenance problem.

A reasonable sequence is: preserve the new failure regressions; extract active
preparation without changing its mathematics; profile and reduce repeated work;
then validate broader targets and any proposed numerical defaults. Each stage
needs its own bounded comparison. Keep candidate-set membership and posterior
assessment separate throughout.

The source recheck inspected Gandy–Scott section 2.2, Algorithm 2/Proposition
2.3, section 3/Algorithm 3/Theorem 3.1, and the Appendix A proof and Appendix B
parameter experiment in the stored paper. Theorem 3.1 assumes independent
p-value vectors across looks. Repeatedly testing the same cumulative sample
would not satisfy that assumption. The paper's mcunit defaults
`alpha=1e-5, k=7, Delta=4` come from an iid-normal KS experiment, not HMC tuning.
For those values the first boundary is about `1.43e-6`; the present 1,999-draw
Monte Carlo null has minimum p-value .0005 even before multiplicity adjustment.
Blindly copying the defaults would make first-look rejection unreachable.
The local `mcunit-expect-invariant.R` calls `expect_mc_test`; the complete
wrapper implementation must be obtained and inspected before implementing that
claim-bearing extension.

For subtle defects, the same paper explicitly permits replacing a frozen
reversible kernel K by a predeclared power K^s (section 2.2, after Proposition
2.3). This is one candidate mechanism to pilot with disjoint design/assessment
streams, not a recommendation to thin SBC output into pretend-independent
draws. A separate acceptance-energy oracle can test a reversed MH ratio
directly; its success would be engineering evidence and would not establish
invariance-test power. The current mechanics engine tests density/score and
the leapfrog map, whereas the historical small-epsilon power results tested
distributional sensitivity. Those are different questions.

## Guide review

The Markdown reference and tuning chapter explain the optional exploration
cap, detailed failures, recovery behavior, shared invalidity, declared global
quantities and limited meaning of `complete`. The diagnostics chapter already
distinguishes equilibration checks from posterior precision and gives the
lugsail formula and assumptions. The active repository governance also states
that R-hat is reporting-only for kernel tuning.

A clean staged full-book build resolves the tuning and diagnostics citations.
The first out-of-tree build used an old `docs/main.aux` for BibTeX and therefore
missed newer entries; the sources already contained those entries. The clean
build still reports three pre-existing citation keys in another chapter:
`Afshar2015`, `Gorinova2020`, and `Pakman2014`. No reference is invented to hide
them. Rendered tuning pages were inspected for readable text, code, equations
and citations. The final result records the exact PDF/build scope and generated
coverage source list.

Documentation distribution was corrected after the owner's clarification:
`docs/main.tex` is the official book source and `docs/main.pdf` its current
compiled copy. The verified build is installed there; dated build artifacts
serve only as historical records. The README now states this workflow and
removes its stale nomination/R-hat tuning advice.
