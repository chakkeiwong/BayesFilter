# Observation-aware TT master program

Execution and amendments have been reconciled under the owner request of
2026-09-14. The original scientific specification remains below; its launch
status text is historical. The exact pre-refresh copy and checksum are preserved under
`artifacts/observation-tt-master-refresh-20260914-01/`.

## Current execution state — 2026-09-14

**Bounded campaign complete; candidate not promoted; numerical continuation not
authorized.** This file is the governing program index. The
[active checkpoint](observation-aware-tt-active-checkpoint.md) is its concise
resumption pointer. All original stages executed; fitting accuracy and
statistical benefit remain unresolved scientific questions.

| Original stage | Required work | Execution and evidence |
| --- | --- | --- |
| 0 | Recursive likelihood-weighted SGQF, signed-mass/SPD checks and observation response | Executed; rough-guide/reference and history-response checks passed in the tested scope. Exact filtering is not established. |
| 1 | Affine charts and exact change of variables | Executed; focused density/Jacobian checks passed. The inspected target already includes both adjacent states. |
| 2 | L1-selected adjacent-state square-root TT fitting, including the previous TT density | Scalar and pair fits executed. Large residuals and the unsuccessful solver repair leave accurate/converged fitting unresolved. |
| 3 | Retain the joint, integrate the suffix and construct conditional KR | Executed. The pair extension and its Gram-index correction have independent marginal/conditional checks. |
| 4 | Independent full-support conditional draws and actual proposal densities | Executed; corrected pair runs pass tested finite/mass/bracket/CDF checks. Finite test coverage is not universal certification. |
| 5 | Exact initial/transition/likelihood correction and post-correction resampling | Executed for d=1,4, all T=20 observations; saved-fit retry completed the real master consumer. |
| 6 | References, conditional heuristic comparisons and analytical frozen-proposal score | Executed. All six proposals pass reference screens; pair heuristic losses block promotion. The score check covers scalar guided frozen proposals only. |

Authorities: [original result](observation-aware-tt-repair-complete-program-20260913-result.md)
and [terminal pair result](observation-tt-pair-block-remedy-20260914-result.md).
Four particle seeds on one observation sequence do not support a statistical
ranking. d1 pairing is a transferred configuration, not separately tuned.
The downstream observations were fresh for the original pair launch and exposed
for recovery; changing output directories did not create another holdout.

The [48-page manuscript](artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.pdf)
and [rendered review](artifacts/observation-tt-pair-block-20260914/rendered-review/review.md)
include the corrected mathematics and terminal comparison. Eight theorem audits
remain inconclusive; scoped identities do not certify the whole document.
Human prose feedback, pair gradients, retraining derivatives and HMC are not
completed or claimed.

## Execution and amendment history

H1–H5 are retrospective index labels, not invented earlier approvals.
Original plans, failures, manifests and result files remain unchanged.

| Record | Evidence-driven reason and action | Result / authority |
| --- | --- | --- |
| H1: complete-program replacement, 2026-09-13 | Earlier scalar runs omitted required lifecycle components and had prior/support/randomness defects. The replacement implemented original stages 0–6. | [Former plan](observation-aware-tt-repair-full-master-20260913.md), [replacement result](observation-aware-tt-repair-complete-program-20260913-result.md). Former runs 01–20 are historical mechanics evidence. Full launch 1 completed; scalar guided TT not promoted. |
| H2: first-transition diagnosis, 2026-09-14 | Large fit residuals persisted despite reference screens. Tested the joint target, ordering/rank obstruction and attainable fitting accuracy. | [Plan](observation-tt-first-transition-root-cause-20260914.md), [result](observation-tt-first-transition-root-cause-20260914-result.md). Two bounded diagnostics supported a representation restriction and unused fitting accuracy. |
| H3: pair-block remedy, 2026-09-14 | Added paired blocks and weighted rows with both marginal and conditional consumers. Calibration and downstream observations were new at first exposure. | [Design](observation-tt-mitigation-design-20260914.md), [test plan](observation-tt-pair-block-remedy-20260914.md). Self-review and focused checks recorded; no independent model review claimed. Diagnostic-01 failed on a harness API; diagnostic-02 exercised the sole solver repair. Full launch 2 failed during d4 sampling. |
| H4: sampler recovery, 2026-09-14 | Reproduced negative conditional mass; corrected a wrong Gram index after an independent coefficient-integral regression. Reused identical fits/settings. | [Recovery plan](observation-tt-pair-sampler-recovery-20260914.md), [terminal result](observation-tt-pair-block-remedy-20260914-result.md). Corrected replay and full launch 3 completed. Candidate still not promoted. |
| H5 / A03: master reconciliation, 2026-09-14 | Owner identified stale master status and an informal next-study/budget suggestion. Reconciles this program and defines how a continuation is to be proposed. | [A03 change record](observation-aware-tt-master-amendment-03-20260914.md). Independent review pending; no numerical continuation activated. |

The original master allowed recorded capacity amendments within remaining
budget. H2–H4 followed evidence about its research question; they did not
establish all the results needed for promotion. No scientific reason requires
abandoning that question. New calibration/replication would extend a completed
campaign, and is not an existing authorized launch.

## Reconciled budget

The original allowance was 2400 numerical seconds and three full master
launches. Failed attempts count. Artifact paths below are relative to
`docs/benchmarks/artifacts/`; each run has a `run_manifest.json`.

| Attempt / charge | Seconds | Artifact directory or evidence |
| --- | ---: | --- |
| CPU mechanics | 4.944046 | `observation_aware_tt_complete_20260913/cpu-mechanics-01/` |
| GPU/XLA admission mechanics | 11.145976 | `observation_aware_tt_complete_20260913/gpu-xla-mechanics-01/` |
| Full launch 1, completed | 99.574079 | `observation_aware_tt_complete_20260913/campaign-01/` |
| First-transition diagnostic | 13.103820 | `observation_tt_first_transition_20260914/run-01/` |
| Quadrature-check diagnostic repair | 11.546747 | `observation_tt_first_transition_20260914/run-02/` |
| Pair diagnostic, harness failure | 6.187937 | `observation_tt_pair_block_remedy_20260914/diagnostic-01/` |
| Pair diagnostic and sole solver repair | 73.699802 | `observation_tt_pair_block_remedy_20260914/diagnostic-02/` |
| Full launch 2, failed | 151.322206 | `observation_tt_pair_block_remedy_20260914/campaign-01/` |
| Earlier 18-test suite | 8.000000 | [Reset memo](codex-session-stall-reset-memo-20260914.md), budget section |
| Replay-01, reproduced defect | 14.598617 | `observation_tt_pair_block_remedy_20260914/replay-01/` |
| Two recovery CPU-test process ceilings | 120.000000 | [Recovery plan](observation-tt-pair-sampler-recovery-20260914.md); before/after logs under pair artifact root |
| Replay-02, completed | 12.479128 | `observation_tt_pair_block_remedy_20260914/replay-02/` |
| Full launch 3, completed from saved fits | 48.464704 | `observation_tt_pair_block_remedy_20260914/campaign-02/` |
| **Charged / remaining** | **575.067063 / 1824.932937** | [Terminal summary](../benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/campaign-02/terminal_summary.json) |

All three full launches and the one scientific solver repair are consumed.
Historical unmetered routine checks remain an accounting limitation. The
balance is the recorded conservative ledger, not proof that all historical
routine work was metered. This refresh/review runs no numerical experiment.

The suggested extra GPU-hour was an unreviewed estimate, not an allocation.
It is withdrawn as an execution recommendation. A new request must follow a
costed protocol. Neither the old time balance nor review alone resets launch
limits or authorizes materially expanded compute.

## Proposed continuation and change control

A03 records the following proposed sequence, not a numerical launch plan:

| Proposed phase | Deliverable and progression rule |
| --- | --- |
| 7: design the numerical protocol | Separate solver realization from representation capacity, define dedicated defensive-mass calibration, fresh partitions, scope-specific fitting/L1 selection, exact commands, total budget and stops. Review the completed protocol before a launch. |
| 8: calibrated fitting/repair | Under that reviewed/authorized protocol, check same-target optimization before changing capacity; evaluate the 5% mass assumption under a dedicated coverage/non-harm contract. Failed fitting is not hidden by broad filtering reference screens. |
| 9: independent-sequence filtering | Freeze the selected candidate; run the complete reference/heuristic ladder by observation regime with predeclared paired uncertainty across sequences. Do not tune on these observations. Insufficient precision is inconclusive. |
| 10: terminal record and manuscript | Update this program, checkpoint, budget, result/inference tables and manuscript from actual outcomes, including failures or skipped dependent phases. |

Pair total gradients, retraining derivatives, HMC, larger dimensions, coupled
charts and source-faithful TT-cross are outside this proposed continuation.
Its detailed protocol and numerical budget are not yet approved or executable.

For a change: identify its stage and evidence; preserve the previous
specification; state changes to target, data, method, criteria, assumptions and
budget; perform the requested independent review and resolve material findings;
record owner authorization at any new scope/budget boundary; update this master
and checkpoint before execution. Local repairs within an admitted unchanged
contract consume its budget and preserve unique attempts; they do not need a
new review/approval for every retry. Update this index after each substantive
phase result.

Exact next action: review the A03 reconciliation and sequence, then prepare the
phase-7 protocol. No numerical experiment is scheduled before that protocol
and its execution allowance are established.


## Frozen initial campaign specification

### Question and scope
Does recursive SGQF observation guidance repair the concentration problem in adjacent-state square-root TT regression sufficiently to yield usable, exactly weighted conditional proposals in SV? Implement the full lifecycle, then distinguish implementation validity from candidate quality. Fixed physical parameters during construction; the candidate is an extension, not source-faithful regression. The independent empirical particle bank and the retained TT density both persist across observations.

### Consolidated pre-execution review
The former scalar driver is incomplete and unsuitable as correctness evidence: uniform initial grid instead of the Gaussian prior; bounded-support proposals; assigned deterministic innovations correlated with ancestors; no SGQF coordinate pullback; missing retained TT marginal; unweighted guide training rows; silently floored weights; no independent filtering reference; and success criteria based on mechanics. Runs 01–20 are preserved as historical diagnostics. Do not reuse their PASS as scientific evidence.

The replacement must execute: (0) likelihood-weighted repository SGQF with signed-mass/SPD checks, recursive moments and observation response; (1) affine charts and exact change of variables; (2) L1-tuned Hermite TT fit of the adjacent target including the previous TT law; (3) retain the full joint for upper, suffix-conditioned KR, integrate the suffix for the next TT law; (4) independent conditional draws and actual full-support proposal densities; (5) exact initial and transition/likelihood weights with ordinary resampling only after correction; (6) references, heuristic comparisons, and analytical frozen-proposal score/finite differences. Each stage writes a separate result. A crashed stage preserves its manifest and traceback.

### Source and classification
Zhao–Cui JMLR 25 (2024), Algorithms 2–3, equations (16), (20)–(23), local text .localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:539–626, 693–719, 808–928. Author source: third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:1–16 and suffix-conditioned branch; marginalise.m:25–85. Identity continuation and exact density ratio adapt Algorithm 3. Gaussian-Hermite basis, fixed ALS/L1 schedule, SGQF guide and defensive mixture are explicit extensions. No author TT-cross equivalence is claimed.

### Evidence contract and research intent
Comparator ladder: transition proposal/bootstrap filter; unconditional stationary-prior proposal; SGQF Gaussian proposal; TT with predictive coordinates; TT with observation-guided coordinates. The three simple proposals are constructed from this SV model. Evaluate ordinary, near-zero and large-magnitude observations separately. None is a tuning target.

Primary candidate screen: finite, valid proposal/weights plus filtering means and log-evidence agreement with a validated scalar grid or replicated larger bootstrap reference, using predeclared uncertainty. Proposal density/CDF/Jacobian mismatches, invalid signed SGQF mass/covariance, non-finite values, and reference disagreement veto promotion. Small ESS, fit residual and timing are explanatory/repair triggers, not correctness proofs. Reference inconsistency, wrong target/density or exhausted budget are continuation vetoes. A failed TT arm does not stop comparisons designed to diagnose it.

Implementation acceptance requires numerical integration and conditional CDF checks on a nonuniform multivariate TT, independent sampling, Gaussian-prior timing, actual SGQF-to-chart-to-fit-to-KR wiring, retained marginalization, and analytical score agreement for the same frozen-proposal scalar. Frozen proposals/ancestry define the score target; this does not differentiate retraining or certify a posterior/HMC run.

### Defaults, assumptions and earliest diagnostics
- SV gamma=.6, beta=.4, sigma=1: paper synthetic setting; correlated dimension-4 transition is the historical C2 extension, frozen matrix imported as data with provenance. Check stability and stationary Lyapunov residual first.
- SGQF levels 2–5: repository rules; signed moments may fail. Escalate deterministically to the next level after mass/SPD or successive-level disagreement, record all attempts, fail if no validated guide. Compare against independent positive quadrature for scalar and particle integration for dimension 4. No covariance clipping.
- Hermite degree 3, rank 3, fixed 4 ALS sweeps initially: capacity hypotheses, not established defaults. Fixed-budget L1 grid 0, 1e-5, 1e-3 selected on separate validation rows; untouched audit rows only veto. Weighted target RMS normalization fixes penalty units. No ridge needed by first-order convex core solves; step size is inverse spectral norm, so singular normal matrices need no inverse. Check per-core objective and finite residuals.
- Defensive Gaussian mass 5% in the joint: declared support/robustness hypothesis. Uses exact full mixture density, not a TT-only denominator. Conditional mixture probabilities follow the fitted suffix mass. This ensures positive support, not bounded importance weights or tail efficiency.
- Gaussian row design, independent training/validation/audit seeds: Monte Carlo regression design; rare target mass may be missed. Audit relative residual, row maximum weight, and downstream reference discrepancy expose this; do not promote by residual alone.
- No adaptive scientific changes on audit data. Later higher-capacity variants require a recorded plan amendment inside remaining budget.

### Budget, environment, execution and stops
Initial complete campaign: at most 40 minutes of numerical execution, at most 3 launches including localized repairs; dimensions 1 and 4, horizon 20, 4 independent particle seeds, 512 candidate particles, independent bootstrap reference replicates. Focused CPU tests are explicitly diagnostic; candidate kernels default to GPU/XLA with memory growth. First run a tiny CPU/non-XLA mechanics test and a bounded GPU/XLA parity/compile smoke. If GPU/XLA cannot pass, record the exception and do not count CPU mechanics as GPU acceptance. Training and every conditional draw have setup-static signatures; loops over cores/time are static or native graph loops, never pfor.

Artifacts: docs/benchmarks/artifacts/observation_aware_tt_complete_20260913/<unique-run>/; manifest before execution includes hashes of changed code, git commit/dirty status, exact command, seeds, device and XLA metadata, wall budget. Run driver with --help for the frozen configuration and --output-root for a new directory.

Pre-mortem: a fit can look good while missing posterior tails; a signed rule can return finite but wrong moments; fitting and evaluation may share the same error; deterministic particles can conceal biased sampling. Independent densities/integrals, disjoint seeds and larger reference runs address these. Self-audit passes for implementation and bounded validation after the former-driver defects above are removed. Scientific success remains an experimental question.

### Frozen implementation screen details (before complete execution)
Use the highest valid SGQF rule among levels 2–5; successive-level discrepancy is retained, and the actual recursive guide must agree with the independent filtering mean to within one process-noise standard deviation at every coordinate/time. This tests the requested rough-guide role, not exactness. Response to the first observation and its effect at the next unchanged observation must each exceed 1e-8. Scalar reference grids use 801 and 1201 nodes over ±12 stationary standard deviations, requiring log-evidence resolution difference <=1e-6. Particle reference uses four seeds and 32768 particles. The candidate screen allows 3.182446 combined MCSE plus .15 in log evidence, and the same combined-MCSE allowance plus .15 sigma for each filtering mean coordinate. These are exploratory agreement screens, not simultaneous confidence guarantees. Conditional observed losses to simple proposals veto promotion and are reported descriptively; no statistical superiority claim is planned. Frozen-score centered differences use h=1e-5 and relative error <=2e-6.

The initial-prior retained density is now a supported t=0 consumer of the shared Hermite density implementation. General KR integration/inversion is shared with the existing retained proposal; the new consumer selects reverse generation with a particle-specific suffix environment. Ten focused tests passed before the driver smoke, including nonuniform density normalization, CDF derivative, upper-axis ordering, retained marginal integration, actual consumer wiring, and observation-history response.

At the original launch, GPU/XLA admission smoke completed on RTX5080 with memory growth: 11.15 s, compared to 4.94 s CPU/non-XLA. All 48 reported numeric decision fields matched within 1.73e-15; both full scalar T=3 lifecycles completed and frozen-score checks passed. Compilation was observed in the GPU log. This admitted the bounded GPU campaign, not a performance/default claim. That initial complete launch was attempt 1 of 3 under the 2400-second total numerical budget; the current exhausted launch status is recorded above.
