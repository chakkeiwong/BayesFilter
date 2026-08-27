# SSL-LSTM q=20 Corrected Parameter-Authority Continuation

Date: 2026-08-26  
Version: `v3.4-fresh-paired-uncertainty-replication`  
Parent terminal record: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-result-2026-08-25.md`  
Status: `IN_PROGRESS_PHASE52_FRESH_PAIRED_UNCERTAINTY_REPLICATION`

This is an append-only continuation of the closed 2026-08-25 particle-authority
program. The parent phases and their artifacts are historical evidence and are
not rewritten. This plan corrects the particle/state measure boundary and
resumes only the surviving parameter-space work.

## 1. Why a corrected version is required

The parent program correctly stopped its direct q=20 LEDH arm. Its Phase 25
transition used a 60-dimensional internal UKF state and a 20-dimensional
innovation. Since the innovation enters only the first 20 coordinates,

`rank(G Q G^T) <= 20 < 60`,

so the induced state transition has no ordinary 60-dimensional Lebesgue
density. Phase 26's 20-dimensional reduced density was a valid mechanics
fixture, but it was not a proposal for the declared four-parameter target.
That route is closed relative to the target; it must not be repaired by
relabeling internal innovations as parameters.

The corrected continuation fixes a second, smaller boundary issue in the
surviving pilot. A chart `theta = c + A z` may be used to sample or mutate, but
the declared target is a density in `theta in R^4`. If `v_theta(theta)` and
`q_theta(theta)` are the target and proposal log densities, then the only
valid importance ratio is

`r(theta) = v_theta(theta) - q_theta(theta)`.

When both are deliberately represented in chart coordinates,
`v_z(z) = v_theta(theta(z)) + log|det A|` and
`q_z(z) = q_theta(theta(z)) + log|det A|`; the Jacobian cancels in the ratio.
It must not be stored as if the theta-density itself contained a chart
Jacobian. ETPF and all particle authorities therefore receive rows of shape
`[N, 4]`, never the internal `[N, 60]` UKF state.

## 2. Governing authority and nonclaims

Active policy has priority over all historical plans. In particular, the
owner's 2026-08-21 LEDH invalidation notice and canonical rebuild policy remain
binding: only the complete LEDH-PF-PF OT route with dual-cap trust-region GenUT,
the per-particle UKF covariance lifecycle, and analytical recursive gradients
may seek claim-bearing LEDH status. A simple theta-space affine/flow scaffold
would be an `extension_or_invention` diagnostic and cannot be called canonical
LEDH or used for HMC admission.

The declared scientific target is the batch-native q=20 SSL-LSTM target
`pi(theta) proportional to exp(V(theta))`, with `theta in R^4`. The 60D state,
20D innovation, sigma points, and UKF covariance are internal terms in `V` and
its score. They are not particles, replay coordinates, or an alternative
target measure.

This continuation can establish engineering and numerical contracts and can
produce role-limited parameter-space evidence. It cannot, from finite runs,
establish posterior correctness, IID Gaussian whitening, exhaustive mode
discovery, an unbiased normalizer, HMC convergence, or statistical
superiority.

## 3. Research-intent ledger

| Field | Corrected statement |
|---|---|
| Main question | Can a fresh authority in the declared four-parameter measure support target-level ETPF/SMC diagnostics and a later NeuTra screen? |
| Candidate mechanism | Parameter-space C0/M0 SMC with explicit theta proposal densities; ETPF acts directly on theta rows; GenUT is a moment/proposal diagnostic only. |
| LEDH boundary | Canonical LEDH is deferred to a separately governed full rebuild. No simplified theta flow is claim-bearing here. |
| Expected failure | Stale chart Jacobian labels, accidental 60D state handoff, invalid target/proposal support, mutation drift, mode loss, finite UKF status, or insufficient evidence for whitening. |
| Primary promotion gate | Exact measure/dimension contract, finite target/status, proposal density in theta measure, and reproducible versioned receipts. |
| Promotion veto | Any `[N,60]` particle input, mixed measure, missing proposal term, non-finite/status-invalid target, unsupported transformed density, stale protocol hash, or role claim exceeding the computed object. |
| Continuation veto | The declared four-dimensional target or a common-support proposal cannot be evaluated, or an exact fixture contradicts the theta-measure identity. |
| Repair trigger | A failed focused test, schema mismatch, harness failure, or candidate diagnostic that can be repaired without changing target, gates, hardware class, privacy boundary, or remaining budget. |
| Explanatory diagnostics | ESS, mode occupancy, covariance/whitening residuals, ETPF negative entries, bridge rows, acceptance, loss, and runtime. |
| Nonclaims | No finite-run mode theorem, IID theorem, posterior theorem, HMC readiness, or default promotion from these phases. |

## 4. Evidence contract

**Question.** Does a fresh, auditable particle cloud in `theta in R^4` provide a
valid input to the modular particle methods, without confusing the internal
UKF state with the target variable?

**Comparator.** The corrected C0/M0 pilot uses the same q=20 target, proposal
family, schedule, and seeds for paired arms. Parent six-bank replay and the
old state-space LEDH artifacts are historical context only. ETPF is compared
to the same weighted theta cloud, not to an unweighted or internal-state
cloud.

**Primary criteria.**

1. A deterministic affine chart fixture verifies the change-of-variables
   ratio and records separately the theta and chart log densities.
2. The q=20 target accepts a static `[N,4]` batch with finite values, scores,
   and status; the target reports `parameter_dim=4` and its signature is
   recorded.
3. Proposal sampling, stored proposal log density, and target value all use
   the same theta measure. The defensive mixture has an explicitly recorded
   positive epsilon and support diagnostic.
4. ETPF receives and returns `[N,4]` rows. Its moment checks are role-limited;
   no density is assigned to the empirical transform.
5. Every run has a unique artifact root, protocol hash, command, environment,
   seed, device policy, source hashes, and decision/inference tables.

**Hard vetoes.** Crash, non-finite value, invalid target status, shape or
measure mismatch, missing density/Jacobian term, stale metadata, overwritten
artifact, noncompliant GPU launch, or a scalar/row-mapped NeuTra optimizer
update. A low ESS or poor whitening is a repair/candidate signal, not a whole
campaign veto.

**Artifact.** Continuation artifacts live below
`docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/`.
The parent artifact root is never overwritten.

## 5. Default and assumption audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|
| `theta` dimension 4 | target API (`parameter_dim`, `[batch,4]`) | accidental internal-state particles | Phase 27 static/runtime shape audit | required fact, checked per run |
| UKF state dimension 60 | target implementation | someone treats latent state as the posterior variable | target signature and source inspection | internal-only fact |
| Geometry chart | prior mode-failure artifact | stale or mode-biased warm start | fresh proposal receipt and support/mode table | calibration warm start only |
| Defensive epsilon | inherited pilot value `0.20` | tail/support or variance distortion | theta-density and tail diagnostics | hypothesis; not a promoted default |
| Safe proposal scale | inherited pilot value `2.0` | poor overlap or unstable tails | finite log-ratio/ESS screen | hypothesis |
| Beta schedules | inherited pilot candidates | schedule-dependent conclusions | calibration-only selection receipt | hypothesis |
| Particle counts | parent `100/300/600` history | budget/scale confounding | paired N ladder when reached | hypothesis |
| ETPF controls | source fixture warm starts | scale and conditioning dependence | theta fixture residuals and support excursions | warm start only |
| CPU reference lane | policy-approved diagnostic exception | cannot support GPU/default claims | explicit hidden-GPU receipt | diagnostic only |
| NeuTra GPU/XLA | owner directive | allocation, batch, or target mismatch | pre-import memory/device probe | later, gated |

No number is promoted merely because it appeared in the parent pilot.

## 6. Skeptical pre-run audit (required before Phase 27)

The old plan was not executable unchanged for the following reasons:

| Audit question | Finding | Repair in this version |
|---|---|---|
| Is the baseline the claimed target? | The old LEDH arm used a 60D internal state for a 4D target. | Close that arm; bind all active particles to theta `[N,4]`. |
| Are proxy metrics promotion criteria? | Whitening/ESS/covariance were sometimes read as authority evidence. | They are explanatory or role-limited; only measure/status/schema gates promote. |
| Are density terms complete? | The pilot's chart Jacobian was added to both terms but labels did not distinguish theta from z. | Store `target_log_theta`, `proposal_log_theta`, and optional chart-coordinate terms separately; test cancellation. |
| Is ETPF a density correction? | The prior q20 probe transformed a retained subset and correctly made no density claim, but it was tied to the old bank. | Run it on a fresh theta bank and retain the empirical-transform nonclaim. |
| Are defaults justified? | Epsilon, scales, schedules, and geometry were inherited. | Mark all as hypotheses and use calibration-only selection. |
| Can a passing command answer the question? | A finite target call alone cannot establish particle authority. | Require common measure, proposal receipt, protocol hash, and role-specific decision tables. |
| Does the canonical LEDH policy permit a shortcut? | No. A simplified theta flow would be a noncanonical extension. | Defer LEDH rebuild; do not create a claim-bearing shortcut. |

The audit passes for the corrected Phase 27 contract because it is a bounded
shape/measure check and does not claim authority or posterior correctness.

## 7. Phase map and inter-phase repair

The continuation uses one remaining campaign pool. The parent recorded
`14780.9 s` of lower-bound wall time against the user-authorized `64800 s` cap;
before each launch, re-read the ledger and reserve only measured remaining
time. Local caps below are ceilings, not additive authority.

| Phase | Purpose | Entry gate | Planned artifact | State |
|---|---|---|---|---|
| 27 | Corrected theta/state measure contract and target API audit | this pre-run audit passes | `phase27-measure-contract/` | `PASS_CORRECTED_PARAMETER_MEASURE_CONTRACT` |
| 28 | Fresh corrected C0/M0 parameter-space pilot | Phase 27 hard gates | `phase28-fresh-theta-pilot/` | `PASS_THETA_MEASURE_PILOT` |
| 29 | Source-faithful ETPF on fresh theta bank | Phase 28 finite/status and density gates | `phase29-fresh-theta-etpf/` | `PASS_FRESH_THETA_ETPF_ROLE_LIMITED` |
| 30 | GenUT scope decision in parameter space | Phase 29 role receipt | `phase30-theta-genut-scope/` | `PARAMETER_GENUT_GLOBAL_INFEASIBLE_SCOPE` |
| 31 | NeuTra data-boundary and batch-native admission screen | a valid fresh theta bank; GPU policy | `phase31-neutra-boundary/` | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` |
| 32 | Fresh-seed theta-authority replication and uncertainty receipt | Phase 31 boundary passes; unchanged target/protocol | `phase32-replication/` | `PASS_THETA_REPLICATION_HARD_GATES_DESCRIPTIVE_UNCERTAINTY` |
| 33 | Longer target-specific GPU/XLA NeuTra trace | Phase 32 hard receipts; fresh bank remains finite | `phase33-neutra-trace/` | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` |
| 34 | Extended optimization-time repair trace | Phase 33 finite trace with whitening residuals | `phase34-neutra-extended-trace/` | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED_REPAIR_TRIGGERED` |
| 35 | Affine theta preconditioning repair (initial full-bank factor) | Phase 34 residuals persist; target/measure unchanged | `phase35-affine-neutra-repair/` | `SUPERSEDED_MEASURE_MISMATCH_REPAIRED` |
| 35R | Affine theta preconditioning bound to training measure | Phase 35 mismatch identified; target/measure unchanged | `phase35r-affine-training-measure-repair/` | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` |
| 36 | Corrected continuation adjudication and support hypothesis | Phase 27--35R artifacts | `phase36-adjudication/` | `PASS_ADJUDICATION_CONTINUE_PHASE37_SUPPORT_LADDER` |
| 37 | Fresh particle-size/support ladder and nominated N=256 boundary | Phase 36 adjudication; unchanged theta protocol | `phase37-support-ladder/` | `PASS_SUPPORT_LADDER_HARD_GATES_NEUTRA_SUPPORT_STRESS_ROLE_LIMITED` |
| 38 | Validation-selected checkpoint repair on nominated N=256 bank | Phase 37 hard gates; unchanged target/measure | `phase38-checkpoint-repair/` | `PASS_CHECKPOINT_SELECTION_AUDIT_REPAIR_TRIGGERED_HISTORICAL_V2_1` |
| 39 | Train/validation/audit empirical-measure separation diagnostic | Phase 38 finite checkpoint receipts; unchanged target/measure | `phase39-measure-separation/` | `PASS_MEASURE_SEPARATION_SPLIT_DEFECT_IDENTIFIED_HISTORICAL_V2_1` |
| 40 | Fresh root-group-stratified NeuTra boundary and checkpoint trace | Phase 39 split-defect audit; target/measure unchanged | `phase40-root-group-stratified-boundary/` | `PASS_V2_2_ROOT_GROUP_BOUNDARY_MEASURE_DIAGNOSTIC_REPAIR_TRIGGERED` |
| 41 | Independent fresh-bank audit of frozen v2.2 training measure | Phase 40 root/measure receipts; target/protocol unchanged | `phase41-independent-audit-bank/` | `PASS_V2_3_INDEPENDENT_AUDIT_REPORT_REPAIR_TRIGGERED` |
| 42 | Two-bank replication under one frozen transport state | Phase 41 report; unchanged target/protocol/objective | `phase42-independent-bank-replication/` | `PASS_V2_4_TWO_BANK_REPLICATION_REPORT_REPAIR_TRIGGERED` |
| 43 | Third-bank support diagnostic with exact v2.4 state-hash reconstruction | Phase 42 report; unchanged target/protocol/objective and old training rows | `phase43-third-bank-support/` | `PASS_V2_5_THREE_BANK_REPORT_REPAIR_TRIGGERED` |
| 44 | Larger-N support diagnostic with one frozen v2.4 trainer per arm | Phase 43 report; unchanged target/protocol/objective and old training rows | `phase44-larger-n-support/` | `PASS_V2_6_LARGER_N_REPORT_REPAIR_TRIGGERED` |
| 45 | Independent N=512 replication with one frozen trainer per arm | Phase 44 report; unchanged target/protocol/objective and old training rows | `phase45-independent-n512-replication/` | `PASS_V2_7_INDEPENDENT_N512_REPORT_REPAIR_TRIGGERED` |
| 46 | Third N=512 support/proposal envelope diagnostic; no trainer retraining | Phase 45 report; unchanged target/protocol and all retained banks | `phase46-support-envelope/` | `PASS_V2_8_SUPPORT_ENVELOPE_REPORT_REPAIR_TRIGGERED` |
| 47 | Paired identity versus theta-space MH invariant mutation diagnostic | Phase 46 report; unchanged target/protocol and mutation-free target | `phase47-invariant-mutation/` | `PASS_V2_9_MUTATION_REPORT_REPAIR_TRIGGERED` |
| 48 | Paired identity versus independent-proposal MH mutation diagnostic | Phase 47 report; unchanged target/protocol and theta proposal density | `phase48-independent-proposal-mutation/` | `PASS_V3_0_INDEPENDENT_MH_REPORT_REPAIR_TRIGGERED` |
| 49 | Independent-proposal MH depth repair (two versus eight proposals per stage) | Phase 48 report; unchanged target/protocol/proposal and exact replay | `phase49-independent-proposal-depth/` | `PASS_V3_1_INDEPENDENT_MH_DEPTH_REPORT_REPAIR_TRIGGERED` |
| 50 | Defensive proposal-support repair with non-symmetric independent MH | Phase 49 report; unchanged target/base bridge and exact replay | `phase50-defensive-proposal-support/` | `PASS_V3_2_DEFENSIVE_SUPPORT_REPORT_REPAIR_TRIGGERED` |
| 51 | Mode-aware proposal-geometry repair with non-symmetric independent MH | Phase 50 report; unchanged target/base bridge, theta measure, and exact replay | `phase51-mode-aware-proposal-geometry/` | `PASS_V3_3_MODE_AWARE_GEOMETRY_REPORT_REPAIR_TRIGGERED` |
| 52 | Fresh paired uncertainty replication of identity, isotropic support, and mode-aware geometry | Phase 51 descriptive nomination; fresh pilot seeds; unchanged target/measure/depth | `phase52-fresh-paired-uncertainty-replication/` | `IN_PROGRESS` |

After every phase: preserve its unique root; run focused tests; classify the
failure; apply the adjacent repair note; record wall time and remaining pool;
refresh the next subplan with measured facts and hashes; then continue if its
entry gate passes. Candidate failure does not silently become direction
failure.

Phase 27's detailed subplan and repair note are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase27-subplan-2026-08-25.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase27-repair-refresh-2026-08-25.md`

Later subplans are written or refreshed only after the preceding receipt, so
their controls remain hypotheses rather than stale defaults.

Phase 37's receipt, repair refresh, and next subplan are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase37-result-2026-08-25.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase37-repair-refresh-2026-08-25.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase38-subplan-2026-08-25.md`

Phase 37 passed the common theta-measure gates and nominated the N=256 M0
receipt by retained-root count. Its held-out NeuTra moments remained poor, so
Phase 38 is an explicitly bounded checkpoint/objective repair. It does not
change the target, measure, proposal, or canonical LEDH boundary.

Phase 38's receipt, repair refresh, and next subplan are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase38-result-2026-08-25.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase38-repair-refresh-2026-08-25.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase39-subplan-2026-08-25.md`

Phase 38 repaired the checkpoint artifact boundary and found a descriptive
benefit for one affine arm, but did not reduce residuals enough for promotion.
Phase 39 therefore measures partition support before any objective or data
generation change.

Phase 39 identified a real design defect in the v2.1 split: validation was
selected by storage order and could be ancestry-correlated with the training
rows. Its report is retained as historical v2.1 evidence. The active repair is
v2.2-root-group-stratified: whole SMC root groups are allocated by a
deterministic subset-sum rule, with sign-balanced validation/audit partitions,
zero root overlap, and a complete row partition. Fresh Phase 40 receipts are
required before interpreting any new whitening diagnostic.

Phase 40's subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase40-subplan-2026-08-25.md`

Phase 40 passed its engineering and root-group boundary gates, but its finite
validation and audit partitions remained materially different from training.
The v2.2 result is therefore retained as a role-limited diagnostic and a
repair trigger, not as whitening evidence. The active version is now
`v2.3-independent-audit-bank`. Phase 41 freezes the v2.2 training rows and
weights, generates a new independent N=256 theta bank with a fresh seed, and
evaluates the unchanged NeuTra arms on that bank without selecting or tuning
on it. This separates a split/finite-holdout explanation from a broader
support or objective explanation.

Phase 40 receipts and repair refresh are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase40-result-2026-08-25.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase40-repair-refresh-2026-08-25.md`

Phase 41's subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase41-subplan-2026-08-25.md`

Phase 41 passed its hard engineering and numerical boundary after preserving
two runner repairs and one reporter repair. The independent N=256 bank was
closer to the frozen training support than the old 12-row holdouts, and the
identity transport residuals decreased descriptively, but all residuals
remained nonzero. This is evidence that the v2.1/v2.2 holdout comparison was
partly defective; it is not evidence of IID whitening. The active version is
now `v2.4-independent-bank-replication`. Phase 42 will generate two new
independent N=256 banks and evaluate both with one frozen trained state, so
bank-to-bank variability is measured before changing the objective or
architecture.

Phase 41 receipts and repair notes are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase41-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase41-repair-refresh-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase41-report-repair-refresh-2026-08-26.md`

Phase 42's subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase42-subplan-2026-08-26.md`

Phase 42 passed all engineering and finite target/status gates, but its
two-bank result triggered the predeclared repair branch. Bank B reproduced the
Phase 41 descriptive residual reduction; bank A was a support/mode outlier and
was worse than the old validation comparator for every arm. The result was
retained without pooling or dropping either bank. Phase 43 then generated an
independent N=256 bank C and evaluated A, B, and C after one reconstructed
trainer per arm. Its exact v2.4 state-hash gates passed and the report selected
the descriptive branch `bank_a_isolated_outlier_descriptive`: B and C were
closer to the old comparator, while A remained the clear outlier. This does
not establish whitening or a statistical ranking. The active version is now
`v2.6-larger-n-support-diagnostic`. Phase 44 generates an independent N=512
bank (with a separately recorded calibration-only 128-row hypothesis) and
evaluates A, B, C, and N=512 after the same frozen trainer state. No objective,
architecture, target, or whitening criterion changes.

Phase 42 result and repair refresh are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase42-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase42-repair-refresh-2026-08-26.md`

Phase 43's subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase43-subplan-2026-08-26.md`

Phase 43's result and repair refresh are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase43-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase43-repair-refresh-2026-08-26.md`

Phase 44's subplan, result, and repair refresh are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase44-subplan-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase44-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase44-repair-refresh-2026-08-26.md`

Phase 44 passed its engineering, target/status, finite, and exact v2.4
state-hash gates after one infrastructure interruption and a successful fresh
retry. The N=512 bank was descriptively better than the isolated N=256 bank A
on both displayed residuals in all four arms, but residuals remained material;
the report therefore kept whitening vetoed and objective change deferred. The
branch is descriptive evidence for a finite-bank/support hypothesis, not a
statistical ranking. Phase 45 generates a second independent N=512 bank and
evaluates both N=512 banks, A, B, and C after one unchanged trainer per arm.
No bank is pooled, selected, or used for training.

Phase 45's active subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase45-subplan-2026-08-26.md`

Phase 45 passed its target, status, measure, independence, finite, GPU/XLA,
transport-parity, fresh-use, and exact frozen-state-hash gates. Both
independent N=512 banks were below bank A on the two displayed transport
residuals in every arm, but N512-b was not below the historical comparator on
one covariance entry. The branch
`n512_replication_order_reproduced_but_support_mixed` is descriptive only.
Whitening remains vetoed, objective changes remain deferred, and no HMC or
canonical LEDH route is admitted. The next smallest artifact is a third N=512
bank plus a fixed-proposal support envelope; it does not retrain NeuTra or
alter the target.

Phase 45's result and repair refresh are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase45-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase45-repair-refresh-2026-08-26.md`

Phase 46's active subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase46-subplan-2026-08-26.md`

Phase 46 passed the pilot, hash, measure, finite-value, and exact proposal
recomputation gates. The third independent N=512 bank fell outside the first
two N=512 scalar support envelope (`n512_c_outside_two_bank_scalar_envelope`).
This is persistent finite-bank variability, not evidence that the proposal
receipt is stale, and it does not promote whitening or an objective change.
The next repair is a paired invariant-mutation diagnostic.

Phase 46's result and repair refresh are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase46-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase46-repair-refresh-2026-08-26.md`

Phase 47's active subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase47-subplan-2026-08-26.md`

Phase 47 passed its analytic fixture, target/status, theta-measure, pairing,
finite-tensor, GPU/XLA, and artifact gates after one harness-only false-gate
repair. The valid paired report found that two local isotropic MH steps moved
particles but did not reduce the declared between-replicate support-spread
vector. This is a candidate-method failure for that local kernel, not a target,
measure, or research-direction failure. Whitening, HMC, and canonical LEDH
remain closed. The next repair tests an independent-proposal MH kernel using
the exact defensive-mixture density in its acceptance ratio, so it can cross
separated proposal components without changing the target or measure.

Phase 47's result and repair refresh are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase47-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase47-repair-refresh-2026-08-26.md`

Phase 48's active subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase48-subplan-2026-08-26.md`

Phase 48 passed its fixture, target/status, proposal-measure, pairing/replay,
finite-tensor, GPU/XLA, and artifact gates. Its independent-MH arm moved in
all three replicates, but the two-step descriptive spread was not lower than
identity overall. This is a depth/candidate-method repair trigger, not a
target, measure, harness, or research-direction veto. The MathDevMCP audit of
the unconstrained bridge-symbol form produced an inadmissible counterexample;
the same identity after direct bridge substitution was certified by SymPy.
Whitening, posterior, HMC, and canonical LEDH remain closed.

Phase 48's result and repair refresh are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase48-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase48-repair-refresh-2026-08-26.md`

The active version is now `v3.1-independent-proposal-depth`. Phase 49 keeps
the exact q=20 target, defensive-mixture density, seeds, identity comparator,
and theta measure, but increases independent-MH depth from two to eight
proposals per nonterminal stage. The Phase 48 two-step receipt is frozen as
the comparator; no rows are pooled or used for training or selection.

Phase 49's active subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase49-subplan-2026-08-26.md`

Phase 49 passed its fixture, target/status, theta-measure, pairing/replay,
finite-tensor, GPU/XLA, and artifact gates. Eight independent proposals moved
particles at every nonterminal stage, but the predeclared three-metric spread
condition failed: negative-mode spread was larger than the frozen depth-two
comparator and ESS variability remained mixed. This is a candidate-depth
repair trigger, not a target, measure, harness, or research-direction veto.
The next version is `v3.2-defensive-proposal-support`: it keeps the q=20
target and annealing base `q` fixed while testing a broader candidate law with
the exact non-symmetric MH correction.

Phase 49's result and repair refresh are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase49-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase49-repair-refresh-2026-08-26.md`

Phase 50's active subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase50-subplan-2026-08-26.md`

Phase 50 passed its fixture, target/status, theta-measure, q-versus-r density,
pairing/replay, finite-artifact, GPU/XLA, and report gates. The isotropic
support component moved particles but did not satisfy the predeclared
three-metric spread condition against frozen Phase 49: only negative-mode
spread decreased, while theta-mean and covariance spreads increased. This is
a negative result for that support law, not a target, measure, harness, or
research-direction veto. Whitening, posterior, HMC, and canonical LEDH remain
closed.

Phase 50's result and repair refresh are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase50-result-2026-08-26.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase50-repair-refresh-2026-08-26.md`

The active version is now `v3.3-mode-aware-proposal-geometry`. Phase 51 keeps
the exact q=20 target, q-based bridge, theta measure, initial clouds,
resampling schedule, depth, seeds, and replay gates. It replaces only the
isotropic candidate component with a frozen two-mode full-covariance mixture
and evaluates the exact non-symmetric independent-MH correction. The direct
comparator is the frozen Phase 50 support arm; Phase 49 remains secondary.

Phase 51's active subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase51-subplan-2026-08-26.md`

Phase 51 passed all engineering, numerical, measure, replay, provenance,
finite-artifact, GPU/XLA, and report gates. Its mode-aware geometry arm was
descriptively below Phase 50's isotropic-support arm on the three primary
between-bank spread metrics, but ESS spread was worse. With only three paired
banks and inherited mode representatives, this is a nomination signal rather
than a statistical ranking or default decision. Whitening, posterior, HMC,
and canonical LEDH remain closed.

Phase 51's result is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase51-result-2026-08-26.md`

The active version is now `v3.4-fresh-paired-uncertainty-replication`.
Phase 52 will generate six fresh q=20 pilot banks and run identity,
isotropic-support, and mode-aware-geometry arms on identical initial clouds and
resampling streams. It will report paired uncertainty diagnostics without
pooling rows, tuning on the claim data, changing the q-based bridge, or
promoting a default.

Phase 52's active subplan is:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase52-subplan-2026-08-26.md`

Phase 35's initial result is retained as a superseded diagnostic. Its full-bank
affine factor did not match the 40-row optimizer measure, so it cannot support
a whitening comparison. Phase 35R is the corrected execution boundary.

Phase 35's detailed subplan, repair note, and result are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase35-subplan-2026-08-25.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase35-repair-refresh-2026-08-25.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase35-result-2026-08-25.md`

The initial affine oracle confirms only the full-bank finite chart construction.
The corrected Phase 35R oracle must be computed on the exact training measure
before any whitening interpretation. This refreshes Phase 35R and defers Phase
36 adjudication; it does not authorize HMC or canonical LEDH.

## 8. Phase 27 executable contract

The runner must:

1. construct the batch-native q=20 target with XLA enabled in a CPU-hidden
   reference lane;
2. assert `target.parameter_dim == 4`, evaluate a static `[N,4]` theta batch,
   and record target status and signature;
3. expose the internal state/innovation dimensions only as metadata and assert
   they are not particle dimensions;
4. evaluate a known affine chart `theta=c+A z` and verify
   `(v_theta-q_theta) == (v_z-q_z)` to floating-point tolerance;
5. call `second_order_etpf_transform` on a small weighted theta cloud and
   assert output shape `[N,4]`, finite target/status, and source-moment role
   diagnostics; and
6. emit a structured JSON/Markdown receipt with no overwritten files.

The phase is a contract audit, not an SMC authority admission. It intentionally
does not use a 60D state cloud, does not call HMC, and does not train NeuTra.

## 9. MathDevMCP audit protocol

MathDevMCP is an audit assistant, not a proof oracle. Before executing the
target runner, use these bounded checks and preserve their raw output under
the Phase 27 artifact root:

```text
PYTHONPATH=/home/ubuntu/python/MathDevMCP/src \
  /home/ubuntu/.venvs/mathdevmcp-mcp/bin/python -m mathdevmcp.cli \
  prove-or-counterexample \
  "(v_theta(theta)-q_theta(theta))=(v_theta(theta)+log_abs_det_A)-(q_theta(theta)+log_abs_det_A)" \
  --assumption "det(A) != 0"

PYTHONPATH=/home/ubuntu/python/MathDevMCP/src \
  /home/ubuntu/.venvs/mathdevmcp-mcp/bin/python -m mathdevmcp.cli \
  audit-math-to-code \
  "rank(G Q G^T) <= min(rank(G),rank(Q),rank(G^T))" \
  bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py
```

The first checks the narrow Jacobian cancellation claim. The second is a
scope-limited code-structure check for the already-closed rank argument. Any
`unverified`, `needs_boundary_clarification`, or tool failure is recorded as a
limitation; it cannot be converted into a pass by prose.

## 10. Stop conditions

Stop the continuation only if the declared four-dimensional target or common
support is unavailable, an exact fixture contradicts the corrected measure
identity with no in-scope repair, three focused infrastructure repairs repeat
without progress, the platform blocks the required diagnostic, or the
remaining campaign pool is exhausted. Poor whitening, low ESS, mode imbalance,
ETPF support excursions, or a failed NeuTra screen trigger repair and
adjudication, not an automatic whole-program stop.

## 11. Closeout requirements

The final continuation note must distinguish engineering correctness,
numerical validity, and scientific interpretation; include a decision table,
stochastic inference-status table, run manifests, MathDevMCP limitations,
strongest alternative explanation, overturning evidence, and the next smallest
artifact. It must state explicitly that the target is four-dimensional and the
UKF state is internal.
