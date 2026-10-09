# Recovered history of multimodal NeuTra repairs

Historical review, 2026-09-25. Question: which earlier multimodal remedies
worked, and within what scope? This review reads existing reports and selected
structured results. It launches no training, target evaluations or sampling.

The old NeuTra architectures have the historical, author-code-unfaithful status
set by the [canonical policy](bayesfilter-neutra-canonical-architecture-policy-2026-09-25.md).
Their observations are recalled as historical evidence, not reused as current
canonical IAF validation. Physical-coordinate SMC and replica exchange are
separate methods, not the retired neural architecture. The review must
distinguish representing known modes, discovering unknown modes, estimating
relative weights and obtaining converged global chains.

## Findings

| Remedy | Historical result | Boundary |
|---|---|---|
| Defensive importance-weighted forward-KL training | The final two-mode confirmation passed its predeclared component-weight screen across eight fresh training seeds; subsequent three-mode work had three viable component-aware training/HMC seeds | Proposals knew component locations; these tests did not establish unknown-mode discovery or q20 transfer |
| Exact HMC after the weighted transport | A two-mode run made hundreds of retained mode transitions per chain and passed the declared mass screen; two fresh three-mode replications passed sequential and exact-law screens, with every chain visiting every component | Known analytic targets and mode-aware support; not a general success rate or method ranking |
| Physical annealed SMC on q20 | Passed the declared reliability checks for relative mass over the two already-known proposal-supported sign regions | No mutation crossed signs; unknown regions and full-posterior correctness remained unresolved |
| Physical parallel tempering / replica exchange on q20 | Eventually achieved repeated temperature round trips and hot local sign changes | Failed cold-chain warm-up convergence at the declared cap; zero retained posterior draws |
| Weighted-transport HMC on q20 | Training produced nominees, but both completed August 20 kernel verifications failed observation-weight folded R-hat | No admitted sequential posterior; rejects those pairs, not the entire method |
| Adaptive replay training | August 24 paired fixed/adaptive proposal screens executed, but whitening remained poor and adaptive importance ESS was weak | Twenty-four updates per seed were an implementation screen; no HMC or superiority result |
| Tempered reverse-KL transport ensemble | Implemented and exercised through staged feasibility, qualification and pricing work | The inspected August/September records do not supply a completed validated q20 posterior; resource, source and candidate failures must remain distinct |

## Strongest positive evidence and its sources

The [two-mode confirmation](bayesfilter-defensive-weighted-neutra-width128-updates10000-confirmation-result-2026-08-12.md)
used a historical six-stage, width-128 weighted IAF, batch 4,096 and up to
10,000 updates. Eight fresh fits estimated minority mass 0.20000077 with a
95% interval [0.19782562, 0.20217592], containing the known 0.2. This passes
the declared screen; containing truth is not an equivalence proof. Four of
eight matched reverse-KL fits omitted the minority mode. No paired superiority
test was declared. The structured confirmation result's `passed=true` and
interval were inspected in this recovery.

The [two-mode HMC result](bayesfilter-defensive-weighted-neutra-analytic-hmc-result-2026-08-12.md)
records 566--645 retained cross-mode transitions per chain. The minority mass
estimate was 0.18417 with 99% MCSE interval [0.16566, 0.20267], containing 0.2.
One mean and one covariance marginal diagnostic missed; the report preserves
those uncertainties rather than imposing an undeclared joint test.

The [three-mode closure](bayesfilter-neutra-three-mode-provenance-and-evidence-closure-result-2026-08-17.md)
reports the original viable seed plus two fresh training/HMC replications.
Every chain visited every component and all component 99% intervals contained
their known masses. Both fresh raw HMC results were inspected here and record
`three_mode_hmc_candidate_passed`, `sequential_passed=true` and
`analytic_primary_screens_passed=true`. The mode-blind centered Student-t
proposal failed importance-support checks before training. This distinction
is why the successful known-mode fits cannot answer mode discovery.

The [q20 annealed-SMC result](bayesfilter-ssl-lstm-q20-physical-annealed-smc-material-result-2026-08-10.md)
is a useful partial success omitted from the recent training handoff. Across
eight independent runs, estimated negative-region mass was 0.47087 with 95%
interval [0.40573, 0.53602]. Its terminal artifact was inspected here: all
declared weight, ancestry, precision and schedule-sensitivity gates passed,
with status `SMC_WEIGHT_EVIDENCE_PASSED`. This is explicitly two-known-region
evidence. The earlier [direct importance calculation](bayesfilter-ssl-lstm-q20-physical-global-repair-result-2026-08-10.md)
and [sparse AIS](bayesfilter-ssl-lstm-q20-physical-ais-repair-result-2026-08-10.md)
failed weight-reliability or schedule/movement checks; SMC repaired part of
that problem through weighting, resampling and preservation of both ancestries.

The old SMC report and the September 24 IAF training manifest name target
signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
That makes it relevant history, but matching a target label does not revalidate
the complete numerical/source provenance. No current-reference admission or
new estimate is issued by this review; the old neural policy is not bypassed.

## What remained unsuccessful or unfinished

The [physical replica-travel terminal addendum](bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-result-2026-08-10.md)
supersedes its earlier short-canary sections: the material continuation achieved
20 round trips, with returns [6,4,5,5] across chains, but its final warm-up
R-hat was 1.141610 against 1.05. All 1,000 transitions were discarded warm-up;
zero posterior draws were retained. It demonstrated a crossing mechanism
without completing posterior equilibration.

The [August 20 weighted-NeuTra continuation](bayesfilter-ssl-lstm-q20-neutra-global-mixing-continuation-result-2026-08-20.md)
finished two full kernel verifications, with maximum folded R-hat 1.08760 and
1.10207 against 1.01. Neither proceeded to sequential posterior sampling.
The [adaptive replay screen](bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-ab-comparison-result-2026-08-24.md)
also did not establish an adequate map or posterior.

For the tempered transport ensemble, the
[August feasibility result](bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-phase8-c1-result-2026-08-30.md),
[September recovery record](bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-reset-memo-2026-09-06.md),
and [later pricing continuation](bayesfilter-q20-ensemble-pricing-continuation-2026-09-21.md)
record graph costs, source/candidate failures and unfinished affordability work.
These are not evidence that the ensemble mathematics failed. The
[September 24 canonical-IAF study](bayesfilter-neutra-training-evaluation-results-2026-09-24.md)
also ended with incomplete HMC tuning and no retained posterior samples.

## Decision and inference status

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Remember weighted forward KL as a historically successful coverage mechanism | Known two/three-mode screens passed | No reported terminal hard veto in those selected tests | Proposal knowledge and architectural transfer | Use the source mechanism as a candidate for an explicitly justified canonical-map study if requested | Automatic reuse of old architecture or current superiority |
| Preserve q20 SMC as partial historical mass evidence | Declared known-region reliability checks passed | No recorded material gate failure | Unseen regions and current numerical provenance | Inspect full provenance before any new reference use | Full q20 posterior solved |
| Keep global q20 sampling unresolved | No accepted posterior in the inspected follow-ups | Cold convergence/tuning failed or required phases unfinished | Current canonical map plus effective global exploration | Resume only under the current authorized method/plan and remaining resources | Rejection of tempering, weighted training or NeuTra generally |

| Inference status | Finding |
|---|---|
| Hard veto screen | Known-mode successes coexist with mode-blind proposal failure, q20 convergence failures and unfinished ensemble work |
| Statistically supported ranking | None established by this historical recovery |
| Statistical evidence | Existing fresh-seed component-mass intervals and independent-run SMC interval, within their recorded scopes |
| Descriptive-only differences | Crossings, round trips, runtime and individual diagnostic values |
| Default readiness | Historical neural architectures remain retired; no replacement of canonical IAF |
| Next evidence needed | Current-configuration coverage-sensitive downstream results and reconciled reference provenance |

No archived tensor inventory was fully rehashed in this recovery. Conclusions
are a synthesis of the cited terminal reports, their supersession notices and
the selected structured results listed above. A later superseding result or
provenance failure would require updating this note. The strongest common
limitation is that successful training proposals already covered the modes;
the work did not establish a general unknown-mode discovery procedure.
