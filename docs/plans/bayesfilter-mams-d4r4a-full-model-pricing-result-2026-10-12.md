# Adaptive MAMS full-model cost result (D4R4a)

Date: 12 October 2026. The [bounded plan](bayesfilter-mams-d4r4a-full-model-pricing-2026-10-12.md)
asked whether the shared discarded MAMS proposer could run on four full-data
nonlinear targets, preserve provenance and candidate denominators, and price
short native trajectories within 7,200 GPU seconds. **It can do those things
after a status-only SIR source repair.** It did not learn a reliable trajectory
length on any of the four targets under the short pilot settings. These are
cost and short-path health results, not tuning membership, posterior delivery,
method ranking or release evidence.

| Full-data target | Pilot actual/reserved gradient work | Pilot / target wall seconds | Frozen `(epsilon, max count)` | 32-step native call seconds |
| --- | ---: | ---: | --- | ---: |
| PP-SGQF, 20 observations | 9,360 / 9,920 | 121.25 / 380.48 | `(2.2021,1)`, `(2.2436,3)` | 3.02, 5.94 |
| PP-UKF, 20 observations | 11,880 / 12,704 | 204.56 / 245.90 | `(1.7797,2)`, `(1.8187,4)` | 7.74, 11.78 |
| SIR-SGQF, 20 observations | 9,296 / 9,824 | 213.78 / 312.19 | `(1.7321,1)`, `(1.6949,3)` | 7.80, 14.84 |
| Structural UKF, 100 observations | 13,016 / 14,048 | 182.62 / 220.31 | `(1.4672,2)`, `(1.5657,5)` | 7.16, 13.37 |

All eight 32-transition-per-chain native prices had finite status, movement
health and no recorded hard veto. The prices reset four declared prepared
starts and use separate pilot/price seeds. They do **not** show that any policy
mixes globally, converges, reaches a reference posterior or is more efficient.
Across all four, neither the source-harmonic coordinate IACT nor the separate
slow-observable estimate met the pilot's reliability conditions. Every frozen
proposal therefore came from `sqrt(d)` or its declared factor-two neighborhood,
not from an empirically usable learned-length update. Several compiled count
supports are only one or two steps; this is a genuine limitation of the tested
pilot, not evidence of adequate travel.

Attempt 01 used commit `ca3c26034565aabb6eaab329714edb7bf0d8101c` and
priced both predator–prey targets. SIR preparation correctly rejected an
incomplete pair of target-status conditioning fields, leaving structural
unstarted. The source snapshot had omitted an existing SIR status-only
change in the shared working tree; earlier preflight tests had run in that
working tree. Attempt 02 used commit
`9c33936693d3f96816bd84fd723cd7dfa65e0bd6`, added the missing innovation
condition statistic and priced only SIR and structural. The old SIR and
repaired SIR gave exactly equal value, score, status code and validity at two
fixed full-horizon parameter points; the new status carried one additional
condition field. Nine isolated CPU-hidden-GPU tests passed, including all four
real full-horizon tiny proposer/native-price paths. These finite-point checks
supplement the inspected status-only source diff; they do not requalify old
posterior references.

Both GPU attempts used host GPU 1, TensorFlow 2.20.0, TFP 0.25.0, XLA and
verified preinitialization memory growth. The terminal audits checked 177
source hashes per attempt, compiled operator and geometry hashes, 80 distinct
recorded pilot/price seed pairs, every candidate/omission and durable price
bundle. The source commits differ; the four rows are combined for **cost
planning only**, not a same-source paired method comparison. The attempts
charged 652.049396 and 539.426641 supervisor seconds, respectively:
1,191.476036 of the original 7,200-second GPU allocation. The 6,008.523964
unused GPU seconds are released. Conservatively charging the full 1,800-second
local-check reserve leaves 151,019.615518 aggregate worker-seconds from the
post-D4R3 balance, with the preexisting overhead reserve unchanged.

The exact commands, attempt classification and checks are in the
[execution record](bayesfilter-mams-d4r4a-execution-2026-10-12.md). Structured
evidence is in the [combined terminal audit](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/full-model-pricing-terminal-audit.json)
and each attempt's `manifest.json`, `result.json`, `supervisor.json` and
`terminal-audit.json` under `N5/generic-adaptation/full-model-pricing-01/`
and `full-model-pricing-02/`. The run manifest fields include source commit,
tree, command, environment, GPU/memory policy, seeds, output paths and wall
time; the saved geometry design supplies the full observation-data hash.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close D4R4a cost preflight | Four full targets and eight proposals priced with intact provenance inside the original allocation | No terminal target/short-path veto after SIR repair; Attempt 01 source-integrity failure remains historical evidence | Unusable length update and one-seed, local-start short paths | Audit reference/source compatibility and price an adequately long discarded-window ladder before fresh qualification | Posterior validity, robust adaptation, comparative efficiency |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | The missing SIR conditioning field blocked Attempt 01; the scoped repair and isolated four-model tests passed. All eight final short paths recorded no numerical hard veto. |
| Statistically supported ranking | None. One seed and short paths provide no uncertainty-supported method or candidate ranking. |
| Descriptive-only differences | Epsilon/count, acceptance, pilot IACT status, native work and elapsed seconds in the table. |
| Default-readiness | No. MAMS remains experimental and the claim-bearing retained builder stays closed. |
| Next evidence needed | Versioned full-model reference compatibility or new references; longer discarded pilot windows with reliability outcomes; independently verified members; posterior/reference checks and fresh paired baselines with uncertainty. |

The strongest alternative explanation for finite local paths is that a
warm-start local factor and short durations explore too little posterior mass.
An independent reference or dispersed-start discrepancy would overturn any
optimistic reading of local health. The weakest evidence is the absent usable
IACT update and missing same-source posterior reference bridge. The next
phase must separate those defects from sampler mechanics without putting
R-hat, ESS or MCSE into tuning membership.

The reviewed [preconfirmation continuation](bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md)
specifies the reference audit and the 256/64 then conditional 1024/256 discarded
window ladder, including source guards, seed preservation, ceilings and the
subsequent confirmation-design refresh. It is the next active phase.
