# SSL-LSTM q=20 Gap-Closure Plan (2026-08-18)

Status: `PLAN_REVIEWED_READY_FOR_EXECUTION`

## Research Intent Ledger

| Field | Declaration |
|---|---|
| Main question | Can the q=20 SSL-LSTM target acquire an eligible global posterior archive, then support the previously specified posterior-predictive output-law comparison? |
| Candidate mechanism | Physical-coordinate distributed replica-exchange fixed HMC with the reviewed dense local-precision mass, followed only after admission by globally weighted NeuTra training and fixed-HMC validation. |
| Baseline | The failed identity-mass physical campaign: six-temperature ratio-0.50 ladder, four chains, 24 one-row CPU/XLA workers, warm-up R-hat `1.141610`, zero retained draws. |
| Positive control | Annealed SMC over the two known sign-separated regions: eight independent runs, ESS fraction `>=0.8783`, max normalized weight `<=0.03854`, mass interval `[0.405731, 0.536018]`. This is a two-known-region weight authority only. |
| Expected failure modes | Dense mass may still fail global warm-up; target may contain an undiscovered region; the physical archive may pass sampler diagnostics but not predictive equivalence; the GPU/XLA target identity or batch-native contract may drift. |
| Promotion criterion | A fresh dense-mass physical run passes finite/status/invalid-path/swap gates, warm-up modern R-hat, retained modern R-hat and ESS, repeated cold-hot-cold travel, and hot forgetting; then a frozen globally weighted NeuTra candidate passes proposal support and the same sequential HMC policy; finally posterior-predictive path tests run on retained posterior draws. |
| Promotion veto | Any nonfinite state/target/score/log-acceptance, invalid target status, broken permutation, stale identity, scalar training fallback, failed support, failed warm-up/retained convergence, failed travel/forgetting, or missing hash-bound artifact. |
| Continuation veto | The target adapter is not batch-native/XLA-valid, dense-mass path cannot fit the authorized compute cap, or no proposal can pass support after the bounded discovery hypotheses. |
| Repair trigger | Dense-mass warm-up failure triggers a bounded mass/step or ladder hypothesis test; support failure triggers proposal enrichment; predictive-test calibration failure triggers discriminator redesign before material comparison. |
| Explanatory diagnostics | Loss, acceptance, raw occupancy, SMC point estimates, energy tails, runtime, mode counts, and per-horizon effect sizes. These do not promote a posterior or rank methods. |
| Must not be concluded | No exhaustive mode-discovery claim from two known modes; no posterior correctness from R-hat/ESS alone; no NeuTra transfer from analytic controls; no parameter-equality claim for the non-identifiable LSTM. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Physical rather than failed NeuTra coordinates for global authority | SSL-LSTM root-cause result, 2026-08-10 | The learned chart separated the two stationary regions by about 23.7 latent units and was locally tuned | A physical chart may still be expensive or poorly conditioned | Dense-mass 100-transition canary and source-MAP parity | Reviewed repair hypothesis |
| Dense mass = mean of two mapped local precisions | 2026-08-11 dense-mass canary | Measured local precision span was approximately `0.5--367`; identity mass failed warm-up | Two-mode local geometry may not represent the full target | Fresh dense-mass warm-up and hot forgetting | Target-specific hypothesis |
| `step=0.35`, `L=8`, ratio `0.50` | Dense-mass canary; only arm passing all 100-transition selection screens | It passed finite/status, acceptance, communication, and hot-forgetting screens | Short canary may overstate stability | Fresh material warm-up/retained gates | Warm-start, not default |
| Four chains, six temperatures, 24 one-row workers | Existing distributed exact-HMC design | Preserves row independence and required multi-chain diagnostics | Cost may exceed budget | Checkpoint-equivalent runtime and hard wall | Reviewed execution setting |
| Posterior predictive: parameter draw then path simulation | User-established scientific endpoint | LSTM parameters are non-identifiable; output law is the target | Single plug-in mean can reject for irrelevant reasons | True-vs-true canary and five separate horizons | Reviewed endpoint |
| `n=1000`, `T=10,20,30,50,100`, alpha `0.01` | User-established diagnostic request | Fixed finite-horizon output-law tests with no omnibus test | Low power or dependence across horizons | True-vs-true canary and per-horizon receipts | Reviewed diagnostic default |

## Evidence Contract

### Phase A: Adapter and artifact preflight

- **Question:** Is the current SSL-LSTM target/value-score route batch-native, XLA-compiled, GPU-policy compliant for training, and deterministic in its target identity?
- **Baseline:** Existing target-integration receipt and q=20 batch-native profile.
- **Primary criterion:** focused contract tests pass; target signature and source hashes are recorded; no scalar or row-mapped optimizer path.
- **Vetoes:** target identity mismatch, nonfinite status, visible GPU in CPU diagnostic lane, missing memory-growth setup in GPU lane, or scalar training fallback.
- **Explanatory:** compile time, allocator bytes, one-step loss, gradient norm.
- **Artifact:** `docs/plans/artifacts/ssl-lstm-q20-gap-closure-2026-08-18/preflight/`.

### Phase B: Known-region mass and global-sampler canaries

- Re-verify the existing SMC recovery artifact without target reevaluation.
- Re-run the reviewed dense-mass `step=0.35/L=8` 100-transition canary in a fresh output root and seed. This is a mechanics/selection canary, not posterior evidence.
- **Primary criterion:** all finite/status/swap/invalid-path gates; every chain observes both signs and at least one hot local sign change; adjacent communication and acceptance band pass.
- **Veto:** any hard numerical or identity failure. A short-run lack of round trips is a repair trigger, not a direction veto.

### Phase C: Hypothesis test for full physical convergence

Run one fresh dense-mass material campaign only after Phase B passes. Use the existing distributed one-row worker route, with warm-up excluded from retained draws.

| Hypothesis | Test | Interpretation if it passes | Interpretation if it fails |
|---|---|---|---|
| H1: local geometry caused the identity-mass failure | Dense mass `0.35/L=8`, ratio `0.50`, balanced sign starts, four chains | Proceed to posterior archive validation | Test H2/H3; do not call target invalid |
| H2: the ladder endpoint/temperature spacing is the remaining problem | Bounded hotter-ladder canary using the previously tested ratio/endpoint arms, only if H1 fails | Fresh material run with selected ladder | Stop the sampler branch if no arm passes hot forgetting/communication |
| H3: the two known regions are not the full posterior | Target-query-driven discovery: independent prior/tempered/multistart proposals, cluster target-valid high-density points, compare discovered clusters with the two known regions; no posterior claim | Enrich SMC proposal and restart global validation | Treat two-region scope as an explicit limitation, not exhaustive evidence |

**Material gates:** warm-up modern rank/folded R-hat `<=1.05` on the recent window; retained R-hat `<=1.01`; bulk ESS `>=1000`; tail ESS `>=400`; at least one cold-hot-cold identity return per chain; hot local-HMC sign forgetting in every chain; finite/status/invalid-path/permutation checks; hard wall and artifact receipts. Warm-up is discarded.

### Phase D: Globally weighted NeuTra training

Only if Phase C yields an eligible physical archive, build a training cloud from retained physical draws plus the independently verified SMC weighting/proposal evidence. Tune architecture, learning-rate schedule, and update budget on disjoint calibration/selection/audit partitions. Use GPU/XLA, TensorFlow float64, memory growth before initialization, and batch size greater than one. Test at least two independent training seeds.

- **Primary criterion:** untouched proposal-law support and target/log-Jacobian finite checks on global held-out rows; no component collapse relative to the physical archive.
- **Veto:** support ESS failure, mode loss, scalar fallback, nonfinite checkpoint, stale checkpoint identity, or training on audit data.
- **Nonclaim:** training loss does not establish posterior correctness.

### Phase E: NeuTra sequential HMC and posterior-predictive endpoint

For each passing training seed, tune fixed HMC over `L >= 2` only; **No NUTS**. Use the shared sequential controller. Exclude warm-up. Require modern R-hat, bulk/tail ESS, finite/status/movement/energy diagnostics, and mode occupancy/crossing against the physical archive. Native TFP divergence remains `not_exposed_by_kernel`.

Then run the scientific endpoint: draw `n=1000` parameter values from retained posterior draws with replacement, simulate one complete path per draw, and compare with `n=1000` paths from the true generating parameters at each `T in {10,20,30,50,100}`. Use separate whole-path two-sample energy/permutation tests at alpha `0.01`; no omnibus or combined p-value. A true-vs-true canary must pass before material tests.

## Skeptical Plan Audit

1. **Wrong baseline:** avoided. The prior identity-mass physical run is the declared failure baseline; the dense candidate is bound to the separate dense-mass canary and will not reuse the identity checkpoint/kernel.
2. **Proxy promotion:** avoided. Loss, acceptance, SMC occupancy, and short canary ESS are explanatory or nomination diagnostics. Posterior promotion requires the physical convergence/travel gates and downstream predictive endpoint.
3. **Missing stop conditions:** present. Hard numerical, identity, budget, warm-up, retained, support, and predictive-calibration vetoes are explicit.
4. **Unfair comparison:** avoided. Candidate changes are isolated: dense mass changes only the metric in H1; ladder hypotheses are tested only after H1 failure; all target/data/seeds/artifact roots are disjoint.
5. **Stale context:** guarded by source/checkpoint/target signatures and unique output roots. Existing seed-B NeuTra draws are never reused as posterior evidence.
6. **Environment mismatch:** CPU global-authority routes hide GPUs before import and record XLA/CPU provenance; NeuTra training is GPU/XLA with memory growth and device receipts.
7. **Meaningless artifact:** every serious phase writes a manifest, exact command, environment, seeds, hardware, wall time, plan/result paths, and hashes. A run that stops before its primary criterion is classified as incomplete, not promoted.

**Audit verdict:** `PASS_WITH_BOUNDED_HYPOTHESES`. The plan is scientifically executable. A failed dense canary or material campaign triggers the named repair tests; it does not reject the SSL-LSTM target or NeuTra direction.

## Execution Order And Budget

1. Preflight and focused tests: short, no material compute.
2. Fresh dense-mass canary: bounded `<=2400 s`.
3. Full physical material campaign: measured dense candidate estimate about `14,405 s`; hard cap `28,800 s`, with 300 s finalization reserve.
4. Discovery preflight: bounded target-query diagnostic, `<=3600 s`, only if physical campaign does not establish a full-mode limitation.
5. NeuTra retraining/HMC/predictive endpoint: separate follow-up campaign after an eligible physical archive; do not launch implicitly from a failed Phase C.

The present execution will complete Phases A--C and the H3 discovery preflight if the material sampler is not admitted. It will not silently claim or launch Phases D--E without an eligible physical archive.

## Planned Artifacts

- Plan: this file.
- Phase A: `docs/plans/artifacts/ssl-lstm-q20-gap-closure-2026-08-18/preflight/`.
- Phase B/C: `docs/plans/artifacts/ssl-lstm-q20-gap-closure-2026-08-18/physical/`.
- Phase H3: `docs/plans/artifacts/ssl-lstm-q20-gap-closure-2026-08-18/discovery/`.
- Terminal result: `docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-result-2026-08-18.md`.
