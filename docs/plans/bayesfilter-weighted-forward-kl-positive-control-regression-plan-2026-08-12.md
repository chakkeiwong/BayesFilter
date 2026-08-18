# Weighted forward-KL NeuTra positive-control regression plan (2026-08-12)

Status: `AUDITED_EXECUTION_AUTHORIZED`

## Research intent ledger

| Item | Contract |
|---|---|
| Main question | Does target-weighted forward-KL NeuTra retain valid downstream corrected-HMC behavior on targets where plain reverse-KL NeuTra previously worked, and does it remain viable on a genuinely three-mode target? |
| Mechanism under test | Fit the same dense autoregressive transport family either by self-normalized target-weighted forward KL or matched reverse KL, freeze each transport, retune fixed-length HMC in its latent coordinates, and assess retained physical draws against the strongest target-specific authority. |
| Expected failure mode | Importance weights may degenerate; an inherited proposal may omit a mode or relevant tail; forward KL may need different capacity or optimizer settings; heldout NLL may improve while corrected HMC remains poorly conditioned; or short chains may look stable inside one region. |
| Promotion criterion | Per-target corrected HMC passes the canonical sequential R-hat/ESS and numerical screens and the target-specific posterior-reference screens. A weighted arm must pass independently; it need not rank above reverse KL. |
| Promotion veto | Nonfinite value/score/state/Jacobian, invalid target status, corrupted identity/archive, missing component, failed retained R-hat/ESS, or failed target-specific reference screen. |
| Continuation veto | Wrong mathematical target, broken target/score derivative parity, proposal with zero support over required target regions, GPU memory-policy failure, artifact corruption, or exhaustion of the eight-hour campaign cap. A valid but failed candidate is a repair trigger, not a research-direction veto. |
| Repair trigger | Weight ESS collapse triggers proposal repair; only one missing mixture component triggers mode-blind versus component-aware proposal comparison; failed HMC after valid training triggers target-specific capacity/optimizer or HMC retuning; reference disagreement after sampler readiness triggers diagnostic and target audit. |
| Explanatory diagnostics | Training/heldout NLL, importance ESS and maximum weight, latent weighted moments, acceptance, finite energy-error tails, runtime, component transition matrix, and descriptive arm differences. |
| Must not be concluded | Passing does not prove exhaustive mode discovery, equality of stochastic methods, sampler superiority, real-SmallNK validity, or readiness to make weighted forward KL the repository default. |

## Target ledger and exact comparators

| Target | Classification | Reference authority | Required claim |
|---|---|---|---|
| Correlated ill-conditioned Gaussian, dimension 4 | Analytic control; old timed-out paper-suite run is not prior success | Exact normalized Gaussian moments and density | Both arms pass corrected-HMC and marginal analytic screens. |
| Unequal-weight two-component mixture, dimension 4 | Completed weighted success | Exact mixture law; terminal four-root result dated 2026-08-12 | Existing result is preserved as prior evidence, not rerun unless a shared-harness regression fails. |
| Unequal-weight three-component mixture, dimension 4 | New scientific target | Exact normalized mixture with weights `(0.5, 0.3, 0.2)`, non-collinear means, distinct rotated covariances | Weighted arm observes all components, each 99% responsibility-mass interval contains truth, each component has transitions, and corrected HMC passes sequential screens. No joint test. |
| `nk_like_mild_smooth`, dimension 9 | Credible one-seed prior plain-NeuTra success | Exact local target value/score plus historical plain result (`R-hat=1.001575`, ESS min `2541.5`, zero reported divergences) | Fresh matched reverse and weighted corrected-HMC arms independently pass current sequential screens; descriptive agreement is not a ranking. |
| `nk_like_strong_smooth`, dimension 9 | Credible one-seed prior plain-NeuTra success | Exact local target value/score plus historical plain result (`R-hat=1.002277`, ESS min `1234.6`, zero reported divergences) | Same as mild; this is the primary varying-Hessian regression. |
| German credit `gamma_scales2`, dimension 51 | Credible three-seed prior local plain-NeuTra success | Committed data and Stan/PyStan reference moments; historical square R-hat about `1.009`, ESS min about `794--877` | Fresh target-specific weighted arm and matched reverse comparator pass current corrected-HMC/reference screens. |
| Funnel and dimension-100 ill-conditioned Gaussian | Fresh matched-baseline targets, not prior successes | Analytic density; exact moments where finite/available | Establish a fresh reverse-KL baseline before interpreting weighted results. |
| BayesFilter LGSSM Phase 20 | Reference-capable but weak historical evidence | Deterministic two-dimensional quadrature | A new proper baseline is required; the 256-draw historical run cannot promote either arm. |

The active execution order is three-component mixture, mild/strong varying
Hessian, German, then fresh-baseline targets. This order maximizes scientific
discrimination under the bounded campaign and does not use failure on one target
to skip a later repair target unless a true continuation veto fires.

## Three-component mathematical target

Use a four-dimensional normalized Gaussian mixture

```text
p(theta) = sum(k=1..3) pi_k N(theta; mu_k, Sigma_k),
pi = (0.5, 0.3, 0.2).
```

The means form a non-collinear triangle in their first two coordinates and also
differ in coordinates three and four. Each covariance is generated by a distinct
full-rank lower-triangular factor, giving different scales and rotations. Exact
responsibilities are

```text
r_k(theta) = pi_k N(theta; mu_k, Sigma_k) / p(theta).
```

For retained draws, estimate each component mass by the mean responsibility,
with chain-aware batch-means MCSE and a separate two-sided 99% interval. Report
all marginal mean, covariance, and component-conditional moment intervals, but
do not combine them into a joint test or omnibus p-value. Hard assignments are
used only for component-presence and transition diagnostics.

Two proposal questions are deliberately separate:

- component-aware overdispersed mixture: tests whether weighted training can
  represent a known globally covered target;
- one mode-blind overdispersed Gaussian: tests recovery without supplying mode
  locations and is a later difficulty rung, not a prerequisite for validating
  the generic three-component mechanics.

## Evidence contract

| Role | Evidence |
|---|---|
| Primary promotion | Corrected HMC under `bayesfilter_neutra_sequential_hmc_v1`: retained maximum modern R-hat over latent/physical coordinates `<=1.01`, bulk/tail ESS `>=400`, plus the target-specific reference screen. |
| Hard veto | Any nonfinite required tensor, target-status invalidity, no movement, positive native divergence if exposed, corrupt receipt, missing required mixture component, or posterior-reference failure. |
| Repair only | Heldout NLL regression, importance ESS shortfall, HMC tuning rejection, warm-up/retained cap, or a candidate-specific component miss when the target/harness remain valid. |
| Explanatory only | Acceptance, finite energy-error tails, training loss, per-seed runtime, tail maxima, and uncalibrated differences between viable arms. |
| Exact baseline | Same target, transport class, initialization distribution, candidate capacity, update count, batch size, precision, hardware class, HMC controller, and reference diagnostics. Training objectives necessarily consume different row laws, which are recorded. |
| Artifact | Unique versioned roots below `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/`, with run manifest, target/proposal signatures, seeds, tuning, archived warm-up/retained draws, result, and hashes. |

No continuous metric ranks weighted versus reverse KL without paired uncertainty.
Passing means that an arm remains viable under the declared screen.

## Default and assumption audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| TensorFlow/TFP, float64, GPU/XLA | Repository default plus prior analytic campaign | Matches serious NeuTra and corrected-HMC path | device or compilation drift | trusted GPU preflight, memory-growth verification, XLA canary |
| Dense IAF | Matched baseline family | Existing shared weighted/reverse trainer permits exact architectural parity | insufficient capacity for a target | target-specific capacity ladder |
| Three-mode initial `(64,64)`, 3 stages | Convenience hypothesis, not a default | Smallest nontrivial step above mechanics | underfits separated modes | 200-update canary and disjoint heldout/component coverage |
| Three-mode serious `(128,128)`, 6 stages, 10,000 updates | Warm start from completed two-mode campaign, not promoted across targets | Direct capacity stress for an additional mode | wasted budget or still inadequate | canary precedes serious rung; checkpoint curve and heldout audit |
| Batch 4096, LR `1e-3` | Inherited two-mode hypothesis | Stable prior weighted training | target-specific optimization mismatch | clipping frequency and 200/1,000-update budget ladder; test `3e-4` or `3e-3` only if triggered |
| Component-aware proposal covariance multiplier 1.5 | Inherited analytic proposal hypothesis | Full support and modest overdispersion | proposal leaks mode locations and makes discovery too easy | label result as representation test; run mode-blind rung separately |
| 99% per-component intervals | Owner-selected prior diagnostic level | Individual conservative diagnostics without fabricated joint test | autocorrelation MCSE underestimation | chain-aware batch means; report every interval separately |
| Four chains, mode-aware starts for mixture | Canonical minimum and anti-initialization diagnostic | Ensures all basins are tested | starts can mask transition failure | require transitions involving every component; occupancy alone is insufficient |
| HMC `L=(3,5,10,15,20,25)` and target-tuned epsilon | Existing reviewed fixed-grid policy; `L=1` forbidden | Covers materially different trajectory lengths | grid misses stable region | one bounded smaller-epsilon repair grid |
| Mild/strong source parameters | Directly anchored to `dsge_hmc` benchmark source and June 4 result | Reproduces the actual prior-success geometry | source drift during port | fixed probe value/score parity before training |
| German architecture/budget | Historical values are warm starts only | Dimension 51 and prior local success make them useful first arms | forward KL needs different proposal/capacity | target-specific importance/proposal and capacity canary before serious run |

Every substantially different target gets its own canary, heldout selection,
capacity/optimizer decision, HMC tuning, and reference screen. A setting passing
one target is never silently promoted to another.

## Execution ladder

1. Implement target-independent TensorFlow Gaussian-mixture diagnostics for any
   static component count: target moments, responsibilities, component mass and
   conditional moments, chain-aware intervals, hard-assignment presence, and a
   full transition-count matrix.
2. Add the exact three-component target and proposal to the matched analytic
   training runner. Remove hard-coded dimension/two-mode assumptions from the
   touched path while preserving historical two-mode schemas and artifacts.
3. Add focused tests for one-component, two-component, and three-component
   truth; component permutation; missing-component detection; conditional
   moments; transition matrix; nonfinite rejection; and the explicit absence of
   a joint-test decision.
4. Run CPU-only focused reference/mechanics tests. These cannot support a
   scientific or GPU-readiness claim.
5. Run a trusted GPU/XLA 200-update three-mode canary with disjoint selection and
   audit clouds. If valid, run the target-specific budget/capacity ladder rather
   than immediately inheriting the two-mode 10,000-update arm.
6. Freeze a weighted candidate by the pre-HMC heldout rule, tune fixed-length HMC
   on disjoint seeds, then run canonical archived sequential HMC and exact
   three-component diagnostics.
7. Port and probe-tie the mild/strong varying-Hessian targets. For each, run
   matched reverse and weighted proposal/architecture canaries, freeze viable
   candidates, retune HMC separately, and apply current sequential screens.
8. Run German only after its committed data/reference loader and batched target
   path pass parity. Then establish fresh baselines for funnel, dimension-100
   Gaussian, and quadrature LGSSM as time remains.
9. Write a result note separating hard vetoes, viable arms, descriptive-only
   differences, statistical ranking status, default-readiness, and next evidence.

## Compute and stop budget

- Total campaign wall cap: eight hours from serious execution start, inherited
  from the user's explicit allowance. Test and implementation time is recorded
  separately.
- GPU processes must set `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import
  and verify growth. At most one process per GPU unless a measured memory canary
  supports co-residency; no full-device preallocation.
- Three-mode canary: 15 minutes; serious training ladder: 90 minutes; HMC tuning
  plus sequential confirmation: 90 minutes.
- Mild and strong: 75 minutes each including matched arms and HMC.
- German: up to 120 minutes after preflight.
- Remaining time is reserved for fresh-baseline targets and terminal artifacts.
- One infrastructure retry is allowed per unchanged rung. Scientific/tuning
  repairs must remain within the target allocation and preserve prior failures.

## Pre-mortem

- A run could pass heldout weighted NLL because a component-aware proposal
  supplies all modes, yet HMC could still be unable to transition. Corrected HMC
  component transitions and mass intervals are primary; NLL is not.
- A run could pass R-hat with chains stationary in separate modes. Per-component
  mass intervals and transitions involving every component distinguish this.
- A three-mode run could look wrong because diagnostics assume component 1 is a
  special minority. Generic vector-valued diagnostics and permutation tests
  distinguish this before execution.
- Weighted versus reverse comparisons could be unfair because weighted training
  evaluates target and proposal while reverse training evaluates the transformed
  target. The manifest reports target-evaluation counts and wall time separately;
  no speed or efficiency ranking is made without equal-cost analysis.
- Varying-Hessian failure could be a bad transferred MAP/Hessian proposal rather
  than evidence against weighted forward KL. Importance ESS and proposal-shell
  diagnostics fire the proposal repair before transport interpretation.
- A historical “success” could be only a short or timed-out run. Only German and
  the two June 4 varying-Hessian artifacts are regression successes; funnel,
  ill-conditioned dimension 100, and LGSSM require fresh baselines.

## Skeptical pre-execution audit

| Audit question | Finding and repair |
|---|---|
| Wrong baseline? | Corrected. The matched reverse arm uses the identical transport family and downstream controller. Exact analytic/Stan/quadrature authorities remain primary where available. Historical funnel/ill-Gaussian timeout is excluded from success claims. |
| Proxy promoted? | Corrected. Loss, NLL, proposal ESS, canaries, acceptance, and short chains only nominate or explain. Corrected sequential HMC plus reference agreement is required. |
| Missing stop conditions? | Corrected. Harness/target/device corruption are continuation vetoes; candidate training or HMC failure triggers repair and does not reject the research direction. |
| Unfair comparison? | Corrected. Architecture, initialization, precision, controller, and diagnostics match. Objective-specific row generation is explicit; runtime/target-call differences are descriptive unless equal-cost uncertainty is run. |
| Hidden inherited defaults? | Corrected. Architecture, LR, proposal scale, intervals, chain starts, grid, and budgets are classified above. Cross-target values are warm starts only. |
| Stale context? | Corrected. The completed four-root two-mode HMC result is preserved. The timed-out paper-suite target and weak 256-draw LGSSM artifact are not promoted to success controls. |
| Environment mismatch? | Corrected. Serious paths are TensorFlow/TFP GPU/XLA float64 with verified memory growth. CPU is only for focused independent checks. |
| Artifacts answer the question? | Corrected. Every claim-bearing arm preserves frozen transport identity, tuning, all warm-up/retained chunks, sequential diagnostics, target-specific reference diagnostics, and hashes. Training output alone cannot answer the question. |

Audit verdict: `PASS_FOR_STAGED_EXECUTION`. The exact three-component rung is
ready after generic diagnostic tests pass. Later targets are authorized only
after their target-specific proposal, capacity, reference, and source-parity
preflights; that staging is scientific scope control, not a reason to stop after
one candidate failure.
