# Structural UKF NeuTra Truth-Tail Campaign

Date: 2026-07-17

Status: `CLOSED_QUALIFIED_OWNER_ADJUDICATED_PASS`

Close note: the original evidence contract below is preserved, including its
conservative `bulk ESS >=1000` gate. The completed 4,000-draw-per-chain result
reached minimum bulk ESS `971.06`, while R-hat, tail ESS, target health, and all
five truth tails passed strongly. By explicit owner direction after inspecting
the result, the campaign closes as a qualified pass under an owner-adjudicated
bulk-ESS sufficiency threshold of `900`. See
`docs/plans/bayesfilter-structural-ukf-neutra-phase3-owner-adjudicated-result-2026-07-17.md`
and `docs/plans/bayesfilter-structural-ukf-neutra-campaign-close-2026-07-17.md`.

## Objective and research intent

Test whether a target-specific learned NeuTra transport can support valid,
converged HMC for the already admitted Chapter 18b structural UKF posterior,
and whether the five data-generating parameter values are non-extreme under
the retained posterior for the frozen synthetic fixture.

| Intent field | Frozen value |
| --- | --- |
| Main question | Does learned NeuTra produce health-valid, converged transformed HMC whose retained structural-UKF posterior does not place a generating parameter in an extreme tail? |
| Candidate | Fresh target-specific plain dense-IAF NeuTra transport followed by fixed-kernel HMC in latent coordinates |
| Expected failure modes | invalid target status, unstable reverse-KL training, unsuitable capacity or learning rate, HMC energy error, non-convergence by a 10,000-draw cap, or an extreme truth tail |
| Primary promotion criterion | health-valid transformed HMC; modern rank/folded split R-hat and ESS gates; all five `p_truth >= 0.05` on the frozen fixture |
| Promotion veto | any sampler hard veto, warm-up or retained convergence failure, or any `p_truth < 0.05` |
| Continuation veto | target identity/hash drift, target-status invalidity, non-finite computation, corrupted/missing evidence, exhausted campaign attempt budget, or `p_truth < 0.003` requiring scientific investigation |
| Repair trigger | localized harness, serialization, compilation, memory, tuning-grid, or sampler-kernel failure that leaves the target, data, method, criteria, hardware class, and total budget unchanged |
| Explanatory diagnostics | reverse-KL loss, heldout reverse-KL, acceptance, energy error, runtime, posterior means/intervals, and per-parameter tails |
| Forbidden conclusion | no filter exactness, calibration theorem, universal reliability, superiority over plain HMC, posterior correctness beyond the declared UKF target, production/default readiness, or frequentist p-value claim |

The existing fixture is intentionally retained because its target identity and
data hashes already passed independent checks. Its physical truth
`[0.8, 0.5, 0.7, 0.4, 0.25]` is not the center of the uniform prior box, whose
mean is `[0.515, 0.65, 0.515, 0.51, 0.51]`. A numerical pass is therefore a
`NONCENTRAL_ONE_SEED_TRUTH_TAIL_PASS`, not a central-truth coverage claim.

## Evidence contract

- Exact target: typed structural UKF posterior with target signature
  `e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665`,
  frozen observation hash
  `ab7885b135d8098c6e516e06733ef99399ea07f4a39292670b578da4a0efbae3`,
  horizon 100, and five probit source coordinates.
- Comparator/baseline: the exact standard-normal source-prior geometry,
  implemented as zero affine center and identity factor. This is a neutral
  warm start and a heldout reverse-KL comparator, not an admitted posterior
  geometry or an HMC promotion baseline.
- Training screen: four fresh 500-step GPU/XLA arms crossing hidden widths
  `(15,15)` and `(30,30)` with learning rates `1e-3` and `5e-3`; batch size 128,
  three IAF stages, ELU, and identical heldout batches. Heldout loss nominates
  a recipe but cannot establish posterior validity or rank viable arms without
  uncertainty support.
- Final training: one fresh 5,000-step GPU/XLA run using the nominated recipe;
  screen weights are not reused. Numerical validity and frozen/trainable
  parity are required.
- HMC: a fresh tuning probe and verification select a fixed step size and
  leapfrog count. Sequential warm-up is retained separately and excluded from
  posterior summaries. Warm-up and retained sampling each continue in chunks
  until their declared gates pass or 10,000 results per chain are reached.
- Convergence: modern R-hat is the maximum of rank-normalized split R-hat and
  folded rank-normalized split R-hat. Retained gates are max R-hat `<=1.01`,
  minimum bulk ESS `>=1000`, and minimum tail ESS `>=400`, with four chains.
- Health vetoes: non-finite states/log density, non-finite gradients, invalid
  target telemetry, or energy error below the declared log-accept threshold.
  TFP native divergence is recorded as unavailable when it is unavailable.
- Truth-tail diagnostic: for pooled retained physical draws,
  `F=(n_less+0.5*n_equal+0.5)/(N+1)` and
  `p_truth=2*min(F,1-F)`. This is a posterior-tail diagnostic, not a frequentist
  p-value. `p_truth>=0.05` passes this one fixture; `0.003<=p_truth<0.05`
  triggers exactly one fresh data-seed continuation; `p_truth<0.003` is a
  severe failure and stops for investigation.
- Artifact: unique attempt directories below
  `docs/plans/artifacts/structural-ukf-neutra-truth-tail-20260717/`, including
  configs, checkpoints, frozen transport, raw warm-up and retained tensors,
  hashes, result JSON/Markdown, and run manifests.

## Default and assumption audit

| Choice | Provenance and justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Frozen structural fixture | P5 admitted target/identity evidence; avoids changing data and target together | fixture-specific result mistaken for calibration | target and data hash replay; noncentral label | reviewed baseline |
| Zero/identity affine warm start | the uniform-box plus probit chart gives an exact standard-normal source prior | posterior far from source prior and training starts poorly | four-arm short screen and heldout comparison | warm-start hypothesis |
| Dense-IAF, three stages | existing BayesFilter NeuTra implementation and successful prior model campaigns | insufficient capacity for structural posterior | width cross and downstream HMC | baseline hypothesis |
| Widths 15 and 30 | dimension-scaled versions of established `3d` and `6d` widths | both underfit or wider arm overfits | common-heldout reverse KL and final HMC | hypothesis |
| Learning rates `1e-3`, `5e-3` | established BayesFilter screen values, not assumed transferable | divergence or slow learning | non-finite/gradient/status telemetry and heldout loss | hypothesis |
| 500/5,000 training steps | existing short-screen/serious-rung convention | screen misnominates or final undertrains | fresh final heldout check and actual HMC validity | bounded budget choice |
| Four HMC chains | minimum supported by modern split-chain diagnostics | insufficient ESS or poor mode exploration | adaptive caps and per-parameter diagnostics | reviewed diagnostic minimum |
| One frozen seed first | owner cost policy | isolated stochastic pass/failure overinterpreted | explicit one-seed wording and conditional second seed | reviewed policy |

## Phases, repair, and handoff

### Phase 1: harness and canary

Implement structural training, selection, HMC, archival, and truth-tail paths.
Run import/compile/unit checks and one batched GPU/XLA target canary. Required
artifacts are source files, test output, and canary JSON. Stop on target/hash
drift, invalid telemetry, GPU absence after trusted probing, or a mathematical
target mismatch. A passing canary hands the exact target and commands to Phase
2.

### Phase 2: target-specific training

Run the four 500-step screens, write the selection record, then run one fresh
5,000-step final job. Required checks are GPU placement, XLA compilation,
memory growth, no NumPy or sample-axis Python loop in the active training path,
target health, common-heldout evaluation, and frozen/trainable parity. Stop on
non-finite or target-invalid training, or after two localized repair attempts.
A finite selected frozen transport with valid hashes hands its exact payload to
Phase 3.

### Phase 3: transformed HMC and truth tail

Tune only the frozen transformed target, verify tuning on a disjoint seed, then
run adaptive sequential warm-up and retained sampling with the stated caps.
Archive warm-up and retained latent/source/physical draws separately. Compute
modern convergence, ESS, health, and truth tails. Stop on a hard sampler veto,
cap failure, severe truth tail, or after two localized repair attempts. A
complete result hands its exact classification and artifact paths to Phase 4.

### Phase 4: documentation and closeout

Write a dedicated NeuTra LaTe chapter covering the method, implementation,
diagnostics, cost policy, limitations, cross-model evidence ledger, and the new
structural result. Update `docs/main.tex`, build the monograph, and write the
campaign result/close record. Stop if the document does not build or if a claim
cannot be tied to an inspected artifact.

At the end of every phase: run the focused checks; write or refresh the result
record; refresh the next phase instructions; review suitability, consistency,
artifact coverage, and boundary safety; continue unless a real continuation
veto fired. Localized repairs use fresh versioned attempt directories and never
overwrite prior evidence.

## Commands and budget

The harness supplies exact stage commands after implementation. All GPU stages
must run with trusted/elevated GPU access and TensorFlow memory growth enabled.
The campaign allows four screen jobs, one final job, one HMC campaign, and at
most two localized repair retries per phase. The first-seed campaign is capped
at eight GPU wall-clock hours. A second data seed is outside the first-seed
budget and is authorized only by the predeclared marginal-tail rule.

## Skeptical pre-execution audit

Audit verdict: `PASS_AFTER_REVISION`.

- Wrong baseline risk: the historical plain-HMC and failed posterior-mode
  affine route do not answer the truth-tail question. They were removed from
  promotion and continuation gates. The standard-normal source-prior geometry
  is retained only as a transparent initialization and loss comparator.
- Proxy risk: heldout reverse KL nominates training settings only. Actual
  transformed-HMC health, convergence, ESS, and truth tails decide the result.
- Missing stop-condition risk: target invalidity, non-finite computation,
  sampler health, convergence caps, truth-tail thresholds, attempt count, and
  total wall time are explicit.
- Unfair comparison risk: no claim of superiority is made and no plain-HMC
  ranking is attempted.
- Hidden-default risk: architecture, optimizer, learning rates, affine start,
  seeds, thresholds, and budgets are recorded above with failure diagnostics.
- Stale-context risk: P5's `COMPARATOR_BLOCKED_GEOMETRY` answered an older
  comparator contract and is historical under the owner's revised criterion;
  the already admitted target identity remains current.
- Environment risk: GPU/XLA and memory growth are checked before serious work;
  CPU-only results cannot support training or HMC claims.
- Artifact-answerability risk: raw retained physical draws and hashes make the
  truth-tail calculation independently reproducible; run manifests bind code,
  command, environment, seeds, device, plan, output, and wall time.

No material flaw remains that makes a successful command unable to answer the
declared one-fixture question. Execution may begin.
