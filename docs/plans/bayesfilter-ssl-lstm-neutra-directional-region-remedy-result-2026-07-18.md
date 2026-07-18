# SSL-LSTM NeuTra Directional-Region Remedy Result

Date: 2026-07-18

Decision: `LOCKED_CONTROLLED_AUDIT_PASSED_TARGET_CONFIRMATION_STILL_CLOSED`

## Outcome

The repaired controlled procedure passed every prospective operating gate.
The evidence supports split average/horizon inference with the locked
Rao-Blackwell/HAC candidate on the declared controlled laws. It does not
support a ranking among viable split candidates and does not establish target
HMC validity, G/H predictive equivalence, posterior correctness, model
adequacy, or default readiness.

The main repair was dimension-matched inference: one 20-dimensional average
region at `alpha=0.025` and ten two-dimensional horizon regions at
`alpha=0.0025`, with union-bound familywise control. Development nominated
Rao-Blackwell conditional moments, growing Bartlett HAC multiplier `3.0`, zero
ridge, and 12,288 retained draws per chain for the locked controlled audit.

## Primary Audit Results

The audit used 1,536 replications per family and 77 simultaneous one-sided
operating claims. All 11 primary families passed coverage, required-decision,
false-decision, and invalid-procedure gates.

| Family class | Weakest observed coverage | Weakest simultaneous coverage lower | Weakest required-decision rate | Weakest required-decision lower | Maximum false-decision upper | Invalid count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eleven primary families | `93.36%` | `91.07%` | `99.02%` | `97.93%` | `0.477%` | `0` |

All primary material families made the required decision in `1,536/1,536`
replications. The persistent negligible mean family made the equivalence
decision in `1,521/1,536`; all other primary equivalence families made it in
`1,536/1,536`.

Exact-boundary families were correctly inconclusive except for at most five
decisive outcomes in a family. Their largest simultaneous boundary-leakage
upper bound was `1.106%`, below the 5% contract. Near-boundary guard families
also passed their wrong-direction and invalid-procedure screens; high decision
power was intentionally not required arbitrarily close to the boundary.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit the controlled split-region procedure | All 77 locked operating claims passed | No numerical, coverage, false-decision, invalid, artifact, GPU, or XLA veto | Generalization beyond declared controlled laws | Integrate independently scaled target conditional moments | Target equivalence or posterior correctness |
| Preserve `K_avg=K_max=0.0068491` | Declared anchors and boundary audit passed | No threshold veto | Scientific relevance remains application-specific | Keep threshold fixed for target preflight | Universal optimality of the threshold |
| Keep G/H confirmation closed | Controlled evidence passed, but target scale and conditional-variance wiring are incomplete | Target-evidence boundary remains active | Actual target forecast dependence and scale provenance | Build and test target adapter under a new plan | G/H equivalence, material difference, or sampler ranking |

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | Passed: zero invalid audit replications; finite zero-ridge HAC and trust-region results; bound capacity receipt; GPU/XLA/TF32 provenance; no target access |
| Viable candidates | The locked split/Rao/`kappa=3` candidate passed. Other split candidates remain viable from development evidence. |
| Statistically supported ranking | None. Development arms used only 96 paired replications and no predeclared uncertainty analysis for ranking. |
| Descriptive-only differences | Split versus full decision rates, path versus Rao rates, HAC-multiplier differences, condition numbers, and runtimes |
| Default readiness | Not established; this was controlled statistical validation only |
| Next evidence needed | Independent target forecast scales, SSL-LSTM conditional observation variances, target adapter tests, then a separately reviewed G/H confirmation contract |

## Development And Capacity Evidence

At 8,192 draws, the historical full-20D/path/`kappa=1` baseline remained
underpowered: persistent negligible mean equivalence was `1/96`, and local
variance-ratio-1.25 detection was `19/96`. Split geometry raised the weakest
primary decision rate to at least `75/96` in every development arm. The
Rao-Blackwell arms detected the local variance family in `96/96` runs.

These differences are descriptive. The lexicographic development rule, which
considered validity and minimum coverage before false decisions and power,
nominated split/Rao/`kappa=3`. It did not prove that setting superior.

The first capacity receipt reported failure because its minimum-across-96
coverage gate was mathematically defective. For one family with true 95%
coverage, observing less than 93% coverage in 96 trials has probability about
20.48%; across 22 independent families, at least one such miss has probability
about 99.35%. Post-run review preserved the immutable receipt and replaced that
gate with development-only pooled coverage plus decision, false-direction, and
invalid screens. The first qualifying rung was 12,288 draws: pooled coverage
`2009/2112=95.12%`, weakest primary decision `95/96=98.96%`, maximum
wrong/boundary decision `1/96=1.04%`, and zero invalids.

## Negative-Result Separation

- Implementation failure: not supported. Focused scalar/batched, projection,
  authentication, conditional-moment, scale, boundary, and harness tests pass.
- Numerical failure: not supported. Audit invalid count was zero throughout.
- Tuning failure: not supported for the locked controlled candidate; no claim
  of optimal tuning is made.
- Evidence-design failure: supported and repaired for the original
  minimum-across-96 capacity coverage rule.
- Evidence against predictive validation, NeuTra, HMC, or the SSL-LSTM: not
  supported by this controlled program.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `b1606a3ec19643356705cf9d08ccf7c6495b6186` with unrelated dirty worktree preserved |
| Environment | conda `tfgpu`; TensorFlow `2.20.0`; TensorFlow Probability `0.25.0` |
| Device | physical GPU 1 through `CUDA_VISIBLE_DEVICES=1`; GPU 0 lane left untouched |
| Numerical policy | TensorFlow/TFP `float64`; XLA JIT; TF32 enabled; zero ridge; Bartlett bandwidth `69` |
| Locked candidate | split regions; Rao-Blackwell conditional moments; `kappa_HAC=3`; 12,288 draws/chain; four chains/arm |
| Audit seeds | `(28301, 28302)` |
| Audit size | 1,536 replications/family; 25 families total; 22 gated; 77 simultaneous operating claims |
| Audit wall time | `3317.2830` seconds |
| Total controlled GPU wall time | `4867.8235` seconds = `1.3522` GPU-hours, below the 2-hour cap |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-neutra-directional-region-remedy-plan-2026-07-18.md` |
| Result | This file |

Immutable receipts:

| Receipt | SHA-256 |
| --- | --- |
| Smoke | `89cc01955d088598d452abeda34c7f69623262d97b5b4b69c8ff029ac9604a77` |
| Development | `7ee2bff948fc174c689efa7162b83aef780332e87e838231b328eb3d11414593` |
| Capacity | `41ed67e1bcd304e7e97303eb7184f3b0f1ccc74bbd67b6be67a7fbc721045025` |
| Locked audit | `8653e21d3a964e7766fd87a8ffaa07593d480368a9e1580e294bb190968c5eaa` |

The locked audit records source hashes
`eb82b990...f3c063` for `predictive_equivalence.py`,
`9fd1b6e3...df858f` for the runner, and
`67d9f48d...830f76` for the pre-close plan. The plan and chapter were updated
after the immutable audit only to record the close; no audit receipt was
rewritten.

## Verification

| Check | Result |
| --- | --- |
| Full predictive-equivalence and remedy suite | `103 passed`; two dependency deprecation warnings only |
| Python compilation | Passed for predictive module, runner, and focused tests |
| Scoped diff whitespace | Passed |
| LaTeX build | Passed; `docs/main.pdf`, 407 pages |
| Chapter 28a references/citations | Resolved; three minor layout warnings only |
| Trusted GPU/XLA smoke | Passed; finite rows, zero ridge, no statistical claim |
| Development | Completed; nomination only, no ranking claim |
| Capacity ladder | Completed; original defective coverage gate preserved and reviewed |
| Locked audit | Passed all 77 claims in 3,317.283 seconds |
| Cleanup | GPU 1 released; no remedy runner process remains |

The full book still has 11 undefined citations and four multiply defined
labels in unrelated chapters. They predate and do not reference Chapter 28a.

## Post-Run Red Team

The strongest alternative explanation is that the controlled Gaussian laws
are easier than the actual SSL-LSTM posterior predictive computation. The
audit establishes the statistical procedure on those laws, not target
readiness. A target adapter could still fail because its scale is estimated
from confirmation data, because conditional observation variance is extracted
incorrectly, or because actual chain dependence differs materially from the
controlled AR families.

The conclusion would be overturned by a discovered feature-order projection
error, incorrect conditional-variance transformation, invalid union-bound
allocation, audit-seed reuse, or receipt/source mismatch. Focused tests and
receipt review found none of those defects. The weakest remaining evidence is
external validity to the target forecast path, which has deliberately not been
tested.

The execution harness should gain non-outcome progress telemetry for long
runs. Its atomic receipt preserved no-peeking, but candidate/family completion
heartbeats would improve supervision without exposing partial metrics.

## Handoff

Create a new concise target-integration plan before any G/H work. It should:

1. derive and test per-draw conditional observation variances from the frozen
   SSL-LSTM forecast equations;
2. freeze horizon scales from an independent calibration-only bank and record
   their provenance;
3. validate path versus conditional-moment estimates on target-shaped fixtures
   without opening G/H comparisons;
4. preflight actual target covariance conditioning and resource capacity;
5. authorize one locked G/H confirmation only if those gates pass.
