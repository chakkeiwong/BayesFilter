# NeuTra Three-Mode Provenance And Evidence Closure Result (2026-08-17)

Plan: `docs/plans/bayesfilter-neutra-three-mode-provenance-and-evidence-closure-plan-2026-08-17.md`

## Verdict

The stale-checkpoint problem is closed. The active three-mode runner now
defaults to and fail-closed binds the reviewed `(128,128)`, six-stage checkpoint
selected at update 8,750, SHA-256
`b39c682030fb3ba8bafe863c747674db40b5d7c13e164c8445ddfab649ad93f6`.
The obsolete 1,000-update checkpoint is preserved as a failed capacity baseline
and can no longer be silently treated as the current candidate.

The one-seed concern is also closed at the bounded level tested here. Two fresh
component-aware training seeds independently passed their disjoint support
screens, fresh HMC tuning, shared sequential HMC, and exact three-component law
screens. Together with the original passing seed, there are now three viable
component-aware seeds. This is replication evidence for this analytic target,
not a universal success rate or method ranking.

Unknown-mode discovery is not closed. A genuinely mode-blind centered iid
Student-`t(3)` proposal family was tested and decisively failed importance
support before training. That family is rejected; mode-blind proposal design in
general remains an open repair problem.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Checkpoint provenance | Active runner binds reviewed SHA, target, architecture, selected step, and XLA identity | No mismatch in focused tests | Historical callers outside this runner may still name old artifacts | Retain fail-closed tests | No scientific result from identity alone |
| Fresh component-aware seed 1 | Training support, tuning, sequential HMC, and exact-law screens passed | No hard numerical, convergence, movement, component-law, or hash veto | One of two new seeds | Retain as viable replicate | No mode discovery or cross-target transfer |
| Fresh component-aware seed 2 | Training support, tuning, sequential HMC, and exact-law screens passed | No hard numerical, convergence, movement, component-law, or hash veto | One of two new seeds | Retain as viable replicate | No mode discovery or cross-target transfer |
| Naive mode-blind Student-t | Best global/median-batch ESS fractions `0.00400/0.00471`, required `0.0625/0.0625` | Proposal-support veto fired before training/HMC | Only one centered iid heavy-tail family tested | Design a target-query-driven discovery proposal without exact mode centers | No rejection of mode discovery in general |
| Native divergence | TFP HMC did not expose a native boolean/count | Status remains `not_exposed_by_kernel` | Native divergence unavailable | Preserve explicit nonclaim; do not relabel energy proxy | No zero-divergence claim |
| Geometry/application transfer | Not run in this campaign | Not assessed | Target-specific protocols remain incomplete | Continue varying-Hessian/KSC/German/SSL-LSTM lanes separately | No transfer or default-readiness claim |

## Fresh Replication Results

| Metric | Fresh seed 1 | Fresh seed 2 |
|---|---:|---:|
| Training checkpoint SHA-256 | `f999b051...6227b` | `cdb2a79a...51a5` |
| Selected update | `9,500` | `9,750` |
| Heldout weighted NLL | `4.42207` | `4.42540` |
| Proposal ESS fraction | `0.69353` | `0.69224` |
| Training support maximum component-mass error | `0.00488` | `0.00360` |
| Selected HMC kernel | `L=3`, epsilon `0.5047103` | `L=5`, epsilon `0.3359571` |
| Warmup / retained per chain | `2,000 / 1,000` | `2,000 / 1,000` |
| Retained maximum R-hat | `1.00584` | `1.00276` |
| Retained minimum bulk ESS | `1,059.42` | `1,437.97` |
| Retained minimum tail ESS | `700.17` | `1,455.02` |
| Exact component masses | `(0.51050, 0.28575, 0.20375)` | `(0.49900, 0.30675, 0.19425)` |
| All component 99% intervals contain truth | Yes | Yes |
| Every chain visited every component | Yes | Yes |
| Exact-law primary screens | Passed | Passed |

Fresh seed 1 aggregate hard-assignment transitions were:

```text
[[1491, 329, 218],
 [ 343, 691, 109],
 [ 207, 122, 486]]
```

Fresh seed 2 aggregate transitions were:

```text
[[1434, 342, 219],
 [ 334, 780, 113],
 [ 226, 104, 444]]
```

The selected kernels differ across seeds. The evidence supports viability, not
a ranking or a universal `L`/epsilon default. Replication 2 had a warmup
HMC-coordinate tail ESS of `380.06`, below the retained ESS requirement but not
a warmup veto under the shared controller; its retained minimum tail ESS was
`1,455.02`. No threshold was changed.

## Mode-Blind Preflight

The proposal construction used only a fixed zero center, iid Student-`t(3)`
marginals, and scale ladder `(1,2,4,8)`. Exact mixture parameters were used only
after sampling to evaluate importance support. The predeclared ESS fraction
threshold `1/16` corresponds to at least 256 effective rows in each nominal
4,096-row batch.

| Scale | Global ESS fraction | Median batch ESS fraction | Minimum batch ESS fraction | Maximum normalized weight |
|---:|---:|---:|---:|---:|
| `1` | `0.002409` | `0.003552` | `0.001116` | `0.03686` |
| `2` | `0.004001` | `0.004707` | `0.002260` | `0.02398` |
| `4` | `0.000855` | `0.001417` | `0.000719` | `0.07019` |
| `8` | `0.000152` | `0.000439` | `0.000269` | `0.25873` |

The best median batch corresponds to about 19 effective rows, not 256. Training
and HMC were correctly not launched. The next proposal must discover separated
regions through target queries, tempering, or another exploration mechanism;
simply making one centered distribution heavier-tailed is insufficient.

## Engineering Repairs

- The three-mode HMC runner now defaults to the reviewed capacity checkpoint.
- Reviewed-candidate admission binds checkpoint SHA, exact target signature,
  `(128,128)` architecture, six stages, selected update 8,750, and XLA.
- Fresh-replication admission additionally validates sibling artifact hashes,
  active plan, target tensors, passed support decision, selected step, memory
  growth, XLA, and fresh seed identity.
- Target covariance provenance comparison permits only `1e-14` absolute
  floating-point roundoff; material changes still fail.
- The training runner accepts an explicit plan binding and executes Git
  provenance commands with `cwd=ROOT`, so detached/user-service launches work.
- Active full-validation and gap-closure notes now classify the 2026-08-17
  failure as the obsolete small baseline rather than the capacity repair.

## Run Manifest

| Field | Value |
|---|---|
| Git commit recorded by serious runs | `3030d86df9cb00346df82c7c19f015c09c7c6e1f` |
| Environment | `tfgpu`; TensorFlow/TFP GPU/XLA; float64; TF32 disabled |
| Hardware | One RTX 4080 SUPER per concurrent lane |
| GPU allocation policy | Memory growth verified before logical-device initialization |
| Fresh training seeds | Replication IDs `1`, `2`; initialization roots `20360811`, `20460811` |
| Training batch / updates | `4,096 / 10,000` per fresh seed |
| HMC | Four chains; `L=(3,5,10,15,20,25)` tuning grid; shared sequential controller |
| Fresh seed 1 training/HMC wall | `736.44 s / 415.48 s` |
| Fresh seed 2 training/HMC wall | `654.31 s / 438.34 s` |
| Mode-blind preflight wall | `10.50 s` |
| Primary artifact root | `docs/plans/artifacts/neutra-three-mode-provenance-evidence-closure-2026-08-17/` |

One sandbox launch failed closed with no visible GPU and produced no scientific
artifact. The first user-service attempt for fresh seed 1 completed optimizer
work but crashed before artifact creation because Git provenance ran outside the
repository. It is infrastructure-only evidence. After the `cwd=ROOT` repair,
the same seed/settings were retried in fresh root `component-aware-replication-1-r3`.

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Both fresh component-aware seeds passed; naive mode-blind proposal failed support before training |
| Viable candidates | Original reviewed seed plus fresh component-aware seeds 1 and 2 |
| Statistically supported ranking | None; kernel and continuous-metric differences are descriptive only |
| Descriptive-only differences | Heldout NLL, ESS fractions above the support gate, selected kernels, runtime, acceptance, energy tails, and marginal moment misses |
| Default-readiness | Not established |
| Next evidence needed | Target-query-driven mode discovery proposal; separate geometry/application/SSL-LSTM confirmation |

## Post-Run Red Team

The strongest alternative explanation is that component-aware proposals and
starts make representation and mixing much easier than unknown-mode discovery.
The failed centered Student-t arm reinforces this distinction. The result would
be overturned for this target-specific replication claim by a valid hash or
target mismatch, a fresh reanalysis showing the exact-law gates were computed
from the wrong retained draws, or additional fresh component-aware seeds showing
systematic failure. The weakest evidence is the bounded seed count and absence
of native TFP divergence telemetry.
