# NeuTra Three-Mode Provenance And Evidence Closure Plan (2026-08-17)

## Research Intent Ledger

| Field | Predeclared statement |
|---|---|
| Main question | Which apparent three-mode NeuTra gaps are stale-checkpoint defects, which are failures of the small transport, and which remain unsupported scientific claims? |
| Candidate under test | The reviewed `(128,128)`, six-stage weighted forward-KL transport selected at update 8,750, plus fresh independently initialized replicas and a separately labeled mode-blind proposal arm. |
| Exact comparator | The analytic three-component Gaussian mixture with weights `(0.5, 0.3, 0.2)`, means/covariances fixed by target signature `3f5c692fa2d6c985c652ddad7394031d837f3dbd3e31ee14bbc8db62ad4a3a55`. |
| Expected failure modes | Obsolete checkpoint selection; insufficient flow capacity; seed-sensitive optimization; importance-weight collapse under a mode-blind proposal; locally moving but mode-trapped HMC. |
| Promotion criterion | A claimed closure must have its own evidence: fail-closed checkpoint identity tests for provenance; fresh target-law and downstream HMC evidence for replication; proposal-support and downstream HMC evidence for mode-blind coverage. |
| Promotion veto | Wrong checkpoint/target/config identity; nonfinite training or target; invalid hashes; `L < 2`; failed modern R-hat/ESS/movement/energy checks; failed exact component-mass or transition screens. |
| Continuation veto | Broken target/score/Jacobian parity, unavailable required artifact, GPU memory-growth/XLA violation, resource conflict, or exhaustion of the bounded campaign budget. A failed candidate is not a continuation veto when the next rung tests its stated repair. |
| Repair trigger | Fresh-seed failure triggers target-specific architecture/training diagnosis. Mode-blind proposal ESS/support failure triggers proposal redesign before HMC. |
| Explanatory only | Loss, acceptance, runtime, isolated moment error, and latent interpolation geometry. |
| Must not be concluded | No unknown-mode discovery, universal NeuTra validity, cross-target transfer, SSL-LSTM readiness, or method superiority from component-aware three-mode success. |

## Evidence Contract

| Question | Baseline | Primary criterion | Veto diagnostics | Artifact |
|---|---|---|---|---|
| Does the active runner select the reviewed candidate? | Obsolete SHA `57b21c...` versus reviewed SHA `b39c68...` | Runner default and eligibility check bind the reviewed SHA, target signature, architecture, XLA status, and selected update | Any mismatch fails before tuning/HMC | Focused tests and this result note |
| Was the 2026-08-17 failure a current-candidate failure? | Small 1,000-update result versus existing 10,000-update result | Active notes distinguish the failed baseline from the already-passing capacity repair | Contradictory active reset/result instruction | Corrected result/reset notes |
| Is the repaired result repeatable? | Existing seed is a baseline, not a replicate | At least two fresh training seeds pass disjoint heldout/support checks and shared sequential HMC/exact-law screens under the same target-specific protocol | Any seed's hard numerical, support, tuning, sampler, or exact-law veto | Fresh versioned training/HMC roots |
| Does the method discover modes without known component centers? | Component-aware proposal/starts | A proposal constructed without component labels/centers has adequate importance support, then its frozen transport passes shared sequential HMC and exact component-law screens | Proposal-support failure blocks HMC; local movement alone does not pass | Fresh versioned mode-blind root |

Native divergence availability is not a separate algorithmic pass criterion that can
be invented for TFP `HamiltonianMonteCarlo`. If the kernel does not expose a native
boolean, the artifact must continue to say `not_exposed_by_kernel`; finite energy
error is retained as an explicit numerical veto but is not relabeled as native
divergence.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Reviewed SHA `b39c68...` | Existing successful result and hashed checkpoint | This is the capacity repair actually admitted to sequential HMC | Constants drift from artifact | Loader plus exact metadata test | Reviewed baseline |
| `(128,128)`, six stages, update 8,750 | Existing disjoint-heldout selection | Already passed one downstream three-mode run | One-seed selection luck | Fresh-seed rung | Baseline hypothesis |
| Weighted forward KL | Checked objective derivation and implementation | Target-covering objective when proposal support is adequate | Low-density bridges receive little absolute weight | Component-wise support, transitions, HMC R-hat | Reviewed mechanism |
| Component-aware proposal | Existing serious successful run | Separates representation/mixing viability from mode discovery | Cannot prove discovery | Separate mode-blind arm | Baseline only |
| Four HMC chains and `L=(3,5,10,15,20,25)` | Shared sequential/tuning policy | Enables modern multi-chain diagnostics and forbids `L=1` | Grid may miss another viable kernel | Fresh fixed-kernel verification | Reviewed default |
| Fresh seed count of two initially | Bounded discriminating rung | Enough to falsify deterministic one-seed success cheaply, not enough for a publication-scale rate | Both may pass by chance | Report as bounded replication, no ranking | Convenience budget rung |
| Mode-blind proposal family | To be selected only from existing repository implementation or a separately audited construction | Needed to test discovery without component centers | Hidden use of exact modes, or unusably low ESS | Static source audit and proposal-only preflight | Unproven hypothesis |

The first mode-blind proposal preflight uses an iid centered Student-`t(3)`
family with scale ladder `(1, 2, 4, 8)`. The center, degrees of freedom, and
scales are convenience hypotheses and are not promoted defaults. Proposal
construction may use only the fixed coordinate origin and the stated ladder;
the exact component means, covariances, weights, labels, or responsibilities
may be used only after sampling to evaluate support. Admission requires both
global and median 4,096-row batch ESS fraction at least `1/16`. This threshold
is derived from requiring at least 256 effective rows in every nominal 4,096-row
training batch. If no arm passes, training and HMC stop for this proposal family.

## Skeptical Plan Audit

- The wrong baseline is the obsolete small checkpoint. The reviewed capacity
  repair is the baseline for new serious validation.
- Checkpoint file integrity is not scientific eligibility. Eligibility must bind
  target and selected configuration, not merely accept any internally valid state.
- Existing successful HMC cannot count as a fresh seed. New training initialization
  and sample seeds are required for replication.
- A component-aware proposal cannot answer mode discovery. The mode-blind arm must
  be isolated and must fail before HMC if its importance weights lack support.
- Acceptance, loss, and runtime remain explanatory. They cannot replace R-hat,
  ESS, mode transitions, or exact component-law screens.
- Missing native TFP divergence telemetry is an explicit limitation, not evidence
  of zero divergences and not a reason to invent a mislabeled proxy.
- Cross-target geometry and application gaps require their own target-specific
  protocols. This campaign may inventory them but cannot close them using the
  analytic mixture.

Audit verdict: the provenance and record repairs are ready for immediate execution.
Fresh replication may proceed only after those tests pass. The mode-blind arm may
proceed only after source inspection identifies a construction that does not encode
the exact component centers and after a cheap importance-support preflight passes.

## Execution And Stop Conditions

1. Change the active three-mode runner to the reviewed checkpoint and add a
   repository-owned eligibility check for SHA, target signature, architecture,
   selected step, and XLA status.
2. Replace the test that enforces the obsolete checkpoint with positive and
   negative eligibility tests.
3. Correct the active 2026-08-17 result/reset notes without deleting historical
   artifacts.
4. Run the focused CPU contract suite. A failure blocks research execution.
5. Inspect available fresh-seed and mode-blind harnesses. Reuse only a path whose
   target, objective, batching, GPU/XLA, and evidence boundaries match this plan.
6. Run proposal/training canaries before any full run. Stop a lane on a true
   continuation veto; preserve candidate failures as evidence and follow only the
   predeclared repair rung that remains within budget.
7. Record a terminal result and reset memo separating closed engineering gaps,
   rejected candidates, viable candidates, and claims still requiring evidence.

Campaign bounds: no more than two fresh component-aware training seeds, one
mode-blind proposal preflight and one mode-blind training seed, and no more than
one serious HMC run per passing fresh transport in this campaign. Each GPU process
must use memory growth before TensorFlow initialization and XLA for training/HMC.
