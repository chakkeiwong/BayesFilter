# BayesFilter LGSSM NeuTra Target-Specific Training Protocol Amendment

Date: 2026-07-14  
Status: `SMOKES_PASSED_AWAITING_EXPANDED_COMPUTE_APPROVAL`  
Parent plan: `docs/plans/bayesfilter-lgssm-neutra-knowledge-transfer-and-serious-validation-plan-2026-07-13.md`  
Owner: Codex supervisor/executor  
Reviewer: Claude Code, read-only and advisory  
Versioned output root: `docs/benchmarks/artifacts/lgssm_neutra_target_specific_protocol_2026_07_14/`

## 2026-07-14 Graph-Native Execution Amendment

The later plan
`docs/plans/bayesfilter-lgssm-neutra-graph-native-training-migration-plan-2026-07-14.md`
supersedes this document's training-execution mechanics while preserving its
scientific target, recipes, seeds, objective, budgets, held-out rule, and HMC
evidence contract. Active training now executes all optimization steps in one
`tf.function(jit_compile=True)` program using `tf.while_loop`, with no
repository-owned NumPy dependency or per-step host loop. It emits one terminal
checkpoint per invocation under
`checkpoint_policy=terminal_only_graph_native_v1`; the 50-step checkpoint and
automatic infrastructure-resume text below is retained as historical planning
context and is no longer active.

## Why This Amendment Exists

The parent plan correctly transferred the NeuTra composition, exact reverse-KL
objective, checkpointing, frozen-score bridge, HMC tuning, and serious
diagnostics. It did not transfer the source program's serious-training budget
or perform a target-specific training-protocol audit.

The parent Phase 4 fixed two dense candidates at 1,000 steps, batch 256,
learning rate `0.001`, and linear decay. Those values were convenience choices.
The inspected source program instead records this serious baseline anchor:

- 5,000 steps;
- batch 128;
- constant Adam learning rate `0.005`;
- three dense IAF stages with hidden layers `[dim, dim]`;
- `s_max=1`, initialization scale `0.02`, and clipping norm `10`.

Source anchors:

- `/home/chakwong/python/docs/plans/neutra-paper-style-at-long-budget-reset-memo-2026-05-19.md`
- `/home/chakwong/python/docs/plans/neutra-paper-style-at-stage-ladder-and-tuning-result-2026-05-19.md`
- `/home/chakwong/python/docs/plans/neutra-paper-style-long-budget-training-tuning-result-2026-05-19.md`
- `/home/chakwong/python/scripts/run_neutra_paper_style_at_baseline.py`

The source evidence is from a different target. It makes the source recipe a
strong prior and baseline, not a BayesFilter LGSSM default. The current
`AGENTS.md` also requires objective/scaling checks, capacity and optimizer
search, a budget ladder, held-out criteria, seed policy, and downstream
validation for a substantially different target. This amendment supplies that
protocol before any serious LGSSM claim.

## Supersession Boundary

This amendment supersedes only the parent plan's Phase 4 dense-candidate
configuration and its assumption that 1,000 steps constitute serious training.
It preserves:

- the exact target, fixture, signatures, parameter order, and score bridge;
- the affine control;
- the mathematical convention `T(z) = affine(dense_iaf_stack(z))`;
- GPU/XLA float64 training and CPU-hidden XLA HMC separation;
- Phase 5 fresh modern rank/folded R-hat admission;
- Phase 6 independent four-chain, 4,000-draw confirmation;
- the immutable tuned plain-HMC comparator and all final nonclaims.

The earlier 1,000-step root remains historical evidence. It must not be
overwritten, completed, or promoted:

`docs/benchmarks/artifacts/lgssm_neutra_serious_validation_2026_07_13/`

## Crash And Attempt Record

Two original convenience-configuration jobs were interrupted when VS Code and
its managed process tree crashed. Neither produced a frozen transport or a
candidate result.

| Candidate | Last heartbeat | Last immutable checkpoint | Checkpoint SHA-256 | Classification |
| --- | ---: | ---: | --- | --- |
| `dense_seed1201` | 80 | 50 | `a6b9006fe52b6e89c2dbb4b7fc3bb56054ff454e9f04a0c73966752d3818636d` | infrastructure interruption of a diagnostic convenience arm |
| `dense_seed1202` | 60 | 50 | `b79e50240f15ef151149a7050cdf94041db600762d14801d225fdfa0fec5ceba` | infrastructure interruption of a diagnostic convenience arm |

All observed heartbeats had finite values, valid target status, and zero
target floors. This establishes early engineering viability only. Replaying
those checkpoints would continue the wrong serious-training question, so they
are preserved but not resumed.

## Research Intent Ledger

| Field | Predeclared value |
| --- | --- |
| Main question | Which of four predeclared, source-grounded training recipes should be nominated for two independent 5,000-step dense-IAF runs on the exact 18D LGSSM target, and do the resulting frozen transports pass independent transformed-HMC admission and confirmation? |
| Mechanism under test | Target-specific recipe nomination for plain reverse-KL dense-IAF NeuTra, with the affine-last map and target mathematics fixed. The four-arm screen does not isolate a causal capacity or learning-rate effect. |
| Baselines | Fixed affine control; source 5,000-step dense-IAF recipe; immutable tuned plain HMC for final same-target comparison. |
| Training-screen criterion | Engineering validity plus common-held-out reverse-KL estimates may nominate one recipe. This is a proxy nomination only. |
| Promotion computation | Phase 5 modern diagnostic admission followed by Phase 6 independent serious HMC on both final training seeds. |
| Promotion vetoes | Target/signature drift, invalid target status, nonfinite training or frozen state, reload/score mismatch, CPU fallback, non-XLA execution, failed Phase 5 modern R-hat admission, or any Phase 6 convergence/integrity/agreement/recovery veto. |
| Continuation vetoes | Invalid target or score, common implementation failure across all candidates, zero surviving screen recipes for any combination of predeclared vetoes, corrupted artifacts, unavailable trusted GPU after a trusted probe, or exhaustion of the bounded compute/attempt budget. |
| Repair triggers | Local infrastructure interruption may resume from the last immutable checkpoint into a fresh attempt directory. Numerical failure rejects that screen arm; it does not silently change learning rate because lower learning rate is already an explicit arm. |
| Explanatory diagnostics | Training and held-out reverse KL, paired held-out differences and MCSE, gradient norms, clipping frequency, log determinant, target conditioning telemetry, acceptance, runtime, and short-screen HMC metrics. |
| Forbidden conclusions | A screen winner is not a superior method; 500-step behavior is not serious training; training loss does not prove transport quality; one fixture does not prove calibration, robustness, generalization, posterior correctness, production readiness, or a new default. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Main failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Exact reverse KL | Parent Phases 0–3 and source implementation | Tests the plain NeuTra mechanism without comparator leakage | Correct predictor-like loss but wrong target gradient | Exact value/score parity and target-status telemetry | reviewed fixed choice |
| Affine-last composition | Parent mathematical convention and source paper-style route | Keeps frozen target geometry and flow convention identical across arms | Accidental affine-first or inverse-density training | Forward/logdet/score parity | reviewed fixed choice |
| Three stages, width `[18,18]` | Source serious baseline | Strong cross-target recipe prior at matching dimension | Too little or excessive capacity for this LGSSM | Compare complete recipe behavior with the other arms | baseline recipe hypothesis |
| Two stages, width `[18,18]` at `0.005` | Bounded lower-capacity recipe | Tests one plausible lower-capacity recipe without claiming a capacity effect | Underfitting or a rate-capacity interaction | Held-out objective and final downstream veto if nominated | recipe hypothesis; capacity effect not isolated |
| Three stages, width `[36,36]` at `0.005` | Source screen recommendation, width multiplier 2 | Tests one plausible wider recipe without claiming a width effect | Optimization instability, unnecessary capacity, or a rate-capacity interaction | Finite gradients, clipping, held-out objective | recipe hypothesis; capacity effect not isolated |
| `s_max=1`, init scale `0.02` | Frozen schema and source baseline | Reuses checked transport scale while holding the first screen small | Saturation or slow departure from identity | logdet, gradient, and target-status trajectory | baseline hypothesis, not tuned here |
| Constant Adam `0.005` | Source long-budget incumbent; cosine did not robustly improve it | Correct source serious anchor | Too aggressive for LGSSM, chronic clipping, invalid target visits | 50-step smoke and 500-step screen | baseline hypothesis |
| Constant Adam `0.001` on the source architecture | Conservative LGSSM-specific alternative | Tests one lower-rate recipe while keeping the source architecture fixed | Slow training at 5,000 steps | 500-step paired held-out objective | recipe hypothesis; no cross-capacity rate claim |
| Batch 128 | Source serious baseline | Halves per-step exact-target cost relative to the interrupted batch-256 arms | Noisy gradient estimates | common held-out batches and cross-seed Phase 4 results | baseline hypothesis |
| Clip norm 10 | Source baseline and parent canaries | Bounded an initially large gradient without invalid state | Chronic clipping hides a bad rate | clipping fraction and raw norm trajectory | baseline hypothesis |
| 500-step screen | Diagnostic budget only | Rejects gross target-specific failures before long training | Short-budget proxy nominates the wrong recipe | mandatory 5,000-step downstream validation | convenience diagnostic |
| 5,000-step final budget | Source reset memo's minimum serious anchor | Avoids repeating known short-budget drift | Still insufficient or inefficient on LGSSM | checkpoints at 1,000/3,000 and final Phase 5/6 | transferred baseline, target validation required |
| Two final seeds | Parent plan | Detects seed-specific training failure without claiming a population estimate | Too few seeds for ranking | candidate-by-candidate Phase 5/6 results | bounded validation choice |
| Truth-centered affine geometry | Existing exact fixture | Required for same-target continuity | Makes the problem unusually favorable | mandatory closeout limitation | fixed fixture limitation |

## Explicit Candidate Screen

All arms use batch 128, 500 steps, one screen seed
`(20260714, 1401)`, constant learning rate, ELU, `s_max=1`, initialization
scale `0.02`, clip norm `10`, float64, GPU/XLA, and the same fixed affine map.

| Candidate | Stages | Hidden layers | Learning rate | Question |
| --- | ---: | --- | ---: | --- |
| `source_anchor_lr5e3` | 3 | `[18,18]` | `0.005` | Does the mature source recipe transfer without target-specific failure? |
| `lower_lr1e3` | 3 | `[18,18]` | `0.001` | Does the exact LGSSM target need a more conservative rate? |
| `shallow_2stage_lr5e3` | 2 | `[18,18]` | `0.005` | Is this lower-capacity recipe viable enough to nominate? |
| `wide_2x_lr5e3` | 3 | `[36,36]` | `0.005` | Is this source-recommended wider recipe viable enough to nominate? |

No random or Optuna search is allowed in this campaign. Schedules, `s_max`,
initialization scale, optimizer family, and HMC policy are held fixed to keep
the first target-specific protocol bounded. The shallow and wide recipes are
not crossed with the lower learning rate, so their outcomes cannot establish
a causal capacity effect or distinguish capacity from rate-capacity
interaction. The screen compares four complete recipes only.

## Held-Out Nomination Rule

Each screen arm is evaluated on eight common, independent held-out base batches
of 128 draws, using seeds disjoint from screen and final training seeds.
Per-batch reverse-KL means are preserved so paired differences and their MCSE
can be computed.

1. Reject any arm with a target, finite-state, device, XLA, reload, or score
   veto.
2. Training loss, clipping, and force summaries are explanatory. They cannot
   pass an arm by themselves.
3. Nominate the arm with the lowest common-held-out mean reverse KL among
   non-vetoed arms. If its mean paired difference from the source anchor is
   within one paired-difference MCSE of zero, prefer the source anchor to avoid
   selecting noise. The same paired-MCSE equivalence rule applies to
   deterministic tie priority when the source anchor is vetoed.
4. If the source anchor is vetoed, apply the same rule among the remaining
   arms by selecting the lowest common-held-out mean. Exact numerical ties use
   deterministic recipe order `lower_lr1e3`, `shallow_2stage_lr5e3`, then
   `wide_2x_lr5e3`; the paired-MCSE source preference does not transfer to a
   non-source recipe.
5. If zero arms survive, emit a terminal Phase 4 screen result with no
   nominated recipe and a complete candidate-failure table. Stop this campaign;
   any additional recipe or search requires a new amendment and compute budget.
6. This rule nominates a representative training protocol only. It does not
   statistically establish that the recipe is best or that held-out reverse KL
   predicts HMC quality.

The nominated recipe is then trained from fresh initialization for both parent
seeds `(20260713,1201)` and `(20260713,1202)` to 5,000 steps. Screen weights
are never reused by final training.

## Budget And Attempt Contract

Training target evaluations:

- wiring smoke: `4 * 5 * 128 = 2,560`;
- screen: `4 * 500 * 128 = 256,000`;
- final: `2 * 5,000 * 128 = 1,280,000`;
- held-out screen: `4 * 8 * 128 = 4,096`;
- planned total: `1,542,656` exact-target evaluations.

This is about three times the parent Phase 4 training budget of 512,000 target
evaluations. It is a material compute expansion and therefore requires a plain
human approval after review; no special wording or hash binding is required.

Attempt limits:

- one screen attempt per arm;
- one final attempt per seed;
- at most one infrastructure resume per job, from its latest immutable
  checkpoint into a fresh attempt directory;
- every screen and final job writes an immutable checkpoint every 50 steps;
- replay caused by all infrastructure resumes must remain below 10% of the
  planned training target-evaluation budget;
- no numerical-failure rerun with an unplanned configuration;
- Phase 5 retains its one declared fixed-grid repair;
- Phase 6 retains its no-rerun confirmatory rule.

Observed interrupted throughput makes the wall-time estimate uncertain. The
campaign should budget up to 48 hours of trusted GPU wall time for Phase 4,
then the already planned CPU-hidden Phase 5/6 budget. Stop and report if the
training wall budget reaches 48 hours before both final candidates freeze.
With six planned jobs and 50-step checkpoint spacing, the worst-case replay
under the one-resume-per-job rule is fewer than 300 steps, or fewer than 38,400
target evaluations at batch 128. This is below 2.5% of the planned total and
therefore satisfies the 10% replay cap.

## Execution Phases

### A. Implementation And Contract V2

- Add explicit training recipes and deterministic screen/final seed ledgers.
- Add constant-learning-rate support through the existing
  `final_learning_rate_fraction=1.0` path; do not invent a second optimizer.
- Add common multi-batch held-out summaries with batch-level observations and
  MCSE.
- Add diagnostic screen freeze/reload checks and final recipe selection.
- Write a new campaign contract under the versioned output root. Preserve the
  parent contract unchanged.
- Add focused tests for candidate enumeration, nomination/tie behavior,
  versioned no-overwrite output, checkpoint recovery, and the 5,000-step final
  handoff.

### B. Trusted GPU/XLA Screen

- Run a five-step wiring smoke for every arm.
- Run all four 500-step arms on the screen seed.
- Preserve checkpoints at 50-step intervals and heartbeats at 10-step
  intervals.
- Evaluate common held-out batches, emit the nomination ledger, and freeze the
  selected recipe specification before final training.

Screen failure does not refute NeuTra. A common target/math failure is a
continuation veto; an individual arm failure rejects only that arm.

### C. Long-Budget Final Training

- Train the frozen selected recipe from fresh initialization for seeds 1201
  and 1202 to 5,000 steps.
- Preserve immutable checkpoints every 50 steps, including 1,000, 3,000, and
  5,000. This bounds worst-case replay for two resumed final jobs to fewer than
  100 training steps, below 1% of planned training evaluations. Intermediate
  checkpoints are failure-localization artifacts, not candidate-selection
  opportunities.
- Freeze only the completed 5,000-step transports.
- Require exact trainable/frozen forward, logdet, and explicit-score parity.
- Write Phase 4 result and hand every engineering-valid final seed to Phase 5.

### D. Existing HMC Admission And Confirmation

- Run Phase 5 for the affine control and both engineering-valid final seeds.
- Freeze all admission decisions before any Phase 6 run.
- Run Phase 6 once for each admitted pair using the existing independent
  four-chain, 4,000-draw contract.
- Do not retune, retrain, reseed, thin, exclude a chain, or rerun a Phase 6
  pair after observing its result.

### E. Closeout

- Write the final result note, decision table, inference-status table,
  engineering/numerical/scientific ledgers, run manifests, and post-run red
  team.
- Ask Claude for one bounded read-only terminal result review.
- State directly which candidates passed hard vetoes, whether any ranking is
  statistically supported, which differences are descriptive only, and what
  evidence would be needed for a broader claim.

## Skeptical Pre-Execution Audit

| Risk | Finding and repair |
| --- | --- |
| Wrong baseline | Replaced the convenience 1,000-step recipe with the source 5,000-step anchor plus a target-specific screen. |
| Proxy promotion | Held-out reverse KL nominates only; serious HMC remains the promotion computation. |
| Missing stop conditions | Added common-invalidity, zero-survivor, artifact, GPU, attempt-budget, and 48-hour wall-budget continuation vetoes. |
| Unfair comparison | Screen arms share target, seed, batch, objective, affine map, dtype, device, and budget. They compare complete recipes; no causal capacity/rate claim is allowed because the factors are not fully crossed. |
| Hidden defaults | Recorded provenance, failure mode, early diagnostic, and status for every material training choice. |
| Stale context | Inspected the source May 19 reset memo, stage result, and long-budget tuning result rather than relying on the old parent summary. |
| Environment mismatch | Training remains trusted GPU/XLA; HMC remains CPU-hidden XLA. |
| Artifacts do not answer question | The screen answers nomination only; two 5,000-step frozen seeds plus Phase 5/6 answer the fixed-fixture serious-validation question. |
| Compute creep | Predeclared exact target-evaluation count, attempt limits, replay cap, and wall-time stop. |
| Misleading pass | Truth-centered geometry and one-fixture scope remain mandatory limitations. |

Audit verdict: `PASS_FOR_IMPLEMENTATION_AND_REVIEW_ONLY`. Long GPU execution
requires this material amendment to pass bounded review and the owner to
approve the expanded compute budget.

## Review Record

Claude Code health probe returned `CLAUDE_PROBE_OK`. Opus at max effort then
reviewed exactly this amendment path with read/search-only tools.

Initial verdict: `REVISE`.

Material findings and repairs:

1. The original four-arm screen confounded capacity with learning rate while
   using capacity-causal language. The plan now compares complete recipes only
   and explicitly forbids causal capacity/rate conclusions.
2. The original `1,000/3,000/5,000` final checkpoint cadence could violate the
   10% replay cap. Every job now checkpoints every 50 steps; worst-case replay
   over all six jobs is below 2.5% of planned evaluations.
3. The zero-survivor screen branch was undefined. It now emits a terminal
   no-recipe result and requires a new amendment for further search.

Focused convergence review: `VERDICT: AGREE`. No remaining material defects
were found in those repairs. Claude remained a read-only advisory reviewer;
Codex remains supervisor and executor.

## Exact Handoff Conditions

Implementation may start when this file passes bounded review. The GPU screen
may start when focused tests and trusted device/XLA probes pass. Long final
training may start only when:

- all screen artifacts are complete and immutable;
- the nomination rule emits exactly one recipe;
- no common target/math continuation veto fired;
- the selected recipe spec is frozen under the versioned root; and
- the owner has approved the stated expanded compute budget.

Phase 5 may start only after both planned final seed jobs either freeze valid
5,000-step transports or have terminal candidate-failure records, and at least
one target-specific learned transport survives. The reused affine control alone
cannot pass the amended Phase 4. Phase 6 may start only after all Phase 5
admission decisions are frozen.
