# q=20 SSL-LSTM NeuTra Two-Architecture Loss Gate

Date: 2026-07-21  
Tier: 2 material GPU/XLA architecture comparison  
Status: `COMPLETED_UNRESOLVED`

## Research Intent Ledger

| Role | Contract |
| --- | --- |
| Main question | Under one loss-driven training protocol, do `(32,32)` and `(64,64)` NeuTra transports achieve different terminal held-out losses, and is either result stable across seeds? |
| Candidate mechanism | Dense IAF hidden widths `(32,32)` versus `(64,64)`; all target, optimizer, batch, validation, and stopping settings held fixed. |
| Exact baseline | q=20, batch size 100, fixed-smoke parameters `(learning_rate=4e-4, initialization_scale=0.01, gradient_clip_norm=10)`, validation every 250 steps, 2,000-step maximum. |
| Primary endpoint | Mean loss of the final selected-best transport on a separate 256-draw audit cloud never used for optimization, repair, stopping, or checkpoint selection. |
| Secondary endpoint | Terminal controller-validation loss, selected-best controller-validation loss and step, terminal-minus-best loss, and full loss trajectory. |
| Promotion criterion | Both seeds of an architecture complete hard-valid terminal exports and the predeclared paired comparison supports a consistent terminal-loss result. |
| Hard vetoes | Nonfinite target/transport values, failed support or round-trip check, failed frozen-artifact reload, corrupted checkpoint, GPU/XLA/memory-growth failure, host-memory cap, or missing terminal artifact. |
| Explanatory diagnostics | Saturation by stage, scale-logit tails, hidden preactivation tails, learning-rate path, gradient clipping, runtime, memory, and target signatures. Saturation has no control or promotion role. |
| Continuation veto | Only invalid evidence, unavailable resources, or a broken target/training contract. A candidate with a higher loss remains evidence against that candidate, not against the architecture-comparison question. |
| Nonclaims | No posterior oracle, posterior correctness, HMC readiness, convergence proof, statistical superiority from two seeds alone, or default promotion. |

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| `(32,32)` and `(64,64)` | User-requested architecture hypotheses; `(32,32)` is the current reference arm | Directly tests capacity without changing the target or training protocol | Width may interact with optimization; fixed-protocol results do not establish architecture-specific optimum |
| Same fixed-smoke parameters | Existing q=20 diagnostic parameters | Isolates width in the primary comparison | One shared setting may favor one width; report as a fixed-protocol comparison and do not call it tuned |
| Seed-a and seed-b | Existing independent stream definitions | Two independent starts provide a minimal robustness check | Two seeds are descriptive and underpowered for a general ranking claim |
| Maximum step 2,000 | Existing bounded training budget | Defines a maximum before observing results while retaining the user-requested loss-plateau stop | Arms may stop at different steps; the independent audit evaluates each protocol-selected best transport |
| Audit batch size 256 | Derived as four times the existing 64-draw controller-validation batch | Reduces Monte Carlo error without changing training cost materially | Still finite Monte Carlo evidence; preserve per-sample losses and paired intervals |
| Loss-only repair | User decision and corrected scientific target | Learning-rate repair is triggered only by paired validation-loss plateau, not saturation | A poor schedule can affect both arms; record all repairs and treat the outcome as protocol-specific |
| GPU 0 fallback | Repository policy and current GPU occupancy | Avoids preempting another lane on GPU 1 | GPU contention may alter runtime; record selected device and do not rank runtime |

## Loss-Only Controller Contract

Saturation is retained in every validation row as telemetry and in checkpoint
diagnostics, but it must not:

- reduce the learning rate;
- reset or extend the plateau window;
- make a finite/support-valid checkpoint ineligible; or
- veto promotion.

Learning-rate reduction occurs only when paired held-out loss has failed to
improve for `patience_steps=250`; after the reduction, training stops after two
additional no-improvement validation cycles or at step 2,000. The exact
controller configuration is stored in every run manifest and checkpoint.

## Evidence Contract

| Item | Contract |
| --- | --- |
| Comparator | Four arms: `(32,32)/seed-a`, `(32,32)/seed-b`, `(64,64)/seed-a`, `(64,64)/seed-b`. |
| Primary comparison | Within each seed, compare independent-audit per-sample losses using the same 256 audit draws; summarize paired mean difference and a two-sided 95% Student-t interval. Across seeds, report the two arm-level results without ranking unless the predeclared consistency rule is met. |
| Consistency rule | Both seeds must complete without hard vetoes. A directional fixed-protocol nomination requires both paired audit intervals to exclude zero in the same direction. With two training seeds this remains a nomination, not broad statistical superiority. |
| Diagnostics that can veto | Only the hard vetoes above. Saturation cannot veto. |
| Artifact | One result JSON per arm, one combined gate summary, exact commands/manifests, and this plan/result note under `docs/plans/artifacts/ssl-lstm-q20-two-architecture-loss-gate-2026-07-21/`. |
| What will not be concluded | No claim that the lower-loss architecture is globally optimal, posterior-correct, converged, or HMC-ready. |

## Skeptical Pre-Execution Audit

- Wrong baseline: checked; the reference arm is the completed `(32,32)` loss-first q=20 run, but fresh arms are required because its learning-rate path used the old saturation repair trigger.
- Proxy promotion: addressed; terminal held-out loss is primary, while saturation and runtime are explanatory only.
- Hidden schedule confounding: addressed; both widths use the same fixed parameters and loss-only controller.
- Missing stop conditions: addressed; hard validity checks, 250-step validation, 250-step plateau patience, two post-repair cycles, 2,000-step maximum, resource cap, and resumable checkpoints are required.
- Unfair seed comparison: addressed; each architecture uses the same two independent stream seeds and validation draws.
- Resource contention: addressed; do not preempt GPU 1; use GPU 0 when available and record occupancy.
- Selection bias: addressed; the 64 controller-validation draws select checkpoints, while a separate predeclared 256-draw audit cloud is final-only and shared across architectures within seed.
- Artifact adequacy: addressed; each arm must include terminal result, independent audit losses, best state, frozen payload, checkpoint history, manifest, and hashes; the combined summary must preserve all four arm statuses.

Audit decision: `PASS_AFTER_LOSS_ONLY_CONTROLLER_REPAIR`.

Launch-time occupancy differed from the earlier audit snapshot: GPU 0 was busy
and GPU 1 was idle. Both architecture runs therefore used GPU 1 without
preempting another process. Device choice is a runtime provenance fact and has
no role in the loss comparison.

## Execution Repair Note

The first `(32,32)/seed-a` launch reached the predeclared 2,000-step maximum
and persisted a verified stopped joint checkpoint, but final export failed on
an internal API mismatch (`FrozenDenseIAFTransport` has public batch methods,
not `forward_and_logdet`). The runner now uses
`forward_batch`/`log_abs_det_jacobian_batch` for frozen audits and permits a
final-mode orphan resume only when a verified stream `progress.json` exists.
This recovery path performs no additional optimizer steps. Focused tests cover
the public batch API and the loss-only contract; the failed export is an
implementation failure, not evidence against either architecture.

## Implementation Scope

1. Add an explicit loss-only controller mode and make the two-architecture gate use it.
2. Preserve saturation telemetry and support diagnostics without allowing saturation to trigger a repair or veto.
3. Update controller and runner tests for loss-only behavior and retain hard numerical veto tests.
4. Run the existing two-seed `final` harness once per width and add a combined audit-loss summary.
5. Execute the four arms sequentially on an available trusted GPU, with a bounded cap per arm and no HMC.
6. Write a result note with terminal/best losses, paired comparisons, hard-veto status, uncertainty limitations, and next action.

## Resource And Stop Contract

- Maximum: 7,200 wall-clock seconds per two-seed architecture run, 14,400 seconds cumulative.
- Stop an arm at a hard veto, resource cap, interruption receipt, ordinary loss plateau, or step 2,000.
- Do not rerun a completed arm without recording a new arm identity.
- Do not preempt or kill processes outside this gate.

## Planned Commands

The two architecture commands invoke `final` mode with `--loss-only-control`,
`--hidden-layers 32,32` or `64,64`, fixed-smoke parameters, batch size 100,
and a 7,200-second two-seed cap. Separate output roots hold each architecture;
the selected physical GPU and all environment settings are recorded in each
manifest.

## Handoff And Stop Conditions

The gate is complete when all four arms have terminal summaries or explicit
hard/resource stop records, the combined summary and result note are written,
and focused tests plus artifact-integrity checks pass. If any arm is invalid,
classify that arm separately and do not rank the architectures. If both seeds
of one architecture are hard-valid but the terminal-loss direction is mixed,
report the architectures as unresolved and proceed to architecture-specific
loss tuning rather than HMC.

## Close Record

All four arms completed without hard vetoes. The paired audit direction was not
significant in both seeds, so the predeclared result is `UNRESOLVED` and HMC was
not launched. Full results, resource accounting, export repair, concurrent
source-drift limitation, statistical nonclaims, and next justified action are
recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-two-architecture-loss-gate-result-2026-07-21.md`.
