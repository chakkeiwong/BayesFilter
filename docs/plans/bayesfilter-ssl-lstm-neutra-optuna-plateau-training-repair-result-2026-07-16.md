# SSL-LSTM NeuTra Optuna And Plateau Repair Result

Date: 2026-07-16

Status: `TRIAL0_GH_FRESH_CONFIRMATION_PASSED_PHASE4_COMPLETE`

## Outcome

The tuned `(32,32)` NeuTra family, mutable serialized learning rate, plateau
repair controller, Optuna harness, and fresh-confirmation runner are
implemented. Sixty focused CPU-hidden tests pass. A trusted one-step GPU/XLA
smoke completed, and the bounded six-trial Optuna study completed in
`8,144.25` seconds (`2.2623` GPU-hours), below its `4.0`-hour cap.

Two trials survived both historical A/B streams through the common step-400
rung and all terminal transport screens. Four trials were pruned on stream A
by the declared Optuna pruner. There were no hard vetoes, saturation vetoes,
resource stops, corrupt artifacts, or target/runtime failures in the material
study.

Trial 2 is frozen as the representative Optuna nomination proxy:

- initial learning rate `0.0018996814203826532`;
- initialization scale `0.01`;
- per-variable clip norm `5.0`; and
- worst-stream terminal heldout reverse-KL proxy `41.33128700938453`.

Trial 0 also remains viable. Its proxy was `41.39404496266031`. The difference
`0.06276` is descriptive only and does not support a statistical ranking.

## Fresh Confirmation And Repair Diagnosis

The frozen C/D confirmation completed in `9,513.14` seconds (`2.6425`
GPU-hours) with decision `FRESH_CONFIRMATION_NOT_PASSED`:

| Stream | Stop | Best/terminal step | Support radius | Decision |
| --- | --- | --- | ---: | --- |
| Fresh C | Plateau after LR repair | `100 / 1100` | `5.527918` | Vetoed: exported best missed moderate-shell support |
| Fresh D | Plateau after LR repair | `700 / 1700` | `3.032461` | Passed all declared transport-training gates |

One pass cannot promote the procedure. C/D provide no posterior, HMC, or
predictive evidence.

A read-only audit then evaluated existing C checkpoints under the unchanged
support threshold. Step 100 failed, but steps 300, 600, 800, and 1,100 had
moderate-shell radii `3.2267`, `3.1429`, `3.1193`, and `3.0147`, respectively,
and passed the support screen. This supports
`CHECKPOINT_SELECTION_MISALIGNMENT_SUPPORTED`: loss-only selection exported an
inadmissible early checkpoint even though later C states were support-admissible.
It does not authorize retrospective promotion of any audited checkpoint.

The prospective repair makes finite, saturation, roundtrip, and moderate-shell
checks part of checkpoint eligibility. An ineligible state cannot initialize or
replace best. The first eligible state initializes patience; after that, only a
meaningful paired heldout-loss improvement among eligible states resets patience.
Untouched E/F seeds are required for repaired confirmation, and HMC remains
closed until both pass.

## Repaired E/F Confirmation

The repaired support-aware confirmation completed in `9,102.96` seconds
(`2.5286` GPU-hours), below its `17,500`-second cap, with decision
`FRESH_CONFIRMATION_NOT_PASSED`.

| Stream | Best/terminal | LR repair | Best support/saturation | Decision |
| --- | --- | --- | --- | --- |
| Fresh E | `600 / 1600` | One reduction at step 1,100 | radius `3.17633`; saturation `0.0` | Passed |
| Fresh F | `700 / 1000` | None before veto | radius `3.04746`; saturation `0.01432` | Vetoed when terminal saturation reached `0.05339 > 0.05` |

The selector repair behaved correctly on F. Its step-100 state had lower loss
but failed moderate-shell support (`5.70531`) and could not replace best. Later
support-eligible states advanced best through step 700. Saturation then rose to
`0.02865` at step 800, `0.04948` at step 900, and `0.05339` at step 1,000,
before plateau patience would have reduced LR at step 1,200. Thus the supported
failure classification is trial-2 optimizer-policy instability on one fresh
stream, not checkpoint-selection, target, XLA, resume, or artifact failure.

Trial 2 is not confirmed, and E's pass cannot be promoted alone. The next
justified candidate is the predeclared trial-0 alternative, which survived both
historical A/B streams with lower LR `0.0011219623709077644`, initialization
scale `0.02`, and clip norm `5.0`. This alternative changes both LR and
initialization scale, so its outcome cannot be interpreted as a pure LR effect.

## Trial-0 G/H Resource Result

The trial-0 alternative used the remaining authorized envelope and stopped
cleanly at the resource boundary after `8,440.86` seconds (`2.3447`
GPU-hours).

| Stream | State | Evidence |
| --- | --- | --- |
| Fresh G | Passed | best step `1000`; terminal step `2000`; one LR reduction; radius `3.01857`; saturation `0.00911`; paired improvement upper `-22.203`; no vetoes |
| Fresh H | Resource stop, confirmation open | optimizer step `446`; latest validation/best step `400`; radius `3.87146`; saturation `0.01042`; roundtrip `8.88e-15`; eligible and meaningfully improved; no candidate veto |

H's joint checkpoint validates and preserves trainer, Adam, effective LR,
controller, best-state, and checkpoint-history hashes. The stop is not evidence
against trial 0. It also does not permit promotion because both G and H must
complete and pass.

Total charged or conservatively counted work is `35,905.30` seconds
(`9.9737` GPU-hours), leaving only `94.70` seconds under the authorized 10-hour
envelope. No useful resume can occur inside that remainder. A separately
authorized cumulative cap of `15,500` seconds for the trial-0 G/H runner is the
recommended extension: it adds at most `7,000` seconds (`1.9444` GPU-hours) and
resumes only H from optimizer step 447. It does not authorize HMC or another
candidate search.

### Final H Resume

The owner authorized increasing the cumulative trial-0 cap to `15,500`
seconds solely to resume H. After a metadata-only fix normalizing persisted JSON
seed lists against in-memory tuples, the original step-446 joint checkpoint was
resumed without numerical-state changes. H completed with:

- support-eligible best step `900` and terminal step `1900`;
- one LR reduction at step `1400`;
- plateau stop at step `1900`;
- best moderate-shell radius `3.00731`;
- best roundtrip residual `1.60e-14`;
- best saturation `0.00651`; and
- paired best-minus-initial one-sided upper bound `-25.035`.

Both G and H therefore passed. The final trial-0 cumulative charge was
`13,730.28` seconds (`3.8140` GPU-hours), below its `15,500`-second cap. Total
program use was `41,194.72` seconds (`11.4430` GPU-hours), below the cumulative
ceiling created by the original envelope plus the explicitly authorized H-only
extension.

The supported conclusion is that trial 0 remains a viable transport-training
procedure under two untouched confirmation streams and the declared screens.
No statistical ranking over trial 2 is supported because trial 0 was tested
after trial-2 instability, both LR and initialization scale differ, and the
sample of fresh runs is small.

The plateau policy is frozen at validation every 100 steps, patience `n=500`,
LR factor `0.5`, stop at `2n` without meaningful improvement, maximum 5,000
steps, LR floor `initial/16`, and paired one-sided 95% improvement with
`absolute_min_delta=0.0`. Every observed survivor rung was a meaningful paired
improvement; the observed gaps were at most 200 steps, so the prospective
`[500,1000]` lower bound selected `n=500`.

## Evidence

| Trial | Status | LR | Init scale | Clip | A loss/saturation/support radius | B loss/saturation/support radius |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Joint survivor | `0.00112196` | `0.02` | `5` | `41.2764` / `0.0130` / `3.3220` | `41.3940` / `0.0013` / `3.5373` |
| 2 | Joint survivor; representative proxy | `0.00189968` | `0.01` | `5` | `40.8084` / `0.0026` / `4.1053` | `41.3313` / `0.0039` / `3.3968` |
| 1 | Pruned A | `0.00064214` | `0.02` | `10` | Pruned at first reported rung | Not run |
| 3 | Pruned A | `0.00016010` | `0.01` | `5` | Pruned at first reported rung | Not run |
| 4 | Pruned A | `0.00180808` | `0.01` | `5` | Pruned after second reported rung | Not run |
| 5 | Pruned A | `0.00022124` | `0.005` | `10` | Pruned at first reported rung | Not run |

Both survivors had finite target/transport state, exact artifact reload,
roundtrip residual below `7e-15`, saturation below `0.05`, moderate-shell
radius below `4.30`, and strongly negative paired final-minus-initial upper
bounds. These are nomination screens, not posterior or HMC validation.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Freeze trial 2 as representative policy | Passed: at least one joint survivor; two observed | No hard or nomination veto for trials 0/2 | Only historical A/B and 400 steps; reverse-KL proxy may not predict full training or HMC | Run both frozen fresh streams if the confirmation resource cap is authorized | Trial 2 is superior, posterior-correct, or HMC-ready |
| Retain trial 0 as viable alternative | Passed same screens | No veto | Objective difference has no uncertainty-supported ranking | Preserve as repair alternative if fresh trial-2 policy is unstable | Trial 0 is inferior |
| Freeze `n=500` and zero min delta | Historical survivor rungs all improved meaningfully | No calibration invalidity | No historical plateau was actually observed by step 400 | Use prospectively on fresh runs; do not recalibrate there | `n=500` is optimal or universal |
| Reject original C/D confirmation | Both fresh runs were required; only D passed | C support promotion veto fired | C's loss-selected best was not representative of its later support-valid trajectory | Repair checkpoint eligibility prospectively and confirm on untouched E/F | Trial 2, NeuTra, or C's whole trajectory is invalid |
| Admit support-aware selector for E/F testing | Deterministic repair tests pass | No implementation veto found locally | Downstream behavior on E/F is unknown | Focused review, then bounded E/F GPU/XLA confirmation | Repaired trial 2 is HMC-ready |
| Reject repaired trial-2 confirmation | E passed but F crossed saturation cap | Valid F promotion veto; no continuation veto | Only one fresh pair and a near-threshold late crossing | Test the predeclared trial-0 alternative on untouched G/H under a new cap | NeuTra direction, architecture, or support selector is invalid |
| Historical interim: keep trial-0 confirmation open | G passed; H then had eligible partial evidence | Resource stop only; no candidate veto | H terminal outcome was then unknown; superseded by completed H | Resume H with a separately authorized cumulative cap | Trial 0 passes, is superior, or is HMC-ready |
| Close Phase 4 with trial 0 viable | G and H both passed the prospective confirmation gates | No G/H transport, artifact, runtime, or resource veto | Two fresh streams do not establish broad stability or ranking | Write a separate exact transformed-target preflight plan for immutable G/H payloads | Posterior correctness, support completeness, superiority, HMC readiness, or default readiness |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | G/H trial-0 confirmation passed; trial 2 was rejected by Fresh F's saturation veto |
| Viable candidates | Trial 0 is the only fresh-confirmed training procedure; trials 0 and 2 were historical tuning survivors |
| Statistically supported ranking | None |
| Descriptive-only differences | Trial losses, saturation, shell radii, runtime, and proxy difference |
| Default readiness | Not established |
| Next evidence needed | Exact transformed-target value/score, serialization, original-start, and GPU/XLA preflight on immutable G/H payloads before any HMC tuning |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `20835ecf90bff78ca93c5d401f231e4aa94e63ce` with unrelated dirty worktree preserved |
| Environment | conda `tfgpu`; TensorFlow `2.20.0`; Optuna `4.6.0`; `float64` |
| Device | Physical GPU 1, process-visible GPU 0, NVIDIA GeForce RTX 4080 SUPER |
| Runtime | XLA JIT on, TF32 enabled, soft placement off |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Study command | Frozen exact command in the live plan Phase 4 execution note |
| Study charged/cap | `8,144.2544 / 14,400` seconds |
| Timing smoke | `302.7857` seconds; result SHA-256 `6d9f06f70515ed71d83866117d0be62be480eaf1143b5b09755881f8f37a79ef` |
| Final focused tests | `72 passed`; compilation and `git diff --check` passed |
| Trial-0 G/H charged/cap | `13,730.2811 / 15,500` cumulative seconds; includes the authorized H resume |
| Total program charged/conservative accounting | `41,194.7243` seconds (`11.4430` GPU-hours), below the successively authorized ceiling |
| Study summary | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/optuna-plateau-repair-study-2026-07-15/study-summary.json`; SHA-256 `186f32bd54f97575ce236d89ebefe07d1c1af7063f0055adc05209f0c5c0475f` |
| Frozen policy | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/optuna-plateau-repair-study-2026-07-15/frozen-tuning-policy.json` |
| C/D confirmation | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/plateau-confirmation-2026-07-16/confirmation-summary.json`; SHA-256 `fc2d6f200a027cd52f4a90209f851d2d9b8ca75692e16a5ef5e2dafdeca1ae41` |
| C support audit | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/plateau-confirmation-2026-07-16/fresh-c-support-audit.json`; SHA-256 `7cc0a4a7fd3e29d29f42597601a8c2db96e411eb7436519126e2e13de1411998` |
| E/F confirmation | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/support-eligible-confirmation-2026-07-16/confirmation-summary.json`; SHA-256 `358bd1117e775a622b26df26d0c459058633af70d7cd94b3ee2f99b7b328d24f` |
| Trial-0 G/H summary | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/trial0-alternative-confirmation-2026-07-16/confirmation-summary.json`; SHA-256 `f1b7aa7858a8e97f2870b5035efa91742065e2f58620026e641bb389fcb93eb5` |
| G best frozen payload | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/trial0-alternative-confirmation-2026-07-16/fresh-g/best-frozen-payload.json`; SHA-256 `6e147d5b33d003e0c895f294fc6b33523dcf97dc24af794d26a677886dedc354` |
| H best frozen payload | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/trial0-alternative-confirmation-2026-07-16/fresh-h/best-frozen-payload.json`; SHA-256 `ed0e42602aa39788ca1ea8d3c881d8bf85e15b91a687ef9adbe00a7b2c9120fb` |

The first non-trusted smoke failed GPU discovery and is sandbox evidence only.
A later trusted smoke on contended physical GPU 0 hit its 300-second resource
cap and is not timing or candidate evidence. Neither is included in the
material study charge or interpreted scientifically.

## Negative-Result Classification

- Implementation failure: not supported.
- Runtime/XLA failure: not supported.
- Resume implementation failure: supported and repaired. Persisted JSON seed
  lists were compared with tuples on the first H resume attempt; it stopped
  before training, preserved all artifact hashes, and was covered by a focused
  canonical-stream regression test before the successful resume.
- Search failure: not supported; two joint survivors were found.
- Statistically supported optimizer ranking: not established.
- Posterior, HMC, predictive, or scientific validity: not tested.

## Post-Run Red Team

The strongest alternative explanation is shared reverse-KL mode seeking: G and
H can both pass the finite support probe bank while mapping the same incomplete
posterior region. Two fresh training streams and fixed shell probes do not prove
complete mode or tail coverage. Exact transformed-target checks followed by
independently tuned, admitted HMC and predictive replication remain the actual
learned-component computation gates.

The Phase 4 conclusion would be overturned by target/source or payload-hash
drift, seed overlap, loader mismatch, replay showing that G or H crossed a
prospective veto, or failure to reproduce exact trainer/controller resume. The
weakest evidence is replication breadth: two passing trial-0 streams support
viability under the declared screens, not a statistical ranking or broad
stability claim. `n=500` also remains a prospective lower-bound policy rather
than a universally identified optimum.
