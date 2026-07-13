# BayesFilter SSL-LSTM Completion Visible Execution Ledger

Date: 2026-07-11

Program status: `PHASE_A3_HARNESS_REVIEW_AGREED_FOCUSED_TESTS_NEXT`

## 2026-07-13 - Phase A3 - HARNESS REVIEW CONVERGED

Final bounded review set:

| Artifact | Accepted identity |
| --- | --- |
| A3 generator SHA-256 | `3c8dcdc2a87d4e282a7a19bd33a2e1b1965afe0293562fff901146fd8d41899c` |
| A3 verifier SHA-256 | `c669babc5e5e77293607f566177185c33cd99f1bc12b46aaa4932766e6b4696e` |
| Boundary semantic SHA-256 | `74b36be88874d9f11b1179c4b4f6e2ff05f4d43462d4618782eaed1df5ff2540` |
| Fixture semantic SHA-256 | `a1133af4913ea78103b8dd6e62735c6cfa0d8b060dadaf5b71ac12407dce95af` |
| Harness review SHA-256 | `c5d26a3fed42bc9fd0be429527304aece393906ffa8d6cdeddddf57c3395e2fb` |
| Signed harness anchor SHA-256 | `985222a2f70194551d26f16377efbcb69a306dee4c765578873398d5b074a294` |
| Signed boundary SHA-256 | `e15eb8b420d7dbffb26bdfa80ef2a8336fd877deda37474be4cadbe335aaec58` |
| Signed fixture SHA-256 | `811c2bbb64f939945d696fa71d64dba40bc98114620fe4e5ede5cd3f3e6033ad` |

Final review outcome:

- The final bounded native Codex substitute returned `VERDICT: AGREE`; it is
  explicitly weaker than Claude and inferred no runtime success.
- All ten traced roots require exact argv, allowed writes, and exactly one
  `exit`/`exit_group(0)`. The trusted syntax probe used only `/bin/true` under
  `/tmp`; it was not A3 evidence.
- The signed anchor is acyclic: reviewed harness and semantic-contract hashes
  bind the anchor; the boundary binds the anchor; the fixture binds the exact
  boundary file.
- Terminal verification recursively reopens post-result ledger, final
  checkpoint, executor ledger, current sources/tests/harness, frozen `HEAD`,
  closure receipt, and closure-generation trace without launching a child
  process.
- Both generator and verifier pre-runtime contract loaders accept the signed
  chain. Compilation, scoped whitespace, ten-role census, root-exit adversarial
  cases, backend/RNG/placeholder scans, and A3 cache scans passed.

Skeptical pre-execution audit remains `PASS`: the exact baseline is the analytic
scalar LGSSM plus direct equation simulation; no proxy was promoted; low power
is a repair trigger; all runtime commands preserve the reviewed artifact and
stop boundaries; and no HMC/NeuTra or concurrent HMC/Kalman path is admitted.

Gate status: `A3_FOCUSED_TESTS_AUTHORIZED`.

Next action: execute the exact frozen CPU-hidden focused-test command. Runtime
artifact generation remains contingent on a passing, role-authenticated focused
trace.

## 2026-07-13 - Phase A3 - HARNESS REVIEW ROUND 5 REPAIR

Review finding and classification:

- Fresh hash-specific bounded Codex-substitute review returned
  `VERDICT: REVISE`; it is explicitly weaker than Claude and inferred no
  runtime success.
- The trace parser and role table correctly defined all ten traced commands,
  but the closure machinery authenticated only the focused-test, CPU/GPU
  generation, CPU/GPU verification, and executor-ledger stages. The
  final-checkpoint, post-result-ledger, and closure-generation traces could be
  hash-bound without first proving their exact root argv and allowed writes.
- This was a pre-runtime harness-authentication defect. No A3 focused,
  CPU-reference, or trusted GPU/XLA evidence command had run, so it invalidated
  no numerical, statistical, model, target, data, or scientific result.

Visible repair:

- The executor ledger now authenticates and binds the first five completed
  trace roles.
- Final-checkpoint generation verifies that signed ledger and accumulated
  trace chain, then authenticates the executor-ledger trace.
- Post-result-ledger generation verifies the signed checkpoint and its chain,
  then authenticates the final-checkpoint trace.
- Closure generation verifies the signed post-result ledger and accumulated
  chain, then authenticates the post-result-ledger trace.
- Fresh closure verification validates all signed member/trace bindings and
  authenticates the closure-generation trace. The terminal stdout-only audit
  remains the final role authentication for the closure-verification trace.

Gate status: `A3_HARNESS_ROUND_5_REPAIRED_FRESH_REVIEW_REQUIRED`

No A3 runtime evidence is authorized until the repaired verifier passes
focused static/adversarial checks and a fresh hash-specific bounded review
returns `VERDICT: AGREE`.

## 2026-07-13 - Phase A3 - IMPLEMENTATION REVIEW PASSED, HARNESS GATE ACTIVE

Entry and evidence contract:

- The repaired A2 post-result closure SHA-256 is
  `38724b7064a112c37f71b55349cb8347b1991652d9cb267c0ed8c482af869ac5`.
- Fresh closure verification returned `A2_POST_RESULT_CLOSURE_VERIFIED`; its
  log SHA-256 is
  `8059cc58ef6d9b2ce8bdb64c7f52e9296619a36788eebb6a79db64acadf1d815`.
- The exact terminal verification trace SHA-256 is
  `df559bd15eaa23971f959582bf1f0b988e2b59ee1fa8a98fb63f6b08ed613774`
  and its hardened syscall audit returned
  `A2_TERMINAL_WRITE_TRACE_AUDIT_PASSED`.
- The A3 scoped boundary and fixture contract are frozen at SHA-256
  `31d354a8e00ba8c00107f17fa7f0a3fd51e3292c1bcf23b8f3c9d0a64d6ebc8a`
  and
  `e36e8b27350a3b25e5537a2723e82aea8772387535d6c2bd8cbd05c8c142a085`.
- The skeptical plan audit remains passed: the comparator is the analytic
  scalar LGSSM plus direct equation simulation; proxy diagnostics remain
  explanatory; low power is a repair trigger; no sampler command is admitted.

Implementation and repair evidence:

- Accepted oracle source/test SHA-256 values are
  `74889d699e3575ee163c64d9a67325f0376e161106e9b36fb6b61453c3a5eb43`
  and
  `977134cbc92b63ca6d8dab7a1e6ca25eb58137cb27430518a1aacc120cecfab8`.
- Statistics review Round 1 returned `VERDICT: REVISE` because forged public
  interval containers could manufacture `PASS`, a caller label could make the
  quadratic MMD appear inferential, and dynamically valued scale-floor use
  could fail open.
- The repaired statistics source/test SHA-256 values are
  `99ddaa1dcb15e9f3ec7a5a18f96ebd0f656848c40ea76c896b387cace294bc16`
  and
  `5e6a137c12b3131c8ff7471d74abd4a877777ef6432a2c51f5c62cceedf9290d`.
- Constructor-bound live-object authentication now protects feature and MMD
  intervals; quadratic U/V forms are unconditionally descriptive; runtime
  scale-floor admission fails closed; and roundoff-degenerate long-run MMD
  uncertainty produces an authenticated hard veto.
- The combined CPU-hidden oracle/statistics suite passed `65/65`; the hardened
  statistics subset passed `44/44`. Compilation, scoped whitespace, backend/RNG,
  and A3 cache scans passed.
- Fresh bounded statistics Round 2 returned `VERDICT: AGREE` at the exact
  repaired source hash. The implementation review record SHA-256 is
  `5be205dadb47456758dab78f1b2f8ee35f56af03a71a7e8f83767ac9f5fa5821`.
  This is a Codex substitute review, explicitly weaker than Claude.

Harness gate:

- The prior generator/verifier drafts are invalid and cannot run. They
  regenerated banks from seeds, trusted placeholder checks, directly forged
  decision containers, and did not independently recompute the full evidence
  contract.
- The active repair must make the CPU artifact materialize both independent
  arm banks and the GPU consume those exact persisted bytes. Seed metadata is
  never cross-backend floating-normal replay authority.
- Fresh verification must reconstruct the persisted banks, recompute
  authenticated decisions in process, validate coverage and controlled
  alternatives, verify HLO/device placement and CPU/GPU parity, and fail on any
  unconditional or empty evidence row.

Gate status: `A3_HARNESS_REWRITE_AND_INDEPENDENT_REVIEW_REQUIRED`

No CPU or GPU A3 evidence command is authorized until the repaired generator
and verifier pass static checks and a fresh bounded read-only harness review.

## 2026-07-13T11:27:42+08:00 - Phase A2 - TERMINAL TRACE PARSER REPAIR

Observed failure and classification:

- The prior closure member verification succeeded, but its terminal write
  audit failed because an unanchored `link(` matcher classified read-only
  `readlink(...)` as a filesystem mutation.
- This invalidated the terminal audit and all prior closure ledgers/checkpoints;
  it did not invalidate the A2 forecast implementation, target, CPU/GPU
  numerics, XLA placement, data, model mathematics, or research direction.

Visible repair and fresh evidence:

- Verifier SHA-256 is
  `d0195063a1686a5332b6788bd1171ffc998370bd3578ceeb64edea240a2511ee`;
  focused-test SHA-256 is
  `1812b338ff90633d2fa627642af8ba65425bdaf1c11211f8944d7207ecbded2c`.
- The exact focused suite passed `87 passed, 15112 warnings in 6795.67s`;
  focused trace SHA-256 is
  `9bc681ce9071cc73e94ee0be85e809871cd3486ce778bb6a591bf3cb3471cdaf`.
- The refrozen boundary SHA-256 is
  `674d5124c92fe093f807195039c9d4a243c5ec403563631d5569ea77d2259cfc`;
  boundary trace SHA-256 is
  `78b3d81cdf1f378d54fbe4baa9f8726aa4bb3a032482bfa74d8dfbe5e53fa6be`.
- Fresh CPU artifact SHA-256/evidence signature are
  `8bd1ed508e90674521774f73332e73e2a2f198a057879448dcddc0e30ed35df2` /
  `912ae925d4e7edde6980a84b75b524aab85f9194733f057513c01c6d430a6d52`.
- Fresh trusted GPU artifact SHA-256/evidence signature are
  `0294b06527620336e970bf6a57fd2e0f1a8466502bf47f9595a533d10ca23521` /
  `9bb522772e7cc42aced7f5a7ebbf79fc9579537b35ad219f0d370415c25bebcc`.
- The bounded narrow-contract review is included in exact review SHA-256
  `1210e2fcced29448cbcdba7a4ce1dcee93326e3f317e27ec65d45c30364f23fb`
  and returned `VERDICT: AGREE`; it is weaker than Claude.

Execution-safety amendment:

- Every newly traced A2 closure command must use exactly
  `/usr/bin/strace -f -qq -yy -s 65535 -e trace=%file`.
- The terminal audit accepts exactly one explicit PID and complete parsed
  syscalls; it fails on empty, malformed, truncated, unfinished/resumed,
  multi-PID, unannotated, descriptor-only, metadata, rename, link, xattr, or
  out-of-root mutations.
- The accepted A2 subplan remains byte-for-byte unchanged. The user's explicit
  "fix that and continue" direction authorizes this narrow trace-contract
  repair for the stale closure-regeneration chain, as recorded in the approval
  ledger; it is not an executor-created expansion of authority.

Gate status: `A2_RESULT_AND_A3_PLAN_REFRESH_REVIEW_THEN_CLOSURE_REGENERATION`

Next action: refresh and rereview the exact A2 result and A3 subplan, regenerate
the A2 ledgers/checkpoint/closure under the hardened trace flags, fresh-process
verify closure, and audit the exact final verification trace before any A3
source edit.

## 2026-07-13 - Phase A2/A3 - RESULT CLOSE AND A3 PLAN REVIEW CONVERGED

Observed evidence:

- A2 result status is `PASSED_FOR_A3_PLANNING_ONLY`; result SHA-256 is
  `1585e227371e673ebf9f3af0c9a007dcd0c02cc8945448f9f2a72ddff16d2e9d`.
- A2 bounded result review returned `VERDICT: AGREE`; review SHA-256 is
  `e0a1f6e31b8793778ae1305d17eba1f60cb08421e96ae36cd00ff5c325e1b8f4`.
- The A3 forecast-oracle/statistics subplan was drafted from the accepted A2
  interfaces and passed local structure, ASCII, whitespace, scoped diff, and
  boundary-language checks.
- The trusted one-path Claude Opus gate was rejected before process creation or
  disclosure by the environment's external-data policy. No A3 content was sent
  and no indirect retry was attempted.
- Fresh bounded Codex substitute Round 1 returned `VERDICT: REVISE` on four
  statistical issues: dependent quadratic-MMD unbiasedness, null-degenerate
  bootstrap, missing joint feature/MMD alpha, and tiny-chain resampling.
- The same A3 plan was repaired to separate descriptive IID/dependent quadratic
  U/V forms from a distinct-chain linear MMD inferential estimand, preserve
  forecast clusters and independent-bank semantics, require four chains/two
  disjoint pairs, use chain-stratified block inference, enforce joint alpha,
  and prevent mechanics-only evidence from emitting `PASS`.
- Fresh bounded Codex substitute Round 2 found no material issue and returned
  `VERDICT: AGREE`. This review is explicitly weaker than Claude and is not
  Claude convergence.

Skeptical audit:

- The analytic scalar LGSSM and direct equation simulation remain the baseline;
  no sampler or A2 shared implementation is the oracle.
- Quadratic MMD on dependent paths, common-random-number MMD, high moments,
  quantiles, runtime, equality tests, and one-seed power remain explanatory.
- Decision-bound MMD now has an explicit independent-chain/bank estimand,
  nondegenerate near-null interval route, cluster unit, stationarity/mixing
  admission, and coverage validation.
- Numerical A3 fixtures are labeled test-only; A4 owns final margins,
  bandwidths, tolerances, blocks, alpha allocation, counts, and seeds.
- No HMC, NeuTra, sampler comparison, scientific claim, package/network action,
  Git publication, or concurrent HMC/Kalman edit is authorized.

Gate status: `A3_SUBPLAN_AGREED_A2_POST_RESULT_CLOSURE_REQUIRED`

Next action: generate and verify the A2 post-result ledger and closure, then
terminally audit and record the closure-verification write trace before any A3
source edit.

## 2026-07-12 - Phase A2 - PRE-RUNTIME EVIDENCE CONTRACT

Binding evidence contract:

| Field | A2 contract |
| --- | --- |
| Engineering/scientific question | Does the A2 implementation faithfully and replayably realize the locked approximate-filter predictive law? |
| Exact baseline/comparator | Accepted A1 target and a direct TensorFlow recursion using the same constrained parameters and explicit zero/fixed innovations |
| Primary pass criterion | Complete A2 focused suite plus structured CPU and trusted GPU/XLA canaries pass conjunctively |
| Promotion vetoes | Entry/hash drift; filter or total-target mismatch; invalid covariance; recursion/noise/replay/batch/XLA/GPU failure; invalid artifact |
| Explanatory diagnostics | Runtime, trace count, sub-tolerance parity residuals, covariance spectrum/projection residuals, and changed-seed deltas |
| Not concluded even on pass | Posterior/sampler correctness, predictive equivalence, calibration, model adequacy, superiority, public/default/product readiness, or scientific validity |
| Preservation artifact | A2 JSON/log artifacts, A2 Markdown result, exact commands/run manifests, hashes, reviews, ledgers, and A3 handoff |

Research-intent guard:

- Mechanism under test: target-preserving terminal extraction plus explicit
  Philox terminal/process/observation innovations and ten-step SSL-LSTM
  recursion.
- Expected failure mode: target parity, covariance validity, noise placement,
  replay, scalar/batch, eager/XLA, CPU/GPU, provenance, or write-boundary
  failure.
- Promotion criterion: the conjunctive engineering gates above.
- Promotion veto: any declared hard gate failure.
- Continuation veto: an invalid A1 entry, changed target/model semantics,
  unavailable trusted GPU/XLA route, invalid oracle, corrupted artifact, or
  unexplained mutation of an A2-owned artifact.
- Repair trigger: a localized implementation, test, assertion, serialization,
  or XLA-compatibility defect inside the reviewed A2 write set.
- Explanatory diagnostics cannot promote A2 or support a scientific ranking.
- A failed A2 candidate does not reject forecast-moment validation. Before any
  stop, classify whether the harness, implementation, target, data, math, or
  artifact is invalid, or whether only the current candidate failed.

Skeptical pre-execution audit:

- Baseline is the accepted A1 target and historical `tf_svd_ukf` route, not a
  stale HMC chain or alternate filter.
- Replay, hashes, runtime, HLO, and local residuals remain engineering checks;
  none is silently promoted into predictive equivalence or scientific
  validity.
- Stop conditions cover entry drift, parity, covariance, recursion, XLA/GPU,
  artifact, review, and write-boundary failures.
- Shared diagnostic and independent-arm innovation roles remain distinct, so
  A2 cannot preselect a favorable later comparison.
- Approximate terminal Gaussian, PSD policy, complete-state draw, noise
  placement, timing, horizon, dtype, cluster unit, and CPU/GPU roles are
  explicit.
- The structured artifacts contain the tensors and provenance needed to answer
  the A2 engineering question while preserving all forbidden claims.

Audit disposition: `PASS_FOR_HASH_BOUND_A2_IMPLEMENTATION_AND_BOUNDARY_FREEZE`.

Review authorization:

- Agreed subplan SHA-256:
  `6b6b9799782be3304ecbd2dee465c52285688b5e2d1b3087d911ccad1279bbb0`.
- Exceptional Round 6 review:
  `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-subplan-codex-substitute-review-round6-2026-07-12.md`.
- Verdict: `AGREE`.
- Review class: `CODEX_SUBSTITUTE_REVIEW`, explicitly weaker than Claude;
  Claude remains policy-unavailable and will not be probed.

Implementation state:

- Drafted only the reviewed A2 production module, five additive lazy dataclass
  exports, focused tests, generator, and independent verifier.
- No A2 TensorFlow or pytest runtime has run yet.
- Next gate is the scoped pre-run boundary, followed by static checks and the
  exact CPU-hidden focused test command.

Boundary attempt 1:

- The default-sandbox `strace` invocation failed before the verifier process
  started because `ptrace` was not permitted; no boundary JSON was written.
- The identical trusted invocation completed, but its closed trace showed that
  `git status` transiently created and removed `.git/index.lock` while
  refreshing the index.
- This violates the reviewed no-outside-write trace contract even though the
  accepted source/content hashes remained unchanged. The first boundary and
  trace are therefore invalid and cannot support runtime execution.
- Repair: set `GIT_OPTIONAL_LOCKS=0` for every verifier/generator Git
  subprocess and use captured pipes rather than `/dev/null` for tracked-state
  probes, then regenerate the same boundary and trace before any TensorFlow
  import.
- Classification: localized verifier/write-attribution defect and A2 repair
  trigger, not target, data, model, math, GPU, or research-direction failure.

Focused-test attempt 1:

- The exact CPU-hidden focused run reached `..F` and exited on
  `test_terminal_covariance_fail_closed_policy` before the compiled fixture.
- A one-test traced diagnostic showed the production covariance audit returned
  valid status for exact-singular and roundoff-negative PSD inputs as intended.
- The test incorrectly indexed tuple element `6` (scalar minimum eigenvalue)
  and then attempted `[0]`; clipped eigenvalues are tuple element `5`.
- Repair: change only that test index from `6` to `5`, refreeze the source
  boundary, and rerun the complete exact focused command.
- Classification: localized test-oracle indexing defect and A2 repair trigger,
  not implementation, target, covariance-math, model, GPU, or research-
  direction failure.

Focused-test orchestration attempts 2 and 3:

- The tool wrapper yielded partial pytest output (`...`) but did not relay the
  nested process session identifier. Treating that yield as completion caused
  a second identical traced run to start while the first was still compiling.
- A trusted process-table check found both A2-only `strace`/pytest pairs still
  active, with no traceback, signal, kernel OOM record, or disk exhaustion.
- Terminated exactly PIDs `2660356`, `2660359`, `2665496`, and `2665499`, all
  launched by this A2 lane. No unrelated process was touched.
- Their shared overwritten focused trace is invalid and will not support A2
  evidence.
- Repair: launch one clean exact focused command, retain its returned unified
  process session identifier, and poll that session to a real exit code before
  any further run.
- Classification: executor/tool-session orchestration invalidity, not a test,
  implementation, XLA, target, model, GPU, or research-direction failure.

Focused-test valid attempt 4:

- The single traced process completed with `2 failed, 60 passed` in
  `6848.26s`; the A1 regression module and all terminal extraction, covariance,
  replay, zero-bank, HLO, provenance, invalid-input, and lazy-export checks
  outside the two named tests passed.
- `test_forecast_recursion_noise_placement_and_replay` used bitwise equality
  for the subtraction identity
  `state_z - deterministic_z == process_innovation`. The printed values agreed,
  but cancellation produced sub-ulp differences. The reviewed contract assigns
  this identity the `128 * eps` recursion tolerance, not bitwise equality.
- `test_scalar_batch_order_and_eager_xla_parity` ran the historical filter and
  analytic comparator eagerly but compared their total to the compiled
  `SSLLSTMPosteriorTarget.value`, producing status `64`. Production compiled
  admission remains correctly pinned to `64 * eps`; eager/XLA comparison has a
  separate reviewed `512 * eps` gate.
- Repair: use `128 * eps` for process/observation subtraction and model-scale
  assertions; for the eager debug terminal only, compare against
  `SSLLSTMPosteriorTarget.eager_debug_value_and_score`, then retain the existing
  `512 * eps` eager-versus-XLA path comparison.
- Classification: one test-oracle tolerance defect and one eager-reference
  comparator-selection defect. Neither invalidates the compiled terminal gate,
  accepted A1 target, model/data/math, GPU route, or research direction.

## 2026-07-12 - Phase A1 - FINAL RESULT ACCEPTED

Actions and evidence:

- Added one function-local `import tensorflow as tf` to
  `bayesfilter/inference/hmc.py::static_unroll_chain_value_and_score`.
- The previously failing focused XLA integration test passed.
- The exact complete four-file suite passed `75/75` in `812.42s`.
- All `23` protected dependencies, A1 source, golden, entry, boundary, and
  evidence hashes remained unchanged; `HEAD` remained the A0 anchor with no
  intervening commits.
- CPU evidence fully recomputed with unchanged signature
  `c208b513e2fbf74d654b3b349695a7fcb811b2a6c36f5c2fa76a30dd5e9c922d`.
- Trusted GPU/XLA evidence fully recomputed on two RTX 4080 SUPER devices with
  unchanged signature
  `077abbd5d5d8dc1068d99aba90fc8b6dd5b74001cda1dd1fe4428d13a0b4631c`.
- Final A1 result SHA-256
  `78f269a53fb0536017d32bd12c2b36967cd013a85dcb1102936ed79ae95e34b5`
  received `CODEX_SUBSTITUTE_REVIEW` `VERDICT: AGREE`.

Gate status: `PASSED_FOR_A2_PLANNING_ONLY`

Nonclaims: no HMC/NeuTra, posterior, predictive-equivalence, calibration,
model-adequacy, performance, default, product, release, or scientific result.

Next action: draft the dedicated A2 terminal-state/forecast API subplan from
the accepted A1 contracts, audit and review it, and do not implement A2 before
that review agrees.

## 2026-07-12T05:27:06+08:00 - Phase A1 - EXCLUDED HMC INTEGRATION BLOCKER

Evidence contract:

- Question: does the A1-owned masked posterior preserve the locked target and
  pass the reviewed CPU and trusted GPU/XLA ten-point gates plus the complete
  final local checkpoint?
- Baseline: A0-locked historical SVD-UKF target and exact CPU artifact.
- Primary criterion: all A1 source/artifact gates and the exact four-file
  pytest command pass conjunctively.
- Veto: any mandatory integration-test failure blocks A2 handoff.
- Nonclaims: no HMC/NeuTra, posterior, predictive, calibration, adequacy,
  performance, default, product, or scientific result.

Actions and evidence:

- Reconfirmed accepted A1 source, golden, entry, boundary, CPU, and review
  hashes and `23/23` protected dependency rows.
- Generated the trusted GPU/XLA artifact on two RTX 4080 SUPER devices. All ten
  frozen points passed; maximum CPU/GPU value residual was `0.0` and score
  infinity residual was `3.552713678800501e-15`.
- Verified the GPU artifact twice by full fresh-process recomputation with
  unchanged signature `077abbd5d5d8dc1068d99aba90fc8b6dd5b74001cda1dd1fe4428d13a0b4631c`.
- Verified the CPU artifact by full fresh-process recomputation with unchanged
  signature `c208b513e2fbf74d654b3b349695a7fcb811b2a6c36f5c2fa76a30dd5e9c922d`.
- Static compilation, forbidden-source scans, whitespace audit, and the
  A1-owned module's `52` tests passed.
- The exact broader suite returned `1 failed, 74 passed`: excluded concurrent-
  lane `bayesfilter/inference/hmc.py` calls `tf.TensorArray` without defining
  `tf`. This lane did not edit that file.

Gate status: `BLOCKED_PHASE_A1_EXCLUDED_HMC_INTEGRATION_TEST`

Next action: after the owning HMC lane stabilizes, rerun the single failing
test and exact 75-test suite. Only a clean final checkpoint permits A1 result
review and A2 subplan drafting.

Blocker-result review:

- Round 1 found incomplete CPU/GPU manifests, inconsistent resume-verifier
  logic, missing review hashes, and compressed finite/reject scope wording.
- The same result was repaired without changing source or evidence artifacts.
- Round 2 accepted exact result SHA-256
  `84f843b7c6bc49cb2a3d1db5af85e24d24cbe7cbb5b931f84dfb9de0d232d1d7`
  with `VERDICT: AGREE`.
- This is a `CODEX_SUBSTITUTE_REVIEW`, explicitly weaker than Claude, and
  accepts only the blocker classification and resume boundary.

## 2026-07-11T04:18:35+08:00 - Program - PRECHECK

Evidence contract:

- Question: define and execute a gated path from the historical scalar fixture
  to an engineering-complete, predictively validated SSL-LSTM vertical slice.
- Baseline: fresh ordinary HMC on one A0-locked target; Phase 2V and failed
  parameter-reference artifacts are context only.
- Primary criterion: the roadmap, active subplan, results, next-phase handoffs,
  local checks, and bounded reviews all pass in sequence.
- Vetoes: wrong/stale target, proxy promotion, missing stop/handoff, unfair
  comparison, environment mismatch, invalid artifact, or authority violation.
- Nonclaims: no implementation, HMC/NeuTra validity, predictive equivalence,
  calibration, GPU/default/release readiness, or scientific claim yet.

Actions:

- Loaded the scalar reset memo, governing predictive-equivalence program,
  LaTeX model chapter, roadmap draft, Claude review-gate guide, review/runbook
  templates, and `claude-code-workers` skill.
- Inspected the historical target, fixture, parameter chart, Phase 2S context,
  current repository status, and current `dsge_hmc` NeuTra governance.
- Preserved the heavily dirty worktree and restricted writes to new SSL-LSTM
  completion planning/review artifacts.

Skeptical audit:

- Wrong baseline: repaired by requiring fresh four-chain ordinary HMC and
  keeping Phase 2V diagnostic only.
- Proxy promotion: loss, hashes, smoke checks, higher moments, and parameter
  similarity are non-promoting outside their declared gates.
- Missing stops: target/oracle/sampler/artifact/review/boundary stops are explicit.
- Fairness: sampler arms share target/data/forecast/design and tune separately.
- Hidden assumptions: SVD-UKF approximation, unnormalized prior convention,
  four-parameter mask, simulated data, and Phase 2S role are explicit.
- Stale context: opening/closing hashes and the 2026-07-10 reset memo are bound.
- Environment: A0 CPU replay is deliberately GPU-hidden; serious later routes
  remain GPU/XLA.
- Artifact sufficiency: A0 requires semantic component hashes, not file names only.

Gate status: `PASSED_FOR_PLANNING_AND_PHASE_A0_REVIEW_ONLY`

Next action: review and repair the roadmap, A0 subplan, and visible runbook.

## 2026-07-11T04:18:35+08:00 - Program - GOVERNANCE_REPAIR

Actions:

- Separated target-semantic, execution, sampler-geometry, and forecast-design
  identities.
- Made the historical unnormalized Gaussian prior-value convention explicit.
- Classified Phase 2S center/covariance as tuning context, not target identity.
- Added a just-in-time A6 NeuTra governance audit; `dsge_hmc` evidence remains
  design context and cannot validate BayesFilter.
- Created the A0 subplan, visible runbook, approval ledger, and stop handoff.

Gate status: `IN_PROGRESS_REVIEW_REQUIRED`

Next action: run scoped local checks, create one-path review bundles, and invoke
the trusted Claude review gate.

## 2026-07-11T04:29:00+08:00 - Program - CLAUDE_REVIEW_BOUNDARY_BLOCKED

Evidence contract:

- Question: may the single roadmap path be disclosed to Claude Opus for the
  requested bounded read-only material review?
- Baseline: no repository content has been sent to Claude in this program.
- Primary criterion: trusted execution permits the exact review-gate wrapper,
  one compact bundle, and one exact target path.
- Vetoes: trusted approval rejection, broader disclosure, indirect workaround,
  or false classification of policy rejection as reviewer death.
- Nonclaims: no Claude review, no Claude liveness result, no roadmap
  convergence, no A0 authorization, and no Codex substitute review.

Actions:

- A dry run validated the bounded command/prompt and wrote
  `.claude_reviews/20260711-042900-bayesfilter-ssl-lstm-completion-roadmap-r1/status.json`
  with `REVIEW_STATUS=dry_run` and `VERDICT=NONE`.
- Requested trusted execution of
  `/home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh` using Claude
  Opus/max, a 90-second low-effort probe, a 180-second material timeout, one
  retry, and the one-path roadmap bundle.
- The trusted execution layer rejected the action before process creation as
  external third-party repository-content disclosure risk.
- No probe ran and no roadmap content was sent externally.
- Did not retry indirectly, invoke a different Claude wrapper, execute A0, or
  claim Claude is dead. The fresh Codex substitute condition is not met because
  transport/liveness was never tested.

Artifacts:

- `docs/reviews/bayesfilter-ssl-lstm-completion-roadmap-review-bundle-2026-07-11.md`
- `docs/plans/bayesfilter-ssl-lstm-completion-claude-review-boundary-blocker-2026-07-11.md`
- `.claude_reviews/20260711-042900-bayesfilter-ssl-lstm-completion-roadmap-r1/status.json`

Gate status: `BLOCKED_PENDING_EXPLICIT_INFORMED_EXTERNAL_DISCLOSURE_APPROVAL`

Next action: obtain explicit informed user approval to disclose the exact
roadmap path to Claude, then retry the same narrow gate. If approval is not
granted, obtain explicit direction to replace Claude reviews with bounded
`CODEX_SUBSTITUTE_REVIEW`; do not infer either authority.

## 2026-07-11T05:02:26+08:00 - Program - CLAUDE_DISCLOSURE_APPROVED

Actions:

- The user explicitly approved sending exact one-path BayesFilter planning and
  result documents to Claude through the bounded read-only review gate and
  acknowledged the external-disclosure risk.
- This approval does not cover whole-repository review, multiple-path packets,
  source/model-file disclosure unless separately named as the one exact review
  target, mutation, execution, or any scientific/product/default boundary.
- The roadmap review will be retried with the same compact bundle and exact
  target path used by the rejected attempt.

Gate status: `EXTERNAL_DISCLOSURE_BOUNDARY_RESOLVED_REVIEW_PENDING`

Next action: invoke the trusted roadmap review gate, then repair and rereview
according to the visible runbook.

## 2026-07-11T05:03:00+08:00 - Program - CLAUDE_POLICY_UNAVAILABLE

Actions:

- Retried the identical one-path roadmap gate after the user's explicit
  informed approval.
- The trusted execution layer again rejected the action before process
  creation, explicitly stating that private-workspace disclosure to the
  external service is forbidden even with user approval.
- No Claude process or probe ran and no repository content was sent.
- Claude is not classified as dead or transport-down. The external review route
  is policy-unavailable in this environment.
- Per the execution policy's requirement to use a materially safer alternative
  and the user's requested fresh-Codex fallback, activated bounded native Codex
  read-only substitute review. It must be labeled `CODEX_SUBSTITUTE_REVIEW`, is
  weaker than Claude review, and cannot be called Claude convergence.

Gate status: `CLAUDE_POLICY_UNAVAILABLE_CODEX_SUBSTITUTE_REVIEW_ACTIVE`

Next action: obtain a fresh Codex roadmap review, repair visibly, and repeat up
to five substantive rounds before proceeding to independent A0/runbook reviews.

## 2026-07-11 - Program - ROADMAP_SUBSTITUTE_REVIEW_CONVERGED

Actions:

- Round 1 found premature M2 closeout, contradictory Track B NeuTra promotion,
  and missing bounded A6/A7 exhaustion handoffs.
- Round 2 found missing ordinary-HMC-only A9 selection and an overbroad
  transform-only M1 veto.
- Round 3 found uncovered A9 outcomes when NeuTra/M2 is governance-blocked,
  materially different, underpowered, or fails audit replication.
- Patched the same roadmap after each round and reran focused document checks.
- Fresh Round 4 returned `VERDICT: AGREE` with no material findings.

Artifacts:

- `docs/reviews/bayesfilter-ssl-lstm-completion-roadmap-codex-substitute-review-round1-2026-07-11.md`
- `docs/reviews/bayesfilter-ssl-lstm-completion-roadmap-codex-substitute-review-round2-2026-07-11.md`
- `docs/reviews/bayesfilter-ssl-lstm-completion-roadmap-codex-substitute-review-round3-2026-07-11.md`
- `docs/reviews/bayesfilter-ssl-lstm-completion-roadmap-codex-substitute-review-round4-2026-07-11.md`

Gate status: `PASSED_WEAKER_CODEX_SUBSTITUTE_REVIEW`

Nonclaims: not Claude convergence and no phase runtime or scientific gate pass.

Next action: converge the A0 subplan review, then review the A0 harness before
the dependency-discovery/reference replay.

## 2026-07-11 - Phase A0 - SUBSTITUTE_REVIEW_RETRY

Actions:

- A0 review rounds 1 and 2 returned material schema, probe, dependency,
  fingerprint, geometry, and stop-semantics findings; the same subplan was
  visibly repaired after each round.
- The first Round 3 substitute reviewer remained active through repeated
  bounded waits and a direct request to conclude, returning no verdict.
- Interrupted that stalled reviewer. The stalled attempt is not an agreement,
  not a substantive review round, and not Claude evidence.
- Launched one fresh bounded Round 3 retry against the same exact A0 path with
  an eight-finding response cap.

Gate status: `A0_CODEX_SUBSTITUTE_REVIEW_ROUND3_RETRY_ACTIVE`

Next action: assess the fresh verdict; repair or proceed to harness creation
only if no material finding remains.

## 2026-07-11T06:06:17+08:00 - Phase A0 - REVIEW_NONCONVERGENCE_STOP

Evidence contract:

- Question: did the terminal fifth substantive A0 subplan review leave any
  material execution blocker?
- Baseline: the Round 4-repaired A0 subplan, reviewed as one exact path by a
  fresh bounded native Codex substitute reviewer.
- Primary criterion: explicit `VERDICT: AGREE` with no material finding.
- Veto: any material Round 5 finding; the predeclared five-round cap then
  requires a blocker result and human direction rather than a sixth repair.
- Nonclaims: no runtime, target replay, TensorFlow, HMC, NeuTra, predictive,
  calibration, GPU/XLA, model-adequacy, or scientific result.

Actions:

- Local contradiction, placeholder, whitespace, and scoped source-context
  checks completed before the terminal review.
- The first bounded Round 5 attempt stalled without a verdict and was
  interrupted; it did not consume a substantive round.
- A fresh replacement Round 5 reviewer returned `VERDICT: REVISE` because the
  mandatory phase-end sequence orders all required checks before the reference
  replay writes the lock those checks must parse and verify.
- Preserved the exact proposed sequencing repair in the blocker result, but did
  not patch the A0 subplan after the review cap was exhausted.
- Stopped before creating or executing the A0 target-lock harness.

Artifacts:

- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-codex-substitute-review-round5-2026-07-11.md`
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a0-review-nonconvergence-blocker-result-2026-07-11.md`
- `docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md`

Gate status: `BLOCKED_A0_REVIEW_CAP_EXHAUSTED_HUMAN_DIRECTION_REQUIRED`

Next action: obtain explicit human approval for one focused patch plus one
extra bounded substitute-review round, or a manual waiver of the extra review.

## 2026-07-11 - Phase A0 - HUMAN_AUTHORIZED_FOCUSED_RECOVERY

Actions:

- The user authorized the recommended over-cap recovery work.
- Patched only the mandatory phase-end sequence so the replay generates the
  structured lock before immediate strict verification, result/A1 drafting and
  review occur next, and final strict verification/rehash occurs immediately
  before handoff.
- Authorized exactly one exceptional bounded native Codex substitute-review
  round of the same A0 subplan. No manual waiver or unbounded review loop is
  inferred.

Gate status: `A0_EXCEPTIONAL_FOCUSED_REVIEW_AUTHORIZED`

Next action: run focused checks and the exceptional one-path review. Proceed to
the A0 harness only on explicit `VERDICT: AGREE`.

## 2026-07-11 - Phase A0 - FOCUSED_RECOVERY_CONVERGED

Actions:

- The human-authorized exceptional one-path Codex substitute review found no
  material issue in the repaired lifecycle.
- The review returned `VERDICT: AGREE` and preserved the unconditional final
  rehash and immutable-member restart requirements.
- The prior review-cap blocker is resolved only for this exact sequencing
  repair. All A0 implementation, harness-review, replay, artifact, and handoff
  gates remain active.

Artifact:

- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-codex-substitute-review-exceptional-2026-07-11.md`

Gate status: `A0_SUBPLAN_PASSED_HUMAN_AUTHORIZED_EXCEPTIONAL_SUBSTITUTE_REVIEW`

Next action: implement the A0-only target-lock harness, run local syntax checks,
and obtain an independent one-path harness review before execution.

## 2026-07-11 - Phase A0 - HARNESS_REVIEW_CONVERGED

Actions:

- Implemented the A0-only discovery/generation/strict-verification harness.
- Local `py_compile`, CLI-help, whitespace, negative-path, and focused contract
  checks passed.
- Substitute review found and repaired self-certified target/probe fields,
  missing exact implementation bindings, path aliasing, rank-zero tensor
  canonicalization, and premature opening/closing lifecycle checks.
- The strict verifier now performs a fresh CPU-hidden target replay after cheap
  structural/hash checks and compares observations and complete probe payloads.
- Final bounded review returned `VERDICT: AGREE` with no material finding.

Artifacts:

- `docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py`
- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-harness-codex-substitute-review-2026-07-11.md`

Gate status: `A0_HARNESS_PASSED_WEAKER_CODEX_SUBSTITUTE_REVIEW`

Next action: run the exact CPU-hidden non-evidentiary dependency-discovery pass.

## 2026-07-11 - Phase A0 - DEPENDENCY_DISCOVERY_PREFLIGHT_FAILED

Command:

`CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_DETERMINISTIC_OPS=1 TF_ENABLE_ONEDNN_OPTS=0 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/benchmark_ssl_lstm_completion_phase_a0_target_lock_2026_07_11.py --discover-dependencies docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json --log-path docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-discovery.log`

Result:

- Exit code `1` before TensorFlow import and before any manifest/log write.
- The invoked interpreter was the reviewed symlink path
  `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, but the harness resolved it to
  its binary target `/home/ubuntu/anaconda3/envs/tfgpu/bin/python3.13` before
  comparing it to the reviewed command path.
- The artifact directory remained empty. No immutable attempt opened.

Classification: fixable harness provenance bug, not environment, target, data,
math, or scientific failure.

Repair: compare the absolute invoked `sys.executable` path to the reviewed
command path without resolving its symlink, then focused-review the repair
before retrying discovery.

Gate status: `A0_HARNESS_REPAIR_AND_REREVIEW_REQUIRED`

## 2026-07-11 - Phase A0 - DEPENDENCY_DISCOVERY_GEOMETRY_CONTRACT_FAILED

Result:

- After the interpreter repair passed focused review, the exact CPU-hidden
  discovery command reached target/geometry warm-up and exited `1` before
  writing a manifest or log.
- TensorFlow emitted a failed `cuInit` startup message despite
  `CUDA_VISIBLE_DEVICES=-1`. This is nontrusted framework-startup evidence only;
  it is not a GPU run, GPU probe result, or diagnosis of machine/driver state.
- The decisive failure was `Phase 2S geometry identities failed`.

Focused NumPy diagnostics:

| Check | Residual | Reviewed tolerance | Status |
| --- | ---: | ---: | --- |
| `factor_z @ factor_z.T = covariance_z` | `1.2212453270876722e-15` | `1.1465297583454372e-13` | passed |
| raw `D @ covariance_z @ D = covariance_theta` | `7.5035342522733472e-10` | `1.4210854715202004e-14` | failed |
| raw `Dinv @ precision_z @ Dinv = precision_theta` | `1.0000462680181954e-9` | `1.0059070632816792e-12` | failed |
| `precision_z @ covariance_z = I` | `1.3224903417406733e-15` | `9.4588741971174453e-13` | passed |
| `precision_theta @ covariance_theta = I` | `4.8006170146228212e-15` | `9.458874189703217e-13` | passed |

Source audit:

- Phase 2S transforms `precision_z` to raw theta precision, then
  `covariance_from_precision` symmetrizes it, adds `jitter=1e-9`, applies the
  eigenvalue-floor/condition-cap rule, and inverts the regularized precision.
- Therefore the two raw cross-coordinate equalities were incompatible with the
  cited source route. The historical artifact is internally consistent; the A0
  plan asserted the wrong identities.
- Source-aware reconstruction passed: regularized theta precision residual
  `4.6185277824406512e-14` versus tolerance `1.0059070632816792e-12`, and
  rebuilt theta covariance residual `2.0469737016526324e-16` versus tolerance
  `1.4210854715202004e-14`.

Classification: fixable plan/schema/harness defect. It does not invalidate the
target, data, Phase 2S artifact, sampler idea, predictive-validation proposal,
or SSL-LSTM research direction. No immutable attempt opened and the artifact
directory remained empty.

Repair: encode the cited source regularization path as the pass/fail identity,
retain raw-to-regularized cross-coordinate residuals as explanatory metadata,
and focused-review the same A0 subplan/harness before retrying discovery.

Gate status: `A0_GEOMETRY_CONTRACT_REPAIR_AND_REREVIEW_REQUIRED`

## 2026-07-11 - Phase A0 - GEOMETRY_CONTRACT_REPAIR_CONVERGED

Actions:

- Patched the A0 plan/schema and harness to reproduce the cited Phase 2S
  precision transformation plus `covariance_from_precision` regularization.
- Preserved raw-to-stored covariance/precision discrepancies as exact
  explanatory diagnostics only.
- Added the effective eigenvalue floor and clipped-eigenvalue count to the
  structured checks.
- NumPy-only source-aware reconstruction passed.
- Focused harness and final plan reviews both returned `VERDICT: AGREE`.

Artifact:

- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-geometry-contract-codex-substitute-review-2026-07-11.md`

Gate status: `A0_GEOMETRY_CONTRACT_PASSED_WEAKER_CODEX_SUBSTITUTE_REVIEW`

Next action: retry the exact CPU-hidden non-evidentiary dependency-discovery
pass. No immutable attempt has opened and the artifact directory is empty.

## 2026-07-11 - Phase A0 - DEPENDENCY_DISCOVERY_PASSED

Result:

- Exact CPU-hidden discovery command exited `0`.
- Dependency closure stabilized after `2` cycles with `43` runtime-loaded local
  Python modules.
- The harness strictly reloaded and verified the dependency manifest, including
  exact keys, byte hashes, Git statuses, semantic aggregate, environment, and
  stable module ordering.
- Manifest semantic aggregate:
  `62204165dc6fe2d46951adaaaa17423e0681affc79c9933952789a93c0c4af25`.
- Manifest exact-file SHA-256:
  `f17dce0f37466199404224168f3848fc781d85498aa70a668ddc2f867908c4e1`.
- Harness SHA-256:
  `f4ebd8ab30369bb0aff052a33ba12a02ff09c171c465d7d2af62f834a4edd4cb`.
- TensorFlow again emitted a failed `cuInit` startup message under
  `CUDA_VISIBLE_DEVICES=-1`; this is recorded as a CPU-hidden framework-startup
  anomaly, not trusted GPU evidence or a GPU/machine diagnosis.

Artifacts:

- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-manifest.json`
- `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/dependency-discovery.log`

Evidence role: non-evidentiary preflight closure only. No immutable attempt had
opened at discovery completion, and no target lock existed.

Gate status: `A0_DEPENDENCY_DISCOVERY_PASSED_READY_FOR_EVIDENTIARY_ATTEMPT`

## 2026-07-11T20:37:25+08:00 - Phase A0 - EVIDENTIARY_ATTEMPT_01_CLOSED

Result:

- The first CPU-hidden evidentiary generation exited `0` and wrote the target
  lock and generation log. Immediate fresh-process strict verification failed.
- The verifier compared fresh binary64 evaluations with decimal-serialized
  historical JSON anchors using exact equality. The observed value pairs were:
  `-37.847429129540124` versus `-37.84742912954012` at `truth_free`, and
  `-37.77528495512358` versus `-37.77528495512359` at `phase2s_center`.
- Each absolute value difference was `7.105427357601002e-15`, within the
  existing `8 * eps64` scale-aware numerical envelope. Every score coordinate
  matched exactly.
- Fresh-process equality against the newly generated lock was not weakened and
  did not run past the earlier historical-anchor rejection. The failed attempt
  cannot be promoted or reused as the successful A0 attempt.

Classification: fixable verifier/contract defect caused by historical JSON
decimal round-tripping. This does not establish target drift and does not
invalidate the target, data, source geometry, model, or research direction.

Preserved artifacts and exact SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| `failed-attempt-01/dependency-discovery.log` | `877246ce4537426eb3dcf8890aabbeed842f57903d1d3f36d5fbab1dc3686946` |
| `failed-attempt-01/dependency-manifest.json` | `f17dce0f37466199404224168f3848fc781d85498aa70a668ddc2f867908c4e1` |
| `failed-attempt-01/target-lock.json` | `a30b413a96a56d321f9451e02a9201aec1635ee9a790efdd1641352da264f766` |
| `failed-attempt-01/target-lock.log` | `a04e80a24a0c605294eb2a66b3a685b3db13ea8f4b1a366599bd3b22c49acdae` |

The preserved-attempt immutable aggregate is
`6dc7b4942a96cae4a10e77313b54eef60406bf8826ae872763f2c0d9a02e2f2c`.

Repair contract: use
`8 * eps64 * max(1, abs(current), abs(historical))` for historical scalar
anchors and the corresponding vector infinity-norm formula for historical
scores only. Keep fresh-process lock replay, tensor/hash, dependency-closure,
and immutable-fingerprint checks exact. Repair and rereview before regenerating
discovery or opening a new attempt.

Gate status: `A0_FAILED_ATTEMPT_01_PRESERVED_TOLERANCE_REPAIR_REVIEW_REQUIRED`

## 2026-07-11 - Phase A0 - HISTORICAL_ANCHOR_REPAIR_REVIEW_CONVERGED

Actions:

- Patched the A0 contract and verifier with scale-aware comparisons limited to
  the two decimal-serialized historical JSON value/score anchors.
- Kept fresh-process generated-lock replay, tensor and component hashes,
  dependency closure, and immutable fingerprints exact.
- `py_compile`, scoped trailing-whitespace scan, archive rehash, and scoped
  `git diff --check` passed. The whitespace scan returned `1` because it found
  no matching trailing whitespace, which is the expected clean result.
- Fresh independent plan and harness `CODEX_SUBSTITUTE_REVIEW` verdicts both
  returned `AGREE`. The initial no-verdict reads were prompt/tool-truncation
  recovery and were not substantive rounds.

Review artifact:

- `docs/reviews/bayesfilter-ssl-lstm-completion-phase-a0-historical-anchor-tolerance-codex-substitute-review-2026-07-11.md`

Review status: weaker than Claude; Claude remains policy-unavailable, was not
probed, and was not sent repository content.

Gate status: `A0_HISTORICAL_ANCHOR_REPAIR_PASSED_READY_FOR_FRESH_DISCOVERY`

## 2026-07-11 - Phase A0 - EVIDENTIARY_ATTEMPT_02_IMMEDIATE_VERIFIER_PASSED

Evidence contract:

- Question: does a fresh immutable attempt reproduce and strictly verify the
  A0-locked historical target after the reviewed historical-anchor repair?
- Baseline: the same two historical target anchors and exact A0 semantic/
  implementation/geometry/forecast contracts; failed attempt 01 is archival
  only.
- Primary criterion: generation and immediate fresh-process strict verification
  both exit `0`; all integrity/replay checks other than the explicitly scoped
  historical JSON anchor tolerance remain exact.
- Vetoes: target/closure/hash/immutable drift, nonfinite probe, source-aware
  geometry failure, or fresh-lock replay mismatch.
- Nonclaims: no posterior, sampler, NeuTra, forecast, GPU, default, or
  scientific evidence.

Result:

- Fresh discovery stabilized in `2` cycles with `43` local modules.
- Manifest semantic aggregate:
  `9718ba393521486d1b63ae19c31c59e2ec636889002d2363c696523ac8ca5f9b`.
- Manifest exact-file SHA-256:
  `2a1e3dcd89c0c5e24f892d14a29ef44329ef3e50c4af40093428082de6806517`.
- Harness SHA-256:
  `e8bb6e8dbc861f9c63982e8ea4f67d2cfa4c6cf413ab9e5d5ec5763858af6954`.
- Generation exited `0` with immutable aggregate
  `6bac37213729c06dfbf6953f60637ffe7f6c0da11cb60a62eba5786b1e44801f`.
- Immediate fresh-process verification exited `0` with the same immutable
  aggregate and signature aggregate
  `af75eda2abb9a4e0260e7dc0921ee43ddd7b0c37c61bafabba8683d70ad155cc`.
- Target-lock exact-file SHA-256:
  `82db318580a44aaff87955aa7fed6b880d25d3fb5c107cf77f55735ff671b62a`.
- TensorFlow again emitted a failed `cuInit` startup message under deliberate
  CPU hiding. It remains a CPU-hidden framework anomaly only.

Gate status: `A0_ATTEMPT_02_IMMEDIATE_VERIFIER_PASSED_RESULT_A1_REVIEW_REQUIRED`

## 2026-07-11T21:42:18+08:00 - Phase A0 - RESULT_REVIEW_AGREED_A1_ROUND2_ACTIVE

Actions:

- Wrote the provisional A0 close record with final rehash explicitly pending.
- A fresh one-path A0 result `CODEX_SUBSTITUTE_REVIEW` returned `VERDICT: AGREE`
  with no material finding.
- Drafted the just-in-time A1 subplan from the verified A0 lock.
- A1 Round 1 returned eight material operational findings: inexact entry/write
  boundaries, self-certified signatures, underspecified callable and reject
  semantics, incomplete artifact/command contracts, proxy GPU promotion,
  insufficient reviewer independence, stale closure ordering, and undefined
  repair bounds.
- Patched the same A1 subplan. The repair now has exact A0 entry command/hashes,
  paths and scratch controls, independent golden signatures, compiled scalar/
  batch/custom-gradient semantics, nonfinite-input-only reject behavior,
  strict CPU/GPU schemas and commands, frozen-point-only GPU status,
  hash-pinned independent reviews, five-cycle repair bounds, and a final
  nonmutating evidence checkpoint before result/A2 review.
- Scoped whitespace and `git diff --check` passed. A fresh Round 2 review is
  active against exact subplan SHA-256
  `a2db5a8a00fb1f6ce8e8864c8a288daebc86e2419b2cd6f4af1a6e52c702f69e`.

Review status: Claude remains policy-unavailable; all substitute reviews are
explicitly weaker and no Claude process/content disclosure occurred.

Gate status: `A0_RESULT_AGREED_A1_SUBPLAN_CODEX_SUBSTITUTE_REVIEW_ROUND2_ACTIVE`

## 2026-07-11T23:24:00+08:00 - Phase A0 - FINAL_REHASH_MUTABLE_PROVENANCE_VETO

Evidence contract:

- Question: did the final A0 verifier reproduce the accepted immutable target
  evidence after A1 planning converged?
- Exact baseline: Attempt 02 target lock SHA-256
  `82db318580a44aaff87955aa7fed6b880d25d3fb5c107cf77f55735ff671b62a`
  under the unchanged reviewed harness.
- Primary criterion: the identical CPU-hidden verifier exits `0` with immutable
  aggregate `6bac37213729c06dfbf6953f60637ffe7f6c0da11cb60a62eba5786b1e44801f`
  and signature aggregate
  `af75eda2abb9a4e0260e7dc0921ee43ddd7b0c37c61bafabba8683d70ad155cc`.
- Vetoes: immutable, target, probe, signature, dependency, or unexplained
  provenance drift.
- Nonclaims: no A1 implementation, HMC, NeuTra, forecast, GPU, posterior,
  predictive, calibration, default, product, or scientific evidence.

Result:

- The identical verifier exited nonzero before target replay with
  `ContractError: governance provenance drift`.
- Exactly one governance descriptor was stale: the A0 subplan changed from
  locked SHA-256
  `330b4f1bfd0820700e2dcc91d982a63e5f086281bc4d4ee140879ab48ccdb53b`
  to reviewed current SHA-256
  `cabf1439c4702515e8591c1865db4cca5a1f143f3111cfab510ece26faebc947`.
- The LaTeX model chapter, reset memo, scalar predictive-equivalence program,
  roadmap, dependency manifest, harness, and all other checked governance
  descriptors remained byte-identical.
- The A0 contract classifies governance inputs as mutable, excludes them from
  the immutable fingerprint and all component signatures, permits them to
  change between verifier calls, and requires a visible provenance refresh
  before handoff. This is a fixable provenance-lifecycle veto, not target,
  data, math, implementation, or research-direction invalidation.

Repair contract:

- Refresh only the stale A0-subplan descriptor in
  `source_provenance.governance_inputs`.
- Preserve every timestamp, run-manifest field, immutable member/fingerprint,
  target/probe payload, implementation, geometry, forecast payload, and all
  five signature hashes byte-for-byte.
- Rerun the unchanged strict verifier, assert the immutable/signature hashes
  are unchanged, then refresh all downstream exact lock-file bindings and their
  bounded reviews.
- Do not edit the immutable harness or regenerate target evidence.

Gate status: `A0_MUTABLE_GOVERNANCE_PROVENANCE_REFRESH_REQUIRED`

## 2026-07-12T00:45:00+08:00 - Phase A1 - CPU_XLA_VALUE_PARITY_PLAN_REPAIR

Evidence contract before the diagnostic:

- Question: does the production-owned target preserve the historical route and
  meet the predeclared eager/CPU-XLA value/score parity bounds at all ten frozen
  points?
- Baseline: eager execution of the identical production finite branch; the
  historical route remains a separate exact estimand-preservation comparator.
- Primary criterion: absolute value residual at most `1e-10` and absolute score
  infinity residual at most `1e-8`, with no point deletion.
- Vetoes: compile failure, nonfinite output, target/signature drift, or residual
  above either bound.
- Nonclaims: target-only engineering diagnostic; no posterior, HMC, NeuTra,
  predictive, calibration, model-adequacy, performance, or default evidence.

Result and localization:

- Historical eager parity and all-ten-point centered finite differences passed.
- CPU XLA compiled and the reviewed custom-gradient test passed.
- Eager/CPU-XLA value residuals ranged from
  `2.3845814212108962e-11` to `1.9697381503647193e-09`; nine of ten failed the
  absolute `1e-10` value bound. Score residuals remained within the unchanged
  `1e-8` bound, with maximum `5.6968261219481064e-09`.
- CPU XLA fast-math disabling produced identical residuals.
- At the worst point, final filtered mean/covariance residuals were
  `3.469446951953614e-18` and `8.978549240895584e-20`. The mismatch is
  accumulated log-likelihood arithmetic, not target, state recursion,
  derivative, branch, or wrapper drift.
- No CPU/GPU evidence artifact had been opened, so no observed artifact was
  selected, discarded, or relabeled.

Repair:

- Amend value parity to the symmetric formula
  `abs(a-b) <= 1e-10 * max(1,abs(a),abs(b))` for eager/CPU-XLA and
  CPU-XLA/GPU-XLA.
- Keep score, historical, and finite-difference bounds, all points, target,
  signatures, and nonclaims unchanged.
- Treat `value_parity_atol_hex` as the formula coefficient, not an absolute-only
  library tolerance.
- Rereview the amended A1 plan and golden consistency, rerun the strict A1
  entry preflight, then rerun the full focused checks before evidence generation.

Gate status: `A1_PLAN_AMENDMENT_REVIEW_REQUIRED_BEFORE_EVIDENCE`

## 2026-07-12 - Phase A1 - SCOPED_CONCURRENCY_PLAN_AMENDMENT

Skeptical audit before resumed execution:

- Wrong baseline: the accepted A0 target lock and hash-pinned historical
  SVD-UKF constructor remain the baseline; no HMC/Kalman artifact is promoted.
- Proxy promotion: ten-point parity remains an A1 engineering gate only.
- Stop conditions: exact `HEAD`, A0/golden bytes, target-critical dependencies,
  and A1-owned paths remain hard boundaries.
- Shared-worktree mismatch: the old 15,063-row whole-repository inventory was
  invalidated by a user-authorized concurrent lane and would turn unrelated
  activity into a false continuation veto.
- Artifact sufficiency: the replacement v2 scoped boundary preserves exact
  target-critical hashes and owned namespaces while recording unrelated Git
  state as explanatory provenance only.
- Stale entry verifier: the A0 discovery closure includes package-init HMC
  modules not used numerically by A1. Its live equality is replaced for A1 by
  exact accepted-lock bytes/identities, protected dependency rehashes, and
  all-ten-point historical semantic replay.
- Contract mismatch: the lazy-export allowance is corrected to the implemented
  and tested `locked_ssl_lstm_posterior_target` symbol.

The existing scale-aware value-parity amendment is retained unchanged:
`abs(a-b) <= 1e-10 * max(1,abs(a),abs(b))`. Score, historical, and finite-
difference bounds, frozen points, target, signatures, and nonclaims remain
unchanged. No CPU/GPU A1 evidence artifact has been opened.

User boundary:

- The user directed this executor to focus only on the SSL-LSTM lane while
  another agent works elsewhere.
- Unrelated concurrent changes are preserved and non-vetoing.
- Protected target-critical or A1-owned-path drift remains a continuation veto.

Gate status: `A1_AMENDED_PLAN_AND_GOLDEN_REVIEW_REQUIRED_BEFORE_EXECUTION`

## 2026-07-12T02:53:27+08:00 - Phase A1 - TERMINAL PLAN REVIEW NONCONVERGENCE

Evidence contract:

- Question: is the final amended A1 plan executable under the user-authorized
  cooperative concurrent-lane model?
- Exact baseline: A0 anchor
  `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163`, accepted A0 target lock, and
  plan candidate SHA-256
  `4c130734fe9913643eaeced43069fe6b6ed3ae0b97bc80cd2b6a429246ed2ae8`.
- Primary criterion: terminal substantive Round 5 returns `VERDICT: AGREE`.
- Veto: any material Round 5 finding under the predeclared five-round cap.
- Nonclaims: no target invalidity, runtime, CPU/GPU evidence, HMC, NeuTra,
  predictive, calibration, default, product, or scientific result.

Actions and checks:

- Hash-pinned the legacy inventory in the entry preflight and boundary immutable
  inputs without restoring live whole-worktree equality.
- Clarified `initial_owned_state` as creation-time provenance so authorized
  later A1 writes do not create a false boundary failure.
- Verified all 23 protected hashes, the exact 23/28/51 manifest partition, plan
  whitespace, nine Bash blocks, and three embedded Python programs.
- Terminal Round 5 independently reproduced the plan hash and accepted the
  direct final-path `O_EXCL|O_NOFOLLOW` publication repair.
- Round 5 returned `VERDICT: REVISE`: the final exact entry-preflight rerun can
  rewrite live commit fields after a permitted unrelated `HEAD` advance, while
  the scoped boundary freezes the original entry-file SHA-256.
- Did not patch into an unapproved sixth substantive round. Wrote the blocker
  result and refreshed the stop handoff.

Classification:

- Invalidated: executable governance/evidence lifecycle.
- Not invalidated: target, data, mask, historical SVD-UKF route, existing A1
  implementation candidate, mathematics, value-parity amendment, or research
  direction.
- No v2 boundary, evidence harness, CPU artifact, or GPU artifact was created.

Gate status: `BLOCKED_PHASE_A1_PLAN_REVIEW_NONCONVERGENCE`

Safest resume action: obtain explicit human authorization for one exceptional
focused repair that keeps the entry artifact immutable and adds a separate live
anchor-to-current history attestation. One fresh focused review may then assess
only that repair; otherwise the phase remains stopped.

## 2026-07-12 - Phase A1 - OWNER-AUTHORIZED LIFECYCLE RECOVERY AGREED

Actions:

- The owner granted five additional substantive review rounds for the stopped
  A1 lifecycle blocker.
- Patched only the lifecycle contract: the entry artifact is written once and
  remains immutable; later checkpoints verify it read-only and attest live
  anchor-to-current commit history separately.
- Preserved original Rounds 1-5 and labeled the new budget Recovery E1-E5.
- Recovery Round E1 independently reproduced plan SHA-256
  `43a671b3ed9d651ea2d3c4622c5667da0128e91cd4a71d6d7c2ef25dc840cb72`
  and returned `VERDICT: AGREE` with no material finding.
- A fresh golden consistency review reproduced golden SHA-256
  `04e237ab955172f675320216d50e87c8df27b8b9e57d7dc8234601ce1f930c34`
  and returned `GOLDEN VERDICT: AGREE` against the exact repaired plan.
- Rebound the required stable review-record paths to those accepted hashes.

Classification:

- Resolved: `entry_artifact_live_history_lifecycle`.
- Unchanged: A0 target/data, A1 target/mask/prior/filter, implementation bytes,
  tolerances, schemas, evidence roles, and all nonclaims.
- Review type remains `CODEX_SUBSTITUTE_REVIEW`, weaker than Claude.

Gate status: `A1_REPAIRED_PLAN_AND_GOLDEN_AGREED_ENTRY_PREFLIGHT_PENDING`

## 2026-07-12 - Phase A2 - FOCUSED GATE PASS AND HARNESS IMPORT REPAIR

Evidence contract:

- Question: does the repaired A2 terminal-state and forecast implementation
  pass its complete CPU-hidden engineering/regression gate, and can the exact
  reviewed file-path evidence commands import the local package?
- Baseline: accepted A1 target and artifacts, the hash-bound A2 Round 6
  subplan, and the frozen ten-row/two-draw A2 design.
- Primary criterion: all focused A2 and A1 regression tests pass; runtime
  artifacts remain ineligible until their generators and fresh verifiers pass.
- Vetoes: protected A1 drift, failed focused test, forbidden write, invalid
  artifact, or a semantic change to the target/model/tolerances.
- Nonclaims: no posterior correctness, HMC/NeuTra readiness, predictive
  equivalence, calibration, product/default readiness, or scientific claim.

Observed result:

- Exact focused command passed: `62 passed, 15112 warnings in 6055.42s`.
- Predictive source SHA-256 remained
  `85c6592753c41ba92fb29a32cc4c36134b033c7948e59c3938801d681f760942`;
  focused-test SHA-256 remained
  `f313f9f65e469bb1869f5fb81077aee278d80c4811bb460d514cfc4576f5c871`.
- Closed focused trace SHA-256 is
  `1ecbdd012bbc36399135e3dce750fb2362f059d7d35a1c37a03dc7c6696515cd`;
  it contains no repository cache, `.git`, source, or unrelated artifact
  mutation attributable to A2.
- The first exact CPU artifact command exited before numerical work with
  `ModuleNotFoundError: No module named 'bayesfilter'`; its closed failed-attempt
  trace SHA-256 is
  `149fe1361174abdee5e03db81199260ad75dc1cdc7c3a487c4c8949ad2938bbd`.
- No innovation bank, CPU reference JSON, or CPU log was created by that
  attempt. This invalidated only the failed command attempt, not the completed
  focused gate or the scientific/engineering target.

Repair:

- Add the standard repository-root `sys.path` bootstrap before local-package
  imports in the permitted A2 generator and independent verifier. This is an
  executable-harness repair only; production source, A1 target, tests, model,
  covariance policy, RNG semantics, tolerances, criteria, and nonclaims are
  unchanged.
- Recompile and scan both harnesses, refreeze the A2 source boundary, then
  rerun the exact CPU generator and fresh verifier. Do not promote the failed
  attempt or focused suite into runtime evidence.

Gate status: `A2_HARNESS_BOOTSTRAP_REPAIR_REFREEZE_REQUIRED`

## 2026-07-12 - Phase A2 - PERSISTED-BANK GPU VERIFIER REPAIR

Observed evidence before repair:

- The refrozen boundary bound generator SHA-256
  `76534b169b537d64341e52b7a80ac24a70b3e7c8227db4e7f5d769e979cd19c7`
  and verifier SHA-256
  `a4435f5d6f7cd04e6034a7e964742e68de0d050003c06295f0d688b3f1ca87d3`.
- CPU generation emitted `CPU_REFERENCE_CONTRACT_PASSED` with evidence
  signature
  `8bc9345ad0d2570dbf89c408a410d921fedac24c4f5ab2d1fa97bd7e3b0a91d9`;
  a fresh CPU verifier returned `A2_RUNTIME_ARTIFACT_VERIFIED` and all 15
  hard checks passed.
- Trusted GPU generation saw two RTX 4080 SUPER devices, emitted
  `GPU_XLA_CANARY_PASSED` with evidence signature
  `2193a40d101be9db2da0221c14846f7e9d7334634740eb4412883eebda4262e2`,
  and reported all 17 hard checks passing with maximum CPU/GPU residual
  `4.440892098500626e-16`.
- The fresh GPU verifier then failed before path replay because it regenerated
  standard normals on GPU and demanded bitwise equality with the exact
  CPU-persisted bank. One terminal-bank element differed below printed
  precision. The failed verifier trace SHA-256 is
  `76a24da5780c536a18f4c405c664d0db0c8cb235749e1f7e66370e953e9510bd`.

Classification:

- This is a verifier implementation failure, not a failed terminal, forecast,
  CPU/GPU parity, device, covariance, or XLA candidate check.
- The reviewed contract explicitly states that CPU and GPU consume the same
  already materialized hexadecimal bank and must not regenerate innovations
  inside execution. Stateless seed derivation is replay metadata; floating
  normal-transform bytes are preserved by the bank artifact.
- The prior CPU/GPU artifacts are invalidated by the verifier-source repair
  because both artifacts hash-bind the verifier. They remain historical
  attempted evidence only and must be regenerated before acceptance.

Repair:

- Reconstruct the typed `SSLLSTMInnovationBank` directly from the independently
  validated hexadecimal tensor rows and stored integer seed metadata.
- Recompute and validate the typed bank content signature and derived integer
  seed metadata, but do not regenerate floating normal tensors on the current
  backend.
- Preserve all forecast, CPU/GPU, tensor-replay, HLO, device, status, hash,
  tolerance, target, model, and nonclaim checks unchanged.
- Recompile and scan the verifier, refreeze the boundary, and freshly
  regenerate and verify CPU and GPU artifacts in order.

Gate status: `A2_PERSISTED_BANK_VERIFIER_REPAIR_REFREEZE_REQUIRED`

## 2026-07-12 - Phase A2 - IMPLEMENTATION REVIEW FINITE-ADMISSION REPAIR

Review result:

- A bounded Codex substitute implementation review of the exact production
  module returned `VERDICT: REVISE`; this review is weaker than Claude.
- The material defect was missing fail-closed finite-value admission for the
  three materialized innovation tensors and the seven forecast outputs.
- Two provenance fields also required clarification: Philox seeds describe
  generation metadata but do not prove cross-backend bitwise regeneration, and
  the accepted A1 adapter signature was stored under an ambiguous field name.

Classification:

- This is an engineering admission/provenance defect, not evidence against the
  terminal Gaussian, forecast recursion, A1 target, or predictive-validation
  research direction.
- The accepted subplan already makes persisted materialized tensor hashes the
  replay authority. No floating normal regeneration is added to GPU execution.
- All existing boundary, focused-test, CPU, and GPU artifacts are stale after
  the production/test/generator repair and are ineligible for A2 closeout.

Repair:

- Reject nonfinite free draws and each innovation tensor before terminal or
  forecast execution, even when a nonfinite bank is self-consistently rehashed.
- Reject any nonfinite compiled or eager forecast output before runtime
  provenance construction and return.
- Rename typed provenance to `a1_adapter_signature`, populate it through
  `SSLLSTMPosteriorTarget.adapter_signature()`, and state explicitly that
  materialized tensor hashes are authoritative while seed data is generation
  metadata rather than cross-backend bitwise-regeneration evidence.
- Add focused negative tests for NaN/Inf banks, free draws, and forecast output
  admission; update only the permitted generator field access.

Gate status: `A2_IMPLEMENTATION_REPAIR_CHECKS_AND_REFREEZE_REQUIRED`

## 2026-07-13 - Phase A2 - REPAIRED FOCUSED GATE AND REVIEW PASS

Observed result:

- The refrozen boundary status is `A2_SCOPED_BOUNDARY_FROZEN`; boundary
  SHA-256 is
  `6b928c78075c2993bb131219afc345da68ab91890d5c019741bb0cb66606b0ae`.
- The exact CPU-hidden A2+A1 focused command passed `69 passed, 15112
  warnings in 6526.75s`. Seven additional parameterized finite-admission cases
  account for the increase from the prior 62-test run.
- Closed focused trace SHA-256 is
  `7fe5d3091ee05406d364d2bc24dc0a1c45761b08675387af374d7e4927202e74`.
  Its mutations are confined to the reviewed `/tmp/bayesfilter-a2-*` roots;
  no A2-named repository cache exists.
- Repaired production SHA-256 is
  `0dad54c239de11f105f541527447d167114073ab046c796a813b5c1e867452ed`;
  repaired focused-test SHA-256 is
  `6e2345f3f75d28d076a47b7362693282bd01127d4069f648c74e3be6854f2c05`.
- The protected A1 target remains exactly
  `6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667`.
- Bounded implementation review Round 2 returned `VERDICT: AGREE` with no
  material finding. It is a Codex substitute review, weaker than Claude, and
  does not establish runtime or scientific validity.

Decision:

- The implementation-review and focused-regression vetoes are cleared.
- Prior CPU/GPU artifacts remain stale and ineligible because production,
  tests, and generator bytes changed. Fresh CPU generation/verification and
  trusted GPU/XLA generation/verification are still mandatory.

Gate status: `A2_FOCUSED_AND_IMPLEMENTATION_REVIEW_PASSED_RUNTIME_REGEN_REQUIRED`
