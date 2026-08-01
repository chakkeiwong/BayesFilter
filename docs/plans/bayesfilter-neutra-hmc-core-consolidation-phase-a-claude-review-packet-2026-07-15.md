# NeuTra HMC Terminal Audit Read-Only Review Packet

Date: 2026-07-15  
Packet status: `READY_FOR_BOUNDED_READ_ONLY_REVIEW`

## Reviewer Contract And Question

Claude is a read-only reviewer. Do not edit files, run commands, launch agents,
or authorize execution. This packet is self-contained for the gate; do not read
other paths unless a material finding cannot be resolved from this packet.

Question: does the completed program, including its local terminal audit and
repairs, support closing without missed requirements or scope drift, with the
narrow conclusion that the shared sequential controller is the active
claim-bearing NeuTra HMC route and that the tested procedure passed one
additional training seed plus one independently generated fixture in the same
18-dimensional LGSSM family?

End the review with exactly `VERDICT: AGREE` if no material implementation,
numerical, statistical, provenance, or claim-scope defect remains. Otherwise
state concrete fixable findings and end with exactly `VERDICT: REVISE`.

## Reviewed Program Contract

The program required three independently reported outcomes:

1. C0-C2: consolidate claim-bearing NeuTra HMC into one TensorFlow/TFP
   sequential controller, migrate the active LGSSM route, and make bypass or
   omitted-route drift machine-detectable.
2. S1: train one fresh third seed on the existing fixture and pass fresh tuning,
   sequential admission, independent confirmation, comparator agreement, and
   recovery.
3. F0-F2: generate a new fixture and exact target signature, admit a freshly
   tuned plain-HMC comparator, perform target-specific screened GPU training,
   and pass fresh NeuTra admission and independent confirmation.

Every serious sequential run had four chains, retained warm-up archived
separately from posterior draws, recent-window modern warm-up R-hat, cumulative
retained diagnostics, 10,000 warm-up/retained caps, distinct seeds, and hard
finite/status/movement/energy-error vetoes. Modern R-hat is the per-parameter
maximum of rank-normalized split and folded rank-normalized split R-hat.

The exact confirmation gates were declared before execution: modern R-hat
`<=1.01`, bulk ESS `>=1000`, tail ESS `>=400`, candidate/comparator posterior
mean difference `<=4` combined MCSE for every parameter, and truth recovery
`<=3` candidate posterior SD for every parameter. Admission warm-up used recent-
window modern R-hat `<=1.05`; confirmation required at least 2,000 warm-up and
4,000 retained draws per chain regardless of an earlier diagnostic pass.

Acceptance and reverse-KL loss were nomination/explanatory diagnostics only.
Downstream HMC convergence, health, comparator agreement, and recovery were the
scientific gates. A candidate/configuration veto was not a continuation veto
unless it invalidated target, comparator, harness, evidence, or budget.

## Engineering Execution

| Phase | Executed outcome | Evidence boundary |
| --- | --- | --- |
| C0 | `PASS_C0_ROUTE_AND_POLICY_CONTRACT` | Canonical policy `bayesfilter_neutra_sequential_hmc_v1`; versioned discovery ledger; exact-one classification; negative omitted/stale/duplicate/bypass fixtures. |
| C1 | `PASS_C1_SHARED_CONTROLLER` | Generic TensorFlow/TFP controller; no NumPy/host callback/model path; at least four chains; separate warm-up/retained seeds; archive/diagnostic callbacks; 10k caps. |
| C2 | `PASS_C2_ACTIVE_DEFAULT_MIGRATION` | Active LGSSM campaign delegates to core; old local controller removed; historical routes retained without reinterpretation. |

The route guard fails for an unledgered qualifying source, stale or duplicate
ledger entries, a missing shared-core/policy binding, fixed terminal sampling,
or reachable local `HamiltonianMonteCarlo`/`sample_chain` bypass. A bounded
fixed probe is allowed only as explicitly declared kernel nomination before
shared sequential admission.

## Scientific Execution

| Arm | Exact execution | Result |
| --- | --- | --- |
| S1 third seed | Fresh seed `(20260715,1203)`, 5,000 batched GPU/XLA steps, no weight reuse; fresh fixed-kernel grid/tuning nominated step `0.8`, then fresh shared-controller admission and independent confirmation | pass; tuning/admission artifact `sha256:efd05306...74e7`; confirmation max R-hat `1.0027965461`, min bulk ESS `5394.2862`, min tail ESS `4381.2278`, max plain-HMC difference `1.9669` combined MCSE, max truth distance `1.6456` posterior SD, no veto |
| F0 fixture/comparator | New simulation seed `(20260715,701)`, new target signature `312d2f4c...d283`; original signature `f4761932...2f30` | pass; selected comparator step `0.3`, 2,000 warm-up and 4,000 retained per chain; max R-hat `1.0053980349`, min bulk ESS `1758.5945`, min tail ESS `3982.9331`, max truth distance `1.3839` posterior SD, no admitted-kernel veto |
| F1 target-specific training | Three fresh 500-step GPU/XLA screens; selected recipe retrained from scratch for 5,000 steps with seed `(20260715,8201)` | engineering pass; exact frozen parity; memory growth; one compiled batched loop; no NumPy/host callback/fallback; final payload SHA-256 `cab56a88...5920` |
| F2 new-fixture NeuTra | Fresh admission selected step `0.8`; independent confirmation used distinct seeds, 2,000 warm-up and 4,000 retained per chain | pass; max R-hat `1.0036022359`, min bulk ESS `4073.1309`, min tail ESS `3154.1754`, max comparator difference `1.7740` combined MCSE, max truth distance `1.3420` posterior SD, no veto |

F0's first comparator grid was not promoted: step `0.8` had 256 declared
energy-error events, while healthy steps did not resolve the acceptance
bracket. A predeclared localized fresh-grid repair selected healthy step `0.3`.
The harness was repaired to reject a bad configuration individually instead of
treating it as a veto of all candidates. This did not change target, method,
gates, hardware class, or budget.

F1 screen heldout reverse-KL means nominated `inherited_wide_lr5e3`.
Differences from source-width and lower-LR arms were `0.0920` (paired MCSE
`0.0206`) and `0.3684` (paired MCSE `0.0332`). They are explicitly descriptive
and do not support a statistically general recipe ranking. F2, not loss,
promoted the frozen candidate.

## Terminal Local Audit And Repairs

The local Phase A audit found no implementation, target, numerical,
statistical, or artifact-corruption veto. It found four closeout issues:

1. stale ready-state labels on completed subplans, repaired to closed;
2. one missing stable self-hash on the F0 fixture identity ledger, added and
   recomputed as `sha256:1936ff2a...e56`;
3. distributed F1 run facts without one consolidated command manifest, repaired
   with a terminal manifest binding environment, seeds, hardware, wall times,
   paths, file hashes, artifact hashes, plan, and result;
4. F1 result JSON did not preserve contemporaneous shell command strings. The
   exact invocations were reconstructed from the frozen CLI and artifact routes
   and explicitly labeled as reconstructed, not as shell transcripts. Valid GPU
   training was not rerun merely to manufacture paperwork.

A crash handoff also alleged duplicated C2 prose, but an exact current-file scan
did not reproduce it. No speculative deletion was made.

## Verification Performed After Repair

- Broad CPU-hidden suite: `92 passed, 2 warnings in 34.09s`; warnings were only
  third-party TFP `distutils.version` deprecations.
- Active modules and CLIs compiled with `py_compile`.
- JSON parse and `git diff --check` passed.
- 48 ordinary file hashes plus 120 tensor archive hashes, 35 byte counts, and 75
  stable self-hashes in the active program root verified with zero errors.
- Active source/route-policy scan found local TFP HMC only in the canonical
  core, with no active NumPy or host callback bypass.
- Every admitted serious run has separate warm-up and retained archives, 10k
  caps, distinct seed roles, and no hard veto.

GPU devices were deliberately hidden for the terminal logic tests. The serious
S1/F1 training artifacts themselves record RTX 4080 SUPER execution,
TensorFlow 2.19.1, JIT/XLA, memory growth before initialization, one compiled
batched training loop, seeds, wall time, status, parity, and hashes.

## Proposed Terminal Decision

| Decision item | Proposed status |
| --- | --- |
| Engineering correctness | pass under focused tests, policy guard, active scan, and artifact audit |
| Hard veto screen | pass for admitted S1/F0/F2 kernels; rejected F0 step 0.8 remains a configuration veto only |
| S1 additional-seed viability | supported for exactly one additional seed on the original fixture |
| F2 additional-fixture viability | supported for exactly one candidate on one independently generated fixture |
| Joint narrow program answer | pass: both predeclared arms passed independently |
| Statistically supported ranking | none |
| Broad robustness/default readiness | not established scientifically |
| Remaining provenance limitation | Close with disclosed non-material caveat: F1 command strings are reconstructed transparently; numerical/device/seed/wall-time evidence is contemporaneous and hashed |

The claimed target is narrow LGSSM NeuTra procedure viability under the exact
downstream gates above. The quantities actually computed are finite/status and
parity checks, modern R-hat, bulk/tail ESS, combined-MCSE posterior agreement,
truth recovery, route-policy invariants, and artifact integrity. They support
that narrow target but are different from and insufficient for calibration,
superiority, population reliability, broader-model robustness, production
readiness, or universal reliability.

The missing contemporaneous F1 shell strings are a real provenance weakness,
not silently treated as absent. It is non-material to this close because the
only CLI exposes exact enumerated stage/job/recipe arguments, each immutable
output path identifies that selection, and each result contemporaneously binds
the target, recipe, seed, steps, device, XLA/memory policy, wall time, payload,
and stable/file hashes. Thus the numerical run and an exact reproducible
invocation remain identified, while the stronger historical claim that a shell
transcript was preserved is explicitly forbidden. The terminal decision is
`complete_with_disclosed_command-transcript_caveat`, not perfect provenance.

## Required Nonclaims

- no sampler or recipe superiority;
- no calibrated coverage claim;
- no population seed-failure-rate estimate;
- no robustness beyond these fixtures in this same 18D LGSSM family;
- no production readiness or universal reliability;
- no interpretation of training/heldout loss or acceptance as convergence; and
- no claim that the new fixture is generally harder, only that it is genuinely
  different and used a less privileged non-truth-centered training center.
