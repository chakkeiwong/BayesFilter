# NeuTra HMC Program Phase A Terminal Audit Result

Date: 2026-07-15  
Decision: `PASS_PHASE_A_CLOSE_WITH_DISCLOSED_COMMAND_TRANSCRIPT_CAVEAT`

## Outcome

The terminal audit closes the NeuTra HMC core-consolidation and LGSSM
robustness program. No implementation, numerical, statistical, target,
comparator, active-route, or artifact-corruption veto remains.

The active claim-bearing LGSSM NeuTra HMC route uses the shared TensorFlow/TFP
sequential controller under policy `bayesfilter_neutra_sequential_hmc_v1`.
Warm-up is retained and separately archived but excluded from posterior draws;
warm-up readiness uses recent-window modern R-hat; retained sampling grows
cumulatively under modern R-hat/full-convergence gates; warm-up and retained
counts each cap at 10,000 per chain. The route guard detects omitted, stale,
duplicate, missing-core, fixed-terminal, and reachable local-sampler bypasses.

Both predeclared scientific arms passed independently:

| Arm | Primary status | Hard veto status | Narrow conclusion |
| --- | --- | --- | --- |
| S1 fresh third training seed | pass | no admitted-run veto | viability supported for seed `(20260715,1203)` on the original fixture |
| F2 fresh new-fixture candidate | pass | no admitted-run veto | viability supported for one candidate on fixture seed `(20260715,701)` with new target signature `312d2f4c...d283` |

This supports a narrow joint statement for one additional training seed and one
additional fixture in the same 18-dimensional LGSSM family. It does not support
broad robustness, sampler or recipe superiority, calibration, a seed-failure
rate, production readiness, cross-model transfer, or universal reliability.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close engineering consolidation | passed shared-core, migration, policy-guard, compatibility, static, and focused tests | no active bypass or archive-semantics veto | route discovery is syntax-marker based and remains a maintained guard | use the shared controller for future claim-bearing NeuTra HMC | no universal sampler correctness |
| Close S1 | passed fresh tuning/admission/confirmation, convergence, comparator agreement, and recovery | no health/status/movement/energy-error veto | one additional seed cannot estimate failure probability | retain as seed-specific evidence | no population reliability or ranking |
| Close F0-F2 | passed new target/comparator, target-specific training, fresh admission/confirmation, agreement, and recovery | bad F0 step 0.8 rejected; admitted kernels healthy | only one new fixture in the same model family | a future program may test another model/dimension if scientifically needed | no broad robustness or cross-model transfer |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | passed for every admitted S1/F0/F2 kernel; F0 step 0.8 remains rejected configuration evidence |
| Statistically supported ranking | none |
| Descriptive-only differences | acceptance, runtime, losses, per-arm ESS magnitudes, fixture spread, screen mean differences |
| Default-readiness | engineering route default established; broad scientific default readiness not established |
| Next evidence needed | multiple fixtures/seeds with uncertainty for a reliability rate, and another model family/dimension for broader transfer |

## Terminal Verification

- Broad CPU-hidden focused suite: `92 passed, 2 warnings in 34.09s`; warnings
  were third-party TFP deprecations only.
- Post-close core/route/campaign recheck: `42 passed`; public API and common
  inference-runtime compatibility recheck: `58 passed`.
- Active modules and CLIs compile.
- Phase A JSON parse and `git diff --check` pass.
- 48 file hashes plus 120 tensor hashes, 35 byte counts, and 75 stable self-
  hashes verify with zero errors.
- F0 fixture identity ledger now has a recomputed stable self-hash.
- F1 now has a consolidated terminal run manifest.
- All five F1 manifest rows match result-file and embedded artifact hashes.
- Claude Round 1 returned `REVISE`; the packet was patched for exact thresholds,
  S1 fresh-tuning evidence, and provenance classification. Round 2 returned
  `VERDICT: AGREE`.

## Residual Risk And Post-Run Red Team

The strongest alternative explanation is that both fixtures are favorable
members of one 18D LGSSM family and that this architecture/procedure may fail on
other dimensions, models, posterior geometry, or seeds. A well-powered
multi-fixture/multi-seed evaluation or a valid failure in another model family
would materially change the broader interpretation; no such broader claim is
made here.

F1's original result payloads did not preserve contemporaneous shell command
strings. The terminal manifest reconstructs exact invocations from the frozen
enumerated CLI and labels them as reconstructed. Contemporaneous target,
recipe, seed, steps, hardware, XLA/memory-growth, wall-time, payload, path, and
hash evidence remains preserved. Therefore this is a disclosed provenance
caveat, not an unrecorded gap or a scientific-run veto.

No next phase remains in this program. Future scientific expansion requires a
new question and plan; routine use and maintenance of the shared controller do
not require reopening this closeout.
