# Reply to the response on F1

Reviewer: Claude Opus 5 (1M context)  
Date: 2026-09-18  
Re: [Response to verdict](surrogate-hmc-all-models-opus-response-2026-09-18.md)

## F1 is withdrawn

**I agree that F1's claimed requirement is mathematically incorrect.** The response's detailed-balance derivation is sound, and the counterexample demonstrates that force and endpoint can use different frozen inputs while preserving the target posterior.

### Where F1 went wrong

My finding conflated two distinct requirements:

1. **Endpoint consistency** (necessary): All endpoint evaluations—U_v(z) current, U_v(z') proposed, cached values, and replays—must use the same frozen realization (omega_v, c_v) to maintain detailed balance for the declared target.

2. **Force-endpoint matching** (not necessary): The force g_f and endpoint U_v must use the same frozen particle bank and controls.

Requirement (1) is correct and follows from the acceptance ratio comparing rho_v(x) with rho_v(T_f(x)). Requirement (2) is what F1 incorrectly claimed was necessary for detailed balance. The response correctly shows that detailed balance depends only on:
- The involution property of T_f (from symmetric leapfrog + momentum reversal)
- Volume preservation
- Consistent use of the same target density rho_v in the acceptance ratio

The force affects only the proposal map T_f. As long as T_f is an involution and volume-preserving (which it is for any finite deterministic position-only field with the stated qualifications), detailed balance holds for the endpoint target, regardless of whether g_f = grad U_v or whether g_f and U_v share inputs.

The counterexample—Gaussian endpoint with zero force giving random-walk Metropolis—is valid and sufficient to disprove the claimed necessity.

### What remains useful

The response's corrected consumer requirement is appropriate and clearer than what the master currently states:

> Every current, cached, proposed and replayed endpoint evaluation uses the same declared endpoint binding (omega_v, c_v) throughout retained sampling. Every forward and reverse force evaluation uses its own declared frozen force binding (omega_f, c_f). Verify each binding independently and prevent force evaluation from mutating endpoint state or its cache. The two bindings may differ.

This clarification is useful for implementation but is not a defect requiring revision before execution. The master's existing language—"Keep the current endpoint cached without changing its random inputs or controls" (section 3.1, line 169) and "Never select a convenient omega because its posterior happens to agree with the reference. Each bank defines its own finite target" (section 3, lines 134-137)—already requires frozen endpoint consistency. Making the force-endpoint separation explicit would reduce ambiguity but does not establish a new mathematical requirement.

### Disposition

**F1 is withdrawn.** The master program's detailed-balance argument in section 3.1 is correct as written. The involution proof does not assume g = grad U, and different frozen force/endpoint inputs do not violate detailed balance.

**Implementation guidance** (not a defect): Test force and endpoint binding independence; verify endpoint cache consistency; test deterministic replay. Do not require omega_f = omega_v as a validity condition.

## Updated verdict summary

With F1 withdrawn:

- **F2** (mixed-precision classification): Useful reporting clarification. The master already distinguishes FP64 integration from FP32 DCTR (section 9, lines 656-662); structured precision fields in decision artifacts would make scope identity machine-readable. Agree with response that per-arm dtype differences (B: FP64 Kalman endpoint, C: promoted FP32 DCTR endpoint) should be recorded separately rather than obscured by a single label.

- **F3** (noise-scale priors): Useful clarification. Agree with response that synthetic benchmark priors can have convenient scales; require defensible interpretation for real-model inference claims.

- **F4** (measured budget): Clarification only, already covered in the master.

**No material defect requiring revision was established by the verdict.** The master program's research design, probability-target separation, detailed-balance argument, comparison structure, evidence rules, and budget accounting are sound.

The master's existing prerequisites—P0/P1 implementation of runners, scope cards, consumer wiring, qualified references, and measured forecasts—remain necessary for execution readiness, as the program explicitly states.

## Source verification note

The response inspected three sources:
- [neural_force_hmc.py](../../bayesfilter/inference/neural_force_hmc.py) line 603: accepts separate force and target bindings architecturally.
- [test_neural_force_hmc.py](../../tests/test_neural_force_hmc.py) line 379: tests scaled Gaussian force against different Gaussian endpoint.
- [tuning_contract.py](../../bayesfilter/inference/tuning_contract.py) line 850: advertises endpoint-corrected position-field route support.

These checks address items I listed as unchecked (actual implementation support for the typed binding). They do not close the full-DCTR consumer wiring work, which the master explicitly lists as a P0/P1 prerequisite.

---

**Final verdict: The master program defines a scientifically valid surrogate-HMC test. No revision is required before proceeding to implementation prerequisites and execution authorization.**
